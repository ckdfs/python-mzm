"""Dither-based closed-loop MZM bias controller (reusable module).

What this implements
- You can ONLY observe PD low-frequency pilot (dither) 1f/2f power.
- Features are normalized by DC photocurrent to handle optical power fluctuations.
- You do NOT assume signed I/Q lock-in outputs.

To recover direction (which is lost if you only use power magnitudes), the controller
uses a *history finite difference* (more realistic than taking an extra probe sample):
    raw_k = [H1_k, H2_k] where H* = (harmonic amplitude) / pd_dc
    shape_k = raw_k / sqrt(H1_k^2 + H2_k^2 + eps)
    x_k = [shape1_k, shape2_k, dshape1, dshape2, dV_{k-1}, sin(theta*), cos(theta*)]
where dshape* = shape*(k) - shape*(k-1).

DC Normalization: By dividing harmonic amplitudes by pd_dc (mean photocurrent),
the features become robust against input optical power fluctuations (e.g., laser
aging, fiber vibrations). This is critical for real-world deployment.

The policy outputs an incremental update ΔV.
Bias is clamped to [0, Vpi] so the target range is 0..180 degrees.

Artifacts
- Dataset: artifacts/dither_dataset_dbm_hist.npz
- Model:   artifacts/dither_policy_dbm_hist.pt

NOTE
- This file is intended to be imported (Notebook calls it). It does not act as a CLI script.
- To run the default pipeline without notebooks, call run_pipeline() (or main()) from Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import torch
from torch import nn

from mzm.model import (
    measure_pd_dither_normalized_batch_torch,
    bias_to_theta_rad,
    theta_to_bias_V,
    wrap_to_pi,
)
from mzm.utils import select_device, make_lockin_refs

_DEFAULT_FEATURE_MODE = "theta_est_hist"
_DPHI_CLIP_RAD = float(np.pi / 4.0)
_CONF_GATE_EMA_ALPHA = 0.02
_CONF_GATE_RATIO = 0.5
_CONF_GATE_MIN_SCALE = 0.2
_CONF_GATE_POWER = 1.0


def _bessel_j1_small(x: float) -> float:
    """Approximate J1(x) for small x using a 3-term series."""

    x2 = float(x) * float(x)
    return float(x) / 2.0 - (float(x) * x2) / 16.0 + (float(x) * x2 * x2) / 384.0


def _bessel_j2_small(x: float) -> float:
    """Approximate J2(x) for small x using a 3-term series."""

    x2 = float(x) * float(x)
    return x2 / 8.0 - (x2 * x2) / 96.0 + (x2 * x2 * x2) / 3072.0


def _bessel_j0_small(x: float) -> float:
    """Approximate J0(x) for small x using a 3-term series."""

    x2 = float(x) * float(x)
    return 1.0 - x2 / 4.0 + (x2 * x2) / 64.0


def _gamma_from_er_db(er_db: float) -> float:
    er_linear = 10.0 ** (float(er_db) / 20.0)
    return float((er_linear - 1.0) / (er_linear + 1.0))


def _estimate_theta_from_harmonics_np(
    h1_norm: np.ndarray,
    h2_norm: np.ndarray,
    *,
    theta_prior_rad: np.ndarray | None = None,
    dither_params: "DitherParams",
    device_params: "DeviceParams",
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate theta in [0, pi] from (h1_norm, h2_norm), eliminating RF scaling.

    Uses the analytical small-signal model described (with TeX equations) in:
      docs/theta_inversion_theory.md

    In short, we first compute:
      p = h1_norm / (2*J1(beta_d)), q = h2_norm / (2*J2(beta_d))
      theta_abs = atan2(p, q)
    which cancels the RF scaling nuisance parameter via the ratio p/q = |tan(theta)|.

    We resolve the sign ambiguity (theta vs pi-theta) using an external prior
    (typically the commanded bias converted to phase via Vpi), which is available
    in closed-loop operation. This avoids brittle inference near deep nulls.
    """

    h1 = np.asarray(h1_norm, dtype=np.float32)
    h2 = np.asarray(h2_norm, dtype=np.float32)

    beta_d = float(np.pi) * float(dither_params.V_dither_amp) / float(device_params.Vpi_DC)
    j0 = _bessel_j0_small(beta_d)
    j1 = _bessel_j1_small(beta_d)
    j2 = _bessel_j2_small(beta_d)
    j1 = float(j1) if abs(j1) > 1e-12 else 1e-12
    j2 = float(j2) if abs(j2) > 1e-12 else 1e-12

    gamma = _gamma_from_er_db(float(device_params.ER_dB))
    a = 1.0 + gamma * gamma
    b_max = 2.0 * gamma

    p = h1 / np.float32(2.0 * j1)
    q = h2 / np.float32(2.0 * j2)

    theta_abs = np.arctan2(p, q).astype(np.float32)  # [0, pi/2]
    sin_abs = np.sin(theta_abs).astype(np.float32)
    cos_abs = np.cos(theta_abs).astype(np.float32)

    w = (p / (sin_abs + np.float32(eps))).astype(np.float32)  # b / (a + b*j0*cos)

    # Estimate b for both sign hypotheses (cos positive/negative).
    denom_plus = (1.0 - w * np.float32(j0) * cos_abs).astype(np.float32)
    denom_minus = (1.0 + w * np.float32(j0) * cos_abs).astype(np.float32)
    b_plus = (w * np.float32(a)) / (denom_plus + np.float32(eps))
    b_minus = (w * np.float32(a)) / (denom_minus + np.float32(eps))

    if theta_prior_rad is None:
        # Fallback: choose the hypothesis whose b stays in-range and closer to b_max (more physical for high ER).
        ok_plus = (b_plus > 0.0) & (b_plus <= np.float32(b_max + 1e-6))
        ok_minus = (b_minus > 0.0) & (b_minus <= np.float32(b_max + 1e-6))
        # Prefer valid; if both valid, prefer larger b (shallower null is more stable numerically).
        choose_plus = np.where(ok_plus & ~ok_minus, True, np.where(ok_minus & ~ok_plus, False, b_plus >= b_minus))
    else:
        prior = np.asarray(theta_prior_rad, dtype=np.float32)
        cand_plus = theta_abs
        cand_minus = (np.float32(np.pi) - theta_abs).astype(np.float32)
        choose_plus = (np.abs(cand_plus - prior) <= np.abs(cand_minus - prior))

    theta_est = np.where(choose_plus, theta_abs, (np.float32(np.pi) - theta_abs)).astype(np.float32)
    b_est = np.where(choose_plus, b_plus, b_minus).astype(np.float32)
    b_est = np.clip(b_est, 0.0, np.float32(b_max)).astype(np.float32)
    return theta_est, b_est


def _estimate_theta_from_harmonics_torch(
    h1_norm: torch.Tensor,
    h2_norm: torch.Tensor,
    *,
    theta_prior_rad: torch.Tensor | None = None,
    dither_params: "DitherParams",
    device_params: "DeviceParams",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    h1 = h1_norm.to(dtype=torch.float32)
    h2 = h2_norm.to(dtype=torch.float32)

    beta_d = float(np.pi) * float(dither_params.V_dither_amp) / float(device_params.Vpi_DC)
    j0 = float(_bessel_j0_small(beta_d))
    j1 = float(_bessel_j1_small(beta_d))
    j2 = float(_bessel_j2_small(beta_d))
    j1 = j1 if abs(j1) > 1e-12 else 1e-12
    j2 = j2 if abs(j2) > 1e-12 else 1e-12

    gamma = float(_gamma_from_er_db(float(device_params.ER_dB)))
    a = float(1.0 + gamma * gamma)
    b_max = float(2.0 * gamma)

    p = h1 / float(2.0 * j1)
    q = h2 / float(2.0 * j2)

    theta_abs = torch.atan2(p, q)  # [0, pi/2]
    sin_abs = torch.sin(theta_abs)
    cos_abs = torch.cos(theta_abs)

    w = p / (sin_abs + float(eps))

    denom_plus = 1.0 - w * float(j0) * cos_abs
    denom_minus = 1.0 + w * float(j0) * cos_abs
    b_plus = (w * float(a)) / (denom_plus + float(eps))
    b_minus = (w * float(a)) / (denom_minus + float(eps))

    if theta_prior_rad is None:
        ok_plus = (b_plus > 0.0) & (b_plus <= float(b_max + 1e-6))
        ok_minus = (b_minus > 0.0) & (b_minus <= float(b_max + 1e-6))
        choose_plus = torch.where(ok_plus & ~ok_minus, True, torch.where(ok_minus & ~ok_plus, False, b_plus >= b_minus))
    else:
        prior = theta_prior_rad.to(dtype=torch.float32)
        cand_plus = theta_abs
        cand_minus = float(np.pi) - theta_abs
        choose_plus = torch.abs(cand_plus - prior) <= torch.abs(cand_minus - prior)

    theta_est = torch.where(choose_plus, theta_abs, (float(np.pi) - theta_abs))
    b_est = torch.where(choose_plus, b_plus, b_minus)
    b_est = torch.clamp(b_est, 0.0, float(b_max))
    return theta_est, b_est


def _bessel_equalized_shape_normalize_np(
    h1: np.ndarray,
    h2: np.ndarray,
    *,
    V_dither_amp: float,
    Vpi_DC: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bessel-equalize 1f/2f before shape normalization.

    For small dither depth beta_d = pi*V_dither/Vpi, the 2f component is scaled by
    J2(beta_d), which is orders of magnitude smaller than J1(beta_d). If we
    shape-normalize [h1, h2] directly, the vector collapses toward the 1f axis.

    We first equalize:
      z1 = h1 / J1(beta_d), z2 = h2 / J2(beta_d)
    then shape-normalize z to remove common scaling (including RF J0(beta_rf)).
    """

    beta_d = float(np.pi) * float(V_dither_amp) / float(Vpi_DC)
    j1 = _bessel_j1_small(beta_d)
    j2 = _bessel_j2_small(beta_d)
    j_eps = 1e-12
    j1 = float(j1) if abs(j1) > j_eps else (j_eps if j1 >= 0 else -j_eps)
    j2 = float(j2) if abs(j2) > j_eps else (j_eps if j2 >= 0 else -j_eps)

    h1 = np.asarray(h1, dtype=np.float32)
    h2 = np.asarray(h2, dtype=np.float32)
    z1 = (h1 / np.float32(j1)).astype(np.float32)
    z2 = (h2 / np.float32(j2)).astype(np.float32)

    s = np.sqrt(z1 * z1 + z2 * z2 + float(eps)).astype(np.float32)
    return (z1 / s).astype(np.float32), (z2 / s).astype(np.float32), s


def _bessel_equalized_shape_normalize_torch(
    h1: torch.Tensor,
    h2: torch.Tensor,
    *,
    V_dither_amp: float,
    Vpi_DC: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta_d = float(np.pi) * float(V_dither_amp) / float(Vpi_DC)
    j1 = _bessel_j1_small(beta_d)
    j2 = _bessel_j2_small(beta_d)
    j_eps = 1e-12
    j1 = float(j1) if abs(j1) > j_eps else (j_eps if j1 >= 0 else -j_eps)
    j2 = float(j2) if abs(j2) > j_eps else (j_eps if j2 >= 0 else -j_eps)

    h1 = h1.to(dtype=torch.float32)
    h2 = h2.to(dtype=torch.float32)
    z1 = h1 / float(j1)
    z2 = h2 / float(j2)
    s = torch.sqrt(z1 * z1 + z2 * z2 + float(eps))
    return z1 / s, z2 / s, s


@dataclass
class DeviceParams:
    Vpi_DC: float = 5.0
    ER_dB: float = 30.0
    IL_dB: float = 6.0
    Pin_dBm: float = 10.0
    Responsivity: float = 0.786
    R_load: float = 50.0


@dataclass
class DitherParams:
    V_dither_amp: float = 0.05
    f_dither: float = 10e3
    Fs: float = 2e6
    n_periods: int = 120


class DeltaVPolicyNet(nn.Module):
    def __init__(self, in_dim: int = 7, hidden: int = 64, depth: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.depth = depth
        
        layers: list[nn.Module] = []
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _target_encoding(theta_target_rad: np.ndarray) -> np.ndarray:
    th = np.asarray(theta_target_rad, dtype=np.float32)
    return np.stack([np.sin(th), np.cos(th)], axis=1)


def generate_dataset_dbm_hist(
    *,
    device_params: DeviceParams,
    dither_params: DitherParams,
    n_samples: int = 8000,
    seed: int = 0,
    teacher_gain: float = 0.5,
    max_step_V: float = 0.2,
    feature_mode: str = _DEFAULT_FEATURE_MODE,
    accel: str = "auto",
    torch_batch: int = 512,
    V_rf_amp: float = 0.0,
    f_rf: float = 1e9,
) -> dict:
    """Generate supervised dataset for realistic dbm_hist controller.

    Parameters
    ----------
    V_rf_amp : float, optional
        RF signal amplitude (V) to include during training. Default 0.0 (no RF).
        Set to a positive value (e.g., 0.2) to train the model with RF interference,
        which can improve robustness to RF power variations during inference.
    f_rf : float, optional
        RF signal frequency (Hz). Default 1e9 (1 GHz).

    Returns a dict containing Xn, y, and normalization stats.
    """

    feature_mode_norm = str(feature_mode).lower().strip()
    if feature_mode_norm not in {"dc_norm_hist", "shape_norm_bessel", "theta_est_hist", "phi_bessel_hist"}:
        raise ValueError(
            "feature_mode must be one of: 'dc_norm_hist', 'shape_norm_bessel', 'theta_est_hist', 'phi_bessel_hist'"
        )

    rng = np.random.default_rng(seed)

    Vpi = float(device_params.Vpi_DC)

    # Current bias in [0, Vpi] -> theta in [0, pi]
    V_bias = rng.uniform(0.0, Vpi, size=n_samples).astype(np.float32)

    # Target theta in [0, pi] (0..180 deg)
    theta_target = rng.uniform(0.0, float(np.pi), size=n_samples).astype(np.float32)

    # Create a "previous" step consistent with a realistic controller history.
    dv_prev = rng.uniform(-max_step_V, max_step_V, size=n_samples).astype(np.float32)
    V_prev = np.clip(V_bias - dv_prev, 0.0, Vpi).astype(np.float32)

    device = select_device(accel)

    print(f"generate_dataset_dbm_hist using device: {device}")
    if V_rf_amp > 0:
        print(f"  RF training enabled: V_rf_amp={V_rf_amp:.3f} V, f_rf={f_rf/1e9:.1f} GHz")
    else:
        print(f"  RF training disabled (V_rf_amp=0)")

    # Precompute lock-in references once per call, on the selected device.
    refs = make_lockin_refs(dither_params, device)

    def _measure_normalized_torch(v_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Measure DC-normalized harmonic amplitudes (h1/pd_dc, h2/pd_dc)."""
        h1_out = np.empty((v_arr.size,), dtype=np.float32)
        h2_out = np.empty((v_arr.size,), dtype=np.float32)
        bs = int(max(1, torch_batch))
        for start in range(0, v_arr.size, bs):
            end = min(v_arr.size, start + bs)
            vb = torch.from_numpy(v_arr[start:end]).to(device=device, dtype=torch.float32)
            h1b, h2b, _ = measure_pd_dither_normalized_batch_torch(
                V_bias=vb,
                V_dither_amp=float(dither_params.V_dither_amp),
                f_dither=float(dither_params.f_dither),
                Fs=float(dither_params.Fs),
                n_periods=int(dither_params.n_periods),
                Vpi_DC=float(device_params.Vpi_DC),
                ER_dB=float(device_params.ER_dB),
                IL_dB=float(device_params.IL_dB),
                Pin_dBm=float(device_params.Pin_dBm),
                Responsivity=float(device_params.Responsivity),
                R_load=float(device_params.R_load),
                refs=refs,
                V_rf_amp=float(V_rf_amp),
                f_rf=float(f_rf),
            )
            h1_out[start:end] = h1b.detach().cpu().numpy()
            h2_out[start:end] = h2b.detach().cpu().numpy()
        return h1_out, h2_out

    h1, h2 = _measure_normalized_torch(V_bias)
    h1_prev, h2_prev = _measure_normalized_torch(V_prev)

    te = _target_encoding(theta_target.astype(np.float32))

    if feature_mode_norm == "dc_norm_hist":
        dh1 = (h1 - h1_prev).astype(np.float32)
        dh2 = (h2 - h2_prev).astype(np.float32)
        X = np.stack(
            [
                h1.astype(np.float32),
                h2.astype(np.float32),
                dh1,
                dh2,
                dv_prev.astype(np.float32),
                te[:, 0].astype(np.float32),
                te[:, 1].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        feature_version = "dc_norm_hist_v1"
    elif feature_mode_norm == "shape_norm_bessel":
        u1, u2, _ = _bessel_equalized_shape_normalize_np(
            h1, h2, V_dither_amp=float(dither_params.V_dither_amp), Vpi_DC=float(device_params.Vpi_DC)
        )
        u1_prev, u2_prev, _ = _bessel_equalized_shape_normalize_np(
            h1_prev, h2_prev, V_dither_amp=float(dither_params.V_dither_amp), Vpi_DC=float(device_params.Vpi_DC)
        )
        du1 = (u1 - u1_prev).astype(np.float32)
        du2 = (u2 - u2_prev).astype(np.float32)
        X = np.stack(
            [
                u1.astype(np.float32),
                u2.astype(np.float32),
                du1,
                du2,
                dv_prev.astype(np.float32),
                te[:, 0].astype(np.float32),
                te[:, 1].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        feature_version = "shape_norm_bessel_v1"
    elif feature_mode_norm == "phi_bessel_hist":
        beta_d = float(np.pi) * float(dither_params.V_dither_amp) / float(device_params.Vpi_DC)
        j1 = float(_bessel_j1_small(beta_d))
        j2 = float(_bessel_j2_small(beta_d))
        j1 = float(j1) if abs(j1) > 1e-12 else 1e-12
        j2 = float(j2) if abs(j2) > 1e-12 else 1e-12

        p = h1.astype(np.float32) / np.float32(2.0 * j1)
        q = h2.astype(np.float32) / np.float32(2.0 * j2)
        p_prev = h1_prev.astype(np.float32) / np.float32(2.0 * j1)
        q_prev = h2_prev.astype(np.float32) / np.float32(2.0 * j2)

        phi = np.arctan2(p, q).astype(np.float32)  # ~ [0, pi/2]
        phi_prev = np.arctan2(p_prev, q_prev).astype(np.float32)
        dphi = (phi - phi_prev).astype(np.float32)
        dphi = np.clip(dphi, -_DPHI_CLIP_RAD, _DPHI_CLIP_RAD).astype(np.float32)

        X = np.stack(
            [
                phi,
                dphi,
                dv_prev.astype(np.float32),
                te[:, 0].astype(np.float32),
                te[:, 1].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        feature_version = "phi_bessel_hist_v1"
    else:
        theta_prior = bias_to_theta_rad(V_bias.astype(float), Vpi_DC=device_params.Vpi_DC).astype(np.float32)
        theta_prior_prev = bias_to_theta_rad(V_prev.astype(float), Vpi_DC=device_params.Vpi_DC).astype(np.float32)
        theta_est, _ = _estimate_theta_from_harmonics_np(
            h1,
            h2,
            theta_prior_rad=theta_prior,
            dither_params=dither_params,
            device_params=device_params,
        )
        theta_prev_est, _ = _estimate_theta_from_harmonics_np(
            h1_prev,
            h2_prev,
            theta_prior_rad=theta_prior_prev,
            dither_params=dither_params,
            device_params=device_params,
        )
        s = np.sin(theta_est).astype(np.float32)
        c = np.cos(theta_est).astype(np.float32)
        s_prev = np.sin(theta_prev_est).astype(np.float32)
        c_prev = np.cos(theta_prev_est).astype(np.float32)
        ds = (s - s_prev).astype(np.float32)
        dc = (c - c_prev).astype(np.float32)
        X = np.stack(
            [
                s,
                c,
                ds,
                dc,
                dv_prev.astype(np.float32),
                te[:, 0].astype(np.float32),
                te[:, 1].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        feature_version = "theta_est_hist_v2"

    # Teacher label: proportional step in wrapped phase error
    th_c = bias_to_theta_rad(V_bias.astype(float), Vpi_DC=device_params.Vpi_DC).astype(np.float32)
    err = wrap_to_pi(theta_target.astype(float) - th_c.astype(float)).astype(np.float32)
    dv = float(teacher_gain) * theta_to_bias_V(err.astype(float), Vpi_DC=device_params.Vpi_DC).astype(np.float32)
    dv = np.clip(dv, -max_step_V, max_step_V).astype(np.float32)
    y = dv.reshape(-1, 1).astype(np.float32)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    Xn = (X - mu) / sigma

    return {
        "Xn": Xn,
        "y": y,
        "mu": mu,
        "sigma": sigma,
        "feature_mode": feature_mode_norm,
        "feature_version": feature_version,
        "device_params": device_params,
        "dither_params": dither_params,
        "teacher_gain": float(teacher_gain),
        "max_step_V": float(max_step_V),
        "V_rf_amp": float(V_rf_amp),
        "f_rf": float(f_rf),
    }


def save_dataset(dataset: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        Xn=dataset["Xn"],
        y=dataset["y"],
        mu=dataset["mu"],
        sigma=dataset["sigma"],
        feature_version=np.array(dataset.get("feature_version", "legacy"), dtype=object),
        feature_mode=np.array(dataset.get("feature_mode", "legacy"), dtype=object),
        device_params=np.array(list(dataset["device_params"].__dict__.items()), dtype=object),
        dither_params=np.array(list(dataset["dither_params"].__dict__.items()), dtype=object),
        teacher_gain=np.array(dataset["teacher_gain"], dtype=np.float32),
        max_step_V=np.array(dataset["max_step_V"], dtype=np.float32),
        V_rf_amp=np.array(dataset.get("V_rf_amp", 0.0), dtype=np.float32),
        f_rf=np.array(dataset.get("f_rf", 1e9), dtype=np.float32),
    )


def load_dataset(path: str | Path) -> dict:
    z = np.load(Path(path), allow_pickle=True)

    device_params = DeviceParams(**dict(z["device_params"]))
    dither_params = DitherParams(**dict(z["dither_params"]))

    # Load RF parameters (with backward compatibility for old datasets)
    V_rf_amp = float(z["V_rf_amp"]) if "V_rf_amp" in z.files else 0.0
    f_rf = float(z["f_rf"]) if "f_rf" in z.files else 1e9
    feature_version = str(z["feature_version"]) if "feature_version" in z.files else "legacy"
    feature_mode = str(z["feature_mode"]) if "feature_mode" in z.files else "legacy"

    return {
        "Xn": z["Xn"].astype(np.float32),
        "y": z["y"].astype(np.float32),
        "mu": z["mu"].astype(np.float32),
        "sigma": z["sigma"].astype(np.float32),
        "feature_version": feature_version,
        "feature_mode": feature_mode,
        "device_params": device_params,
        "dither_params": dither_params,
        "teacher_gain": float(z["teacher_gain"]),
        "max_step_V": float(z["max_step_V"]),
        "V_rf_amp": V_rf_amp,
        "f_rf": f_rf,
    }


def train_policy(
    *,
    Xn: np.ndarray,
    y: np.ndarray,
    epochs: int = 2000,
    batch: int = 256,
    lr: float = 1e-3,
    hidden: int = 64,
    depth: int = 3,
    seed: int = 0,
    accel: str = "auto",
) -> nn.Module:
    if seed is not None:
        torch.manual_seed(seed)

    device = select_device(accel)

    print(f"train_policy using device: {device}")

    # Optimization: Move entire dataset to device upfront.
    # For small datasets (like 8000 samples), this eliminates the CPU->GPU transfer overhead
    # in every batch iteration, which is the main bottleneck for MPS/GPU on small models.
    X_dev = torch.from_numpy(Xn).to(device)
    y_dev = torch.from_numpy(y).to(device)

    # Optimization: Remove DataLoader overhead for small datasets.
    # DataLoader adds significant Python overhead per batch which dominates for small models.
    # We use manual shuffling and slicing instead.
    n_samples = X_dev.shape[0]
    indices = torch.arange(n_samples, device=device)

    model = DeltaVPolicyNet(in_dim=Xn.shape[1], hidden=hidden, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Scheduler to reduce LR when loss plateaus, helping to reduce fluctuations and find deeper minima
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=300, min_lr=1e-6
    )

    # Note: We intentionally skip torch.compile here because it can be unstable on some
    # macOS/MPS/CPU configurations, causing crashes mid-training.
    # The manual batching above already provides the majority of the speedup.

    for epoch in range(1, int(epochs) + 1):
        model.train()
        
        # Manual shuffle
        indices = indices[torch.randperm(n_samples, device=device)]
        
        running = 0.0
        
        # Manual batch loop
        for start in range(0, n_samples, batch):
            end = start + batch
            idx = indices[start:end]
            xb = X_dev[idx]
            yb = y_dev[idx]
            
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)

        avg_loss = running / n_samples
        scheduler.step(avg_loss)

        if epoch == 1 or epoch % 200 == 0 or epoch == int(epochs):
            current_lr = opt.param_groups[0]['lr']
            print(f"epoch {epoch:5d} | train_mse={avg_loss:.4e} | lr={current_lr:.2e}")

    # If model was compiled, we need to access the original module to move it or save it properly if needed,
    # but .cpu() usually works on the wrapper too.
    return model.cpu()


def save_model(
    *,
    model: nn.Module,
    mu: np.ndarray,
    sigma: np.ndarray,
    device_params: DeviceParams,
    dither_params: DitherParams,
    feature_mode: str | None = None,
    feature_version: str = "legacy",
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state": model.state_dict(),
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "feature_mode": None if feature_mode is None else str(feature_mode),
        "feature_version": str(feature_version),
        "arch": {
            "in_dim": getattr(model, "in_dim", int(mu.shape[0])),
            "hidden": getattr(model, "hidden", 64),
            "depth": getattr(model, "depth", 3),
        },
        "device_params": device_params.__dict__,
        "dither_params": dither_params.__dict__,
    }
    torch.save(ckpt, path)


def load_model(path: str | Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    
    # Infer architecture if not fully specified (backward compatibility)
    arch = ckpt.get("arch", {})
    in_dim = int(arch.get("in_dim", 7))
    
    state_dict = ckpt["model_state"]
    
    # Try to infer hidden and depth from state_dict if not in arch
    if "hidden" not in arch or "depth" not in arch:
        # Infer hidden from first layer weight: [hidden, in_dim]
        if "net.0.weight" in state_dict:
            hidden = state_dict["net.0.weight"].shape[0]
        else:
            hidden = 64 # Fallback
            
        # Infer depth from max layer index
        max_idx = 0
        for k in state_dict.keys():
            if k.startswith("net."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    idx = int(parts[1])
                    if idx > max_idx:
                        max_idx = idx
        # Last layer index is 2*depth
        depth = max_idx // 2
    else:
        hidden = int(arch["hidden"])
        depth = int(arch["depth"])

    model = DeltaVPolicyNet(in_dim=in_dim, hidden=hidden, depth=depth)

    # Fix for torch.compile prefix if model was saved from a compiled state
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    meta = {
        "mu": np.asarray(ckpt["mu"], dtype=np.float32),
        "sigma": np.asarray(ckpt["sigma"], dtype=np.float32),
        "feature_mode": str(ckpt.get("feature_mode", _DEFAULT_FEATURE_MODE) or _DEFAULT_FEATURE_MODE),
        "feature_version": str(ckpt.get("feature_version", "legacy")),
        "device_params": DeviceParams(**ckpt["device_params"]),
        "dither_params": DitherParams(**ckpt["dither_params"]),
    }
    return model, meta


@torch.no_grad()
def rollout_dbm_hist(
    *,
    model: nn.Module,
    mu: np.ndarray,
    sigma: np.ndarray,
    device_params: DeviceParams,
    dither_params: DitherParams,
    theta_target_deg: float,
    V_init: float,
    steps: int = 60,
    accel: str = "auto",
    V_rf_amp: float = 0.0,
    f_rf: float = 1e9,
    max_step_V: float = 0.2,
    feature_mode: str = _DEFAULT_FEATURE_MODE,
    confidence_gate: bool = True,
    gate_ema_alpha: float = _CONF_GATE_EMA_ALPHA,
    gate_ratio: float = _CONF_GATE_RATIO,
    gate_min_scale: float = _CONF_GATE_MIN_SCALE,
    gate_power: float = _CONF_GATE_POWER,
) -> dict:
    """Single-trajectory rollout with optional RF signal.
    
    Parameters
    ----------
    V_rf_amp : float, optional
        RF signal amplitude (V). Default 0.0 (no RF signal).
    f_rf : float, optional
        RF signal frequency (Hz). Default 1e9 (1 GHz).
    """
    device = select_device(accel)

    # print(f"rollout_dbm_hist using device: {device}")

    model = model.to(device)

    # Precompute lock-in references once on the chosen device.
    refs = make_lockin_refs(dither_params, device)

    Vpi = float(device_params.Vpi_DC)
    th_t = float(np.deg2rad(theta_target_deg))

    V = float(np.clip(V_init, 0.0, Vpi))

    feature_mode_norm = str(feature_mode).lower().strip()
    if feature_mode_norm not in {"dc_norm_hist", "shape_norm_bessel", "theta_est_hist", "phi_bessel_hist"}:
        raise ValueError(
            "feature_mode must be one of: 'dc_norm_hist', 'shape_norm_bessel', 'theta_est_hist', 'phi_bessel_hist'"
        )

    prev_u1 = None
    prev_u2 = None
    prev_s = None
    prev_c = None
    prev_phi = None
    prev_dv = 0.0
    prev_snr = None

    V_hist: list[float] = []
    err_deg_hist: list[float] = []
    dv_hist: list[float] = []
    h1_hist: list[float] = []
    h2_hist: list[float] = []
    dh1_hist: list[float] = []
    dh2_hist: list[float] = []
    h1_raw_hist: list[float] = []
    h2_raw_hist: list[float] = []
    theta_deg_hist: list[float] = []
    theta_est_deg_hist: list[float] = []
    b_est_hist: list[float] = []
    phi_hist: list[float] = []
    dphi_hist: list[float] = []
    s_conf_hist: list[float] = []
    dv_scale_hist: list[float] = []

    te = _target_encoding(np.array([th_t], dtype=np.float32))[0]

    for _ in range(int(steps)):
        vb_t = torch.tensor([float(V)], device=device, dtype=torch.float32)
        h1_t, h2_t, _ = measure_pd_dither_normalized_batch_torch(
            V_bias=vb_t,
            V_dither_amp=float(dither_params.V_dither_amp),
            f_dither=float(dither_params.f_dither),
            Fs=float(dither_params.Fs),
            n_periods=int(dither_params.n_periods),
            Vpi_DC=float(device_params.Vpi_DC),
            ER_dB=float(device_params.ER_dB),
            IL_dB=float(device_params.IL_dB),
            Pin_dBm=float(device_params.Pin_dBm),
            Responsivity=float(device_params.Responsivity),
            R_load=float(device_params.R_load),
            refs=refs,
            V_rf_amp=float(V_rf_amp),
            f_rf=float(f_rf),
        )

        h1 = float(h1_t[0].item())
        h2 = float(h2_t[0].item())

        if feature_mode_norm == "dc_norm_hist":
            dh1 = 0.0 if prev_u1 is None else (h1 - float(prev_u1))
            dh2 = 0.0 if prev_u2 is None else (h2 - float(prev_u2))
            x0, x1, x2, x3 = h1, h2, dh1, dh2
            x = np.array([x0, x1, x2, x3, float(prev_dv), float(te[0]), float(te[1])], dtype=np.float32)
            theta_est_deg = float("nan")
            b_est = float("nan")
        elif feature_mode_norm == "shape_norm_bessel":
            u1, u2, _ = _bessel_equalized_shape_normalize_np(
                np.array([h1], dtype=np.float32),
                np.array([h2], dtype=np.float32),
                V_dither_amp=float(dither_params.V_dither_amp),
                Vpi_DC=float(device_params.Vpi_DC),
            )
            u1 = float(u1[0])
            u2 = float(u2[0])
            du1 = 0.0 if prev_u1 is None else (u1 - float(prev_u1))
            du2 = 0.0 if prev_u2 is None else (u2 - float(prev_u2))
            x0, x1, x2, x3 = u1, u2, du1, du2
            x = np.array([x0, x1, x2, x3, float(prev_dv), float(te[0]), float(te[1])], dtype=np.float32)
            theta_est_deg = float("nan")
            b_est = float("nan")
        elif feature_mode_norm == "phi_bessel_hist":
            beta_d = float(np.pi) * float(dither_params.V_dither_amp) / float(device_params.Vpi_DC)
            j1 = float(_bessel_j1_small(beta_d))
            j2 = float(_bessel_j2_small(beta_d))
            j1 = float(j1) if abs(j1) > 1e-12 else 1e-12
            j2 = float(j2) if abs(j2) > 1e-12 else 1e-12

            p = float(h1) / float(2.0 * j1)
            q = float(h2) / float(2.0 * j2)
            phi = float(np.arctan2(np.float32(p), np.float32(q)))
            snr_s = float(np.sqrt(np.float32(p) * np.float32(p) + np.float32(q) * np.float32(q)))
            dphi = 0.0 if prev_phi is None else float(phi - float(prev_phi))
            dphi = float(np.clip(dphi, -_DPHI_CLIP_RAD, _DPHI_CLIP_RAD))

            x = np.array([phi, dphi, float(prev_dv), float(te[0]), float(te[1])], dtype=np.float32)

            # For notebook/debug, reuse legacy slots.
            x0, x1, x2, x3 = phi, dphi, float("nan"), float("nan")
            theta_est_deg = float("nan")
            b_est = float("nan")
        else:
            theta_est, b_est_arr = _estimate_theta_from_harmonics_np(
                np.array([h1], dtype=np.float32),
                np.array([h2], dtype=np.float32),
                theta_prior_rad=np.array([bias_to_theta_rad(V, Vpi_DC=device_params.Vpi_DC)], dtype=np.float32),
                dither_params=dither_params,
                device_params=device_params,
            )
            theta_est = float(theta_est[0])
            b_est = float(b_est_arr[0])
            s = float(np.sin(theta_est))
            c = float(np.cos(theta_est))
            ds = 0.0 if prev_s is None else (s - float(prev_s))
            dc = 0.0 if prev_c is None else (c - float(prev_c))
            x0, x1, x2, x3 = s, c, ds, dc
            theta_est_deg = float(np.rad2deg(theta_est))
            x = np.array([x0, x1, x2, x3, float(prev_dv), float(te[0]), float(te[1])], dtype=np.float32)
        xn = (x - mu) / sigma

        dv = float(model(torch.from_numpy(xn).to(device).unsqueeze(0)).cpu().numpy().reshape(-1)[0])
        dv = float(np.clip(dv, -float(max_step_V), float(max_step_V)))
        dv_scale = 1.0
        if feature_mode_norm == "phi_bessel_hist":
            if prev_snr is None:
                prev_snr = float(snr_s)
            else:
                prev_snr = (1.0 - float(gate_ema_alpha)) * float(prev_snr) + float(gate_ema_alpha) * float(snr_s)
            if bool(confidence_gate):
                s_ref = float(max(1e-12, float(prev_snr) * float(gate_ratio)))
                dv_scale = float(np.clip(float(snr_s) / s_ref, 0.0, 1.0)) ** float(gate_power)
                dv_scale = float(max(float(gate_min_scale), float(dv_scale)))
                dv = float(np.clip(dv * dv_scale, -float(max_step_V), float(max_step_V)))
        V = float(np.clip(V + dv, 0.0, Vpi))

        if feature_mode_norm == "dc_norm_hist":
            prev_u1, prev_u2 = h1, h2
        elif feature_mode_norm == "shape_norm_bessel":
            prev_u1, prev_u2 = x0, x1
        elif feature_mode_norm == "phi_bessel_hist":
            prev_phi = float(x0)
        else:
            prev_s, prev_c = x0, x1
        prev_dv = dv

        th_c = float(bias_to_theta_rad(V, Vpi_DC=device_params.Vpi_DC))
        err = float(wrap_to_pi(th_t - th_c))

        V_hist.append(V)
        err_deg_hist.append(float(np.rad2deg(err)))
        dv_hist.append(dv)
        # Keep legacy key names for notebook/debug. Meanings depend on feature_mode.
        h1_hist.append(float(x0))
        h2_hist.append(float(x1))
        dh1_hist.append(float(x2))
        dh2_hist.append(float(x3))
        h1_raw_hist.append(h1)
        h2_raw_hist.append(h2)
        theta_deg_hist.append(float(np.rad2deg(th_c)))
        theta_est_deg_hist.append(float(theta_est_deg))
        b_est_hist.append(float(b_est))
        phi_hist.append(float(x0))
        dphi_hist.append(float(x1))
        s_conf_hist.append(float(snr_s) if feature_mode_norm == "phi_bessel_hist" else float("nan"))
        dv_scale_hist.append(float(dv_scale) if feature_mode_norm == "phi_bessel_hist" else float("nan"))

    out: dict = {
        "V": np.asarray(V_hist, dtype=float),
        "err_deg": np.asarray(err_deg_hist, dtype=float),
        "dv": np.asarray(dv_hist, dtype=float),
        "h1_norm": np.asarray(h1_hist, dtype=float),
        "h2_norm": np.asarray(h2_hist, dtype=float),
        "dh1_norm": np.asarray(dh1_hist, dtype=float),
        "dh2_norm": np.asarray(dh2_hist, dtype=float),
        "h1_norm_raw": np.asarray(h1_raw_hist, dtype=float),
        "h2_norm_raw": np.asarray(h2_raw_hist, dtype=float),
        "theta_deg": np.asarray(theta_deg_hist, dtype=float),
        "theta_est_deg": np.asarray(theta_est_deg_hist, dtype=float),
        "b_est": np.asarray(b_est_hist, dtype=float),
    }
    if feature_mode_norm == "phi_bessel_hist":
        out["phi"] = np.asarray(phi_hist, dtype=float)
        out["dphi"] = np.asarray(dphi_hist, dtype=float)
        out["s_conf"] = np.asarray(s_conf_hist, dtype=float)
        out["dv_scale"] = np.asarray(dv_scale_hist, dtype=float)
    return out


@torch.no_grad()
def rollout_dbm_hist_batch(
    *,
    model: nn.Module,
    mu: np.ndarray,
    sigma: np.ndarray,
    device_params: DeviceParams,
    dither_params: DitherParams,
    theta_target_deg: np.ndarray,
    V_init: np.ndarray,
    steps: int = 60,
    accel: str = "auto",
    V_rf_amp: float = 0.0,
    f_rf: float = 1e9,
    Pin_dBm: float | None = None,
    max_step_V: float = 0.2,
    feature_mode: str = _DEFAULT_FEATURE_MODE,
    confidence_gate: bool = True,
    gate_ema_alpha: float = _CONF_GATE_EMA_ALPHA,
    gate_ratio: float = _CONF_GATE_RATIO,
    gate_min_scale: float = _CONF_GATE_MIN_SCALE,
    gate_power: float = _CONF_GATE_POWER,
) -> dict:
    """Batch rollout with optional RF signal and optical power override for robustness testing.
    
    Parameters
    ----------
    V_rf_amp : float, optional
        RF signal amplitude (V). Default 0.0 (no RF signal).
        Use this to test model robustness under RF modulation.
    f_rf : float, optional
        RF signal frequency (Hz). Default 1e9 (1 GHz).
    Pin_dBm : float, optional
        Input optical power (dBm). If None, uses device_params.Pin_dBm.
        Use this to test model robustness under optical power fluctuations.
    """
    device = select_device(accel)

    # print(f"rollout_dbm_hist_batch using device: {device}")

    model = model.to(device)

    # Convert inputs to tensors on device
    V = torch.from_numpy(V_init).to(device=device, dtype=torch.float32)
    th_t_rad = torch.deg2rad(torch.from_numpy(theta_target_deg).to(device=device, dtype=torch.float32))

    # Constants
    Vpi = float(device_params.Vpi_DC)
    mu_t = torch.from_numpy(mu).to(device=device, dtype=torch.float32)
    sigma_t = torch.from_numpy(sigma).to(device=device, dtype=torch.float32)

    # Precompute lock-in references
    refs = make_lockin_refs(dither_params, device)

    feature_mode_norm = str(feature_mode).lower().strip()
    if feature_mode_norm not in {"dc_norm_hist", "shape_norm_bessel", "theta_est_hist", "phi_bessel_hist"}:
        raise ValueError(
            "feature_mode must be one of: 'dc_norm_hist', 'shape_norm_bessel', 'theta_est_hist', 'phi_bessel_hist'"
        )

    # State initialization
    prev_u1 = None
    prev_u2 = None
    prev_s = None
    prev_c = None
    prev_phi = None
    prev_dv = torch.zeros_like(V)
    prev_snr = None

    # History storage (on CPU to save GPU memory if steps are large, or keep on GPU?)
    # Keeping on CPU is safer for large batches.
    err_deg_hist = []

    # Target encoding (constant throughout rollout)
    # sin(th), cos(th)
    te_sin = torch.sin(th_t_rad)
    te_cos = torch.cos(th_t_rad)

    # Use provided Pin_dBm or default from device_params
    actual_Pin_dBm = Pin_dBm if Pin_dBm is not None else float(device_params.Pin_dBm)

    for _ in range(int(steps)):
        # Measure DC-normalized features
        h1, h2, _ = measure_pd_dither_normalized_batch_torch(
            V_bias=V,
            V_dither_amp=float(dither_params.V_dither_amp),
            f_dither=float(dither_params.f_dither),
            Fs=float(dither_params.Fs),
            n_periods=int(dither_params.n_periods),
            Vpi_DC=float(device_params.Vpi_DC),
            ER_dB=float(device_params.ER_dB),
            IL_dB=float(device_params.IL_dB),
            Pin_dBm=actual_Pin_dBm,
            Responsivity=float(device_params.Responsivity),
            R_load=float(device_params.R_load),
            refs=refs,
            V_rf_amp=float(V_rf_amp),
            f_rf=float(f_rf),
        )

        if feature_mode_norm == "dc_norm_hist":
            x0 = h1
            x1 = h2
            if prev_u1 is None:
                x2 = torch.zeros_like(h1)
                x3 = torch.zeros_like(h2)
            else:
                x2 = h1 - prev_u1
                x3 = h2 - prev_u2
            prev_next_0 = h1
            prev_next_1 = h2
        elif feature_mode_norm == "shape_norm_bessel":
            u1, u2, _ = _bessel_equalized_shape_normalize_torch(
                h1,
                h2,
                V_dither_amp=float(dither_params.V_dither_amp),
                Vpi_DC=float(device_params.Vpi_DC),
            )
            if prev_u1 is None:
                du1 = torch.zeros_like(u1)
                du2 = torch.zeros_like(u2)
            else:
                du1 = u1 - prev_u1
                du2 = u2 - prev_u2
            x0, x1, x2, x3 = u1, u2, du1, du2
            prev_next_0 = u1
            prev_next_1 = u2
        elif feature_mode_norm == "phi_bessel_hist":
            beta_d = float(np.pi) * float(dither_params.V_dither_amp) / float(device_params.Vpi_DC)
            j1 = float(_bessel_j1_small(beta_d))
            j2 = float(_bessel_j2_small(beta_d))
            j1 = j1 if abs(j1) > 1e-12 else 1e-12
            j2 = j2 if abs(j2) > 1e-12 else 1e-12
            p = h1 / float(2.0 * j1)
            q = h2 / float(2.0 * j2)
            phi = torch.atan2(p, q)
            if prev_phi is None:
                dphi = torch.zeros_like(phi)
            else:
                dphi = phi - prev_phi
            dphi = torch.clamp(dphi, -_DPHI_CLIP_RAD, _DPHI_CLIP_RAD)
            prev_phi = phi
            snr_s = torch.sqrt(p * p + q * q + 1e-24)
            if prev_snr is None:
                prev_snr = snr_s
            else:
                prev_snr = (1.0 - float(gate_ema_alpha)) * prev_snr + float(gate_ema_alpha) * snr_s
        else:
            theta_est, _ = _estimate_theta_from_harmonics_torch(
                h1,
                h2,
                theta_prior_rad=(V / Vpi) * float(np.pi),
                dither_params=dither_params,
                device_params=device_params,
            )
            s = torch.sin(theta_est)
            c = torch.cos(theta_est)
            if prev_s is None:
                ds = torch.zeros_like(s)
                dc = torch.zeros_like(c)
            else:
                ds = s - prev_s
                dc = c - prev_c
            x0, x1, x2, x3 = s, c, ds, dc
            prev_next_0 = s
            prev_next_1 = c

        if feature_mode_norm == "phi_bessel_hist":
            x = torch.stack([phi, dphi, prev_dv, te_sin, te_cos], dim=1)
        else:
            x = torch.stack([x0, x1, x2, x3, prev_dv, te_sin, te_cos], dim=1)
        xn = (x - mu_t) / sigma_t

        # Inference
        dv = model(xn).squeeze(1)  # (N, 1) -> (N,)
        dv = torch.clamp(dv, -float(max_step_V), float(max_step_V))
        if feature_mode_norm == "phi_bessel_hist" and bool(confidence_gate):
            s_ref = torch.clamp(prev_snr * float(gate_ratio), min=1e-12)
            dv_scale = torch.clamp(snr_s / s_ref, 0.0, 1.0) ** float(gate_power)
            dv_scale = torch.clamp(dv_scale, min=float(gate_min_scale), max=1.0)
            dv = torch.clamp(dv * dv_scale, -float(max_step_V), float(max_step_V))

        # Update
        V = torch.clamp(V + dv, 0.0, Vpi)

        # Update state
        if feature_mode_norm in {"dc_norm_hist", "shape_norm_bessel"}:
            prev_u1 = prev_next_0
            prev_u2 = prev_next_1
        elif feature_mode_norm == "theta_est_hist":
            prev_s = prev_next_0
            prev_c = prev_next_1
        prev_dv = dv

        # Calculate error for history
        # We need bias_to_theta_rad but vectorized.
        # bias_to_theta_rad is: (V / Vpi_DC) * pi
        th_c = (V / Vpi) * np.pi

        # wrap_to_pi: (angle + pi) % (2*pi) - pi
        # But here we want simple difference wrapped.
        diff = th_t_rad - th_c
        err = (diff + np.pi) % (2 * np.pi) - np.pi

        err_deg_hist.append(torch.rad2deg(err).cpu().numpy())

    # Stack history: (steps, N) -> (N, steps)
    err_deg_hist = np.stack(err_deg_hist, axis=1)

    return {"err_deg": err_deg_hist}


def run_pipeline() -> int:
    """Run the default dataset->train->rollout pipeline."""

    artifacts = Path("artifacts")
    ds_path = artifacts / "dither_dataset_dbm_hist.npz"
    model_path = artifacts / "dither_policy_dbm_hist.pt"

    device_params = DeviceParams()
    dither_params = DitherParams()

    # 1) dataset (reusable)
    if not ds_path.exists():
        print(f"generating dataset: {ds_path}")
        ds = generate_dataset_dbm_hist(
            device_params=device_params,
            dither_params=dither_params,
            n_samples=8000,
            seed=0,
            teacher_gain=0.5,
            max_step_V=0.2,
        )
        save_dataset(ds, ds_path)
    else:
        print(f"using existing dataset: {ds_path}")

    ds = load_dataset(ds_path)

    # 2) train (reusable)
    print(f"training model -> {model_path}")
    model = train_policy(Xn=ds["Xn"], y=ds["y"], epochs=2000)
    save_model(
        model=model,
        mu=ds["mu"],
        sigma=ds["sigma"],
        device_params=ds["device_params"],
        dither_params=ds["dither_params"],
        path=model_path,
    )

    # 3) inference / rollout sanity
    model, meta = load_model(model_path)
    rng = np.random.default_rng(1)
    V_init = float(rng.uniform(0.0, float(meta["device_params"].Vpi_DC)))
    for tgt in [0.0, 30.0, 90.0, 150.0, 180.0]:
        r = rollout_dbm_hist(
            model=model,
            mu=meta["mu"],
            sigma=meta["sigma"],
            device_params=meta["device_params"],
            dither_params=meta["dither_params"],
            theta_target_deg=float(tgt),
            V_init=V_init,
            steps=60,
            accel="auto",
        )
        final_err = float(r["err_deg"][-1]) if r["err_deg"].size else float("nan")
        print(f"rollout target={tgt:6.1f} deg | initV={V_init:.3f} V | final_err={final_err:+.2f} deg")

    return 0


def main() -> int:
    return run_pipeline()
