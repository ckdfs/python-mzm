"""Dither-based closed-loop MZM bias controller (reusable module).

What this implements
- You can ONLY observe PD low-frequency pilot (dither) 1f/2f power (dBm).
- You do NOT use DC optical power.
- You do NOT assume signed I/Q lock-in outputs.

To recover direction (which is lost if you only use power magnitudes), the controller
uses a *history finite difference* (more realistic than taking an extra probe sample):
    x_k = [P1_dBm(k), P2_dBm(k), dP1_dBm, dP2_dBm, dV_{k-1}, sin(theta*), cos(theta*)]
where dP*_dBm = P*_dBm(k) - P*_dBm(k-1).

The policy outputs an incremental update ΔV.
Bias is clamped to [0, Vpi] so the target range is 0..180 degrees.

Artifacts
- Dataset: artifacts/dither_dataset_dbm_hist.npz
- Model:   artifacts/dither_policy_dbm_hist.pt

NOTE
- This file is intended to be imported (Notebook calls it). It does not act as a CLI script.
- For a CLI entry, use scripts/train_mzm_dither_controller.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mzm.model import measure_pd_dither_1f2f, bias_to_theta_rad, theta_to_bias_V, wrap_to_pi


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
) -> dict:
    """Generate supervised dataset for realistic dbm_hist controller.

    Returns a dict containing Xn, y, and normalization stats.
    """

    rng = np.random.default_rng(seed)

    Vpi = float(device_params.Vpi_DC)

    # Current bias in [0, Vpi] -> theta in [0, pi]
    V_bias = rng.uniform(0.0, Vpi, size=n_samples).astype(np.float32)

    # Target theta in [0, pi] (0..180 deg)
    theta_target = rng.uniform(0.0, float(np.pi), size=n_samples).astype(np.float32)

    X = np.zeros((n_samples, 7), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)

    for i in range(n_samples):
        vb = float(V_bias[i])
        th_t = float(theta_target[i])

        # Create a "previous" step consistent with a realistic controller history.
        dv_prev = float(rng.uniform(-max_step_V, max_step_V))
        v_prev = float(np.clip(vb - dv_prev, 0.0, Vpi))

        meas_prev = measure_pd_dither_1f2f(
            V_bias=v_prev,
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
        )
        meas = measure_pd_dither_1f2f(
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
        )

        p1 = float(meas["p1_dBm"])
        p2 = float(meas["p2_dBm"])
        dp1 = p1 - float(meas_prev["p1_dBm"])
        dp2 = p2 - float(meas_prev["p2_dBm"])
        te = _target_encoding(np.array([th_t], dtype=np.float32))[0]

        X[i, :] = np.array([p1, p2, dp1, dp2, dv_prev, te[0], te[1]], dtype=np.float32)

        # Teacher label: proportional step in wrapped phase error
        th_c = float(bias_to_theta_rad(vb, Vpi_DC=device_params.Vpi_DC))
        err = float(wrap_to_pi(th_t - th_c))
        dv = float(teacher_gain) * float(theta_to_bias_V(err, Vpi_DC=device_params.Vpi_DC))
        dv = float(np.clip(dv, -max_step_V, max_step_V))
        y[i, 0] = np.float32(dv)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    Xn = (X - mu) / sigma

    return {
        "Xn": Xn,
        "y": y,
        "mu": mu,
        "sigma": sigma,
        "device_params": device_params,
        "dither_params": dither_params,
        "teacher_gain": float(teacher_gain),
        "max_step_V": float(max_step_V),
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
        device_params=np.array(list(dataset["device_params"].__dict__.items()), dtype=object),
        dither_params=np.array(list(dataset["dither_params"].__dict__.items()), dtype=object),
        teacher_gain=np.array(dataset["teacher_gain"], dtype=np.float32),
        max_step_V=np.array(dataset["max_step_V"], dtype=np.float32),
    )


def load_dataset(path: str | Path) -> dict:
    z = np.load(Path(path), allow_pickle=True)

    device_params = DeviceParams(**dict(z["device_params"]))
    dither_params = DitherParams(**dict(z["dither_params"]))

    return {
        "Xn": z["Xn"].astype(np.float32),
        "y": z["y"].astype(np.float32),
        "mu": z["mu"].astype(np.float32),
        "sigma": z["sigma"].astype(np.float32),
        "device_params": device_params,
        "dither_params": dither_params,
        "teacher_gain": float(z["teacher_gain"]),
        "max_step_V": float(z["max_step_V"]),
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
) -> nn.Module:
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

    model = DeltaVPolicyNet(in_dim=Xn.shape[1], hidden=hidden, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)

        if epoch == 1 or epoch % 200 == 0 or epoch == int(epochs):
            print(f"epoch {epoch:5d} | train_mse={running / len(ds):.4e}")

    return model.cpu()


def save_model(
    *,
    model: nn.Module,
    mu: np.ndarray,
    sigma: np.ndarray,
    device_params: DeviceParams,
    dither_params: DitherParams,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state": model.state_dict(),
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "arch": {"in_dim": int(mu.shape[0])},
        "device_params": device_params.__dict__,
        "dither_params": dither_params.__dict__,
    }
    torch.save(ckpt, path)


def load_model(path: str | Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(Path(path), map_location="cpu")
    model = DeltaVPolicyNet(in_dim=int(ckpt["arch"]["in_dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta = {
        "mu": np.asarray(ckpt["mu"], dtype=np.float32),
        "sigma": np.asarray(ckpt["sigma"], dtype=np.float32),
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
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    Vpi = float(device_params.Vpi_DC)
    th_t = float(np.deg2rad(theta_target_deg))

    V = float(np.clip(V_init, 0.0, Vpi))

    prev_p1 = None
    prev_p2 = None
    prev_dv = 0.0

    V_hist: list[float] = []
    err_deg_hist: list[float] = []
    dv_hist: list[float] = []
    p1_hist: list[float] = []
    p2_hist: list[float] = []
    dp1_hist: list[float] = []
    dp2_hist: list[float] = []
    theta_deg_hist: list[float] = []

    for _ in range(int(steps)):
        meas = measure_pd_dither_1f2f(
            V_bias=float(V),
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
        )

        p1 = float(meas["p1_dBm"])
        p2 = float(meas["p2_dBm"])
        dp1 = 0.0 if prev_p1 is None else (p1 - float(prev_p1))
        dp2 = 0.0 if prev_p2 is None else (p2 - float(prev_p2))

        te = _target_encoding(np.array([th_t], dtype=np.float32))[0]
        x = np.array([p1, p2, dp1, dp2, float(prev_dv), float(te[0]), float(te[1])], dtype=np.float32)
        xn = (x - mu) / sigma

        dv = float(model(torch.from_numpy(xn).to(device).unsqueeze(0)).cpu().numpy().reshape(-1)[0])
        V = float(np.clip(V + dv, 0.0, Vpi))

        prev_p1, prev_p2 = p1, p2
        prev_dv = dv

        th_c = float(bias_to_theta_rad(V, Vpi_DC=device_params.Vpi_DC))
        err = float(wrap_to_pi(th_t - th_c))

        V_hist.append(V)
        err_deg_hist.append(float(np.rad2deg(err)))
        dv_hist.append(dv)
        p1_hist.append(p1)
        p2_hist.append(p2)
        dp1_hist.append(dp1)
        dp2_hist.append(dp2)
        theta_deg_hist.append(float(np.rad2deg(th_c)))

    return {
        "V": np.asarray(V_hist, dtype=float),
        "err_deg": np.asarray(err_deg_hist, dtype=float),
        "dv": np.asarray(dv_hist, dtype=float),
        "p1_dBm": np.asarray(p1_hist, dtype=float),
        "p2_dBm": np.asarray(p2_hist, dtype=float),
        "dp1_dBm": np.asarray(dp1_hist, dtype=float),
        "dp2_dBm": np.asarray(dp2_hist, dtype=float),
        "theta_deg": np.asarray(theta_deg_hist, dtype=float),
    }


def main() -> int:
    """Default pipeline used by the CLI runner in scripts/."""

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
        )
        final_err = float(r["err_deg"][-1]) if r["err_deg"].size else float("nan")
        print(f"rollout target={tgt:6.1f} deg | initV={V_init:.3f} V | final_err={final_err:+.2f} deg")

    return 0
