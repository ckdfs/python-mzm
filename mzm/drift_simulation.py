"""Drift simulation module for MZM control loop.

This module provides tools to simulate the closed-loop controller's behavior
under various drift conditions (bias drift, Vpi drift) to verify robustness.
Typically used after training a model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import replace
from .model import (
    _measure_pd_dither_1f2f_batch_torch,
    measure_pd_dither_normalized_batch_torch,
    bias_to_theta_rad,
    wrap_to_pi,
)
from .dither_controller import DeviceParams, DitherParams, _estimate_theta_from_harmonics_np
from .utils import make_lockin_refs


def _estimate_theta_signed_from_lockin_np(
    *,
    h1_I_norm: float,
    h2_Q_norm: float,
    dither_params: DitherParams,
    Vpi_DC: float,
    eps: float = 1e-12,
) -> float:
    beta_d = float(np.pi) * float(dither_params.V_dither_amp) / float(Vpi_DC)
    j1 = beta_d / 2.0 - (beta_d**3) / 16.0 + (beta_d**5) / 384.0
    j2 = (beta_d**2) / 8.0 - (beta_d**4) / 96.0 + (beta_d**6) / 3072.0
    j1 = float(j1) if abs(float(j1)) > 1e-12 else (1e-12 if j1 >= 0 else -1e-12)
    j2 = float(j2) if abs(float(j2)) > 1e-12 else (1e-12 if j2 >= 0 else -1e-12)

    p = (-float(h1_I_norm)) / (2.0 * j1 + float(eps))
    q = (float(h2_Q_norm)) / (2.0 * j2 + float(eps))
    theta = float(np.arctan2(p, q))
    if theta < 0.0:
        theta = theta + float(np.pi)
    return float(np.clip(theta, 0.0, float(np.pi)))

def get_bias_drift(t, step_rate=0.002):
    """
    Generate a bias drift value at time step t.
    Models unidirectional drift (linear ramp) which shifts the MZM transfer curve.
    """
    # Linear drift only
    linear = step_rate * t
    return linear

def get_vpi_true(
    t: int,
    *,
    Vpi0: float,
    rel_step_rate: float = 0.0,
    abs_step_rate: float = 0.0,
    min_Vpi: float = 1e-3,
) -> float:
    """Generate a Vpi value at time step t.

    Vpi drift is modeled either as:
    - multiplicative: Vpi(t) = Vpi0 * (1 + rel_step_rate * t)
    - additive:       Vpi(t) = Vpi0 + abs_step_rate * t
    """

    if float(rel_step_rate) != 0.0 and float(abs_step_rate) != 0.0:
        raise ValueError("Specify only one of rel_step_rate or abs_step_rate")

    Vpi0 = float(Vpi0)
    tt = float(t)
    if float(abs_step_rate) != 0.0:
        Vpi_t = Vpi0 + float(abs_step_rate) * tt
    else:
        Vpi_t = Vpi0 * (1.0 + float(rel_step_rate) * tt)

    return float(max(float(min_Vpi), float(Vpi_t)))

def simulate_control_loop_with_drift(
    model,
    mu,
    sigma,
    device_params,
    dither_params,
    n_steps=200,
    target_theta_deg=90.0,
    max_step_V=0.2,
    drift_step_rate=0.002
):
    """
    Simulate the closed-loop control under bias drift using a trained model.
    Feature Mode: 'theta_est_hist'
    """
    print(f"\\n--- Starting Bias Drift Simulation (Steps: {n_steps}) ---")
    print(f"Target Theta: {target_theta_deg} deg")
    print(f"Dither Amp: {dither_params.V_dither_amp} V")
    print(f"Drift Rate: {drift_step_rate} V/step")
    
    # Simulation state
    # Start at 'perfect' guess for 2.5V (Quadrature) if target is 90
    V_bias_current = 2.5 # Initial controller guess
    theta_target = np.radians(target_theta_deg)
    
    # Target state encoding (sin, cos)
    target_sin = np.sin(theta_target)
    target_cos = np.cos(theta_target)
    
    # History state
    prev_s = 0.0
    prev_c = 0.0
    prev_dv = 0.0
    prev_theta_est = None
    
    # Logging
    history = []
    
    model.eval()
    
    # Pre-convert stats to tensor
    if isinstance(mu, np.ndarray):
        mu_t = torch.from_numpy(mu)
    else:
        mu_t = mu
        
    if isinstance(sigma, np.ndarray):
        sigma_t = torch.from_numpy(sigma)
    else:
        sigma_t = sigma

    # Precompute lock-in references once (significantly reduces per-step overhead).
    refs = make_lockin_refs(dither_params, torch.device("cpu"))
    
    for t in range(n_steps):
        # 1. Calculate Shift
        v_drift = get_bias_drift(t, step_rate=drift_step_rate)
        
        # 2. Measurement (Physics Simulation)
        v_batch = torch.tensor([V_bias_current], dtype=torch.float32)
        drift_batch = torch.tensor([v_drift], dtype=torch.float32)
        
        with torch.no_grad():
            out = _measure_pd_dither_1f2f_batch_torch(
                V_bias=v_batch.to(dtype=torch.float64),
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
                V_rf_amp=0.0,
                V_drift=drift_batch.to(dtype=torch.float64),
            )

            pd_dc = float(out["pd_dc"][0].item())
            h1_I = float(out["h1_I"][0].item())
            h2_Q = float(out["h2_Q"][0].item())

            h1_I_norm = h1_I / (pd_dc + 1e-12)
            h2_Q_norm = h2_Q / (pd_dc + 1e-12)

            theta_est = _estimate_theta_signed_from_lockin_np(
                h1_I_norm=h1_I_norm,
                h2_Q_norm=h2_Q_norm,
                dither_params=dither_params,
                Vpi_DC=float(device_params.Vpi_DC),
            )
            
            # 2. Compute Sin/Cos features
            s = np.sin(theta_est)
            c = np.cos(theta_est)
            
            # 3. Compute Delta Sin/Cos
            if t == 0:
                # Initialize history with current state to avoid jump
                prev_s = s
                prev_c = c
                
            ds = s - prev_s
            dc = c - prev_c
            
            # 4. Construct Feature Vector
            # [s, c, ds, dc, dv_prev, target_sin, target_cos]
            obs_list = [
                s,
                c,
                ds,
                dc,
                prev_dv,
                target_sin, 
                target_cos
            ]
            
            obs_raw = torch.tensor([obs_list], dtype=torch.float32) # [1, 7]
            
            # Normalize
            obs_norm = (obs_raw - mu_t) / sigma_t
            
            # 5. Controller Action
            delta_v = model(obs_norm)[0].item()
            
        # Clip step size
        delta_v_clipped = np.clip(delta_v, -max_step_V, max_step_V)
        
        # Update Controller
        V_bias_current += delta_v_clipped
        
        # Update History
        prev_s = s
        prev_c = c
        prev_dv = delta_v_clipped # Use the action we actually took
        
        # --- Logging ---
        v_actual = V_bias_current + v_drift
        theta_actual = bias_to_theta_rad(v_actual, Vpi_DC=device_params.Vpi_DC)
        phase_err = np.degrees(wrap_to_pi(theta_actual - theta_target))
        
        history.append({
            'step': t,
            'v_drift': v_drift,
            'v_ctrl': V_bias_current,
            'v_actual': v_actual,
            'theta_err': phase_err,
            'theta_est_deg': np.degrees(theta_est),
            'h1': float(h1_I_norm),
            'h2': float(h2_Q_norm),
        })
        
    return history

def simulate_control_loop_with_vpi_drift(
    model,
    mu,
    sigma,
    device_params: DeviceParams,
    dither_params: DitherParams,
    *,
    n_steps: int = 200,
    target_theta_deg: float = 90.0,
    max_step_V: float = 0.2,
    vpi_rel_step_rate: float = 0.0,
    vpi_abs_step_rate: float = 0.0,
    controller_knows_vpi: bool = False,
    clamp_bias: bool = True,
) -> list[dict]:
    """Simulate the closed-loop control under Vpi drift using a trained model.

    This intentionally separates:
    - Vpi_true(t): used by the *physics* measurement model
    - Vpi_used(t): used by the *controller/estimator* (theta prior + inversion)
    """

    Vpi_nom = float(device_params.Vpi_DC)

    print(f"\\n--- Starting Vpi Drift Simulation (Steps: {n_steps}) ---")
    print(f"Target Theta: {target_theta_deg} deg")
    print(f"Dither Amp: {dither_params.V_dither_amp} V")
    if float(vpi_abs_step_rate) != 0.0:
        print(f"Vpi Drift: {vpi_abs_step_rate} V/step")
    else:
        print(f"Vpi Drift: {vpi_rel_step_rate} (relative)/step")
    print(f"Controller knows Vpi: {controller_knows_vpi}")

    theta_target = np.radians(float(target_theta_deg))
    target_sin = float(np.sin(theta_target))
    target_cos = float(np.cos(theta_target))

    # Controller state
    V_bias_current = Vpi_nom / 2.0  # start near quadrature in nominal calibration

    prev_s = 0.0
    prev_c = 0.0
    prev_dv = 0.0
    history: list[dict] = []

    model.eval()

    if isinstance(mu, np.ndarray):
        mu_t = torch.from_numpy(mu)
    else:
        mu_t = mu

    if isinstance(sigma, np.ndarray):
        sigma_t = torch.from_numpy(sigma)
    else:
        sigma_t = sigma

    # Precompute lock-in references once.
    refs = make_lockin_refs(dither_params, torch.device("cpu"))

    for t in range(int(n_steps)):
        Vpi_true = get_vpi_true(
            t,
            Vpi0=Vpi_nom,
            rel_step_rate=float(vpi_rel_step_rate),
            abs_step_rate=float(vpi_abs_step_rate),
            min_Vpi=1e-3,
        )
        Vpi_used = float(Vpi_true) if bool(controller_knows_vpi) else float(Vpi_nom)

        v_batch = torch.tensor([float(V_bias_current)], dtype=torch.float32)

        with torch.no_grad():
            # Physics measurement under Vpi_true
            h1_t, h2_t, _ = measure_pd_dither_normalized_batch_torch(
                V_bias=v_batch,
                V_dither_amp=dither_params.V_dither_amp,
                f_dither=dither_params.f_dither,
                Fs=dither_params.Fs,
                n_periods=dither_params.n_periods,
                Vpi_DC=float(Vpi_true),
                ER_dB=device_params.ER_dB,
                IL_dB=device_params.IL_dB,
                Pin_dBm=device_params.Pin_dBm,
                Responsivity=device_params.Responsivity,
                R_load=device_params.R_load,
                refs=refs,
                V_rf_amp=0.0,
                V_drift=0.0,
            )
            h1 = float(h1_t.item())
            h2 = float(h2_t.item())

            # Estimation prior uses Vpi_used
            theta_prior = float(bias_to_theta_rad(V_bias_current, Vpi_DC=Vpi_used))

            # Theta inversion uses Vpi_used (affects beta_d)
            device_params_used = (
                device_params if float(device_params.Vpi_DC) == float(Vpi_used) else replace(device_params, Vpi_DC=Vpi_used)
            )
            theta_est_rad, _ = _estimate_theta_from_harmonics_np(
                np.array([h1]),
                np.array([h2]),
                theta_prior_rad=np.array([theta_prior]),
                dither_params=dither_params,
                device_params=device_params_used,
            )
            theta_est = float(theta_est_rad[0])

            s = float(np.sin(theta_est))
            c = float(np.cos(theta_est))

            if t == 0:
                prev_s = s
                prev_c = c

            ds = s - prev_s
            dc = c - prev_c

            obs_list = [s, c, ds, dc, prev_dv, target_sin, target_cos]
            obs_raw = torch.tensor([obs_list], dtype=torch.float32)  # [1, 7]
            obs_norm = (obs_raw - mu_t) / sigma_t
            delta_v = float(model(obs_norm)[0].item())

        delta_v_clipped = float(np.clip(delta_v, -float(max_step_V), float(max_step_V)))
        V_bias_current = float(V_bias_current + delta_v_clipped)
        if bool(clamp_bias):
            V_bias_current = float(np.clip(V_bias_current, 0.0, Vpi_nom))

        prev_s = s
        prev_c = c
        prev_dv = delta_v_clipped

        theta_actual = float(bias_to_theta_rad(V_bias_current, Vpi_DC=float(Vpi_true)))
        phase_err = float(np.degrees(wrap_to_pi(theta_actual - float(theta_target))))

        history.append(
            {
                "step": int(t),
                "v_ctrl": float(V_bias_current),
                "dv": float(delta_v_clipped),
                "vpi_nom": float(Vpi_nom),
                "vpi_true": float(Vpi_true),
                "vpi_used": float(Vpi_used),
                "vpi_ratio": float(Vpi_true / Vpi_nom),
                "theta_err": float(phase_err),
                "theta_est_deg": float(np.degrees(theta_est)),
                "theta_prior_deg": float(np.degrees(theta_prior)),
                "h1": float(h1),
                "h2": float(h2),
            }
        )

    return history

def simulate_adaptive_control_loop(
    model,
    mu,
    sigma,
    device_params: DeviceParams,
    dither_params: DitherParams,
    *,
    n_steps: int = 200,
    target_theta_deg: float = 90.0,
    max_step_V: float = 0.2,
    vpi_rel_step_rate: float = 0.0,
    vpi_abs_step_rate: float = 0.0,
    clamp_bias: bool = True,
    est_alpha: float = 0.05,
    conf_margin_rad: float = 0.01,
    conf_h2_frac: float = 0.08,
    conf_h2_min_abs: float = 1e-6,
    update_max_rel: float = 0.05,
    buffer_len: int = 5,
) -> list[dict]:
    """Simulate adaptive control loop with online Vpi estimation via grid search."""

    Vpi_nom = float(device_params.Vpi_DC)

    print(f"\n--- Starting Adaptive Vpi Control (Steps: {n_steps}) ---")
    print(f"Target Theta: {target_theta_deg} deg")
    if float(vpi_abs_step_rate) != 0.0:
        print(f"Vpi Drift: {vpi_abs_step_rate} V/step")
    else:
        print(f"Vpi Drift: {vpi_rel_step_rate} (relative)/step")

    theta_target = np.radians(float(target_theta_deg))
    target_sin = float(np.sin(theta_target))
    target_cos = float(np.cos(theta_target))

    V_bias_current = Vpi_nom / 2.0
    Vpi_est = Vpi_nom

    prev_s = 0.0
    prev_c = 0.0
    prev_dv = 0.0
    h2_abs_ema = 0.0
    vpi_buf: deque[float] = deque(maxlen=int(buffer_len))

    history: list[dict] = []

    model.eval()

    if isinstance(mu, np.ndarray):
        mu_t = torch.from_numpy(mu)
    else:
        mu_t = mu

    if isinstance(sigma, np.ndarray):
        sigma_t = torch.from_numpy(sigma)
    else:
        sigma_t = sigma

    refs = make_lockin_refs(dither_params, torch.device("cpu"))

    vpi_min = 0.5 * Vpi_nom
    vpi_max = 3.0 * Vpi_nom
    vpi_grid = np.linspace(vpi_min, vpi_max, 61, dtype=np.float32)

    for t in range(int(n_steps)):
        Vpi_true = get_vpi_true(
            t,
            Vpi0=Vpi_nom,
            rel_step_rate=float(vpi_rel_step_rate),
            abs_step_rate=float(vpi_abs_step_rate),
            min_Vpi=1e-3,
        )

        v_batch = torch.tensor([float(V_bias_current)], dtype=torch.float32)

        with torch.no_grad():
            h1_t, h2_t, _ = measure_pd_dither_normalized_batch_torch(
                V_bias=v_batch,
                V_dither_amp=dither_params.V_dither_amp,
                f_dither=dither_params.f_dither,
                Fs=dither_params.Fs,
                n_periods=dither_params.n_periods,
                Vpi_DC=float(Vpi_true),
                ER_dB=device_params.ER_dB,
                IL_dB=device_params.IL_dB,
                Pin_dBm=device_params.Pin_dBm,
                Responsivity=device_params.Responsivity,
                R_load=device_params.R_load,
                refs=refs,
                V_rf_amp=0.0,
                V_drift=0.0,
            )
            h1 = float(h1_t.item())
            h2 = float(h2_t.item())

            best_vpi = float(Vpi_est)
            best_cost = float("inf")
            second_cost = float("inf")
            best_theta = None

            for vpi_cand in vpi_grid:
                dp = replace(device_params, Vpi_DC=float(vpi_cand))
                theta_prior = float(bias_to_theta_rad(V_bias_current, Vpi_DC=float(vpi_cand)))
                theta_est_rad, _ = _estimate_theta_from_harmonics_np(
                    np.array([h1], dtype=np.float32),
                    np.array([h2], dtype=np.float32),
                    theta_prior_rad=np.array([theta_prior], dtype=np.float32),
                    dither_params=dither_params,
                    device_params=dp,
                )
                theta_cand = float(theta_est_rad[0])
                cost = abs(float(wrap_to_pi(theta_cand - theta_prior)))
                if cost < best_cost:
                    second_cost = best_cost
                    best_cost = cost
                    best_vpi = float(vpi_cand)
                    best_theta = theta_cand
                elif cost < second_cost:
                    second_cost = cost

            # Confidence gating: update only when the estimate is identifiable.
            # - Near quadrature (theta≈90°), h2 tends to be very small and Vpi becomes weakly observable.
            # - The grid-search cost curve becomes flat; we use the gap between best and second-best.
            h2_abs = abs(float(h2))
            h2_abs_ema = 0.99 * float(h2_abs_ema) + 0.01 * float(h2_abs)
            h2_gate = max(float(conf_h2_min_abs), float(conf_h2_frac) * max(float(h2_abs_ema), float(conf_h2_min_abs)))
            margin = float(second_cost - best_cost) if np.isfinite(second_cost) else 0.0
            vpi_update_ok = (h2_abs >= h2_gate) and (margin >= float(conf_margin_rad))

            if vpi_update_ok:
                vpi_buf.append(float(best_vpi))
                vpi_meas = float(np.median(np.asarray(vpi_buf, dtype=float)))
                ratio = vpi_meas / float(Vpi_est)
                ratio = float(np.clip(ratio, 1.0 - float(update_max_rel), 1.0 + float(update_max_rel)))
                Vpi_est = float(Vpi_est) * (1.0 + float(est_alpha) * (ratio - 1.0))
                Vpi_est = float(np.clip(Vpi_est, vpi_min, vpi_max))

            dp_est = replace(device_params, Vpi_DC=float(Vpi_est))
            theta_prior = float(bias_to_theta_rad(V_bias_current, Vpi_DC=float(Vpi_est)))
            theta_est_rad, _ = _estimate_theta_from_harmonics_np(
                np.array([h1], dtype=np.float32),
                np.array([h2], dtype=np.float32),
                theta_prior_rad=np.array([theta_prior], dtype=np.float32),
                dither_params=dither_params,
                device_params=dp_est,
            )
            theta_est = float(theta_est_rad[0])
            s = float(np.sin(theta_est))
            c = float(np.cos(theta_est))

            if t == 0:
                prev_s = s
                prev_c = c

            ds = s - prev_s
            dc = c - prev_c

            obs_list = [s, c, ds, dc, prev_dv, target_sin, target_cos]
            obs_raw = torch.tensor([obs_list], dtype=torch.float32)
            obs_norm = (obs_raw - mu_t) / sigma_t
            
            delta_v_nn = float(model(obs_norm)[0].item())

            gain_factor = float(np.clip(Vpi_est / Vpi_nom, 0.5, 3.0))
            delta_v = float(delta_v_nn) * gain_factor

        delta_v_clipped = float(np.clip(delta_v, -float(max_step_V), float(max_step_V)))
        V_bias_current = float(V_bias_current + delta_v_clipped)
        if bool(clamp_bias):
            V_bias_current = float(np.clip(V_bias_current, 0.0, Vpi_nom))

        prev_s = s
        prev_c = c
        prev_dv = delta_v_clipped

        theta_actual = float(bias_to_theta_rad(V_bias_current, Vpi_DC=float(Vpi_true)))
        phase_err = float(np.degrees(wrap_to_pi(theta_actual - float(theta_target))))

        history.append(
            {
                "step": int(t),
                "v_ctrl": float(V_bias_current),
                "dv": float(delta_v_clipped),
                "dv_nn": float(delta_v_nn),
                "vpi_nom": float(Vpi_nom),
                "vpi_true": float(Vpi_true),
                "vpi_est": float(Vpi_est),
                "vpi_update_ok": bool(vpi_update_ok),
                "vpi_margin": float(margin),
                "h2_abs": float(h2_abs),
                "h2_gate": float(h2_gate),
                "theta_err": float(phase_err),
                "theta_est_deg": float(np.degrees(theta_est)),
                "h1": float(h1),
                "h2": float(h2),
            }
        )

    return history


def plot_results_vpi_drift(history, target_deg, show=True, save_path=None, *, vpi_y: str = "ratio"):
    steps = [h["step"] for h in history]
    vpi_ratio = [h.get("vpi_ratio", np.nan) for h in history]
    vpi_true = [h.get("vpi_true", np.nan) for h in history]
    vpi_used = [h.get("vpi_used", np.nan) for h in history]
    vpi_nom = [h.get("vpi_nom", np.nan) for h in history]
    v_ctrl = [h["v_ctrl"] for h in history]
    errs = [h["theta_err"] for h in history]

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    vpi_y_norm = str(vpi_y).lower().strip()
    ax0b = None
    if vpi_y_norm == "ratio":
        ax0.plot(steps, vpi_ratio, color="purple", label="Vpi_true / Vpi_nom")
        ax0.axhline(1.0, color="k", linestyle="--", linewidth=0.5)
        ax0.set_ylabel("Vpi ratio")
    elif vpi_y_norm == "true":
        ax0.plot(steps, vpi_true, color="purple", label="Vpi_true (V)")
        ax0.plot(steps, vpi_used, color="grey", linestyle="--", alpha=0.8, label="Vpi_used (V)")
        ax0.set_ylabel("Vpi (V)")
    elif vpi_y_norm == "both":
        ax0.plot(steps, vpi_true, color="purple", label="Vpi_true (V)")
        ax0.plot(steps, vpi_used, color="grey", linestyle="--", alpha=0.8, label="Vpi_used (V)")
        ax0.plot(steps, vpi_nom, color="k", linestyle=":", alpha=0.6, label="Vpi_nom (V)")
        ax0b = ax0.twinx()
        ax0b.plot(steps, vpi_ratio, color="purple", alpha=0.25, label="Vpi_true / Vpi_nom")
        ax0b.set_ylabel("Vpi ratio")
    else:
        raise ValueError("vpi_y must be one of: 'ratio', 'true', 'both'")

    ax0.set_title("Vpi Drift")
    ax0.grid(True, alpha=0.3)
    if ax0b is None:
        ax0.legend()
    else:
        h0, l0 = ax0.get_legend_handles_labels()
        h1, l1 = ax0b.get_legend_handles_labels()
        ax0.legend(h0 + h1, l0 + l1)

    ax1.plot(steps, v_ctrl, color="blue", label="Controller Output (V_bias)")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("Controller Bias Command")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(steps, errs, "r-", label="Phase Error")
    ax2.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax2.set_ylabel("Error (deg)")
    ax2.set_xlabel("Step")
    ax2.set_title(f"Tracking Error (Mean Abs: {np.mean(np.abs(errs)):.2f} deg, Target: {target_deg} deg)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()

def plot_vpi_drift_comparison(
    *,
    hist_baseline: list[dict],
    hist_adaptive: list[dict],
    vpi_drift_total_V: float,
    show: bool = True,
):
    steps = [h["step"] for h in hist_baseline]

    err_base = [h["theta_err"] for h in hist_baseline]
    err_adap = [h["theta_err"] for h in hist_adaptive]

    vpi_true = [h.get("vpi_true", np.nan) for h in hist_adaptive]
    vpi_est = [h.get("vpi_est", np.nan) for h in hist_adaptive]

    v_base = [h["v_ctrl"] for h in hist_baseline]
    v_adap = [h["v_ctrl"] for h in hist_adaptive]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax1.plot(steps, err_base, "r-", alpha=0.5, label="Baseline Error")
    ax1.plot(steps, err_adap, "g-", linewidth=1.5, label="Adaptive Error")
    ax1.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("Phase Error (deg)")
    ax1.set_title(f"Tracking Error Comparison (Vpi Drift: +{float(vpi_drift_total_V):.1f}V)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, vpi_true, "k--", label="True Vpi")
    ax2.plot(steps, vpi_est, "b-", label="Estimated Vpi")
    ax2.set_ylabel("Vpi (V)")
    ax2.set_title("Vpi Estimation Performance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(steps, v_base, "r-", alpha=0.5, label="Baseline V_bias")
    ax3.plot(steps, v_adap, "g-", label="Adaptive V_bias")
    ax3.set_ylabel("Bias Voltage (V)")
    ax3.set_xlabel("Step")
    ax3.set_title("Controller Output Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def run_vpi_drift_comparison(
    model,
    mu,
    sigma,
    device_params: DeviceParams,
    dither_params: DitherParams,
    *,
    n_steps: int = 10000,
    target_theta_deg: float = 90.0,
    vpi_drift_total_V: float = 1.0,
    est_alpha: float = 0.02,
    max_step_V: float = 0.2,
    conf_margin_rad: float = 0.01,
    conf_h2_frac: float = 0.08,
    conf_h2_min_abs: float = 1e-6,
    update_max_rel: float = 0.05,
    buffer_len: int = 5,
    show: bool = True,
):
    vpi_abs_step_rate = float(vpi_drift_total_V) / float(n_steps)

    print("Running Baseline Controller...")
    hist_baseline = simulate_control_loop_with_vpi_drift(
        model,
        mu,
        sigma,
        device_params,
        dither_params,
        n_steps=int(n_steps),
        target_theta_deg=float(target_theta_deg),
        vpi_abs_step_rate=float(vpi_abs_step_rate),
        max_step_V=float(max_step_V),
        controller_knows_vpi=False,
    )

    print("Running Adaptive Controller...")
    hist_adaptive = simulate_adaptive_control_loop(
        model,
        mu,
        sigma,
        device_params,
        dither_params,
        n_steps=int(n_steps),
        target_theta_deg=float(target_theta_deg),
        vpi_abs_step_rate=float(vpi_abs_step_rate),
        max_step_V=float(max_step_V),
        est_alpha=float(est_alpha),
        conf_margin_rad=float(conf_margin_rad),
        conf_h2_frac=float(conf_h2_frac),
        conf_h2_min_abs=float(conf_h2_min_abs),
        update_max_rel=float(update_max_rel),
        buffer_len=int(buffer_len),
    )

    fig = plot_vpi_drift_comparison(
        hist_baseline=hist_baseline,
        hist_adaptive=hist_adaptive,
        vpi_drift_total_V=float(vpi_drift_total_V),
        show=bool(show),
    )

    return {
        "hist_baseline": hist_baseline,
        "hist_adaptive": hist_adaptive,
        "figure": fig,
        "vpi_abs_step_rate": vpi_abs_step_rate,
    }

def plot_results(history, target_deg, show=True, save_path=None):
    steps = [h['step'] for h in history]
    v_drift = [h['v_drift'] for h in history]
    v_ctrl = [h['v_ctrl'] for h in history]
    v_act = [h['v_actual'] for h in history]
    errs = [h['theta_err'] for h in history]
    ests = [h['theta_est_deg'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(steps, v_drift, '--', label='Drift (Disturbance)', color='grey', alpha=0.7)
    ax1.plot(steps, v_ctrl, label='Controller Output', color='blue')
    ax1.plot(steps, v_act, label='Actual Bias (Net)', color='green', linewidth=2)
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"Bias Drift Compensation (Target: {target_deg} deg)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, errs, 'r-', label='Phase Error')
    ax2.plot(steps, [(e - target_deg) for e in ests], 'k:', alpha=0.5, label='Est Error') # Approximate check
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax2.set_ylabel("Error (deg)")
    ax2.set_xlabel("Step")
    ax2.set_title(f"Tracking Error (Mean Abs: {np.mean(np.abs(errs)):.2f} deg)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        
    if show:
        plt.show()
