
import torch
import numpy as np
import matplotlib.pyplot as plt
from .model import measure_pd_dither_normalized_batch_torch, bias_to_theta_rad, wrap_to_pi
from .dither_controller import DeviceParams, DitherParams, _estimate_theta_from_harmonics_np

def get_bias_drift(t, step_rate=0.002):
    """
    Generate a bias drift value at time step t.
    Models unidirectional drift (linear ramp) which shifts the MZM transfer curve.
    """
    # Linear drift only
    linear = step_rate * t
    return linear

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
    
    for t in range(n_steps):
        # 1. Calculate Shift
        v_drift = get_bias_drift(t, step_rate=drift_step_rate)
        
        # 2. Measurement (Physics Simulation)
        v_batch = torch.tensor([V_bias_current], dtype=torch.float32)
        drift_batch = torch.tensor([v_drift], dtype=torch.float32)
        
        with torch.no_grad():
            # The model computes V_total = V_bias_current + V_drift internally
            h1_t, h2_t, _ = measure_pd_dither_normalized_batch_torch(
                V_bias=v_batch,
                V_dither_amp=dither_params.V_dither_amp,
                f_dither=dither_params.f_dither,
                Fs=dither_params.Fs,
                n_periods=dither_params.n_periods,
                Vpi_DC=device_params.Vpi_DC,
                ER_dB=device_params.ER_dB,
                IL_dB=device_params.IL_dB,
                Pin_dBm=device_params.Pin_dBm,
                Responsivity=device_params.Responsivity,
                R_load=device_params.R_load,
                V_rf_amp=0.0,
                V_drift=drift_batch
            )
            
            h1 = h1_t.item()
            h2 = h2_t.item()
            
            # --- Feature Engineering: theta_est_hist ---
            
            # 1. Estimate Theta from h1, h2
            # We need a prior for the estimator (where we *think* we are)
            # bias_to_theta_rad(V_bias_current, ...) gives the controller's belief of phase
            # if drift was 0.
            theta_prior = bias_to_theta_rad(V_bias_current, Vpi_DC=device_params.Vpi_DC)
            
            theta_est_rad, _ = _estimate_theta_from_harmonics_np(
                np.array([h1]), 
                np.array([h2]), 
                theta_prior_rad=np.array([theta_prior]),
                dither_params=dither_params,
                device_params=device_params
            )
            theta_est = float(theta_est_rad[0])
            
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
            'h1': h1
        })
        
    return history

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
