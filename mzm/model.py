"""MZM simulation with thermal, shot, RIN noise and RBW.
Ported from MATLAB version.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class NoiseResult:
    P_thermal_W: float
    P_shot_W: float
    P_rin_W: float
    P_noise_total_W: float
    P_noise_floor_dBm: float
    P_density_dBmHz: float


@dataclass
class SpectrumResult:
    f_opt: np.ndarray
    P_opt_spec_dBm: np.ndarray
    f_elec: np.ndarray
    P_elec_spec_dBm: np.ndarray
    val_1G_dBm: float
    val_2G_dBm: float


@dataclass
class BiasScanResult:
    V_scan: np.ndarray
    P_scan_mW: np.ndarray
    V_bias: float
    curr_P_mW: float
    curr_P_dBm: float


@dataclass
class SimulationResult:
    t: np.ndarray
    RBW_Hz: float
    E_out: np.ndarray
    P_opt_inst_W: np.ndarray
    P_pd_avg_dBm: float
    I_pd: np.ndarray
    noise: NoiseResult
    spectrum: SpectrumResult
    bias_scan: BiasScanResult
    Pin_dBm: float
    pd_tap: float = 1.0


def simulate_mzm(
    SymbolRate: float = 10e9,
    Fs: float = 100e9,
    T_total: float = 10e-6,
    Vpi_RF: float = 5.0,
    Vpi_DC: float = 5.0,
    ER_dB: float = 30.0,
    IL_dB: float = 6.0,
    Responsivity: float = 0.786,
    R_load: float = 50.0,
    Pin_dBm: float = 10.0,
    pd_tap: float = 1.0,
    Temp_K: float = 290.0,
    RIN_dB_Hz: float = -145.0,
    f_rf: float = 1e9,
    V_rf_amp: float = 0.2,
    V_bias: float | None = None,
    vpi_compatible_dbm: bool = False,
) -> SimulationResult:
    """Run MZM simulation with noise and spectrum analysis.

    Parameters mirror the original MATLAB script defaults.
    
    Args:
        vpi_compatible_dbm: If True (default), electrical power is calculated as I^2 (normalized to 1 Ohm)
                            to match VPI's default behavior, which is ~17 dB lower than I^2 * 50.
                            If False, power is calculated as I^2 * R_load (physical power in 50 Ohm).
    """

    # 1. Time and basic parameters
    dt = 1.0 / Fs
    t = np.arange(0, T_total, dt)
    L = t.size

    # Resolution bandwidth
    RBW_Hz = Fs / L

    # Device parameters
    Pin_W = 10 ** ((Pin_dBm - 30.0) / 10.0)
    E_in = np.sqrt(Pin_W)

    loss_factor = 10 ** (-IL_dB / 10.0)
    er_linear = 10 ** (ER_dB / 20.0)
    gamma = (er_linear - 1.0) / (er_linear + 1.0)

    # 2. Signal generation
    if V_bias is None:
        V_bias = Vpi_DC / 2.0
    RF_signal = V_rf_amp * np.sin(2.0 * np.pi * f_rf * t)

    V1 = (V_bias + RF_signal) / 2.0
    V2 = -(V_bias + RF_signal) / 2.0

    # 3. MZM modulation
    phi1 = (np.pi / Vpi_RF) * V1
    phi2 = (np.pi / Vpi_RF) * V2

    E_out = E_in * np.sqrt(loss_factor) * 0.5 * (
        np.exp(1j * phi1) + gamma * np.exp(1j * phi2)
    )

    if float(pd_tap) <= 0:
        raise ValueError("pd_tap must be > 0")

    P_opt_inst_W = np.abs(E_out) ** 2
    # Optical power seen by PD after an optional tap (e.g., 1:9 coupler).
    P_pd_inst_W = P_opt_inst_W * float(pd_tap)
    P_pd_avg_W = np.mean(P_pd_inst_W)
    P_pd_avg_dBm = 10.0 * np.log10(P_pd_avg_W * 1000.0 + 1e-30)

    I_pd = Responsivity * P_pd_inst_W

    # 4. Noise calculation
    I_av = float(np.mean(I_pd))

    K_Boltzmann = 1.38e-23
    q_electron = 1.6e-19

    P_thermal_W = K_Boltzmann * Temp_K * RBW_Hz
    P_shot_W = 2.0 * q_electron * I_av * RBW_Hz * R_load
    P_rin_W = (10.0 ** (RIN_dB_Hz / 10.0)) * (I_av ** 2) * R_load * RBW_Hz

    P_noise_total_W = P_thermal_W + P_shot_W + P_rin_W

    # [Fix] Scale down by R_load (50 Ohm) to match VPI simulation results.
    # VPI likely displays power into 50 Ohm as V^2/50 (or equivalent), whereas
    # the previous Python code calculated I^2 * 50. The difference is a factor of 50 (17 dB).
    # We divide by R_load to align with VPI's magnitude.
    if vpi_compatible_dbm:
        scale_factor = 1.0 / R_load
    else:
        scale_factor = 1.0

    P_noise_total_W_scaled = P_noise_total_W * scale_factor
    P_noise_floor_dBm = 10.0 * np.log10(P_noise_total_W_scaled * 1000.0 + 1e-30)

    P_density_W_Hz_scaled = P_noise_total_W_scaled / RBW_Hz
    P_density_dBmHz = 10.0 * np.log10(P_density_W_Hz_scaled * 1000.0 + 1e-30)

    noise_res = NoiseResult(
        P_thermal_W=P_thermal_W * scale_factor,
        P_shot_W=P_shot_W * scale_factor,
        P_rin_W=P_rin_W * scale_factor,
        P_noise_total_W=P_noise_total_W_scaled,
        P_noise_floor_dBm=P_noise_floor_dBm,
        P_density_dBmHz=P_density_dBmHz,
    )

    # 5. Spectrum analysis
    # Optical spectrum
    E_spec = np.fft.fftshift(np.fft.fft(E_out)) / L
    f_opt = (np.arange(-L / 2, L / 2) * (Fs / L))
    P_opt_spec_W = np.abs(E_spec) ** 2 * 1000.0
    P_opt_spec_dBm = 10.0 * np.log10(P_opt_spec_W + 1e-20)

    # Electrical spectrum
    Y_elec = np.fft.fft(I_pd)
    P2 = np.abs(Y_elec / L)
    half = L // 2
    P1 = P2[: half + 1].copy()
    if P1.size > 2:
        P1[1:-1] = 2.0 * P1[1:-1]
    f_elec = Fs * np.arange(0, half + 1) / L

    # [Fix] Remove * R_load factor to match VPI (17 dB lower).
    # Effectively calculating Power = I^2 (or V^2/50 if V=I*50).
    if vpi_compatible_dbm:
        P_sig_W = 0.5 * (P1 ** 2)
        if P1.size > 0:
            P_sig_W[0] = (P1[0] ** 2)
    else:
        P_sig_W = 0.5 * (P1 ** 2) * R_load
        if P1.size > 0:
            P_sig_W[0] = (P1[0] ** 2) * R_load

    rng = np.random.default_rng()
    noise_trace_W = P_noise_total_W_scaled * (-np.log(rng.random(P_sig_W.shape)))

    P_total_W = P_sig_W + noise_trace_W
    P_elec_spec_dBm = 10.0 * np.log10(P_total_W * 1000.0 + 1e-20)

    idx_1G = int(np.argmin(np.abs(f_elec - 1e9)))
    idx_2G = int(np.argmin(np.abs(f_elec - 2e9)))
    val_1G = float(P_elec_spec_dBm[idx_1G])
    val_2G = float(P_elec_spec_dBm[idx_2G])

    spectrum_res = SpectrumResult(
        f_opt=f_opt,
        P_opt_spec_dBm=P_opt_spec_dBm,
        f_elec=f_elec,
        P_elec_spec_dBm=P_elec_spec_dBm,
        val_1G_dBm=val_1G,
        val_2G_dBm=val_2G,
    )

    # 6. Bias scan
    V_scan = np.linspace(-2.0 * Vpi_DC, 2.0 * Vpi_DC, 1000)
    phi1_scan = (np.pi / Vpi_DC) * (V_scan / 2.0)
    phi2_scan = (np.pi / Vpi_DC) * (-V_scan / 2.0)

    E_scan = E_in * np.sqrt(loss_factor) * 0.5 * (
        np.exp(1j * phi1_scan) + gamma * np.exp(1j * phi2_scan)
    )
    P_scan_mW = np.abs(E_scan) ** 2 * 1000.0

    curr_P_mW = float(np.interp(V_bias, V_scan, P_scan_mW))
    curr_P_dBm = 10.0 * np.log10(curr_P_mW + 1e-20)

    bias_res = BiasScanResult(
        V_scan=V_scan,
        P_scan_mW=P_scan_mW,
        V_bias=V_bias,
        curr_P_mW=curr_P_mW,
        curr_P_dBm=curr_P_dBm,
    )

    return SimulationResult(
        t=t,
        RBW_Hz=RBW_Hz,
        E_out=E_out,
        P_opt_inst_W=P_opt_inst_W,
        P_pd_avg_dBm=P_pd_avg_dBm,
        I_pd=I_pd,
        noise=noise_res,
        spectrum=spectrum_res,
        bias_scan=bias_res,
        Pin_dBm=Pin_dBm,
        pd_tap=float(pd_tap),
    )




def bias_to_theta_rad(V_bias: np.ndarray | float, *, Vpi_DC: float = 5.0) -> np.ndarray:
    """Map bias voltage to an operating angle theta (rad).

    We define:
        theta = (pi / Vpi_DC) * V_bias

    Under this definition, with the current DC transfer model:
        theta=0   -> maximum point (0°)
        theta=pi/2-> quadrature (90°)
        theta=pi  -> minimum point (180°)
    """

    return (np.pi / float(Vpi_DC)) * np.asarray(V_bias, dtype=float)


def theta_to_bias_V(theta_rad: np.ndarray | float, *, Vpi_DC: float = 5.0) -> np.ndarray:
    """Inverse mapping of bias_to_theta_rad."""

    return (float(Vpi_DC) / np.pi) * np.asarray(theta_rad, dtype=float)


def wrap_to_pi(theta_rad: np.ndarray | float) -> np.ndarray:
    """Wrap angle to [-pi, pi)."""

    th = np.asarray(theta_rad, dtype=float)
    return (th + np.pi) % (2.0 * np.pi) - np.pi


@torch.no_grad()
def _measure_pd_dither_1f2f_batch_torch(
    *,
    V_bias: torch.Tensor,
    V_dither_amp: float,
    f_dither: float,
    Fs: float,
    n_periods: int,
    Vpi_DC: float,
    ER_dB: float,
    IL_dB: float,
    Pin_dBm: float,
    pd_tap: float = 1.0,
    Responsivity: float,
    R_load: float,
    refs: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    # RF signal parameters (optional, for robustness testing)
    V_rf_amp: float = 0.0,
    f_rf: float = 1e9,
    # Drift parameters
    V_drift: float | torch.Tensor = 0.0,
) -> dict[str, torch.Tensor]:
    """Torch batch core for PD dither 1f/2f measurement.

    Returns tensors with batch dimension [B]. Internal math uses float64
    for stability (especially when 2f is extremely small).
    
    Parameters
    ----------
    V_rf_amp : float, optional
        RF signal amplitude (V). Default 0.0 (no RF signal).
        When > 0, a high-frequency RF signal is added to simulate
        real-world conditions where data modulation coexists with dither.
    f_rf : float, optional
        RF signal frequency (Hz). Default 1e9 (1 GHz).
    """

    if f_dither <= 0 or Fs <= 0:
        raise ValueError("f_dither and Fs must be > 0")

    device = V_bias.device
    dtype = torch.float32
    Vb = V_bias.reshape(-1).to(dtype=dtype)
    B = int(Vb.numel())

    n_samples = int(round((float(n_periods) / float(f_dither)) * float(Fs)))
    if n_samples < 16:
        raise ValueError("n_samples too small; increase Fs or n_periods")

    # Time base and dither waveform
    t = torch.arange(n_samples, device=device, dtype=dtype) / float(Fs)
    w1 = 2.0 * float(np.pi) * float(f_dither)
    dither = float(V_dither_amp) * torch.sin(w1 * t)  # [N]
    
    # Add RF signal if specified (high-frequency modulation)
    # Note: For realistic simulation, RF frequency >> dither frequency
    # The RF signal adds perturbation to the optical power, affecting dither measurement
    #
    # FIX (2025-12-30): Instead of adding random noise or aliased sampling,
    # we use the analytical "J0 scaling" effect.
    # The fast RF modulation V_rf * sin(w_rf * t) effectively scales the
    # interference term of the MZM transfer function by Bessel function J0(beta).
    # This is the physically correct model for a bandwidth-limited PD detecting
    # a bias dither in the presence of high-speed RF modulation.
    
    # V_t only contains bias + dither (slow signals) + drift
    # V_drift can be a scalar or a tensor broadcastable to Vb
    V_total_bias = Vb + V_drift
    V_t = V_total_bias[:, None] + dither[None, :]  # [B, N]

    # DC transfer in torch
    Pin_W = 10.0 ** ((float(Pin_dBm) - 30.0) / 10.0)
    if float(pd_tap) <= 0:
        raise ValueError("pd_tap must be > 0")
    
    loss_factor = 10.0 ** (-float(IL_dB) / 10.0)
    er_linear = 10.0 ** (float(ER_dB) / 20.0)
    gamma = (er_linear - 1.0) / (er_linear + 1.0)

    # Calculate RF scaling factor J0(beta)
    # beta = pi * V_rf / Vpi
    if float(V_rf_amp) > 0:
        beta_rf = (float(np.pi) / float(Vpi_DC)) * float(V_rf_amp)
        # Use torch.special.bessel_j0 if available, else fallback or assume recent torch
        j0_scale = float(torch.special.bessel_j0(torch.tensor(beta_rf)))
    else:
        j0_scale = 1.0

    # Calculate Power directly using the intensity formula
    # P_out = P_in * loss * 0.25 * [ (1 + gamma^2) + 2*gamma*J0*cos(pi*V/Vpi) ]
    # This avoids complex number operations and allows easy J0 scaling.
    
    P_scale = Pin_W * float(np.sqrt(loss_factor))**2 * 0.25
    term_const = 1.0 + float(gamma)**2
    
    # Phase difference phi1 - phi2 = (pi/Vpi) * V_t
    theta_t = (float(np.pi) / float(Vpi_DC)) * V_t
    term_interf = 2.0 * float(gamma) * j0_scale * torch.cos(theta_t)
    
    P_W = P_scale * (term_const + term_interf)  # [B, N] (MZM output)
    # Tap/coupler after MZM, before PD.
    P_W = P_W * float(pd_tap)

    # PD current
    I_pd = float(Responsivity) * P_W
    pd_dc = I_pd.mean(dim=1)
    I_ac = I_pd - pd_dc[:, None]

    # Lock-in refs
    if refs is None:
        refs = {
            1: (torch.sin(w1 * t), torch.cos(w1 * t)),
            2: (torch.sin(2.0 * w1 * t), torch.cos(2.0 * w1 * t)),
        }
    s1, c1 = refs[1]
    s2, c2 = refs[2]
    if s1.device != device or s2.device != device:
        raise ValueError("refs must be on the same device as V_bias")

    # Ensure dtype consistency for stable dot-products.
    s1 = s1.to(dtype=dtype)
    c1 = c1.to(dtype=dtype)
    s2 = s2.to(dtype=dtype)
    c2 = c2.to(dtype=dtype)
    I_ac = I_ac.to(dtype=dtype)

    N = float(n_samples)
    I1 = (2.0 / N) * torch.sum(I_ac * s1[None, :], dim=1)
    Q1 = (2.0 / N) * torch.sum(I_ac * c1[None, :], dim=1)
    A1 = torch.sqrt(I1 * I1 + Q1 * Q1)

    I2 = (2.0 / N) * torch.sum(I_ac * s2[None, :], dim=1)
    Q2 = (2.0 / N) * torch.sum(I_ac * c2[None, :], dim=1)
    A2 = torch.sqrt(I2 * I2 + Q2 * Q2)

    p1_W = 0.5 * (A1 ** 2) * float(R_load)
    p2_W = 0.5 * (A2 ** 2) * float(R_load)

    p1_dBm = 10.0 * torch.log10(p1_W * 1000.0 + 1e-30)
    p2_dBm = 10.0 * torch.log10(p2_W * 1000.0 + 1e-30)

    theta_rad = (float(np.pi) / float(Vpi_DC)) * Vb

    if B == 0:
        # Keep shapes predictable even for empty batches.
        return {
            "pd_dc": pd_dc,
            "h1_I": I1,
            "h1_Q": Q1,
            "h1_A": A1,
            "h2_I": I2,
            "h2_Q": Q2,
            "h2_A": A2,
            "p1_W": p1_W,
            "p1_dBm": p1_dBm,
            "p2_W": p2_W,
            "p2_dBm": p2_dBm,
            "theta_rad": theta_rad,
        }

    return {
        "pd_dc": pd_dc,
        "h1_I": I1,
        "h1_Q": Q1,
        "h1_A": A1,
        "h2_I": I2,
        "h2_Q": Q2,
        "h2_A": A2,
        "p1_W": p1_W,
        "p1_dBm": p1_dBm,
        "p2_W": p2_W,
        "p2_dBm": p2_dBm,
        "theta_rad": theta_rad,
    }


def measure_pd_dither_1f2f(
    *,
    V_bias: float,
    V_dither_amp: float = 0.05,
    f_dither: float = 50e3,
    Fs: float = 5e6,
    n_periods: int = 200,
    Vpi_DC: float = 5.0,
    ER_dB: float = 30.0,
    IL_dB: float = 6.0,
    Pin_dBm: float = 10.0,
    pd_tap: float = 1.0,
    Responsivity: float = 0.786,
    R_load: float = 50.0,
    feature: str = "signed",
) -> Dict[str, float]:
    """Simulate PD output under low-frequency dither and extract 1f/2f features.

    This models what you can measure in practice:
    - You *cannot* observe optical power directly.
    - You *can* observe PD electrical signal around the dither frequency.

    Parameters
    - feature:
        "signed": returns lock-in I components (signed) for 1f and 2f.
        "power":  returns A^2 (magnitude squared) for 1f and 2f.

    Returns
    - pd_dc: average PD current (A)
    - h1: 1f feature
    - h2: 2f feature
    - theta_rad: underlying operating angle (rad) from V_bias
    """

    feature = feature.lower().strip()
    if feature not in {"signed", "power"}:
        raise ValueError("feature must be 'signed' or 'power'")

    # Keep this API pure-CPU to avoid surprising CUDA allocations
    # during interactive use.
    out = _measure_pd_dither_1f2f_batch_torch(
        V_bias=torch.tensor([float(V_bias)], dtype=torch.float64, device=torch.device("cpu")),
        V_dither_amp=float(V_dither_amp),
        f_dither=float(f_dither),
        Fs=float(Fs),
        n_periods=int(n_periods),
        Vpi_DC=float(Vpi_DC),
        ER_dB=float(ER_dB),
        IL_dB=float(IL_dB),
        Pin_dBm=float(Pin_dBm),
        pd_tap=float(pd_tap),
        Responsivity=float(Responsivity),
        R_load=float(R_load),
        refs=None,
    )

    pd_dc = float(out["pd_dc"][0].item())
    h1_I = float(out["h1_I"][0].item())
    h1_Q = float(out["h1_Q"][0].item())
    h1_A = float(out["h1_A"][0].item())
    h2_I = float(out["h2_I"][0].item())
    h2_Q = float(out["h2_Q"][0].item())
    h2_A = float(out["h2_A"][0].item())
    p1_W = float(out["p1_W"][0].item())
    p2_W = float(out["p2_W"][0].item())
    p1_dBm = float(out["p1_dBm"][0].item())
    p2_dBm = float(out["p2_dBm"][0].item())
    theta = float(out["theta_rad"][0].item())

    if feature == "signed":
        h1 = h1_I
        h2 = h2_A
    else:
        h1 = float(h1_A ** 2)
        h2 = float(h2_A ** 2)

    return {
        "pd_dc": pd_dc,
        "h1": h1,
        "h2": h2,
        "h1_I": h1_I,
        "h1_Q": h1_Q,
        "h1_A": h1_A,
        "h2_I": h2_I,
        "h2_Q": h2_Q,
        "h2_A": h2_A,
        "p1_W": p1_W,
        "p1_dBm": p1_dBm,
        "p2_W": p2_W,
        "p2_dBm": p2_dBm,
        "R_load": float(R_load),
        "theta_rad": theta,
    }


@torch.no_grad()
def measure_pd_dither_1f2f_dbm_batch_torch(
    *,
    V_bias: torch.Tensor,
    V_dither_amp: float = 0.05,
    f_dither: float = 50e3,
    Fs: float = 5e6,
    n_periods: int = 200,
    Vpi_DC: float = 5.0,
    ER_dB: float = 30.0,
    IL_dB: float = 6.0,
    Pin_dBm: float = 10.0,
    pd_tap: float = 1.0,
    Responsivity: float = 0.786,
    R_load: float = 50.0,
    refs: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    V_rf_amp: float = 0.0,
    f_rf: float = 1e9,
    V_drift: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch/GPU-accelerated batch version of dither measurement.

    Computes 1f/2f *power* features in dBm for a batch of bias voltages.

    Parameters
    - V_bias: 1D tensor of shape [B] (or any shape that flattens to B)
    - refs: optional precomputed {harmonic: (sin_ref, cos_ref)} where each ref
      has shape [N] on the same device as V_bias.
    - V_rf_amp: RF signal amplitude (V). Default 0.0 (no RF).
    - f_rf: RF signal frequency (Hz). Default 1e9 (1 GHz).

    Returns
    - p1_dBm: tensor [B]
    - p2_dBm: tensor [B]
    """

    out = _measure_pd_dither_1f2f_batch_torch(
        V_bias=V_bias,
        V_dither_amp=float(V_dither_amp),
        f_dither=float(f_dither),
        Fs=float(Fs),
        n_periods=int(n_periods),
        Vpi_DC=float(Vpi_DC),
        ER_dB=float(ER_dB),
        IL_dB=float(IL_dB),
        Pin_dBm=float(Pin_dBm),
        pd_tap=float(pd_tap),
        Responsivity=float(Responsivity),
        R_load=float(R_load),
        refs=refs,
        V_rf_amp=float(V_rf_amp),
        f_rf=float(f_rf),
        V_drift=V_drift,
    )
    return out["p1_dBm"].to(dtype=torch.float32), out["p2_dBm"].to(dtype=torch.float32)


@torch.no_grad()
def measure_pd_dither_normalized_batch_torch(
    *,
    V_bias: torch.Tensor,
    V_dither_amp: float = 0.05,
    f_dither: float = 50e3,
    Fs: float = 5e6,
    n_periods: int = 200,
    Vpi_DC: float = 5.0,
    ER_dB: float = 30.0,
    IL_dB: float = 6.0,
    Pin_dBm: float = 10.0,
    pd_tap: float = 1.0,
    Responsivity: float = 0.786,
    R_load: float = 50.0,
    refs: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    V_rf_amp: float = 0.0,
    f_rf: float = 1e9,
    V_drift: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch/GPU-accelerated batch version with DC-normalized outputs.

    Computes DC-normalized 1f/2f amplitude features for a batch of bias voltages.
    The normalization (h1_A/pd_dc, h2_A/pd_dc) makes features robust against
    input optical power fluctuations (e.g., laser aging, fiber vibrations).

    Parameters
    - V_bias: 1D tensor of shape [B] (or any shape that flattens to B)
    - refs: optional precomputed {harmonic: (sin_ref, cos_ref)} where each ref
      has shape [N] on the same device as V_bias.
    - V_rf_amp: RF signal amplitude (V). Default 0.0 (no RF).
    - f_rf: RF signal frequency (Hz). Default 1e9 (1 GHz).

    Returns
    - h1_norm: tensor [B] - h1_A / pd_dc (dimensionless, normalized 1f amplitude)
    - h2_norm: tensor [B] - h2_A / pd_dc (dimensionless, normalized 2f amplitude)
    - pd_dc: tensor [B] - mean photocurrent (A), useful for diagnostics
    """

    out = _measure_pd_dither_1f2f_batch_torch(
        V_bias=V_bias,
        V_dither_amp=float(V_dither_amp),
        f_dither=float(f_dither),
        Fs=float(Fs),
        n_periods=int(n_periods),
        Vpi_DC=float(Vpi_DC),
        ER_dB=float(ER_dB),
        IL_dB=float(IL_dB),
        Pin_dBm=float(Pin_dBm),
        pd_tap=float(pd_tap),
        Responsivity=float(Responsivity),
        R_load=float(R_load),
        refs=refs,
        V_rf_amp=float(V_rf_amp),
        f_rf=float(f_rf),
        V_drift=V_drift,
    )

    pd_dc = out["pd_dc"]
    h1_A = out["h1_A"]
    h2_A = out["h2_A"]

    # Normalize by DC photocurrent (add small epsilon to avoid division by zero)
    eps = 1e-12
    h1_norm = h1_A / (pd_dc + eps)
    h2_norm = h2_A / (pd_dc + eps)

    return (
        h1_norm.to(dtype=torch.float32),
        h2_norm.to(dtype=torch.float32),
        pd_dc.to(dtype=torch.float32),
    )
