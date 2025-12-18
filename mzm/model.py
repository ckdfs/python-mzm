"""MZM simulation with thermal, shot, RIN noise and RBW.
Ported from MATLAB version.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


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
    Temp_K: float = 290.0,
    RIN_dB_Hz: float = -145.0,
    f_rf: float = 1e9,
    V_rf_amp: float = 0.2,
    V_bias: float | None = None,
) -> SimulationResult:
    """Run MZM simulation with noise and spectrum analysis.

    Parameters mirror the original MATLAB script defaults.
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

    P_opt_inst_W = np.abs(E_out) ** 2
    P_pd_avg_W = np.mean(P_opt_inst_W)
    P_pd_avg_dBm = 10.0 * np.log10(P_pd_avg_W * 1000.0 + 1e-30)

    I_pd = Responsivity * P_opt_inst_W

    # 4. Noise calculation
    I_av = float(np.mean(I_pd))

    K_Boltzmann = 1.38e-23
    q_electron = 1.6e-19

    P_thermal_W = K_Boltzmann * Temp_K * RBW_Hz
    P_shot_W = 2.0 * q_electron * I_av * RBW_Hz * R_load
    P_rin_W = (10.0 ** (RIN_dB_Hz / 10.0)) * (I_av ** 2) * R_load * RBW_Hz

    P_noise_total_W = P_thermal_W + P_shot_W + P_rin_W
    P_noise_floor_dBm = 10.0 * np.log10(P_noise_total_W * 1000.0 + 1e-30)

    P_density_W_Hz = P_noise_total_W / RBW_Hz
    P_density_dBmHz = 10.0 * np.log10(P_density_W_Hz * 1000.0 + 1e-30)

    noise_res = NoiseResult(
        P_thermal_W=P_thermal_W,
        P_shot_W=P_shot_W,
        P_rin_W=P_rin_W,
        P_noise_total_W=P_noise_total_W,
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

    P_sig_W = 0.5 * (P1 ** 2) * R_load
    if P1.size > 0:
        P_sig_W[0] = (P1[0] ** 2) * R_load

    rng = np.random.default_rng()
    noise_trace_W = P_noise_total_W * (-np.log(rng.random(P_sig_W.shape)))

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
    )


def mzm_dc_power_mW(
    V_bias: np.ndarray | float,
    *,
    Vpi_DC: float = 5.0,
    ER_dB: float = 30.0,
    IL_dB: float = 6.0,
    Pin_dBm: float = 10.0,
) -> np.ndarray:
    """Compute MZM DC (no-RF) optical power vs bias voltage.

    This matches the bias-scan model used inside simulate_mzm().
    Returns optical power in mW.
    """

    V_bias_arr = np.asarray(V_bias, dtype=float)

    Pin_W = 10 ** ((Pin_dBm - 30.0) / 10.0)
    E_in = np.sqrt(Pin_W)

    loss_factor = 10 ** (-IL_dB / 10.0)
    er_linear = 10 ** (ER_dB / 20.0)
    gamma = (er_linear - 1.0) / (er_linear + 1.0)

    phi1 = (np.pi / Vpi_DC) * (V_bias_arr / 2.0)
    phi2 = (np.pi / Vpi_DC) * (-V_bias_arr / 2.0)

    E_out = E_in * np.sqrt(loss_factor) * 0.5 * (np.exp(1j * phi1) + gamma * np.exp(1j * phi2))
    P_mW = (np.abs(E_out) ** 2) * 1000.0
    return P_mW


def mzm_dc_power_curve_mW(
    *,
    Vpi_DC: float = 5.0,
    ER_dB: float = 30.0,
    IL_dB: float = 6.0,
    Pin_dBm: float = 10.0,
    V_min: float | None = None,
    V_max: float | None = None,
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (V_scan, P_scan_mW) for DC bias sweep."""

    if V_min is None:
        V_min = 0.0
    if V_max is None:
        V_max = Vpi_DC

    V_scan = np.linspace(float(V_min), float(V_max), int(n_points))
    P_scan_mW = mzm_dc_power_mW(
        V_scan,
        Vpi_DC=Vpi_DC,
        ER_dB=ER_dB,
        IL_dB=IL_dB,
        Pin_dBm=Pin_dBm,
    )
    return V_scan, P_scan_mW


def simulation_summary(sim: SimulationResult) -> Dict[str, Any]:
    """Return a compact dict summary for quick inspection/printing."""

    return {
        "Pin_dBm": sim.Pin_dBm,
        "P_pd_avg_dBm": sim.P_pd_avg_dBm,
        "RBW_Hz": sim.RBW_Hz,
        "P_noise_floor_dBm": sim.noise.P_noise_floor_dBm,
        "P_density_dBmHz": sim.noise.P_density_dBmHz,
        "val_1G_dBm": sim.spectrum.val_1G_dBm,
        "val_2G_dBm": sim.spectrum.val_2G_dBm,
    }


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


def lockin_harmonics(
    signal: np.ndarray,
    *,
    Fs: float,
    f_ref: float,
    harmonics: tuple[int, ...] = (1, 2),
) -> Dict[int, Dict[str, float]]:
    """Extract lock-in components at n*f_ref.

    Returns a dict:
        {n: {"I": in_phase_sin, "Q": quadrature_cos, "A": magnitude}}

    Notes:
    - We use sin/cos references and scale by 2/N so that a pure tone
      signal(t)=a*sin(2*pi*f*t) yields I≈a, Q≈0.
    """

    x = np.asarray(signal, dtype=float)
    N = x.size
    if N == 0:
        return {n: {"I": 0.0, "Q": 0.0, "A": 0.0} for n in harmonics}

    t = np.arange(N, dtype=float) / float(Fs)
    out: Dict[int, Dict[str, float]] = {}
    for n in harmonics:
        w = 2.0 * np.pi * (float(n) * float(f_ref))
        s = np.sin(w * t)
        c = np.cos(w * t)
        I = float((2.0 / N) * np.sum(x * s))
        Q = float((2.0 / N) * np.sum(x * c))
        A = float(np.sqrt(I * I + Q * Q))
        out[int(n)] = {"I": I, "Q": Q, "A": A}
    return out


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

    if f_dither <= 0 or Fs <= 0:
        raise ValueError("f_dither and Fs must be > 0")

    n_samples = int(round((float(n_periods) / float(f_dither)) * float(Fs)))
    if n_samples < 16:
        raise ValueError("n_samples too small; increase Fs or n_periods")

    t = np.arange(n_samples, dtype=float) / float(Fs)
    V_t = float(V_bias) + float(V_dither_amp) * np.sin(2.0 * np.pi * float(f_dither) * t)

    # Use the same DC transfer model as mzm_dc_power_mW, then convert to PD current.
    P_mW = mzm_dc_power_mW(V_t, Vpi_DC=Vpi_DC, ER_dB=ER_dB, IL_dB=IL_dB, Pin_dBm=Pin_dBm)
    P_W = P_mW / 1000.0
    I_pd = float(Responsivity) * P_W

    pd_dc = float(np.mean(I_pd))
    I_ac = I_pd - pd_dc

    h = lockin_harmonics(I_ac, Fs=Fs, f_ref=float(f_dither), harmonics=(1, 2))

    feature = feature.lower().strip()
    if feature not in {"signed", "power"}:
        raise ValueError("feature must be 'signed' or 'power'")

    # Always expose full lock-in components for downstream feature engineering.
    h1_I = float(h[1]["I"])
    h1_Q = float(h[1]["Q"])
    h1_A = float(h[1]["A"])
    h2_I = float(h[2]["I"])
    h2_Q = float(h[2]["Q"])
    h2_A = float(h[2]["A"])

    # Convert harmonic current amplitude (peak) into electrical power at load.
    # If i(t) = A*sin(wt), then P_avg = (A^2 / 2) * R.
    def _amp_to_power_dBm(A: float, R: float) -> tuple[float, float]:
        P_W = 0.5 * (float(A) ** 2) * float(R)
        P_dBm = 10.0 * np.log10(P_W * 1000.0 + 1e-30)
        return float(P_W), float(P_dBm)

    p1_W, p1_dBm = _amp_to_power_dBm(h1_A, R_load)
    p2_W, p2_dBm = _amp_to_power_dBm(h2_A, R_load)

    # Two-dimensional features for convenience.
    # - For 1f, the signed in-phase component preserves slope direction.
    # - For 2f, magnitude is robust to reference phase offsets.
    if feature == "signed":
        h1 = h1_I
        h2 = h2_A
    else:
        h1 = float(h1_A ** 2)
        h2 = float(h2_A ** 2)

    theta = float(bias_to_theta_rad(V_bias, Vpi_DC=Vpi_DC))
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
