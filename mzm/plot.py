"""Plot helpers for MZM simulation.

Provide reusable functions to plot optical spectrum, electrical spectrum,
and bias scan from SimulationResult.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from mzm.model import SimulationResult


def _annotate_optical_orders(f_plot_opt: np.ndarray, p_plot_opt: np.ndarray, f_rf_hz: float, max_order: int = 2) -> None:
    """Annotate carrier and ±n-th optical sidebands on an optical spectrum.

    f_plot_opt: frequency relative to carrier (GHz)
    p_plot_opt: power (dBm)
    f_rf_hz: RF frequency (Hz)
    """

    if f_rf_hz is None or f_rf_hz <= 0:
        return

    f_rf_ghz = f_rf_hz / 1e9

    def find_peak(center_ghz: float, span_factor: float = 0.2):
        # search for peak within center_ghz ± span_factor*f_rf_ghz
        delta = span_factor * f_rf_ghz
        mask = (f_plot_opt > center_ghz - delta) & (f_plot_opt < center_ghz + delta)
        if not np.any(mask):
            return None, None
        local_f = f_plot_opt[mask]
        local_p = p_plot_opt[mask]
        idx = int(np.argmax(local_p))
        return float(local_f[idx]), float(local_p[idx])

    # carrier around 0 GHz
    c_f, c_p = find_peak(0.0, span_factor=0.1)
    if c_f is not None:
        plt.scatter(c_f, c_p, color="black", s=25, zorder=3)
        plt.text(
            c_f,
            c_p + 1.5,
            f"Carrier\n{c_p:.1f} dBm",
            ha="center",
            fontsize=8,
        )

    # ±n-th orders
    for n in range(1, max_order + 1):
        # +n
        center_pos = n * f_rf_ghz
        pf, pp = find_peak(center_pos)
        if pf is not None:
            plt.scatter(pf, pp, color="blue", s=20, zorder=3)
            plt.text(
                pf,
                pp + 1.5,
                f"+{n}th\n{pp:.1f} dBm",
                ha="center",
                fontsize=8,
            )

        # -n
        center_neg = -n * f_rf_ghz
        nf, np_ = find_peak(center_neg)
        if nf is not None:
            plt.scatter(nf, np_, color="blue", s=20, zorder=3)
            plt.text(
                nf,
                np_ + 1.5,
                f"-{n}th\n{np_:.1f} dBm",
                ha="center",
                fontsize=8,
            )


def plot_optical_spectrum_osa(sim: SimulationResult, f_rf_hz: float = 1e9, span_factor: float = 2.5, max_order: int = 2) -> None:
    """Plot optical spectrum around carrier with carrier and ±1/±2 orders annotated.

    The spectrum is computed from the complex field sim.E_out, using the
    same normalization as in the reference OSA-style code.
    """

    E_out_t = sim.E_out
    t = sim.t
    dt = t[1] - t[0]
    Fs = 1.0 / dt
    L = len(t)

    # FFT + shift
    E_spec = np.fft.fft(E_out_t)
    E_spec = np.fft.fftshift(E_spec)
    P_spec = np.abs(E_spec) ** 2

    # normalize such that total spectral power matches time-domain average
    P_time_avg = np.mean(np.abs(E_out_t) ** 2)
    scale_factor = P_time_avg / np.sum(P_spec)
    P_spec = P_spec * scale_factor

    P_spec_dBm = 10 * np.log10(P_spec * 1000 + 1e-20)

    f_opt = Fs * np.arange(-L // 2, L // 2) / L
    disp_span = span_factor * f_rf_hz
    idx_opt = (f_opt >= -disp_span) & (f_opt <= disp_span)

    f_plot_opt = f_opt[idx_opt] / 1e9
    p_plot_opt = P_spec_dBm[idx_opt]

    plt.figure(figsize=(8, 5))
    plt.plot(f_plot_opt, p_plot_opt, color="#cc3333", linewidth=1)

    _annotate_optical_orders(f_plot_opt, p_plot_opt, f_rf_hz=f_rf_hz, max_order=max_order)

    plt.xlabel("Frequency Relative to Carrier (GHz)")
    plt.ylabel("Optical Power (dBm)")
    plt.title("Optical Spectrum (OSA View)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim(bottom=-100)
    plt.tight_layout()


def _annotate_rf_orders(f_cpu: np.ndarray, p_cpu: np.ndarray, f_rf_hz: float, orders=(1, 2), color: str = "orange") -> None:
    """Annotate given RF harmonic orders on an electrical spectrum.

    f_cpu: frequency axis (Hz)
    p_cpu: power (dBm)
    f_rf_hz: fundamental RF frequency (Hz)
    orders: tuple of harmonic orders, e.g. (1, 2)
    """

    if f_rf_hz is None or f_rf_hz <= 0:
        return

    if f_cpu.size == 0 or np.all(~np.isfinite(p_cpu)):
        return

    for n in orders:
        target = n * f_rf_hz
        delta = 0.05 * f_rf_hz
        mask = (f_cpu > target - delta) & (f_cpu < target + delta)
        if not np.any(mask):
            continue

        local_f = f_cpu[mask]
        local_p = p_cpu[mask]
        idx = int(np.argmax(local_p))
        peak_f = float(local_f[idx])
        peak_p = float(local_p[idx])

        x_mhz = peak_f / 1e6
        plt.scatter(x_mhz, peak_p, color=color, s=20, zorder=3)
        plt.text(
            x_mhz,
            peak_p + 2,
            f"{n}×RF\n{peak_p:.1f} dBm",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_electrical_spectrum(sim: SimulationResult, f_rf_hz: float = 1e9, harmonic_orders=(1, 2)) -> None:
    """Plot electrical spectrum at PD output, with noise floor and RF harmonics.

    Uses the electrical spectrum and noise information already computed in
    SimulationResult.spectrum and SimulationResult.noise.
    """

    f_cpu = sim.spectrum.f_elec
    p_cpu = sim.spectrum.P_elec_spec_dBm

    RBW_Sim_Hz = sim.RBW_Hz
    P_noise_density_dBm_Hz = sim.noise.P_density_dBmHz
    Calculated_Floor_dBm = P_noise_density_dBm_Hz + 10 * np.log10(RBW_Sim_Hz)

    f_show_max = f_rf_hz * 5.0
    idx_show = f_cpu <= f_show_max

    f_plot = f_cpu[idx_show]
    p_plot = p_cpu[idx_show]

    plt.figure(figsize=(10, 5))
    plt.plot(f_plot / 1e6, p_plot, color="#3366cc", linewidth=1)
    plt.axhline(
        y=Calculated_Floor_dBm,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=f"Noise Floor @ RBW={RBW_Sim_Hz/1e3:.1f} kHz",
    )

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("RF Power (dBm)")
    plt.title("Electrical Spectrum at PD Output")
    plt.grid(True, linestyle="--", alpha=0.7)
    if p_plot.size:
        plt.ylim(bottom=-160, top=float(np.max(p_plot)) + 10)

    _annotate_rf_orders(f_plot, p_plot, f_rf_hz, orders=harmonic_orders)

    plt.legend()
    plt.tight_layout()


def plot_bias_scan(sim: SimulationResult) -> None:
    """Plot bias scan curve, highlighting current bias point."""

    plt.figure(figsize=(6, 4))
    plt.plot(sim.bias_scan.V_scan, sim.bias_scan.P_scan_mW, "k", linewidth=1.5)
    plt.scatter([sim.bias_scan.V_bias], [sim.bias_scan.curr_P_mW], color="r", zorder=3)
    plt.axvline(sim.bias_scan.V_bias, color="r", linestyle="--")
    plt.title("偏压扫描曲线")
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Optical Power (mW)")
    plt.grid(True)
    plt.tight_layout()
