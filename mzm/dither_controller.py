"""Dither-based closed-loop MZM bias controller (reusable module).

What this implements
- You can ONLY observe PD low-frequency pilot (dither) 1f/2f power.
- Features are normalized by DC photocurrent to handle optical power fluctuations.
- You do NOT assume signed I/Q lock-in outputs.

To recover direction (which is lost if you only use power magnitudes), the controller
uses a *history finite difference* (more realistic than taking an extra probe sample):
    x_k = [P1/pd_dc, P2/pd_dc, dP1_norm, dP2_norm, dV_{k-1}, sin(theta*), cos(theta*)]
where dP*_norm = (P*/pd_dc)(k) - (P*/pd_dc)(k-1).

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
- For a CLI entry, use scripts/train_mzm_dither_controller.py.
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
    accel: str = "auto",
    torch_batch: int = 512,
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

    # Create a "previous" step consistent with a realistic controller history.
    dv_prev = rng.uniform(-max_step_V, max_step_V, size=n_samples).astype(np.float32)
    V_prev = np.clip(V_bias - dv_prev, 0.0, Vpi).astype(np.float32)

    accel_norm = str(accel).lower().strip()
    if accel_norm not in {"cpu", "auto", "cuda", "mps"}:
        raise ValueError("accel must be one of: 'cpu', 'auto', 'cuda', 'mps'")

    if accel_norm == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("accel='cuda' requested but torch.cuda.is_available() is False")
    if accel_norm == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("accel='mps' requested but MPS is not available")

    if accel_norm == "cpu":
        device = torch.device("cpu")
    elif accel_norm == "cuda":
        device = torch.device("cuda")
    elif accel_norm == "mps":
        device = torch.device("mps")
    else:
        # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"generate_dataset_dbm_hist using device: {device}")

    # Precompute lock-in references once per call, on the selected device.
    n_samples_time = int(
        round((float(dither_params.n_periods) / float(dither_params.f_dither)) * float(dither_params.Fs))
    )
    t = torch.arange(n_samples_time, device=device, dtype=torch.float32) / float(dither_params.Fs)
    w = 2.0 * float(np.pi) * float(dither_params.f_dither)
    refs = {
        1: (torch.sin(w * t), torch.cos(w * t)),
        2: (torch.sin(2.0 * w * t), torch.cos(2.0 * w * t)),
    }

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
            )
            h1_out[start:end] = h1b.detach().cpu().numpy()
            h2_out[start:end] = h2b.detach().cpu().numpy()
        return h1_out, h2_out

    h1, h2 = _measure_normalized_torch(V_bias)
    h1_prev, h2_prev = _measure_normalized_torch(V_prev)

    dh1 = (h1 - h1_prev).astype(np.float32)
    dh2 = (h2 - h2_prev).astype(np.float32)
    te = _target_encoding(theta_target.astype(np.float32))

    X = np.stack(
        [
            h1.astype(np.float32),   # DC-normalized 1f amplitude
            h2.astype(np.float32),   # DC-normalized 2f amplitude
            dh1,                      # delta h1_norm
            dh2,                      # delta h2_norm
            dv_prev.astype(np.float32),
            te[:, 0].astype(np.float32),
            te[:, 1].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

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
    accel: str = "auto",
) -> nn.Module:
    torch.manual_seed(seed)

    accel_norm = str(accel).lower().strip()
    if accel_norm not in {"cpu", "auto", "cuda", "mps"}:
        raise ValueError("accel must be one of: 'cpu', 'auto', 'cuda', 'mps'")

    if accel_norm == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("accel='cuda' requested but torch.cuda.is_available() is False")
    if accel_norm == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("accel='mps' requested but MPS is not available")

    if accel_norm == "cpu":
        device = torch.device("cpu")
    elif accel_norm == "cuda":
        device = torch.device("cuda")
    elif accel_norm == "mps":
        device = torch.device("mps")
    else:
        # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

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
        opt, mode='min', factor=0.5, patience=500, min_lr=1e-6
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
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state": model.state_dict(),
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
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
) -> dict:
    accel_norm = str(accel).lower().strip()
    if accel_norm not in {"cpu", "auto", "cuda", "mps"}:
        raise ValueError("accel must be one of: 'cpu', 'auto', 'cuda', 'mps'")
    if accel_norm == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("accel='cuda' requested but torch.cuda.is_available() is False")
    if accel_norm == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("accel='mps' requested but MPS is not available")

    if accel_norm == "cpu":
        device = torch.device("cpu")
    elif accel_norm == "cuda":
        device = torch.device("cuda")
    elif accel_norm == "mps":
        device = torch.device("mps")
    else:
        # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # print(f"rollout_dbm_hist using device: {device}")

    model = model.to(device)

    # Precompute lock-in references once on the chosen device.
    n_samples_time = int(
        round((float(dither_params.n_periods) / float(dither_params.f_dither)) * float(dither_params.Fs))
    )
    t_ref = torch.arange(n_samples_time, device=device, dtype=torch.float32) / float(dither_params.Fs)
    w = 2.0 * float(np.pi) * float(dither_params.f_dither)
    refs = {
        1: (torch.sin(w * t_ref), torch.cos(w * t_ref)),
        2: (torch.sin(2.0 * w * t_ref), torch.cos(2.0 * w * t_ref)),
    }

    Vpi = float(device_params.Vpi_DC)
    th_t = float(np.deg2rad(theta_target_deg))

    V = float(np.clip(V_init, 0.0, Vpi))

    prev_h1 = None
    prev_h2 = None
    prev_dv = 0.0

    V_hist: list[float] = []
    err_deg_hist: list[float] = []
    dv_hist: list[float] = []
    h1_hist: list[float] = []
    h2_hist: list[float] = []
    dh1_hist: list[float] = []
    dh2_hist: list[float] = []
    theta_deg_hist: list[float] = []

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
        )

        h1 = float(h1_t[0].item())
        h2 = float(h2_t[0].item())
        dh1 = 0.0 if prev_h1 is None else (h1 - float(prev_h1))
        dh2 = 0.0 if prev_h2 is None else (h2 - float(prev_h2))

        te = _target_encoding(np.array([th_t], dtype=np.float32))[0]
        x = np.array([h1, h2, dh1, dh2, float(prev_dv), float(te[0]), float(te[1])], dtype=np.float32)
        xn = (x - mu) / sigma

        dv = float(model(torch.from_numpy(xn).to(device).unsqueeze(0)).cpu().numpy().reshape(-1)[0])
        V = float(np.clip(V + dv, 0.0, Vpi))

        prev_h1, prev_h2 = h1, h2
        prev_dv = dv

        th_c = float(bias_to_theta_rad(V, Vpi_DC=device_params.Vpi_DC))
        err = float(wrap_to_pi(th_t - th_c))

        V_hist.append(V)
        err_deg_hist.append(float(np.rad2deg(err)))
        dv_hist.append(dv)
        h1_hist.append(h1)
        h2_hist.append(h2)
        dh1_hist.append(dh1)
        dh2_hist.append(dh2)
        theta_deg_hist.append(float(np.rad2deg(th_c)))

    return {
        "V": np.asarray(V_hist, dtype=float),
        "err_deg": np.asarray(err_deg_hist, dtype=float),
        "dv": np.asarray(dv_hist, dtype=float),
        "h1_norm": np.asarray(h1_hist, dtype=float),
        "h2_norm": np.asarray(h2_hist, dtype=float),
        "dh1_norm": np.asarray(dh1_hist, dtype=float),
        "dh2_norm": np.asarray(dh2_hist, dtype=float),
        "theta_deg": np.asarray(theta_deg_hist, dtype=float),
    }


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
) -> dict:
    accel_norm = str(accel).lower().strip()
    if accel_norm not in {"cpu", "auto", "cuda", "mps"}:
        raise ValueError("accel must be one of: 'cpu', 'auto', 'cuda', 'mps'")

    if accel_norm == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("accel='cuda' requested but torch.cuda.is_available() is False")
    if accel_norm == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("accel='mps' requested but MPS is not available")

    if accel_norm == "cpu":
        device = torch.device("cpu")
    elif accel_norm == "cuda":
        device = torch.device("cuda")
    elif accel_norm == "mps":
        device = torch.device("mps")
    else:
        # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"rollout_dbm_hist_batch using device: {device}")

    model = model.to(device)

    # Convert inputs to tensors on device
    V = torch.from_numpy(V_init).to(device=device, dtype=torch.float32)
    th_t_rad = torch.deg2rad(torch.from_numpy(theta_target_deg).to(device=device, dtype=torch.float32))

    # Constants
    Vpi = float(device_params.Vpi_DC)
    mu_t = torch.from_numpy(mu).to(device=device, dtype=torch.float32)
    sigma_t = torch.from_numpy(sigma).to(device=device, dtype=torch.float32)

    # Precompute lock-in references
    n_samples_time = int(
        round((float(dither_params.n_periods) / float(dither_params.f_dither)) * float(dither_params.Fs))
    )
    t_ref = torch.arange(n_samples_time, device=device, dtype=torch.float32) / float(dither_params.Fs)
    w = 2.0 * float(np.pi) * float(dither_params.f_dither)
    refs = {
        1: (torch.sin(w * t_ref), torch.cos(w * t_ref)),
        2: (torch.sin(2.0 * w * t_ref), torch.cos(2.0 * w * t_ref)),
    }

    # State initialization
    prev_h1 = None
    prev_h2 = None
    prev_dv = torch.zeros_like(V)

    # History storage (on CPU to save GPU memory if steps are large, or keep on GPU?)
    # Keeping on CPU is safer for large batches.
    err_deg_hist = []

    # Target encoding (constant throughout rollout)
    # sin(th), cos(th)
    te_sin = torch.sin(th_t_rad)
    te_cos = torch.cos(th_t_rad)

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
            Pin_dBm=float(device_params.Pin_dBm),
            Responsivity=float(device_params.Responsivity),
            R_load=float(device_params.R_load),
            refs=refs,
        )

        # Deltas
        if prev_h1 is None:
            dh1 = torch.zeros_like(h1)
            dh2 = torch.zeros_like(h2)
        else:
            dh1 = h1 - prev_h1
            dh2 = h2 - prev_h2

        # Construct input
        # x: [h1, h2, dh1, dh2, prev_dv, sin(th), cos(th)]
        x = torch.stack([h1, h2, dh1, dh2, prev_dv, te_sin, te_cos], dim=1)
        xn = (x - mu_t) / sigma_t

        # Inference
        dv = model(xn).squeeze(1)  # (N, 1) -> (N,)

        # Update
        V = torch.clamp(V + dv, 0.0, Vpi)

        # Update state
        prev_h1 = h1
        prev_h2 = h2
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
            accel="auto",
        )
        final_err = float(r["err_deg"][-1]) if r["err_deg"].size else float("nan")
        print(f"rollout target={tgt:6.1f} deg | initV={V_init:.3f} V | final_err={final_err:+.2f} deg")

    return 0
