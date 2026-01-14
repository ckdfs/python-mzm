"""Common utilities for MZM dither controller.

Provides shared logic for device selection, lock-in reference generation,
and resource cleanup to avoid code duplication across modules.
"""

from __future__ import annotations

import gc
import numpy as np
import torch

# Type checking import only to avoid circular dependency issues at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mzm.dither_controller import DitherParams


def select_device(accel: str = "auto") -> torch.device:
    """Select PyTorch device based on preference and availability.

    Args:
        accel: One of 'cpu', 'cuda', 'mps', 'auto'.

    Returns:
        torch.device
    """
    accel_norm = str(accel).lower().strip()
    if accel_norm not in {"cpu", "auto", "cuda", "mps"}:
        raise ValueError("accel must be one of: 'cpu', 'auto', 'cuda', 'mps'")

    if accel_norm == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("accel='cuda' requested but torch.cuda.is_available() is False")
    if accel_norm == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("accel='mps' requested but MPS is not available")

    if accel_norm == "cpu":
        return torch.device("cpu")
    elif accel_norm == "cuda":
        return torch.device("cuda")
    elif accel_norm == "mps":
        return torch.device("mps")
    else:
        # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


def make_lockin_refs(
    dither_params: "DitherParams",
    device: torch.device
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Generate sin/cos reference waveforms for lock-in detection.

    Args:
        dither_params: Object containing f_dither, Fs, n_periods.
        device: The torch device to store tensors on.

    Returns:
        Dict mapping harmonic order (1, 2) to (sin_ref, cos_ref) tensors.
    """
    n_samples_time = int(
        round((float(dither_params.n_periods) / float(dither_params.f_dither)) * float(dither_params.Fs))
    )
    t = torch.arange(n_samples_time, device=device, dtype=torch.float32) / float(dither_params.Fs)
    w = 2.0 * float(np.pi) * float(dither_params.f_dither)
    
    return {
        1: (torch.sin(w * t), torch.cos(w * t)),
        2: (torch.sin(2.0 * w * t), torch.cos(2.0 * w * t)),
    }


def cleanup_torch() -> None:
    """Force garbage collection and empty PyTorch caches (CUDA/MPS)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
