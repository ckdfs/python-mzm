"""Evaluation utilities for MZM dither controller.

Provides reusable helpers to run batch rollouts for angle sweep, RF-power
robustness, multi-target RF robustness, and optical-power robustness.

Each helper loads the model from disk, performs batch rollout using the
existing torch-accelerated routines in ``mzm.dither_controller`` and returns
simple NumPy dictionaries that plotting functions can consume.
"""

from __future__ import annotations

import gc
import hashlib
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import mzm.dither_controller as dc


def _load_model(model_path: str | Path):
    """Load model and metadata; re-raise FileNotFoundError with path info."""

    try:
        return dc.load_model(Path(model_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found: {model_path}") from None


def _file_hash(filepath: str | Path) -> str | None:
    path = Path(filepath)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _save_eval_npz(path: str | Path, stats: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        targets=stats["targets"],
        mean=stats["mean"],
        std=stats["std"],
        min=stats["min"],
        max=stats["max"],
    )


def evaluate_model(
    model_path: str | Path,
    *,
    steps: int = 60,
    n_repeats: int = 50,
    V_rf_amp: float = 0.0,
    f_rf: float = 1e9,
    accel: str = "auto",
    seed: int | None = None,
    label: str | None = None,
) -> dict:
    """Angle sweep 0–180° with repeats; returns mean/std/min/max per angle."""

    model, meta = _load_model(model_path)

    targets = np.arange(0, 181, 1, dtype=float)
    device_params = meta["device_params"]
    dither_params = meta["dither_params"]
    mu = meta["mu"]
    sigma = meta["sigma"]

    rng = np.random.default_rng(seed)
    targets_repeated = np.repeat(targets, n_repeats)
    n_total = targets_repeated.size
    V_init_batch = rng.uniform(0.0, float(device_params.Vpi_DC), size=n_total)

    r = dc.rollout_dbm_hist_batch(
        model=model,
        mu=mu,
        sigma=sigma,
        device_params=device_params,
        dither_params=dither_params,
        theta_target_deg=targets_repeated,
        V_init=V_init_batch,
        steps=steps,
        accel=accel,
        V_rf_amp=float(V_rf_amp),
        f_rf=float(f_rf),
    )

    final_errs = np.abs(r["err_deg"][:, -1])
    del r, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    final_errs_reshaped = final_errs.reshape(len(targets), n_repeats)
    means = np.mean(final_errs_reshaped, axis=1)
    stds = np.std(final_errs_reshaped, axis=1)
    mins = np.min(final_errs_reshaped, axis=1)
    maxs = np.max(final_errs_reshaped, axis=1)

    return {
        "targets": targets,
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
        "label": label or Path(model_path).name,
    }


def evaluate_current_and_best(
    model_path: str | Path,
    *,
    best_model_path: str | Path,
    best_eval_path: str | Path,
    steps: int = 60,
    n_repeats: int = 50,
    V_rf_amp: float = 0.0,
    f_rf: float = 1e9,
    accel: str = "auto",
    seed: int | None = None,
) -> tuple[dict, dict | None, str | None]:
    """
    评估当前模型，并与历史最佳进行对比与更新。

    返回 (current_stats, best_stats, update_action)。best_stats 可能为 None。
    """

    model_path = Path(model_path)
    best_model_path = Path(best_model_path)
    best_eval_path = Path(best_eval_path)

    current_stats = evaluate_model(
        model_path,
        steps=steps,
        n_repeats=n_repeats,
        V_rf_amp=V_rf_amp,
        f_rf=f_rf,
        accel=accel,
        seed=seed,
        label="Current Model",
    )

    # If feature versions differ, comparisons are meaningless; keep best untouched.
    try:
        _, current_meta = _load_model(model_path)
        current_feature_version = str(current_meta.get("feature_version", "legacy"))
    except FileNotFoundError:
        current_feature_version = "legacy"

    best_stats: dict | None = None
    best_feature_version: str | None = None
    if best_model_path.exists():
        try:
            _, best_meta = _load_model(best_model_path)
            best_feature_version = str(best_meta.get("feature_version", "legacy"))
        except FileNotFoundError:
            best_feature_version = None

    if best_feature_version is not None and best_feature_version != current_feature_version:
        update_action = (
            f"Best model feature_version mismatch (best={best_feature_version}, current={current_feature_version}); "
            "skipping best comparison/update."
        )
        return current_stats, None, update_action

    if best_eval_path.exists():
        data = np.load(best_eval_path)
        best_stats = {k: data[k] for k in data.files}
        best_stats["label"] = "Best Model (Cached)"
    elif best_model_path.exists():
        best_stats = evaluate_model(
            best_model_path,
            steps=steps,
            n_repeats=n_repeats,
            V_rf_amp=V_rf_amp,
            f_rf=f_rf,
            accel=accel,
            seed=seed,
            label="Best Model",
        )

    update_action: str | None = None

    if best_stats is None:
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, best_model_path)
        _save_eval_npz(best_eval_path, current_stats)
        best_stats = current_stats
        update_action = f"已初始化最佳模型与评估缓存：{best_model_path} / {best_eval_path}"
    else:
        curr_mae = float(np.mean(current_stats["mean"]))
        best_mae = float(np.mean(best_stats["mean"]))
        if curr_mae < best_mae:
            curr_hash = _file_hash(model_path)
            best_hash = _file_hash(best_model_path)
            if curr_hash is not None and curr_hash == best_hash:
                _save_eval_npz(best_eval_path, current_stats)
                best_stats = current_stats
                update_action = "模型文件相同，更新最佳评估缓存。"
            else:
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(model_path, best_model_path)
                _save_eval_npz(best_eval_path, current_stats)
                best_stats = current_stats
                update_action = "当前模型更优，已覆盖最佳模型并刷新评估缓存。"

    return current_stats, best_stats, update_action


def evaluate_rf_power_robustness(
    model_path: str | Path,
    *,
    V_rf_amp_values: Iterable[float] | None = None,
    f_rf: float = 1e9,
    target_deg: float = 90.0,
    steps: int = 60,
    n_repeats: int = 100,
    accel: str = "auto",
    seed: int = 42,
) -> dict:
    """Evaluate robustness vs RF amplitude at a single target angle."""

    if V_rf_amp_values is None:
        V_rf_amp_values = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    V_rf_amp_values = np.asarray(V_rf_amp_values, dtype=float)

    model, meta = _load_model(model_path)
    device_params = meta["device_params"]
    dither_params = meta["dither_params"]
    mu = meta["mu"]
    sigma = meta["sigma"]

    rng = np.random.default_rng(seed)

    mean_errors: list[float] = []
    std_errors: list[float] = []
    min_errors: list[float] = []
    max_errors: list[float] = []

    for V_rf_amp in V_rf_amp_values:
        targets_repeated = np.full(n_repeats, float(target_deg), dtype=float)
        V_init_batch = rng.uniform(0.0, float(device_params.Vpi_DC), size=n_repeats)

        r = dc.rollout_dbm_hist_batch(
            model=model,
            mu=mu,
            sigma=sigma,
            device_params=device_params,
            dither_params=dither_params,
            theta_target_deg=targets_repeated,
            V_init=V_init_batch,
            steps=steps,
            accel=accel,
            V_rf_amp=float(V_rf_amp),
            f_rf=float(f_rf),
        )

        final_errs = np.abs(r["err_deg"][:, -1])
        mean_errors.append(float(np.mean(final_errs)))
        std_errors.append(float(np.std(final_errs)))
        min_errors.append(float(np.min(final_errs)))
        max_errors.append(float(np.max(final_errs)))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "V_rf_amp_values": V_rf_amp_values,
        "f_rf": float(f_rf),
        "mean_errors": np.array(mean_errors, dtype=float),
        "std_errors": np.array(std_errors, dtype=float),
        "min_errors": np.array(min_errors, dtype=float),
        "max_errors": np.array(max_errors, dtype=float),
    }


def evaluate_rf_robustness_multi_target(
    model_path: str | Path,
    *,
    V_rf_amp_values: Iterable[float] | None = None,
    f_rf: float = 1e9,
    target_angles: Iterable[float] | None = None,
    steps: int = 60,
    n_repeats: int = 50,
    accel: str = "auto",
    seed: int = 42,
) -> dict:
    """Evaluate RF robustness across multiple target angles and amplitudes."""

    if V_rf_amp_values is None:
        V_rf_amp_values = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
    if target_angles is None:
        target_angles = np.array([0, 45, 90, 135, 180])

    V_rf_amp_values = np.asarray(V_rf_amp_values, dtype=float)
    target_angles = np.asarray(target_angles, dtype=float)

    model, meta = _load_model(model_path)
    device_params = meta["device_params"]
    dither_params = meta["dither_params"]
    mu = meta["mu"]
    sigma = meta["sigma"]

    rng = np.random.default_rng(seed)

    mean_errors = np.zeros((len(V_rf_amp_values), len(target_angles)), dtype=float)
    std_errors = np.zeros_like(mean_errors)

    for i, V_rf_amp in enumerate(V_rf_amp_values):
        for j, target in enumerate(target_angles):
            targets_repeated = np.full(n_repeats, float(target), dtype=float)
            V_init_batch = rng.uniform(0.0, float(device_params.Vpi_DC), size=n_repeats)

            r = dc.rollout_dbm_hist_batch(
                model=model,
                mu=mu,
                sigma=sigma,
                device_params=device_params,
                dither_params=dither_params,
                theta_target_deg=targets_repeated,
                V_init=V_init_batch,
                steps=steps,
                accel=accel,
                V_rf_amp=float(V_rf_amp),
                f_rf=float(f_rf),
            )

            final_errs = np.abs(r["err_deg"][:, -1])
            mean_errors[i, j] = float(np.mean(final_errs))
            std_errors[i, j] = float(np.std(final_errs))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "V_rf_amp_values": V_rf_amp_values,
        "f_rf": float(f_rf),
        "target_angles": target_angles,
        "mean_errors": mean_errors,
        "std_errors": std_errors,
    }


def evaluate_optical_power_robustness(
    model_path: str | Path,
    *,
    Pin_dBm_values: Iterable[float] | None = None,
    target_deg: float = 90.0,
    steps: int = 60,
    n_repeats: int = 100,
    V_rf_amp: float = 0.0,
    f_rf: float = 1e9,
    accel: str = "auto",
    seed: int = 42,
) -> dict:
    """Evaluate robustness vs input optical power (Pin_dBm)."""

    if Pin_dBm_values is None:
        Pin_dBm_values = np.array([0.0, 3.0, 6.0, 8.0, 10.0, 12.0, 15.0])
    Pin_dBm_values = np.asarray(Pin_dBm_values, dtype=float)

    model, meta = _load_model(model_path)
    device_params = meta["device_params"]
    dither_params = meta["dither_params"]
    mu = meta["mu"]
    sigma = meta["sigma"]

    training_Pin = float(device_params.Pin_dBm)
    rng = np.random.default_rng(seed)

    mean_errors: list[float] = []
    std_errors: list[float] = []
    min_errors: list[float] = []
    max_errors: list[float] = []

    for Pin in Pin_dBm_values:
        targets_repeated = np.full(n_repeats, float(target_deg), dtype=float)
        V_init_batch = rng.uniform(0.0, float(device_params.Vpi_DC), size=n_repeats)

        r = dc.rollout_dbm_hist_batch(
            model=model,
            mu=mu,
            sigma=sigma,
            device_params=device_params,
            dither_params=dither_params,
            theta_target_deg=targets_repeated,
            V_init=V_init_batch,
            steps=steps,
            accel=accel,
            V_rf_amp=float(V_rf_amp),
            f_rf=float(f_rf),
            Pin_dBm=float(Pin),
        )

        final_errs = np.abs(r["err_deg"][:, -1])
        mean_errors.append(float(np.mean(final_errs)))
        std_errors.append(float(np.std(final_errs)))
        min_errors.append(float(np.min(final_errs)))
        max_errors.append(float(np.max(final_errs)))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "Pin_dBm_values": Pin_dBm_values,
        "training_Pin_dBm": training_Pin,
        "mean_errors": np.array(mean_errors, dtype=float),
        "std_errors": np.array(std_errors, dtype=float),
        "min_errors": np.array(min_errors, dtype=float),
        "max_errors": np.array(max_errors, dtype=float),
    }
