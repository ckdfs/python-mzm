# python-mzm agent notes

This repo is a research/prototype implementation of an MZM bias dither controller trained with supervised learning.

## Key entry points

- Notebook pipeline (dataset/train/inference/eval): `mzm_dither_controller.ipynb`
- Main controller module (features, dataset, rollout): `mzm/dither_controller.py`
- Physical + measurement model (dither lock-in, RF `J0(beta_rf)` scaling): `mzm/model.py`
- Evaluation helpers (angle sweep, RF/optical robustness): `mzm/eval.py`
- Project technical guide: `docs/mzm_dither_controller_guide.md`

## Current feature modes

`mzm/dither_controller.py` supports `feature_mode`:

- `dc_norm_hist`: legacy baseline (`[h1,h2,dh1,dh2,dv_prev,sinθ*,cosθ*]`)
- `shape_norm_bessel`: experimental; can increase ambiguity/variance in closed loop
- `theta_est_hist`: current recommended; inverts `(h1,h2)` to `theta_hat` and uses `[sin(theta_hat),cos(theta_hat),d_sin,d_cos,...]`

The notebook sets `FEATURE_MODE` in the dataset cell and passes it to `generate_dataset_dbm_hist(...)`.

## Robustness notes

- RF modulation is modeled analytically as `J0(beta_rf)` scaling on the interference term; RF power changes act as a nuisance scaling.
- `theta_est_hist` treats the RF scaling as a nuisance parameter and uses a bias-based prior (`theta_prior=pi*V/Vpi`) to resolve the `theta` vs `pi-theta` ambiguity inherent to magnitude-only observations.
- Model checkpoints store `feature_version`; do not compare/update “best model” across mismatched feature versions.

## Repro workflow

1) Run dataset cell (writes `artifacts/dither_dataset_dbm_hist.npz`)
2) Run training cell (writes `artifacts/dither_policy_dbm_hist.pt`)
3) Run evaluation cells (angle sweep + RF/optical robustness)

If results look inconsistent, confirm the printed `feature_mode` and `feature_version` match the intended configuration.
