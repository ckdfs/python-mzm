"""python-mzm reusable modules.

This package contains the reusable simulation, plotting, and controller code.
Notebook(s) should import from here.
"""

from .model import (
    simulate_mzm,
    measure_pd_dither_1f2f,
    bias_to_theta_rad,
    theta_to_bias_V,
    wrap_to_pi,
)

from .dither_controller import (
    DeviceParams,
    DitherParams,
    generate_dataset_dbm_hist,
    train_policy,
    save_model,
    load_model,
    rollout_dbm_hist,
    rollout_dbm_hist_batch,
)

from .eval import (
    evaluate_model,
    evaluate_current_and_best,
    evaluate_rf_power_robustness,
    evaluate_rf_robustness_multi_target,
    evaluate_optical_power_robustness,
)
