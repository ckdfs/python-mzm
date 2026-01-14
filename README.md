# python-mzm

- MZM 仿真物理模型
- 基于深度学习的 MZM 导频法闭环控制（数据生成 / 训练 / 闭环验证）。

## 环境
- 建议在 conda 环境中运行
- 依赖见 `requirements.txt`

## 训练入口

### dither 闭环控制（仅用 PD 导频 1f/2f 功率）

推荐用 Notebook：
- `mzm_dither_controller.ipynb`（数据集生成 / 训练 / 推理 分开，可复用保存的产物）

数据集生成可选 GPU 加速（需要 CUDA 可用的 PyTorch）：
- 在 `generate_dataset_dbm_hist(..., accel='auto')` 时会自动使用 GPU（否则回退 CPU）
- 可通过 `torch_batch` 调整 batch 大小以避免显存不足

更完整的理论推导与代码对应说明：
- `docs/mzm_dither_controller_guide.md`
