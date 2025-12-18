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

更完整的理论推导与代码对应说明：
- `docs/mzm_dither_controller_guide.md`

也可以直接跑精简脚本（默认就是接近真实的控制方式，无需额外参数）：
- `python scripts/train_mzm_dither_controller.py`
