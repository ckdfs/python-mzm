---
marp: true
paginate: true
size: 16:9
math: katex
---

<!-- _class: lead -->

# 基于深度学习的 MZM 导频法闭环控制

**详细版 Slides（Marp）**

- 工程文档来源：`docs/mzm_dither_controller_guide.md`
- 导出：Marp for VS Code → Export → PowerPoint (`.pptx`)

---

## 目录

1. 项目目标与约束
2. MZM 物理模型
3. 导频与 lock-in 测量（1f/2f → dBm）
4. 闭环控制问题表述
5. 幅度观测的可观测性问题与解决
6. 策略网络与监督信号（Teacher）
7. 数据集生成（dbm_hist）
8. 训练与工件保存
9. 闭环回放（rollout）与 trace
10. 快速上手、参数核对、局限性、代码对应

---

## 1. 项目目标与约束（问题定义）

目标：控制 Mach–Zehnder Modulator（MZM）偏置工作点，使角度 $\theta$ 锁定到指定目标 $\theta^*$。

关键约束（与实现保持一致）：
- 不直接观测 DC 光功率
- 不依赖高速 RF 可观测
- 仅使用 PD 输出中由低频导频产生的 **1f/2f 分量功率（dBm）**
- 策略输出为偏置电压增量 $\Delta V$，每步更新后执行限幅

---

## 1. 可观测量与控制量

- 状态（需要控制）：偏置电压 $V$（或工作角度 $\theta$）
- 可观测量：$P_1^{\mathrm{dBm}}, P_2^{\mathrm{dBm}}$（导频 1f/2f）
- 控制量：$\Delta V$（每一拍输出一次）
- 约束：$V \in [0, V_\pi]$（等价 $\theta \in [0,\pi]$）

更新律：
$$
V_{k+1}=\mathrm{clip}(V_k+\Delta V_k,\ 0,\ V_\pi)
$$

---

## 1. 代码入口（你会用到哪些文件）

- 物理模型与导频测量：`mzm/model.py`
- 数据集/训练/回放：`mzm/dither_controller.py`
- Notebook 流水线：`mzm_dither_controller.ipynb`

加速说明（实现现状）：
- 导频测量 / lock-in 的核心已统一为 torch 批量实现
- `accel='auto'`：CUDA 可用走 GPU，否则 torch-on-CPU
- `torch_batch`：控制每次送入 device 的样本数（显存不够可调小）

---

## 2. MZM 物理模型：场叠加（与实现一致）

输出光场（考虑插入损耗 IL 与有限消光比 ER 导致的幅度不平衡）：
$$
E_{\text{out}} = E_{\text{in}}\,\sqrt{\eta}\,\frac{1}{2}\left(e^{j\phi_1} + \gamma\,e^{j\phi_2}\right)
$$

- $\eta = 10^{-\mathrm{IL}/10}$：功率衰减因子
- $\gamma$：由消光比得到的幅度不平衡系数

若 $\mathrm{ER}_{\text{field}}=10^{\mathrm{ER}_{\text{dB}}/20}$：
$$
\gamma = \frac{\mathrm{ER}_{\text{field}}-1}{\mathrm{ER}_{\text{field}}+1}
$$

---

## 2. MZM 物理模型：电压到相位映射

归一化映射：
$$
\phi = \frac{\pi}{V_\pi}V
$$

DC 偏置在两臂对称施加：
$$
V_1=\frac{V_{\text{bias}}}{2},\quad V_2=-\frac{V_{\text{bias}}}{2}
$$

相位：
$$
\phi_1=\frac{\pi}{V_{\pi,DC}}\frac{V_{\text{bias}}}{2},\quad
\phi_2=\frac{\pi}{V_{\pi,DC}}\left(-\frac{V_{\text{bias}}}{2}\right)
$$

强度（光功率）：
$$
P_{\text{out}}=|E_{\text{out}}|^2
$$

---

## 2. 工作点定义（角度 $\theta$）

项目用角度统一描述偏置点：
$$
\theta \triangleq \frac{\pi}{V_{\pi,DC}}V_{\text{bias}}
$$

含义：
- $\theta=0$（0°）：强度极大点
- $\theta=\pi/2$（90°）：四分之一点
- $\theta=\pi$（180°）：强度极小点

控制范围：$\theta\in[0,\pi]$ ⇔ $V_{\text{bias}}\in[0,V_{\pi,DC}]$

---

## 3. 导频：注入方式

偏置上叠加低频正弦导频：
$$
V(t)=V_{\text{bias}} + A\sin(2\pi f_dt)
$$

仿真测量链路：
- 由 DC 传输得到 $P(t)=P(V(t))$
- PD 响应得到电流
$$
I(t)=\mathcal{R}\,P(t)
$$

其中 $\mathcal{R}$ 为 responsivity（A/W）。

---

## 3. 导频：小信号展开（1f/2f 来源）

设 $\Delta V(t)=A\sin(\omega t)$，$\omega=2\pi f_d$。

泰勒展开：
$$
I(V_{\text{bias}}+\Delta V) \approx I_0 + I'\Delta V + \frac{1}{2}I''\Delta V^2 + \cdots
$$

代入 $\sin^2(\omega t)=\tfrac{1}{2}(1-\cos(2\omega t))$：
- 1f：由 $I' A\sin(\omega t)$ 产生（与局部斜率相关）
- 2f：由二阶项产生（与局部曲率相关，同时包含 DC）

---

## 3. lock-in 检测：I/Q 与幅度

采样 $x[n]$（交流分量）与参考正交信号：
$$
\sin(n\Omega),\ \cos(n\Omega),\quad \Omega=2\pi\frac{f_d}{F_s}
$$

第 $k$ 阶谐波（$k=1,2$）：
$$
I_k = \frac{2}{N}\sum_{n=0}^{N-1} x[n]\sin(k\Omega n),\quad
Q_k = \frac{2}{N}\sum_{n=0}^{N-1} x[n]\cos(k\Omega n)
$$

幅度：
$$
A_k = \sqrt{I_k^2+Q_k^2}
$$

实现里训练/生成默认用 torch 批量核心；NumPy 版用于对照调试。

---

## 3. 幅度 → 功率 → dBm

若电流纯正弦 $i(t)=A\sin(\omega t)$，负载为 $R$：
$$
P_{\text{avg}} = \frac{1}{T}\int_0^T i^2(t)R\,dt = \frac{A^2}{2}R
$$

对 1f/2f：
$$
P_1=\tfrac{1}{2}A_1^2 R,\quad P_2=\tfrac{1}{2}A_2^2 R
$$

dBm 转换：
$$
P_{\mathrm{dBm}} = 10\log_{10}(P_{\mathrm{W}}\cdot 1000)
$$

最终观测：$P_1^{\mathrm{dBm}}, P_2^{\mathrm{dBm}}$。

---

## 4. 闭环控制：状态、观测、动作

- 状态：$V_k$（或 $\theta_k$）
- 观测：$\big(P_{1,k}^{\mathrm{dBm}}, P_{2,k}^{\mathrm{dBm}}\big)$
- 动作：$\Delta V_k$

更新：
$$
V_{k+1}=\mathrm{clip}(V_k+\Delta V_k,0,V_\pi)
$$

关键：策略要在**“只能看到幅度 dBm”**的条件下，决定 $\Delta V_k$ 的方向和大小。

---

## 4. 目标角度编码（避免不连续）

目标角度限制：$\theta^*\in[0,\pi]$。

策略输入对目标使用正余弦编码：
$$
t_1=\sin(\theta^*),\quad t_2=\cos(\theta^*)
$$

直觉：
- 避免直接回归角度导致的周期边界不连续
- 提供连续、可学习的目标表示

---

## 4. 相位误差定义与 wrap

误差在 $[-\pi,\pi)$ 内连续：
$$
 e_\theta = \mathrm{wrap}(\theta^*-\theta)
$$

wrap 定义：
$$
\mathrm{wrap}(x)=(x+\pi)\bmod 2\pi-\pi
$$

即使目标范围为 $0\sim\pi$，迭代更新中误差仍可能跨越边界，因此 wrap 是必要的。

---

## 5. 仅保留幅度的可观测性问题

理想小信号下：
$$
I_{1f}\propto I'(V)\,A
$$

但幅度观测：
$$
A_1=\sqrt{I_1^2+Q_1^2}\ge 0
$$

结果：
- 1f 的符号信息丢失
- 仅靠当前时刻 $(P_1,P_2)$ 难以判断“该往哪个方向调偏置”

---

## 5. 解决策略：引入差分 + 上一拍动作

恢复方向信息的核心思想：
- 已知上一拍动作 $\Delta V_{k-1}$（方向已知）
- 观测差分携带局部梯度方向的统计信息

定义（dBm 差分）：
$$
\Delta P_{1,k}=P_{1,k}-P_{1,k-1},\quad
\Delta P_{2,k}=P_{2,k}-P_{2,k-1}
$$

将 $\Delta P$ 与 $\Delta V_{k-1}$ 结合，作为策略输入的一部分。

---

## 5. 最终策略输入向量（7 维）

与实现一致：
$$
\mathbf{x}_k=[P_{1,k}^{\text{dBm}},\ P_{2,k}^{\text{dBm}},\ \Delta P_{1,k}^{\text{dBm}},\ \Delta P_{2,k}^{\text{dBm}},\ \Delta V_{k-1},\ \sin\theta^*,\ \cos\theta^*]
$$

- 输入维度：7
- 直观分组：
  - 当前观测：$P_1,P_2$
  - 变化趋势：$\Delta P_1,\Delta P_2$
  - 已知历史动作：$\Delta V_{k-1}$
  - 目标：$\sin\theta^*,\cos\theta^*$

---

## 6. 策略网络（Policy）：MLP 回归 $\Delta V$

策略形式：
$$
\Delta V = \pi_\psi(\hat{\mathbf{x}})
$$

默认网络（实现中的 `DeltaVPolicyNet`）：
- in_dim = 7
- hidden = 64
- depth = 3（每层 Linear + ReLU）
- out_dim = 1（无激活，直接回归）

结构（示意）：

```
7  →  64  →  64  →  64  →  1
      ReLU    ReLU    ReLU
```

---

## 6. 输入归一化（与模型一起保存）

标准化：
$$
\hat{\mathbf{x}} = \frac{\mathbf{x}-\mu}{\sigma}
$$

- $(\mu,\sigma)$ 由训练集统计
- 推理与回放必须使用同一组 $(\mu,\sigma)$

工件中会保存：
- `model_state`
- `mu`, `sigma`
- `device_params`, `dither_params`

---

## 6. Teacher：相位误差比例控制（生成监督标签）

由偏置得到角度：
$$
\theta_k = \frac{\pi}{V_\pi}V_k
$$

误差：
$$
 e_k=\mathrm{wrap}(\theta^*-\theta_k)
$$

由 $\theta=\frac{\pi}{V_\pi}V$ 得到理想电压修正：
$$
\Delta V_{\text{ideal}} = \frac{V_\pi}{\pi}e_k
$$

标签（限幅）：
$$
 y_k = \mathrm{clip}(g\,\Delta V_{\text{ideal}},\ -\Delta V_{\max},\ \Delta V_{\max})
$$

默认：`teacher_gain = 0.5`, `max_step_V = 0.2`。

---

## 7. 数据集生成：为什么要模拟“真实可用历史”

目标：让 $\Delta P$ 与 $\Delta V_{k-1}$ 的组合在物理上自洽。

数据集函数：`generate_dataset_dbm_hist()`（dbm_hist 特征）。

每个样本都包含：
- 当前时刻观测（在 $V_k$ 下测量）
- 上一时刻观测（在 $V_{k-1}$ 下测量）
- 上一拍动作 $\Delta V_{k-1}$

这样构造出来的差分对应“系统经历上一拍动作后观测发生的变化”。

---

## 7. 数据集生成：逐样本构造流程（详细）

对每个样本：
1) 采样当前偏置 $V_k\sim\mathcal{U}(0,V_\pi)$
2) 采样目标角度 $\theta^*\sim\mathcal{U}(0,\pi)$
3) 采样上一拍动作 $\Delta V_{k-1}\sim\mathcal{U}(-\Delta V_{\max},\Delta V_{\max})$
4) 反推上一拍偏置：
$$
V_{k-1}=\mathrm{clip}(V_k-\Delta V_{k-1},0,V_\pi)
$$
5) 分别在 $V_{k-1}$ 与 $V_k$ 下测量 $P_1^{\mathrm{dBm}},P_2^{\mathrm{dBm}}$
6) 构造差分 $dp1, dp2$
7) 目标角度编码 $(\sin\theta^*,\cos\theta^*)$，拼接 7 维输入

---

## 7. 差分特征的定义（与实现一致）

对导频测量得到：
$$
P_{1,k}^{\mathrm{dBm}},\ P_{2,k}^{\mathrm{dBm}}
$$

差分：
$$
dp1=P_{1,k}^{\mathrm{dBm}}-P_{1,k-1}^{\mathrm{dBm}},\quad
\,dp2=P_{2,k}^{\mathrm{dBm}}-P_{2,k-1}^{\mathrm{dBm}}
$$

于是输入：
$$
\mathbf{x}_k=[P_{1,k}^{\text{dBm}},\ P_{2,k}^{\text{dBm}},\ dp1,\ dp2,\ \Delta V_{k-1},\ \sin\theta^*,\ \cos\theta^*]
$$

---

## 7. 数据集工件（NPZ）保存字段

工件：`artifacts/dither_dataset_dbm_hist.npz`

包含字段：
- `Xn`：归一化特征，形状 $(N,7)$
- `y`：标签（teacher 输出 $\Delta V$），形状 $(N,1)$
- `mu, sigma`：归一化统计，形状 $(7,)$
- `device_params`：器件参数（`Vpi/ER/IL/Pin/Responsivity/R_load`）
- `dither_params`：导频参数（`V_dither_amp/f_dither/Fs/n_periods`）
- `teacher_gain, max_step_V`

备注：实现默认不逐样本保存隐藏变量（如 $V_k, V_{k-1}, \theta^*$）。

---

## 8. 训练：目标函数与数据流

训练目标（MSE）：
$$
\min_\psi\ \mathbb{E}\left[\left(\pi_\psi(\hat{\mathbf{x}})-y\right)^2\right]
$$

实现：`train_policy()`
- 直接使用 NPZ 中 `Xn`（已归一化）与 `y`
- DataLoader 按 batch 打乱训练

默认超参数：
- `epochs=2000`
- `batch=256`
- `lr=1e-3`（Adam）
- 网络：`hidden=64`, `depth=3`

---

## 8. 模型保存（可复用工件）

模型工件：`artifacts/dither_policy_dbm_hist.pt`

保存内容：
- `model_state`：网络参数
- `mu, sigma`：归一化统计
- `device_params, dither_params`：仿真/测量参数

推理要点：
- 必须用同一组 `mu, sigma` 标准化输入，否则输出尺度会失真

---

## 9. 闭环回放（rollout）：用于验证收敛性

函数：`rollout_dbm_hist()`

输入：
- 目标角度（度）`theta_target_deg`
- 初始偏置 `V_init`
- 迭代 `steps`

每步：
1) measure：在 $V_k$ 下测量 $P_{1,k}^{\mathrm{dBm}},P_{2,k}^{\mathrm{dBm}}$
2) feature：构造 $dp1,dp2$ 并拼接上一拍 `prev_dv`
3) normalize：用 `mu,sigma` 标准化
4) policy：得到 $\Delta V_k$
5) update：限幅更新 $V_{k+1}$
6) record：记录 wrapped 误差

---

## 9. 回放的数学形式（对齐实现）

1) 测量：
$$
(P_{1,k}^{\mathrm{dBm}},P_{2,k}^{\mathrm{dBm}}) \leftarrow \mathrm{measure}(V_k)
$$

2) 差分：
$$
dp1_k=P_{1,k}^{\mathrm{dBm}}-P_{1,k-1}^{\mathrm{dBm}},\quad
\,dp2_k=P_{2,k}^{\mathrm{dBm}}-P_{2,k-1}^{\mathrm{dBm}}
$$

3) 标准化输入：
$$
\hat{\mathbf{x}}_k = \frac{[P_{1,k}^{\mathrm{dBm}},P_{2,k}^{\mathrm{dBm}},dp1_k,dp2_k,\Delta V_{k-1},\sin\theta^*,\cos\theta^*]-\mu}{\sigma}
$$

4) 策略 + 更新：
$$
\Delta V_k = \pi_\psi(\hat{\mathbf{x}}_k),\quad
V_{k+1}=\mathrm{clip}(V_k+\Delta V_k,0,V_\pi)
$$

---

## 9. 回放 trace 字段（便于逐轮展示/调试）

除 `V/err_deg` 外，还返回：
- `dv`：每轮输出 $\Delta V_k$
- `p1_dBm, p2_dBm`：每轮观测
- `dp1_dBm, dp2_dBm`：每轮差分
- `theta_deg`：每轮角度轨迹

Notebook `mzm_dither_controller.ipynb` 默认会打印这些 trace，便于审阅每轮推理过程。

---

## 10. 快速上手：Notebook

推荐入口：`mzm_dither_controller.ipynb`

按顺序运行（逻辑分段通常为）：
- 数据集生成 → 写入 `artifacts/`
- 训练 → 保存模型工件
- 推理/回放 → 输出误差与 trace

优点：
- 过程可视化、可逐步调参
- 工件可复用

---

## 10. 快速上手：脚本一键流程

在仓库根目录：

```bash
python scripts/train_mzm_dither_controller.py
```

脚本逻辑：
1) 数据集不存在则生成
2) 训练并保存模型
3) 对多个目标角执行一次 rollout 并打印最终误差

（如果你的仓库里暂时没有 `scripts/`，可以只用 Notebook 流水线。）

---

## 11. 参数与单位核对（与实现一致）

- `Vpi_DC`：V（默认 5.0）
- `V_dither_amp`：V（默认 0.05）
- `f_dither`：Hz（默认 10 kHz）
- `Fs`：Hz（默认 2 MHz）
- `n_periods`：采样周期数（默认 120）
- `R_load`：Ω（默认 50）
- `Responsivity`：A/W（默认 0.786）
- `p1_dBm` / `p2_dBm`：dBm（按 $P_{avg}=(A^2/2)R$ 计算）

---

## 12. 局限性（工程上需要注意）

1) **$V_\pi$ 漂移敏感**
- $\theta \leftrightarrow V$ 映射依赖 $V_\pi$
- 硬件 $V_\pi$ 漂移会导致控制退化

2) **限幅引入非线性**
- 靠近边界时，$\Delta V$ 会被 clip 截断
- 收敛行为会与“内点”不同，评估时要区分

---

## 13. 与源代码的对应关系（查阅导航）

`mzm/model.py`
- `mzm_dc_power_mW()`：DC 传输曲线
- `lockin_harmonics()`：NumPy 参考版 lock-in
- `_measure_pd_dither_1f2f_batch_torch()`：torch 批量导频测量核心（CPU/GPU）
- `measure_pd_dither_1f2f()`：单点导频测量（返回字段不变）
- `measure_pd_dither_1f2f_dbm_batch_torch()`：批量输出 `p1_dBm/p2_dBm`
- `bias_to_theta_rad()`, `theta_to_bias_V()`, `wrap_to_pi()`

`mzm/dither_controller.py`
- `generate_dataset_dbm_hist()` / `train_policy()` / `save_model()` / `load_model()`
- `rollout_dbm_hist()`

---

<!-- _class: lead -->

# Q & A

- 需要我把这份 slides 再进一步「按你的汇报时长」压缩/扩展页数吗？
- 或者想加入实测/仿真曲线截图（需要你提供图或我从 Notebook 里生成）？
