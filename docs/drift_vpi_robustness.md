# 漂移鲁棒性：偏压漂移与 $V_\pi$ 漂移的实现方法

本文档解释两类常见慢漂移下，本仓库控制闭环如何保持鲁棒，并把“实现方法”明确对应到代码路径：

- **偏压漂移（bias drift）**：加性电压扰动，等价于传输曲线在电压轴上整体平移。
- **$V_\pi$ 漂移（Vpi drift）**：相位映射系数漂移，等价于“同一个电压对应的相位角”随时间变化。

相关代码入口：

- 漂移仿真：`mzm/drift_simulation.py`
- 导频测量：`mzm/model.py` 的 `_measure_pd_dither_1f2f_batch_torch()` / `measure_pd_dither_normalized_batch_torch()`
- 幅度反演（需要先验选分支）：`mzm/dither_controller.py` 的 `_estimate_theta_from_harmonics_np()`

---

## 1. 两种漂移分别“破坏”了什么

### 1.1 偏压漂移：控制器输出 $V_k$，真实作用电压变成 $V_k + V_{\mathrm{drift}}(k)$

把导频与漂移写进“真实作用电压”：

$$
V_{\mathrm{act}}(t) = V_k + V_{\mathrm{drift}}(k) + V_d\sin(\omega_d t).
$$

真实相位角（物理链路）由它决定：

$$
\theta_{\mathrm{act}}(t) = \frac{\pi}{V_\pi}V_{\mathrm{act}}(t).
$$

关键点：**漂移改变的是物理作用点**，因此 PD 测到的导频谐波对应的是 $\theta_{\mathrm{act}}$，而不是控制器“以为的” $\theta(V_k)$。

### 1.2 $V_\pi$ 漂移：控制器输出 $V_k$ 不变，但 $\theta=\pi V/V_\pi$ 的斜率变了

设真实 $V_\pi$ 随时间变为 $V_{\pi,\mathrm{true}}(k)$，则

$$
\theta_{\mathrm{act}} = \frac{\pi V_k}{V_{\pi,\mathrm{true}}(k)}.
$$

更麻烦的是：导频深度也依赖 $V_\pi$：

$$
\beta_d = \frac{\pi V_d}{V_\pi}.
$$

所以 $V_\pi$ 漂移同时会造成：

- **相位映射错误**（$V\rightarrow\theta$ 变了）；
- **测量模型系数错误**（$J_1(\beta_d),J_2(\beta_d)$ 变了，影响反演/特征）。

---

## 2. 偏压漂移下的鲁棒性：用“带符号 lock-in”直接估计 $\hat\theta$（不依赖先验）

### 2.1 为什么“只有幅度”的反演在大漂移时会变脆

如果只用无符号幅度 $(h_1,h_2)$ 做反演，会遇到“分支二义性”：

$$
\theta \in \{\theta_{\mathrm{abs}},\ \pi-\theta_{\mathrm{abs}}\}.
$$

在无漂移/小漂移时，用先验 $\theta_{\mathrm{prior}}=\pi V_k/V_\pi$ 选分支一般没问题；但当偏压漂移变大，$\theta_{\mathrm{prior}}$ 与真实 $\theta_{\mathrm{act}}$ 会偏离很多，先验可能把分支选错并“自洽锁死”，表现为闭环偶发坏图。

### 2.2 解决思路：恢复符号信息，让 $\hat\theta$ 不再需要先验选分支

`mzm/drift_simulation.py` 的偏压漂移仿真不再用幅度反演，而是从 lock-in 的 **带符号分量** 直接构造 $\hat\theta$。

在小信号模型下（忽略公共缩放因子），1f 与 2f 的带符号分量可写成

$$
h_{1,I} \propto -\sin\theta,\qquad
h_{2,Q} \propto \cos\theta.
$$

两者共享同一个公共缩放（包含输入光功率、RF 缩放、器件系数等），做 DC 归一化后仍然可写成“公共缩放 $\times$ 三角函数”的形式，因此可以用比例消掉缩放并得到角度：

$$
p := \frac{-h_{1,I}/I_{\mathrm{dc}}}{2J_1(\beta_d)},\qquad
q := \frac{h_{2,Q}/I_{\mathrm{dc}}}{2J_2(\beta_d)},\qquad
\hat\theta = \mathrm{atan2}(p,q).
$$

再把 $\hat\theta$ 包到 $[0,\pi]$：

$$
\hat\theta \leftarrow
\begin{cases}
\hat\theta + \pi, & \hat\theta < 0,\\
\hat\theta, & \text{otherwise.}
\end{cases}
$$

这条链路的核心性质是：**$\hat\theta$ 来自测量本身，不需要依赖 $V_k$ 的先验**。因此即使偏压漂移让 $V_k$ 与真实作用点偏离，闭环仍然能围绕“估计到的真实相位”做反馈。

### 2.3 代码对应关系

- 在物理测量里注入偏压漂移：`mzm/model.py` 的 `V_total_bias = Vb + V_drift`（被 `_measure_pd_dither_1f2f_batch_torch()` 使用）
- 偏压漂移仿真主循环：`mzm/drift_simulation.py` 的 `simulate_control_loop_with_drift(...)`
  - 读取 `out["h1_I"]`, `out["h2_Q"]`
  - 做 DC 归一化得到 `h1_I_norm`, `h2_Q_norm`
  - 角度估计：`_estimate_theta_signed_from_lockin_np(...)`

---

## 3. $V_\pi$ 漂移下的鲁棒性：分离 “真实 $V_\pi$” 与 “控制器使用的 $V_\pi$”

`mzm/drift_simulation.py` 有意把 $V_\pi$ 拆成两条时间序列：

- $V_{\pi,\mathrm{true}}(k)$：**物理测量**使用（决定真实 $\theta$ 与真实 $\beta_d$）
- $V_{\pi,\mathrm{used}}(k)$：**控制器/估计器**使用（决定 $\theta_{\mathrm{prior}}$ 与反演时的 $J_1,J_2$）

这允许你测试两种场景：

1) **已知 $V_\pi$**：令 $V_{\pi,\mathrm{used}}=V_{\pi,\mathrm{true}}$（`controller_knows_vpi=True`），等价于系统能实时校准/读取 $V_\pi$；
2) **标定失配**：令 $V_{\pi,\mathrm{used}}=V_{\pi,\mathrm{nom}}$ 固定（默认），评估失配对闭环的影响。

对应函数：`simulate_control_loop_with_vpi_drift(...)`。

---

## 4. $V_\pi$ 漂移自适应（在线估计）的实现方法：网格搜索 + 置信度门限 + 稳健更新

仅靠固定标称 $V_{\pi,\mathrm{nom}}$，当 $V_{\pi,\mathrm{true}}$ 漂移较大时会出现两类问题：

- 角度反演用错了 $\beta_d=\pi V_d/V_{\pi,\mathrm{used}}$，导致 $\hat\theta$ 系统性偏差；
- 策略网络输出的 $\Delta V$ 其实对应“标称 $V_\pi$ 下的相位-电压增益”，当真实 $V_\pi$ 改变时，闭环等效增益失配，容易慢/振荡。

### 4.1 在线估计的目标：找到能让“先验-反演”自洽的 $V_{\pi,\mathrm{est}}$

自适应版本在每一步维护一个估计 $V_{\pi,\mathrm{est}}$，并用它来同时计算：

$$
\theta_{\mathrm{prior}}(V_{\pi,\mathrm{est}}) = \frac{\pi V_k}{V_{\pi,\mathrm{est}}}
$$

以及用同一个 $V_{\pi,\mathrm{est}}$ 做角度反演，得到 $\hat\theta(V_{\pi,\mathrm{est}})$。

当 $V_{\pi,\mathrm{est}}$ 接近真实值时，这两者应该“更一致”。因此定义代价函数：

$$
\mathrm{cost}(V_\pi) = \left|\mathrm{wrap}\left(\hat\theta(V_\pi)-\theta_{\mathrm{prior}}(V_\pi)\right)\right|.
$$

在一组候选网格上选最小者作为观测到的 $V_\pi$：

$$
V_{\pi,\mathrm{meas}} = \arg\min_{V_\pi\in\mathcal{G}} \mathrm{cost}(V_\pi).
$$

这一步对应 `simulate_adaptive_control_loop(...)` 中的 `vpi_grid` 与 `best_cost` 搜索逻辑。

### 4.2 为什么需要置信度门限：接近四分之一波点时 $V_\pi$ 变“不可辨”

在 $\theta\approx 90^\circ$ 附近，2f 信号通常很弱（$\cos\theta\approx 0$），导致不同的 $V_\pi$ 候选可能给出相近的代价（代价曲线变平），此时强行更新会造成估计抖动并放大到闭环。

因此实现里用两类门限做 “update gating”：

- **幅度门限**：$|h_2|$ 足够大（并使用 EMA 形成动态阈值 `h2_gate`）
- **间隔门限**：最优与次优的代价差 `margin = second_cost - best_cost` 足够大

只有满足条件才允许更新 `Vpi_est`（字段 `vpi_update_ok`）。

### 4.3 为什么还要稳健更新：避免单步跳变把环路打爆

即使允许更新，也不直接把 `best_vpi` 赋值给 `Vpi_est`，而是三层“稳健化”：

1) **滑窗中位数**：对最近若干次 `best_vpi` 取中位数；
2) **相对步长限幅**：限制单次相对变化不超过 `update_max_rel`；
3) **EMA 平滑**：用 `est_alpha` 做指数平滑：

$$
V_{\pi,\mathrm{est}} \leftarrow V_{\pi,\mathrm{est}}\left(1+\alpha\left(\frac{V_{\pi,\mathrm{meas}}}{V_{\pi,\mathrm{est}}}-1\right)\right).
$$

这样做的目的不是让估计“更快”，而是让闭环整体更稳、更不容易出现周期性爆炸。

### 4.4 对策略输出做增益校正：$\Delta V$ 需要按 $V_\pi$ 比例缩放

网络是在标称 $V_{\pi,\mathrm{nom}}$ 的电压尺度上学到的控制增益。若真实 $V_\pi$ 变大，同样的相位修正需要更大的电压变化；反之亦然。因此实现里对网络输出做比例缩放：

$$
\Delta V = \Delta V_{\mathrm{nn}}\cdot \mathrm{clip}\left(\frac{V_{\pi,\mathrm{est}}}{V_{\pi,\mathrm{nom}}},\,0.5,\,3.0\right).
$$

对应 `simulate_adaptive_control_loop(...)` 中的 `gain_factor` 与 `delta_v = delta_v_nn * gain_factor`。

---

## 5. 实用参数含义（对应 `simulate_adaptive_control_loop`）

- `est_alpha`：$V_\pi$ 估计更新速度（越大越快，但越容易抖）
- `conf_margin_rad`：最优与次优代价的最小间隔（越大越保守）
- `conf_h2_frac/conf_h2_min_abs`：2f 幅度门限（用于判定“信息量是否够”）
- `update_max_rel`：单次相对更新限幅（防止跳变）
- `buffer_len`：中位数缓冲长度（抗离群）

---

## 6. 建议的阅读顺序

1) 先读 `docs/theta_inversion_theory.md` 理解“由导频反演 $\hat\theta$”与 RF 缩放消除；
2) 再看 `mzm/drift_simulation.py`：
   - 偏压漂移：为什么改成用带符号 lock-in；
   - $V_\pi$ 漂移：为什么需要在线估计与增益校正。

