# Theta 反演：如何在 RF 功率变化下保持鲁棒（`theta_est_hist`）

这份文档只回答一件事：为什么在 RF 功率（等价于 $J_0(\beta_{rf})$）变化时，仍然能从观测到的 $(h_{1,\mathrm{norm}},h_{2,\mathrm{norm}})$ 稳定反推出工作角 $\hat\theta$；以及代码里所谓“RF 归一化/消除 RF 缩放”到底是怎么做的。

对应实现：

- DC 归一化测量：`mzm/model.py` 的 `measure_pd_dither_normalized_batch_torch()`
- 角度反演（消除 RF 缩放）：`mzm/dither_controller.py` 的 `_estimate_theta_from_harmonics_np()` / `_estimate_theta_from_harmonics_torch()`

---

## 1. 先把问题说清楚：观测量、两类“缩放”、我们要消掉哪一个

系统能直接得到的量是导频（dither）产生的 1f/2f 谐波“幅度”（无符号）以及 DC 电流：

- $A_1,A_2$：lock-in 得到的 1f/2f 幅度（实现里对应 `h1_A`,`h2_A`）
- $I_{\mathrm{dc}}$：PD 电流均值（实现里对应 `pd_dc`）

第一步做的 DC 归一化是：

$$
h_1 := \frac{A_1}{I_{\mathrm{dc}}},\qquad
h_2 := \frac{A_2}{I_{\mathrm{dc}}}.
$$

它的作用是消掉“光功率整体缩放”（例如 $P_{in}$ 变化带来的整体乘法系数 $K$）。但它并不能自动消掉 RF 带来的缩放，原因是：RF 不只影响分子 $A_1,A_2$，它也会改变分母 $I_{\mathrm{dc}}$ 的结构（见第 3 节）。

我们要消掉的 RF 影响是：高速 RF 调制导致 MZM 干涉项被一个因子 $J_0(\beta_{rf})$ 缩放，等价为一个未知的“nuisance 参数” $b$ 在测量方程里变化。

---

## 2. 物理模型：RF 只进入一个 nuisance 参数 $b$

用项目里一致的强度模型写 PD 电流：

$$
I(t) = K\left[a + b\cos\left(\theta + \beta_d\sin(\omega_d t)\right)\right].
$$

其中：

- $\theta=\dfrac{\pi V_{\mathrm{bias}}}{V_\pi}\in[0,\pi]$：工作角
- $\beta_d=\dfrac{\pi V_{\mathrm{dither}}}{V_\pi}$：导频深度
- $a=1+\gamma^2$，$\gamma$ 由消光比（ER）决定（实现里由 `ER_dB` 计算）
- **RF 缩放被压缩到一个参数里**：

$$
b = 2\gamma\,J_0(\beta_{rf}),\quad \beta_{rf}=\frac{\pi V_{rf}}{V_\pi}
$$

所以“RF 功率变化”在这个模型下等价为 $b$ 变化（而 $\theta,\beta_d,a$ 不变）。

---

## 3. 从模型到测量方程：为什么 DC 归一化后仍然残留 $b$

对 $\cos(\theta+\beta_d\sin(\omega_d t))$ 做 Jacobi–Anger 展开，可以把 DC、1f、2f 主项写成（省略更高次谐波）：

$$
\cos(\theta+\beta_d\sin x)
\approx
J_0(\beta_d)\cos\theta
-2J_1(\beta_d)\sin\theta\sin x
+2J_2(\beta_d)\cos\theta\cos 2x
$$

因此（lock-in 取幅度，丢符号），1f/2f 幅度满足：

$$
A_1 = 2KbJ_1(\beta_d)\,|\sin\theta|,\quad
A_2 = 2KbJ_2(\beta_d)\,|\cos\theta|
$$

而 DC 分量是：

$$
I_{\mathrm{dc}} = K\left[a+bJ_0(\beta_d)\cos\theta\right]
$$

把它们做 DC 归一化，得到两条核心方程：

$$
h_1 = \frac{2bJ_1(\beta_d)\,|\sin\theta|}{a+bJ_0(\beta_d)\cos\theta},\qquad
h_2 = \frac{2bJ_2(\beta_d)\,|\cos\theta|}{a+bJ_0(\beta_d)\cos\theta}.
$$

读法很关键：

- $K$ 被消掉了 $\Rightarrow$ 对输入光功率整体缩放鲁棒；
- 但 $b$ 同时在分子、分母里出现 $\Rightarrow$ **仅 DC 归一化并不能让 $h_1,h_2$ 对 RF 缩放不敏感**。

这就是“RF 归一化/消除 RF 缩放”要解决的剩余问题：从 $(h_1,h_2)$ 里把 $b$ 消掉，反推出 $\theta$。

---

## 4. RF 归一化（消除 RF 缩放）的核心：取比值把 $b$ 和分母一起消掉

把上面的两条方程做一次“Bessel 去尺度”（实现里叫 `p`,`q`）：

$$
p := \frac{h_1}{2J_1(\beta_d)},\qquad
q := \frac{h_2}{2J_2(\beta_d)}.
$$

代入可得：

$$
p = \frac{b|\sin\theta|}{D},\qquad
q = \frac{b|\cos\theta|}{D},\qquad
D := a+bJ_0(\beta_d)\cos\theta.
$$

现在做关键一步：取比值

$$
\frac{p}{q}=\left|\tan\theta\right|
$$

注意这里同时消掉了两样东西：

- RF 缩放 $b$（含 $J_0(\beta_{rf})$）
- “共同分母” $D$（它也含 $b$）

所以，$\theta$ 的“折叠角”（只在 $[0,\pi/2]$ 内）可以直接由观测给出：

$$
\theta_{\mathrm{abs}} = \mathrm{atan2}(p,q)\in[0,\pi/2].
$$

这一步就是本项目里真正意义上的“RF 归一化”：你并不需要显式知道 RF 功率，也不需要测 $J_0(\beta_{rf})$，只要用 $(h_1,h_2)$ 的结构关系取比值，就把 RF 缩放从角度估计里消掉了。

---

## 5. 幅度观测的先天歧义：为什么还需要一个先验来选分支

因为 lock-in 只输出幅度 $|\cdot|$，$\cos\theta$ 的符号丢失了，因此真实角度只能是两种之一：

$$
\theta \in \{\theta_{\mathrm{abs}},\ \pi-\theta_{\mathrm{abs}}\}
$$

闭环控制里我们天然知道自己当前输出的偏置电压 $V$，因此可以构造先验角度：

$$
\theta_{\mathrm{prior}}=\frac{\pi V}{V_\pi}
$$

然后按“离先验更近”的规则选分支：

$$
\hat\theta=
\begin{cases}
\theta_{\mathrm{abs}}, & |\theta_{\mathrm{abs}}-\theta_{\mathrm{prior}}|\le|\pi-\theta_{\mathrm{abs}}-\theta_{\mathrm{prior}}|,\\
\pi-\theta_{\mathrm{abs}}, & \text{otherwise.}
\end{cases}
$$

这一步对应 `_estimate_theta_from_harmonics_*()` 里 `theta_prior_rad` 的逻辑；它不是为了“消 RF”，而是为了解决幅度观测导致的不可辨识性，使 $\hat\theta\in[0,\pi]$ 可稳定落在正确分支。

---

## 6. 代码里的实现细节：`p/q` 之外还做了什么（以及为什么）

`mzm/dither_controller.py::_estimate_theta_from_harmonics_np()` 与 torch 版本的核心步骤是一一对应的：

1) 用已知导频幅度和 $V_\pi$ 得到 $\beta_d=\pi V_{\mathrm{dither}}/V_\pi$，并计算 $J_0,J_1,J_2$（实现里用小信号近似 `_bessel_j*_small()`）。

2) 计算

$$
p=\frac{h_1}{2J_1(\beta_d)},\quad q=\frac{h_2}{2J_2(\beta_d)},\quad
\theta_{\mathrm{abs}}=\mathrm{atan2}(p,q)
$$

3) 用先验选择 $\hat\theta\in\{\theta_{\mathrm{abs}},\pi-\theta_{\mathrm{abs}}\}$。

4) （可选诊断）估计 $b$：令

$$
w := \frac{p}{|\sin\hat\theta|}
=\frac{b}{a+bJ_0(\beta_d)\cos\hat\theta}
$$

对两种 $\cos\hat\theta=\pm|\cos\theta_{\mathrm{abs}}|$ 分别解出

$$
b_{\pm}=\frac{wa}{1\mp wJ_0(\beta_d)|\cos\theta_{\mathrm{abs}}|}
$$

并裁剪到 $[0,2\gamma]$（实现里是 `b_max=2*gamma`），用于分支 fallback 或调试输出。

---

## 7. 什么时候“看起来不鲁棒”：你最可能踩到的坑

- **RF=0 反而更难**：$J_0(\beta_{rf})=1$ 时 $b$ 最大，更容易把分母 $D=a+bJ_0(\beta_d)\cos\theta$ 推到很小（接近深消光），噪声/量化误差会被放大。
- **导频参数必须稳定**：上面的消除依赖 $\beta_d$ 已知且不漂；如果实际导频幅度变化但模型仍用旧的 $\beta_d$，$J_1,J_2$ 去尺度会出错，进而影响 $\theta_{\mathrm{abs}}$。
- **只用幅度就一定有分支歧义**：如果你希望完全不依赖先验，必须引入带符号的 lock-in I/Q（或额外激励/观测）来恢复符号信息。
