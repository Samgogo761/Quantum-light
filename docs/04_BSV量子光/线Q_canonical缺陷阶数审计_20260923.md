# 线 Q：canonical 缺陷的耦合阶数与所需噪声结构（2026-09-23）

人审判断：Kubo 期望值响应

\[
\delta\langle I\rangle=\mathcal R^A[\delta A]
\]

被当成精确的 Heisenberg 构成 \(\delta\hat I=\hat\xi+\mathcal R^A[\delta\hat A]\) 再无限迭代，并不自动给出合法量子 I/O。本轮只做诊断：\(\gamma\to\lambda\gamma\)，以及

\[
\Delta C=\mathbb I-\bigl(SS^\dagger-TT^\dagger+C_{\mathrm{el}}\bigr).
\]

\(\Delta C\) 不写入生产、不做成人工浴。不加 \(T_2\)，不启动非线性，不做 BZ，不重写 \(n,m\)。

代码：`q8_m0_order_audit.py`、`q8_m0_stage1o_order.py`。  
`STAGE1_STATUS = stage1_canonical_order_audit_20260923`。  
JSON：`results/q_line/stage1o_order_20260923/stage1o_order.json`。  
图：`stage1o_delta_vs_lambda.png`、`stage1o_scaled_delta.png`、`stage1o_neumann.png`、`stage1o_deltaC_heatmap.png`、`stage1o_deltaC_diag.png`。  
固定 MP8。CJJ \(11\,\mathrm{s}\)，审计墙时 \(160\,\mathrm{s}\)。诊断 \(\lambda\) 只乘在 \(G_{\mathrm{self}}\) 上；电流-K 的 \(\eta^2 N_{\mathrm{cell}}\) 保持物理值，仅作对照。

---

## 物理问题

canonical defect \(\Delta_{\mathrm{can}}=[b_{\mathrm{out}},b_{\mathrm{out}}^\dagger]-1\) 的最低非零阶在哪里？若在 Born/Kubo 已经消掉的那一阶之后才出现，就可以把 1N 的失败读成理论闭合问题，而不是 H2 模、\(Q/S\) 或漏模的实现错误。

---

## 阶数

场 I/O，H2 输出，左右两端口。\(\lambda=0\) 时三类都是 \(0\)（残差 \(3\times10^{-10}\)）。Born 电流-K 的 \(2\mathrm{Re}A_2+\mathrm{comm}_s\) 仍是 \(2.35\times10^{-8}\)。

| \(\lambda\) | \(\Delta\) 重求 n=1 | \(\Delta\) 重求 n=4 | \(\Delta\) Born n=1 | \(\Delta\) 电流-K 两模 |
|---|---|---|---|---|
| 0 | \(0\) | \(0\) | \(0\) | \(0.029\) |
| \(1/64\) | \(-6.3\times10^{-4}\) | \(-6.3\times10^{-4}\) | \(-4.7\times10^{-4}\) | \(0.012\) |
| \(1/32\) | \(-1.7\times10^{-3}\) | \(-1.8\times10^{-3}\) | \(-9.3\times10^{-4}\) | \(0.0038\) |
| \(1/16\) | \(-6.7\times10^{-3}\) | \(-6.9\times10^{-3}\) | \(-1.8\times10^{-3}\) | \(0.020\) |
| \(1/8\) | \(-0.037\) | \(-0.038\) | \(-0.0034\) | \(0.12\) |
| \(1/4\) | \(-0.26\) | \(-0.23\) | \(-0.0059\) | \(-0.66\) |
| \(1/2\) | \(-1.52\) | \(-0.40\) | \(-0.0080\) | \(-15.0\) |
| 1 | **\(1.18\)** | **\(35.2\)** | **\(-0.0013\)** | **\(151\)** |

小 \(\lambda\) 窗（\(1/64\) 到 \(1/4\)）拟合：

| 曲线 | \(p_\gamma\) | \(p_\eta=2p_\gamma\) | \(r^2\) |
|---|---|---|---|
| 重求 n=1 | \(2.18\) | \(4.35\) | \(0.985\) |
| 重求 n=4 | \(2.15\) | \(4.29\) | \(0.988\) |
| Born n=1 | \(0.91\) | \(1.82\) | — |
| \(C_{\mathrm{el}}\) | \(3.18\) | \(6.4\) | — |

相邻倍增给出的瞬时幂次从 \(1.47\) 升到 \(2.79\)：越小的 \(\lambda\) 越能看见 Born 地板，越大的 \(\lambda\) 越高阶开始进入。\(\Delta/\lambda^2\) 在 \(1/32\)–\(1/16\) 大约是 \(-1.7\)，不是 \(\Delta/\lambda^4\) 那种平台。

因此：诊断 \(\lambda\)（\(\gamma\to\lambda\gamma\)）上，重求和缺陷从 **\(\lambda^2\)** 开始；换成现有 \(\eta\sim\sqrt{\gamma}\) 记账，就是 **\(\lambda_\eta^4\)**。这正是「Kubo 最低阶之后的下一非零阶」。

Born 场 \(\lvert\Delta\rvert\) 全程小于 \(0.01\)。最小 \(\lambda\) 上重求和与 Born 只差 \(1.6\times10^{-4}\)；到物理 \(\gamma\)，Born 仍是 \(-0.001\)，重求和已是 \(1.18\)。多出来的那一块（重求 − Born）从 \(\lambda^2\) 附近涨起来。

\(C_{\mathrm{el}}\) 在小 \(\lambda\) 上可忽略（\(10^{-9}\)），物理 \(\gamma\) 上仍只有 \(-0.10\)。失败不在电子噪声的最低阶。

---

## Neumann：缺口从 \(L^2\) 开始

| \(n\) | \(\Delta(\lambda=1/8)\) | \(\Delta(\lambda=1)\) |
|---|---|---|
| 1（Born） | \(-0.0034\) | \(-0.0013\) |
| 2 | \(-0.0090\) | \(0.345\) |
| 3 | \(-0.030\) | \(6.95\) |
| 4 | \(-0.041\) | \(-355\) |
| 全阶 Volterra | \(-0.037\) | \(1.18\) |

\(\lambda=1\) 时 \(n=1\) 几乎闭合，\(n=2\) 第一次出现 \(O(1)\) 缺口。裸的 \(C_\xi\) 没有因为你把经典 susceptibility 平方、立方而自动长出匹配项。

---

## \(\Delta C\) 结构（诊断，不是补丁）

8 个 \(R_{\mathrm{out}}\) 包：H2、H1、H3–H6、最低两档红外箱。

| \(\lambda\) | \(\lVert\Delta C\rVert_F\) | \(\Delta C_{H2,H2}\) | \(\Delta C_{\mathrm{bin0}}\) | \(\Delta C_{\mathrm{bin1}}\) |
|---|---|---|---|---|
| \(1/16\) | \(4.4\) | \(0.0017\) | **\(1.59\)** | \(-0.024\) |
| \(1/4\) | \(96\) | \(0.15\) | 大 | 大 |
| 1 | \(6580\) | \(-139\) | **\(1205\)** | **\(2773\)** |

物理 \(\gamma\) 上对角元：H1 为 \(506\)，H2 为 \(-139\)，H3–H6 为 \(O(10)\)，两个最低红外箱为 \(10^3\)。热图上的大交叉项在 H1–IR 和 H2–IR。这和已经看见的 H2\(\to\)IR\(\to\)H2 是同一条通道。

没有把 \(\Delta C\) 当成 Langevin 浴去把对易补成 1。

---

## 备选解释

**C1.** 当前 Gaussian closure（\(\mathcal R^A+\xi\) 再 resolvent）在期望值层数学成立，但不是 symplectic 的量子线性网络。Kubo 只保证 \(S=I+\delta S\) 的最低阶；\(\mathcal L\mathbb J\mathcal L^\dagger\) 及更高项没有对应的噪声。支持：Born / \(n=1\) 几乎不坏；重求和 \(p_\gamma\approx2\)（\(p_\eta\approx4\)）；Neumann 从 \(n=2\) 起跳；\(\Delta C\) 在红外。反对：MP8 仍无 BZ 资格，高阶系数的数值大小不能当材料答案。

**C2.** 场 \(Q\) 分析在红外箱上不是等距的，\(\Delta C\) 的红外峰是分析缺陷。支持：最低箱 \(\omega=2\pi/T\)。反对：同一套 \(Q\) 在空片 / \(\gamma=0\) / Born 上把 H2 对易保持在 1；红外峰随 \(\lambda^2\) 出现，不是固定偏移。

**C3.** 实现里 Volterra 伴随和 \(S,T\) 反演在 \(O(\lambda)\) 仍有漏项。支持：最小 \(\lambda\) 上 Born 场 \(\Delta\sim0.03\lambda\)。反对：同一 \(\lambda\) 上电流-K Kubo 是 \(10^{-8}\)；Born 场 \(\lvert\Delta\rvert\) 从不大于 \(0.01\)；物理 \(\gamma\) 的 \(O(1)\) 来自 \(n\ge2\)。

---

## 冻结

\[
\begin{array}{l}
\text{Born：量子一致，只是一阶 perturbative benchmark；}\\
\text{期望值层的线性 feedback resolvent：数学成立；}\\
\text{把它直接提升成量子 I/O：canonical consistency 失败；}\\
\text{缺陷最低非零阶：诊断 }\lambda^2\text{（}\eta\text{ 记账 }\lambda_\eta^4\text{），Neumann }n=2\text{；}\\
\text{漏正交真空模：排除；}\\
\text{Born 层 }\xi\text{-}R\text{ 不匹配：排除；}\\
\text{resummed }n,m,V\text{：无量子物理资格；}\\
\text{真空 RMS：stress diagnostic；}\\
\Delta C\text{：结构诊断，不是生产补丁；}\\
\text{非线性量子反馈：未启动；}\\
T_2/\text{耗散：未加入；}\\
\text{RPA / 准玻色联合线性理论：未采纳。}
\end{array}
\]

生产 \(V(\theta)\) 仍关。

---

## 下一道人审（本轮不代选）

阶数审计支持：继续修补 \(\mathcal R^A+\xi\) 的闭合，优先级低于换一个**本身保对易**的线性量子理论。

你写下的候选是：在强 P1 背景上，把 8 带非相互作用费米子的粒子—空穴涨落 \(B_{cv,k}\) 与光子连续模写成联合线性 Heisenberg 方程。那仍是线性量子反馈，不是完整非线性量子光。

本轮不采纳、不实现。你要拍板的只是：是否把这条 RPA / 准玻色联合线性模型，定为线 Q 的下一版量子主框架。
