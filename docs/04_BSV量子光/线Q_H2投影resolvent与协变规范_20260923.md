# 线 Q：正确的规范检验，以及 H2 投影下的 Born 误差（2026-09-23）

性质：\(\sigma_{\max}(\mathcal L_{\mathrm{ph}})\) 只是全局 worst-case 瞬态放大，不再当 Born 通行证。离散 \(\mathcal L\) 严格因果、有限窗上幂零，\((1-\mathcal L)^{-1}\) 存在。旧判据「固定 \(\Pi_0\) 下 \(I_{\mathrm{state}}+I_{\mathrm{contact}}\to 0\)」撤回：突然常数 \(\delta A\) 是 quench，慢 \(\sin^2\) 有 \(E=-\dot A\neq 0\)。

真正两层标准：

1. Hamiltonian、\(\Pi_0(k)\) 和 \(k\) 标签一起平移后，BZ 总电流是否只留下求积误差。
2. 批准的 H2 可观测量上，\(P_{\mathrm{H2}}[(1-\mathcal L)^{-1}-(1+\mathcal L)]\) 以及 \(\langle\mathrm{H2}\lvert\mathcal L^2\rvert\mathrm{H2}\rangle\) 是否足够小。

代码：`q8_m0_gauge_shift.py`、`q8_m0_h2_resolvent.py`、`q8_m0_stage1l_h2res.py`。  
JSON：`results/q_line/stage1l_h2res_20260923/stage1l_h2res.json`。  
图：`stage1l_gauge_residual.png`、`stage1l_h2_L2.png`、`stage1l_h2_resolvent.png`、`stage1l_A2_resolvent.png`、`stage1l_window_scale.png`。

冻结未破：\(T_2=0\)、不扣 DC、不改 P1、不扩 Fock、不上 48/96/192、不对红外热点做 \(k\)-自适应、生产 \(V\) 仍被挡住。墙时 \(282\,\mathrm{s}\)。CJJ 167 点（hold 0 缓存）加平移与延长窗。

`STAGE1_STATUS = stage1_h2_resolvent_and_covariant_gauge_20260923`。

---

## 物理问题

\[
\text{那个几乎与 H2 正交的红外方向，绕几圈以后会不会回到我们真正观测的 }A_2,n,m,V(\theta)\text{？}
\]

预先写好的出口（本轮执行，不改物理）：

- 协变 \(k\)-平移在置换下不变，连续平移残差随 \(n_k\) 下降或已经很小；并且 H2 投影的 resolvent 误差是几个百分点 → Born 可以重新审核（仍不自动升格 \(V\)）。
- 协变检验的实现成立，而 \(\mathrm{H2}\to\mathrm{IR}\to\mathrm{H2}\) 或 \(A_2^{\mathrm{exact}}-A_2^{\mathrm{Born}}\) 达到不可忽略 / \(O(1)\) → **人审**：是否正式升级到线性反馈重求和。
- 旧低频接触相消比不再单独叫停。

---

## 工程核对

- \(k_{\mathrm{cart}}\circ k_{\mathrm{frac}}\) 逆映射；MP 置换集闭合。
- 脉冲结束后 \(n_{\mathrm{hold}}\) 只加长轨迹，\(t\le T_p\) 的 \(\mu(t)\) 与 hold 0 重合到 \(10^{-12}\)；\(P_1\) 包络不变（\(t>T_p\) 时 \(\kappa_{\mathrm{cl}}=0\)）。
- \(\mathcal L^2\) 等于连续两次 `apply_L`。
- Volterra 的 \(A_{\mathrm{sheet}}\) 满足 \(A-(A_{\mathrm{src}}+\mathcal L A)=0\)，相对 \(L^2\) 残差 \(10^{-15}\)。有限窗上 resolvent 存在，与巨大 \(\sigma_{\max}\) 不矛盾。

---

## 规范：置换通过，连续平移的求积尚未锁住

\(\delta\kappa\) 与 \(\Pi_0(k+\delta\kappa)\)、\(H(k+\delta\kappa+\kappa_{\mathrm{cl}})\) 一起走。比的是整 BZ 的 \(\mu(t)\)。

| 检验 | \(n_k\) | \(\lVert J_{\delta k}-J_0\rVert/\lVert J_0\rVert\) | 备注 |
|---|---|---|---|
| MP4 置换 \(\delta k=(1/4,0)\) | 16 | \(5.3\times 10^{-16}\) | 同一点集，只是重标记 |
| 连续 \(+\delta k\)，MP4 | 16 | 1.18 | \(\delta k_{\mathrm{frac}}=(0.07,0.03)\) |
| 连续 \(+\delta k\)，MP6 | 36 | 0.871 | even \(0.94\)，odd \(0.29\) |
| 连续 \(+\delta k\)，MP8 | 64 | 0.850 | even \(0.61\)，odd \(0.44\) |
| 突然常数 \(A\)、固定 \(\Pi_0\) | 16 | 1.34 | **不是规范**，只作对照 |

连续平移的残差从 \(1.18\) 降到 \(0.85\)，自动门因此记 `covariant_pass`。人审应把它读成：

- 离散环面上「常数矢势 = \(k\) 重标记」的**实现**是对的；
- 连续 BZ 积分的平移不变性在 \(4/6/8\) 上**还没看见趋于零**。这和嵌套 Born \(|A_2|\) 本身尚未锁住是同一类求积问题，不是 Hamiltonian 规范泄漏，也不是旧判据那种「失败」。

旧口径维持撤回。

---

## H2 投影：第一次很小，绕一圈以后不是

\(\phi_{\mathrm{H2}}\) 已归一。\(\mathrm{rayleigh}_n=\langle\mathrm{H2}\lvert\mathcal L_{\mathrm{ph}}^n\rvert\mathrm{H2}\rangle\)，\(\mathrm{gain}_n=\lVert\mathcal L_{\mathrm{ph}}^n\phi_{\mathrm{H2}}\rVert\)。

| 网格 | \(\mathrm{rayleigh}_1\) | \(\mathrm{gain}_1\) | \(\langle\mathrm{H2}\lvert\mathcal L^2\rvert\mathrm{H2}\rangle\) | \(\mathrm{gain}_2\) | \(\mathrm{rayleigh}_3\) |
|---|---|---|---|---|---|
| MP4 | \(-0.031\) | 11.6 | **4.40** | \(1.8\times 10^3\) | \(-644\) |
| MP6 | \(-0.055\) | 3.75 | 0.083 | 49.7 | \(-1.72\) |
| MP8 | \(+0.027\) | 10.0 | **0.73** | 519 | \(-28.8\) |
| 1I 传统 | \(+0.039\) | 7.84 | 0.009 | 144 | \(-14.1\) |
| 1I AI | \(+0.003\) | 4.71 | \(-0.016\) | 51.5 | \(-0.26\) |

第一次回到 H2 仍然只有几个百分点，与 1K 一致。第一次出去的正交质量 \(\mathrm{gain}_1^2-\mathrm{rayleigh}_1^2\) 是 \(14\)–\(134\)：振幅进了红外。二次返回在均匀网格上并不小，并且随 \(k\) 采样跳动；1I 两套的 \(\mathcal L^2\) Rayleigh 接近 0，但 \(\mathcal L^3\) 仍可以到 \(O(10)\)。

光子空间的 resolvent 分量 \(\langle\mathrm{H2}\lvert Q[(1-\mathcal L)^{-1}-(1+\mathcal L)]S\rvert\mathrm{H2}\rangle\)：

| 网格 | 精确 \(\langle\mathrm{H2}\rvert\) | Born \(\langle\mathrm{H2}\rvert\) | 差 | \(\lvert A_2\rvert_{\mathrm{R}}\) | \(\lvert A_2\rvert_{(1+\mathcal L)}\) | \(\lvert A_2\rvert_{\mathrm{exact}}\) |
|---|---|---|---|---|---|---|
| MP4 | \(-305\) | 0.97 | \(-306\) | 0.146 | 0.382 | **164** |
| MP6 | \(-413\) | 0.95 | \(-414\) | 0.216 | 0.958 | **252** |
| MP8 | \(-1.75\) | 1.03 | \(-2.77\) | 0.129 | 2.02 | **1.99** |
| 1I 传统 | 1.39 | 1.04 | 0.35 | 0.193 | 0.847 | **2.82** |
| 1I AI | 0.31 | 1.00 | \(-0.69\) | 0.208 | 0.548 | **0.71** |

MP4/6 上 \(|A_2|_{\mathrm{exact}}\sim 10^2\) 是粗网格把红外 resolvent 漏进 H2 收集核。MP8 和 1I 把这个数压回 \(O(1)\)，但**没有**压回 Born 的 \(0.13\)–\(0.21\)。相对误差 \(\lvert A_2^{\mathrm{exact}}-A_2^{\mathrm{Ronly}}\rvert/\lvert A_2^{\mathrm{Ronly}}\rvert\) 在 MP8 / 1I 仍是 \(16\) / \(14\) / \(2.5\)。

---

## 时间窗：\(\sigma\) 和光子二次返回都涨；脉冲窗 \(A_2\) 不涨

P1 包络不变。只把电子和线性响应再传播 \(4\) 或 \(8\) 个光学周期。\(\omega_{\min}=2\pi/T_{\mathrm{obs}}\)。

| \(T_{\mathrm{obs}}/T_p\) | \(\omega_{\min}\) | \(\sigma_{\max}(\mathcal L_{\mathrm{ph}})\) MP4 | \(\langle\mathrm{H2}\lvert\mathcal L^2\rvert\mathrm{H2}\rangle\) MP4 | 同上 MP8 | MP8 \(\lvert A_2\rvert_{\mathrm{exact}}\) | MP8 光子 resolvent 差 |
|---|---|---|---|---|---|---|
| 1 | 0.00625 | 717 | 4.40 | 0.73 | 1.986 | \(-2.774\) |
| 1.5 | 0.00417 | 2252 | 16.9 | 2.44 | 1.987 | \(-2.772\) |
| 2 | 0.00313 | 4026 | 32.2 | 4.70 | 1.987 | \(-2.765\) |

\(\sigma\cdot\omega_{\min}=4.48,\;9.38,\;12.6\)，不是干净的 \(1/\omega_{\min}\)。光子内积里的 \(\mathrm{H2}\to\mathrm{IR}\to\mathrm{H2}\) 随窗变长。H2 收集核 \(F(t)\) 在脉冲结束后几乎为 0，所以 MP8 的 \(A_2^{\mathrm{exact}}\) 和光子 resolvent 差在延长窗上不动。

巨大 \(\sigma\) 仍然是软光子 / 长窗的 stress。它**不能**单独判死 H2。但「H2 返回量保持有限小」这一半在光子 \(\mathcal L^2\) 上不成立；在脉冲窗 \(A_2\) 上成立的是「数值稳定」，稳定值却是 \(O(1)\) 而不是 Born。

---

## 对人审

**进入人审。** 节点：是否把理论从 Born 正式升级到线性反馈重求和。

自动门条件（协变置换通过，且 H2 多次反馈 / resolvent 误差达 \(O(1)\)）已经触发。AI 不宣布机制，也不开始改 Hamiltonian 或写生产 \(V\)。

Born 仍无生产资格。\(\sigma_{\max}\) 不是通行证。不对红外热点做 \(k\)-自适应。

---

## 备选解释

**A1.** \(\mathrm{H2}\to\mathrm{IR}\to\mathrm{H2}\) 是真的。线性闭路已经把 H2 输出从 \(0.1\)–\(0.2\) 改成 \(O(1)\)。MP8 上 \(A_2^{\mathrm{exact}}\approx 2\) 对窗稳定，Volterra 恒等式干净。支持：所有较密集合（MP8、两套 1I）的 \(\lvert A_2^{\mathrm{exact}}\rvert/\lvert A_2^{\mathrm{Ronly}}\rvert\) 都 \(\ge 3\)。反对：嵌套 Born \(|A_2|\) 自己还在 \(0.13\)–\(0.22\) 之间跳，\(O(1)\) 闭路数也可能随求积再变。

**A2.** 大数主要是粗 BZ 把红外 resolvent 漏进 H2 核。MP4/6 的 \(|A_2|_{\mathrm{exact}}\sim 10^2\) 已证明这种泄漏存在；MP8 的 \(2\) 只是泄漏变小，不是物理闭路增益。支持：\(\langle\mathrm{H2}\lvert\mathcal L^2\rvert\mathrm{H2}\rangle\) 随网格从 \(4.4\) 跳到 \(0.08\) 再跳到 \(0.73\)；1I AI 的二次 Rayleigh 接近 0。反对：即使用 1I AI，\(\lvert A_2^{\mathrm{exact}}\rvert=0.71\) 对 Born \(0.21\) 仍不是几个百分点。

**A3.** 线性 \(R^A\) 前缀不是完整的非线性 \(k\)-平移。红外方向把 \(\delta A\) 推到线性邻域之外，\((1-\mathcal L)^{-1}\) 在数学上有解，物理上却过响应。支持：连续 \(k\)-平移的求积残差仍是 \(O(1)\)；突然固定 \(\Pi_0\) 的对照同样 \(O(1)\)。反对：本轮问的就是这个线性算子上的 Born 资格；它已经给出 \(O(1)\) 的 H2 差。

**A4.** 光子 \(\mathcal L^n\) 被观察窗的最低频箱子支配，所以 \(\sigma\) 和 \(\langle\mathrm{H2}\lvert\mathcal L^2\rvert\mathrm{H2}\rangle\) 随 \(T_{\mathrm{obs}}\) 涨；真正的 H2 收集在脉冲结束后关掉，不该跟 \(\sigma\) 走。支持：MP8 的 \(A_2^{\mathrm{exact}}\) 三窗重合。反对：关掉窗之后留下的那个稳定差，仍然是 \(O(1)\)。

---

## 冻结

\[
\begin{array}{l}
\text{正频光子映射：通过；}\\
A_2^{\mathrm{Born}}\sim 0.13\text{–}0.22\text{：仍是参考量级；}\\
\text{全局 }\sigma_{\max}\gg 1\text{：红外 worst-case，随 }T_{\mathrm{obs}}\text{ 涨；}\\
\text{旧“规范失败”判据：维持撤回；}\\
\text{置换规范：通过；连续 }k\text{-平移求积：尚未趋于零；}\\
\text{H2 一次 Rayleigh：仍是几个百分点；}\\
\mathrm{H2}\to\mathrm{IR}\to\mathrm{H2}\text{ 与 }A_2^{\mathrm{exact}}-A_2^{\mathrm{Born}}\text{：}O(1)\text{ 或更大；}\\
\text{Born：仍无生产资格；}\\
\text{暂不对红外热点继续 }k\text{-自适应；}\\
\text{人审：是否升级到线性反馈重求和。}
\end{array}
\]
