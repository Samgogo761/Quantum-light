# 磁性 HHG：平顶 D1 的少数 \(k\) Floquet 占据 / 相干贡献

`20261120` 为任务标签；实际执行 2026-10-22。外审批准的最小机制检验 `floquet_occupation_few_k`。Hamiltonian、初态、μ、Peierls、a₂、A\*=0.8、ω₀、CEP、生产 `k_grid` 默认均未改。不引入 \(T_2\)。`propagate_called` 仅限这 8 个 \(k\) × \(\pm N\) 的平台诊断重传；全 BZ 旧 D1 归档只读。`inputs_unchanged = true`。`d2_started = qed_started = t2_introduced = false`。

**结论：平台内电子态确实含有非对角 Floquet 相干，而且在 \(k/-k\) 配对的 pair-odd 电流上，半周期残差主要由相干块贡献。但这还不能定量解释 D1 的 BZ 求和 \(J^o\)。** 进入 BZ 求和的是 same-\(k\) 奇电流 \(\tfrac12\bigl(J_+(k)-J_-(k)\bigr)\)；它在普通点上是混合甚至对角主导的。自动化标签 `coh_dominates_quantitatively` / `quantitatively_supported` 只对 pair-odd 成立，报告正文不以它们作为最终物理结论。

一句话：**Floquet 相干假说得到部分支持；BZ 求和 \(J^o\) 的非平稳性仍无法归因。**

脚本：[tools/b0_small_model/q8_m0_magnetic_hhg_flat_d1_floquet_fewk_20261120.py](../../tools/b0_small_model/q8_m0_magnetic_hhg_flat_d1_floquet_fewk_20261120.py)。产物：[results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/)。墙时 7.56 s，预算 120 s，未超时。

## 1. 本轮问的是因果关系，不是“有没有相干”

上一轮诊断已经支持：周期驱动不自动给出周期电流；中央六周期 \(C_o=0.009\)，\(R_o\approx\sqrt{2}\) 就是弱相关。外审要求把检验从“\(\rho^F_{\alpha\beta}\) 非零”推进到

\[
J(t)=J_{\mathrm{diag}}(t)+J_{\mathrm{coh}}(t),
\]

看磁序奇电流的半周期残差、逐周期相位和非整数谱是不是主要由 \(J_{\mathrm{coh}}\) 贡献。

本轮冻结协议 `P1_a2_flat_2_8_2_A0.8`，\(\mu=-0.009010620372198276\)，`ntpc=400`，中点酉指数，Floquet 原点取平台周期 3 的起点 \(t_0=3T_0\)。每个 \(k\)、每个 Néel 畴构造单周期演化算符 \(U_F\)，把 \(\rho(t_0)\) 投到 Floquet 基底，再按近简并分组拆成对角 / 非对角块后分别传播电流。

硬约束全部守住：不改 \(H\)、不加 \(T_2\)、不开 D2 / Γ192 / 延长平台 / 压缩光 / QED / G1/H2 / 服务器。少数 \(k\) **不能**代表 BZ 求和。

## 2. \(k\) 点选择

Γ48 Monkhorst–Pack，`k_grid_xy` 的 `ij` 索引，`shift=(0,0)`。反演伴侣 \((i,j)\mapsto\bigl((n_k-i)\bmod n_k,(n_k-j)\bmod n_k\bigr)\)。不是只选高对称点。

| 名称 | \((i,j)\) | 角色 | 伴侣 |
|---|---|---|---|
| Γ | (0,0) | 反演不动 | 自身 |
| M（沿 \(b_1\)） | (24,0) | 反演不动 | 自身 |
| K | (16,32) | \(K\) | \(K'\) |
| \(K'\) | (32,16) | \(K'\) | \(K\) |
| G1 | (5,11) | 普通内点 | G1p (43,37) |
| G2 | (7,20) | 离开高对称 | G2p (41,28) |

共 8 点、2 畴。选择依据写入 [kpoint_selection.json](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/kpoint_selection.json)。

## 3. \(U_F\) 数值质量与 stroboscopic 重构

近简并阈值：硬简并 \(\lvert\Delta\varepsilon\rvert\le 10^{-10}/T_0\)；近简并 \(\lvert\Delta\varepsilon\rvert\le 10^{-4}\omega_0\)。八带全部落成四对 \([0,1],[2,3],[4,5],[6,7]\)，符合 PT-AFM 配对结构。未分组的 off-diag 会把简并对内的规范相干算进去，因此主验收用近简并分组后的 `offdiag_frac_near`。

| 检查 | 结果 |
|---|---|
| \(U_F^\dagger U_F-I\) | \(2.2\times10^{-14}\)–\(4.3\times10^{-14}\) |
| \(\lvert\det U_F\rvert\) | \(1\pm 10^{-13}\) |
| 本征重构 \(\max\lvert U_F\phi-\lambda\phi\rvert\) | \(7.7\times10^{-16}\)–\(1.9\times10^{-15}\) |
| \(\mathrm{tr}\,\rho(t_0)\) | \(4\) 到 \(10^{-13}\) |
| 平台重传 vs 原 D1 电流 | \(0\) |
| \(J_{\mathrm{diag}}+J_{\mathrm{coh}}\) 重构 | \(5.3\times10^{-15}\) |
| stroboscopic \(\rho\)（周期 3–10） | Γ \(\sim10^{-13}\)；G2 \(\sim4\times10^{-12}\) |
| \(\rho^F\) 相对相位 vs 准能差 | 同期 \(10^{-13}\) 量级，随周期线性累积的是数值噪声而非物理漂移 |

平台内部的单粒子密度矩阵就是 Floquet 演化：占据冻结，非对角项按准能差走相位。数值上没有“传播器把周期态走丢”的迹象。

占据本身已经不是单一 Floquet 本征态。Γ / G1 接近 4 个模式满占据；M 在一对近简并簇之间有 \(\sim0.01/0.99\) 的泄漏；G2 / G2p 出现 \(0.09\)–\(0.32\) 量级的簇间占据混合。

## 4. 问题 9a：有没有显著的非对角 Floquet 相干？

**有，但是强烈依赖 \(k\)。** 近简并分组后，16 个（\(k\), 畴）样本的平均

\[
\frac{\lVert\rho^F_{\mathrm{off,near}}\rVert_F}{\lVert\rho^F\rVert_F}=0.111.
\]

| \(k\) | \(+N\) near 相干份额 | \(-N\) | 未分组份额（含简并对内） |
|---|---:|---:|---:|
| Γ | \(1.5\times10^{-7}\) | \(1.5\times10^{-7}\) | 0.136 / 0.049 |
| M | 0.106 | 0.111 | 0.195 / 0.116 |
| K | 0.022 | 0.022 | 0.179 / 0.265 |
| \(K'\) | 0.015 | 0.018 | 0.531 / 0.251 |
| G1 | \(5.6\times10^{-5}\) | \(5.8\times10^{-5}\) | 0.237 / 0.025 |
| G1p | \(7.9\times10^{-5}\) | \(8.9\times10^{-5}\) | 0.454 / 0.043 |
| G2 | 0.302 | 0.281 | 0.321 / 0.441 |
| G2p | 0.451 | 0.455 | 0.477 / 0.460 |

脚本自动化 `q9a = yes_significant` 来自平均值 \(>0.05\)。这个平均值被 G2/G2p 和 M 拉高。Γ 与 G1 在去掉简并对内相干之后几乎没有簇间相干。因此 9a 的正确表述是：**相干存在，且在部分普通点上很大；不是每个 \(k\) 都有同等强度的簇间相干。**

未分组 off-diag 普遍更大，正是外审要求近简并分组的原因：不能把本征矢规范选择读成真实相干。

![Floquet 相干份额](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/figure_floquet_coherence.png)

## 5. 两种“奇电流”必须分开

脚本同时存了两种组合：

\[
\begin{aligned}
J^{\mathrm{pair\text{-}odd}}(k)&=\tfrac12\bigl(J_+(k)-J_-(-k)\bigr),\\
J^{\mathrm{same\text{-}k}}(k)&=\tfrac12\bigl(J_+(k)-J_-(k)\bigr).
\end{aligned}
\]

D1 的 BZ 求和 \(J^o\) 用的是 same-\(k\)。pair-odd 在反演不动点（Γ、M）与 same-\(k\) 相同；在 \(K/K'\)、G1、G2 上它主要是 **不同 \(k\) 的总电流差**，RMS 可以到 \(0.40\)–\(0.46\)，比 D1 的 BZ \(J^o\sim10^{-4}\) 大约三个数量级。后面三张图都是 **G1 的 pair-odd**，不能读成 BZ \(J^o\)。

| 对 | pair-odd \(R_o\) | pair-odd 相干 MS 份额 | same-\(k\) \(R_o\) | same-\(k\) 相干 / 对角 MS | pair-odd RMS |
|---|---:|---:|---:|---:|---:|
| Γ/Γ | 0.539 | \(\approx 1.00\) | 同左 | 相干主导，但 \(\mathrm{rms}\,J^o\sim5\times10^{-10}\) | \(5.3\times10^{-10}\) |
| M/M | 1.547 | \(\approx 1.00\) | 同左 | 相干主导 | \(1.09\times10^{-3}\) |
| K/\(K'\) | 0.0029 | 0.999 | **1.42** | 0.65 / 0.36 | 0.404 |
| G1/G1p | \(1.2\times10^{-5}\) | 1.00 | **1.41** | **0.093 / 0.91（对角主导）** | 0.465 |
| G2/G2p | 0.551 | 0.68 | **1.53** | 0.72 / 0.33 | 0.140 |

K 与 G1 的 pair-odd \(R_o\) 可以小到 \(10^{-5}\)–\(10^{-3}\)，只是因为两个 \(k\) 的大电流几乎被半周期映射对上；那不是磁序奇通道已经协变。G1 pair-odd 的 H2 看起来“稳”（`rel_std` \(\sim5\times10^{-7}\)），是对角背景极大；其相干块 H2 仍 `rel_std=0.583`、相位扫过 \(353^\circ\)。

![G1 pair-odd 电流拆分](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/figure_jodd_split.png)

![G1 pair-odd 半周期残差](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/figure_halfperiod_split.png)

![G1 pair-odd 逐周期 H2](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/figure_cycle_h2.png)

## 6. 问题 9b：相干能否定量解释 \(J^o\) 的非平稳性？

**对 pair-odd：可以。对 BZ 求和用的 same-\(k\)：不能。**

pair-odd 中央六周期残差的平均相干 MS 份额是 0.937；对角块 \(R_o\) 平均 0.107，远小于总残差平均 0.528。这就是脚本写出 `coh_dominates_quantitatively` 的原因。

same-\(k\) 完全是另一回事：

- 普通点的 \(R_o\) 都在 \(1.41\)–\(1.53\)，相关 \(C\approx 0\) 到 \(-0.17\)，与 D1 全 BZ 的 \(R_o\sim 1.4\)、\(C_o\approx 0\) 同形态。
- G1 的 same-\(k\) 残差 **90.8% 来自对角块**。这里簇间 \(\rho^F\) 相干本来就 \(\sim 10^{-4}\)，相干解释不了这个点的奇电流非平稳。
- K、G2 是混合：相干占 65% 和 72%，对角仍有 33%–36%。
- Γ 的 same-\(k\) \(J^o\) 小到 \(5\times10^{-10}\)，相干主导没有磁学权重。

对角块的逐周期 H2 在所有对上都几乎冻结（`rel_std` 到 \(10^{-8}\)–\(10^{-14}\)）。非平稳、相位乱跳、非整数功率集中在相干块。这支持“**只要相干电流进了奇通道，它就会制造 D1 看到的那种不平稳**”；但不支持“**BZ 求和 \(J^o\) 的残差已经由这些相干项定量占满**”。G1 明确给出反例：same-\(k\) 奇电流可以主要来自对角块，而对角块本身已经半周期失败。

因此 H1 的本轮判断是：**部分支持，仍无法区分。** 自动化 `hypothesis_H1 = quantitatively_supported` 不得写入最终物理结论。

## 7. 问题 9c：磁畴 Floquet 响应态是否另有映射失败？

Hamiltonian / 电流算符的反演 + sewing 在这 8 个点上成立：\(H\) 误差 \(0\)–\(1.4\times10^{-16}\)，sewing 后 \(10^{-15}\)，电流算符 \(10^{-16}\)。正向半周期

\[
U_{\mathrm{late}}(+N,k)\;\stackrel{?}{=}\;\text{映射}\bigl(U_{\mathrm{early}}(-N,-k)\bigr)
\]

的 \(\max\lvert\cdot\rvert\) 为 \(10^{-14}\)–\(4\times10^{-12}\)。**驱动在平台上的半周期动力学对称，已经下到单 \(k\) 的 \(U_F\)。**

反向 \(U\) 映射 \(0.34\)–\(1.66\) **不要当成物理失败**。它把 \(+N\) 的早期半周期和 \(-N\) 伴侣的晚期半周期用同一套 sewing 伴随去比，约定不闭合；正向已经通过，说明问题在伴随约定，不在 Hamiltonian。

\(\rho(t_0+T_0/2;+N,k)\) 对 \(\rho(t_0;-N,-k)\) 的映射误差是 \(k\) 依赖的：

| \(k\) | \(\rho\) 映射 \(\max\lvert\cdot\rvert\) | 总电流映射相关 \(\langle J_+(t+T_0/2),-J_-(-k,t)\rangle\) |
|---|---:|---:|
| Γ | \(1.7\times10^{-7}\) | \(1.000\) |
| M | 0.067 | 0.997 |
| K | 0.0037 | \(1.000\) |
| \(K'\) | 0.019 | \(1.000\) |
| G1 | \(7.8\times10^{-5}\) | \(1.000\) |
| G2 | 0.260 | 0.912 |
| G2p | 0.134 | 0.974 |

平均总电流映射相关 0.985。外审已经指出：\(\lvert J^o\rvert\sim 10^{-3}\lvert J^e\rvert\) 时，总电流相关几乎完全由偶通道控制。same-\(k\) 奇电流在普通点上 \(R_o\sim 1.4\)、\(C\sim 0\)，说明 **奇通道并没有继承总电流的映射精度**。

本轮不把“磁畴响应态映射失败”升级为主因，也不排除它。G2 的 \(\rho\) 误差 0.26 和 G1 same-\(k\) 的对角主导残差，都是独立迹象，但还没有单独解释 D1。脚本 `q9c = independent_signs_present` 可以保留，前提是同时写上：正向 \(U\) 已过；反向 \(U\) 是约定问题；总电流相关不能为奇通道背书。

## 8. 外审九条的逐条回答

1. **协议**：复用 h8/R0、2+8+2、A\*=0.8、a₂、μ、CEP、中点酉指数。未改传播物理，未加 \(T_2\)。
2. **\(k\) 点**：8 个，含反演不动点、\(K/K'\) 和两个普通内点对。依据已存档。
3. **\(U_F\)**：酉性、行列式、本征重构均到双精度。
4. **\(\rho^F\)**：占据与近简并分组后的簇间相干均已保存。四对近简并是结构而不是偶然。
5. **stroboscopic**：平台内 \(\rho\) 与 Floquet 重构一致到 \(10^{-12}\) 或更好；准能相位对应成立。
6. **电流拆分**：pair-odd 残差相干主导；same-\(k\) 混合，G1 对角主导。不以 \(\rho^F\) 非对角非零宣布机制成立。
7. **磁畴映射**：反演涉及 \(k\to -k\) 时在伴侣点之间比较。正向 \(U\) 成立；\(\rho\) 映射 \(k\) 依赖。
8. **BZ 限制**：少数 \(k\) 的强相干不能代表求和后的 \(J^o\)。K/G1 的 pair-odd 大电流差尤其不能外推。
9. **三个问题**：
   - **9a** 平台内电子态是否含显著非对角 Floquet 相干？**是，且 \(k\) 依赖。** 平均 near 份额 0.111；Γ/G1 分组后几乎没有，G2/M 很大。
   - **9b** 这些相干是否真正主导 \(J^o\) 的非平稳部分？**在 pair-odd 上是；在 same-\(k\) / BZ \(J^o\) 上尚未建立。**
   - **9c** 是否存在磁畴 Floquet 响应态映射不满足协变条件的独立迹象？**有限、混杂。** 正向 \(U\) 不支持“Hamiltonian 映射坏了”；奇通道的 same-\(k\) 残差和部分 \(\rho\) 误差不能用总电流相关 0.985 排除。

## 9. 对 H1 / H2 / H3 的更新

| 假设 | 本轮 |
|---|---|
| H1 Floquet 相干导致 \(J^o\) 非平稳 | **部分支持。** 相干块确实制造不平稳的奇电流；G1 的 same-\(k\) 对角主导说明它还不是 BZ \(J^o\) 的定量归因 |
| H2 磁畴周期响应态不满足协变映射 | **仍不能排除，也未升级为主因。** 总电流映射好，奇通道映射不好；正向 \(U_F\) 已协变 |
| H3 BZ 差分不稳定 | **本轮无新证伪。** 少数 \(k\) 不能回答求和相消 |

## 10. 本轮明确不宣称的事情

- 没有证明 Floquet 相干就是 D1 全 BZ \(J^o\) 失败的主要原因。
- 没有证明无耗散体系无法实现磁性偶次 HHG。
- 没有把 pair-odd 的“稳定 H2”读成磁性选择定则。
- 没有批准 D2、延长平台、Γ192、压缩光、QED、G1/H2、耗散模型或服务器。
- 没有在“制备对称适配的周期响应态 / 改观测量 / 冻结经典筛选回到量子光”三者中作选择。

源码哈希：classical `8fac7a95…`，model `5d53090c…`，D1 `99094af3…`，本脚本 `6c2ba1e2…`。

JSON：[fewk_status.json](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/fewk_status.json)，[pair_metrics.json](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/pair_metrics.json)，[floquet_per_k.json](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/floquet_per_k.json)，[mapping_checks.json](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/mapping_checks.json)。

一页：[线Q_八周期磁性HHG_平顶D1_Floquet少数k_一页_20261120.md](线Q_八周期磁性HHG_平顶D1_Floquet少数k_一页_20261120.md)。

**停机。** 旧 D1 与上一轮诊断归档未改。下一步需要你决定研究线，本轮不自动开算。
