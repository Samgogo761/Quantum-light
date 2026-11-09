# 磁性 HHG：D1 少数 \(k\) Floquet 占据（一页）

`20261120`；执行 2026-10-22。`floquet_occupation_few_k`。8 个 \(k\) × \(\pm N\)。协议 `P1_a2_flat_2_8_2_A0.8`。墙时 7.56 s / 预算 120 s。未改 \(H\)、未加 \(T_2\)。`d2 = blocked`。旧 D1 归档只读。

**平台内确有非对角 Floquet 相干，但还不能定量解释 BZ 求和的 \(J^o\)。** 脚本标签 `yes_significant` / `coh_dominates_quantitatively` / `quantitatively_supported` 只对 pair-odd 成立，不作最终结论。

一句话：相干假说得到部分支持；进入 BZ 求和的 same-\(k\) 奇电流仍无法归因。

| 问题 | 判断 |
|---|---|
| 9a 显著非对角相干？ | **是，且 \(k\) 依赖。** 近简并分组后平均 \(\lVert\rho^F_{\mathrm{off}}\rVert/\lVert\rho^F\rVert=0.111\)。Γ/G1 分组后 \(\sim10^{-7}\)–\(10^{-4}\)；G2/M 0.11–0.45。未分组 off-diag 含简并对内规范相干，不能用。 |
| 9b 相干主导 \(J^o\) 非平稳？ | **pair-odd 是；same-\(k\) / BZ 否。** pair-odd 残差相干 MS=0.937，但 K/G1 的 pair-odd RMS 达 0.40–0.46，是不同 \(k\) 总电流差，不是 BZ \(J^o\sim10^{-4}\)。same-\(k\) 普通点 \(R_o\sim1.41\)–1.53、\(C\sim0\)；**G1 对角主导（0.093 / 0.91）**。 |
| 9c 磁畴映射失败？ | **混杂。** 正向 \(U\) \(\sim10^{-12}\)（Hamiltonian 半周期对称成立）。反向 \(U\) 0.3–1.66 是 sewing 约定，不是物理失败。\(\rho\)：Γ \(1.7\times10^{-7}\)，G1 \(\sim8\times10^{-5}\)，M 0.067，G2 0.26。总电流相关 0.985 不能排除奇通道映射失败。 |

H1：**部分支持，BZ \(J^o\) 仍无法定量归因。** 相干块确实制造不平稳奇电流；G1 same-\(k\) 对角主导是反例。H2 不排除、不升级。H3 本轮无新证伪。

8 点：Γ(0,0)，M(24,0)，K(16,32)/\(K'\)(32,16)，G1(5,11)/G1p(43,37)，G2(7,20)/G2p(41,28)。电流拆分 maxabs \(5.3\times10^{-15}\)；平台重传 vs 原传播 0；\(U_F\) 酉性 \(10^{-14}\)；stroboscopic \(\rho\) \(10^{-13}\)–\(4\times10^{-12}\)。

下面三张电流图都是 **G1 pair-odd**，不是 BZ \(J^o\)。

![Floquet 相干份额](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/figure_floquet_coherence.png)

![G1 pair-odd 电流拆分](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/figure_jodd_split.png)

![G1 pair-odd 半周期残差](../../results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120/figure_halfperiod_split.png)

停机。不在“制备周期响应态 / 换观测量 / 冻结经典筛选回量子光”中作选择。不自动开 D2、延长平台、Γ192、压缩光、QED、\(T_2\)。

完整报告：[线Q_八周期磁性HHG_平顶D1_Floquet少数k_20261120.md](线Q_八周期磁性HHG_平顶D1_Floquet少数k_20261120.md)
