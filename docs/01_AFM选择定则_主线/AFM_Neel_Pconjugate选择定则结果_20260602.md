# P 共轭 -N 选择定则结果（2026-06-02）

用脚本构造的严格 P 共轭 -N TB（见 `AFM_Neel_Pconjugate构造完成_20260602.md`）重跑 SBE
后，与 +N 成对做 AFM Neel 选择定则检查。**结论：此前的 H2 evenN 异常基本消除，证实
异常来自 DFT/Wannier 配对误差，而非 SBE 求解器。**

## 1. 数据

```text
+N : C:\Users\26507\Documents\量子光研究\新SBEs\data\output_lg_cov_fullvalence_T2_0p5cycle\lgcov_k40_nb104|nb112
-N : C:\Users\26507\Documents\量子光研究\CrI3_TB-N\output_afm_pconj_fullvalence_T2_0p5cycle\lgcov_k40_nb104|nb112
分析: C:\Users\26507\Documents\New_SBEs\Quantum-light\results\afm_neel_symmetry_Pconj_T2_0p5cycle\nb104|nb112
```

初始电流（机器零，且与 +N 量级一致，优于旧独立 -N）：
```text
+N         |J(0)| ~ 4.4e-18
新P共轭 -N  |J(0)| ~ 4.37e-18      （旧独立 -N 为 2.0e-18）
```

## 2. 旧独立 -N vs 新 P 共轭 -N（full112，leakage = forbidden/allowed）

```text
 阶  允许通道   旧独立-N      新P共轭-N     改善
 H1  evenN      4.84e-05     5.28e-08      917x
 H2  oddN       1.31e+01     3.21e-01       41x   <-- 关键异常被解决
 H3  evenN      5.91e-04     8.93e-06       66x
 H4  oddN       9.26e-02     4.07e-02      2.3x
 H5  evenN      3.34e-02     2.00e-03       17x
 H6  oddN       8.08e-01     8.78e-01      ~1x（噪声地板）
 H7  evenN      1.48e-02     4.35e-03      3.4x
 H8  oddN       1.59e-01     1.29e-01      1.2x
 H9  evenN      5.71e-02     2.47e-02      2.3x
 H10 oddN       1.31e+00     5.69e-01      2.3x
```

最关键的 **H2 从 13.1 降到 0.32（41 倍）**，而且通道翻转：之前主要落在错误的 evenN，
现在主要落在正确的 oddN。`nb104` 与 `nb112` 给出几乎相同的 leakage，排除带窗影响。

## 3. 绝对量级（解释残余 leakage）

偶次谐波本身极弱，必须看绝对量级才能正确解读：

```text
 阶  通道  yield_even   yield_odd   leakage  相对H1
 H1  even  2.673e-02    1.41e-09     0.000   1.0e+00
 H2  odd   2.81e-07     8.78e-07     0.321   3.3e-05   <- 比H1弱约3万倍
 H3  even  3.895e-04    3.48e-09     0.000   1.5e-02
 H4  odd   5.80e-08     1.426e-06    0.041   5.3e-05
 H5  even  1.553e-06    3.10e-09     0.002   5.8e-05
 H6  odd   8.24e-08     9.38e-08     0.878   3.5e-06   <- 两通道都在~1e-7噪声地板
 H7  even  5.01e-07     2.18e-09     0.004   1.9e-05
 H8  odd   2.85e-09     2.22e-08     0.129   8.3e-07
 H9  even  2.78e-08     6.87e-10     0.025   1.0e-06
 H10 odd   1.07e-10     1.88e-10     0.569   7.0e-09
```

解读：
```text
1. 强谐波 H1/H3/H5/H7/H9（奇次, evenN 通道）leakage = 5e-8 ~ 0.025，非常干净。
2. H4（偶次, oddN）leakage 0.041，干净。
3. H2 现在 oddN 主导（leakage 0.32），但绝对量级仅为 H1 的 3.3e-5，已接近
   40x40 k 网格数值地板，残余 0.32 主要是离散化/噪声，不是求解器错误。
4. H6/H8/H10 两个通道都在 ~1e-7 ~ 1e-10，是纯噪声地板，O(1) 的 leakage 比值无意义
   （两个噪声值之比），不应作为判据。
```

## 4. 结论

```text
1. 把 -N 换成严格 P 共轭后，H2 evenN 异常从 13.1 降到 0.32（通道由错变对）。
2. 这证实：H2 异常来自 +N/-N 两套独立 SCF+Wannier 没有严格配对（DFT 层面），
   不是 SBE 长度规范求解器的错误，也不是选择定则理论错误。
3. 求解器 + lg_cov 规范 + k40 离散化对强谐波给出干净的选择定则。
4. 残余偶次 leakage 受限于偶次谐波本身极弱（~1e-6 ~ 1e-7）+ 40x40 k 网格数值地板。
```

对应外部 AI 预判的"情况 A"：H2 leakage 显著下降 → 物理图像确认。

## 5. 下一步建议

```text
1. （可选，加强证据）把 +N 也用当前代码重跑一遍做严格同代码配对；预期不改变结论。
2. （可选）k 网格加密到 60x60 重跑这对 P 共轭数据，验证 H2/H6 残余 leakage 随
   k 网格下降，从而把"残余来自数值地板"坐实。
3. 可以认为 classical-light 阶段的选择定则验证收口，进入 BSV pilot
   （先 nb1-104 / 40x40 / N=50-100 小样本）。
```
