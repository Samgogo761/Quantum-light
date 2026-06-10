# 偶次谐波的 intra/inter 通道分解（2026-06-03，零成本后处理）

本文记录一项**不需要新计算**的分析：用已有的 `Jt_decomposed.dat` 验证偶次谐波
（H2/H4/H6...）由哪一路电流通道携带。这是对论文论断"偶次谐波来自带间极化电流"的
数值支撑。

## 1. 背景与一个诚实的限制

按 Yue & Gaarde 框架，总电流可分为：

```text
J = J_intra + J_pol + J_anom + J_mix
```

其中 `J_anom` 是 Berry 曲率驱动的反常（横向 Hall 型）电流。对 PT 对称的 AFM，
`J_anom` 被对称性强制为零，所有偶次谐波应来自带间极化 `J_pol`。

**限制（必须诚实写明）**：当前 Fortran 求解器只输出 **intra + inter 两路**
（见 `src/mod_current.f90: write_current_decomposed`），其中

```text
J_inter = J_pol + J_anom + J_mix
```

即 `J_anom` 被包含在 `J_inter` 内，**没有单独拆出**。因此用现有文件**无法**直接做
"`|J_anom| << |J_pol|`"的字面检查——那需要在求解器里单独隔离 Berry 曲率项（一个小的
代码改动，可在未来跑 BSV 时顺带加）。

本文能严格给出的是：
```text
(1) 分解闭包：J_tot = J_intra + J_inter（机器精度）
(2) 每个谐波由 intra 还是 inter 通道主导（平行 Jx 与横向 Jy 都看）
```

## 2. 方法

脚本：
```text
tools/analysis/harmonic_intra_inter_decomp.py
```
对 `Jt_decomposed.dat` 的每一路用与 `HHG.dat` 相同的 **Hann 窗 + FFT** 约定
（见 `src/mod_hhg.f90`）做谐波分解，在每个整数谐波附近取峰值幅度，报告
`inter/tot = |J_inter| / (|J_intra| + |J_inter|)`。

数据：lg_cov / 40x40 / T2_cycles=0.5 / λ=3200 nm / 线偏振 θ=0°（沿 x）。
`Jx` 为平行分量，`Jy` 为横向分量。

## 3. 结果

闭包检验（应 ~1e-12）：
```text
+N: ||J_tot-(J_intra+J_inter)||/||J_tot||  x=1.4e-11  y=1.5e-11
-N: 同量级
```

谐波分辨的 `inter/tot`（+N full112，节选）：

| 阶 | 宇称 | inter/tot (Jx) | inter/tot (Jy) |
|---:|:---:|---:|---:|
| 1 | odd  | 0.962 | 0.889 |
| 2 | EVEN | 0.947 | 0.797 |
| 3 | odd  | 0.959 | 0.886 |
| 4 | EVEN | 0.983 | 0.971 |
| 5 | odd  | 0.864 | 0.947 |
| 6 | EVEN | 0.869 | 0.866 |
| 7 | odd  | 0.901 | 0.986 |
| 8 | EVEN | 0.946 | 0.879 |

完整数据（+N/-N × nb104/nb112）：
```text
results/harmonic_intra_inter/plusN_nb104.csv
results/harmonic_intra_inter/plusN_nb112.csv
results/harmonic_intra_inter/minusN_nb104.csv
results/harmonic_intra_inter/minusN_nb112.csv
```

## 4. 结论

```text
1. 分解闭包成立到机器精度，J_intra/J_inter 的拆分是自洽的。
2. 所有谐波——包括偶次 H2/H4/H6/H8——都由 INTERBAND（极化）通道主导，
   inter/tot ~ 0.78-0.98，在平行 Jx 和横向 Jy 分量都成立。
3. 带间主导与带窗无关：nb104 与 full112 给出几乎相同的比值。
4. +N 与 P 共轭 -N 的 intra/inter 结构几乎一致，符合 P 共轭。
```

物理含义：偶次谐波**不是**带内（band-velocity）电流贡献，而是带间极化电流主导。这与
"偶次谐波来自 J_pol"的论断一致；但要把 J_pol 与 J_anom 进一步分离、明确证明
`J_anom ≈ 0`，仍需在求解器里加一个 Berry 曲率反常电流的单独输出。

## 5. 后续（非阻塞）

```text
1. 进入 BSV 时，可顺带在 mod_current.f90 增加 J_anom 单独输出，做到真正的四路分解。
2. 若要论文级证据，再对 J_anom 的 H2/H4/H6 幅度与 J_pol 比较，确认 << 1。
```
