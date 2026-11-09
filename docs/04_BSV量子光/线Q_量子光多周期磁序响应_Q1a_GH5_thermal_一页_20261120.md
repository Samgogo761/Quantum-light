# 量子光多周期磁序响应：位移热态 GH5（一页）

`20261120`；执行 2026-10-23。协议 `Qin_a2_1_4_1`。冻结 QED 与两压缩角 GH5。`stopped = true`。

**\(R_2\) 很小：主窗 \(\varepsilon_2=0.00171\)。与二阶输入统计主导相容，不是量子优势证明，也不是全部经典输入的不可模拟性。**

| 量 | 结果 |
|---|---|
| 资源 | \(\langle X\rangle=1\)，\(\langle n\rangle=0.3427\)，nth=0.0927（仅噪声），Var(X)_th=1.185，\(M=0\) |
| 正 \(P\) 区间 | \(\mathrm{Var}(X)\in[1,1.37093]\)；热态在内，两压缩态 0.549 / 1.822 在外 |
| 作业 | 两路 GH5 各 25 节点，约 120 s，范数误差 \(\sim10^{-14}\) |
| 主窗 \(J_{2,x}^o\) | mix \(3.887\times10^{-8}\)，热态 \(3.903\times10^{-8}\)，相位差 \(\approx0.031^\circ\) |
| \(R_2\) / \(\varepsilon_2\) | \((-1.604+0.575i)\times10^{-10}\) / 0.00171 |
| 三窗 \(\varepsilon_2\) | 0.00197 / 0.00171 / 0.00220，均 \(\ll0.20\) |
| 分类 | `second_order_statistics_compatible_not_quantum_advantage` |
| 新 QED / 服务器 | 现在不启动 |

![H2 复平面：θ=0、θ=π、mix 与位移热态几乎重合](C:/Users/26507/Documents/New_SBEs/Quantum-light/results/q_line/qinput_qin_a2_141_gh5_thermal_20261120/figures/h2_complex_plane_thermal_vs_mix.svg)

**外审问题：** 具有相同平均光场、平均光子数及平均协方差，但不同高阶统计的位移热态，是否产生与两压缩态等权混合相同的磁序奇 H2 复响应？

**是，在当前 GH5/MTEF 的 H2 奇通道上几乎相同。\(\varepsilon_2\approx0.17\%\)。这与低阶统计主导相容；不证明量子优势，也不证明电流本身不可经典模拟。**

停机。不自动开 QED、Cn、SV、GH7 或服务器。

