# 量子光多周期磁序响应：位移热态 GH5/MTEF 对照

`20261120`；执行 2026-10-23。协议 `Qin_a2_1_4_1`。
冻结 θ=0 / θ=π 精确 QED 与 GH5，全部只读，不重算。本轮仅两路新 GH5：位移热态 \(N=\pm1\)。
\(\rho=D(0.5)\,\rho_{\mathrm{th}}(\sinh^2 0.3)\,D^\dagger(0.5)\)。\(\bar n_s=\sinh^2 r\approx0.09273\) **只是热噪声**，总 \(\langle n\rangle=|\alpha|^2+\bar n_s\approx0.34273\)。
`stopped = true`。位移热态是一个正 \(P\) 对照，不是全部经典输入的最优界。

**结论：主窗磁序奇 H2 上，位移热态与两压缩态等权混合几乎重合。完整复差 \(\varepsilon_2=|R_2|/|J(\pi)-J(0)|=0.00171\ll0.20\)。这与“当前 H2 响应主要受输入光一、二阶统计控制”相容，但不能证明量子优势，也不能把单个正 \(P\) 热态对照写成全部经典输入的不可模拟性。**

## 数值作业

| 项 | 结果 |
|---|---|
| 新传播 | 仅 `th_p`、`th_m` 两路 GH5，各 25 节点 |
| 墙时 | 并发约 119 s / 路，驱动器约 120 s；预算 600 s |
| 协议 | 11154 步，\(dt\approx0.067597\)，1+4+1 |
| 电子范数 | \(\max|\|e\|-1|\sim1.2\times10^{-14}\) |
| 求积 | 各向同性高斯 Wigner，\(s_X=s_P=\sqrt{\cosh 0.6}\)；未调用 `mp.analytic_xp` |
| 冻结数据 | θ=0、θ=π GH5 只读构造 \(J_{\mathrm{mix}}=[J(0)+J(\pi)]/2\) |

## 输入资源（固定位移、固定平均光子数、平均协方差）

| 量 | θ=0 | θ=π | 位移热态 |
|---|---|---|---|
| \(\langle X\rangle\) | 1.000000 | 1.000000 | 1.000000 |
| \(\langle P\rangle\) | 0.000000 | 0.000000 | 0.000000 |
| \(\langle n\rangle\) | 0.342733 | 0.342733 | 0.342733 |
| nth（仅热噪声） | 0.092733 | 0.092733 | 0.092733 |
| \(\mathrm{Var}(X)\) | 0.548812 | 1.822119 | 1.185465 |
| \(\mathrm{Var}(P)\) | 1.822119 | 0.548812 | 1.185465 |
| \(M=\langle(\Delta a)^2\rangle\) | −0.31833 | +0.31833 | 0 |

资源匹配 `resource_match_ok=true`。\((V_0+V_\pi)/2=V_{\mathrm{th}}\) 成立，\(M_{\mathrm{th}}=0\)。
在固定 \(\langle a\rangle=0.5\)、\(\langle n\rangle=0.34273\) 下，正 \(P\) 输入的 \(\mathrm{Var}(X)\) 区间约 \([1,1.37093]\)。热态 1.185 在区间内；两压缩态 0.5488 与 1.8221 在区间外。

该差别证明**输入资源**的非经典性，但不直接证明**电流响应**的非经典性。

## 主窗 H2 磁序奇复源（Jx，\([2T_0,4T_0]\)）

| 量 | θ=0 GH5 | θ=π GH5 | mix | 位移热态 |
|---|---|---|---|---|
| \(J_{2,x}^o\) | \(9.709-6.127i\) | \(-85.55+23.17i\) | \(-37.92+8.519i\) | \(-38.08+8.577i\) \(\times10^{-9}\) |
| \(\lvert J_{2,x}^o\rvert\) | \(1.148\times10^{-8}\) | \(8.863\times10^{-8}\) | \(3.887\times10^{-8}\) | \(3.903\times10^{-8}\) |
| \(\arg J_{2,x}^o\) | −0.563 | 2.877 | 2.921 | 2.920 |
| 2D \(\lvert\mathcal J_2^o\rvert\) | \(1.192\times10^{-8}\) | \(9.097\times10^{-8}\) | \(4.001\times10^{-8}\) | \(4.020\times10^{-8}\) |

\[
R_2^o=J_{\mathrm{th}}^o-J_{\mathrm{mix}}^o=(-1.604+0.575i)\times10^{-10},\qquad
|R_2|=1.704\times10^{-10},\qquad
|J(\pi)-J(0)|=9.966\times10^{-8}.
\]

\(\varepsilon_2=0.00171\)。热态与 mix 的相位差约 \(0.031^\circ\)。

| 窗 | \(\lvert R_2\rvert\) | \(\arg R_2\) | \(\varepsilon_2\) |
|---|---|---|---|
| 前 \([1,3]T_0\) | \(1.960\times10^{-10}\) | −1.862 | 0.00197 |
| 主 \([2,4]T_0\) | \(1.704\times10^{-10}\) | 2.797 | 0.00171 |
| 后 \([3,5]T_0\) | \(2.192\times10^{-10}\) | 0.695 | 0.00220 |

三窗 \(\varepsilon_2\) 均 \(\ll0.20\)。\(R_2\) 本身已是 \(10^{-10}\) 残差量级，其相位随窗旋转（\(\delta_{\mathrm{complex}}(R_2)\approx2.33\)）不应读成物理信号不稳定；应读成 \(J_{\mathrm{th}}\approx J_{\mathrm{mix}}\) 后的数值余项。

工程分类标记（不是物理定律）：\(\varepsilon_2<0.20\) 记为与二阶输入统计主导相容；\(\varepsilon_2\ge0.20\) 记为相同一、二阶统计不足以决定该响应。

verdict `second_order_statistics_compatible_not_quantum_advantage`。

## 偶通道对照（避免把极弱奇电流写成巨大输出）

时间域 \(\mathrm{RMS}(J_x^o)/\mathrm{RMS}(J_x^e)\approx3.43\times10^{-7}\)。主窗偶电流 H2 上，热态与 mix 几乎不可分辨：\(\lvert J_{2}^e\rvert\approx0.00567320\) vs \(0.00567321\)。压缩角主要调制的是已经极弱的磁序奇 H2，而不是偶背景。

## 图

![H2 复平面：θ=0、θ=π、mix、位移热态](C:/Users/26507/Documents/New_SBEs/Quantum-light/results/q_line/qinput_qin_a2_141_gh5_thermal_20261120/figures/h2_complex_plane_thermal_vs_mix.svg)

![三窗残差 R2（残差尺度，不是物理大漂移）](C:/Users/26507/Documents/New_SBEs/Quantum-light/results/q_line/qinput_qin_a2_141_gh5_thermal_20261120/figures/h2_R2_three_windows.svg)

归档：`results/q_line/qinput_qin_a2_141_gh5_thermal_20261120/gh5_thermal.json`。

## 解释边界

- 本轮是 GH5/MTEF，不是精确 QED。
- \(R_2\) 小 ≠ 证明量子优势；也不等于已经证明电子“只响应二阶关联”。它只说明：在当前协议和观测量上，相同一、二阶输入统计的位移热态与两压缩态等权混合给出几乎相同的 H2 奇复源。
- 位移热态是一个正 \(P\) 对照，不是全部经典随机输入的最优界。
- 两压缩态 \(\mathrm{Var}(X)\) 落在正 \(P\) 区间外，只证明输入资源非经典。
- 两周期矩形 Fourier ≠ 已认证 HHG。
- 未启动新 QED、Cn、SV、GH7、额外 k、nmax、G1/H2 或服务器。

完成后停机。

