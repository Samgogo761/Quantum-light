# 量子光多周期磁序响应：Q3 精确 QED 正交协方差验证

`20261120`；执行 2026-10-24。协议 `Qin_a2_1_4_1`。`stopped = true`。

冻结只读：θ=0 QED、θ=π QED、四角 GH5。本轮仅新增 θ=π/2 与 3π/2 的 N=±1 精确 QED。输入是位移压缩态 \(D(\alpha)S(re^{i\theta})|0\rangle\)，\(\alpha=0.5\)，\(r=0.3\)，**不是纯 BSV**。

**判定：** `qed_confirms_covsign_phase_rotation_not_certified_HHG`

**是：** 在固定平均位移、平均光子数，以及相同初始 \(\operatorname{Var}(X)\)、\(\operatorname{Var}(P)\) 下，仅翻转 \(\operatorname{Cov}(X,P)\) 符号，精确 QED 主窗 Néel-odd H2 复相位旋转约 \(107^\circ\)，振幅比仅约 \(1.06\)。与冻结 GH5 复差向量的相对偏差约 \(3.4\%\)、相位偏差约 \(1.3^\circ\)。这支持压缩椭圆相位调控，但不是已认证 HHG，也不是经典不可复制性或反铁磁特殊性证明。

## 一、覆盖与资源

| 批次 | 作业 | 状态 | \(t/T_0\) | 主窗 \([2,4]T_0\) | 后窗 \([3,5]T_0\) | 墙时 | 高端 Fock |
|---|---|---|---|---|---|---|---|
| Q3a \(\theta=\pi/2\) | \(N=+1\) | timeout_partial | 5.736 | 完整 | 完整 | 870 s | \(6.36\times10^{-5}\) |
| Q3a \(\theta=\pi/2\) | \(N=-1\) | timeout_partial | 5.704 | 完整 | 完整 | 870 s | \(6.36\times10^{-5}\) |
| Q3b \(\theta=3\pi/2\) | \(N=+1\) | ok | 6.000 | 完整 | 完整 | 781 s | \(6.34\times10^{-5}\) |
| Q3b \(\theta=3\pi/2\) | \(N=-1\) | ok | 6.000 | 完整 | 完整 | 781 s | \(6.34\times10^{-5}\) |

范数误差 \(\sim5\times10^{-13}\)，Taylor 尾项 \(\sim5\times10^{-16}\)，Fock 报警阈值 0.05 未触发。Q3a 未覆盖完整 \(6T_0\)；按合同不续算最后约 0.26 个周期。冻结 θ=π QED 仍只到约 \(4.75T_0\)，其后窗标为 **unavailable**，不填零。

本协议敏感性窗口是 \([1,3]T_0\)（前）和 \([3,5]T_0\)（后），不是 \([4,6]T_0\)。Q3a 已覆盖 \([3,5]T_0\)。

\(n_{\max}=8\) 实际输入（两角相同资源，截断误差同量级）：

| 量 | 解析 | \(n_{\max}=8\) |
|---|---|---|
| \(\langle X\rangle\) | 1 | 0.99983 |
| \(\langle n\rangle\) | 0.342733 | 0.342540（相对差 \(0.056\%\)） |
| \(\operatorname{Var}(X)\) | 1.185465 | 1.185058 |
| \(\operatorname{Var}(P)\) | 1.185465 | 1.185435 |
| \(\operatorname{Cov}(X,P)\) | \(\mp0.636654\) | \(\mp0.635594\) |

\(P(n\ge9)\approx2.31\times10^{-5}\)。未提高截断。

## 二、主窗 Néel-odd H2（主观测量：\(J_x\)）

约定：\(J_q=\frac{1}{t_1-t_0}\int_{t_0}^{t_1}e^{+iq\omega_0 t}J(t)\,dt\)，窗口 \([2T_0,4T_0]\)。整数 Fourier 投影 **不是** 已认证 HHG。

| \(\theta\) | \(J_x^o\) | \(\lvert J_x^o\rvert\) | \(\arg J_x^o\) | 2D \(\lvert\mathcal J_2^o\rvert\) |
|---|---|---|---|---|
| 0（冻结） | \((+1.164-0.672i)\times10^{-8}\) | \(1.344\times10^{-8}\) | \(-30.0^\circ\) | \(1.537\times10^{-8}\) |
| \(\pi/2\) | \((-2.030+5.615i)\times10^{-8}\) | \(5.971\times10^{-8}\) | \(+109.87^\circ\) | \(6.101\times10^{-8}\) |
| \(\pi\)（冻结） | \((-8.323+2.603i)\times10^{-8}\) | \(8.721\times10^{-8}\) | \(+162.63^\circ\) | \(8.913\times10^{-8}\) |
| \(3\pi/2\) | \((-5.059-3.780i)\times10^{-8}\) | \(6.316\times10^{-8}\) | \(-143.23^\circ\) | \(6.466\times10^{-8}\) |

正交协方差对照：

\[
\delta_\perp\mathcal J_{2,x}^{o}
= \mathcal J_{2,x}^{o}(3\pi/2)-\mathcal J_{2,x}^{o}(\pi/2)
= (-3.030-9.396i)\times10^{-8}.
\]

| 量 | QED | 事前 GH5（只读，不重跑） |
|---|---|---|
| \(\delta_\perp\) | \((-3.030-9.396i)\times10^{-8}\) | \((-3.323-9.553i)\times10^{-8}\) |
| \(\lvert\delta_\perp\rvert\) | \(9.872\times10^{-8}\) | \(1.011\times10^{-7}\) |
| 振幅比 \(3\pi/2\div\pi/2\) | 1.058 | 1.113 |
| \(\Delta\phi\)（wrapped） | \(106.90^\circ\) | \(104.57^\circ\) |
| \(\arg\delta_\perp\) | \(-107.87^\circ\) | \(-109.18^\circ\) |

QED 对 GH5 复差向量：相对偏差 **3.36%**，相位偏差 **1.31°**。预先声明的工程标记是 20% / 20°，不是物理定律。

偶电流主窗 \(\lvert J_{x}^{\mathrm{even}}(\mathrm{H2})\rvert\approx5.45\times10^{-3}\)。奇/偶约 \(1.1\times10^{-5}\)。不要把相位旋转写成巨大磁性源。

冻结 H1–H6 列表上，\(\lvert\delta_\perp J_x^o\rvert\)：H2 \(9.87\times10^{-8}\)；其余 \(1.66\)–\(4.26\times10^{-9}\)。H2 至少大约 20 倍。

## 三、窗口敏感性（不是独立重复实验）

| 窗口 | \(\theta=\pi/2\) 相位 | \(\theta=3\pi/2\) 相位 | \(\Delta\phi\) | 振幅比 | \(\lvert\delta_\perp\rvert\) |
|---|---|---|---|---|---|
| 前 \([1,3]T_0\) | \(123.71^\circ\) | \(-137.68^\circ\) | \(98.6^\circ\) | 1.14 | \(1.008\times10^{-7}\) |
| 主 \([2,4]T_0\) | \(109.87^\circ\) | \(-143.23^\circ\) | \(106.9^\circ\) | 1.06 | \(9.872\times10^{-8}\) |
| 后 \([3,5]T_0\) | \(100.23^\circ\) | \(-155.52^\circ\) | \(104.3^\circ\) | 0.91 | \(9.932\times10^{-8}\) |

重叠两周期窗上，相位旋转保持在约 \(99^\circ\)–\(107^\circ\)。振幅比在 0.91–1.14 间滑动，因此这不是振幅增强故事。这些窗口彼此重叠，不能当作三次独立实验，也不能认证稳定谐波。冻结 θ=π 的 \([3,5]T_0\) 后窗 **unavailable**。

连续谱在低频仍有大幅值，这与此前 QED 一致。主对照仍是冻结的整数 H1–H6 列表，而不是事后改选峰位。

## 四、四角拟合 \(B+Ae^{i\theta}+Ce^{-i\theta}\)

| 系数 | QED | 冻结 GH5 |
|---|---|---|
| \(B\) | \((-3.562+0.942i)\times10^{-8}\) | \((-3.805+0.862i)\times10^{-8}\) |
| \(A\) | \((+4.721-1.576i)\times10^{-8}\) | \((+4.770-1.563i)\times10^{-8}\) |
| \(C\) | \((+0.229-0.613i)\times10^{-9}\) | \((-0.067+0.983i)\times10^{-9}\) |
| \(\lvert C\rvert/\lvert A\rvert\) | 0.013 | 0.020 |
| 相对残差 | 0.342% | 0.263% |

图像仍然是：背景复响应 \(B\) + 随压缩角旋转的项 \(Ae^{i\theta}\)。四个等间隔角不能把 \(e^{i5\theta}\) 与 \(e^{i\theta}\) 分开。近圆拟合 **不是** 微观 \(\chi^{(2)}\)，也 **不能** 证明电流只响应 \(M=\langle(\Delta a)^2\rangle\)。

\(B_{\mathrm{GH5}}\) 与此前位移热态 GH5 H2 \(\approx(-3.808+0.858i)\times10^{-8}\) 接近，这是一致性检查，不是机制证明。

初始 \(\operatorname{Var}(X)=\operatorname{Var}(P)\) 相同，并不表示演化中 \(\operatorname{Var}[X(t)]\) 相同。自由场协方差差为 \(-2\sinh(2r)\sin[\omega_0(t+t')]\); 电子反作用下不能把它当作完整联合动力学的精确公式。

## 五、物理边界

- 已确认：Q2 的 GH5 相位候选，在精确 QED 主窗中同样出现。
- 未确认：稳定 HHG、完整 BZ、Fock 截断误差界、正 \(P\) 经典不可复制、反铁磁相对非磁介质的特殊性。
- 未启动：第三批、服务器、Cn、纯 BSV、GH7、额外 \(k\)、\(n_{\max}\) 扫描、经典 D2/Floquet、R2 优化。
- 未重跑：θ=0 / θ=π QED 与四角 GH5。

## 六、图

- 四角复平面：`C:/Users/26507/Documents/New_SBEs/Quantum-light/results/q_line/qinput_qin_a2_141_qed_q3_covsign_20261120/figures/h2_four_angle_qed_vs_gh5.svg`
- 复差向量：`C:/Users/26507/Documents/New_SBEs/Quantum-light/results/q_line/qinput_qin_a2_141_qed_q3_covsign_20261120/figures/h2_delta_perp_qed_vs_gh5.svg`
- 连续谱：`C:/Users/26507/Documents/New_SBEs/Quantum-light/results/q_line/qinput_qin_a2_141_qed_q3_covsign_20261120/figures/h2_continuum_qed_covsign.svg`

账本：`C:/Users/26507/Documents/New_SBEs/Quantum-light/results/q_line/qinput_qin_a2_141_qed_q3_covsign_20261120/qed_q3_covsign.json`

完成后停机。
