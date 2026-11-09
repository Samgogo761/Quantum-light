# 线Q 量子光多周期磁序响应：θ=π QED 机制诊断（20261120）

只读后处理与自由光场二阶关联推导。未进行新的电子传播。
重叠的 2T0 窗口不是独立重复实验。整数阶 Fourier 不是已认证 HHG。

## 一、反常二阶关联与自由光场协方差

对输入态 \(|\psi_\theta\rangle=D(\alpha)S(re^{i\theta})|0\rangle\)：

\[
M_\theta=\langle(\Delta a)^2\rangle=-\frac12 e^{i\theta}\sinh(2r).
\]

数值：\(r=0.3\)，\(M_0=-0.318327\)，\(M_\pi=0.318327\)。

自由光场正交分量 \(X(t)=ae^{-i\omega_0 t}+a^\dagger e^{i\omega_0 t}\) 满足

\[
\operatorname{Var}_\theta[X(t)]=\cosh(2r)-\sinh(2r)\cos(2\omega_0 t-\theta),
\]

因此压缩角反转给出确定的双频差

\[
\operatorname{Var}_\pi[X(t)]-\operatorname{Var}_0[X(t)]=2\sinh(2r)\cos(2\omega_0 t).
\]

振幅 \(2\sinh(2r)=1.2733\)。
对称二时刻协方差为

\[
C_\theta(t_1,t_2)=\cosh(2r)\cos[\omega_0(t_1-t_2)]-\sinh(2r)\cos[\omega_0(t_1+t_2)-\theta],
\]

其差 \(C_\pi-C_0=2\sinh(2r)\cos[\omega_0(t_1+t_2)]\) 依赖 \(t_1+t_2\) 的双频相位。
若存在磁序敏感的二阶电子响应核，该结构可以进入 H2 电流差。
**这不是已证明的响应核。** 自由光场公式不等于有电子反作用的 QED。

## 二、固定 2T0 平移窗的 H2 复振幅

全局约定：\(J_q=1/(t_1-t_0)\int e^{+iq\omega_0 t}J(t)\,dt\)，矩形窗。

| 窗口 | QED \(\theta=0\) | QED \(\theta=\pi\) | QED \(\delta_\theta\) | GH5 \(\delta_\theta\) | 完整 |
| --- | --- | --- | --- | --- | --- |
| [1,3]T0 | 5.520e-09-7.135e-09i | -9.102e-08+1.317e-08i | -9.654e-08+2.031e-08i | -9.605e-08+1.934e-08i | True |
| [2,4]T0 | 1.164e-08-6.716e-09i | -8.323e-08+2.603e-08i | -9.487e-08+3.275e-08i | -9.526e-08+2.929e-08i | True |
| [2.5,4.5]T0 | 1.016e-08-4.956e-09i | -8.230e-08+3.209e-08i | -9.246e-08+3.705e-08i | -9.540e-08+3.685e-08i | True |
| [2.7,4.7]T0 | 1.276e-08-3.194e-09i | -8.209e-08+3.592e-08i | -9.485e-08+3.911e-08i | -9.700e-08+4.030e-08i | True |

主窗相对 [1,3] 的复差向量变化 \(\delta_\mathrm{complex}=0.125\)。
[2.5,4.5] 对 [2.7,4.7]：\(\delta_\mathrm{complex}=0.031\)，
观测相位差 0.6°，线性泄漏模板相位差 144.0°。
这些窗口相互重叠，不能当成独立重复实验。

## 三、慢变背景与 2ω0 联合拟合

预先固定模型：常数 / 线性 / 二次多项式，加上 \(A\cos(2\omega_0 t)+B\sin(2\omega_0 t)\)。
未再搜索其他背景函数。

| 窗口 | 原始 Fourier | 常数+2ω0 | 线性+2ω0 | 二次+2ω0 | 相对差 const/lin/quad |
| --- | --- | --- | --- | --- | --- |
| [1,3]T0 | -9.654e-08+2.031e-08i | -9.824e-08+2.020e-08i | -9.823e-08+2.012e-08i | -9.806e-08+2.014e-08i | 0.017 / 0.017 / 0.016 |
| [2,4]T0 | -9.487e-08+3.275e-08i | -9.623e-08+3.269e-08i | -9.623e-08+3.266e-08i | -9.618e-08+3.266e-08i | 0.014 / 0.014 / 0.013 |
| [2.5,4.5]T0 | -9.246e-08+3.705e-08i | -9.364e-08+3.701e-08i | -9.363e-08+3.697e-08i | -9.362e-08+3.697e-08i | 0.012 / 0.012 / 0.012 |
| [2.7,4.7]T0 | -9.485e-08+3.911e-08i | -9.496e-08+3.919e-08i | -9.510e-08+3.900e-08i | -9.498e-08+3.891e-08i | 0.001 / 0.003 / 0.002 |

线性泄漏若主导，[2.5,4.5]→[2.7,4.7] 的 H2 相位应跳约 144.0°。
实测 QED \(\delta_\theta\) 相位差为 0.6°。

主窗连续谱：单态 \(\theta=0\) 与 \(\theta=\pi\) 的 \(|J_x^o|\) 峰值都在 \(q=0.025\)；\(\delta_\theta\) 峰值在 \(q=2.025\)，且 \(|J(2.025)|\approx|J_2|\)。压缩角差集中在 H2 邻域，不是把两态共有的低频背景泄漏进 \(q=2\)。

## 四、偶/奇通道尺度

| 量 | RMS | 主窗 H2 \(|J_{2,x}|\) |
| --- | --- | --- |
| \(J^e(\theta=0)\) | 2.859e-01 | 5.474e-03 |
| \(J^o(\theta=0)\) | 2.874e-07 | 1.344e-08 |
| \(J^o(\theta=\pi)\) | 3.171e-07 | 8.721e-08 |
| \(\delta_\theta J^o\) | 1.519e-07 | 1.004e-07 |
| \(\delta_\theta J^e\) | 7.382e-04 | 4.690e-05 |

主窗 \(|J^o|/|J^e|\) RMS 比约 1.005e-06。
不能把很小的奇电流相对变化写成巨大的实际磁性输出。

## 五、GH5 同窗对照

主窗 QED/GH5 复差向量相对偏差 0.035，
相位差 1.95°。
相位与自由场 \(\cos(2\omega_0 t)\) 结构可以相容，但这不等于已经证明响应核。

## 六、图

![h2_windows](C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line\qinput_qin_a2_141_qed_theta_pi_mech_20261120\figures\h2_dtheta_sliding_windows_qed.svg)

![h2_states](C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line\qinput_qin_a2_141_qed_theta_pi_mech_20261120\figures\h2_theta0_vs_pi_main_qed.svg)

![continuum](C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line\qinput_qin_a2_141_qed_theta_pi_mech_20261120\figures\h2_continuum_main_qed.svg)

![phase_leakage](C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line\qinput_qin_a2_141_qed_theta_pi_mech_20261120\figures\h2_phase_vs_linear_leakage.svg)

## 七、两个问题的回答

**A. 当前 H2 压缩角调控是否可能主要由低频泄漏造成？** 不太可能主要由低频泄漏造成。原始 H2 Fourier 与预先固定的二次多项式+2ω0 联合拟合相差 1.3%；[2.5,4.5]→[2.7,4.7] 观测相位差 0.6°，而线性泄漏模板要求约 144.0°。重叠窗仍有复漂移，故不能认证稳定谐波，但现有证据不支持“H2 只是慢变背景泄漏”。

**B. 反常二阶光场关联是否构成目前最合理、但仍待验证的微观机制？** 反常二阶关联是目前最合理、但仍待验证的微观机制。它同时解释了：θ 只旋转涨落椭圆、自由场 Var X 出现确定的 2ω0 差、观测量恰好落在 H2 邻域、以及 GH5/Wigner（只输入二阶统计）能够重现 QED 差向量。尚未证明电子响应核；自由场协方差不是完整 QED；相位相容不等于因果证明。

## 八、边界

- 整数 Fourier 投影不是已认证 HHG。
- 同 α、近似同 ⟨n⟩ 不是完整非经典性判据。
- QED 电流差不能证明输出光压缩或纠缠。
- 不得与 A*=0.8 经典强场谱拼接。
- 后窗 [3,5]T0 仍缺失；本诊断不续跑。

完成后停机。禁止续跑 QED、Cn/SV、提高 GH 阶、改 nmax、加 k、Floquet 或服务器。

