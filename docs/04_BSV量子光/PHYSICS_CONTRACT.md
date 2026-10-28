# 线 Q × Q-MATH 物理契约（R0，只读冻结）

**日期：2026-10-23。本文件是契约，不是新传播批准书。R1.5 是数学-only 续篇，不改下列冻结值；R2 仍冻结。**

## 0. 本轮允许与禁止

| 允许 | 禁止 |
| --- | --- |
| R0 档案哈希、R1 解析推导、既有 JSON 只读重算 | 任何新的电子—光子传播 |
| R1.5 纯数学：正 \(P\) 可达集、对偶二次上包络证书、\(n_{\max}=8\) 截断限制 | 把热态残差 \(\varepsilon_2\) 或有限维相干态优化写成无限维正 \(P\) 界 |
| 有限差分接触项核验（不演化态） | QED / GH5 / Cn / SV / GH7 / extra k / nmax / Floquet / 服务器 |
| 标明定理适用条件 | 执行 R2 正 P 优化或反例搜索作业 |
| 主动尝试推翻反常关联假说 | 宣称已认证 HHG、输出纠缠、量子优势、纯 BSV |

输入态是位移压缩态 \(D(\alpha)S(re^{i\theta})|0\rangle\)，\(\alpha=0.5\)，**不是**纯压缩真空 BSV。本文「正 P」始终表示非负 Glauber–Sudarshan \(P\) 测度，**不是** stochastic positive-P SDE。

## 1. 冻结模型与协议

| 项目 | 冻结值 |
| --- | --- |
| 电子模型 | 现象学 M0 八带 `h8`，参数集 R0 |
| 几何 | `build_geometry()`；共享 \(k/-k\)，`K_FRAC=[0.173, 0.271]` |
| 偏振 | \(a_2\) 单位向量；`e = a2_cart[:2]/‖·‖` |
| 光子 | 单共享模式；\(\omega_0=0.05\)；\(n_{\max}=8\) |
| 耦合 | \(\hat A = g s(t)\hat X\)，\(g=0.005\)，\(X=a+a^\dagger\)，Schrödinger 绘景 |
| 协议 | `Qin_a2_1_4_1`：1 周期升沿 + 4 周期平台 + 1 周期降沿 |
| 时间网格 | 11154 步，`dt≈0.067597` |
| 磁畴 | Néel \(N=\pm 1\) |
| 算符顺序 | \(D(\alpha)S(\xi)|0\rangle\)，不得写成 \(S D\) |
| Peierls | 完整 Fock 投影 \(P\exp(i\eta X)P\)，**禁止**换成 \(\exp(i\eta PXP)\) |
| 电流 | \(j_a=-\partial H/\partial k_a\)（\(e=\hbar=1\)） |
| API | `from q8_m0_model import R0, build_geometry, h8, current_ops, dh8` |

源码哈希（独立脚本核验，`matches_expected=True`）：

- `q8_m0_model.py`：`5d53090cabb60238c6c16a74b92ddc061d5adf2d8081e48b46b9bdf0089261d3`
- `q8_m0_qinput_dynamics_20261026.py`：`bf6d4e407bf55f1e6ebc3767f1b2bd6075ac7ef564f195abc1217c7d51671f94`

## 2. 观测量与 Fourier 约定

磁序偶/奇电流：

\[
J^{e,o}(t)=\frac{J^{+N}(t)\pm J^{-N}(t)}{2}.
\]

矩形窗 Fourier（**不是已认证 HHG**）：

\[
\mathcal J_q=\frac{1}{t_1-t_0}\int_{t_0}^{t_1} e^{+iq\omega_0 t}\,J(t)\,dt.
\]

主窗 \([2T_0,4T_0]\)。辅助窗 \([1T_0,3T_0]\)、\([3T_0,5T_0]\)。\(\theta=\pi\) QED 只覆盖到约 \(4.76T_0\)：主窗完整，**后窗缺失**。缺失不得写成物理零；`abs_back=0` 是数据缺口。

核心对象：

\[
\delta_\theta\mathcal J_{2,x}^{o}=\mathcal J_{2,x}^{o}(\theta=\pi)-\mathcal J_{2,x}^{o}(\theta=0),\qquad
\Lambda_{\mathrm{eff}}^{o}=\frac{\delta_\theta\mathcal J_{2,x}^{o}}{M_\pi-M_0}.
\]

\(\Lambda_{\mathrm{eff}}\) 是两点割线，**不是**微观 \(\chi^{(2)}\)。

精确线性响应算符（定义，本轮不重构矩阵元）：

\[
\mathcal J_{2}^{o}[\rho_{\mathrm{in}}]=\operatorname{Tr}\bigl(\rho_{\mathrm{in}}\,\mathsf K_{2}^{o}\bigr).
\]

电流期望对初始光子密度矩阵线性。因此两压缩态等权混合的电流等于两电流的算术平均，无需新传播。

## 3. 代码层已核验的结构事实

1. `dh8` **丢掉** `neel`。因此 `current_ops = -dh8` 在本 \(k\) 点 Néel 无关：独立有限差分给出 \(\max|j(+)-j(-)|=0\)。磁序奇来自 \(\rho_N\)，不是 \(j_a(N)\)。
2. 接触恒等式 \(V=\partial_A H|_0=-j_0\cdot e\) 在机器精度成立（差 \(0\)）。
3. \(W_{\mathrm{FD}}\) 与 \(-j_1\cdot e\) 在 \(\varepsilon=10^{-4}\) 下最大元差 \(\sim 4\times10^{-7}\)；\(j_2\) 的 Frobenius 范数 \(\approx 213\neq 0\)。二阶电流接触项必须保留，不能只写普通 Kubo。
4. 真空方差约定：\(X=a+a^\dagger\)，\(\mathrm{Var}_{\mathrm{vac}}(X)=1\)。

## 4. 解释边界（写进任何后续句子之前）

- 同 \(\alpha\) 对照 **不是** 同平均光子数对照；\(\theta=0\) 与 \(\theta=\pi\) 才是同 \(\langle a\rangle\)、同 \(\langle n\rangle\)。
- GH5 重现差向量 \(\neq\) 输入光是经典态；压缩态可有正 Wigner、无正 \(P\)。
- GH5 与 QED 不一致 \(\neq\) 输出纠缠。
- 整数频率投影 \(\neq\) 已认证谐波峰。
- 输入 \(|M|\) 超出正 \(P\) 界 \(\neq\) 电流输出不可经典模拟。
- 低占据量子模型不得与 \(A^*=0.8\) 经典强场谱拼接。
- Q-MATH-16.10 定理 B 针对**零均值、最低阶双光子相干、归一化图样**；不直接覆盖本协议的位移、有限门、非微扰 \(g=0.005\) 电流 Fourier。
- Q-MATH-16.2 接触核源码**不在** 16.11 包内。不得声称已复现 16.2。

## 5. 状态词

| 词 | 含义 |
| --- | --- |
| `timeout_partial` | 墙时内未跑完协议；已覆盖区间仍可用 |
| `full_switch_incomplete` | 降沿未完成；平台分析仍可做 |
| 后窗缺失 | 数据缺口，不是窗口不稳定的物理结论 |

本契约在未获用户批准前不得被新传播改写。

R1.5 文档：`docs/04_BSV量子光/线Q_QMATH_R1.5_正P经典可达界_20261023.md`。证书方法已建立，数值证书没有；一个位移热态不是完整经典优化。未经另行批准，不得进入 R2，不得计算真实 \(f(z)=\langle z|\mathsf K_2^o|z\rangle\) 的数值包络。
