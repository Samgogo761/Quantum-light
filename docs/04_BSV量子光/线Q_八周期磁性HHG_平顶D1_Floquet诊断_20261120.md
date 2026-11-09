# 磁性 HHG：平顶 D1 的 Floquet / 半周期诊断

`20261120` 为任务标签；实际执行 2026-10-22。方案 2：只读后处理与数学推导。Hamiltonian、初态、μ、Peierls、a₂、A\*=0.8、ω₀、CEP、生产 `k_grid` 默认均未改。旧 V48 与 D1 npz 只读。`propagate_called = false`。`inputs_unchanged = true`。`d2_started = qed_started = t2_introduced = false`。

**结论：周期驱动不自动给出周期电流；D1 的 \(R_o\approx\sqrt{2}\) 就是 \(J^o(t)\) 与 \(J^o(t+T_0/2)\) 几乎不相关。逐周期偶次奇源不平稳。优先假设是无耗散 Floquet 相干，而不是磁畴总电流映射失败，也还轮不到独立 BZ 或现象学 \(T_2\)。只推荐下一步：少数 \(k\) 的 Floquet 占据诊断。本轮不执行。**

一句话：**驱动已经周期化；电子仍是 Floquet 叠加，不是对称适配的周期响应态。**

脚本：[tools/b0_small_model/q8_m0_magnetic_hhg_flat_d1_floquet_diag_20261120.py](../../tools/b0_small_model/q8_m0_magnetic_hhg_flat_d1_floquet_diag_20261120.py)。产物：[results/q_line/magnetic_hhg_flat_d1_floquet_diag_20261120/](../../results/q_line/magnetic_hhg_flat_d1_floquet_diag_20261120/)。墙时 2.45 s。

## 1. 周期 Hamiltonian 为什么不自动给出周期电流

平台内部 \(H(t+T_0)=H(t)\)。任意解可按 Floquet 展开（\(\hbar=1\)）

\[
|\psi(t)\rangle=\sum_\alpha c_\alpha\,e^{-i\varepsilon_\alpha t}|u_\alpha(t)\rangle,\qquad |u_\alpha(t+T_0)\rangle=|u_\alpha(t)\rangle.
\]

Peierls 电流算符随 \(A(t)\) 周期，矩阵元 \(\langle u_\alpha(t)|\hat{\jmath}(t)|u_\beta(t)\rangle\) 本身 \(T_0\) 周期，可写成 \(\sum_m j_{\alpha\beta m}e^{-im\omega_0 t}\)。于是

\[
\boxed{J(t)=\sum_{\alpha\beta m} C_{\alpha\beta m}\,e^{-i[(\varepsilon_\alpha-\varepsilon_\beta)+m\omega_0]t},\qquad C_{\alpha\beta m}=c_\alpha^* c_\beta\,j_{\alpha\beta m}.}
\]

两项来源必须分开：

| 贡献 | 条件 | 频谱 |
|---|---|---|
| Floquet 对角 | \(\alpha=\beta\)，或 \(\varepsilon_\alpha-\varepsilon_\beta\in\omega_0\mathbb{Z}\) | 整数谐波 |
| Floquet 相干 | \(\alpha\neq\beta\) 且准能差不是 \(\omega_0\) 的整数倍 | 拍频 / 非整数旁带 |

无耗散有限系统中，\(c_\alpha\) 在平台 Hamiltonian 的 Floquet 基下是常数，由升沿历史固定。Hamiltonian 周期化**不会**把非对角相干清掉。这是 Shirley 理论的直接推论，不是程序错误。

HHG 动力学对称选择定则（Alon, Averbukh, Moiseyev, PRL **80**, 3743 (1998)；Neufeld, Podolsky, Cohen, Nat. Commun. **10**, 405 (2019)）要求波函数是动力学对称算符的本征 Floquet 态，而不是升沿制备的相干叠加。固体中这些选择定则本来就可以被破坏（Moiseyev, PRA **91**, 053811 (2015)）。D1 检验的是更强的命题：平台驱动满足半周期反号之后，无耗散电子**会不会自己走进**那个本征态。

## 2. \(J_{-N}(t)=-J_{+N}(t+T_0/2)\) 需要什么

已有反演恒等式

\[
J_N[A](t)=-J_{-N}[-A](t).
\]

若再假设**整段历史** \(A(s+T_0/2)=-A(s)\)，并且 \(+N\) 在 \(t+T_0/2\) 的响应态是 \(-N\) 在 \(t\) 的对称像，才得到

\[
J_{-N}(t)=-J_{+N}(t+T_0/2)\quad\Leftrightarrow\quad
J^e(t+T_0/2)=-J^e(t),\quad
J^o(t+T_0/2)=+J^o(t).
\]

四件事情**没有一件自动成立**：

1. **Hamiltonian 的动力学对称**：平台上 \(A(t+T_0/2)=-A(t)\) 已经用双精度验证；升沿 \(0\le t<2T_0\) 不满足。
2. **初态准备**：2 周期 \(\sin^2\) 升沿把电子送进一组 \(c_\alpha\)，不必是半周期对称算符的本征组合。
3. **Floquet 占据**：两畴的 \(|c_\alpha|^2\) 必须互为对称像。
4. **相干项**：即便占据正确，残留 \(c_\alpha^*c_\beta\) 仍会在 \(J^o\) 里制造拍频。

D1 只保证了第 1 条的平台部分。

## 3. 共同窗口周期 3–5：修正混窗比较

原 D1 报告把旧 Γ48 的周期 3–5 和新平台的周期 3–9 直接相比，写出「\(R_e\) 降至 51%」。外审要求同一时间窗。本轮重算：

| 量 | 旧 Γ48 3–5 | D1 3–5 | D1 3–9（长期对照） |
|---|---:|---:|---:|
| 驱动 rel | 0.09417 | \(2.47\times10^{-15}\) | \(3.70\times10^{-15}\) |
| \(R_e\) | 0.19015 | **0.09137** | 0.09689 |
| \(R_o\) | 1.26905 | **1.27929** | 1.39851 |
| \(\mathrm{rms}\,J^e\) | 0.02483 | 0.02589 | 0.02587 |
| \(\mathrm{rms}\,J^o\) | \(6.014\times10^{-5}\) | \(1.363\times10^{-4}\) | \(1.706\times10^{-4}\) |
| \(J^o\) 绝对残差 | \(7.63\times10^{-5}\) | \(1.74\times10^{-4}\) | \(2.39\times10^{-4}\) |
| \(\mathrm{rms}\,J^o/\mathrm{rms}\,J^e\) | \(2.42\times10^{-3}\) | \(5.26\times10^{-3}\) | \(6.59\times10^{-3}\) |

同窗比：\(R_e^{\mathrm{D1}}/R_e^{\mathrm{old}}=0.480\)。混窗比是 0.510。**定性不变**：偶电流背景改善；奇电流没有改善。原报告「降至 51%」略偏保守，不能用来宣称 \(R_e\) 已经过关，更不能批 D2。

中央六周期仍作独立长期诊断，不替代同窗对照。

![共同窗口](../../results/q_line/magnetic_hhg_flat_d1_floquet_diag_20261120/figure_common_window.png)

## 4. \(R_o\approx\sqrt{2}\) 的精确含义

令 \(r=\mathrm{rms}\,J^o(t+T_0/2)/\mathrm{rms}\,J^o(t)\)，\(C\) 为平移前后的归一化相关。恒等式

\[
R_o=\sqrt{1+r^2-2rC}
\]

在全部窗口上与直接残差**逐位一致**。

| 窗口 | \(C=\langle J^o(t),J^o(t+T_0/2)\rangle\) | \(r\) | \(R_o\) | \(\lvert R_o-\sqrt{2}\rvert\) |
|---|---:|---:|---:|---:|
| 旧 Γ48 3–5 | 0.216 | 1.027 | 1.269 | 0.145 |
| D1 3–5 | 0.155 | 0.968 | 1.279 | 0.135 |
| D1 3–9 | **0.00926** | 0.987 | 1.399 | **0.016** |

中央六周期上 \(C\approx0\)、\(r\approx1\)，所以 \(R_o\approx\sqrt{2}\) **就是弱相关**，不是「差了一点点才到选择定则」。逐周期 \(C\) 在 \(-0.68\) 到 \(+0.33\) 之间振荡，没有趋向 \(+1\)。

对照：同一窗口 \(\langle J^e(t),-J^e(t+T_0/2)\rangle=0.995\)，\(\langle J_{+N}(t+T_0/2),-J_{-N}(t)\rangle=0.995\)，总电流映射残差 0.0975，与 \(R_e\) 同量级。**大的偶电流已经近似服从半周期反号；失败集中在千分之几的奇电流。**

![半周期平移](../../results/q_line/magnetic_hhg_flat_d1_floquet_diag_20261120/figure_odd_half_shift.png)

## 5. 逐周期复谐波：不平稳

对平台每个完整周期做矩形 Fourier，\(q=1\ldots9\)。H1–H9 **全部** `stationary=false`。偶次奇源：

| 周期 | \(\lVert J^o_{\mathrm{H2}}\rVert\) | H2 奇/偶 | H2 相位 (°) | \(\lVert J^o_{\mathrm{H4}}\rVert\) | \(\lVert J^o_{\mathrm{H6}}\rVert\) | \(\lVert J^o_{\mathrm{H8}}\rVert\) |
|---|---:|---:|---:|---:|---:|---:|
| 2 | \(1.11\times10^{-5}\) | 0.0110 | −101 | \(1.58\times10^{-6}\) | \(9.12\times10^{-6}\) | \(6.09\times10^{-6}\) |
| 3 | \(3.27\times10^{-5}\) | 0.0479 | −171 | \(7.44\times10^{-6}\) | \(1.58\times10^{-5}\) | \(1.32\times10^{-5}\) |
| 4 | \(2.00\times10^{-5}\) | 0.0531 | −80 | \(2.43\times10^{-5}\) | \(1.48\times10^{-5}\) | \(1.47\times10^{-5}\) |
| 5 | \(1.68\times10^{-5}\) | 0.0411 | −73 | \(3.34\times10^{-5}\) | \(3.10\times10^{-5}\) | \(1.46\times10^{-5}\) |
| 6 | \(7.22\times10^{-5}\) | 0.236 | −104 | \(4.22\times10^{-5}\) | \(1.42\times10^{-5}\) | \(4.00\times10^{-6}\) |
| 7 | \(7.76\times10^{-5}\) | 0.634 | −115 | \(2.80\times10^{-5}\) | \(2.22\times10^{-5}\) | \(1.41\times10^{-5}\) |
| 8 | \(2.45\times10^{-5}\) | 0.223 | −103 | \(7.49\times10^{-6}\) | \(2.65\times10^{-5}\) | \(4.62\times10^{-6}\) |
| 9 | \(7.26\times10^{-5}\) | 0.133 | −6 | \(9.29\times10^{-6}\) | \(1.64\times10^{-5}\) | \(1.19\times10^{-5}\) |

H2 奇源：相对标准差 0.645，最大/最小 = 7.01，逐周期相位增量标准差 57°。不是缓慢趋近一个复常数，而是周期间振荡。H4/H6/H8 同样。两畴重叠在多数周期仍接近 1、相对相位 \(\sim0\)，与 D1 主验收一致：不是 π 翻转的纯磁序奇通道。

激发数 \(\langle n_{\mathrm{exc}}\rangle\) 在平台上于 0.354–0.369 之间摆动，相对漂移 4.2%，\(\pm N\) 两边到 \(10^{-8}\) 一致。没有明显长期加热，也没有冻结成单一 Floquet 占据。

![逐周期](../../results/q_line/magnetic_hhg_flat_d1_floquet_diag_20261120/figure_cycle_harmonics.png)

## 6. 非整数频谱：有，但不能认证为准能差

中央六周期 \(J^o\) 的矩形 FFT：频率分辨率 \(\Delta(\nu/\omega_0)=1/6\)。

| 窗 | 整数带功率比（半宽 0.25） | 非整数功率比 | 半整数带（半宽 0.08） | 主峰 \(\nu/\omega_0\) |
|---|---:|---:|---:|---:|
| rect 6 | 0.156 | **0.844** | 0.153 | 0.333 |
| sin² 6 | 0.210 | 0.790 | 0.218 | 0.667 |
| Hann 6 | 0.210 | 0.790 | 0.217 | 0.667 |
| rect 8（全平台） | 0.269 | 0.731 | 0.065 | 0.375 |

非整数功率占优，与「平台电流仍含拍频」相容。八周期平台分辨不出准能差；主峰落在 \(1/3\) 或 \(2/3\) 只说明窗函数把慢包络泄漏到低频，**不得把旁带认证为 \(\varepsilon_\alpha-\varepsilon_\beta\)**。

![频谱](../../results/q_line/magnetic_hhg_flat_d1_floquet_diag_20261120/figure_odd_spectrum.png)

## 7. 三个可证伪假设

| 假设 | 内容 | 本轮 | 区分实验 |
|---|---|---|---|
| **H1 Floquet 相干** | 非对角 \(c_\alpha^*c_\beta\) 持续存在，\(J\) 含准能拍频，不是 \(T_0\) 周期 | **支持**。H2 奇源不平稳；\(C=0.009\)；非整数功率 84% | 少数 \(k\) 上读 \(\lvert c_\alpha\rvert^2\) 与 \(\lvert c_\alpha^*c_\beta\rvert\)；看 \(\arg J_q^o\) 是否按 \((\varepsilon_\alpha-\varepsilon_\beta)/\omega_0\) 推进 |
| **H2 磁畴映射失败** | 即使每畴已经 \(T_0\) 周期，仍不必有 \(J_{-N}(t)=-J_{+N}(t+T_0/2)\) | **当前不支持**。总电流映射相关 0.995，残差与 \(R_e\) 同量级；失败的是 \(J^o\) 的半周期重复，而其前提「每畴已周期」尚未成立 | 若少数 \(k\) 上单畴已 stroboscopic 周期、但两畴 Floquet 对不是反演像，则 H2 升级、H1 降为次要 |
| **H3 BZ 差分不稳** | \(J^o\sim10^{-3}J^e\)，Γ48 相消可主导 \(R_o\) | **未证伪，但不是解释 D1 所必需**。单网格内部已经不平稳 | shift48/D2 才能测 H3；在占据诊断之前 D2 仍封锁 |

H1 是目前唯一被单网格数据正向支持的机制。H3 仍然可能放大 \(R_o\) 的数值，但不能单独解释逐周期相位乱跳。

## 8. 只推荐一项最小实验（不执行）

**选择：`floquet_occupation_few_k`。**

不选独立 BZ 网格：单网格非平稳已经足够提出 H1，加密网格回答不了 \(c_\alpha\)。不选对称约束耗散：还没有说明 \(T_2\) 如何作用在密度矩阵上、是否保留磁性动力学对称；现在加任意退相干是新物理，不是诊断。

| 项目 | 内容 |
|---|---|
| 做什么 | 同一 `P1_a2_flat_2_8_2_A0.8` 协议，取 8 个代表 \(k\)（Γ、M、K 附近及 1–2 个普通点），\(\pm N\)，平台上存储 stroboscopic \(\rho_k\) 并/或对角化平台 Floquet Hamiltonian |
| 规模 | 8 \(k\) × 2 畴 = 16 个单 \(k\) 作业；相对 D1 的 2304 \(k\) 约 1/300 |
| 墙时 | 本机约 10–60 s（含存 \(\rho(t)\) 与 \(8\times(2N_h+1)\) Floquet 对角化）；不需要服务器 |
| 改 Hamiltonian | 否 |
| 加 \(T_2\) | 否 |
| 新全 BZ | 否 |
| 本轮 | **不执行** |

通过标准（以后才用，现在不定阈值）：若少数 \(k\) 上 \(\lvert c_\alpha^*c_\beta\rvert\) 与 \(J_q^o\) 的逐周期旋转同量级，H1 成立，再讨论如何制备对称适配的响应态；若单 \(k\) 已经周期而 BZ 求和后 \(J^o\) 仍乱，再打开 H3。

## 9. 本轮明确不宣称的事情

- 没有证明无耗散体系永远无法实现磁性偶次 HHG。
- 没有把非整数峰认证为准能差。
- 没有批准 D2、延长平台、Γ192、压缩光、QED、G1/H2 或服务器。
- 没有引入现象学 \(T_2\)。
- 没有把新旧协议的谐波绝对值差称为磁性增强。
- 低占据 QED 与 \(A^*=0.8\) 仍差约 40 倍，本轮完全未碰量子光。

## 10. 停机节点

JSON：[floquet_diag_status.json](../../results/q_line/magnetic_hhg_flat_d1_floquet_diag_20261120/floquet_diag_status.json)。一页：[线Q_八周期磁性HHG_平顶D1_Floquet诊断_一页_20261120.md](线Q_八周期磁性HHG_平顶D1_Floquet诊断_一页_20261120.md)。

源码哈希：model `5d53090c…`，classical `8fac7a95…`，D1 脚本 `99094af3…`，诊断脚本 `67405df2…`。

**停机。** 下一步需要你决定，本轮不自动开算：

是否批准「少数 \(k\) 的 Floquet 占据诊断」。它不改物理模型，但会第一次把 \(c_\alpha\) 当作可观测量。若问题主要是升沿制备的相干，就研究如何形成对称适配的响应态；不要靠加长脉冲或加密网格碰运气。若将来确需耗散，必须先写清它如何作用于密度矩阵、是否保留磁性动力学对称。
