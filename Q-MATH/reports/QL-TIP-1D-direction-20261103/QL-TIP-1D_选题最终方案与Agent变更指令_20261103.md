# QL-TIP-1D：选题最终方案与本地 Agent 变更指令（2026-11-03）

## 1. 项目定位与结论

**正式选题**：静态偏压对明亮压缩真空（BSV）驱动钨金属针尖电子发射的统计响应与高能电子产额的影响。

**建议论文题目**：*Bias-Dependent Photoelectron Emission from Metallic Nanotips Driven by Bright Squeezed Vacuum: A One-Dimensional TDSE Study*。

目标是普通专业期刊的可复现、小规模理论补充，不宣称首次实现 BSV 电子发射、首次发现偏压抑制再散射或严格量子优势。

**仅一个核心自变量**：尖端表面的局域静态提取场强 `F_dc`（单位 V/nm）。其余光波长、脉冲、空间近场模型和表面有效势保持固定。**唯一必要的输入光比较**：强相干光基准 vs 近似单模明亮 BSV；经典热强度噪声分布可以作为很低成本的辅助对照。初稿不研究 DSV 相位、双频光、多模 BSV、波长宽扫描或三维电磁。

**开放的研究假设，不是预设成果**：在偏压抑制平均强度相干光的高能再散射响应时，BSV 的稀有高强度事件是否使高能电子贡献保留得更久？不预先承诺“增强比非单调峰值”“临界偏压”“平台保留”，要由计算判定。

## 2. 关键文献核查结论

1. Heimerl et al., *Multiphoton electron emission with non-classical light*, Nat. Phys. 20, 945–950 (2024), DOI https://doi.org/10.1038/s41567-024-02472-6 —— 已证明光统计到电子数统计的映射；不得重复作为新发现。
2. Heimerl et al., *Quantum light drives electrons strongly at metal needle tips*, Nat. Phys. 21, 1899–1904 (2025), DOI https://doi.org/10.1038/s41567-025-03087-1 —— 已将 1D TDSE + Husimi 强度权重用于 BSV 驱动钨针尖，并在 TDSE 中含约 -0.5 V/nm 静态场、FE=3、近场衰减尺度 20 nm；所以本项目的增量不是“首次加入 DC”，而是**对偏压进行系统调谐与光统计交叉比较**。
3. Hergert, Lampe & Lienau, *Quenching Strong-Field Rescattering of Photoemitted Electrons from Metallic Nanotapers by Using Moderate Bias Fields*, ACS Photonics 12, 2219–2225 (2025), DOI https://doi.org/10.1021/acsphotonics.5c00207 —— 已验证经典强场偏压抑制再散射；需先复现此经典趋势。
4. Heimerl et al., *Attosecond physics in optical near fields*, Nat. Phys. 21, 1893–1898 (2025), DOI https://doi.org/10.1038/s41567-025-03093-3 —— 已研究近场梯度与电子谱结构；故不再以单独近场衰减变化为主要创新点。
5. Jiang et al., *Nonlinear atomic tunnelling boosted by bright squeezed vacuum*, Nature 654, 356–360 (2026), DOI https://doi.org/10.1038/s41586-026-10485-9 —— 已在原子体系研究固定平均能量下调节 BSV 的 g2 与隧穿响应。
6. Liu et al., *Energy- and Momentum-Resolved Single-Shot Yield Fluctuations in Bright-Squeezed-Vacuum-Driven Atomic Strong-Field Ionization*, PRL 137, 153201 (2026), DOI https://doi.org/10.1103/pgkx-8t5c —— 已研究能量/动量分辨的 BSV 逐发产额涨落。
7. Li et al., *Control of tunneling ionization and subcycle interference dynamics with squeezed coherent light*, Phys. Rev. Research 8, 023047 (2026), DOI https://doi.org/10.1103/pbc2-3r6m —— DSV 的相位控制已有相邻研究。
8. Ma et al., *Photoelectrons resolve the phase-dependent variance of bright squeezed vacuum*, arXiv:2610.31811 (submitted 2026-10-22), https://arxiv.org/abs/2610.31811 —— BSV 相位敏感光电子读出也已出现预印本。

截至本次检索，没有从已查阅的核心来源确认**完全相同的“钨针尖偏压扫描 × BSV-Husimi × 高能响应对比”组合**，但这不是证明绝无前人；投稿前仍要按题目、作者及引用链继续核对。

## 3. 最小模型

电子只沿金属针尖轴向 x 运动，x=0 为金属表面；使用一维有效束缚势阱与一维 TDSE：

\[
 i\hbar\partial_t\psi = [-\hbar^2\partial_x^2/(2m_e)+V_{\rm well}(x)+V_{\rm ext}(x,t)]\psi.
\]

**束缚势**：优先沿用已核对的文献有限势阱参数（如 a=0.135 nm、深度 10.8 eV，具体束缚能由本地数值确定，不硬编码为 5.6 eV）。绝对不宣称是精确钨能带。

**光学场**：先完整复现 Heimerl 2025 的近场定义 `xi(x)=1+(FE-1)exp(-x/zeta)`，其中 FE=3, zeta=20 nm，且标出光场在 x<0 的屏蔽近似。注意该定义在远处趋于入射场而不是零；不得把它和纯 `exp(-x/zeta)` 模型混同。初稿模型选定后不再改动。可在附录做纯局域场模型的一次敏感性检查，但非必要。

**静态场**：明确区分表面局域场强 `F_dc` 与仪器施加电压 `U_dc`，两者不可互换。不能直接把无限空间的 `-e F_dc x` 塞进一个有限传播盒，随后把盒边吸收损失当成光电子能谱。可使用来自 2025 Hergert 文献或补充材料的有效一维静电场剖面；若不得已用简化模型，可取在真空侧随 x 衰减且在远处可积的 `F_dc(x)=F_dc(0)/(1+x/L_dc)^2` 并固定 `L_dc`，明示其是有效简化，不是已知的真实针尖电势。电势由空间积分电场构造，确保电荷符号、电子加速方向、能量零点一致。

**静态偏压初态**：如果 DC 场本身使束缚态变为亚稳共振态，需先说明初态准备方案，例如从无场基态缓慢开启静电场，并单独做 DC-only 对照。检验无光偏压发射、场切换引起的虚假电子以及数值边界反射；在 DC-only 主导时不采用简单比值宣称光诱导信号。

**激光**：固定 1600 nm、25 fs（与已发表计算保持可比）、同一脉冲包络。光强统一定义为尖端表面的平均光学峰值强度 \(\bar I_{\rm apex}\)，不与入射焦点强度混用。第一轮选择 \(\bar I_{\rm apex}\approx 1.5\times10^{12}\,\mathrm{W/cm^2}\) 作为代表值（只作初值）。场幅 E(V/nm) 与强度的换算：\(I\approx 1.327\times 10^{11}[E({\rm V/nm})]^2\,\mathrm{W/cm^2}\)。其他强度仅在确有必要时增加 1-2 个。

## 4. Husimi / 光强加权，不做联合 QED

每个光强 I、偏压 F、CEP 相位 phi 先求一个单电子基准电子能谱：`S_coh(E|I,F,phi)`。若 CEP 未锁定，对 CEP 做小数目的收敛平均，并将其纳入 BSV 使用的相位约定；BSV 的强压缩相空间可能在大压缩下近似一维，但不等于随意忽略正负振幅和 CEP 效应。

\[
 S_j(E;\bar I,F)=\int_0^\infty p_j(I;\bar I)\overline S_{\rm coh}(E|I,F)\,\mathrm dI.
\]

强压缩、明亮、近单模 BSV 的光强权重：

\[
 p_{\mathrm{BSV}}(I)=\frac{\exp[-I/(2\bar I)]}{\sqrt{2\pi \bar I I}},\quad I>0,
\]

归一化及平均强度由程序单独检验。比较 coherent (固定 I=Ibar)，可选 thermal (指数分布) 作为正 P / 经典强度涨落对照。**不可宣称统计加权 TDSE 证明了光—电子纠缠、不可经典模拟性或严格量子优势**。如需加入多模 BSV，必须核查模式的时间、频率与空间结构，不能仅改 Gamma 分布形状就声称做了真实多模电离动力学。

**计算成本纠正**：绝对不要进行“189 个条件×10,000 次 shot×独立 TDSE”的盲目模拟。只对每组偏压求一小套 `I` 网格和少量 CEP 代表相位的基准 TDSE，保存响应库，随后用确定性求积或经验证的强度采样做所有光态重加权。BSV 形状参数 k=1/2 的 Gamma 权重可用广义 Gauss–Laguerre 求积；优先检查尾部截断误差。RTX 5060 Ti 和 CPU 的性能以本机 benchmark 为准，不能接受未测量的 8–12 小时承诺。

## 5. 论文主观测量和判据

**主观测量**：相同平均尖端光强下的高能窗电子产额
\[
 Y_{H,j}(F)=\int_{E>E_*}S_j(E;F)\,dE,
 \quad R_H(F)=\frac{Y_{H,\mathrm{BSV}}(F)}{Y_{H,\mathrm{coh}}(F)}.
\]

其中 E_* 由**先完成的 coherent 基准谱**确定后固定，不能为了得到“好看”结果事后改阈值。能量轴必须去除静态电势的统一加速偏置，并确认能谱提取方法在 DC 场下成立。

**辅助量**：总发射 Y_total，归一化高能比例 \(f_{H,j}=Y_{H,j}/Y_{{\rm total},j}\)，以及偏压淬灭的相对比值 \(Q_j(F)=Y_{H,j}(F)/Y_{H,j}(F_{\rm ref})\)。同时报告 R_H 和绝对 Y_H，避免分母趋零时出现虚高。

**候选效应**：`Q_BSV(F)` 比 `Q_coh(F)` 下降更慢或出现某段差异；也可能同向、无显著差异。我们不预置符号或峰位。若 BSV 的平均电子谱没有平台，不能为了突出“再散射平台保留”强行这样命名；统一讨论**高能电子贡献与偏压响应**。可以把“强度涨落的稀有事件”和偏压抑制的竞争作为待检验物理解释。

## 6. 计算顺序与停机判据

**Stage 0 — 保留已进行工作**：继续原始 QL-TIP-1D 一维势阱、无场传播和 coherent 800 nm 基准（仅作为 solver 验证）。必要时 Mathematica 对有限势阱本征值作独立检查。

**Stage 1 — 1600 nm coherent 校准**：固定 FE=3, zeta=20 nm，建立几组强度下的光电子能谱；给出谱的归一化、自由电子的渐近能量定义、CEP 平均结果、网格/时间步/吸收边界测试。先至少确认高能谱结构在数值上可信。若没有可信的高能谱，停止后续 BSV 讨论。

**Stage 2 — 少量偏压点**：只选 `F_dc=0.0,0.5,1.0,1.5 V/nm`（若 DC-only 发射过大或电子逃逸定义不稳，则立即缩小范围），每点先用 10–20 个光强的粗网格、至少两种 CEP 约定，形成初版 `S_coh(E|I,F)`；通过后再加密。测试静态场开关、表面电场符号、势垒形状和静态能量增益。

**Stage 3 — BSV 确定性加权**：利用相同 `S_coh` 数据构造 coherent/BSV；可增加 thermal 负控。先只报告总产额与固定高能窗产额。做强度网格、强尾积分与 CEP 收敛，避免在 BSV 尾部数据不足时外推/截断。

**Stage 4 — 决定是否继续**：若存在大于数值/积分误差、在不同能量阈值小幅扰动下也稳定的偏压依赖对比，增加偏压中间点和 1 个辅助平均强度，完成 4 张论文图。若没有可靠差异，停止扩展，转为**空间近场衰减长度 × BSV** 的单变量备选（先固定 DC=0.5，改变 zeta=5,10,20,40nm）；若备选也只有平凡统计缩放，不强行投稿。

## 7. 最少四张正式图

1. Fig.1：一维有效束缚势、静态提取场和光学近场示意；coherent 参考谱与网格可靠性（主图/附录）。
2. Fig.2：固定两三个代表性偏压，coherent 与 BSV 的电子能谱（统一静态能量校正和强度）；展示高能窗定义。
3. Fig.3：随 `F_dc` 的高能窗绝对产额 `Y_H`、统计增强比 `R_H`、高能占比 `f_H`；阴影或误差条注明数值与加权误差。
4. Fig.4：强度分辨电子响应 `Y_H(I,F)` 与 BSV 权重 `p_BSV(I)` 的耦合、代表性稀有高强度区对积分的贡献（可画累积分布），回答“为什么偏压会改变统计结果”。

如果发现统计比值无明显变化，不要以美术方式制造效果，应在 Stage 4 决定切换备选题。

## 8. 对 Gemini 报告的科学与工程纠错

- *不能预设* `R_H(F)` 或总产额增强比随偏压必然出现非单调峰值：结果符号与形状由完整 `Y(I,F)` 决定。
- 将低偏压到高偏压简单说成“肖特基势垒下降”需要镜像势等具体模型；若没有 image-charge 修正，只称外加电场导致的势垒倾斜/降低。
- 真实静态提取场不可能在整个无限真空域保持同一线性电势。必须设定有限、物理可解释的场剖面，明确能量零点与电子逃逸判据。
- 电子 Fano 因子 / `g2` 不是仅由单电子 TDSE 的发射概率就唯一确定。若论文报告电子计数统计，必须额外明示电子发射事件与探测计数的随机过程模型；首稿不报告这些量。
- `10,000 shots × 189 条件` 不是合理的独立 TDSE 任务矩阵。不同光态共用 TDSE 响应库；大量采样只作用于廉价插值/加权步骤。
- 单模 BSV Husimi 的相位结构与多模 BSV 的真实时空模式需要区分；优先单模明亮极限。
- 光学场幅、强度与近场增强的定义必须统一，报告 `I_inc`、`I_apex`、`E_apex`，但论文中用同一基准做公平比较。
- 不提供未实测的 GPU 机时或不存在的“1.5%必然收敛”保证；用本地 benchmark 和误差预算确定。

## 9. Origin / 本地数据交付格式

所有运行保存实际物理数据，不以图片代替：
- `model_params.yaml`：运行配置与单位；
- `run_manifest.csv`：每次 TDSE 的场强、I、CEP、网格、时间步、收敛标签、代码 commit；
- `bound_state.csv`：x_nm, V_eV, psi_real, psi_imag, density；
- `coherent_spectrum_<runid>.csv`：energy_eV, spectrum_per_eV, absolute_yield_density；
- `response_library.h5` 或 `.npz`：F_dc、I、CEP、energy、S(E)、Y_tot、Y_H、配置哈希；
- `statistical_yield_origin.csv`：F_dc_V_per_nm, meanI_W_per_cm2, state, Y_total, Y_high, f_high, RH, uncertainty；
- `spectra_origin.csv`：energy_eV, state, F_dc_V_per_nm, spectrum_per_eV（按 Origin 长表或宽表均可，但要有明确元数据）；
- `integration_audit.csv`：求积阶数、强度上界、尾部贡献、累计误差；
- `validation.md`：每个验收关口的具体结果、不通过的项目与可复现运行指令。

Mathematica 负责一维有限势阱独立本征值与少量积分核对；Python/NumPy/SciPy 是基准，CuPy 在 RTX 5060 Ti 上通过正确性和测速才启用。Origin 承接所有正式图件。

## 10. 发给本地 Agent 的即时变更命令

继续当前任务书的 Stage 0 与一维 TDSE 数值验证，不得清空当前进度。**在进入高成本大网格之前，把主研究自变量锁定为 `F_dc`，固定 1600 nm、FE=3、zeta=20 nm、单模明亮 BSV。**

先复现 2025 Nature Physics 一维相干基准和 2025 ACS Photonics 偏压抑制再散射的经典趋势；特别检查 DC 场剖面、静态场单独发射、渐近能量定义、CEP、吸收边界。只在这些基准过关后建立偏压 × 光强的 TDSE 响应库，再用确定性 Husimi 求积得到 BSV 谱。不要运行 10,000 个独立 TDSE shots，不要急着加入多模 BSV、DSV、两色光或三维 COMSOL。

项目首要要回答的不是“有没有一个预先指定的非单调峰值”，而是**相同平均光强条件下，BSV 的稀有高强度事件能否系统改变静态偏压对高能电子产额的抑制曲线**，以及这种差别是否超过收敛误差。对全部结果输出带单位的 CSV 和完整数值库供 Origin 作图。未见可信差异就停机审计或切换备选，不得虚构可发表发现。
