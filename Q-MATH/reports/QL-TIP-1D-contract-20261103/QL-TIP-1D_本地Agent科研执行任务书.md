# QL-TIP-1D｜量子光统计调控金属纳米针尖电子发射：本地 Agent 科研执行任务书

**任务日期：2026-11-03**  
**项目定位：第一篇规模适中、理论严谨、可复现的一维模型论文**  
**执行环境：本地电脑；RTX 5060 Ti（显存容量待检测）；Wolfram Mathematica；Origin**

## 0. 给 Agent 的总指令

你是本项目的本地科研执行 Agent，需要阅读用户提供的背景调查报告以及三篇主要原始论文，并在**本地电脑**完成模型核对、Python 计算程序、数值验证、统计平均、参数扫描和可供 Origin 直接导入的原始数据生产。你可以使用互联网检索官方文档及开源代码，但**不得编造参考文献、下载结果、数值结果或宣称未运行的测试已通过**。任务不是写一份空泛计划，而是分阶段运行、保存数据、记录误差并形成初步论文的可审计证据。

**硬约束：只使用一个一维单电子表面发射 TDSE 模型。** 不需要服务器、COMSOL、FDTD、真实钨的三维电子结构、全电子 TDDFT、多体 QED 或完整的等离激元量子化。用户将用 Origin 进行论文正式作图：程序必须先交付 CSV 数值数据及其物理定义，图像只是调试预览。

**优先顺序：物理正确性 > 可复现性 > 节省运行时间 > GPU 加速 > 图像外观。** 不得为了得到“量子优势”而调整模型或选择性删数据。

## 1. 研究现状与适宜的论文问题

2024 年 Heimerl 等在 Nature Physics 报告非经典光驱动金属针尖的多光子电子发射及光子统计到电子统计的映射。2025 年 Heimerl 等在 Nature Physics 报告 BSV 驱动的金属针尖强场电子发射、再散射能谱和后选择；其理论使用一维单电子 TDSE，并按 BSV Husimi 强度权重平均相干光能谱。2025 年 Hergert 等在 ACS Photonics 使用明确的一维有限方势阱模拟钨针尖电子发射。这些构成直接文献基础，**故本项目不能声称首次研究量子光驱动金属针尖，不能把“相干光与 BSV 有不同发射产额”本身包装为全新发现**。

**候选论文题目：** *Photon-Statistics-Dependent Multiphoton Photoemission from a Metallic Nanotip: A One-Dimensional Model Study*。

**最小科学问题：** 对于固定的钨针尖有效势阱，输入相干光、经典热光和明亮压缩真空近似的统计涨落如何影响电子发射概率？随入射波长改变最低多光子阶数（3、4、5 光子附近），以及随近场衰减长度改变时，微扰区统计矩预测在何处失效？可否提供可信的一维 TDSE 趋势，而不仅是复述普适阶乘矩因子？

文章定位应为**“量子光驱动统计的有效随机场/半经典 TDSE 研究”**，不是完整联合光—电子态演化。即使 BSV 具有非经典性质，这种正的强度权重加权方法本身也不能证明任何结果排除了经典随机场解释。热光是经典统计对照；如果不同经典光源拥有同样的完整强度/相位分布，本模型将给出相同预测，必须明确此限制。

## 2. 需要先阅读的论文（用户会提供本地 PDF）

**必读 A：** J. Heimerl et al., *Multiphoton electron emission with non-classical light*, Nature Physics **20**, 945–950 (2024), DOI: https://doi.org/10.1038/s41567-024-02472-6 。阅读多光子统计映射及其适用区间。

**必读 B：** J. Heimerl et al., *Quantum light drives electrons strongly at metal needle tips*, Nature Physics **21**, 1899–1904 (2025), DOI: https://doi.org/10.1038/s41567-025-03087-1 。特别检查 Methods 中的 1D TDSE、近场函数、CEP、BSV Husimi 强度分布、统计平均公式，以及 Supplementary/Extended Data 和 Source Data。原始文章提供可下载的相关 Source Data（若下载，记录官方 URL、文件名、下载日期和校验和），可用于近似定性验证，但不要把不同参数下的曲线硬性要求重合。

**必读 C：** G. Hergert, R. Lampe, C. Lienau, *Quenching Strong-Field Rescattering of Photoemitted Electrons from Metallic Nanotapers by Using Moderate Bias Fields*, ACS Photonics **12**, 2219–2225 (2025), DOI: https://doi.org/10.1021/acsphotonics.5c00207 。其正文描述宽度 0.135 nm、深度 10.8 eV 的一维有限势阱及光学近场函数。先核对其 Supporting Information 对数值边界、初态和能谱的具体方法，记录是否有额外静电场及其符号。不得把本文 1950 nm、静电偏压实验参数和另一篇 1600 nm 量子光实验参数混成“同一实验”。

**背景参考（用户上传）：** 《量子光注入金属纳米尖端的背景调查与可行性评估.docx》。这是一份调研综述，不可替代原始文献。与本项目不相干的 Au/Ag SPP 传输、单光子保真度、三维 FEM 参数扫描不实施。

**先生成：** `docs/literature_audit.md`，逐篇列出“原文明确给出 / 我们主动简化 / 尚待确认”三栏。任何未能从原文核实的具体参数必须标为 provisional，不得伪称论文原值。

## 3. 唯一物理模型与规范

设表面在 x=0，金属有效势阱位于 [-a,0]，右侧 x>0 为真空发射方向。首版先取无静态偏压：

\[
i\hbar\partial_t\psi(x,t)=\left[-\frac{\hbar^2}{2m_e}\partial_x^2+V_s(x)+U_L(x,t)-iW_{\rm CAP}(x)\right]\psi(x,t),
\]

\[
V_s(x)=\begin{cases}-10.8\ \mathrm{eV}, & -0.135\ \mathrm{nm}\le x\le0,\\0,&\text{其他。}\end{cases}
\]

无场应只有一个主要束缚态，能量接近 -5.6 eV（精确值取决于模型边界和离散；不要把 -5.6 当作必须精确匹配到小数点后的数）。先通过无场定态方程验证。

建议入射电场：
\[
E_{\rm in}(t)=E_0\exp[-2\ln2\,t^2/\tau_I^2]\cos(\omega t+\phi),
\]
其中 \(\tau_I\) 定义为**强度包络 FWHM**，不是场幅包络 FWHM；\(E_0=\sqrt{2I_{\rm inc}/(c\epsilon_0)}\)；请在代码和输出元数据写明。近场函数：
\[
\xi(x)=\begin{cases}0,&x<0,\\1+(\beta-1)e^{-x/\ell},&x\ge0.\end{cases}
\qquad E(x,t)=\xi(x)E_{\rm in}(t).
\]

在长度规范使用**从空间电场积分得到**的连续标量势，而非误用 \(e xE(x,t)\)：
\[
U_L(x,t)=e\int_0^x E(s,t)\,ds
=eE_{\rm in}(t)\left[x+(\beta-1)\ell(1-e^{-x/\ell})\right],\quad x\ge0;
\]
\(U_L=0\) 对 \(x<0\)。此处 \(e>0\) 为元电荷，故电子受力 \(-\partial_x U_L=-eE\)。明确标记长度规范、参考零点、国际单位与原子单位转换；如采用其他规范，需给出一致性推导与数值核对。

首轮参数：W 有效表面，\(\lambda=800\) nm、\(\tau_I=15\) fs、\(\beta=3\)、\(\ell=20\) nm、\(E_{\rm dc}=0\)、\(\phi=0,\pi\) 初始检查。入射峰值强度在 \(10^{10}\)–\(10^{12}\) W/cm² 进行试探，在数值可信且不饱和的范围细化。\(I_{\rm apex}=\beta^2 I_{\rm inc}\) 对场峰值成立；**入射强度与尖端局域强度必须分别记录，不得重复乘场增强因子**。800 nm 光子能量约 1.55 eV，束缚能约 5.5–5.6 eV，所以最低微扰通道预计四光子；不能保证所有强度下对数斜率严格为 4（可能有频道关闭、带宽、非微扰及数值噪声）。

补充注意：这一“一个束缚态”的表面势阱没有完整金属费米海，不可直接输出实验绝对电子数或声称材料级定量预测。没有静态偏压时，最终能谱是发射后相对于真空阈值的**过剩动能谱**，不要混同有静态加速场的探测器动能。

## 4. 数值算法与严格执行次序

### Stage 0 — 本地环境与论文审计

1. 检查 OS、CPU、内存、Python、Mathematica/WolframScript、Origin 版本；运行 `nvidia-smi` 记录 GPU 设备名、驱动、显存大小。RTX 5060 Ti 有不同显存配置，**必须实测**。
2. 创建独立 Python 环境，安装 NumPy、SciPy、pandas、pytest、PyYAML 等 CPU 基础依赖；先不为 GPU 更换全部环境。
3. 将原文 PDF 和 SI 放入 `references/`（遵守获取权限）；按节核对参数与数值方法。落实 `.gitignore`，不把大型论文 PDF 未经许可重新公开。
4. 写出 `docs/model_conventions.md`：电场方向、势符号、CEP、脉宽、强度定义、原子单位、吸收边界、规范及其量纲检查。

### Stage 1 — 定态与自由传播基线

1. Python 用 `scipy.sparse.linalg.eigsh`（或可靠稀疏定态求解）求势阱初始束缚态；另用 Mathematica `NDEigensystem` 或一维有限方势阱的解析超越方程独立核对。由于狭窄方势阱，检查 \(\Delta x\) 和端点表示对束缚能的影响。
2. 无场且无 CAP 参与区域应满足 \(\|\psi(t)\|_2^2=1\)；初始定态密度基本不变化，记录相位演化误差。
3. 自由高斯波包传播需通过解析展宽或收敛解校验 FFT、网格、时间步、动量标定。可参考 qmsolve 算法，但主代码由我们自己的可审计模块组成。

### Stage 2 — 一维 TDSE CPU 求解器

1. 优先使用 Strang split-operator FFT：势半步 → FFT 动能整步 → 势半步，时间依赖势取 \(t+\Delta t/2\)。FFT 需明确 \(k\) 网格、\(\hbar^2k^2/(2m)\) 和归一化。
2. 吸收层可用 \(-iW_{\rm CAP}\) 或吸收掩膜，分别设置左右区域。**电子发射量只能统计真空右侧**，不得把左侧吸收当作发射概率。
3. 建立一致的发射产额定义，例如真空侧远处净流量积分与后脉冲右侧概率/右 CAP 吸收量交叉核验。记录所有电荷流出及概率守恒预算；避免把发射后回到尖端的再散射暂时流量算成永久电子发射。
4. 选做电子过剩动能谱 \(S(E;I)\)：在场消失后提取右向出射波包，采用窗口化 Fourier/动量投影，并按 \(E=\hbar^2k^2/(2m)\) 加 Jacobian 转换为 \(dP/dE\)。检查能谱积分与总出射概率一致；若这一提取不可信，先不画能谱，不伪造截止。
5. 建议起始数值网格（仅供试跑，非固定答案）：\(x_{\min}\approx-20\) nm、\(x_{\max}\approx140\) nm、\(N_x=32768\)，\(\Delta x\approx0.0049\) nm；\(\Delta t\approx0.15\)–0.25 a.u.，整个脉冲前后均保留足够传播时间。根据最大电子能量、真空波包行程及 CAP 反射检查，再调整边界和网格。至少完成 \(N_x\)、\(\Delta t\)、盒长及 CAP 参数四类收敛性检查。
6. 先计算 \(I_{\rm inc}\) 对 \(Y(I)\) 的关系，检查在可解析/数值收敛的弱场区是否接近四光子幂律。不许对数值噪声底以下的微小产额做斜率拟合。

### Stage 3 — 光统计平均（不重新更换电子模型）

定义 CPU TDSE 的可复用单强度响应库 \(Y(I,\phi)\) 和 \(S(E;I,\phi)\)，对每种输入保持**相同的平均入射强度、脉冲包络、波长、近场模型和明确的 CEP 采样规则**。对 2025 年论文的 BSV 强压缩单模近似：

\[
p_{\rm BSV}(I;\mu)=\frac{1}{\sqrt{2\pi\mu I}}\exp[-I/(2\mu)],\quad I>0.
\]

相干态的大光子数窄分布近似：\(p_{\rm coh}(I)=\delta(I-\mu)\)。单模热光的经典强度分布：\(p_{\rm th}(I)=\mu^{-1}\exp(-I/\mu)\)。定义 \(\mu=\langle I\rangle\)；不要把这些强度分布当成有限光子数下的精确 Glauber P、Q 或 Wigner 等价物。BSV 分布仅在强压缩、明亮且与原文测量/伪相干态近似相容的条件下采用；不能将它滥用于少光子 BSV、压缩角测量或量子相干断言。

\[
\overline{Y}_j(\mu)=\int_0^\infty p_j(I;\mu)Y(I)\,dI,\quad
\overline{S}_j(E;\mu)=\int_0^\infty p_j(I;\mu)S(E;I)\,dI.
\]

**优先确定性求积而非大量随机蒙特卡洛：** BSV 可令 \(I=\mu Z^2\)、\(Z\sim\mathcal N(0,1)\)，用 Gauss–Hermite 求积；热光可令 \(I=\mu u\)、\(u\sim\mathrm{Exp}(1)\)，用 Gauss–Laguerre 求积。先验证权重归一化、\(\langle I\rangle=\mu\) 与 \(\langle I^n\rangle/\mu^n\)（n=2,3,4）。对于同一个目标均值，用足够范围的 TDSE 强度响应库及受控插值；积分尾部强度超出库边界时**必须自适应补充求解，不能暗中截断并重归一化**。

理论测试：对于 \(Y(I)\propto I^4\)，经典固定强度/热光/上述明亮 BSV 近似的相对因子分别为 \(1/24/105\)；这是**输入统计的数学测试，不是本文创新**。真正单模有限光子数压缩真空的正规序四阶阶乘矩与这一大光子数极限不完全相同，需要另行标注，不得混用。

关于相位：BSV 近似为沿反压缩正交分量的随机振幅，正负号与相位 \(0,\pi\) 有关；热光具有随机相位。若一维表面模型表现 CEP 敏感，必须对不同态使用物理上正确的 CEP/相位协议并检查 2、4、8 个相位点收敛；不能机械地只用正振幅或统一固定 CEP 后宣称完整光态预测。

### Stage 4 — 最小非平凡扫描

1. **主线先只算 800 nm、\(\beta=3\)、\(\ell=20\) nm。** 先得到约 30–60 个按需自适应强度点（数量由误差和运行成本确定），构建 \(Y(I)\) 库、三种态平均响应及完整 CSV。
2. 基线通过后只增加代表波长 \(\lambda=620,800,1000\) nm。以束缚能约 5.5–5.6 eV 估计，分别对应最低 3、4、5 光子微扰发射通道。注意临界阈值、脉冲带宽和 AC Stark/ponderomotive 效应可能改变有效阶数。
3. 只选择少数有代表性的强度点，扫描 \(\ell=10,20,40\) nm。\(\beta\) 先固定 3。若另扫描 \(\beta\)，分别做“固定入射强度”和“固定 apex 峰值强度”的比较，以隔离单纯强度放大和空间衰减效应。
4. 关注 \(R_{\rm th}=\overline{Y}_{\rm th}/\overline{Y}_{\rm coh}\)、\(R_{\rm BSV}=\overline{Y}_{\rm BSV}/\overline{Y}_{\rm coh}\)、局域有效斜率 \(n_{\rm eff}=d\ln Y/d\ln I\) 及能谱。只在分母远离数值噪声底时计算增强比。
5. 一旦发现结果只等于 3!/4!/5! 等普适统计矩关系，没有超出已有文献的针尖相关新趋势，必须在报告中如实说明，并提出最小修正；不得靠过度解释包装成新机制。

## 5. 软件与 GitHub（以官方来源为准）

- **Python / NumPy / SciPy：正式生产工具。** `scipy.sparse.linalg.eigsh` 处理初态；`scipy.fft` 或 `numpy.fft` 处理传播；`pytest` 用于可重复测试。
- **CuPy：首选可选 GPU 后端。** 官方 https://docs.cupy.dev/en/stable/install.html 、https://docs.cupy.dev/en/stable/user_guide/fft.html ；仓库 https://github.com/cupy/cupy 。Windows 有 CUDA 12.x/13.x 预编译 wheel（如 `cupy-cuda12x`），但安装前先核查本机驱动、Python、CuPy 与 CUDA 版本。不要同时安装多个冲突的 CuPy 包。
- **qmsolve：算法参考/独立验证而非核心依赖。** https://github.com/quantum-visualizations/qmsolve ；具有 split-step、Crank–Nicolson、CuPy 实现，仓库标明 BSD-3-Clause。其通用求解器并非“已验证的金属针尖发射程序”；如复用任何代码，必须保留许可证与出处，且重新核查边界、单位、近场和光电子提取。
- **Mathematica：独立审核。** https://reference.wolfram.com/language/ref/NDEigensystem.html 和 https://reference.wolfram.com/language/ref/program/wolframscript.html.en 。用于独立求有限方阱能级、解析多光子统计矩、求积归一化、单位换算。可写 `validation/finite_well_check.wl` 和 `validation/statistics_check.wl`；不要求 Mathematica 承担整个生产级 TDSE。
- **Origin：用户进行最终科学作图。** 官方 CSV 导入指南 https://docs.originlab.com/user-guide/import-data/ 。Python 负责产出原始 CSV、HDF5/NPZ 与调试图；Origin 负责定稿图、标注、拟合版式。禁止仅提供 PNG 而不提供可重新制图的原始点阵。
- **PyTorch：备选，而非默认。** 如 CuPy 安装或运行失败且有明确性能收益，再考虑与 RTX 5060 Ti 的 Blackwell `sm_120` 兼容的 CUDA 12.8 及以上对应构建；安装前现场验证 `torch.cuda.is_available()`、`get_device_capability()` 和 complex128 FFT。不要因主流深度学习生态而强行把 TDSE 改写为神经网络。

**GPU 性能策略：** 首先测量 CPU 的单条 TDSE 运行时间与内存；安装 CuPy 后在相同双精度 `complex128`、相同网格/时间步/初态上做 3–5 条实际 TDSE 对照，分别计时、检查误差和 GPU 峰值显存。先预热 FFT plan，计时前后同步 GPU，避免 CPU/GPU 往返数据复制。只有在可接受数值误差和实际加速比下才转用 GPU。若 GPU FP64 性能差、显存不足或速度反而慢，继续 CPU；禁止把“启用 GPU”当作科研成就。

## 6. 必须保存的数据与 Origin 接口

每个独立参数组合都要可追溯。每个 CSV 仅一行英文列名、UTF-8 编码、点号小数、逗号分隔；第一列 X，自后依次为 Y 或误差列。所有物理单位写在列名或配套 `metadata.json`，CSV 中不要混入中文说明行，以免 Origin 导入不稳。

**建议文件：**

1. `data/origin/Fig1_potential.csv`：`x_nm, V_surface_eV, psi0_abs2_per_nm, E_enhancement`。
2. `data/origin/Fig2_yield_coherent.csv`：`I_inc_W_cm2, I_apex_W_cm2, yield_coh, numeric_err, local_slope`。
3. `data/origin/Fig3_statistics_yield.csv`：`I_mean_inc_W_cm2, yield_coh, yield_thermal, yield_BSV, R_thermal, R_BSV, integration_err`。
4. `data/origin/Fig4_wavelength_scan.csv`：`lambda_nm, photon_eV, n_min, I_mean_inc_W_cm2, ell_nm, yield_coh, yield_thermal, yield_BSV, R_BSV`；长表格式适合 Origin 分组/透视，另导出每条曲线的独立宽表 CSV。
5. `data/origin/Fig5_spectra_long.csv`：`E_excess_eV, lambda_nm, I_mean_inc_W_cm2, state, dP_dE_per_eV`（若谱已通过提取验证）；另提供 `Fig5_spectra_wide.csv`，同一能量轴每列一个条件，供 Origin 一键作图。
6. `data/origin/convergence.csv`：`case_id, nx, dx_nm, dt_au, x_max_nm, cap_start_nm, yield, relative_delta, norm_budget_error`。
7. `data/raw/` 保存不经归一化改写的中间数值和必要 checkpoint（压缩 NPZ/HDF5），以及 `data/manifest.csv`（run_id、时间、配置 hash、代码 commit、文件路径、计算状态）。运行时仅为调试可以自动导出小尺寸 PNG，不得代替原始数据。

所有批量计算都自动保存 `configs/*.yaml`、`logs/*.log`、`results/summary.csv`、依赖 lockfile、Git commit、GPU/CPU 信息、随机种子（如使用 MC）。谱图须同时保存**绝对概率密度**和**只为比较形状而归一化的版本**，且文件名不同，不可用归一化谱冒充电子产额。

## 7. 验收标准与停机规则

**Gate A（定态与自由传播）：** 初态正确、归一化、无场传播稳定；用 Mathematica 或解析有限阱交叉验证束缚能，明确差异。未通过不可开始批量 TDSE。

**Gate B（单强度 TDSE）：** 网格、时间步、盒长、CAP 均有对照；无场发射近零（量化噪声底）；发射概率非负、不大于 1，左右概率预算合理；数值稳定区域内的主要物理量以约 5% 相对误差为初始收敛目标，更靠近阈值/极小产额时须采取更适合的绝对误差判断。未通过不能拟合多光子斜率。

**Gate C（统计积分）：** 三种分布均满足归一化与平均强度；热光矩 2!/3!/4!，明亮 BSV 近似矩 3/15/105；求积阶数提高后平均产额和比值稳定；不允许未覆盖尾部强度造成虚假增强。CEP 规则单独核查。

**Gate D（正式结果）：** 800 nm 主图通过后才开展 620、1000 nm 与有限的近场扫描；统计比值明确区分入射/局域强度和数值误差；至少一张非微扰趋势图可与解析弱场极限接续，或如实给出否定结论。

**科学边界：** 不能证明量子优势、量子光相干性被电子探测到、真实 W 能带预测、绝对电子计数，也不能将纯粹入射统计矩增强称为纳米针尖特有的发现。若期刊论文无法从已有 2024–2025 工作中分出可信增量，先提交“未达到发表标准”的审计结论及最小建议，不得生成虚构图线。

## 8. 项目目录和 Agent 汇报机制

建议目录：

```text
QL-TIP-1D/
  README.md
  references/                 # 用户提供的原文，谨慎管理版权
  docs/                       # literature_audit, model_conventions, numerical_methods, research_notes
  configs/                    # yaml 参数与扫描批次
  src/ql_tip_1d/              # potentials, pulse, propagator, observables, statistics, io, gpu_backend
  validation/                 # Mathematica .wl + analytic checks
  tests/                      # pytest、回归基准
  scripts/                    # 单次运行、扫描、导出
  data/raw/                   # 原始 npz/hdf5
  data/origin/                # Origin-ready CSV
  data/manifest.csv
  logs/
  reports/                    # 每阶段审核说明、最终论文证据清单
  figures_preview/            # 仅调试预览图
  requirements.txt            # 或 lockfile，固定依赖版本
```

每完成一个 Stage，提交一个 `reports/STAGE_X.md`，写出：已执行内容、运行命令、实际数据路径、关键数值、收敛性、GPU 性能、原文对应位置、尚存问题、下一步。不要只说“代码已实现”而不展示输出和测试。遇到无法修复的错误应保留失败日志、停止生产计算，不得跳过验收关卡。

**第一轮就开始做：** (1) 读取并索引原文与附件；(2) 诊断本地环境；(3) 建立项目目录；(4) 定态有限阱求解与 Mathematica 交叉检查；(5) 运行短时间无场/自由波包传播测试。以上全部通过后，再开始 800 nm 相干光电子发射曲线。不要一口气启动全部波长和统计模型的大型计算。

## 9. 本任务书核验过的外部地址

- Heimerl et al. 2024: https://doi.org/10.1038/s41567-024-02472-6
- Heimerl et al. 2025: https://doi.org/10.1038/s41567-025-03087-1
- Hergert et al. 2025: https://doi.org/10.1021/acsphotonics.5c00207
- qmsolve: https://github.com/quantum-visualizations/qmsolve
- CuPy 安装: https://docs.cupy.dev/en/stable/install.html
- CuPy FFT: https://docs.cupy.dev/en/stable/user_guide/fft.html
- NVIDIA GPU compute capability: https://developer.nvidia.com/cuda/gpus
- SciPy `eigsh`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
- Wolfram `NDEigensystem`: https://reference.wolfram.com/language/ref/NDEigensystem.html
- WolframScript: https://reference.wolfram.com/language/ref/program/wolframscript.html.en
- Origin CSV 导入: https://docs.originlab.com/user-guide/import-data/

请按“最简单的可信模型 → 数值审计 → 统计平均 → 少量物理扫描 → Origin 原始数据 → 论文”执行。项目第一阶段不增加新的模型，不因有 GPU 而引入不必要的复杂软件栈。
