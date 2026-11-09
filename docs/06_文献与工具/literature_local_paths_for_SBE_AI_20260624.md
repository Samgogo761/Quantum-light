# 量子光 CrI3 研究流文献本地位置索引（2026-06-24）

用途：给后续调整 `Quantum-light` SBE 求解器和撰写阶段汇报的本地 AI 使用。本文只做路径索引和用途分流，不直接修改 Zotero 数据库。

## Zotero 整理说明

- 我没有直接改 `zotero.sqlite`。当前可用 Zotero 工具只支持搜索/读取，不支持安全创建或移动 collection；直接改数据库风险较高。
- 建议在 Zotero GUI 中新建或整理 collection：`陆老师/博士课题/量子光HHG-CrI3`。
- 子 collection 建议分为：`SBE方法`、`磁群与非互易`、`量子光HHG`、`Gaussian-QFI-非经典性`、`开放系统-QRT-FCS`。
- Zotero 父 collection 默认不一定显示子 collection 项目。若希望点开 `陆老师` 就能看到核心论文，可把核心论文同时拖入父 collection 和子 collection。Zotero collection 是归类引用，不会复制附件。
- Downloads 中有重复文件：`PhysRevA.95.053825.pdf` 与 `PhysRevA.95.053825 (1).pdf`；`PhysRevLett.132.260402.pdf` 与 `PhysRevLett.132.260402 (1).pdf`。建议 Zotero 中只保留一份附件。

## 代码 AI 优先阅读顺序

1. `SBE-WG-2025`、`Yue-Gaarde-2020`、`Yue-Gaarde-2022`：先审查 SBE 方程、规范选择、退相干和 HHG 谱定义。
2. `Ilin-2026-topology`、`Stammer-2025-QO-coherence`：再设计双时电流关联、协方差矩阵和输出光统计导出。
3. `Xiao-2023-SHG-magnetic`、`Gao-2023-quantum-metric`、`PT-odd-even-2024`：用于 AFM-CrI3 的 N 宇称、磁群选择定则和非互易机制。
4. `Weedbrook-2012-RMP`、`Adesso-2014-CVQI`、`EvenOdd-QFI-2026`、`Ryl-2017-nonclassicality`：用于压缩、纠缠、QFI 与经典不可复现判据。

## 文献路径表

| ID | 文献与本地路径 | 主题 | 给代码/理论 AI 的用途 | Zotero 状态 |
|---|---|---|---|---|
| `SBE-WG-2025` | Thümmler et al., *Semiconductor Bloch equations in Wannier gauge with well-behaved dephasing*。`C:/Users/26507/Downloads/1-s2.0-S001046552500459X-main.pdf` | SBE 方法；Wannier gauge；退相干稳定性 | 检查当前 lg_cov 瞬时本征基退相干在近简并/带交叉处是否稳定，是否需要加入 Wannier-gauge dephasing 诊断。 | 未自动匹配到 Zotero storage。 |
| `Yue-Gaarde-2020` | Yue and Gaarde, *Structure gauges and laser gauges for the semiconductor Bloch equations in high-order harmonic generation in solids*。`C:/Users/26507/Downloads/PhysRevA.101.053411.pdf` | 长度规范/速度规范；结构 gauge 与 laser gauge | 解释为什么 velocity gauge 截断在本项目中失效，为什么 lg_cov 是生产路径；Peierls/full-band 可作为规范交叉验证。 | 未自动匹配到 Zotero storage。 |
| `Yue-Gaarde-2022` | Yue and Gaarde, *Introduction to theory of high-harmonic generation in solids: tutorial*。`C:/Users/26507/Downloads/josab-39-2-535.pdf` | SBE 教程；HHG 数值流程 | 对照运动方程、电流定义、window/FFT、弱场 scaling、k 网格收敛与退相干处理。 | 未自动匹配到 Zotero storage。 |
| `PT-odd-even-2024` | Cheong and Huang, *Emergent phenomena with broken parity-time symmetry: Odd-order versus even-order effects*。`C:/Users/26507/Downloads/PhysRevB.109.104413.pdf` | P/T/PT 对称性；奇偶阶响应 | 支撑双层 AFM-CrI3 中 P 破缺、T 破缺、PT 守恒与奇偶阶非线性响应的叙事。 | 未自动匹配到 Zotero storage。 |
| `Xiao-2023-SHG-magnetic` | Xiao et al., *Classification of second harmonic generation effect in magnetically ordered materials*。`C:/Users/26507/Downloads/s41535-023-00594-3.pdf` | 磁群选择定则；磁有序 SHG 分类 | 支撑 N-even/N-odd、i-type/c-type SHG 分解；说明选择定则检查为什么是 SBE debug 工具。 | 未自动匹配到 Zotero storage。 |
| `Ryl-2017-nonclassicality` | Ryl, Sperling, and Vogel, *Quantifying nonclassicality by characteristic functions*。`C:/Users/26507/Downloads/PhysRevA.95.053825.pdf` | 非经典性；特征函数；矩问题 | 把“经典不可复现”具体化为 characteristic function、Bochner positivity 和 moment matrix 判据。 | Downloads 有重复文件 `(1)`；未自动匹配到 Zotero storage。 |
| `Paulino-2024-FCS` | Paulino et al., *Large Deviation Full Counting Statistics in Adiabatic Open Quantum Dynamics*。`C:/Users/26507/Downloads/PhysRevLett.132.260402.pdf` | 开放系统；全计数统计；临界参数探测 | 支撑从双时电流关联走向 full counting statistics、临界响应和量子传感的路线。 | Downloads 有重复文件 `(1)`；未自动匹配到 Zotero storage。 |
| `Gao-2023-quantum-metric` | Gao et al., *Quantum metric nonlinear Hall effect in a topological antiferromagnetic heterostructure*。`C:/Users/26507/Downloads/science.adf1506.pdf` | PT-AFM；非互易输运；量子几何 | 支撑 PT 守恒时 Berry curvature 可被压制但 quantum metric/几何张量仍可进入非线性响应。 | 未自动匹配到 Zotero storage。 |
| `Phase-space-limit` | *Limitations of an approximative phase-space description in strong-field quantum optics*。`C:/Users/26507/Downloads/cxff-kky6.pdf` | 强场量子光；相空间近似局限 | 界定正 P、Wigner 或半经典随机平均何时会失败，支撑 L2 到 L3 的分界。 | 未自动匹配到 Zotero storage。 |
| `GM-Feist-2025` | González-Monge and Feist, *High-harmonic generation driven by temporal-mode quantum states of light*。`C:/Users/26507/Downloads/2512.06602v2.pdf` | temporal-mode 量子驱动 HHG；经典统计平均边界 | 定义 L2：量子驱动但可由经典强度分布平均复现的情形；本课题要说明 L3 如何超过它。 | 已匹配：`C:/Users/26507/Zotero/storage/CD844BRM/González-Monge和Feist - 2025 - High-harmonic generation driven by temporal-mode quantum states of light.pdf`。 |
| `Paris-2008-QET` | Paris, *Quantum Estimation for Quantum Technology*。`C:/Users/26507/Downloads/0804.2981v3.pdf` | 量子估计；QFI 基础 | 用于 QFI、Cramér-Rao bound、标准量子极限和量子增强传感的基础定义。 | 未自动匹配到 Zotero storage。 |
| `EvenOdd-QFI-2026` | *Even-Odd Splitting of the Gaussian Quantum Fisher Information: From Symplectic Geometry to Metrology*。`C:/Users/26507/Downloads/2601.06513v1.pdf` | Gaussian QFI；even-odd splitting；辛几何 | 把 `Cee/Coo/Ceo/Coe` 四通道与 Gaussian covariance/QFI 的偶奇分块联系起来。 | 未自动匹配到 Zotero storage。 |
| `Adesso-2014-CVQI` | Adesso, Ragy, and Lee, *Continuous variable quantum information: Gaussian states and beyond*。`C:/Users/26507/Downloads/1401.4679v3.pdf` | 连续变量量子信息 | 用于协方差矩阵、辛本征值、对数负性、Gaussian/non-Gaussian 边界。 | 未自动匹配到 Zotero storage。 |
| `Weedbrook-2012-RMP` | Weedbrook et al., *Gaussian quantum information*。`C:/Users/26507/Downloads/RevModPhys.84.621.pdf` | Gaussian QI 标准综述 | 用于协方差矩阵、压缩、纠缠和辛谱的标准定义。 | 未自动匹配到 Zotero storage。 |
| `QuantumLight-SymBreak-2026` | *Symmetry breaking by quantum light in solid-state high-harmonic generation*。`C:/Users/26507/Downloads/2605.28236v1.pdf` | 量子光破缺固体 HHG 对称性 | 说明量子光不仅改变强度平均，也可能改变对称选择和输出量子统计。 | 未自动匹配到 Zotero storage。 |
| `Lange-2024-nonclassical` | Lange, Hansen, and Madsen, *Electron-correlation-induced nonclassicality of light from high-order harmonic generation*。`C:/Users/26507/Downloads/PhysRevA.109.033110.pdf` | HHG 出射光非经典性；电子关联 | 说明出射 HHG 光的非经典性可以由物质电子关联或电流涨落诱导。 | 已匹配：`C:/Users/26507/Zotero/storage/UJVSJZ32/Lange 等 - 2024 - Electron-correlation-induced nonclassicality of light from high-order harmonic generation.pdf`。 |
| `Gothelf-2025-quantum-crystal` | Gothelf, Lange, and Madsen, *High-order harmonic generation in a crystal driven by quantum light*。`C:/Users/26507/Downloads/PhysRevA.111.063105.pdf` | 晶体 HHG + 量子光驱动 | 对照晶体中量子驱动 HHG 的输出统计和可观测量定义。 | 已匹配：`C:/Users/26507/Zotero/storage/SBL9D7UF/Gothelf 等 - 2025 - High-order harmonic generation in a crystal driven by quantum light.pdf`。 |
| `Ilin-2026-topology` | Ilin et al., *Quantum Optical Signatures of Band Topology in Solid-State High Harmonics*。`C:/Users/26507/Downloads/2604.20388v1.pdf` | 固体 HHG 输出光子统计；current-current fluctuation；covariance 模板 | 最高优先级给代码 AI。它给出从密度矩阵演化到 current-current fluctuation、出射谐波 covariance/压缩的实现模板；可类比把 SSH 换成 AFM-CrI3 最小模型。 | 未自动匹配到 Zotero storage。 |
| `Stammer-2025-QO-coherence` | *Theory of quantum optics and optical coherence in high harmonic generation*。`C:/Users/26507/Downloads/2504.13287v1.pdf` | HHG 量子光学；optical coherence；光场关联函数 | 定义输出光的 coherence、`g^(n)`、多模相关以及 HHG 中光场量子态表征。 | 未自动匹配到 Zotero storage。 |

## 给另一个本地 AI 的最小指令模板

请优先阅读 `SBE-WG-2025`、`Yue-Gaarde-2020`、`Yue-Gaarde-2022`、`Ilin-2026-topology` 四篇，按以下问题审查 `Quantum-light`：

- 当前 `lg_cov` 的运动方程、电流算符、FFT/HHG 谱定义是否与 Yue & Gaarde 的 SBE 规范一致。
- 当前退相干实现是否会在近简并、交叉或大带数 Wannier 基中引入非物理不连续；是否需要增加 Wannier-gauge dephasing 或至少输出诊断。
- 如何以最小代码改动增加双时连通电流关联 `C_conn(t,t')`：可先在 4 能带最小模型中实现辅助密度算符/quantum regression channel，再迁移到 112-WF。
- 输出接口建议：`J_mean(t)`、`C_ab(t,t')` 的低秩/频域版本、谐波模式 covariance matrix、压缩谱、对数负性、QFI 初版。
