# 当前研究入口（2026-10-30，2026-11-03 续；协议 A 精确 QED 独立留出已完成；Q1 三窗独立拟合通过；Q3 最小 Fock 核已重构；不开 Q2）

`docs/README.md` 仍是历史目录清单，**从今天起先读本文件**。旧入口停在 2026-09-02，已经不能代表线 Q 的 Q6 / A/B/C 收口。

---

## 0. 2026-11-03 续：协议 A 精确 QED 独立留出已完成；Q1 三窗独立拟合通过；Q3 核重构完成；仍不开 Q2

| 新文件 | 用途 |
|---|---|
| `docs/04_BSV量子光/线Q_Q6审计与1600nm桥接设计_20261102.md` | Q6 完成度、A/B/C 与 \(\Delta n\)、\(g\) 转不成 W/cm²、24.8 μm vs 1600 nm、协议 A/B/C、文献表、论文判断、量子理论层次 |
| `docs/04_BSV量子光/线Q_三问与量子理论层次_一页_20261102.md` | 给自己和 Cohen 前的白话三问 + 量子层次 |
| `docs/04_BSV量子光/线Q_1600nm_协议A_GH5冒烟_20261102.md` | 1600 nm 协议 A runner 与冒烟通过记录 |
| `docs/04_BSV量子光/线Q_1600nm_协议A_GH5_20261102.md` | 协议 A GH5 四角 + \(\pi/4\) 留出的正式结论（旧口径） |
| `tools/b0_small_model/q8_m0_qinput_1600nm_protocolA_gh5_20261102.py` | 协议 A GH5 入口：`--resources-only` / `--mode smoke\|fourangle\|pi4` |
| `results/q_line/qinput_1600nm_protocolA_gh5_20261102/` | 协议 A 结果目录（勿与冻结 Q6 目录混用） |
| `tools/b0_small_model/q8_m0_qinput_1600nm_protocolA_qed_fockaudit_20261102.py` | 1600 nm 精确 QED Fock 截断审计：`--mode clarify\|input\|smoke\|evolve` |
| `results/q_line/qinput_1600nm_protocolA_qed_fockaudit_20261102/` | 精确 QED 截断审计目录；**不要**当四角机制结果 |
| `docs/04_BSV量子光/线Q_1600nm_协议A_QED截断审计_20261102.md` | \(n_{\max}=8\)、\(\theta=\pi/4\) 顶层审计正式结论 |
| `tools/b0_small_model/q8_m0_qinput_1600nm_protocolA_qed_fourangle_20261102.py` | 两磁畴十轴精确 QED runner；**默认拒绝开工** |
| `docs/04_BSV量子光/线Q_1600nm_协议A_QED四角执行指令_20261102.md` | 外审后的 10 轴执行契约；**本文件不是开工令** |
| `docs/04_BSV量子光/线Q_1600nm_协议A_QED四角runner只读审计_20261102.md` | 外审要求的开工前只读审计；**已经做完**。Hann 不收；freeze-fit 已实现 |
| `docs/04_BSV量子光/线Q_1600nm_协议A_QED四角job_row字段说明_20261102.md` | 只读审计里 `job_row` 多抄的状态字段；未改演化 |
| `results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102/` | 四角新目录；不要与 fockaudit 目录混用 |
| `docs/04_BSV量子光/线Q_1600nm_协议A_QED四角阶段一进度_20261102.md` | 八轴活记录；现为 8/8 |
| `docs/04_BSV量子光/线Q_1600nm_协议A_QED四角阶段一完成_20261102.md` | 阶段一闸门冻结；**不是** A/B/C |
| `docs/04_BSV量子光/线Q_1600nm_QED拟合与留出_执行计划_20261102.md` | 批次一/二执行计划：端点积分、x/二维分栏、runner 闸门 |
| `docs/04_BSV量子光/线Q_1600nm_协议A_QED四角拟合冻结_20261102.md` | 精确 QED 四角 A/B/C 冻结（历史：当时留出未做） |
| `docs/04_BSV量子光/线Q_1600nm_协议A_QED留出与对照_20261103.md` | 批次二 \(\pi/4\) 两磁畴留出与频率对照；主窗通过，邻窗依赖仍在，停机 |
| `docs/04_BSV量子光/线Q_1600nm_协议A_Q1三窗独立拟合_20261103.md` | Q1：三窗各自拟合 \(A,B,C\)，该窗自己的 \(\pi/4\) 留出全部通过；近反相三窗都在 |
| `tools/b0_small_model/q8_m0_qinput_1600nm_protocolA_qed_windowQ1_20261103.py` | Q1 后处理；不传播、不改写冻结拟合 |
| `results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102/analysis_windowQ1_20261103/` | Q1 JSON 与图；冻结 SHA 未改 |
| `docs/04_BSV量子光/线Q_Q3光子数相干读出核_数学起底_20261103.md` | Q3 数学起底：\(F_\ell\) ≠ 实验室 \(K_{\Delta n}\)；公式稿，不是数值核 |
| `docs/04_BSV量子光/线Q_Q3最小Fock探针_一页方案_20261103.md` | Q3 最小 Fock 探针一页方案；外审后已执行 8 轴 |
| `tools/b0_small_model/q8_m0_qinput_1600nm_protocolA_qed_q3fock_20261103.py` | Q3 8 轴精确 QED runner；Fock 探针，不是 \(D(\alpha)S(\xi)|0\rangle\) |
| `tools/b0_small_model/q8_m0_qinput_1600nm_protocolA_qed_q3fock_analyze_20261103.py` | Q3 核重构后处理；不传播、不改冻结拟合 |
| `results/q_line/qinput_1600nm_protocolA_qed_q3fock_20261103/` | Q3 八轴时间序列；不要与四角 / fockaudit 目录混用 |
| `docs/04_BSV量子光/线Q_Q3最小Fock探针_核重构_20261103.md` | Q3 \(\{K_{00},K_{22},K_{02},K_{20}\}\) 已重构；主比较 \(K_{22}-K_{00}\) vs \(K_{02}\) |
| `results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102/analysis_endpoint_v1/leaveout_validation.json` | 独立留出数字；冻结 SHA 未改；不要因乱码 verdict 改写 JSON |
| `results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102/analysis_endpoint_v1/comparison.json` | QED / 修正 GH5 / 24.8 μm 对照与边界 |
| `results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102/fourangle_fit.json` | 冻结拟合；SHA256 `17a8e37ec3e0ab54096ff4edd1440c5a3f0412b09450c2fd8960cbc00d8e69db` |
| `results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102/analysis_endpoint_v1/` | 端点修正谐波、修正 GH5 参考、源 SHA；未覆盖八轴或旧 GH5 JSON |
| `docs/00_当前状态/博士开题审阅_20261030.md` | 外面五课题开题的批评；可毕业合同是 1 篇机制 + 1 篇材料或 1600 nm，第三篇选修 |

本轮判断（冻结，不因外面 AI 改口）：

- 内部机制稿：**写**。仅 24.8 μm 投稿：**不**。
- **用户已下令计算。** 协议 A GH5（1600 nm、同一 \(R\)、抽象 \(g=0.005\)）**已经跑完**。不是构想 A 的升沿 \(\Delta J_{\mathrm{odd}}\)，也不是 20261028 未认证统计 SBE。
- \(g\) **不能**换成 W/cm²；禁止擅自 \(g\propto\omega^{-1/2}\)。
- 不要把 4.98 / 110.6° 抄到 1600 nm。旧 GH5 的 8.778 是漏端点 \(x\) 比，也**不要**再当生产参考。
- GH5 ≠ Cohen MTEF。
- 主读出仍是 \(\langle\hat J\rangle\)。纠缠 / \(g^{(2)}\) / 纯 BSV 不抢在协议 A 前面。
- Cohen 材料**已经发出**，本轮不再做 briefing / PPT / 开题。
- 不 `git add .`，不推 public origin。不删 `.venv_q8_cuda`。

协议 A 进度（2026-11-02）：

- `--resources-only`：`omega0=0.775`，\(T_0=5.336\,\mathrm{fs}\)，1+4+1 \(\approx 32\,\mathrm{fs}\)，\(R=1.273\)，`used_pf_OMEGA0=False`。
- `--mode smoke`：`smoke_p_0`/`smoke_m_0`，800 步，**通过**。
- GH5 `--mode fourangle`：8 路 `n_steps=11154`，**完成**。旧口径（漏端点 `rect_jq`，仅 \(x\)）振幅比 \(\pi/0=8.778\)，\(\rho=1.274\)，\(\arg(A/B)=179.04^\circ\)。**历史口径。**
- GH5 `--mode pi4`：旧口径独立留出相对误差 **0.0115%**（漏端点，历史）。批次二同口径修正后 GH5 主窗 \(E_{\mathrm{sym}}=0.0110\%\)。
- 邻窗 \(|H2_{\mathrm{odd}}(\theta=\pi)|\) 仍在变（旧口径：主 \(8.72\times10^{-5}\)，前 \(9.25\times10^{-5}\)，后 \(9.53\times10^{-5}\)）。
- \(\pi/4\) 外层 Weyl \(n\approx 7.91\)。**Weyl \(\neq\) Fock \(\langle\hat a^\dagger\hat a\rangle\)**，公式是 \(0.25(X_f^2+P_f^2)-0.5\)。不得用 7.91 改锁定矩阵。
- 精确 QED 截断审计 **已通过**（仅 \(\theta=\pi/4\)，Néel \(=+1\)）：`n8_p_q` 11154 步，\(t/T_0=6\)，\(\max_t P_8=2.659\times 10^{-5}\)，\(\max_t\langle n\rangle=0.34284\)，`fock_alarm_0p05=false`。verdict=`n8_fock_truncation_pass_theta_pi4_only`。Weyl 7.91 **不是** Fock 数。`n8_p_q` **不能**当独立留出。
- 阶段一 **8/8 已通过**（2026-11-02 19:04）。最高 \(\max P_8=1.008\times10^{-4}\)（\(\theta=\pi\)），报警阈 0.05；\(\max\langle n\rangle\approx0.343\)；范数误差 \(\sim4\times10^{-13}\)。
- **批次一后处理已完成。** 端点修正 `linear_exact`；精确 QED 四角 A/B/C 已冻结。\(\rho=1.26737\)，\(\arg(A/B)=179.391^\circ\)（距 \(\pi\) \(0.609^\circ\)），\(x\) 振幅比 \(\pi/0=8.534\)，二维比 \(8.498\)。同口径修正 GH5：\(\rho=1.272\)，\(178.72^\circ\)，\(x\) 比 \(8.391\)。
- **批次二已完成。** `p_q`/`m_q` 均为 `status=ok`，11154 步，无 Fock 报警。主窗 \(E_{\mathrm{sym}}=0.0906\%\)（闸门 2%），\(\Delta\phi=-0.0021^\circ\)。前/后窗 \(\sim15\%\)，窗口依赖仍在。冻结 SHA 仍为 `17a8e37ec3e0ab54096ff4edd1440c5a3f0412b09450c2fd8960cbc00d8e69db`。
- **Q1 已完成（无新传播）。** 外审纠正：批次二邻窗 \(\sim15\%\) 是**主窗预测去比前后窗实测**。三窗各自拟合后，该窗 \(E_{\mathrm{sym}}(\pi/4)=0.095\%/0.091\%/0.098\%\) 全过；距 \(\pi\) 为 \(0.484^\circ/0.609^\circ/0.509^\circ\)；\(\rho=1.269/1.267/1.264\)。绝对 \(\lvert B\rvert,\lvert A\rvert\) 仍随窗变。冻结 SHA 未改。
- **现在停机，不开 Q2。** 不升 \(n_{\max}\)、不减 \(dt\)、不调 \(g\)、不开纯 BSV / 正 P / M52 / 纠缠，不 `git commit/push`。不要把 0.09% 写成物理精度，也不要把 8.534 抄成产额。
- **2026-11-03 外审后文档修正：** Q1 复系数表头 `C×10²` 已改为与 A、B 相同的 \(\times 10^{-5}\) 单位，数字未改；Q3 已按外审修正（实验室奇数 \(\Delta n\)、合法叠加探针、\(\rho\approx R\) 不推出 \(\lvert k_M\rvert\approx\lvert k_n\rvert\)）。
- **Q3 最小 Fock 核重构已完成（2026-11-03 16:42）。** 8/8 精确 QED 过闸，无 Fock 报警。已重构 \(\{K_{00},K_{22},K_{02},K_{20}\}\)。主比较是 \(\mathbf v_n=K_{22}-K_{00}\) 对 \(\mathbf v_c=K_{02}\)，不是 \(K_{00}\) 对 \(K_{02}\)。主窗 raw \(\phi=28.98^\circ\)，三窗都 \(\ge10^\circ\)；仅 H2 只有 \(11.40^\circ\)，通道权重敏感。2×2 子空间 \(\neq\) 量子优势。冻结 SHA 未改。**不要**因此开 Q2 或增加探针。

本仓库很脏，是因为大量报告、脚本和结果从未按主题提交，不是因为又开了一条新课题。整理原则：

- 不 `git add .`，不推送到目前为 **public** 的 origin。
- 不覆盖历史报告。
- 原始 Q6、Q-MATH、Wannier 数据和 1600 nm 抽检全部保留。
- 已冻结的 Gamma12 / \(C_{JJ}\) 原始数组已删除，JSON 摘要保留。

---

## 1. 现在课题实际停在哪里

| 线 | 状态 | 能写进论文的内容 | 还不能写 |
|---|---|---|---|
| **线 Q / 八带 M0** | Q6 机制候选已闭合；1600 nm 协议 A 精确 QED 主窗留出通过；Q1 三窗三项模型成立；Q3 \(\{|0\rangle,|2\rangle\}\) 核已重构；不开 Q2 | 位移压缩光调控 Néel-odd H2 有限时间复电流；A/B/C；\(R=\|M\|/\alpha^2\)；Néel/SOC 门控；有限时间核；1600 nm 精确 QED 上压缩角仍进入奇 H2；三项模型对独立 \(\pi/4\) 在前/主/后窗都通过；2×2 Fock 子空间四个复矩阵元 | 远场产额、量子优势、真实 CrI₃ 定量预测、稳态 \(\chi(\omega)\)、完整 \(K_{\Delta n}\) 主导性、\(n_{\max}\) 观测量收敛 |
| **Q-MATH** | 归档到 **16.30** | 完整 Peierls 双频因果核、正 P 输入界、若干 no-go | 真实出射 S 矩阵、输出纠缠证书 |
| **线 M / Wannier** | 材料检验未完成 | 1600 nm 相干光下 few-k 存在 Néel-odd H2 | M52 不是生产基准；full112 空间完备性未决；WINDOW144 有插值伪影 |

主结果协议必须每次说清楚：

\[
|\psi_\theta\rangle=D(0.5)S(0.3e^{i\theta})|0\rangle,\quad
\lambda=24.8\,\mu\mathrm{m},\quad
\langle n\rangle=0.343,\quad
R=1.273.
\]

这是低占据、深失谐、单 \(k/-k\) 对、1+4+1 耦合开关的精确 QED。**不是** 25 fs、宏观 BSV。1600 nm 协议 A 精确 QED：**主窗独立留出通过**；Q1：三项模型在三窗各自成立，近反相三窗都在；绝对核仍随窗变；**不是**实验产额或全 BZ。

冻结数字（禁止改写）：振幅比 **4.98**（有限窗，不是产额）；协方差反转 **110.6°**；邻窗 **16%–39%**；\(\rho=\|A\|/\|B\|=1.456\)；\(\arg(A/B)=177.65^\circ\)。

---

## 2. 今天该把哪几份文件交给人

### 给 Cohen 教授（先发英文）

| 文件 | 路径 |
|---|---|
| 英文 16 页 seminar PPT | `docs/04_BSV量子光/cohen_briefing_20261029/Cohen_seminar_Q6_mechanism_20261029.pptx` |
| 英文课题详解 | `docs/04_BSV量子光/cohen_briefing_20261029/Cohen_briefing_project_overview_20261029.md` |
| 中文详解（自己备忘） | `docs/04_BSV量子光/cohen_briefing_20261029/Cohen讨论_课题详解_中文_20261029.md` |

**不要**再发 `Downloads/汇报_量子输入磁序复响应_Q6版_20261024.pptx`，里面有已退役的 5.80 / 106.9°。

开场就说：24.8 μm、每模 0.34 个光子、位移压缩态，不是实验 BSV。请他评的是机制是否值得写成理论文，以及 \(R\) 是否是连到高亮度光源的正确控制量。

### 给导师评阅的科学稿（中文，尚未改成投稿 LaTeX）

| 文件 | 用途 |
|---|---|
| `docs/04_BSV量子光/线Q_外审后结论_20261029.md` | 当前科学结论与不同意外审的地方 |
| `docs/04_BSV量子光/线Q_论文骨架_20261029.md` | 论文结构、摘要草稿、边界 |
| `docs/04_BSV量子光/线Q_理论节草稿_20261029.md` | 理论节 |
| `docs/04_BSV量子光/线Q_论文图注_20261029.md` | Fig.1–8 图注 |
| `results/q_line/paper_figures_20261029/figures/` | 八张论文图 |

### 博士开题（本轮新写）

| 文件 | 用途 |
|---|---|
| `docs/00_当前状态/博士开题审阅_20261030.md` | 对外审开题稿的批评，以及可毕业的修订 |

---

## 3. 目录地图（只保留会用到的）

```text
Quantum-light/
├─ docs/00_当前状态/          本入口、开题审阅、清理记录
├─ docs/04_BSV量子光/         线 Q 报告、Cohen 材料、外审包
├─ docs/08_QE与Wannier/       线 M / full112 / WINDOW144
├─ docs/01_AFM选择定则_主线/  经典磁序 SHG 选择定则
├─ Q-MATH/                    数学档案，入口 README.md，最新 16.30
├─ tools/b0_small_model/      八带 M0、Q6、A/B/C 生产脚本
├─ results/q_line/            线 Q 数值；大数组已清，摘要还在
├─ tb_b1/, tb_m52/            大 Wannier 数据，磁盘保留、不进 git
├─ src/                       Fortran SBE
└─ deploy/                    服务器脚本
```

权威 Q6 / A/B/C 报告：

- `docs/04_BSV量子光/线Q_ABC微观机制_20261029.md`
- `docs/04_BSV量子光/线Q_双线协同后续_20261029.md`
- `Q-MATH/README.md`（16.30）

1600 nm：

- 协议 A GH5（已收口，旧口径）：`docs/04_BSV量子光/线Q_1600nm_协议A_GH5_20261102.md`，`results/q_line/qinput_1600nm_protocolA_gh5_20261102/`
- 协议 A 精确 QED 截断审计（\(\theta=\pi/4\) 已通过，不能当留出）：`docs/04_BSV量子光/线Q_1600nm_协议A_QED截断审计_20261102.md`，`results/q_line/qinput_1600nm_protocolA_qed_fockaudit_20261102/`
- 协议 A 精确 QED 四角阶段一（8/8 过闸）：`docs/04_BSV量子光/线Q_1600nm_协议A_QED四角阶段一完成_20261102.md`
- 协议 A 精确 QED 四角拟合冻结：`docs/04_BSV量子光/线Q_1600nm_协议A_QED四角拟合冻结_20261102.md`，`results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102/fourangle_fit.json`
- 协议 A 精确 QED 独立留出与对照（**主窗通过，停机**）：`docs/04_BSV量子光/线Q_1600nm_协议A_QED留出与对照_20261103.md`
- 协议 A Q1 三窗独立拟合（**该窗留出全过，不开 Q2**）：`docs/04_BSV量子光/线Q_1600nm_协议A_Q1三窗独立拟合_20261103.md`
- Q3 读出核数学起底：`docs/04_BSV量子光/线Q_Q3光子数相干读出核_数学起底_20261103.md`
- Q3 最小 Fock 核重构（**8/8 过闸，2×2 核已写出，不开 Q2**）：`docs/04_BSV量子光/线Q_Q3最小Fock探针_核重构_20261103.md`，`results/q_line/qinput_1600nm_protocolA_qed_q3fock_20261103/`
- 旧对照，**不是**本协议：`docs/04_BSV量子光/线Q_1600nm_BZ物理结论_修订_20261028.md`，`results/q_line/m52_1600nm_fewk_coherent_20261028/`，`results/q_line/b1_1600nm_model_compare_20261029/`

---

## 4. 明确冻结、不要再扫

- 不再扫 SOC / \(\Delta\) / \(n_{\max}=10\) / M52 加密网格。
- 不再把 80%/20% 能量分配当成唯一实验方案。
- 不再把 4.98 和 110.6° 抄到 1600 nm。
- 旧 Gamma12、全 BZ Fock QED、输出 \(g^{(2)}\) 大计算保持冻结，除非导师或 Cohen 明确要求打开其中一项小原型。

`pi_dev()` 已修好，`tools/b0_small_model/test_pi_dev_20261029.py` 覆盖 0°/90°/180°。归档 SOC JSON 不改写。

---

## 5. 磁盘清理摘要

详见 `仓库清理记录_20261030.md`。

| 操作 | 结果 |
|---|---|
| 删除冻结 Gamma12/CJJ 的 `.npy/.npz` | **47.5 GB**，JSON 保留 |
| 保留 | Q6/A/B/C、论文图、1600 nm 抽检、Wannier `tb_*`、Q-MATH |
| 暂不删 | `.venv_q8_cuda/`（4.6 GB，CUDA 环境；确认不再跑 Gamma12 后再删） |

`results/q_line` 从约 49 GB 降到 **0.53 GB**。
