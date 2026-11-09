# CrI₃ Bloch/Wannier 截断：场外 AI 发问的本地答复（2026-07-06）

> 对应场外文档：`CrI3_Wannier_Bloch_truncation_场外AI发问_20260706.md`  
> **核心结论**：当前代码的 `nb_start/nb_end` 是 **Bloch 本征带截断**，与师兄建议一致；**JDOS + f-sum 代理量已在本地 tb.dat 上跑通**（§16）；**40-WF Cr-d-only 试点**为当前 QE/Wannier 主任务（服务器 sbatch）。  
> **口径（2026-07-09）**：`nb1-104` **仅对 H1–H9 生产可信**；H10–H12 谨慎；H13+ / H16–H20 **勿用 nb1-104 宣称收敛**（见 Task4：`量子光研究/.../analysis/nb1-104_高阶可信区间_Task4_20260709.md`）。  
> **分工**：速度规范 PRB 预研 **暂搁置**；本文专注 **QE/Wannier + 40-WF + tb.dat 诊断**。

---

## 决策建议（A–E）

| 选项 | 建议 | 理由 |
|------|------|------|
| **A. 112-WF + nb1-104 生产** | ✅ **H1–H9 继续** | BSV 已跑通；H1–H9 与 full112 <3%；**高阶勿宣称全收敛** |
| **B. 对称窗口作诊断** | ✅ **已整理；不再扩扫** | W30–W60 已判废 |
| **C. 40-WF Cr-d-only 试点** | ▶️ **当前执行** | 新有效模型；`pilot_40wf/` + sbatch 流水线；**N_occ≠84** |
| **D. Cr-d + 部分 I-p** | ❌ 暂不 | 调试成本高；C 失败后再考虑 |
| **E. 112-WF Löwdin/downfolding** | 🔄 并行 R2 | 服务 Tier2/QRT，不替代 R1 |

---

## Q1. `nb_start/nb_end` 截断的是 Bloch 还是 Wannier？

**答：Bloch 本征带（band index）截断，不是 Wannier orbital 子块截断。**

### 源码流程

| 步骤 | 文件 | 函数 |
|------|------|------|
| 1. 读 full `nwann=112` 的 H(R), r(R) | `mod_wannier.f90` | `read_tb` |
| 2. 每 k 点 Fourier → `H^W(k)`，**全 112×112 对角化** | `mod_crystal.f90` | `compute_band_structure` → `zheevd` |
| 3. 取本征值/本征矢子集 | 同上 L61–62 | `eigvals(nb_start:nb_end)`, `eigvecs(:, nb_start:nb_end)` → `U_trunc` |
| 4. 光学矩阵：先在 **full Wannier** 建 H,D,P，再 `U_trunc` 投影到 **Bloch 子空间** | `mod_sbe.f90` | `precompute_lg_matrices` |
| 5. SBE 在 `n_trunc×n_trunc` **Bloch band space** 传播 | `mod_sbe.f90` | `propagate_lg_covariant` |

### 伪代码（与场外 AI 理想流程一致）

```fortran
! mod_crystal.f90
Hk = fourier_H(R -> k)           ! nwann × nwann Wannier representation
call zheevd(Hk, eigvals, U)      ! full 112 bands
Ek_active = eigvals(nb_start:nb_end)
U_active  = U(:, nb_start:nb_end)

! mod_sbe.f90 precompute_lg_matrices
H_full(k), D_full(k), P_full(k)  ! built in FULL nwann space from Hmn_R, rmn_R
H_B = U_active^H H_full U_active
D_B = U_active^H D_full U_active
P_B = U_active^H P_full U_active   ! length-gauge current operator in band basis
```

**不是** `H_W_active = H_W(nb_start:nb_end, nb_start:nb_end)`（后者才是危险的 Wannier orbital 截断）。

`project_to_trunc` 的实现是 `U^H M_W U`，即标准 Bloch 基变换。

---

## Q2. `lg_cov` 中 Berry / dipole / 协变导数如何实现？

| 量 | 生成方式 | 文件 |
|----|----------|------|
| `Dk_eq` (dipole) | full Wannier `rmn_R` → `D_full(k)` → `U^H D U` | `precompute_lg_matrices` |
| `Pk_eq` (current) | `P = ∂H/∂k - i[D,H]`（Wannier）→ 投影到 Bloch | 同上 |
| `Hk_eq` | 同上 | 同上 |
| **协变 k 导数** | `transport_density_to_k`：相邻 k 密度用 `S = U0^H U1` 平行输运 | `finite_diff_rho_parallel` |
| **E·∇_k ρ** | `add_lg_covariant_gradient`：分数坐标有限差分 + 电场系数 | `mod_sbe.f90` L1128+ |

**包含** 隐含 Berry 联络的平行输运（`U_trunc` 在相邻 k 间），**不是** 只在 active 子空间内重算 Wannier 矩阵。

**注意**：`i U^† ∂_k U` 未单独存为 `A_n` 数组，而是并入 `transport_density_to_k` 的 `S` 矩阵。

---

## Q3–Q4. 对称窗口表 + H1–H15 强度比

**条件**：`lg_cov`，`k40`，`T2_cycles=0.5`（对称窗来自 `output_lg_cov_scan_nodeph`；nb1-104/full112 来自 `output_lg_cov_fullvalence_T2_0p5cycle`）。

| 窗口 | band index | 价带 | 导带 | 能量范围 eV | 已有 HHG |
|------|------------|-----:|-----:|-------------|:--------:|
| **W20** | 75–94 | 10 | 10 | −1.28 ~ +2.59 | ❌ **未跑 lg_cov** |
| **W30** | 70–99 | 15 | 15 | −1.53 ~ +2.76 | ✅ |
| **W40** | 65–104 | 20 | 20 | −1.78 ~ +2.97 | ✅ |
| **W50** | 60–109 | 25 | 25 | −2.03 ~ +3.20 | ✅ |
| **W60** | 53–112 | 32 | 28 | −2.30 ~ +3.22 | ✅ |
| **nb1-104** | 1–104 | 84 | 20 | −4.93 ~ +2.97 | ✅ |
| **full112** | 1–112 | 84 | 28 | −4.93 ~ +3.22 | ✅ |

### H_q / H_q(full112)（T2=0.5, k40）

| q | W30 | W40 | W50 | W60 | nb1-104 | full112 绝对强度 |
|---:|---:|---:|---:|---:|---:|---:|
| H1 | 0.46 | 0.52 | 0.65 | 0.75 | **0.96** | 2.67e-2 |
| H2 | 0.92 | 1.74 | 1.76 | 1.40 | **1.01** | 1.92e-5 |
| H3 | 2.65 | 2.17 | 2.19 | 2.25 | **1.00** | 3.90e-4 |
| H4 | 4.26 | 8.12 | 6.23 | 6.27 | **1.00** | 4.78e-6 |
| H5 | 3.10 | 3.38 | 2.19 | 2.77 | **1.00** | 1.60e-6 |
| H6 | 14.8 | 9.80 | 25.8 | 28.1 | **0.99** | 1.18e-7 |
| H7 | 32.2 | 6.36 | 1.18 | 4.72 | **0.99** | 5.37e-7 |
| H8 | 3.99 | 10.9 | 6.47 | 0.81 | **1.01** | 5.42e-8 |
| H9 | 42.0 | 16.1 | 9.40 | 11.0 | **0.98** | 2.71e-8 |
| H10 | 44.0 | 72.8 | 61.9 | 33.5 | 1.03 | 5.23e-10 |
| H11 | 76.8 | 70.2 | 97.6 | 14.6 | 1.08 | 1.53e-9 |
| H12+ | 更大 | 更大 | 更大 | 更大 | H13+ 比值失真（full112 近地板） | — |

**解读**：

- **对称窗口 W30–W60 不能作生产**：砍掉深价带 → H3–H9 可偏 **2–40×**，偶次 H6 可达 **10–28×**。
- **nb1-104 在 H1–H11 稳定**（±3% 量级），与场外 AI「先 Bloch 截断、保留全价带」一致。
- **W40/W50 不能** 作为「费米面少带足够」的正面证据——只有 **nb1-104** 类全价带截断才行。

---

## Q5. 复数 Fourier 分量

**现状**：`HHG.dat` 只输出 `|J(ω)|²`（Hann 窗 FFT 后强度），**未存复振幅**。

**待做（Phase 1b）**：在 `mod_sbe.f90` 的 FFT 段增加 `J_x(ω), J_y(ω)` 复数导出，或后处理从 `Jt.dat` 重算。  
**优先级**：R2 / 量子光统计前必须；R1 强度谱可暂缓。

---

## Q6. ±N 选择定则 leakage

**已有**（`经典测试补充答复_20260701`，k40, T2=0.5，**强度比** leakage = 禁戒/允许）：

| H | 通道 | nb1-104 | full112 |
|---:|------|--------:|--------:|
| H2 | oddN 允许 | 0.35 | 0.32 |
| H4 | oddN | 0.04 | 0.04 |
| H6 | oddN | 0.88 | 0.88 |
| H8 | oddN | 0.13 | 0.13 |

**对称窗口**：未系统做 ±N 测试 → **待跑**（可用 `deploy/run_lg_cov_scan.sh` 改 CASE + P 共轭 TB）。

---

## Q7. 零场 baseline

| 窗口 | \|J(0)\| (T2=0.5, k40) | H1 异常？ |
|------|------------------------:|-----------|
| full112 | 4.40e-18 | 否 |
| nb1-104 | 4.41e-18 | 否 |
| nb1-94 | 4.41e-18 | 否 |
| nb10-104 | **1.10e-7** | 是（截价带） |
| W30–W60 | 未在同一表汇总 | 对称窗 H1 仅达 full112 的 46–75% |

---

## Q8. 光子能量与谐波能量

```text
λ = 3200 nm  →  ℏω₀ ≈ 0.387 eV  (代码用原子单位，文档口径 0.38745 eV)
```

| q | q·ℏω₀ (eV) |
|---:|---:|
| H1 | 0.39 |
| H3 | 1.16 |
| H5 | 1.94 |
| H7 | 2.71 |
| H9 | 3.49 |
| H11 | 4.26 |
| H13 | 5.04 |
| H15 | 5.81 |

被删导带 bands 105–112：**+2.89 ~ +3.22 eV** → 与 H9–H11 能量尺度重叠，可解释 nb1-104 在 H13+ 的偏离。

---

## Q9. JDOS / transition histogram

**状态（2026-07-07）**：✅ **已完成**（读 `CrI3_tb.dat`，**不跑 SBE**）。

| 项 | 位置 |
|----|------|
| 脚本 | `量子光研究/01_输入与模型/wannier/tools/transition_jdos_fsum.py` |
| 报告 | `01_输入与模型/wannier/analysis/jdos_fsum_report_20260707.md` |
| 条件 | 24×24 k，ℏω₀=0.38745 eV，JDOS 能量窗 ±0.15 eV，H1–H20 |

**主要数值结论**（详见 §16）：

- **full112**：H1–H20 权重全在 **active 价带 → active 导带**（无 deleted 通道）。
- **nb1-104**：H1–H8 与 full112 **一致**；**H9 起** 出现 **价带(活)→导带(删)**（bands 105–112），H19–H20 的 active→active 权重大幅下降 → **高阶 HHG 应用 full112 或更多导带**，与 §Q8 能量尺度一致。
- **W40 / W30**：H6 起 **价带(删)→导带(活)** 主导 → 与 §Q3–Q4 HHG 失真机制一致，**不能作生产窗**。

---

## Q10–Q11. f-sum / TRK / velocity-dipole 残差

**状态（2026-07-07）**：**部分完成**（与 JDOS 同脚本中的 **f-sum 代理量**；非 SBE 内建、非逐 k 完整 TRK）。

| 窗口 | n_offdiag | mean \|iΔE·r_x\| | fsum_proxy Σ\|r\|² |
|------|----------:|-----------------:|-------------------:|
| full112 | 12432 | 2.163e-02 | 4.658e+01 |
| nb1-104 | 10712 | 2.087e-02 | 4.491e+01 |
| W40_65-104 | 1560 | 1.215e-02 | 3.401e+01 |
| W30_70-99 | 870 | 1.043e-02 | 3.117e+01 |

**解读**：nb1-104 与 full112 接近；对称窗明显偏低 → 与 HHG 表一致。

**仍待做**（可选 R2）：逐 k 检查 `v^a_mn ≈ (i/ℏ)(ε_m−ε_n) r^a_mn` 残差、TRK 与 SBE **速度规范** 求和规则闭环 — **在 Git 工作区 SBE 侧**（与 PRB 文章线同一 Agent），不在 tb.dat 脚本范围。

---

## Q12. intra/inter 分解

**已有**：`harmonic_intra_inter_decomp.py` + `Jt_decomposed.dat`（full112, +N）。  
**对称窗口**：未逐窗跑 → Phase 1c 可批处理现有 `output_lg_cov_scan_nodeph`。

---

## 可立即执行的任务清单（按优先级）

### Phase 0 — 无需新算例（今天可做）

| # | 任务 | 产出 | 状态 |
|---|------|------|------|
| 0.1 | 归档场外 AI 发问 + 本答复 | `08_QE与Wannier/` | ✅ |
| 0.2 | 汇总对称窗口 HHG 表（本文 §4） | 本文 | ✅ |
| 0.3 | 确认代码为 Bloch 截断 | 本文 Q1 + 源码引用 | ✅ |

### Phase 1 — 本地脚本 / 轻量算例（1–3 天）

| # | 任务 | 输入/脚本 | 依赖 |
|---|------|-----------|------|
| 1.1 | HHG 窗口对比自动化 | `tools/analysis/summarize_band_window_hhg.py` | 已有 HHG.dat |
| 1.2 | JDOS 四类通道直方图 | `wannier/tools/transition_jdos_fsum.py` | ✅ 2026-07-07 |
| 1.3 | f-sum / v–r **代理量** | 同上脚本 §2 | ✅ 代理量；完整 TRK → Git/SBE |
| 1.4 | 复数 J(ω) 导出 | 改 `mod_sbe.f90` + 单点测试 | 编译 |
| 1.5 | W20 (75–94) lg_cov 补跑 | 新 deploy case 或改 `run_lg_cov_scan.sh` | 服务器 |

### Phase 2 — 40-WF 试点（导师批准后，~1–2 周）

| # | 任务 | 说明 |
|---|------|------|
| 2.1 | 准备 `CrI3_donly.win` | 见 `01_输入与模型/wannier/pilot_40wf/` |
| 2.2 | **复用现有 nscf**（576 k, nbnd=200） | 只需 `pw2wannier90` + `wannier90`，**不必重跑 QE DFT** |
| 2.3 | 能带插值 vs full112 | Γ–K 路径 |
| 2.4 | lg_cov HHG + ±N + 弱场 | 对标 nb1-104 |

---

## 40-WF Cr-d-only 试点：输入文件说明

路径：`量子光研究/01_输入与模型/wannier/pilot_40wf/`

| 文件 | 作用 |
|------|------|
| `CrI3_donly.win` | `num_wann=40`，仅 `Cr:d` 投影；**初猜** dis_win/dis_froz |
| `README_pilot.md` | 服务器执行步骤与验证清单 |

**风险**（Soriano 等）：ligand p 影响 CrI₃ 磁性机制 → d-only **不能** 无验证替代 d+p。

**是否重跑 QE**：若服务器仍有 `CrI3.save` + 原 nscf 的 `.mmn/.eig` 对应 **112 投影**，则 40-WF 需 **重新 pw2wan**（投影改变 → 重算 amn/mmn）；nscf 本身可复用。

---

## 给导师/师兄的一句话

> 我们的 SBE 已经在 **完整 112-WF 模型对角化后的 Bloch 本征带** 上做 active-window 截断，不是粗暴删 Wannier 轨道。全价带 + 截高导带（nb1-104）在 H1–H11 与 full112 一致；对称低能窗（W30–W60）因丢失深价带而不可用。下一步用 JDOS 和 f-sum 残差量化「深价带是真实跃迁还是虚跃迁」；若仍要「真正少带模型」，再启动 40-WF Cr-d-only **新模型试点**，full112 作 reference。

---

## 12. SBE 求解器当前能输出什么？（Git：`Quantum-light`）

> **工作区分工**：SBE 代码改动在 `New_SBEs/Quantum-light`；本工作区只做 QE/Wannier。下表供你在 Git 工作区开改前对照。

### 12.1 已实现、可直接打开 `&output` 或默认就有

| 物理量 | 输出文件 | 开关 / 条件 | 说明 |
|--------|----------|-------------|------|
| 总电流时序 | `Jt.dat` | 默认 | J_x, J_y |
| **带内 / 带间电流分解** | `Jt_decomposed.dat` | 默认（`lg_cov` 路径） | `J_intra`, `J_inter`；后处理 `harmonic_intra_inter_decomp.py` |
| HHG **强度**谱 | `HHG.dat` | 默认 | `\|J(ω)\|²`，**无复相位**；谐波阶次写到 `nt/2`（由 `ncyc`、`dt` 决定，可覆盖 H20） |
| 电场 / 矢势 | `Et.dat` | 经典单轨迹 | 原生 E(t), A(t) |
| **Berry 曲率 + 量子度规** | `quantum_geometry.dat` | `save_geometry=.true.`（默认开） | Ω_n(k), g_xx, g_yy, g_xy；`analyze_quantum_geometry.py` |
| Peierls 路径 Berry | `berry_curvature.dat` | `gauge_method=vg` 时 | 生产不用 |
| k 空间占据（汇总） | `occupation_kt.dat` | `save_occupation=.true.` | 每 k 的价带/导带总占据 |
| k 空间占据（逐带） | `occupation_band_kt.dat` | `occ_band_resolved=.true.` | **极大**（112×k×t），慎用 |
| 带间相干标量 | `coherence_kt.dat` | `save_coherence=.true.` | `\|ρ_offdiag\|`，非完整 ρ |
| 能带结构快照 | `bands.dat` | 默认 | 截断窗口内 E_n(k) |
| 谷分辨电流 | `Jt_valley.dat` | 默认 | K / K' |
| 自旋 z 电流 + HHG | `Jt_spin.dat`, `HHG_spin.dat` | `spin_current=.true.` | 需 `lg_cov`；可选 `spin_sz_file` |
| BSV 平均谱 + 标准误 | `HHG_bsv.dat` | `bsv_enabled=.true.` | MC 或求积后处理合并 |
| BSV 16 节点求积 | `HHG_bsv_quad.dat` | Python 合并 | 生产主路径 |

`&output` 段示例见 `docs/02_求解器_理论与验证/可视化导出接口_20260611.md`。

### 12.2 尚未实现、但汇报 / R2 理论需要

| 物理量 | 状态 | 与汇报的关系 |
|--------|------|--------------|
| **完整密度矩阵 ρ(k,t)** | ❌ 未导出 | 只有占据/相干标量；Tier2 QRT 需要演化辅助算符 χ，不是 dump 全 ρ |
| **复数 J(ω)** / HHG 相位 | ❌ `mod_hhg.f90` 只写强度 | N 奇偶通道、C_conn、量子光统计必需 → **Git 工作区待改** |
| **j_anom(t) 四路拆分** | ❌ 只有 intra/inter | PPT 标注「待完成」；对称性论证 Ω≈0，未单独输出 j_anom |
| **C_conn(t,t')** 双时关联 | ❌ Fortran 未实现 | R2 核心；`tier2/` Python 试探；见 `Tier2_QRT编码_20260624` |
| **Σ、压缩谱、g^(2)、QFI** | ❌ 后处理链未接 SBE | 从 C_conn FFT 得到，Tier2 计划 |
| **f-sum / TRK 数值残差** | ❌ 求解器内无 | 用 **JDOS + f-sum 脚本**（读 tb.dat，不改 SBE）先诊断带窗 |

### 12.3 对你科学目标（H1–H9 主、看到 H20）的含义

```text
λ = 3200 nm  →  ℏω₀ ≈ 0.387 eV
H20 ≈ 7.7 eV 辐射能量
```

- **H1–H9**：现有 `HHG.dat` + `Jt_decomposed.dat` + nb1-104 验证 **足够** 写 R1。
- **H11–H15**：你们已用作 **k 网格 / 数值地板** 诊断；full112 在 H13+ 常近噪声底，解读需谨慎。
- **H16–H20**：FFT 网格允许输出更高阶，但 **cutoff 区信噪比** 取决于 `k` 密度、`T2`、Hann 窗；需专门收敛（k60/k80 + 更长 `ncyc` 或更高采样率），不是改 Wannier  alone 能解决的。
- **Liu 2022 EPL**：他们光子能量更大 → 同样阶次对应更低辐射能量；你们 3200 nm 下 H20 探索的是 **更长光程 / 多光子通道**，与 Liu 的截断阶次不可直接比，需在论文里分开表述。

**建议在 Git 工作区优先加的 SBE 改动**（按优先级）：

1. 导出复数 `J_x(ω), J_y(ω)`（或 `Jt.dat` + 后处理脚本带 Hann 窗一致）
2. `save_occupation` / `save_coherence` 在带窗扫描 run 中打开
3. Tier2：`C_conn` 辅助算符演化（R2，与 R1 并行）

---

## 13. JDOS + f-sum 脚本的目的（为何值得做）

**不只是**「把深价带/高导带对 H1–H15 做成一张表」——那张表是 **输出之一**，目的是回答 **带窗截断的物理机制**，服务于你们是否在 nb1-104 停步、是否重做 40-WF：

### 13.1 JDOS（matrix-element-weighted 跃迁直方图）

在 **不动 SBE、不重跑算例** 的前提下，从 `CrI3_tb.dat` 对每个 k 对角化，算 Bloch 基偶极矩阵元，统计跃迁能量 ΔE 落在 **H1…H20**（即 q·ℏω₀）附近的权重：

| 通道类型 | 含义 |
|----------|------|
| active 价带 → active 导带 | 经济窗口 **直接保留** 的 HHG 通道 |
| **deleted 深价带** → active 导带 | 截断后 **丢失的真实多光子通道** |
| active 价带 → **deleted 高导带** | 截掉 bands 105–112 等的影响 |
| deleted ↔ deleted | 完全移出 active space 的通道 |

**若 H9–H20 附近权重主要在 deleted→active**：说明 nb1-104 在 **高阶** 上缺的是 **真实跃迁**，不是数值噪声 alone → 高阶研究应保留 full112 或更多导带。  
**若权重很小但 f-sum 残差大**：说明主要是 **虚跃迁 / 求和规则** 贡献 → 应用 length gauge + 全价带（你们已在用）。

### 13.2 f-sum / velocity–dipole 残差

检查有限窗口是否破坏 **v–r 一致性** 与 TRK/f-sum（顺磁–抗磁抵消的数学基础）：

- 对称窗 W30–W60 预期 **残差大** → 与 HHG 失真一致，**独立验证**「不是 SBE bug 而是截断物理」。
- nb1-104 预期 **接近 full112** → 支撑生产配置。
- 比「只看 HHG 比值」更贴近师兄说的 **求和规则检查**。

### 13.3 执行位置（已跑通）

- **输入**：`CrI3_tb.dat`（`量子光研究/01_输入与模型/wannier/`）
- **脚本 + 报告**：`tools/transition_jdos_fsum.py` → `analysis/jdos_fsum_report_20260707.md`
- **与 QE/Wannier 关系**：无；**不依赖** 重跑 wannier90
- **场外 AI 沟通**：把 §16 + 报告 §1–§3 粘贴即可，不必重复聊天历史

---

## 14. 工作区与服务器分工（2026-07-07 更新）

| 环境 | 用途 |
|------|------|
| **`量子光研究`（当前 Cursor）** | QE/Wannier 输入、`pilot_40wf/`、tb.dat 本地副本 |
| **`New_SBEs/Quantum-light`（后续）** | SBE 改代码、JDOS/f-sum、BSV、Tier2 |
| **服务器** | nscf/save/mmn/eig **仍在** → 40-WF 只需 pw2wan+wannier90 |

**远程登录**：见 **§18 Remote-SSH**（原 MobaXterm：`wangjs@192.168.7.110`）。

---

## 15. 本工作区（QE/Wannier）待办 — 与 SBE 无关

| # | 任务 | 状态 |
|---|------|------|
| W1 | `pilot_40wf/CrI3_donly.win` + `make_donly_win.py` | ✅ |
| W2 | `pw2wan_donly.in` + `README_pilot.md` | ✅ |
| W3 | 服务器执行 wannier90 40-WF | ✅ 2026-07-12（`CrI3_donly_tb.dat` 137M） |
| W4 | 能带插值 vs full112 / 验收 | ✅ Γ,K gap 一致；见 `analysis/donly_validation_report_20260713.md` |
| W5 | `CrI3_donly_tb.dat` | ✅ |

| W6 | 更新本文 §9–§11、§13.3 与 JDOS 报告对齐 | ✅ 2026-07-07 |
| W7 | 配置 Cursor Remote-SSH | ⏳ 见 §18 |

SBE 复数 J(ω)、H20 k 收敛、速度规范求和规则 PRB 线：**在 Git 工作区**（另一 Agent）。

---

## 16. JDOS 数值结论摘要（2026-07-07）

> 完整表格：`量子光研究/01_输入与模型/wannier/analysis/jdos_fsum_report_20260707.md`

| 配置 | H1–H8 | H9–H15 | H16–H20 | 生产建议 |
|------|-------|--------|---------|----------|
| **full112** | active→active | 同左 | 同左 | 高阶 reference |
| **nb1-104** | ≈ full112 | 出现 **活→删导带**；active 通道仍为主 | H19–H20 active 通道 **≪** full112 | **H1–H9 生产**；H16+ 用 full112 或加导带 |
| **W40** | H1–H5 尚可 | H6+ **删价带→活导带** 主导 | 失真 | ❌ 诊断 only |
| **W30** | H1–H4 尚可 | H5+ 删价带通道 | 更严重 | ❌ 诊断 only |

**与 HHG 表（§Q3–Q4）交叉验证**：JDOS 独立说明「对称窗丢深价带 = 假通道」；「nb1-104 丢高导带 = H9+ 真通道损失」。

---

## 17. 本文档覆盖范围审计（供场外 AI / 你自检）

### 已写入、可对外引用的判断

| 主题 | 位置 |
|------|------|
| Bloch vs Wannier 截断机制 + 源码 | Q1 |
| lg_cov Berry / 平行输运 | Q2 |
| 对称窗 vs nb1-104 HHG 表 | Q3–Q4 |
| 3200 nm / H 阶次能量 | Q8, §12.3 |
| ±N leakage（nb1-104 vs full112） | Q6 |
| 零场 baseline / nb10-104 异常 | Q7 |
| JDOS + f-sum 代理（**已完成**） | Q9–Q10, §13, §16 |
| SBE 导出能力边界 | §12 |
| 40-WF 试点路径与风险 | Phase 2, §40-WF |
| A–E 决策矩阵 | 文首 |
| 工作区三分工 | §14–§15 |
| 带数缩减总方案交叉引用 | 见 `CrI3_带数缩减评估方案_20260706.md` |

### 有意不在本文展开（见其他文档 / Agent）

| 主题 | 说明 |
|------|------|
| 速度规范 Peierls VG 求和规则失败、PRB 文章规划 | Git 工作区 / 另一 Agent |
| SBE 改码清单（复数 J(ω)、C_conn） | §12.2–12.3 仅列状态 |
| QE nscf 输入细节、能带图复现 | `CrI3_QE_Wannier工作汇报_202605.md` |
| W20 lg_cov HHG | §Q3 标 ❌ 未跑 |
| 对称窗 ±N leakage | §Q6 标待跑 |
| 逐 k v–r 残差 / TRK 完整数值 | Q10 标为 Git/SBE 待做 |
| Git Agent 首轮阅读顺序 | `01_输入与模型/wannier/GIT工作区Agent阅读指引.md` |

**给场外 AI 的最小附件包**：本文 + `jdos_fsum_report_20260707.md` +（可选）`CrI3_带数缩减评估方案_20260706.md`。

---

## 18. Remote-SSH 配置（原 MobaXterm 登录方式）

### 18.1 你之前的登录方式（2026-07-09 已确认跳板）

| 项 | 值 |
|----|-----|
| 客户端 | **MobaXterm** |
| **SSH 网关（第 1 跳）** | `physics@202.119.84.63`**`:66`** |
| **登录节点（第 2 跳）** | `wangjs@192.168.7.110:22`（hostname: **master01**） |
| 认证 | 密码（MobaXterm 未勾选 private key） |
| 本地 SSH 别名 | `cri3-login`（`ProxyJump cri3-gateway`） |

**不能** 直连 `192.168.7.110`——该地址仅在网关之后可达。

### 18.2 服务器上 QE/Wannier 相关路径（文档归档）

```text
/public/home/wangjs/project/CrI3_TB/wannier/          # 112-WF TB、mmn/eig
/public/home/wangjs/project/CrI3_TB_mN_Pconj/wannier/
/public/home/wangjs/project/New_SBEs/Quantum-light/  # SBE 代码（Git）
/public/home/wangjs/software/q-e-qe-7.2/bin/           # QE 7.2（ZnO 文档同款路径）
```

40-WF 试点：在 `CrI3_TB/wannier/` 下按 `pilot_40wf/README_pilot.md` 执行 pw2wan + wannier90。

### 18.3 Cursor Remote-SSH 步骤（Windows）

1. **扩展**：Cursor 安装 **Remote - SSH**（与 VS Code 同款，`ms-vscode-remote.remote-ssh`）。
2. **SSH 配置**：已生成模板 `C:\Users\26507\.ssh\config`（Host 名 `cri3-login`）。  
   - 若 MobaXterm 用**密码**：删掉或注释 `IdentityFile` 行，连接时输入密码。  
   - 若用**密钥**：在 MobaXterm → Session 右键 → Edit session → Advanced SSH settings 查看 **Use private key** 路径，复制到 `IdentityFile`，或导出为 `C:\Users\26507\.ssh\id_ed25519`（**勿提交 Git**）。
3. **测试**（PowerShell）：`ssh cri3-login` — 能进 shell 再开 Cursor。
4. **Cursor**：`F1` → **Remote-SSH: Connect to Host…** → 选 `cri3-login` → 打开文件夹 `/public/home/wangjs/project/CrI3_TB` 或 `New_SBEs/Quantum-light`。
5. **端口**：默认 22；若 MobaXterm 里改过 SSH port，在 `config` 加 `Port xxxx`。

详细图文与密钥导出：`量子光研究/04_工具与环境/Remote-SSH配置说明.md`。

