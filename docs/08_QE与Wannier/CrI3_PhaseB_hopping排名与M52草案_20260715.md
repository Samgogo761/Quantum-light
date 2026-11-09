# CrI3 Phase B：d–p hopping 排名 + SVD + M52/M64/M88 草案（2026-07-15）

> **用户确认**：暂停 40-WF opt1 收敛扫描，转入 Phase B/C。  
> **本轮交付**：B1–B3 定量结果、能量窗计数、三份候选 `.win`、服务器提交链（`select_projections`，**不重跑 pw2wan**）。

---

## 1. 方法与数据

| 项 | 说明 |
|----|------|
| 输入 | full112 `CrI3_tb.dat`（1.1 GB）、`CrI3.nnkp`（112 自旋投影顺序） |
| 脚本 | `wannier/tools/rank_dp_hoppings.py` |
| 输出 | `analysis/key_Ip/{hopping_rank,p_opt_singular_values,p_opt_vectors,local_orbital_frames}.csv/json` |
| WF 映射 | nnkp 顺序 = site × (pz,px,py) × (↑,↓)；Cr-d 1–40，I-p 41–112；Ω_D≈0.005 → 终态 WF ≈ 初始投影 |

**通道定义**（排除 R=0 同原子 onsite，只计跨原子传输）：

| 通道 | Σ\|t\|² 占比 |
|------|------------:|
| d–d | **0.46%** |
| **d–p** | **67.1%** |
| p–p | 32.4% |

与评分器 L_hop（含 Cr 同原子 d–d SOC/晶体场块）口径不同；**传输权重几乎全经 I-p 中介**。

---

## 2. B2：按 I 原子排名（关键数）

| I 原子 | 层 | 界面 | Σ d–p 同层 | Σ d–p 跨层 | **Σ p–p 层间** |
|-------:|---:|:----:|-----------:|-----------:|--------------:|
| 4–6 | 1 | ✅ | ~14.0 | ~0.03 | **1.33–1.66** |
| 7–9 | 1 | ❌ | ~14.0 | ~0.002 | ~0.02 |
| 10–12 | 2 | ❌ | ~14.0 | ~0.002 | ~0.02 |
| 13–15 | 2 | ✅ | ~14.0 | ~0.03 | **1.32–1.67** |

**结论**：
- 12 个 I 的**层内 d–p 权重几乎相同**（~14）——不能靠「删外侧 I」减维而不破对称性。
- **层间 p–p 通道集中在 6 个界面 I**（4–6, 13–15），比外侧 I 强 **~80×** → M52-interface 有明确物理动机。

---

## 3. B3：SVD 选轨（摘要）

### 3.1 目标 2：层间 d+p（`t_inter_d+p`）— 界面 I

| 界面 I | Kramers-2 捕获 | 主分量 n_pz |
|-------:|---------------:|----------:|
| 4,5,13,14 | **~72%** | **0.987** |
| 6,15 | ~70% | **0.991** |

**层间耦合 ≈ 全局笛卡尔 pz**（van der Waals 方向），与几何帧一致。

### 3.2 目标 1：层内 d（`t_intra_d`）— 所有 I

| 指标 | 值 |
|------|-----|
| Kramers-2 捕获 | **~44%**（谱平坦：s1≈1.31…s6≈0.67） |
| 主分量 | **px/py 混合**，随 I 位点而变（非单一全局方向） |

**结论**：每个 I 只留 1 个 p（M64）对层内 d–p 仍不足；M88（px+py，不含 pz）SVD 捕获 ~**81%** 层内权重。

---

## 4. 能量窗计数（576-k `CrI3.eig`）

| 模型 | dis_win | min 带数 | dis_froz | max frozen |
|------|---------|---------:|----------|----------:|
| **M52** | [-2.0, 3.5] | 54 ✅ | [-1.2, 1.0] | 22 ✅ |
| **M64** | [-2.5, 3.5] | 66 ✅ | [-1.2, 1.0] | 22 ✅ |
| **M88** | [-3.5, 4.5] | 90 ✅ | [-1.2, 1.0] | 22 ✅ |

---

## 5. Phase C 草案（已生成，待 wannier90 跑通）

**路线 A（已采用）**：Wannier90 3.1 `select_projections`，软链 full112 `CrI3.{amn,mmn,eig}`，**省 ~15 h pw2wan**。

| 模型 | num_wann | select_projections（112 基序号） | 目录 |
|------|---------:|----------------------------------|------|
| M52 | 52 | 1–40 + 界面 I 的 pz↑↓（41,42,47,48,53,54,95,96,101,102,107,108） | `models/M52_interface/` |
| M64 | 64 | 1–40 + 12×pz↑↓ | `models/M64_allI_pz/` |
| M88 | 88 | 1–40 + 12×(px,py)↑↓ | `models/M88_allI_2p/` |

本地文件：
- `wannier/tools/make_candidate_wins.py`、`compute_select_projections.py`
- `wannier/models/projection_indices.json`
- `wannier/models/submit_candidates.sh`（M52→M64→M88 依赖链，part_2）

**服务器作业（2026-07-15）**：
- part_2：26659 **FAILED**（spread 已收敛 Ω=427 Å²，写 hr/tb 时 MPI 崩溃）；26660/26661 已取消。
- **part_1 重投**：**26710** (M52) → **26711** (M64) → **26712** (M88)：26710 再次 FAILED（最小化已收敛，MPI 写 hr/tb 崩溃）；26711/26712 已取消。
- **根因**：本机 `wannier90.x` 的 MPI 并行在写 `hr/tb/band` 时多 rank 抢写 → `forrtl CLOSE error` / double free；最小化本身正常。40-WF/112 也曾 FAILED，但常在 `All done` 之后才炸，文件已落盘。
- **修复（node05 独占，2026-07-15 晚）**：**26725** M52 串行 plot → **26726** M64 → **26727** M88  
  - `--nodelist=node05 --exclusive`  
  - 最小化：**mpirun -np 8**（与成功 40-WF 一致）  
  - 写文件：**仅串行** `restart=plot`；验收要求 `All done` + `hr/tb/band` 非空  
  - 脚本：`submit_node05_exclusive.sh`

---

## 6. 下一步（P4）

1. wannier90 收敛后 → **评分器四级检验**（L_E/L_sub/L_loc/L_hop）
2. `check_mesh_occupation.py` 自动 nv → 短 HHG vs nb104
3. 最优候选再进 k/T2/dt 收敛

**暂缓**：40-WF opt1-nv12 的 k80/k120（除非专门研究 d³ 极限模型）。

---

## 7. 与 40-WF 审计的衔接

| 40-WF 问题 | Phase B 回应 |
|-----------|-------------|
| L_sub FAIL（丢 I-p 价带） | M52/M64/M88 显式加 p 子空间 |
| L_hop FAIL（d–p 占 59%） | 保留 d–p 通道而非纯 d³ |
| H1 仅 0.34× nb104 | 需等 P4 HHG 验证；M52 优先测层间/N 奇偶 |
