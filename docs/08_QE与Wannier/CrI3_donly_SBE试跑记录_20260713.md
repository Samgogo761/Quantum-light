# CrI3 d-only SBE 试跑记录（2026-07-13）

> 脚本：`deploy/run_donly_readcheck_login.sh`（Task2）、`deploy/run_donly_lg_cov_pilot.sh`（Task3）

---

## 公共参数

| 项 | 值 |
|----|-----|
| TB | `/public/home/wangjs/project/CrI3_TB/wannier/pilot_40wf/CrI3_donly_tb.dat` |
| `nv_orig` | **22** |
| `nb_start` / `nb_end` | 1 / 40 |
| `gauge_method` | `lg_cov` |
| 激光 | 3200 nm, I=2×10¹¹ W/cm², T2=0.5 cycle（Task3） |

---

## Task2：读入试跑（login, k10）

| 项 | 值 |
|----|-----|
| 输出 | `output_donly_readcheck/k10_b40_nv22/` |
| 墙时 | ~6 min（4 OMP） |
| Γ gap | 0.667 eV |
| `\|J(t=0)\|` | **5.7×10⁻⁷** ⚠️（k10 网格过粗，仅作读入诊断） |
| 产物 | `HHG.dat`, `Jt.dat`, `bands.dat` |

---

## Task3：HHG 试跑（Job 26459, part_1, k40）

| 项 | 值 |
|----|-----|
| 提交 | 2026-07-13，用户批准 |
| 节点 | part_1 / node10，36 核 exclusive |
| 输出 | `output_donly_lg_cov_T2_0p5cycle/lgcov_k40_b40_nv22/` |
| 墙时 | **real ~12 min** |
| `\|J(t=0)\|` | **6.2×10⁻¹³**（优于 k10，仍 **>** full112 ~10⁻¹⁸） |
| 产物 | `HHG.dat`, `Jt.dat`, `Jt_decomposed.dat`, `run.log` |

---

## 待完成

| 项 | 状态 |
|----|------|
| H1–H9 强度 vs full112 / nb1-104 | ✅ 见 [CrI3_donly_HHG对比_20260713.md](CrI3_donly_HHG对比_20260713.md) |
| ±N leakage | ❌ 未跑 |
| 是否判定 pilot 失败/继续 | H1–H9 可作原型；H10+ 失败；见对照文档 |

---

## 对标参考（生产）

| 配置 | 输出目录（示例） |
|------|------------------|
| nb1-104 k40 T2=0.5 | `output_lg_cov_fullvalence_T2_0p5cycle/lgcov_k40_nb104` |
| full112 | `.../lgcov_k40_nb112` |
