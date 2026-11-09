# CrI3 SBE 历史结果 manifest（2026-07-16）

> 用途：新少带模型（M52/M64/M88）与 **同一口径** 比较。  
> 详细路径以「量子光研究/新SBEs/data/」为准；`02_计算输出/新SBEs/data/` 有镜像。

---

## 有效基线（优先引用）

| ID | 模型 / 窗 / Nocc | 参数摘要 | 输出根 | 用途 |
|----|------------------|----------|--------|------|
| A1 | full112 / 1–112 / 84 | lg_cov · k40 · dt=0.35 · T2=0.5cyc · 3200nm · 2e11 | `output_lg_cov_fullvalence_T2_0p5cycle/lgcov_k40_nb112/` | **主参考** |
| A2 | nb1-104 / 1–104 / 84 | 同 A1 | `.../lgcov_k40_nb104/` | 经济参考；H1–H9；BSV 母配置 |
| A4 | full112 / k60 | 同 A1 但 60×60 | `output_lg_cov_k60_b112_T2_0p5cycle/` | k40↔k60 收敛 |
| B2 | ±N · P 共轭 −N | 同 A1；P 构造 TB | `CrI3_TB-N/output_afm_pconj_fullvalence_T2_0p5cycle/` | **选择定则主结果** |
| B3 | evenN/oddN 分析 | — | `New_SBEs/Quantum-light/results/afm_neel_symmetry_Pconj_T2_0p5cycle/` | leakage / Jo Je |
| D1 | BSV 求积 nb1-104 | k40 · T2=**1.0**cyc · 16 节点 | `output_bsv_quad/` | R1 主结果（注意 T2≠0.5） |

文档入口：`带窗经济配置测试_T2_0p5cycle_20260520.md`，`k网格60x60确认_T2_0p5cycle_20260519.md`，`AFM_Neel_Pconjugate选择定则结果_20260602.md`，`BSV确定性求积方案_20260612.md`，`CrI3_nb1-104高阶可信区间_20260709.md`。

---

## 负证 / 作废（勿当生产）

| ID | 内容 | 原因 |
|----|------|------|
| C* | 对称 Bloch 窗 W30–W60 | H3–H9 数量级偏离；截价带 |
| B4 | 旧独立 −N（非 P 共轭） | DFT 未配对；H2 evenN 异常 |
| E1 | 40-opt1 收敛阵列 nv=22 | 真 Nocc=12；谱物理作废（可留墙时标定） |

---

## 默认激光与规范（新作业必须对齐）

```text
gauge_method = lg_cov
wvl_nm = 3200
intensity_Wcm2 = 2.0e11
phi_cep_deg = 90
ncyc = 4
dt = 0.35
T2_cycles = 0.5          # 经典；BSV 历史用 1.0
E_fermi_eV ≈ 0.0843      # 生产 namelist；占据诊断常用 0.081
```

---

## CSV 机读摘要

见同目录 `CrI3_SBE历史结果_manifest_20260716.csv`。
