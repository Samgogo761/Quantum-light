# CrI3 40-WF d-only Task1 验收（2026-07-13）

> 服务器：`/public/home/wangjs/project/CrI3_TB/wannier/pilot_40wf/`  
> 分析脚本：`量子光研究/01_输入与模型/wannier/tools/analyze_donly_validation.py`

---

## 1. Wannier 流水线

| 步骤 | JobID | 结果 |
|------|------:|------|
| `-pp` | 26073 | ✅ |
| `pw2wannier90` | 26074 | ✅ ~15h25m |
| `wannier90` | 26077 | ✅ `CrI3_donly_tb.dat` 137M |

---

## 2. 隙区能带（band.dat）

| 点 | full112 gap | d-only gap | d-only 价带数 |
|----|------------:|-----------:|--------------:|
| Γ | 0.667 eV | **0.667 eV** | **22** |
| K | 1.124 eV | **1.124 eV** | **22** |

VBM/CBM 数值与 full112 对齐（near-gap Cr-d）。

---

## 3. 占据数（SBE）

- **`nv_orig = 22`**（勿用 84 或化学 d³ 的 12）
- `CrI3_donly.eig`（200 DFT 带）**不能**用来定 Wannier 占据数

---

## 4. Spread（wout）

| 量 | d-only | full112 |
|---|---:|---:|
| Ω_I | 354 | 197 |
| Ω_D | 2.22 | 0.005 |
| Ω_total | 427 | 202 |

最终单 WF spread 7–14 Å²；Ω_I 偏大 → **试点谨慎，不自动替代 112-WF**。

---

## 5. 结论

**Task1 通过**（隙区 + tb.dat）；可进 SBE 试跑；是否采纳为有效模型取决于 Task3 HHG/±N。
