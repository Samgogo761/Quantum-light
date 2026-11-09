# CrI3 QE vs Wannier 能带验证（2026-07-13）

> 目的：确认 **40-WF d-only**（以及对照 **112-WF**）Wannier 插值能带与 QE `bands.x` 基准相符。  
> 图仿 ZnO：`examples/ZnO_wurtzite_qe_wannier/output/ZnO_qe_vs_wannier_bands_near_gap_pubstyle.png`。

## 0. 服务器状态

- `squeue -u wangjs`：**当前无作业**（本验证为本地后处理，未向服务器提交新任务）。

## 1. 数据来源

| 数据 | 路径 |
|------|------|
| QE path bands | 服务器 `CrI3_TB/outputs/bands/CrI3_bands.dat.gnu` → 本机 `量子光研究/01_输入与模型/qe_bands/`（原先本机缺 QE 能带，已补下） |
| SCF Fermi | `CrI3_scf_soc.out`：**0.0810 eV** |
| Wannier 40 | `wannier/pilot_40wf/CrI3_donly_band.dat`（40 带，k 路径 638 点） |
| Wannier 112 | `wannier/CrI3_band.dat`（112 带，同路径） |

图中能量均平移为 **$E - E_{\mathrm{VBM}}$**（各自模型在 Γ 的 VBM）。灰线 = QE，蓝线 = 对应 Wannier。

## 2.1 重要限制：QE `bands.x` 能量天花板

当前 QE path 计算 `nbnd=148`，平移到 \(E-E_{\mathrm{VBM}}\) 后 **最高大约只到 ~1.29 eV**。  
因此图上 **1.5–3 eV 的 Wannier 高导带无法用这套 QE 验证**（需加大 `nbnd` 重跑 QE `bands`）。右图 ~2–3 eV 的带及其交叉，**不要直接当成与 QE 不符的证据**。近隙（≲1.3 eV）的对比才有效。

分栏图说明：[CrI3_分栏能带与spread说明_20260713.md](CrI3_分栏能带与spread说明_20260713.md)

## 2. 推荐查看的图

| 文件 | 说明 |
|------|------|
| [figures/CrI3_qe_vs_donly40_sidebyside_near_gap.png](figures/CrI3_qe_vs_donly40_sidebyside_near_gap.png) | ★ **一行两列**：左 QE / 右 40-WF |
| [figures/CrI3_qe_wannier112_donly40_1x3_near_gap.png](figures/CrI3_qe_wannier112_donly40_1x3_near_gap.png) | ★ **一行三列**：QE \| 112 \| 40 |
| [figures/CrI3_qe_vs_donly40_sidebyside_wide.png](figures/CrI3_qe_vs_donly40_sidebyside_wide.png) | 分栏宽窗（深价带） |
| [figures/CrI3_qe_vs_donly40_bands_near_gap_pubstyle.png](figures/CrI3_qe_vs_donly40_bands_near_gap_pubstyle.png) | 旧叠图（易盖住 QE） |
| [figures/CrI3_qe_vs_wannier112_vs_donly40_near_gap.png](figures/CrI3_qe_vs_wannier112_vs_donly40_near_gap.png) | 旧三线叠图 |

脚本：`wannier/tools/plot_qe_vs_wannier_panels.py`

## 3. 定量结果（Γ / K）

| 模型 | Γ gap (eV) | K gap (eV) | Γ 价带数 $N_{\mathrm{occ}}$（$E\le E_F$） |
|------|----------:|----------:|-----------------------------------------------:|
| QE `bands.x` | **0.6671** | 1.1244 | 140（全 148 带中的占据数，非有效模型） |
| Wannier **40 d-only** | **0.6672** | 1.1243 | **22** |
| Wannier **112 d+p** | **0.6672** | 1.1243 | 84 |

近隙窗 $E-E_{\mathrm{VBM}}\in[-1.5,2.0]$ eV、归一化路径上「最近邻带」误差：

| 模型 | MAE (eV) | RMS (eV) | max (eV) |
|------|---------:|---------:|---------:|
| d-only 40 | 0.0107 | 0.0606 | 0.9638 |
| full112 | 0.0007 | 0.0017 | 0.0120 |

**判读**：

- Γ/K **带隙与 QE 一致到 ~1 meV**（0.667 eV）——局域化在 **隙区** 是成功的。
- d-only 的近隙 MAE/RMS 仍可能大于 112：外窗外没有 I-p 深价带，且 disentanglement 子空间与全带不完全重合；**看图比只看 gap 更重要**。
- 宽能量窗图上，40-WF 只在约 **−2 ~ +3 eV**（`dis_win`）有带——这是设计如此，不是绘图错误。

## 4. $N_{\mathrm{occ}}$ 是什么？

**是的：`$N_{\mathrm{occ}}$` = 有效 Wannier 哈密顿量中、费米能以下的占据本征带条数（spinor 带指数）。**

对本 40-WF 模型：

- 投影初猜：`Cr:d` × 4 原子 × 2（SOC spinor）→ **`num_wann = 40`**。
- 简单离子化学：Cr³⁺ 为 d³ → 4×3 = **12 个 d 电子**，若天真地当成「12 条占据 d 带」会得到 **12**，但那不是本模型的占据计数。
- 实际在插值能带上数得 **`$N_{\mathrm{occ}}=22$`**：说明经过 **disentanglement** 后，有效 40 维子空间吸收了更多 **杂化占据权重**（价带顶附近并不纯是原子 Cr-d；I-p 特征被部分投影进该子空间，或外窗内本就有多于 12 条占据 DFT 带参与混合）。
- 因此：**40-WF ≠「只留下纯 Cr-d 轨道」**，而是「以 Cr-d 为初猜的、经 disentangle 的有效近隙子空间」。电子计数与轨道成分要用投影/权重再解释，不能靠化学式直接定 `nv`。

对照：112-WF（d+p）在同一约定下 Γ 有 **84** 条占据带——那才是「几乎全价带」母模型。

## 5. 风险说明（删掉显式 I-p）

即使近隙与 QE 对齐，d-only 仍有结构性风险：

1. **深价带 / 宽光谱**：I-5p 主导的深占据带不在 40-WF 内；高阶谐波、宽带光学可能缺通道。
2. **p–d 杂化与光学矩阵元**：CrI₃ 的磁交换与 SOC 相关光学响应常依赖 I-p；矩阵元被挤进有效 d 子空间后，**形状可能对、权重可能偏**。
3. **Ω_I / spread**：40-WF 的 Ω_total 约为 112 的 ~2×，局域性更差——插值带在路径上可能更「抖」或偏离。
4. **尚未替代生产链**：在 HHG H1–H9 与 ±N 选择定则未过关前，**112 / nb1-104 仍是生产基准**；40-WF 只是减时试点。

## 6. 当前结论（仅局域化，不谈 SBE）

| 检查 | 结果 |
|------|------|
| 服务器是否有任务在跑 | **无** |
| QE 基准文件是否在本机 | **已补齐** `qe_bands/` |
| QE vs 40-WF 近隙叠图 | **已生成**（见上表） |
| Γ/K gap 与 QE | **一致（0.667 eV）** |
| 是否等于「纯 Cr-d、12 电子」 | **否**；`$N_{\mathrm{occ}}=22$`，为 disentangled 有效子空间 |
| 可否宣布取代 112 | **否**；需后续光学/HHG 验证（另议） |

下一步建议（仍属 Wannier 质量，而非 SBE 科学结论）：若近隙叠图在导带/价带顶有可见偏离，优先调 `dis_win` / `dis_froz` 或检查投影，再重跑 **wannier90 only**。
