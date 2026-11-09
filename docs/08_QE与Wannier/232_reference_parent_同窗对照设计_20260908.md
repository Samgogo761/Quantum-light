# 232 reference parent：同窗对照设计（2026-09-08）

```text
LINE                 = M
DECISION             = 112_LARGE_GO_232_WAIT
EXPERIMENT           = 112_old -> 112_large (then 232 only after human)
NOT                  = 232 Wannier ; GH7 ; k40/k50 ; chemical I-d 144 ; WINDOW144 production
NOT                  = Husimi-Q ; SCDM started ; Q is I-5d ; HHG mechanism
HUMAN_NEXT           = after 112_large Wannier/verification (high-CB vs QE; hot-k P and r; small classical H5/H7)
```

用户批准：现在没有新的物理拍板。本地完成 \(112_{\mathrm{large}}\)（`num_wann=112`，Cr-\(d\)+I-\(p\)，冻结窗不变，只把 Bloch 池补全）。**不上 232。**  
C 的 UPF projector 证据见 [expanded_parent_轨道鉴定政策_20260908.md](expanded_parent_轨道鉴定政策_20260908.md) §6。

工作假设（尚未闭合）：

\[
\boxed{
Q_{\mathrm{optically\ active}}
\text{ is strongly enriched in I-}d\text{-like character}
}
\]

不是 \(Q=I\text{-}5d\)。232 是一次**可证伪**的 reference-parent 检验，不是生产 parent。

---

## 已批准的三块（方法约束已写入）

### A. 外窗对照设计

不能只做 \(112_{\mathrm{old}}\to 232\)。那会把三件事绑在一起：更大能量窗、更多 Bloch 带、加入 I-\(d\)。

对照链：

\[
\boxed{
112_{\mathrm{old}}
\;\rightarrow\;
112_{\mathrm{large\text{-}window}}
\;\rightarrow\;
232_{\mathrm{I\text{-}d}}
}
\]

| 比较 | 共用什么 | 问的问题 |
|---|---|---|
| \(112_{\mathrm{old}}\) vs \(112_{\mathrm{large}}\) | 同一套 \(d+p\) 化学试探、同一冻结窗哲学 | 只把高能 Bloch 池加大，问题是否已经动 |
| \(112_{\mathrm{large}}\) vs \(232\) | **同一套新 NSCF、同一外窗、同一 Bloch 池** | 在这个更大池里，是否把完整 I-\(d\) 方向纳入 parent |

KS 势、赝势、结构不变。变的是投影下来的 Wannier Hilbert space，因此 SBE 里的 \(H_W(\mathbf k)\) 会变。这正是需要人审的原因。

同维数的 SCDM 中性控制（232 vs 232）**现在不做**。现在只判断 I-\(d\)-like 假设值不值得继续，不做最终因果归因。

硬约束（不是给你选的数字，是逻辑）：

- 生产 112：`nbnd=200`，`dis_win=[-6,8]\) eV，窗内最少约 144 条带。
- \(N_{\mathrm{win}}(k)\ge N_{\mathrm{wann}}\)。144 < 232，200 < 232，所以必须新的更高 `nbnd` NSCF，以及更高的外窗上沿。
- 外窗**下沿固定** \(-6\) eV（与生产 112 价带覆盖一致）。上沿必须使 **每个** \(k\) 满足 \(N_{\mathrm{outer}}(k)>232\)（有 disentangle 自由度），不能「刚好 232 条」。目标余量与生产 112 同类：窗内带数 / 232 \(\simeq\) 144/112。具体 eV / `nbnd` 由计数 NSCF 决定。

冻结窗：\(112_{\mathrm{large}}\) 与 \(232\) **相同**，默认沿用生产 `dis_froz=[-1.5,0.8]\) eV。近隙锁住，差别只来自外窗（以及 232 的额外 120 维与 I-\(d\) 试探）。

### B. 三个 parent 的定义

| 标签 | 是什么 | 不是什么 |
|---|---|---|
| \(112_{\mathrm{old}}\) | 现用生产 full112 TB：`num_wann=112`，试探 Cr-\(d\) + I-\(p\)，窗 \([-6,8]\)，24×24 | 新化学模型 |
| \(112_{\mathrm{large}}\) | **新** Wannier：同一 Cr-\(d\)+I-\(p\) 试探，`num_wann=112`，与 232 **同一**新 NSCF、**同一**大外窗、同一冻结窗 | 「把旧 112 插值到新窗」；不是 WINDOW144 |
| \(232_{\mathrm{I\text{-}d}}\) | **新** Wannier：试探 Cr-\(d\) + I-\(p\) + **完整** I-\(d\)（4×5×2 + 12×3×2 + 12×5×2 = 40+72+120 = 232） | 化学 I-\(d\) 144；不是把 \(Q_{32}\) 贴上 I-\(d\) 标签 |

I-\(d\) 初猜径向：用方案 C 已冻结的生产 UPF 第一组有限 5D PAW 赝波（`PP_PSWFC` 7/8 的 \(j\) 平均），**不再扫 `zona`**。这是「232 测的是哪条 I-\(d\)」的定义，不是节点脚本问题。

WINDOW144 仍是 \(\Pi_{112}\oplus Q_{32}\) 的诊断完备化，**不**升格，也**不**当作 232 的靶标谱。

SBE 范围（Gate 1–2 之后才考虑）：k10/k20、经典、+N、与 WINDOW144 对照相同的两个场点（\(2\times10^{11}\) 与 \(10^{12}\)）。禁止 GH7、k40/k50、正式 M88 GH5。H5/H7 的变化只当证据，见 §C。

### C. 成功 / 失败（物理判据）

C 的六壳层 \(|A|^2\sim 0.26\)–\(0.28\) **不是** \(P_Q^{232}\)。它只说明初猜；不能用来事先宣判 232 必败或必成。必须等新的 \(U_{\mathrm{dis}}\)。

**约束 1 — 不要按旧 \(Q\) 带号追踪。**  
旧 \(Q_{32}\) 是 \([-6,8]\) eV、144 维外窗里的补空间。新外窗里 Hilbert 变了，不存在「old Q \#\(x\) = new parent \#\(x\)」。

比较对象是旧诊断里**光学最活跃的那些真实 Bloch 态 / 子空间**（A1-O 热点 \(k\) 上，由 \(\lvert\Pi r Q\rvert\) 列权重标出的态）。映射到新 NSCF：同一 \(\mathbf k\)，用能量窗口作候选，再用波函数 overlap / projector（\(\lvert\langle\psi^{\mathrm{old}}_n|\psi^{\mathrm{new}}_m\rangle\rvert^2\)）对齐。Overlap 只在同时存有新旧 \(wfc\) 的 \(k\) 上算（A1-O 星图 52 点；热点不在 24×24 网格上）。具体线性代数由工程脚本做。

\(r\) 仍用 A1-O 同一套长度规范 Berry 定义。

**约束 2 — H5/H7「同向变化」不是成功门。**  
没有已知正确的 H5/H7；WINDOW144 的 ×4.76 不是真值。232 之后 H5/H7 怎么变，只是**证据**，不能写成「朝某方向动 = I-\(d\) 正确」。

成功需要**同时**看到（加入 I-\(d\) 相对 \(112_{\mathrm{large}}\)，同一新窗）：

1. 目标高导带（映射后的光学活跃 Bloch 子空间）的 **projectability** 明显提高；
2. \(E_n(\mathbf k)\) 对 QE **高导带拟合**改善（这些态，不是只看近隙）；
3. 热点上关键 \(r_{mn}\) 的描述改善（同一套 \(r\)）；
4. 经典 HHG 对**进一步小幅扩空间**趋于稳定（k10 vs k20，以及 leftover 光学耦合不再主导）。H5/H7 幅度与相位记入证据表，不单独 PASS/FAIL。

| 结果 | 判读 |
|---|---|
| \(112_{\mathrm{large}}\) 已满足 1–3，232 几乎不再加分 | 主要是更大 Bloch 池 / 外窗，**不是** I-\(d\) 壳层 |
| 相对 \(112_{\mathrm{large}}\)，232 在 1–3 上明显再改善，HHG 只作是否同向的旁证 | I-\(d\) 假设**加分**（仍不是 \(Q=I\text{-}5d\)） |
| 232 相对 \(112_{\mathrm{large}}\) 在 1–3 上几乎不抬，或 HHG 与 projectability 脱节 | **I-\(d\) 假设失败** → 转 projectability/SCDM |
| 232 disentangle / 局域化严重坏（\(\Omega\) 爆炸、游离 WF、谱对 QE 对不上） | **失败** → 不要硬贴 I-\(d\) |

下一轮人机：\(112_{\mathrm{large}}\) 与 232 的 Gate 1–2（及若已跑的 k10/k20）**结果出来之后**。在此之前不宣布机制、不升格生产 parent。

---

## 工程附录（不需要你选）

满足上面三块之后才执行。这里只预登记，避免把设计缺口伪装成「到时候再说」。

1. **计数再定 `nbnd` / \(E_{\max}\)**：作业 **32911 COMPLETED**。见上文「计数结果」。32588（52 点 × 400）SIGKILL ≈ 239 GB。576k 仍走 **part_1**，禁止写回 `temp_scf_soc` 与 `temp_a1h_scf_clean`。
2. **赝势 / SCF**：生产 `Cr.rel-pbe.upf` / `I.rel-pbe.upf` 的 sha256 与 A1-H 门一致；不改结构、U、SOC。
3. **试探**：\(112_{\mathrm{large}}\) 仅 Cr-\(d\)+I-\(p\)；232 另加 I-\(d\)，径向来自 C 的 UPF 表，不再扫 `zona`。私有 `pw2wannier90` 若沿用 C 的补丁，不得覆盖生产 `bin/pw2wannier90.x`。
4. **网格**：Wannier 仍 24×24×1（与 112 可对照）。SBE 只 k10/k20。k20 插值独立性问题与 WINDOW144 相同。H5/H7 差值只进证据表，不得写成成功门，也不得把 k20 相对 k10 的差全部写成「真漏空间」。
5. **可选第四对照（默认不做）**：若 \(112_{\mathrm{old}}\) vs \(112_{\mathrm{large}}\) 分不清「重做 Wannier」和「加大窗」，再加一个新 NSCF 上、外窗仍为 \([-6,8]\) 的 112。v1 不默认付这成本。
6. **硬禁止**：GH7、k40/k50、化学 I-\(d\) 144、WINDOW144 升格、Husimi-Q 生产、宣布 HHG 机制、同维 SCDM 控制（除非 Gate 失败转入 SCDM）。
7. **计数脚本**：`deploy/a0_layerA_m88full/ref232_count/`；后处理 `tools/analysis/count_outer_window_bands.py`。Overlap 映射脚本等新旧 \(wfc\) 都在以后再写，比较对象仍是旧光学活跃 Bloch 子空间，不是旧 \(Q\) 带号。

---

## 人机

下一轮人审：**\(112_{\mathrm{large}}\) 的 Wannier 与验证出来之后**（高导带 vs QE；热点 projectability 与 \(r_{mn}\)；小规模经典 H5/H7 只作证据）。在此之前不宣布机制、不升格、**不投 232 Wannier**。

---

## 计数结果（2026-09-09，作业 32911）

**COMPLETED**（part_1 / node02 / 01:16:44 / exit 0 / MaxRSS ≈ 85 GB）。8 个 \(k\)，`nbnd=400`。JSON：`results/ref232_count_20260909/outer_window_count.json`。A1-O 52 个 200 带 \(wfc\) 仍在。

| 量 | 数 |
|---|---|
| 生产窗 \([-6,8]\) 的真实 \(N\) | **174–185**（不是 144） |
| \(N>232\) 的最低上沿 | 10.62 eV |
| 目标余量 299 的上沿 | **13.15 eV**（瓶颈在 \(\eta\)/H5/H7） |
| 冻结窗 \([-1.5,0.8]\) | 22–26 条，可沿用 |
| 第 400 带最低 | 14.62 eV（相对 13.15 还有 ~1.5 eV） |

**锁定（576k 前可改，计数已够用）：** `nbnd=400`。232 的外窗上沿是技术余量，不是物理能级：在 8 个 \(k\) 上 \(N>232\) 要 10.62 eV，余量 299 条要对到 13.15 eV，取 13.2 eV 只是给 disentangle 留空。\(112_{\mathrm{large}}\) **不需要** 299 条；它只要 \(N_{\mathrm{outer}}(k)>112\) 且有余量。`nbnd=400` 时真实 \([-6,8]\) 已有 174–185 条，对 112 已经够。

**历史文件复核（生产 `.win` / `.eig` / `.chk`，不是只凭 32911）：** 截断成立，32911 与生产 `.eig` 一致。

| 文件 | 事实 |
|---|---|
| `wannier/CrI3.win` | `num_wann=112`，`num_bands=200`，`dis_win=[-6,8]`，`dis_froz=[-1.5,0.8]`，无 `exclude_bands` |
| `inputs/CrI3_nscf_soc.in` | `nbnd=200` |
| `wannier/CrI3.eig`（576k×200） | 每个 \(k\) 恰好 56 条 \(E<-6\)、144 条落入 \([-6,8]\)；**没有任何** \(E>8\) 的本征值 |
| 第 56 / 57 / 200 带 | \(E_{56}\in[-12.09,-12.04]\)，\(E_{57}\in[-5.02,-4.69]\)，\(E_{200}\in[6.093,6.628]\) eV |
| `.chk` \(U_{\mathrm{dis}}\) | 每个 \(k\) 外窗固定 **QE57–200**（144 维） |
| 32911 \(\Gamma\) 第 200 带 | 6.4725 eV，与生产 `.eig` 的 \(\Gamma\) \(E_{200}=6.4725\) eV 一致 |

因此旧 Bloch 池的精确范围是 **QE 带 57–200 ≈ \([-5.02,\,6.63]\) eV**，不是完整 \([-6,8]\)。口头说 \([-6,6.5]\) 可以，但上沿在网格上可到 6.63 eV，下沿实际从 \(E_{57}\sim-5\) eV 开始（−6 eV 落在半芯与价带之间的空隙里）。

WINDOW144 是这个**截断池**里的 \(\Pi_{112}\oplus Q_{32}\)，不是「完整 \([-6,8]\) 的 144 维完备化」。它仍有诊断价值，但不能再读成已经把原设计外窗里缺的高能 Bloch 态全部补回。

顺序不变、且收紧：先做并验收 \(112_{\mathrm{large}}\)（仍是 Cr-\(d\)+I-\(p\) 112 条，`dis_win=[-6,8]` 现已被 `nbnd=400` 真正覆盖），**人看过之后再做 232 Wannier**。576k NSCF 在隔离目录 `ref112_large/`。一次算 576×400 已在作业 32972 OOM（~240 GB SIGKILL）；改为 **72×8 \(k\)**（与成功的 32911 同尺度：`part_1`、16 核、`-nk 2`、`disk_io=low`），再自动合并 xml/wfc 成一个 save 给 `pw2wannier90`。禁止写回 `temp_scf_soc` 与 A1-O save，禁止手工改 xml。
