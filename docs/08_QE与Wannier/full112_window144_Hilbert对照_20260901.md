# WINDOW144 Hilbert 完备性对照（2026-09-01）

```text
LABEL              = WINDOW144_HILBERT_COMPLETION_NOT_CHEMICAL
N_W                = 144 = 112 + 32
FROZEN             = production U_prod = U_dis U_inner
ADDED              = Q_32 energy + mmn parallel transport
GRID               = k20 classical +N only
NOT                = chemical I-d 144 ; GH7 ; k40 ; k50 ; Husimi-Q ; production parent
IMPACT_ON_HHG      = BASELINE_JQ_CHANGED_K20
INTERPRETATION     = Q_INTERP_VS_TRUE_LEAK_OPEN
FULL112_VERIFIED_FINAL_IP_REF = LOST
FULL112_QUANTITATIVELY_WRONG  = NOT_SHOWN
A0_CLOSES_INTERP   = false
SLURM              = 31678 COMPLETED (part_2)
```

本文件只回答一件事：

\[
\boxed{\text{加入缺失的 }Q_{32}\text{ 以后，}J_2,J_5,J_7,J_9\text{ 相对 full112 变多少？}}
\]

它 **不是** 新的化学轨道 parent，也 **不是** 大规模扩算许可。

---

## 0. 为什么现在做、仍然不大做

k20 真 \(\Pi r Q\) 已经给出直接算符证据：部分 \(k\) 上 full112 漏掉了会被场明显耦合的高能空间。最强点 \((0.10,0.55)\) / Wannier 62，基准场 \(\eta_r^{\max}=0.565\)，\(\Delta\varepsilon=0.392\,\mathrm{eV}\approx\hbar\omega_0\)。400 个 k 里 30 个 \(\eta>0.3\)；\(10^{12}\) 上 254/400 超过 0.3。权威：[full112_Pi_r_Q_eta审计_20260831.md](full112_Pi_r_Q_eta审计_20260831.md) §0。

因此：

\[
\boxed{
\begin{array}{ll}
2\times10^{11}: & \text{full112 的 Hilbert 完备性已经值得认真怀疑；}\\
10^{12}: & \text{full112 明显不能再安心当高场最终参考。}
\end{array}}
\]

\(\eta\) 只说明漏掉的通道**能被耦合**，不给出 H2/H7 误差条。所以对照必须看电流，但规模必须小：

- **做**：k20、经典、+N、两个场点。
- **不做**：expanded-parent \(\times\) k40 \(\times\) GH7，也不做默认 k50 / 正式 M88 GH5。

以前 set1 \(k=458\) 和 \(\Gamma\) 没找到问题，不是算错，而是规则分别找的是顶部占据热点和 112 内部光学热点，不是 \(112\to\) 外空间耦合热点。全 BZ 扫 \(\Pi r Q\) 之后逻辑已经闭合。

一句话：

\[
\boxed{\text{现在已经有理由测试扩空间，但还没有理由大规模扩算。}}
\]

---

## 1. 这个 144 是什么（以及不是什么）

| | WINDOW144 | 禁止写成 |
|---|---|---|
| 空间 | 外窗 \(W=[-6,8]\,\mathrm{eV}\) 的 Hilbert 完备化 \(\Pi_{112}\oplus Q_{32}\) | “I-\(d\) 144-WF 模型” |
| 112 列 | 生产 `CrI3.chk` 的 \(U_{\mathrm{prod}}=U_{\mathrm{dis}}U_{\mathrm{inner}}\)，**冻结** | 重新 MLWF 把原来 112 个函数也改掉 |
| 32 列 | 窗内补空间的能量本征基，再用 mmn 做生成树平行输运 | 预设化学壳层 120/128/144 |
| \(N_{\mathrm{occ}}\) | 仍是 84 | 把 Q 当成占据态 |

2026-08-19 设计备忘禁止的是：**未鉴定化学壳层之前不要生成新化学 parent**。本对照不解除那条禁令；它只生成一个**诊断用** Hilbert 完备化 TB。3.5–8 eV 试验投影与 I-\(d\) 重叠仍约 0.056，**不能**把 \(Q\) 叫成 I-\(d\)。

---

## 2. 构造与门禁

生成器：`tools/analysis/build_window144_hilbert_tb.py`  
测试：`tools/analysis/test_build_window144_hilbert_tb.py`  
产物（**不进 git**）：`量子光研究/01_输入与模型/wannier/window144_hilbert/CrI3_window144_tb.dat`  
门禁 JSON：`results/window144_hilbert_20260901/build_gate.json`

硬门（任一 FAIL 则不写 tb、不投 SBE）：

- `HERMITICITY`、`MESH_ROUNDTRIP`、`WINDOW144_VS_QE`（144 谱 vs 窗外窗 QE 本征值）、`PI_BLOCK`（112 块 = \(U_{\mathrm{prod}}^\dagger H U_{\mathrm{prod}}\)）、`U_ORTH_Q`

软门：`Q_LOCALIZATION`（\(Q\) 块 \(H(R)\) 在 \(|R|\ge12\) 的权重）。WARN 仍可写 tb，但对照结论必须带“Q 插值可能偏硬”的脚注。

分区：TB 构造是登录节点 / 本机后处理。本次 k20 对照因 part_1 无空闲改走 **part_2 / 8 核**（与 GH3-v2 k20 先例相同）。GH7 / k40 生产仍禁止，也不因此改回 part_1。

---

## 3. 对照协议（先 2e11，再一个高场节点）

| 任务 | \(I\) (W/cm²) | CEP | full112 参照 | 目的 |
|---|---:|---:|---|---|
| 0 | \(2\times10^{11}\) | \(90^\circ\) | 29458 baseline | 基准场 \(J_q\) 变多少 |
| 1 | \(1.00\times10^{12}\) | \(265.307^\circ\) | 29452_0 | 代表性高场节点 |

共同：k20、`lg_cov`、`dt=0.35`、\(T_2=0.5\) cycle、3200 nm、+N、`bsv_enabled=.false.`、pinned binary `34edab1d…`、`nb_end=144`、`nv_orig=84`。

脚本：

- `deploy/a0_layerA_m88full/sbatch_window144_hilbert_k20.sh`
- `deploy/a0_layerA_m88full/submit_window144_hilbert_k20.sh`（需 `WINDOW144_SUBMIT=I_UNDERSTAND_WINDOW144_K20_CLASSICAL_ONLY`）
- 比较：`tools/analysis/compare_window144_jq.py --dir-112 … --dir-144 …`

读谱：`HHG.dat` 的 \(|J|^2\) 峰（半宽 0.4 阶）；有 `HHG_complex.dat` 时加 \(|J|\) 比。

判读（预先写明，避免事后改口）：

三种结果都只针对 **普通** \(J_q(+N)\)，不是磁性 \(J_o\)：

- **基线和高场都几乎不变**：近共振 \(Q_{32}\) 通道存在，但对 H2/H5/H7/H9 影响有限。暂时**不用**重做化学 Wannier parent。
- **基线小、高场明显变**：最干净的分工——full112 对经典 \(2\times10^{11}\) 还能用，但不能覆盖 Husimi-\(Q\) 高场尾。
- **基线就明显变**：full112 连基准场 HHG 也要重新审视，后面才认真考虑更大物理 parent。

无论哪种，**都不**自动批准 k40/k50/GH7。

本轮只跑 \(+N\) 是对的，只筛普通 \(J_q\)。它**不能**关闭磁性

\[
J_o=\frac{J(+N)-J(-N)}{2}
\]

的 Hilbert 门：单域小误差可能在差分里被放大。若本轮差异很小，下一步是 **k20、同一两点场、\(\pm N\) 的 H2 \(J_o\) 小对照**，不是大规模扩算。

`Q_LOCALIZATION=WARN` 必须留下：若只差 1%–几%，不能立刻说“就是缺失 32 维”；若变化达到很明显的量级，这个警告通常不足以解释全部。

`IMPACT_ON_HHG` 在比较 JSON 写出来之前保持 `UNRESOLVED`。**2026-09-02 已写出**：见 §5。

---

## 4. 当前执行状态

| 步 | 状态 |
|---|---|
| 协议与脚本 | 本文件 |
| TB 门禁 | **OVERALL=PASS**，`Q_LOCALIZATION=WARN`（\(Q\) 块 \(\|R\|\ge12\) 权重 7.2%）。JSON：`results/window144_hilbert_20260901/build_gate.json`（`content_sha256_without_self=75b10183…`） |
| tb.dat | 本机已写，1 993 042 874 字节，sha256 `ece7958c9cd8ca48b061b440e8fbde0643008d249d25aed866ef776bf6b72a07`；**不进 git** |
| k20 SBE | part_1 **31677** 已 scancel。**31678** `[0-1]` **COMPLETED**（part_2：31678_0 node12 05:25:52；31678_1 node13 04:55:33）。两个子作业 `status=PASS`、`SUCCESS`、`nb_end=144`、binary `34edab1d…`、tb `ece7958c…`。OUTROOT：`.../output_a0_window144_k20_d1a143180153_part2` |
| \(J_q\) 比较 | **已完成**。JSON：`results/window144_hilbert_20260901/jq_compare_2e11.json`、`jq_compare_1e12.json`；判读：`jq_verdict_20260902.json` |
| 化学 144 / GH7 / k40 | **禁止**。WINDOW144 **不得**升格为生产 parent |

硬门摘要（网格点含 \(\Gamma\)、Wannier 62、458）：

- 厄米性、\(\Pi\perp Q\)、112 块 vs \(U_{\mathrm{prod}}\)：机器精度。
- 傅里叶往返 / 144 谱 vs 窗外窗 QE：\(\sim 2.5\times10^{-8}\,\mathrm{eV}\)。
- \(Q\) 块远程跳跃 WARN：H5/H7 已明显变，方向（「缺空间可能改 HHG」）成立；但 k20 ≠ 24×24 网格，**仍不能**把全部差值写成真 Hilbert 漏空间。见 §5。

服务器：生产 112 `CrI3_tb.dat` 本来就在（sha `66382a51…`）。WINDOW144 tb **本机生成后 scp**（服务器原先没有 144 文件），sha `ece7958c…` 与本机一致。

投递：

```bash
export WINDOW144_SUBMIT=I_UNDERSTAND_WINDOW144_K20_CLASSICAL_ONLY
export SCRIPT=/public/home/wangjs/project/CrI3_TB/wannier/window144_hilbert/sbatch_window144_hilbert_k20.sh
bash /public/home/wangjs/project/CrI3_TB/wannier/window144_hilbert/submit_window144_hilbert_k20.sh
```

已完成：**31678**（part_2）。判读见 §5。

---

## 5. 2026-09-02：\(J_q\) 对照结果（协议第三种）

两节点都正常结束，不是 cancelled / TIMEOUT。

| 子作业 | \(I\) | CEP | 节点 | 墙时 | `run_status` |
|---|---:|---:|---|---:|---|
| 31678_0 | \(2\times10^{11}\) | \(90^\circ\) | node12 | 5 h 26 min | PASS |
| 31678_1 | \(1.00\times10^{12}\) | \(265.307^\circ\) | node13 | 4 h 56 min | PASS |

预先写明的三种结果，本轮落在 **第三种**：

\[
\boxed{
\text{基线场 }2\times10^{11}\text{ 上 }J_5,J_7\text{ 已经明显变。}
\text{full112 已失去“已验证最终独立粒子参考”的资格，}
\text{但还不能说它已被定量证明是错的。}
}
\]

最干净的两个峰（峰位与整数阶一致，两种读谱法给出同一 \(|J|\) 比）：

| | \(2\times10^{11}\) \(\lvert J_{144}/J_{112}\rvert\) | \(10^{12}\) 同比 |
|---|---:|---:|
| H5 | **4.76** | 6.40 |
| H7 | **3.58** | 3.27 |

H2/H9 **不要**用协议默认的「±0.4 阶取峰」当误差条：full112 在精确 2 阶是局部极小（夹在 H1 连续谱里），WINDOW144 把这个坑填平了。更诚实的读法是取最接近整数阶的 bin：

| | \(2\times10^{11}\) \(\lvert J\rvert\) 比 | \(10^{12}\) |
|---|---:|---:|
| H2 @ 1.9993 | 9.87 | 11.75 |
| H9 @ 8.9968 | 11.29 | 17.04 |

\(\lvert J\rvert^2\) 比是上述数字的平方，看起来更夸张，讨论时用 \(\lvert J\rvert\)。

这 **不是** 新生产 parent。WINDOW144 仍是诊断用 Hilbert 完备化；`Q_LOCALIZATION=WARN`（QQ \(\lvert R\rvert\ge12\) 权重 7.2%，\(\Pi Q\) 块 14.6%），而 SBE 跑在 **k20**，门禁谱符合只在 **24×24 Wannier 网格**上验证过。因此解释仍开着：

\[
\boxed{
\text{数字} =\text{第三种结果；}
\quad
\text{机制} =\text{真漏空间 vs }Q\text{ 插值伪影，尚未分开。}
}
\]

标签：`IMPACT_ON_HHG = BASELINE_JQ_CHANGED_K20`，`INTERPRETATION = Q_INTERP_VS_TRUE_LEAK_OPEN`，`FULL112_QUANTITATIVELY_WRONG = NOT_SHOWN`。

H5/H7 是主要证据。H2/H9 的大倍数不进结论：full112 在那些整数频点接近局部极小，分母一小时倍数会爆。

**仍然禁止**：GH7、k40、k50、化学 I-\(d\) 144、把 144 升格为生产、Husimi-\(Q\) 生产。  
**下一步不是** \(\pm N\) 全套，也 **不是** 144→k40。两条线**并行**，B 不等 A：

- **A0**：24×24 内部一致性，关不了插值。  
- **A1-H**：离网格 QE 本征值，只关 \(H(k)\)。  
- **A1-O**：独立 \(r\)/\(v\)，才能对 HHG 插值作强结论。  
- **B**：候选机制（静态地图 + 总 \(J(t)\) 时频），不是最终归因。  

协议：[window144_k20插值独立性_20260902.md](window144_k20插值独立性_20260902.md)。A1-H PASS \(\neq\) WINDOW144 HHG 已可信。

磁性 \(J_o\) 的 Hilbert 门仍未关：本轮只有 \(+N\)。A1 读完之前不投新的 **SBE 生产**；A1 本身不是 SBE。

原始比较：`tools/analysis/compare_window144_jq.py`；机器判读：`results/window144_hilbert_20260901/jq_verdict_20260902.json`。
