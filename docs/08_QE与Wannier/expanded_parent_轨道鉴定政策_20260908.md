# 扩展轨道鉴定政策（2026-09-08）

```text
LINE                    = M
DECISION                = SCHEME_A_ORBITAL_ID_NO_NEW_PARENT
ACTIVE_MODEL_GATE       = MEDIUM_POLICY_NOT_EXECUTED
WINDOW                  = 3.5-8 eV OR Q32, weighted by |Pi r Q| or eta
NOT                     = new Wannier parent ; chemical I-d 144 ; GH7 ; k40 ; WINDOW144 production
NEXT_HUMAN_GATE         = which complete shell(s), or re-disentangle, after W_S/G_S
```

用户批准（2026-09-08）：先做扩展轨道鉴定，不生成新 Wannier parent。  
active model 中等验收门先写成政策，等大 parent 的经典 \(\pm N\) 结果后再选 \(\rho_{mn}\) 通道。

---

## 1. 批准的两件事

\[
\boxed{\text{先鉴定轨道，不生成新化学 parent。}}
\]

\[
\boxed{
J(t)+H2/H5/H7\text{ 幅度与相位}+J_e/J_o+\text{少量关键 }\rho_{mn}(t)
\quad\text{（政策，暂不执行）}
\]

\(\rho_{mn}\) 追哪几对现在不定。那是路径追踪节点，要等大 parent 经典结果再审。

## 2. 鉴定范围

- 能量：整个 \(Q_{32}\)，并平行报告 3.5–8 eV。
- 加权：\(\lvert\Pi r Q\rvert\) 列权重（主）或 \(\eta\)。
- 通道：Cr-\(s/p/d\)、I-\(s/p/d\)。
- 若 Cr-\(3d\)/I-\(5p\) 在 \(Q\) 里仍然很强：优先考虑重新设计 disentanglement / frozen window，而不是默认加 I-\(d\)。

## 3. 工具核证（不是猜）

生产赝势（A1-H 门，sha256 一致）：

| UPF | `PP_CHI`（`projwfc` 能看见） | `PP_BETA`（不是 projwfc 通道） |
|---|---|---|
| `Cr.rel-pbe.upf` | 3S, 4S, 3P, 3D | 另有 **4P** |
| `I.rel-pbe.upf` | 5S, 5P **only** | 另有 **5D** |

因此：`projwfc` **不能**正鉴定 I-\(5d\)，也 **不能**正鉴定 Cr-\(4p\)。I-\(d\) / Cr-\(p\) 走 Wannier 试探 AMN。

QE 7.2 `pw2wannier90` `radialpart`（与 A1-O 同一二进制）：

- `rvalue=1`，`zona=alpha=1.0 bohr^{-1}`（与现有 112 试探相同）
- \(R(r)=2\alpha^{3/2}e^{-\alpha r}\)
- **不是**光谱学 I \(5d\) 原子径向，也不是 UPF 里的 `PP_CHI`

`rvalue=2,3` 是带径向节点的氢似函数，本轮不用，以免和现有 112 试探换径向。

## 4. 计算

隔离目录仍是 `window144_hilbert/a1_fewk_nscf/`。用 A1-O 已有 52 个 \(k\) 波函数，只算 AMN。  
作业：`sbatch_shell_trial_amn_part2.sh`（part_2，**32491 COMPLETED**，exit 0:0，00:24:52，node12）。  
JSON：`results/mechanism_b_20260902/q32_shell_trial_screen.json`。**不**生成新 TB，**不**升格 WINDOW144。

32491 筛选（氢似试探，不是正交百分比）：四个 \(k\) 上 \(Q_{32}\) 与 3.5–8 eV 都是 32 态。六个壳层里 **I-\(d\) 试探最大**（\(r_x\) 加权约 0.15–0.18），Cr-\(d\)+I-\(p\) 合计只有约 0.05。但六壳层 \(|A|^2\) **总和只有约 0.25**，约 3/4 没被这组氢似试探抓住。这还不够单独宣布「就是 I-\(5d\)，去做 232」。

筛选指标 \(W_S,G_S\) 不是正交百分比。某一壳层明显占优之后，才构造真正的 Wannier projector 去验证——那一步仍要用户看。

---

## 5. 2026-09-08 工作假设与径向稳健性

用户侧决定（外部建议，本地采纳执行边界）：

- **执行上选 B**：暂不生成 232 带 parent。
- **物理工作假设取 C**：光学加权的 \(Q\) 上 I-\(d\) 是领先候选，但 \(Q\) 混合、可能弥散；\(\Gamma\) 低端 \(Q\) 的 Cr-\(s\) 只当线索，不是已鉴定成分。
- Cr-\(d\)+I-\(p\) \(\sim 0.05\) 使「只是 disentangle 丢掉已有 \(d+p\)」变弱，**不等于**已经认出新壳层就是光谱学 I \(5d\)。
- 「75% 没抓住」= 当前氢似试探投影能力不足，不是 75% 电子态完全未知。

径向稳健性（仍是鉴定，不是新 parent）：在同一 52-\(k\) 波函数上加三组试探

- \(r=1,\mathrm{zona}=0.5\)（更弥散）
- \(r=1,\mathrm{zona}=2.0\)（更紧）
- \(r=2,\mathrm{zona}=1.0\)（一个径向节点）

看 \(\eta\)/H5/H7 上 \(\mathrm{I}\!-\!d\) 是否仍稳定领先，以及总覆盖是否明显提高。结果出来后由用户决定：232 参考 parent，还是放弃纯原子壳层、改更一般的子空间构造。**本地 AI 不得自动开工 232 或 SCDM。** 径向作业：**32568 COMPLETED**（part_2，node12，exit 0:0，01:15:45）。JSON：`results/mechanism_b_20260902/q32_shell_radial_robust.json`。

预注册的「所有径向下热点 \(k\) 都是 I-\(d\) 领先」**未通过**（`hot_k_I_d_always_leader=false`）。拆开看：

- 弥散 \(r=1,\mathrm{zona}=0.5\)：\(\eta\)/H5/H7 仍是 I-\(d\)，覆盖从 \(\sim 0.27\) 升到 \(\sim 0.89\)–\(0.97\)。
- 紧 \(r=1,\mathrm{zona}=2.0\)：领先变成 Cr-\(d\)，I-\(d\) 掉到 \(\sim 0.005\)。
- \(r=2,\mathrm{zona}=1.0\)：热点仍 I-\(d\)，但六壳层 \(|A|^2\) 之和 \(>1\)（非正交过计），\(\Gamma\) 上 I-\(s\) 也很强。

因此：**更弥散的 I-\(d\) 试探能抓住光学活跃 \(Q\)，但排序对径向并不无条件稳定；不能用这一轮自动批准 232。**

---

## 6. 2026-09-08 方案 C：一次有物理定义的 projector 验证

用户决定更接近 **C**：不直接做 232（A），也不立刻改 projectability/SCDM（B）。

工作假设（尚未闭合）：光学活跃 \(Q\) 偏弥散，并带明显 I-\(d\) 特征；化学身份尚未闭合。不要写成「\(Q\) 就是 I-\(d\)」。

本轮 **只做一次**，禁止再扫 `zona` / `rvalue`：

- 生产 QE 7.2 **没有** `atom_proj`；I UPF 的 `PP_CHI` 仍无 5D。
- I-\(d\) 径向 = 生产 `I.rel-pbe.upf` 里第一组有限的 5D PAW 赝波（`PP_PSWFC` 7/8，\(j=3/2,5/2\) 平均）。高能 5D 通道 9/10（径向爆炸）不用。
- 其余五壳层用同一套 UPF：Cr `4S`/`3P`/`3D` CHI，I `5S`/`5P` CHI（\(p,d\) 做 SO 平均）。Cr `3S` 半芯不进这六壳层。
- 重叠仍走 A1-O 的 52 \(k\) 波函数 + 现有 MMN 加权；私有修补 `pw2wannier90` 只改 `radialpart`，**不覆盖**生产 `bin/pw2wannier90.x`。
- 只看 \(\eta\)/H5/H7：I-\(d\) 是否仍领先、是否明显高于 Cr-\(d\)/Cr-\(p\)/I-\(p\)、领先是否集中在 \(\lvert\Pi r Q\rvert\) 大的 \(Q\)。

结果出来后 **停**，由用户在「232 参考 parent（仅 k10/k20 经典）」与「放弃硬贴 I-\(d\)，改 projectability/SCDM」之间决定。本地不得自动开工 232 或 SCDM。

作业：**32584**（part_2，node12）。私有 `pw2wannier90_upf.x` 编译成功；生产 `bin/pw2wannier90.x` 未覆盖。AMN 日志含 `UPF_CUSTOM_RADIAL_LOADED n_type=6 mesh=333 n_proj=288`。JSON：`results/mechanism_b_20260902/q32_shell_upf_c.json`。

三个预注册问题（\(\eta\)/H5/H7，\(r_x\) 加权）：

| 问题 | 结果 |
|---|---|
| I-\(d\) 是否仍领先 | **是**（`hot_k_I_d_leader=true`） |
| 是否明显高于 Cr-\(d\)/Cr-\(p\)/I-\(p\) | **是**：对最大竞争者（I-\(p\)）约 7.3–8.6 倍；Cr-\(p\) 可忽略 |
| 领先是否集中在 \(\lvert\Pi r Q\rvert\) 大的 \(Q\) | **是**：top-8 光学权重带全部是 I-\(d\) 赢家（权重份额 1.0） |

径向来源指纹：I-\(d\) \(\langle r\rangle=5.32\) bohr，Cr-\(d\) \(1.63\)，I-\(p\) \(2.39\)。这组 UPF 5D 分波本身就偏弥散。

六壳层覆盖仍只有约 \(0.26\)–\(0.28\)（与氢似 \(z=1\) 基线同量级）。因此：**光学活跃、被抓住的那一块稳定偏 I-\(d\)**；**仍不能说缺的就是完整 I-\(5d\) 壳层**。

用户已批准同窗对照并进入工程（2026-09-09）：\(112_{\mathrm{old}}\to112_{\mathrm{large}}\to232_{\mathrm{I-d}}\)。比较用 QE 波函数 overlap 映射同一物理 Bloch 子空间，不用旧 \(Q\) 带号；H5/H7 同向变化只作证据，不是成功门；外窗 \(N_{\mathrm{outer}}>232\)。当前只投隔离计数 NSCF，尚未 576k / 新 Wannier / SBE。见 [232_reference_parent_同窗对照设计_20260908.md](232_reference_parent_同窗对照设计_20260908.md)。下一轮人审在 \(112_{\mathrm{large}}\) 与 232 结果出来之后。
