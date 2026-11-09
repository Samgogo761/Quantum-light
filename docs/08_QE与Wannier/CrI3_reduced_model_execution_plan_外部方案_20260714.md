# 双层 AFM-CrI₃ 少带模型：40-WF 审计、关键 I-\(p\) 选择与 downfolding 执行方案

> **用途**：交给本地 AI 在 QE + Wannier 工作区中执行。  
> **当前阶段**：只处理电子结构、Wannier 子空间和少带模型构造；暂不开展量子光、QRT 与完整 HHG 验收。  
> **研究条件**：HT-AFM 双层 CrI₃，SOC；驱动光 \(\lambda=3200\) nm，\(I=2\times10^{11}\,\mathrm{W/cm^2}\)。  
> **参考模型**：QE200 与 112-WF Cr-\(d\)+I-\(p\)；候选模型：40-WF Cr-\(d\)-centered。

## 0. 本轮必须回答的核心问题

1. 当前 40-WF 模型到底是什么：纯原子 Cr-\(d\) 模型，还是包含 I-\(p\) 尾部的 Cr-\(d\)-centered effective Wannier 模型？
2. 40-WF 的总 spread = 263 Å² 是否真的已经足够局域？
3. 40-WF 价带偏差来自 frozen/outer window 设置、Cr-\(d\) projectability 不足、40 维子空间本身不够，还是 band matching / 能量零点问题？
4. 哪些 I-\(p\) 组合对层内磁交换、层间磁交换、目标能区能带、光学跃迁矩阵元和后续 HHG 最重要？
5. 是否能从 full112 的 `tb.dat` 构造 Cr-\(d\)+关键 I-\(p\) 中间模型、静态 Löwdin/downfolded d-only 模型，并给出模型评分与排序？

# 1. Phase A：完整审计当前 40-WF 结果

## A1. 收集并输出全部 Wannier 参数

从 40-WF 与 112-WF 的 `.win`、`.wout`、`.eig`、`.amn`、`.mmn` 中整理：

```text
num_bands
num_wann
spinors
dis_win_min / dis_win_max
dis_froz_min / dis_froz_max
dis_num_iter / dis_conv_tol / dis_mix_ratio
num_iter / conv_tol / conv_window
projections
exclude_bands
guiding_centres
write_hr / write_rmn / write_tb
```

产出：

```text
analysis/40wf_audit/input_parameter_comparison.md
analysis/40wf_audit/input_parameter_comparison.csv
```

## A2. 不能只比较 Omega_total：输出归一化 spread 指标

分别提取：

\[
\Omega_{\rm total},\quad
\Omega_I,\quad
\Omega_D,\quad
\Omega_{OD}.
\]

并计算：

\[
\bar\Omega_{\rm total}
=
\frac{\Omega_{\rm total}}{N_{\rm WF}}.
\]

同时输出每个 WF spread 的平均值、中位数、最大值、P90、P95，并按 Cr 原子、层、\(t_{2g}\)-like、\(e_g\)-like 分组；输出 Wannier center 与对应 Cr 原子位置的偏移。

特别比较：

```text
40-WF:  Omega_total / 40
112-WF: Omega_total / 112
```

不要再以“两个 Omega_total 数值同量级”判断局域化质量。

产出：

```text
analysis/40wf_audit/spread_summary.csv
analysis/40wf_audit/spread_histogram.png
analysis/40wf_audit/wannier_centres_vs_atoms.png
```

## A3. 判断 40-WF 是否包含 I-\(p\) 混合尾部

即使初始投影只有 `Cr:d`，最终 40 个 WF 也可能是 Cr-\(d\)-centered、但带有 I-\(p\) 杂化尾部的有效轨道。

至少执行一种分析：

1. Wannier 函数实空间等值面：选取代表性的 \(t_{2g}\)-like、\(e_g\)-like、上下层 WF，查看是否延伸至邻近 I。
2. 回投影 / orbital weight：若工具允许，计算
   \[
   w_{\alpha}^{I-p}
   =
   \langle w_\alpha|P_{I-p}|w_\alpha\rangle.
   \]

输出每个 40-WF 的 Cr-\(d\) weight、I-\(p\) weight，以及 I-\(p\) weight 在不同 I 原子、\(p_x/p_y/p_z\) 上的分布。

产出：

```text
analysis/40wf_audit/wf_orbital_weights.csv
analysis/40wf_audit/wf_orbital_weights_heatmap.png
```

## A4. 重新量化能带拟合，不只画图

参考 QE200，分别计算 112-WF 与 40-WF 在以下能区的误差：

```text
W_near  = [-1.5, +1.5] eV
W_mid   = [-2.0, +3.5] eV
W_HHG10 = 所有可能产生 DeltaE <= 10*hbar*omega0 的价-导带组合
W_HHG15 = 所有可能产生 DeltaE <= 15*hbar*omega0 的价-导带组合
```

其中：

\[
\hbar\omega_0 \approx 0.38745\ {\rm eV},
\]

\[
10\hbar\omega_0\approx3.87\ {\rm eV},\qquad
15\hbar\omega_0\approx5.81\ {\rm eV}.
\]

输出：

\[
\Delta E_{\rm RMS},\quad
\Delta E_{\max},\quad
E_g,\quad
\mathbf k_{\rm VBM},\quad
\mathbf k_{\rm CBM}.
\]

要求实现稳健 band matching，不能只按固定 band index 生硬比较。优先使用：

1. 本征态重叠最大匹配；
2. 若不可用，则使用能量连续性 + Hungarian algorithm；
3. 在简并带处比较子空间 projector，而非逐条带。

产出：

```text
analysis/40wf_audit/band_error_by_window.csv
analysis/40wf_audit/band_error_kresolved.png
analysis/40wf_audit/qe_112_40_overlay_target_window.png
```

## A5. 诊断 frozen/outer window 是否还能优化

对当前 40-WF 输入做 projectability 审计：

\[
p_{n\mathbf k}^{d}
=
\sum_{\alpha\in Cr-d}
|\langle\psi_{n\mathbf k}|g_\alpha\rangle|^2.
\]

输出 QE bands 在 \([-2,+6]\) eV 内的 Cr-\(d\) projectability。

据此判断：

- frozen window 内是否存在大量低 projectability 态；
- 目标价带是否没有被 frozen；
- outer window 是否缺少构造平滑 d-like 子空间所需的候选态；
- 是否适合启用 Wannier90 的 projectability disentanglement。

请准备至少三组 40-WF 窗口候选，但先不批量跑：

```text
40-opt-current：当前参数
40-opt-valence：提高目标价带拟合权重
40-opt-projectability：能量窗 + projectability 联合筛选
```

每组必须列明设计理由。

# 2. Phase B：从 full112 识别“关键 I-p”

## B1. 建立每个 I 原子的局域轨道坐标

不要默认全局 \(p_z\) 就是关键轨道。

对每个 I 原子建立局域正交基：

\[
\hat e_\sigma
=
\frac{\mathbf R_{\rm Cr}-\mathbf R_I}
{|\mathbf R_{\rm Cr}-\mathbf R_I|},
\]

\[
|p_\sigma\rangle
=
e_{\sigma x}|p_x\rangle+
e_{\sigma y}|p_y\rangle+
e_{\sigma z}|p_z\rangle.
\]

再构造两个与 \(\hat e_\sigma\) 正交的方向：

\[
|p_{\pi1}\rangle,\qquad |p_{\pi2}\rangle.
\]

对于界面 I-I 通道，另建立：

\[
\hat e_{\sigma,\mathrm{inter}}
=
\frac{\mathbf R_{I_2}-\mathbf R_{I_1}}
{|\mathbf R_{I_2}-\mathbf R_{I_1}|}.
\]

产出：

```text
analysis/key_Ip/local_orbital_frames.json
analysis/key_Ip/local_orbital_frames_visualization.png
```

## B2. 计算 Cr-d / I-p hopping 排名

从 112-WF `tb.dat` 或 `hr.dat` 提取：

```text
Cr(t2g)-I(p_sigma)
Cr(t2g)-I(p_pi1/pi2)
Cr(eg)-I(p_sigma)
Cr(eg)-I(p_pi1/pi2)
interface I(p)-I(p)
```

对每类通道输出：

\[
|t_{\alpha\beta}(\mathbf R)|,
\qquad
\sum_{\mathbf R}|t_{\alpha\beta}(\mathbf R)|^2.
\]

按层内/层间、上层/下层、内侧 I/外侧 I、\(t_{2g}/e_g\)、\(p_\sigma/p_\pi\) 分组。

产出：

```text
analysis/key_Ip/hopping_rank.csv
analysis/key_Ip/hopping_network.png
```

## B3. 用 SVD 选“一个 I 的最佳 p 方向”

若目标是“每个 I 只保留一个 p 组合”，不要人工指定 \(p_x,p_y,p_z\)。

对某个 I 的三维 p 子空间构造与目标 Cr-d 子空间的 hopping 矩阵：

\[
T_{d,p}^{(I)}.
\]

做奇异值分解：

\[
T_{d,p}^{(I)}
=
U\Sigma V^\dagger.
\]

右奇异向量 \(V_1\) 对应：

\[
|p_{\rm opt}^{(I)}\rangle
=
V_{x1}|p_x\rangle+
V_{y1}|p_y\rangle+
V_{z1}|p_z\rangle,
\]

即与目标 Cr-d 子空间耦合最强的一个 I-p 线性组合。

分别针对：

```text
目标 1：层内 Cr-d 耦合
目标 2：层间 I-I / Cr-d 耦合
目标 3：near-gap Bloch 子空间
```

产出：

```text
analysis/key_Ip/p_opt_vectors.csv
analysis/key_Ip/p_opt_singular_values.csv
```

## B4. 目标能区中的 I-p 权重

对 QE 或 full112 Bloch 本征态计算：

\[
w_{n\mathbf k}^{I,p_\alpha}
=
\langle u_{n\mathbf k}|P_{I,p_\alpha}|u_{n\mathbf k}\rangle.
\]

重点输出：

```text
VBM 到 VBM-若干带
CBM 到 3.5 eV 导带
所有 DeltaE <= 5.81 eV 的主要价-导带对
```

产出：

```text
analysis/key_Ip/orbital_weight_kresolved.h5
analysis/key_Ip/orbital_weight_bandplot.png
analysis/key_Ip/orbital_weight_summary.csv
```

# 3. Phase C：构造候选中间模型

待 Phase A/B 完成后再实际运行。

## C1. 候选模型

### M40

```text
40 spinor WF
Cr-d-centered only
用途：最简基准
```

### M52-interface

```text
40 Cr-d + 6 个界面 I × 1 个 p_opt × 2 spin = 52 WF
用途：优先保留层间磁交换通道
```

### M64-all-I-1p

```text
40 Cr-d + 12 个 I × 1 个 p_opt × 2 spin = 64 WF
用途：每个 I 保留一个最强 p 组合
```

### M88-all-I-2p

```text
40 Cr-d + 12 个 I × 2 个 p_opt × 2 spin = 88 WF
用途：保留主要 sigma/pi 双通道
```

### M-energy

```text
维度由 projectability / frozen window 决定
目标：优先复现指定 HHG 能量窗口中的 composite bands
```

不得在 Phase A/B 结果出来前认定哪个模型最好。

# 4. Phase D：full112 `tb.dat` 的 downfolding 试验

## D1. 静态 Löwdin 诊断代码

从 full112 Wannier 轨道基中划分：

\[
A=\text{active 子空间，例如 Cr-d},
\qquad
B=\text{被消去的 I-p 子空间}.
\]

对每个 \(\mathbf k\) 构造：

\[
H(\mathbf k)
=
\begin{pmatrix}
H_{AA} & H_{AB}\\
H_{BA} & H_{BB}
\end{pmatrix}.
\]

实现：

\[
H_{\rm eff}(E_0,\mathbf k)
=
H_{AA}
+
H_{AB}
(E_0+i\eta-H_{BB})^{-1}
H_{BA}.
\]

扫描参考能量：

```text
E0 = VBM
E0 = gap center
E0 = CBM
E0 = 1, 2, 3 eV above VBM
```

该模型先作为电子结构诊断，不直接进入 HHG。

产出：

```text
tools/lowdin_downfold.py
analysis/downfold/band_comparison_E0_*.png
analysis/downfold/downfold_error.csv
```

## D2. 需要明确的限制

本阶段只 downfold Hamiltonian：

\[
H_{\rm eff}.
\]

不能默认光学位置算符等于裸投影：

\[
r_{\rm eff}\neq P_A r P_A.
\]

后续若用于 HHG，必须通过同一有效变换构造：

\[
r_{\rm eff},
\qquad
v_{\rm eff},
\qquad
J_{\rm eff}.
\]

因此本地 AI 需在代码注释和报告中明确：

```text
Hamiltonian-only downfolding = band diagnostic
not yet an HHG-ready model
```

# 5. 模型评分代码

实现统一评分脚本：

```text
tools/score_reduced_model.py
```

总评分：

\[
\mathcal L_{\rm total}
=
w_E\mathcal L_E
+
w_{\rm sub}\mathcal L_{\rm sub}
+
w_{\rm loc}\mathcal L_{\rm loc}
+
w_{\rm hop}\mathcal L_{\rm hop}
+
w_{\rm opt}\mathcal L_{\rm opt}
+
w_{\rm HHG}\mathcal L_{\rm HHG}.
\]

本轮先实现前四项。

### 能带误差

\[
\mathcal L_E
=
{\rm RMS}_{\mathbf k,n}
\left(
E^{\rm red}_{n\mathbf k}
-
E^{\rm ref}_{n\mathbf k}
\right).
\]

### 子空间误差

\[
\mathcal L_{\rm sub}
=
{\rm RMS}_{\mathbf k}
\left\|
P_{\rm red}(\mathbf k)
-
P_{\rm ref}(\mathbf k)
\right\|_F.
\]

### 局域化代价

\[
\mathcal L_{\rm loc}
=
\frac{\Omega_{\rm total}}{N_{\rm WF}}
+
\lambda_{\max}\max_\alpha\Omega_\alpha.
\]

### 关键 hopping 损失

\[
\mathcal L_{\rm hop}
=
1-
\frac{
\sum_{(\alpha\beta)\in{\rm kept}}
|t_{\alpha\beta}|^2
}{
\sum_{(\alpha\beta)\in{\rm full}}
|t_{\alpha\beta}|^2
}.
\]

预留但暂不在本轮执行：

```text
L_opt：光学矩阵元 / 吸收 / f-sum
L_HHG：复电流与谐波谱
```

权重必须可由 YAML 配置：

```yaml
weights:
  energy: 0.35
  subspace: 0.25
  localization: 0.15
  hopping: 0.25
  optical: 0.0
  hhg: 0.0
```

输出：

```text
analysis/model_score/model_ranking.csv
analysis/model_score/radar_plot.png
analysis/model_score/model_score_report.md
```

# 6. 本轮代码清单

本地 AI 请建立：

```text
tools/
├── parse_wannier_audit.py
├── compare_bands_subspace.py
├── build_local_p_frames.py
├── rank_dp_hoppings.py
├── select_p_by_svd.py
├── lowdin_downfold.py
└── score_reduced_model.py
```

统一要求：

```text
- 支持命令行参数
- 支持 YAML 配置
- 输出 CSV + PNG + Markdown 报告
- 不覆盖原始 QE/Wannier 文件
- 每一步可独立运行
- 写清单位、能量零点、band matching 方法
- 保存 Git commit 前运行最小测试
```

# 7. 推荐执行顺序

```text
Step 1  解析 40/112 的 .win/.wout，生成完整参数和 spread 对比
Step 2  完成 band matching 与能区误差统计
Step 3  计算 QE bands 对 Cr-d 的 projectability
Step 4  判断 40-WF 是否值得继续调 outer/frozen window
Step 5  从 full112 建立 I-p 局域坐标和 hopping 排名
Step 6  用 SVD 得到每个 I 的 p_opt
Step 7  给出 M52/M64/M88/M-energy 输入草案
Step 8  实现 Hamiltonian-only Löwdin downfolding
Step 9  用统一评分函数排序现有 M40 和 downfold 模型
Step 10 等用户确认后，再实际跑新的 Wannier 候选模型
```

# 8. 本地 AI 第一轮只需返回的结果

请不要一次性开始所有重计算。第一轮只返回：

1. 40-WF 和 112-WF 的完整 spread 表，尤其是每 WF 平均值与最大值；
2. 40-WF 当前 frozen/outer window 与投影配置；
3. \([-2,+3.5]\) eV 内 QE/112/40 的 RMS 与最大能带误差；
4. QE bands 对 Cr-d 的 projectability 图；
5. 40-WF 是否含明显 I-p 空间尾部；
6. 是否具备从 full112 `tb.dat` 读取轨道标签、\(H(R)\)、\(r(R)\) 的现成代码；
7. 上述 7 个脚本的实现工时估计和依赖文件清单。

# 9. 当前阶段的停止条件

在以下结果出现前，不要直接宣布 40-WF 可用于 HHG：

```text
- 价带偏差原因已确定；
- spread 已按每 WF 归一化审计；
- Cr-d projectability 已量化；
- 关键 I-p 通道已排序；
- Hamiltonian-only downfolding 与 HHG-ready downfolding 已明确区分。
```
