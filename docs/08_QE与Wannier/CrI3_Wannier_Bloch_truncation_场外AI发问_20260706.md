---
title: CrI3 少带化与 Bloch/Wannier 截断确认方案
date: 2026-07-06
status: 给本地 AI / 导师讨论用
topic:
  - CrI3-AFM
  - QE
  - Wannier90
  - SBE
  - HHG
  - band truncation
  - sum rule
---

# CrI₃ 少带化与 Bloch/Wannier 截断确认方案

## 0. 结论先行

这份文档用于向本地 AI 确认三个问题：

1. 师兄说的「**截断 Wannier 基有问题，截断 Bloch 基相对没问题**」到底是什么意思；
2. 已经做过的**对称窗口截断诊断**是否足够支持下一步决策；
3. 若老师和师兄都建议重做少带模型，应该如何设计一个**少带 Wannier / downfolding 试点**，并如何处理真实跃迁、虚跃迁和求和规则问题。

我的当前判断是：

> **不能直接删除 Wannier 轨道/矩阵行列作为生产模型。**  
> 如果要少带化，优先区分三种操作：
>
> 1. **Bloch 能带截断**：先在完整 112-WF Hamiltonian 中对角化，得到 Bloch 本征能带，再选取某个能带窗口进入 SBE；
> 2. **Wannier 轨道截断**：直接删掉某些 Wannier orbital / HR 矩阵行列，这一般不推荐；
> 3. **重做少带 Wannier / downfolding**：重新构造少带有效模型，这是物理上更干净但需要完整重验的路线。

若老师明确希望尝试「费米面附近少带 HHG」，建议把它作为一个**受控试点**，而不是替代当前 112-band 母模型的生产路线。

---

# 1. 先解释：Wannier 基截断 vs Bloch 基截断

## 1.1 112-WF 模型中的两个基底

当前 112-band Wannier 模型本质上有两个常用表示。

### A. Wannier orbital basis

这是局域轨道基：

\[
|w_{\alpha\mathbf R}\rangle
\]

其中 \(\alpha\) 表示 Cr-\(d\)、I-\(p\)、spinor 等轨道指标，\(\mathbf R\) 是晶格矢量。

在 \(k\)-空间中，Hamiltonian 是：

\[
H^{W}_{\alpha\beta}(\mathbf k)
=
\sum_{\mathbf R}
e^{i\mathbf k\cdot\mathbf R}
H^{W}_{\alpha\beta}(\mathbf R).
\]

这里上标 \(W\) 表示 Wannier basis。

**重要点**：Wannier 轨道不是能量本征态。一个 Wannier orbital 可能参与多个能带，一个能带也通常由多个 Wannier orbital 混合而成。

---

### B. Bloch eigenbasis / band basis

对完整 \(112\times112\) 的 \(H^W(\mathbf k)\) 做对角化：

\[
U^\dagger(\mathbf k) H^W(\mathbf k) U(\mathbf k)
=
\varepsilon(\mathbf k),
\]

其中：

\[
\varepsilon(\mathbf k)
=
{\rm diag}(\varepsilon_1,\varepsilon_2,\ldots,\varepsilon_{112}).
\]

得到的本征态：

\[
|u_{n\mathbf k}\rangle
=
\sum_\alpha
|w_{\alpha\mathbf k}\rangle U_{\alpha n}(\mathbf k).
\]

这就是 Bloch eigenbasis / band basis。

---

## 1.2 什么叫「截断 Wannier 基」？

如果直接删掉某些 Wannier 轨道，比如只保留 Cr-\(d\)，删除 I-\(p\)，相当于：

\[
H^W(\mathbf k)
\rightarrow
P_W H^W(\mathbf k) P_W,
\]

其中 \(P_W\) 是 Wannier orbital 子空间投影。

这不是简单「去掉远能带」，而是**改变了局域轨道空间本身**。

问题是：

1. Wannier orbital 不是能量本征态；
2. 删除 I-\(p\) 会改变 Cr-\(d\) 与 I-\(p\) 杂化；
3. 位置矩阵、Berry connection、dipole matrix element 也会改变；
4. 如果只是删 HR/tb.dat 的行列，而没有重新 downfolding，远带效应没有被折叠回来；
5. 这会破坏原始 112-WF 模型的光学矩阵元和规范结构。

因此师兄说「截断 Wannier 基有问题」，大概率指的是：

> **不要直接从 Wannier orbital basis 中删轨道/删矩阵行列，然后假装它仍然是原 full model 的可靠少带模型。**

---

## 1.3 什么叫「截断 Bloch 基」？

Bloch 截断通常是：

1. 先保留完整 112-WF Hamiltonian；
2. 每个 \(\mathbf k\) 点先对角化完整 \(H^W(\mathbf k)\)；
3. 得到完整的 Bloch bands；
4. 再选择能带窗口，例如 75–94、65–104 或 1–104；
5. 在这个 Bloch band subspace 中传播 SBE。

数学上是：

\[
H^B_{mn}(\mathbf k)
=
\varepsilon_n(\mathbf k)\delta_{mn},
\quad
m,n\in \mathcal W_{\rm act}.
\]

其中 \(\mathcal W_{\rm act}\) 是选中的 Bloch band window。

这个做法比 Wannier orbital 截断更可控，因为：

- 被保留的是能量本征态；
- 能直接定义价带、导带、带隙、跃迁能；
- 可以逐步扩大窗口并做收敛测试；
- 可以直接和 full112 的同一 band index 对比；
- 光学矩阵元可先从完整 Wannier 模型变换到 Bloch basis 后再取子块。

但是，**Bloch 截断也不是绝对没有问题**。它仍然会丢掉远带对求和规则、虚跃迁、Berry/dipole 完备性的贡献。因此更准确的说法是：

> **Bloch 能带截断不是严格无误，而是比直接删 Wannier 轨道更有定义、更容易收敛测试、更容易判断误差。**

---

# 2. 对 SBE-HHG 来说，为什么这件事很重要？

## 2.1 传统 HHG 谱看的是平均电流

SBE 计算的输出通常是：

\[
J^a(t)
=
{\rm Tr}\left[j^a(t)\rho(t)\right].
\]

然后做 Fourier transform：

\[
J^a(\omega)
=
\int dt\, e^{i\omega t}J^a(t),
\]

谐波谱大致为：

\[
S(\omega)\propto |J(\omega)|^2.
\]

如果只保留费米面附近少数 Bloch bands，确实可能能复现低中阶 HHG，尤其当深价带真实跃迁概率很小时。

---

## 2.2 但远带/深价带仍可能通过两条路径影响结果

### A. 真实跃迁贡献

真实跃迁由能量差和矩阵元共同决定：

\[
\Delta E_{vc}(\mathbf k)
=
\varepsilon_c(\mathbf k)-\varepsilon_v(\mathbf k).
\]

若：

\[
\Delta E_{vc}(\mathbf k)
\approx
q\hbar\omega_0,
\]

且跃迁偶极矩阵元：

\[
d^a_{vc}(\mathbf k)
\]

不小，则该跃迁可能贡献第 \(q\) 次谐波附近的响应。

所以判断深价带是否参与，不能只看“离费米面远”，还要看：

1. 驱动光子能量 \(\hbar\omega_0\)；
2. 目标谐波阶数 \(q\)；
3. \(\Delta E_{vc}(\mathbf k)\) 是否落在 \(q\hbar\omega_0\) 附近；
4. 矩阵元 \(|d^a_{vc}(\mathbf k)|^2\) 是否显著；
5. 相关 \(k\)-点是否在强场动力学中被访问。

本项目中 \(3200\) nm 对应的单光子能量约为：

\[
\hbar\omega_0 \approx 0.387\ {\rm eV}.
\]

因此：

\[
H5 \approx 1.94\ {\rm eV},
\quad
H9 \approx 3.49\ {\rm eV},
\quad
H11 \approx 4.26\ {\rm eV},
\quad
H15 \approx 5.81\ {\rm eV}.
\]

这意味着即使驱动光子能量不大，高阶谐波对应的辐射能量仍可能覆盖深价带到高导带的能量差。

---

### B. 虚跃迁 / 求和规则 / 规范完备性贡献

即使远带不真实激发，它们也可能通过虚跃迁参与抵消关系。

在完整空间中有完备性：

\[
\sum_n |u_{n\mathbf k}\rangle\langle u_{n\mathbf k}|=1.
\]

若截断为 active bands：

\[
\sum_{n\in \mathcal W_{\rm act}}
|u_{n\mathbf k}\rangle\langle u_{n\mathbf k}|
\neq 1.
\]

这会影响：

- 速度矩阵与位置矩阵的关系；
- \(f\)-sum rule / TRK sum rule；
- 顺磁与抗磁项抵消；
- Berry connection 的非阿贝尔完备性；
- 基态电流是否为零；
- 低阶谐波是否出现伪增强。

尤其在 velocity gauge 中，有限能带截断更容易出现不物理结果。当前项目已经改用 `lg_cov` length gauge，这是更稳的选择，但 length gauge 也仍然需要窗口收敛验证。

---

# 3. 需要向本地 AI 确认的第一个问题：当前代码到底截断了哪个基？

请本地 AI 明确回答：

## Q1. 当前 SBE 的 `nb_start/nb_end` 是在 Bloch eigenbasis 截断，还是在 Wannier orbital basis 截断？

请检查代码流程，并用伪代码说明。

理想流程应该类似：

```fortran
! 1. read full 112-WF H(R), r(R)
call read_tb(full_tb)

! 2. Fourier transform to H_W(k), r_W(k)
call build_H_W(k)
call build_r_W(k)

! 3. diagonalize full H_W(k)
call diagonalize(H_W, U, eps)

! 4. transform r/current/Berry to Bloch basis
r_B = U^dagger r_W U + gauge_connection_terms
j_B = transform_current(...)

! 5. select Bloch band window nb_start:nb_end
r_active = r_B(nb_start:nb_end, nb_start:nb_end)
j_active = j_B(nb_start:nb_end, nb_start:nb_end)
rho_active = rho(nb_start:nb_end, nb_start:nb_end)

! 6. propagate SBE in active Bloch band space
call propagate_sbe(rho_active)
```

如果代码实际做的是：

```fortran
H_W_active = H_W(nb_start:nb_end, nb_start:nb_end)
```

那就是 Wannier/orbital sub-block 截断，这就需要警惕。

---

## Q2. `lg_cov` 中的 Berry connection / dipole matrix 是如何从 full 112 模型得到的？

请确认：

1. 是否先在完整 112 空间中计算 \(A^a_{mn}(\mathbf k)\)、\(d^a_{mn}(\mathbf k)\)、\(j^a_{mn}(\mathbf k)\)，再截 Bloch band 子块；
2. 还是只在 active subspace 内重新计算；
3. 是否包含 \(iU^\dagger\partial_k U\) 项；
4. 是否使用 covariant derivative：

\[
D_a\rho
=
\partial_{k_a}\rho-i[A_a,\rho].
\]

请给出代码中的函数名/文件名。

---

# 4. 需要向本地 AI 确认的第二个问题：对称窗口截断诊断是否足够？

我知道对称窗口截断诊断已经做过，但目前需要把结果整理成可判断表格。

请本地 AI 输出以下内容。

## Q3. 已有对称窗口有哪些？

请列出完整表格：

| window 名称 | band index | 价带数 | 导带数 | 能量范围 | 是否已有 HHG |
|---|---:|---:|---:|---:|---|
| W20 | 75–94 | 10 | 10 | ? | yes/no |
| W30 | 70–99 | 15 | 15 | ? | yes/no |
| W40 | 65–104 | 20 | 20 | ? | yes/no |
| W50 | 60–109 | 25 | 25 | ? | yes/no |
| W60 | 53–112 | ? | ? | ? | yes/no |
| nb1-104 | 1–104 | 84 | 20 | ? | yes/no |
| full112 | 1–112 | 84 | 28 | ? | yes/no |

---

## Q4. 对每个窗口，输出 H1–H15 的强度比

请输出：

\[
R_q(W)
=
\frac{S_q(W)}{S_q({\rm full112})}.
\]

表格格式：

| q | W20 | W30 | W40 | W50 | W60 | nb1-104 |
|---:|---:|---:|---:|---:|---:|---:|
| H1 | | | | | | |
| H2 | | | | | | |
| ... | | | | | | |
| H15 | | | | | | |

请同时标注 full112 的绝对强度：

\[
S_q({\rm full112}).
\]

原因：若 full112 本身已接近数值地板，则相对误差会被放大。

---

## Q5. 输出复数 Fourier 分量，而不只是强度

请至少输出：

\[
J_x(q\omega_0),\quad J_y(q\omega_0)
\]

对于：

\[
q=1,2,3,5,7,9,11.
\]

表格格式：

| q | window | \(|J_x|\) | arg \(J_x\) | \(|J_y|\) | arg \(J_y\) |
|---:|---|---:|---:|---:|---:|

原因：强度接近但相位错了，后续 \(C_{\rm conn}\)、量子光统计和多模关联仍可能不可靠。

---

## Q6. 输出 \(+N/-N\) 选择定则 leakage

定义：

\[
J_{\rm evenN}(q)
=
\frac{J_q(+N)+J_q(-N)}{2},
\]

\[
J_{\rm oddN}(q)
=
\frac{J_q(+N)-J_q(-N)}{2}.
\]

对于双层 AFM-CrI₃：

- 奇次谐波应主要是 \(N\)-even；
- 偶次谐波应主要是 \(N\)-odd。

请输出每个窗口的 leakage：

| q | allowed channel | W20 | W30 | W40 | W50 | W60 | nb1-104 | full112 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| H1 | evenN | | | | | | | |
| H2 | oddN | | | | | | | |
| H3 | evenN | | | | | | | |
| H4 | oddN | | | | | | | |
| ... | | | | | | | | |

请说明 leakage 是用强度、幅度还是复数分量计算的。

---

## Q7. 输出零场 / 弱场 baseline current

为判断是否出现虚电流，请输出：

1. \(E_0=0\) 时的最大电流：

\[
\max_t |J(t)|.
\]

2. 极弱场下 H1 是否线性缩放：

\[
S_1 \propto E_0^2
\quad \text{or} \quad
|J_1|\propto E_0.
\]

3. 对称窗口是否出现 H1 异常增强。

表格格式：

| window | max \(|J(t)|\) at \(E=0\) | H1 异常？ | 备注 |
|---|---:|---|---|

---

# 5. 需要向本地 AI 确认的第三个问题：真实跃迁是否需要深价带？

## Q8. 基于当前光子能量，输出目标谐波能量表

请确认当前激光参数：

\[
\lambda = 3200\ {\rm nm}
\]

对应：

\[
\hbar\omega_0 = ?\ {\rm eV}.
\]

并输出：

| q | \(q\hbar\omega_0\) eV |
|---:|---:|
| H1 | |
| H3 | |
| H5 | |
| H7 | |
| H9 | |
| H11 | |
| H13 | |
| H15 | |

---

## Q9. 做 matrix-element-weighted JDOS / transition histogram

请对每个窗口外被删除的价带/导带，计算它们到 active bands 的跃迁是否落在 H1–H15。

建议定义：

\[
{\rm JDOS}_{ab}(E)
=
\sum_{\mathbf k}
\sum_{v\in a}
\sum_{c\in b}
w_{\mathbf k}
|d^x_{vc}(\mathbf k)|^2
\delta(E-[\varepsilon_c-\varepsilon_v]).
\]

至少比较：

1. active valence \(\to\) active conduction；
2. deleted deep valence \(\to\) active conduction；
3. active valence \(\to\) deleted high conduction；
4. deleted deep valence \(\to\) deleted high conduction。

输出 H1–H15 附近的积分权重：

\[
W_q
=
\int_{q\hbar\omega_0-\Delta}^{q\hbar\omega_0+\Delta}
{\rm JDOS}(E)\,dE.
\]

建议 \(\Delta=0.1\text{--}0.25\) eV。

目的：判断深价带/高导带是真实跃迁通道，还是主要只是虚跃迁/求和规则通道。

---

# 6. 需要向本地 AI 确认的第四个问题：求和规则 / 虚跃迁问题怎么诊断？

## Q10. 速度-偶极关系残差

在 Bloch eigenbasis 中应有类似关系：

\[
v^a_{mn}(\mathbf k)
=
\frac{1}{\hbar}
\partial_{k_a}\varepsilon_n(\mathbf k)\delta_{mn}
+
\frac{i}{\hbar}
[\varepsilon_m(\mathbf k)-\varepsilon_n(\mathbf k)]
r^a_{mn}(\mathbf k)
\quad (m\ne n).
\]

请用当前代码定义检查：

\[
{\rm Res}_{v-r}(W)
=
\frac{
\|v^a_{\rm active}
-
v^a_{\rm reconstructed}(r,\varepsilon)\|
}{
\|v^a_{\rm active}\|
}.
\]

对比：

\[
W20,W30,W40,W50,W60,nb1\text{-}104,full112.
\]

---

## Q11. f-sum / TRK sum rule 残差

请用代码中可实现的形式检查 selected window 的 sum-rule residual。

概念上，完整空间满足：

\[
[x_a,[H,x_b]]
\sim
{\rm constant}.
\]

有限 active window 中，由于：

\[
P_{\rm act}\neq 1,
\]

残差会增大。

请输出一个归一化指标：

\[
{\rm Res}_{\rm TRK}(W).
\]

如果已有更适合 tight-binding/Wannier 的 f-sum residual，请采用代码实际可计算的定义，并说明公式。

---

## Q12. 顺磁/抗磁或 intra/inter 抵消检查

若代码能输出 \(J_{\rm intra}\)、\(J_{\rm inter}\)，请比较窗口下：

\[
J_{\rm tot}=J_{\rm intra}+J_{\rm inter}.
\]

输出：

| window | H1 inter/total | H2 inter/total | H3 inter/total | H5 inter/total | 备注 |
|---|---:|---:|---:|---:|---|

如果速度规范仍有旧结果，请标记为旧结果，不要和当前 `lg_cov` 生产混用。

---

# 7. 少带 Wannier 重做方案：若必须执行，建议怎么做？

如果老师和师兄都希望尝试少带 Wannier，我建议不要做一堆方案，而是先做一个最小试点。

## 7.1 试点模型 A：40-WF Cr-d-only spinor

双层 CrI₃ 中：

\[
4\ {\rm Cr}
\times
5d
\times
2\ {\rm spin}
=
40\ {\rm WF}.
\]

设置思路：

```text
num_wann = 40
spinors = true
projections:
  Cr: d
```

能量窗口可参考文献中的 d-only 思路，但需结合双层和你的 QE 数据重新调试。初始建议：

```text
dis_win_min  = -1.0 ~ -1.5 eV
dis_win_max  = +2.5 ~ +3.0 eV
dis_froz_min = -0.8 ~ -1.0 eV
dis_froz_max = +0.8 ~ +1.0 eV
```

注意：这只是起点，不是最终参数。

---

## 7.2 试点模型 B：Cr-d + 关键 I-p 少量混合

如果 40-WF d-only 明显失败，再考虑 60–80 WF 的折中模型。

可能方向：

1. Cr-\(d\) 全保留；
2. 只保留靠近 VBM 的 I-\(p\) 组合；
3. 或通过投影分析选择对 VBM/CBM 贡献最大的 I-\(p\) 原子/轨道。

但这条路线调试成本较高，不建议第一步就做。

---

## 7.3 少带 Wannier 必须通过的验证

### A. Wannier 质量

| 指标 | 要求 |
|---|---|
| spread | 无游离 WF |
| Omega_D | 尽量小 |
| band interpolation | 目标能区内与 QE / full112 接近 |
| gap | 接近 0.65 eV 或至少物理合理 |
| band ordering | VBM/CBM 性质不乱 |

---

### B. HHG 质量

与 full112 / nb1-104 对比：

| 检查项 | 要求 |
|---|---|
| H1–H9 强度 | 至少量级和趋势一致 |
| H2 \(N\)-odd | 必须保留 |
| H1/H3/H5 \(N\)-even | 必须保留 |
| interband/intraband | 机制不能完全翻转 |
| 弱场 scaling | Hq scaling 合理 |
| k-grid 收敛 | 至少 k40/k60，最好可推到 k100 |

---

### C. 量子光后续可用性

如果少带模型要进入 BSV/QRT 量子光计算，需要额外检查：

1. \(C_{\rm conn}(t,t')\) 是否稳定；
2. 协方差矩阵 \(\Sigma\) 是否满足物理合法性：

\[
\Sigma+\frac{i}{2}\Omega\ge 0.
\]

3. \(g^{(2)}\)、压缩、纠缠结果是否对窗口和 \(k\)-grid 稳定。

---

# 8. 对本地 AI 的整段发问模板

下面这段可以直接复制给本地 AI。

```text
请你基于当前工作区，确认 CrI3-AFM HHG 项目中“少带化”的几个关键问题，并给出可执行方案。

背景：
- 当前已有 112-WF Cr-d + I-p spinor Wannier 母模型；
- 当前生产 SBE 使用 lg_cov length gauge；
- 已有 nb1-104 active space、full112、对称窗口截断等测试；
- 老师和师兄希望尝试费米面附近少带 HHG，甚至重做少带 Wannier；
- 我需要判断“截断 Bloch 能带”和“截断 Wannier 基”到底差别在哪里，以及少带 Wannier 是否值得执行。

请回答以下问题：

1. 当前代码里的 nb_start/nb_end 到底是在 Bloch eigenbasis 截断，还是在 Wannier orbital basis 截断？
   - 请指出具体源码文件和流程；
   - 请写出伪代码；
   - 请确认是否先 diagonalize full 112 H(k)，再选 band window。

2. 当前 lg_cov 中的 Berry connection / dipole / current matrix 是如何生成的？
   - 是否先从 full 112 Wannier 模型得到完整 Bloch-basis 矩阵，再截 active sub-block？
   - 是否包含 i U† ∂k U 项？
   - covariant derivative D_k rho 的实现在哪个文件？

3. 请整理已经做过的对称窗口截断诊断：
   - W20=75–94, W30=70–99, W40=65–104, W50=60–109, W60=53–112, nb1-104, full112；
   - 输出每个 window 的价带数、导带数、能量范围、运行时间；
   - 输出 H1–H15 相对 full112 的强度比；
   - 同时给出 full112 的绝对强度，避免数值地板导致误判。

4. 请输出复数 Fourier 分量，而不只是强度：
   - 对 q=1,2,3,5,7,9,11；
   - 输出 Jx(q), Jy(q) 的幅度和相位；
   - 对比 W20/W30/W40/W50/W60/nb1-104/full112。

5. 请输出 +N/-N 选择定则 leakage：
   - H1/H3/H5 应为 N-even；
   - H2/H4/H6 应为 N-odd；
   - 标明 leakage 的定义是强度比、幅度比还是复数投影比。

6. 请检查零场/弱场虚电流：
   - E0=0 时每个窗口 max |J(t)|；
   - 极弱场下 H1 是否线性缩放；
   - 哪些窗口出现 H1 伪增强？
   - 请区分旧 velocity gauge 结果和当前 lg_cov 结果。

7. 请基于当前光子能量和目标谐波阶数评估真实跃迁贡献：
   - 当前 lambda=3200 nm 对应 hbar omega0 是多少 eV？
   - H1,H3,H5,H7,H9,H11,H13,H15 对应能量是多少？
   - 做 matrix-element-weighted JDOS / transition histogram：
     active valence -> active conduction；
     deleted deep valence -> active conduction；
     active valence -> deleted high conduction；
     deleted deep valence -> deleted high conduction。
   - 判断深价带/高导带在 H1–H15 中是“真实跃迁贡献”还是主要体现为“虚跃迁/求和规则贡献”。

8. 请检查求和规则 / 虚跃迁问题：
   - 计算 velocity-dipole relation residual；
   - 计算 f-sum / TRK residual；
   - 比较 W20/W30/W40/W50/W60/nb1-104/full112；
   - 如果当前代码没有实现，请给出最小实现方案。

9. 如果执行少带 Wannier，请先设计 40-WF Cr-d-only spinor 试点：
   - num_wann=40；
   - projections=Cr:d；
   - 给出建议 dis_win/dis_froz；
   - 是否可以复用现有 nscf/pw2wan，还是必须重跑 QE？
   - 需要哪些输入文件；
   - 预计工作量和风险。

10. 对 40-WF Cr-d-only 模型，请给出强制验证清单：
   - band interpolation vs QE/full112；
   - gap、VBM/CBM、轨道成分；
   - H1–H9 HHG；
   - +N/-N 选择定则；
   - 弱场 scaling；
   - k-grid 收敛；
   - 是否能进入 BSV/QRT 量子光计算。

请最后给出决策建议：
A. 继续 112-WF + nb1-104；
B. 仅用对称窗口作为诊断；
C. 启动 40-WF Cr-d-only Wannier 试点；
D. 做 Cr-d + 部分 I-p 折中模型；
E. 做 112-WF 到低能有效模型的 Löwdin/downfolding。
请说明每个选择的收益、风险和是否符合导师“少带 HHG”的要求。
```

---

# 9. 当前我自己的建议

## 9.1 如果只是为了给导师看“费米面附近能带是否足够”

先整理已有对称窗口结果，不要立刻重做 Wannier。

因为这能直接回答：

> 费米面附近 Bloch bands 是否能复现 HHG？

如果 W40/W50 结果不错，就可以向老师说明少带 Bloch active space 有希望。  
如果 W20/W30 失败，也可以作为证据说明 CrI₃ 不能过度低能化。

---

## 9.2 如果老师坚持“真正少带模型”

那就做 40-WF Cr-d-only Wannier 试点，但要明确它是**新模型**，不是 full112 的简单裁剪。

汇报用语建议：

> 我们会先测试 Bloch band active-space 截断，确认费米面附近能带对 HHG 的贡献范围。如果简单 Bloch 截断产生伪电流或破坏 \(N\)-选择定则，则不直接用于生产。进一步会尝试重做 Cr-\(d\)-only 40-WF Wannier，这是一个 downfolded 少带有效模型，需要重新验证能带、带隙、光学矩阵元和 HHG 选择定则。最终仍以 112-WF \(d+p\) 模型作为 reference。

---

## 9.3 如果担心虚跃迁/求和规则

优先解决办法是：

1. 使用 length gauge / `lg_cov`，避免 velocity gauge 下强烈的有限带发散；
2. 保留完整 occupied manifold，例如 `nb1-104`；
3. 若必须少带，做系统窗口收敛，而不是单个窗口；
4. 检查 \(E=0\) baseline current；
5. 检查 \(N/-N\) 选择定则；
6. 检查 velocity-dipole relation residual 和 f-sum/TRK residual；
7. 若重做 Wannier，则重新生成 position/dipole matrix，不能沿用 full112 的局部子块。

---

# 10. 可放进汇报的简短版本

```text
少带化不是简单“删带”。

1. Bloch-band 截断：
   先从完整 112-WF 模型得到 Bloch 本征态，再选费米面附近能带进入 SBE。
   优点是可控、可收敛测试、可与 full112 对比。

2. Wannier-orbital 截断：
   直接删除 Wannier 基/轨道会改变局域轨道空间和光学矩阵元，一般不推荐。

3. 重做少带 Wannier：
   是新的 downfolded 有效模型，物理上更干净，但必须重新验证 band interpolation、gap、HHG、N 选择定则和 sum rule。

下一步：
先整理已有 W20/W30/W40/W50/W60 对称窗口诊断；
若 W40/W50 能复现 H1–H9 和 N-选择定则，再启动 40-WF Cr-d-only Wannier 试点。
```

---

# 11. 最终一句话

\[
\boxed{
师兄说的重点不是“少带一定没问题”，而是“在能量本征态 Bloch basis 中做 active-band 截断，比直接删 Wannier orbital 更有定义、更可控”。
}
\]

\[
\boxed{
真正想要少带生产模型，最好重做 Wannier/downfolding，但它必须被当作新模型完整验证，而不是 full112 的无痛裁剪。
}
\]
