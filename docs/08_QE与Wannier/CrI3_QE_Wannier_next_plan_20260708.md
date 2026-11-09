# CrI3 QE + Wannier 少带化与速度规范截断：下一步工作方案（给本地 AI / 服务器 Agent）

更新时间：2026-07-08

## 0. 当前重新定调

### 0.1 不再把 `nb1-104` 说成“高阶完全收敛”

- `nb1-104 = 84 valence + 20 conduction` 对 **H1–H9** 已比较稳。
- **H10–H12** 开始需要谨慎。
- **H13+ / H16–H20** 不应再用 `nb1-104` 直接宣称收敛。
- `nb1-104` 和 `full112` 在 H11 之后的偏差不能简单归因于“机器精度”；至少有一部分来自被删高导带的真实通道和 cutoff 区域的数值敏感性。

### 0.2 对称窗口已判定不能作为生产

对称窗口 W30/W40/W50/W60 的结果已经说明：只保留费米面附近若干价带/导带，会丢掉大量深价带，并显著扭曲 H3–H9 和偶次谐波。JDOS + f-sum proxy 已支持这个机制解释。

### 0.3 40-WF Cr-d-only 可以做，但必须作为新有效模型试点

40-WF Cr-d-only Wannier 不应被视为 full112 的简单截断，而是一个新的低能有效模型。它必须重新验证 Wannier 收敛、能带、占据数、光学矩阵元、HHG、±N 选择定则和零场/弱场虚电流。

---

## 1. Bloch 截断 vs Wannier 截断：需要本地 AI 再确认

当前本地答复称 `nb_start/nb_end` 是 Bloch 本征带截断，不是 Wannier orbital 子块截断。请再次确认源码中是否严格满足：

```fortran
H_W(k) = Fourier[H(R)]                       ! full 112 × 112 Wannier space
call zheevd(H_W(k), eigvals, U_full)         ! full diagonalization
U_active = U_full(:, nb_start:nb_end)        ! select Bloch eigenvectors

M_Bloch = U_active^dagger M_Wannier U_active
```

确认这些矩阵都是先在 full Wannier space 构造，再投影到 active Bloch space：

- `Hk_eq`
- `Dk_eq`
- `Pk_eq`
- current matrix
- covariant-derivative / parallel transport overlap

确认没有出现：

```fortran
H_W_active = H_W(nb_start:nb_end, nb_start:nb_end)
```

---

## 2. 40-WF Cr-d-only Wannier 试点：核心风险

### 2.1 文献风险

CrI3 的 Cr-d-only 模型在文献中并非完全可靠。Soriano et al. 比较了 Cr-d-only 与 Cr-d + I-p 两类 Wannier downfolding，并指出 ligand I-p orbitals 对 CrI3 的磁性机制有决定性影响。d-only 可以在某些层面定性复现 Cr-d 带结构，但不能正确描述 CrI3 的 FM super-exchange microscopic origin。扩展到 d+p 后，才得到更接近 ab initio 的 DOS、gap、spin ordering 和正确交换趋势。

因此 40-WF Cr-d-only 试点必须被标记为：

> low-cost effective model trial, not a validated replacement for full d+p model.

### 2.2 40-WF 试点目标

判断 Cr-d-only 低能模型能不能在 H1–H9 范围内近似复现 full112 / nb1-104 的 HHG 和磁性选择定则。

---

## 3. 40-WF d-only 的占据数必须重新确定

full112 中：

```text
num_wann = 112
occupied bands = 84
```

但 40-WF Cr-d-only 模型只包含：

```text
4 Cr × 5 d × 2 spin = 40 spinor WFs
```

它不再显式包含 I-p 价带，因此不能继续使用 `N_occ = 84`。

如果按 Cr3+ 的 d3 图像：

```text
每个 Cr ≈ d^3
4 个 Cr → 12 个 d 电子
```

那么 40-WF d-only 模型中可能有约 12 个 occupied spinor bands。但这只是化学直觉，最终必须由实际 d-only Wannier / DFT 投影结果决定。

请本地 AI 输出：

```text
1. d-only 模型的 Fermi level
2. 每条 Wannier-interpolated band 的占据判断
3. SBE 初始 occupied band 数 N_occ_donly
4. 与 Cr d projected occupation 的一致性
5. 若出现部分占据或金属性，说明 d-only 模型失败或需调整窗口
```

---

## 4. Wannier90 参数需要检查

请检查 40-WF `CrI3_donly.win` 中：

```text
num_bands
num_wann
spinors
begin projections
  Cr:d
end projections

dis_win_min / dis_win_max
dis_froz_min / dis_froz_max
dis_num_iter
dis_mix_ratio
dis_conv_tol

write_hr = true
write_tb = true
write_rmn = true 或确认 write_tb 中包含 r(R)
bands_plot = true
bands_num_points
```

说明：

- 外窗口 `dis_win` 决定参与 disentanglement 的 DFT Bloch states 范围。
- 内冻结窗口 `dis_froz` 用于强制精确复现某个能量范围内的 DFT bands。
- `dis_froz` 不能决定占据数；占据数由 reduced model 的电子数/Fermi level/band filling 判断。
- 对 d-only 试点，建议至少冻结 near-gap Cr-d 相关 bands。

---

## 5. 40-WF d-only 完成后的验证清单

### 5.1 Wannier 本身

输出：

```text
Omega_Total
Omega_D
Omega_OD
每个 WF spread 最大值与均值
是否有游离 WF
是否有异常中心
```

### 5.2 能带对比

画三种能带叠图：

```text
DFT QE bands
full112 Wannier interpolation
40-WF d-only Wannier interpolation
```

路径至少包含：

```text
Gamma → M → K → Gamma
```

重点比较：

```text
near-gap bands
VBM/CBM 位置
direct/indirect gap
spin/layer ordering
d-only 是否仍是绝缘体
```

### 5.3 光学矩阵元

确认 d-only `tb.dat` 是否包含可用于 `lg_cov` 的 `r(R)`：

```text
CrI3_donly_tb.dat 是否写出 position matrix elements
SBE 能否读入并构造 D(k), P(k)
v-r consistency proxy 是否正常
是否有异常大的 dipole / position matrix elements
```

### 5.4 HHG 对比

用同样参数跑：

```text
lg_cov
k40
T2_cycles = 0.5
lambda = 3200 nm
I = 2e11 W/cm^2
```

输出：

```text
H1-H9 peak intensity ratio to full112
H1-H9 peak intensity ratio to nb1-104
H1-H9 complex Jx/Jy amplitude and phase
zero-field baseline current
```

### 5.5 磁性选择定则

必须做 +N / -N：

```text
H2 / H4 / H6 / H8 allowed vs forbidden leakage
```

如果 d-only 不能保持 H2 的 N-odd 性质，则不能作为磁性 HHG 模型使用。

---

## 6. 真实跃迁：根据光子能量和能隙判断

当前驱动：

```text
lambda = 3200 nm
hbar omega0 ≈ 0.387 eV
gap ≈ 0.65 eV
```

谐波能量：

```text
H1  ≈ 0.39 eV
H3  ≈ 1.16 eV
H5  ≈ 1.94 eV
H7  ≈ 2.71 eV
H9  ≈ 3.49 eV
H11 ≈ 4.26 eV
H13 ≈ 5.04 eV
H15 ≈ 5.81 eV
H20 ≈ 7.75 eV
```

请基于 JDOS / matrix-element-weighted transition histogram 判断：

```text
1. H1-H9 的主要跃迁通道是否都在 nb1-104 active space 内
2. H9-H15 是否开始涉及 deleted conduction bands 105-112
3. H16-H20 是否必须 full112
4. d-only 模型是否有对应能量差的跃迁通道
```

注意：由于光子能量小，HHG 是多光子/强场过程，不应只按单光子共振理解，但 transition-energy histogram 仍能判断哪些能带对可能参与 recombination / polarization response。

---

## 7. 虚跃迁、求和规则与规避方案

直接截断会破坏完备性：

```math
sum_{n in active} |u_n><u_n| != 1
```

这会影响：

```text
TRK / f-sum rule
velocity-dipole consistency
paramagnetic-diamagnetic cancellation
low-frequency divergences in velocity gauge
unphysical ground-state current
```

目前可用规避手段：

```text
1. 优先使用 length gauge / lg_cov，而不是 Peierls velocity gauge
2. 保留完整 occupied manifold：nb1-104 比对称窗口好
3. 做 f-sum proxy / JDOS 诊断
4. 少带模型若要使用，必须重新构造 Wannier effective model，而不是硬删 bands
5. 对 velocity gauge 文章线，需要系统做 TRK residual、baseline current、gauge convergence
```

请补充完整检查：

```text
1. 逐 k 检查 v_mn^a ≈ i (E_m - E_n) r_mn^a / hbar
2. 对 full112、nb1-104、W40、W30、d-only 分别算 residual
3. 统计 f-sum / TRK residual vs band window
4. 统计 zero-field and weak-field current baseline
5. 判断哪些异常来自真实跃迁缺失，哪些来自虚跃迁/求和规则缺失
```

---

## 8. 速度规范截断测试能否成为 PRB 文章线？

已知文献已经证明：

- 有限带截断会破坏 length gauge 与 velocity gauge 的等价性。
- few-band velocity gauge 可产生非物理低频发散。
- length gauge 通常在截断基下收敛更快、更稳定。
- 固体 HHG 中 Berry connection / transition dipole phase / structure gauge 很重要。

因此“速度规范截断会出问题”本身不是新发现。

如果要冲 PRB，需要把文章定位成：

> Real-material multiband benchmark of gauge/basis truncation in magnetic solid HHG.

可能的新点：

```text
1. 真实 112-band CrI3-AFM spinor Wannier model
2. Bloch active-space 截断 vs Wannier orbital 截断的明确区分
3. 对称窗口、全价带+截导带、full112 的系统比较
4. JDOS + f-sum proxy 解释截断失真机制
5. velocity gauge / length gauge / lg_cov 的对比
6. ±N 磁性选择定则作为额外物理判据
7. zero-field current / weak-field linear response / TRK residual 作为数值健康指标
```

最低补足计划：

```text
A. full112, nb1-104, W40/W30, d-only 四类模型
B. length gauge vs velocity gauge 对比
C. zero-field baseline current
D. weak-field linear response
E. H1-H15 HHG spectra
F. complex J(omega) amplitude + phase
G. v-r consistency / TRK residual
H. JDOS weighted transition histogram
I. ±N selection rule leakage
J. k-grid convergence at least k40/k60, 关键图 k80
```

判断：

```text
可以作为内部技术文章 / supplementary-level 方法研究；
若要 PRB，需要强调真实磁性多带材料 + 截断机制诊断，而不能只重复“velocity gauge bad”。
```

---

## 9. 给本地 AI 的执行请求

### Task 1：40-WF Cr-d-only Wannier

```text
1. 在服务器上运行 pw2wannier90 + wannier90
2. 输出 spread、Omega、centres
3. 画 DFT/full112/d-only 能带叠图
4. 判断 d-only 占据数和带隙
5. 生成 CrI3_donly_tb.dat
```

### Task 2：d-only SBE 前置检查

```text
1. 确认 tb.dat 包含 H(R)+r(R)
2. 确认 SBE 能读入 d-only tb.dat
3. 确认 lg_cov 所需 D(k), P(k) 可构造
4. 做零场 baseline
```

### Task 3：d-only HHG 试跑

```text
lg_cov / k40 / T2=0.5 / same laser
输出 H1-H9 与 full112、nb1-104 对比
输出 complex J(omega)
输出 ±N leakage
```

### Task 4：整理 nb1-104 高阶偏差机制

```text
1. 把 nb1-104 vs full112 的 H1-H20 ratio 重新整理
2. 不要把 H13+ 简单写成 machine precision
3. 结合 JDOS 说明 deleted conduction bands 的影响
4. 标明 H1-H9、H10-H12、H13+ 三个可信区间
```

### Task 5：速度规范 PRB 线预研

```text
1. 汇总已有 VG / Peierls / lg_cov 数据
2. 统计 baseline current / TRK residual
3. 判断是否具备 PRB 方法文章雏形
4. 列出还缺哪些图
```

---

## 10. 当前最重要的判断

```text
1. 现在不能再把 nb1-104 描述为高阶全收敛。
2. 对称窗口已失败，不需要继续纠结。
3. 40-WF d-only 可以做，但必须作为新模型验证。
4. d-only 文献上存在磁性机制风险，特别是 ligand I-p 缺失。
5. QE DFT 直接加密可以用于诊断，但生产 SBE/HHG 仍更适合 Wannier 插值。
6. 占据数必须由 reduced model 自己确定，不能沿用 full112 的 84。
7. 速度规范截断线有文章潜力，但必须有新意：真实磁性多带材料 + 截断机制诊断。
```
