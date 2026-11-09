# hBN 二带模型 benchmark 记录（2026-06-21）

## 目的

这一步不是为了建立第一性原理 hBN 生产模型，而是为了快速验证新 `Quantum-light` SBE 求解器在一个文献常用的二带 hBN / gapped graphene 模型上能否稳定给出 HHG。

前一轮 QE + Wannier90 的 monolayer hBN slab 路线遇到的问题是：20 Å 真空层使低导带混入非局域真空态，`num_wann=8` 或 `pz` 两带模型都出现非常大的 spread（约 200-700 Å²），不适合作为快速验证路线。这个失败更像是 slab Wannier 建模问题，不是 SBE 求解器问题。

## 模型来源

新求解器当前以 Wannier90-style TB 文件作为统一入口。为了避免继续纠缠 hBN slab 的真空态 Wannier 化问题，本次用 Python 直接把解析二带 hBN 哈密顿量写成新求解器可读的 `tb.dat`。

模型形式：

```text
H11 = +M0
H22 = -M0
H12(k) = t [1 + exp(i k.a1) + exp(-i k.a2)]
H21(k) = conj(H12(k))
```

参数：

```text
t  = 2.33 eV
M0 = 1.96 eV
a0 = 2.50 Ang
c  = 20.00 Ang
```

对应 K 点直接带隙：

```text
2*M0 = 3.92 eV
```

## 生成文件

目录：

```text
C:\Users\26507\Documents\New_SBEs\Quantum-light\examples\hBN_model_tb
```

关键文件：

```text
generate_hBN_model_tb.py
hBN_model_tb.dat
hBN_model_summary.json
input_hBN_model_lgcov_k30_yue_gaarde.nml
analyze_hBN_model_hhg.py
README.md
```

`hBN_model_tb.dat` 的 R 空间矩阵元包含：

```text
R = (0,0,0), (1,0,0), (0,-1,0), (-1,0,0), (0,1,0)
```

位置矩阵元只保留 R=0 的两个子晶格对角位置。这是最小解析模型约定，不等价于文献中完整的 Berry connection / transition-dipole phase 数据。

## SBE 参数

输入文件：

```text
examples\hBN_model_tb\input_hBN_model_lgcov_k30_yue_gaarde.nml
```

参数：

```text
gauge_method = lg_cov
nkx = nky = 30
nv_orig = 1
nb_start = 1
nb_end = 2
lambda = 1600 nm
intensity = 3.5e12 W/cm^2
ncyc = 11
dt = 0.50 a.u.
T2_fs = 1e30
T2_cycles = -1
```

这组参数用于靠近 Yue/Gaarde 类二带 hBN HHG benchmark 的设置；后续如果要严格复现某篇文献，需要再对齐其包络、CEP、谱定义、k 网格和输出归一化。

## 本地运行结果

运行命令（MSYS 环境，避免 Windows DLL 搜索路径问题）：

```powershell
& "C:\msys64\usr\bin\bash.exe" -lc "export PATH=/mingw64/bin:/usr/bin:$PATH; cd '/c/Users/26507/Documents/New_SBEs/Quantum-light/examples/hBN_model_tb' && ../../hhg_sbe.exe input_hBN_model_lgcov_k30_yue_gaarde.nml > run_hBN_model_lgcov_k30.log 2>&1"
```

输出：

```text
Et.dat
Jt.dat
HHG.dat
bands.dat
Jt_decomposed.dat
Jt_valley.dat
quantum_geometry.dat
harmonics_summary.dat
run_hBN_model_lgcov_k30.log
```

sanity checks：

```text
K 点直接带隙 = 3.920000 eV
Gamma 本征值 = +/-7.259594 eV
TB H/r/v 厄米性误差 = 0（打印精度内）
Initial current |J(t=0)| = 2.6652e-19
lg_cov / 30x30 / 2 bands 本地运行时间 = 9.62 s
```

前 7 阶谐波窗口峰值：

```text
H1 peak_order=0.99960  total=2.0121576e-03  x=2.0121525e-03  y=5.1489329e-09
H2 peak_order=1.99920  total=6.2648336e-05  x=5.0137439e-08  y=6.2598199e-05
H3 peak_order=2.90790  total=4.2613474e-05  x=4.2608621e-05  y=4.8524136e-09
H4 peak_order=3.99840  total=5.0963497e-06  x=4.5099626e-07  y=4.6453534e-06
H5 peak_order=5.08890  total=4.8520354e-05  x=4.8178619e-05  y=3.4173557e-07
H6 peak_order=6.08850  total=2.4893550e-05  x=2.0426333e-05  y=4.4672174e-06
H7 peak_order=6.90640  total=2.5114050e-05  x=1.7888803e-05  y=7.2252468e-06
```

物理上可见：x 偏振主导 H1/H3 等奇次响应；由于 hBN 二带模型本身缺反演，y 分量偶次响应明显存在。这和把 hBN 当作非中心对称 benchmark 的用途一致。

## 当前结论

1. 用 Python 生成解析模型 `tb.dat` 是目前最快、最干净的验证路线。
2. 本地已经证明新求解器可直接读取该解析 `tb.dat`，完成 `lg_cov` 时间传播并输出 HHG。
3. 这条 benchmark 适合证明求解器的数值流程没有基本错误；它不能替代 CrI3 的第一性原理生产结果。

## 下一步

建议做两个很小的后续测试：

1. k 网格收敛：`nk=30, 60, 90` 对比 H1-H15。二带模型很轻，本地或服务器都可以跑。
2. 文献对齐：选定一篇具体二带 hBN HHG 文献后，严格对齐其激光包络、强度、频率、CEP、谱定义（`|J(omega)|^2` 还是 `|omega J(omega)|^2`），再做图形对照。

这两步完成后，可以把 hBN 二带模型作为“求解器 benchmark”，而不是继续纠缠 slab Wannier 的真空态问题。

---

## 2026-06-21 补充：k 收敛与文献对照

### 对照文献

当前最适合对照的是 Yue & Gaarde 的 hBN SBE 工作：

```text
L. Yue and M. B. Gaarde,
Imperfect Recollisions in High-Harmonic Generation in Solids,
arXiv:2001.04626.

L. Yue and M. B. Gaarde,
Structure- and laser-gauges for the semiconductor Bloch equations in
high-harmonic generation in solids,
arXiv:2003.02961.
```

文献中和我们 benchmark 最相关的设置：

```text
monolayer hBN
1 valence band + 1 conduction band
lambda = 1600 nm
peak intensity = 3.5 TW/cm^2
T2 = 5 fs
pulse duration = 35 fs
LPD along Gamma-K
```

文献报告的核心物理特征：

```text
平行于激光偏振方向：主要为奇次谐波
垂直于激光偏振方向：主要为偶次谐波
谐波可延伸到约 25 阶
Berry connection / transition-dipole phase 对 hBN 的偶次和垂直响应很重要
```

### 本地 k 网格扫描

本地完成了：

```text
gauge_method = lg_cov
T2_fs = 5 fs
lambda = 1600 nm
intensity = 3.5e12 W/cm^2
ncyc = 11
nk = 30, 60, 90, 120
```

运行目录：

```text
examples\hBN_model_tb\runs\lgcov_k30_T2_5p0fs
examples\hBN_model_tb\runs\lgcov_k60_T2_5p0fs
examples\hBN_model_tb\runs\lgcov_k90_T2_5p0fs
examples\hBN_model_tb\runs\lgcov_k120_T2_5p0fs
```

汇总文件：

```text
examples\hBN_model_tb\analysis\k_scan_harmonics_summary.dat
examples\hBN_model_tb\analysis\k_scan_spectrum_T2_5fs.svg
```

本地运行时间：

```text
nk=30   9.50 s
nk=60  19.98 s
nk=90  39.28 s
nk=120 69.76 s
```

初始电流均为机器零量级：

```text
|J(t=0)| ~ 5e-19 到 8e-19
```

矩阵诊断：

```text
H(k), r(k), v(k) 的厄米性误差在打印精度内为 0
```

### k 收敛判断

以 `nk=120` 为参考：

```text
H1-H3:  nk=90 与 nk=120 基本一致，误差约 0.05%-0.1%
H4-H8:  nk=90 与 nk=120 大多在 1% 左右
H9:     nk=90 约低 10%
H10-H13: nk=90 与 nk=120 基本在 1%-3% 内
H14-H15: nk=90 约低 5%-6%
H16+：  绝对强度已经很低，比例容易被数值地板放大，不适合作为严格收敛判据
```

结论：

```text
nk=90 已足够作为 hBN 二带 benchmark 的主要本地对照。
nk=120 可作为高网格参考。
本地计算成本很低，不需要服务器。
```

### 35 fs 文献时长对照点

为了更贴近文献中的 35 fs pulse，又额外跑了：

```text
nk = 120
T2_fs = 5 fs
ncyc = 6.557
T_total = 34.99 fs
```

运行目录：

```text
examples\hBN_model_tb\runs\lgcov_k120_T2_5p0fs_ncyc6p557
```

sanity check：

```text
Initial current |J(t=0)| = 8.56e-20
Total wall time = 39.53 s
```

逐阶分量显示：

```text
H1  dominated by Jx
H2  dominated by Jy
H3  dominated by Jx
H4  dominated by Jy
H5  dominated by Jx
H8  dominated by Jy
H9  dominated by Jx
H12 dominated by Jy
H13 dominated by Jx
```

也就是说，本求解器在这个二带 hBN benchmark 上复现了文献最重要的选择定则：

```text
parallel direction  -> odd harmonics
perpendicular direction -> even harmonics
```

同时，谱尾延伸到 H20-H25 附近，但 H18 之后强度已经很低，不能过度解读尾部比例。

### 与文献的一致和差异

可以说一致的部分：

```text
1. 使用同类 hBN 二带 SBE benchmark。
2. 激光频率、峰值强度和 T2 可与 Yue/Gaarde 对齐。
3. 平行奇次、垂直偶次的偏振选择规律被复现。
4. 谱可延伸到约 25 阶附近。
5. 求解器的 lg_cov 时间传播、HHG 输出、Et/Jt 输出和基态电流检查全部正常。
```

不能说成严格定量复现的部分：

```text
1. 当前 `hBN_model_tb.dat` 是解析二带 TB，不是 Yue/Gaarde 使用的 pseudopotential hBN band/dipole/Berry 数据。
2. 当前 position matrix 采用最小二带模型的子晶格位置约定；它不等价于文献中完整的 Berry connection / transition-dipole phase 数据。
3. 谱定义为 `|J(omega)|^2`，并使用 Hanning window；文献图的归一化和谱定义可能不同。
4. pulse envelope 的“35 fs”定义需要和文献补充材料进一步核对。
```

因此这一步的正确定位是：

```text
这是一个成功的 model-based SBE benchmark，
证明新 Quantum-light 求解器在标准 hBN 二带强场 HHG 场景下给出了正确的定性物理结构。

它不是对 Yue/Gaarde 图 2 的逐点定量复现。
如果要做到逐点复现，需要获得或重建他们使用的 hBN band/dipole/Berry 数据，
或者显式实现解析二带模型的完整 Berry connection / transition-dipole phase。
```

### 严格图形复现工作包

已经完成本机可做的第一层图形复现尝试：

```text
输入数据：
examples\hBN_model_tb\runs\lgcov_k120_T2_5p0fs_ncyc6p557\harmonics_summary.dat

输出图：
docs\figures\hBN_benchmark\hbn_model_yue_style_fig2ab_attempt.svg
```

这张图使用和文献一致的主要激光参数：

```text
1600 nm
35 fs
3.5 TW/cm^2
T2 = 5 fs
k = 120 x 120
```

并按 Yue & Gaarde Fig. 2(a,b) 的方式，把谱强度放在横轴、谐波阶数放在纵轴，分别画平行方向 Jx 和垂直方向 Jy。它已经展示出平行奇次、垂直偶次的核心结构。

不过，这还不是严格逐点复现。原因是 Yue & Gaarde 使用 pseudopotential hBN band、Berry connection 和 transition-dipole phase；当前 benchmark 使用解析最小二带模型。尤其是当前模型直接带隙为 3.92 eV，在 1600 nm 下对应约 H5.06，而文献图中的最小带隙线在更高谐波阶附近。

如果导师要求比当前 benchmark 更强的外部证据，建议把 Yue/Gaarde Fig. 2 逐点复现作为独立工作包：

```text
1. 模型对齐：
   重建文献中的 hBN band、Berry connection、transition dipole phase；
   或者获得等价的 pseudopotential 数据。

2. 激光对齐：
   确认 35 fs pulse 的包络定义、CEP、T2=5 fs、k 网格和 LPD 沿 Gamma-K 的方向。

3. 谱定义对齐：
   明确文献图使用 |J(omega)|^2、|omega J(omega)|^2 还是其他归一化；
   统一 Fourier window 和纵轴归一化。

4. 图形对齐：
   先复现 Fig. 2(a,b) 的平行/垂直 HHG 谱；
   再考虑 Gabor time-frequency 图 2(c,d)。
```

这一步的目标是“严格图形复现”，不是发展新的 hBN 物理；它应该服务于求解器可信度证明。
