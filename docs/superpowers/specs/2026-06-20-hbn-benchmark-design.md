# hBN monolayer QE/Wannier/SBE benchmark 设计

日期：2026-06-20

## 1. 目标

建立一套单层 hBN 的 QE + Wannier90 + `Quantum-light` SBE benchmark，用于验证当前二维 `lg_cov` 求解器在一个成熟二维材料体系上的行为。

这套 benchmark 的目标不是做 hBN 新物理，而是回答三个工程/物理验证问题：

1. 当前 Wannier `tb.dat` 读入、矩阵预计算、`lg_cov` 传播和 HHG 输出链条在小型二维体系上是否正常。
2. 是否能复现文献中 hBN HHG 的基本特征：强的面内响应、与 Berry connection / transition-dipole phase 相关的规范敏感性，以及合理的 k 网格收敛趋势。
3. 是否能作为 CrI3 主线之外的“外部材料 sanity check”，避免只依赖双层 AFM CrI3 这个文献较少、能带数较多的体系来证明求解器可靠。

## 2. 文献定位

主 benchmark 文献：

- Lun Yue and Mette B. Gaarde, *Structure- and laser-gauges for the semiconductor Bloch equations in high-harmonic generation in solids*, Phys. Rev. A 101, 053411 (2020), arXiv:2003.02961.

使用方式：

- 作为 SBE/gauge benchmark 的主参考。
- 该文直接讨论二维单层 hBN 的 SBE HHG，并强调在 hBN 中忽略 Berry connections 或 transition dipole phases 会得到非物理谱。
- 对应我们当前路线：使用 Wannier `tb.dat` 中的 `r(k)`/`D(k)`，走 `lg_cov`，而不是人为丢弃 Berry/dipole 相位。

参数参考文献：

- Guillaume Le Breton, Angel Rubio, and Nicolas Tancogne-Dejean, *High-harmonic generation from few layer hexagonal boron nitride: evolution from the monolayer to the bulk response*, arXiv:1807.02961.

使用方式：

- 作为单层 hBN 结构、k 网格和激光尺度参考。
- 该文是 TDDFT，不是 SBE，所以不能要求谱强度逐点相同。
- 可借鉴的量：B-N 键长 `2.73 Bohr`、面内 HHG 使用 `42×42` k 网格、`1600 nm` 激光、面内脉冲约 `30 fs`、强场 HHG intensity scale。

## 3. QE/Wannier 体系设计

材料：单层 hexagonal boron nitride, hBN。

结构：

- B-N 键长：`2.73 Bohr = 1.4449 Å`。
- 六角晶格常数：`a = sqrt(3) × 2.73 Bohr = 4.7286 Bohr = 2.5026 Å`。
- 真空层：`c = 20 Å`。
- 原胞：1 B + 1 N。
- 采用 `CELL_PARAMETERS angstrom`，避免 `ibrav` 与真空方向混淆。

推荐原胞：

```text
a1 = ( 2.5026,  0.0000,  0.0000 ) Å
a2 = (-1.2513,  2.1673,  0.0000 ) Å
a3 = ( 0.0000,  0.0000, 20.0000 ) Å
```

推荐原子位置：

```text
B  0.000000  0.000000  0.500000  crystal
N  0.333333  0.666667  0.500000  crystal
```

QE 设置：

- 泛函：PBE。
- SOC：关闭。
- 磁性：关闭。
- `ecutwfc = 60 Ry`。
- `ecutrho = 480 Ry`。
- SCF k 网格：`18×18×1`。
- NSCF k 点：显式 `18×18×1`，与 Wannier90 `.win` 完全一致。
- `nosym = .true.`、`noinv = .true.` 用于 NSCF/Wannier 数据生成，避免 QE 自动改写 k 点集合。
- `nbnd = 24`。

赝势假设：

- 默认使用 PSLibrary PBE PAW 或 ultrasoft 赝势。
- 输入文件先采用以下文件名：

```text
B.pbe-n-kjpaw_psl.1.0.0.UPF
N.pbe-n-kjpaw_psl.1.0.0.UPF
```

如果服务器实际文件名不同，只需要改 QE 输入文件中的 `ATOMIC_SPECIES`。

Wannier90 设置：

- `num_bands = 24`。
- `num_wann = 8`。
- 投影：`B:s;p` 与 `N:s;p`，共 `4 + 4 = 8` 个 Wannier 函数。
- `mp_grid = 18 18 1`。
- 第一轮 `.win` 不写 `dis_win_*` / `dis_froz_*`，先让 `pw2wannier90` 生成 `hBN.eig`；随后按每个 k 点窗口内带数是否不少于 `num_wann=8` 来设置正式窗口。
- 预期占据带数：无 SOC、非磁、价电子 8 个，对应 `nv_orig = 4`。

输出目标：

- `hBN_hr.dat`
- `hBN_tb.dat`
- `hBN_band.dat`
- `hBN_centres.xyz`
- QE path bands 与 Wannier path bands 对比图

## 4. SBE benchmark 设计

SBE 主参数优先采用 Yue & Gaarde 2020 的 hBN 设置尺度：

- `gauge_method = 'lg_cov'`
- `wvl_nm = 1600.0`
- 文献向量势峰值：`A0 = 0.35 a.u.`
- `omega0 = 0.0285 a.u.`，对应 `lambda = 1600 nm`
- 估算电场峰值：`E0 ≈ omega0 × A0 = 0.00998 a.u.`
- 换算强度：`I ≈ 3.5×10^12 W/cm^2`
- 文献脉冲支撑区间：`t in [-tau, tau]`，`tau = 29.4 fs`
- 总脉冲长度：约 `58.8 fs`
- 在 `1600 nm` 下光周期约 `5.34 fs`，因此我们代码中取 `ncyc = 11.0`
- `env_type = 2`，即电场 cos² 包络；这不是与文献向量势包络完全逐项相同，但总时长和中心频率对齐，足够作为 solver benchmark。

第一轮 SBE 测试：

1. `lg_cov / 8 bands / 20×20 / no dephasing`
2. `lg_cov / 8 bands / 42×42 / no dephasing`
3. 可选：`lg_cov / 8 bands / 42×42 / T2_cycles = 1.0`

第一轮判据：

- `Initial current |J(t=0)|` 应接近机器零。
- `H(k)`、`D(k)`、`P(k)` 厄米性诊断应在可解释范围内。
- `20×20 → 42×42` 的低阶谐波应有收敛趋势。
- 谱形应以对称性允许的谐波为主；若出现奇怪偶/奇通道，需要优先检查晶格方向、偏振角、Berry connection/dipole phase。

## 5. 目录与文件计划

新增目录：

```text
examples/hBN_monolayer_qe_wannier/
```

计划生成：

```text
hBN_scf.in
hBN_bands.in
hBN_bands_pp.in
hBN_nscf.in
hBN.pw2wan
hBN.win
run_all.sh
README.md
pseudo/README.md
sbe_inputs/input_hbn_lgcov_k20_yue_gaarde.nml
```

运行后输出应下载到：

```text
examples/hBN_monolayer_qe_wannier/output/
```

本地分析脚本后续可生成：

```text
hBN_qe_vs_wannier_bands.png
hBN_qe_vs_wannier_bands_near_gap.png
hBN_lgcov_k20_k42_HHG_benchmark.png
hBN_QE_Wannier_SBE数据整理_20260620.md
```

## 6. 不做的事

第一版不做以下内容：

- 不直接追求 TDDFT 文献谱强度逐点一致。
- 不做 exciton/BSE 修正。
- 不做 3D bulk hBN。
- 不引入 SOC。
- 不引入多层堆叠。
- 不把 CrI3 的 `3200 nm`、`2×10^11 W/cm^2` 参数照搬到 hBN benchmark。

## 7. 风险与检查点

主要风险：

1. PBE hBN 带隙低于实验和部分 TDDFT 设置，cutoff 位置可能与文献不同。
2. `env_type=2` 对电场使用 cos² 包络，而 Yue & Gaarde 写的是向量势 cos² 包络，两者不是完全同一个脉冲形状。
3. 如果只用 8 WF，远高导带贡献会缺失；但对第一轮 solver benchmark，8 WF 是合理的最小模型。
4. 如果 Wannier 能带在带边附近不吻合，需要先调能量窗口，不应直接进入 SBE。

成功条件：

- QE/Wannier 能带在 VBM/CBM 附近与 QE path bands 对齐。
- `hBN_tb.dat` 能被 `Quantum-light` 正常读入。
- `lg_cov` 路径能在 `20×20` 和 `42×42` 完整跑通。
- `J(t=0)` 近零，HHG 输出稳定。
- 可以把 hBN 结果作为“二维材料外部 benchmark”写入求解器验证文档。
