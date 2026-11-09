# ZnO QE + Wannier90 数据整理记录（2026-06-20）

本文档整理当前用于验证 `Quantum-light` SBE 求解器的 ZnO 数据链：输入参数文件、服务器运行得到的数据、下载到本地的数据，以及本地后处理结果。

## 1. 目标定位

这套 ZnO 数据用于做 **SBE 求解器健康检查**，不是最终的 ZnO 体材料 HHG 定量论文结果。

主要目的：

- 验证求解器能正确读入一个小型、无 SOC、非磁性的 `tb.dat`。
- 验证 `lg_cov` 路径在简单体系中是否能正常运行。
- 检查 `Initial current |J(t=0)|`、TB 矩阵质量、弱场 scaling 和 HHG 输出是否正常。

重要限制：

- 当前 `Quantum-light` SBE 是二维 `nkx × nky` 求解器，没有 `nkz`。
- 因此 bulk wurtzite ZnO 的 `ZnO_tb.dat` 直接喂给当前 SBE 时，本质上是 **kz=0 切片响应**，不是完整 3D bulk ZnO 响应。
- 这仍然适合做代码健康检查，但不适合直接和 bulk ZnO HHG 文献做定量比较。

## 2. 本地路径与服务器路径

本地工作目录：

```text
C:\Users\26507\Documents\New_SBEs\Quantum-light\examples\ZnO_wurtzite_qe_wannier
```

本地下载与后处理输出目录：

```text
C:\Users\26507\Documents\New_SBEs\Quantum-light\examples\ZnO_wurtzite_qe_wannier\output
```

服务器运行目录：

```text
/public/home/wangjs/project/ZnO_TB
```

赝势本地来源：

```text
C:\Users\26507\Documents\量子光研究\ZnO_wannier_inputs\pseudo
```

服务器运行时赝势目录：

```text
/public/home/wangjs/project/ZnO_TB/pseudo
```

## 3. 体系与 QE 基本参数

体系：

- 材料：wurtzite ZnO
- 原子数：4 个原子，2 Zn + 2 O
- 泛函：PBE
- SOC：无
- 磁性：无
- 赝势：
  - `Zn.pbe-dnl-kjpaw_psl.1.0.0.UPF`
  - `O.pbe-n-kjpaw_psl.1.0.0.UPF`

Zn 赝势信息：

- `z_valence = 12`
- Zn `3d` 作为价态处理。
- 这对 ZnO 很重要，因为 Zn-3d 与 O-2p 杂化影响价带结构。

晶格参数：

```text
ibrav = 4
celldm(1) = 6.138 Bohr
celldm(3) = 1.6020
```

对应 Wannier90 `.win` 中使用的 Å 单位晶格矢量：

```text
a = 3.2480897205 Å
c = 5.2034397323 Å
```

原子位置：

```text
Zn  0.0000000000  0.0000000000  0.0000000000
Zn  0.6666666667  0.3333333333  0.5000000000
O   0.0000000000  0.0000000000  0.3819000000
O   0.6666666667  0.3333333333  0.8819000000
```

截断能：

```text
ecutwfc = 50 Ry
ecutrho = 400 Ry
```

该设置比赝势推荐值更保守，适合验证用途。

## 4. 输入参数文件

### 初始输入文件

| 文件 | 作用 | 关键参数 |
|---|---|---|
| `ZnO_scf.in` | QE SCF | `8×8×6` automatic k grid，`ecutwfc=50`，`ecutrho=400` |
| `ZnO_bands.in` | QE path bands | 路径 `A→L→M→Γ→A→H→K`，每段 20 点，`nbnd=52` |
| `ZnO_nscf.in` | QE NSCF for Wannier | 显式 `8×8×6=384` k 点，`nosym=.true.`，`noinv=.true.`，`nbnd=52` |
| `ZnO.pw2wan` | `pw2wannier90.x` 输入 | `write_mmn=.true.`，`write_amn=.true.`，`write_eig=.true.` |
| `ZnO.win` | 初始 Wannier90 输入 | `num_bands=52`，`num_wann=20`，投影 `Zn:s;d` 和 `O:s;p`，初始版本不含能量窗口 |
| `run_all.sh` | 一键运行草稿 | 顺序为 `scf→bands→nscf→wannier90 -pp→pw2wannier90→wannier90` |

### k 点一致性检查

已经确认：

- `ZnO_nscf.in` 中显式 k 点数：384。
- `ZnO.win` 中 `begin kpoints` 点数：384。
- 两者逐点一致、顺序一致。

这点很重要，因为 `pw2wannier90.x` 要求 NSCF k 点与 `.win` 中的 k 点完全对应。

## 5. Wannier90 能量窗口选择

从 `ZnO.eig` 与 QE 输出得到：

```text
最低能量：-10.809321783950 eV
最高能量： 54.769658625442 eV
highest occupied level = 7.3080 eV
lowest unoccupied level = 8.0858 eV
PBE gap ≈ 0.7778 eV
```

窗口内带数检查：

```text
dis_win [-12, 15] eV: min bands/kpt = 19
dis_win [-12, 18] eV: min bands/kpt = 21
dis_win [-12, 20] eV: min bands/kpt = 21
dis_win [-12, 25] eV: min bands/kpt = 26
```

最终采用第一版窗口：

```text
dis_win_min  = -12.0
dis_win_max  = 18.0
dis_froz_min = -12.0
dis_froz_max = 7.8
```

选择理由：

- `[-12, 15]` 最少只有 19 条带，小于 `num_wann=20`，不安全。
- `[-12, 18]` 最少有 21 条带，刚好够 20 WF 解纠缠。
- `dis_froz_max=7.8 eV` 位于 VBM 和 CBM 之间，可冻结完整占据带，但不强行冻结导带。
- 更宽的 `[-12,25]` 包入更多高导带，可能让 disentanglement 更不稳定；第一版验证不需要。

## 6. 服务器运行得到的数据

下载到本地的主要输出文件：

| 文件 | 大小 | 含义 |
|---|---:|---|
| `output/ZnO.eig` | 579,072 B | NSCF 本征值，供 Wannier90 解纠缠使用 |
| `output/ZnO.win` | 21,129 B | 最终加入能量窗口后的 Wannier90 输入 |
| `output/ZnO.wout` | 2,108,191 B | Wannier90 输出日志 |
| `output/ZnO_band.dat` | 64,080 B | Wannier90 path-band 插值结果 |
| `output/ZnO_centres.xyz` | 1,415 B | 20 个 Wannier center + 4 个原子位置 |
| `output/ZnO_tb.dat` | 28,909,003 B | SBE 读入用核心 TB 文件 |
| `output/ZnO_qe_bands.dat` | 197,791 B | 注意：这是被 NSCF 覆盖后的 8×8×6 均匀网格 bands 数据，不是路径能带 |
| `output/ZnO_qe_bands.dat.gnu` | 0 B | QE 未写入有效内容，可忽略 |
| `output/ZnO_qe_bands_path.dat` | 62,346 B | 正确的 QE path-band 数据，`nks=121` |

Wannier90 成功标志：

```text
<<< Disentanglement convergence criteria satisfied >>>
Final Spread (Ang^2) Omega Total = 13.146671777
All done: wannier90 exiting
```

`ZnO_tb.dat` 头部信息：

```text
nwann = 20
nrpts = 469
```

## 7. 已发现并修正的 bands.x 文件问题

第一次下载的 `ZnO_qe_bands.dat.gnu` 是 0 字节，同时 `ZnO_qe_bands.dat` 虽然非空但第一行为：

```text
&plot nbnd=52, nks=384 /
```

这说明它对应的是 NSCF 的 `8×8×6` 均匀 k 网格，而不是 path bands。原因是：

1. 先运行了 `ZnO_bands.in`。
2. 后来运行 `ZnO_nscf.in`，两者使用相同的 `prefix='ZnO'` 和 `outdir='./tmp'`。
3. `ZnO_nscf.in` 覆盖了 `tmp/ZnO.save` 的 k 点信息。
4. 之后再跑 `bands.x`，读到的就是 NSCF 网格。

修正方法：

```bash
cd /public/home/wangjs/project/ZnO_TB

PW=/public/home/wangjs/software/q-e-qe-7.2/bin/pw.x
mpirun -np 4 $PW < ZnO_bands.in > ZnO_bands_path.out 2>&1

cat > ZnO_bands_path_pp.in <<'EOF'
&BANDS
  prefix  = 'ZnO'
  outdir  = './tmp'
  filband = 'ZnO_qe_bands_path.dat'
/
EOF

BANDS=/public/home/wangjs/software/q-e-qe-7.2/bin/bands.x
mpirun -np 4 $BANDS < ZnO_bands_path_pp.in > ZnO_bands_path_pp.out 2>&1
```

正确结果：

```text
&plot nbnd=52, nks=121 /
```

该文件已下载为：

```text
output/ZnO_qe_bands_path.dat
```

## 8. 本地后处理结果

已生成的图和摘要：

| 文件 | 用途 |
|---|---|
| `output/ZnO_wannier_band_quicklook.svg` | 只看 Wannier90 插值带的 quicklook |
| `output/ZnO_qe_vs_wannier_bands.png` | QE vs Wannier 全能量范围对比图 |
| `output/ZnO_qe_vs_wannier_bands.svg` | 同上，SVG 版本 |
| `output/ZnO_qe_vs_wannier_bands_near_gap.png` | 近带隙放大初版 |
| `output/ZnO_qe_vs_wannier_bands_near_gap_clean.png` | 近带隙放大清理版 |
| `output/ZnO_qe_vs_wannier_bands_near_gap_pubstyle.png` | 推荐使用的近带隙对比图 |
| `output/ZnO_wannier_validation_summary.txt` | 简单误差统计摘要 |
| `output/ZnO_near_gap_plot_note.txt` | 近带隙图说明 |

推荐图：

```text
output/ZnO_qe_vs_wannier_bands_near_gap_pubstyle.png
```

图中：

- 灰线：QE `bands.x` 路径能带。
- 蓝线：Wannier90 20-WF 插值能带。
- 能量已平移为 `E - E_VBM`。
- 黑色虚线：VBM = 0。
- 绿色虚线：CBM = 0.778 eV。
- 纵轴范围：`-5 eV` 到 `15 eV`，便于和文献常见 ZnO 能带图对比。

误差摘要：

```text
QE nbnd=52 nks=121
Wannier bands=20 points_per_band=97
RMS error first20 normalized path: 0.301350 eV
MAE error first20 normalized path: 0.067345 eV
RMS error first20 for energies <=18 eV: 0.301350 eV
MAE error first20 for energies <=18 eV: 0.067345 eV
Max abs error: 2.179561 eV at band,k,qe,w,err=(20, 25, 13.334, 15.513561284586663, 2.179561284586663)
```

解释：

- 平均误差较小，主要带隙附近和占据带匹配良好。
- 最大误差出现在第 20 条带、窗口顶部的高导带区域。
- 对于 SBE 健康检查，这套 20-WF 模型足够使用。
- 如果未来要做 ZnO 文献级定量 HHG，对高导带区域更敏感，建议尝试 `num_wann=22/24/26` 或扩大能量窗口。

## 9. 当前判断

这套 ZnO Wannier 模型已经通过第一版验收：

- QE 与 Wannier 在 VBM/CBM 附近吻合良好。
- Zn-3d/O-2p 价带区域总体复现良好。
- 高导带窗口顶部存在偏差，但不影响当前“求解器健康检查”的目的。
- `ZnO_tb.dat` 文件较小，适合快速测试 SBE。

因此可以进入 SBE 验证阶段。

## 10. 建议的 ZnO SBE 健康检查参数

建议第一版 SBE 输入参数：

```text
wannier_tb_file = '/public/home/wangjs/project/ZnO_TB/ZnO_tb.dat'
SOC = 0
E_fermi_eV = 7.70
nv_orig = 18
nb_start = 1
nb_end = 20
gauge_method = 'lg_cov'
```

参数说明：

- ZnO 每原胞价电子数：`2×12 + 2×6 = 36`。
- 无 SOC、非自旋极化时每条能带可容纳 2 个电子，因此占据带数 `nv_orig=18`。
- `E_fermi_eV=7.70 eV` 位于 `VBM=7.3080 eV` 和 `CBM=8.0858 eV` 之间。
- `nb_start=1, nb_end=20` 使用完整 20-WF 模型。

建议测试顺序：

1. `lg_cov / 20×20 / nb1-20 / no dephasing`
2. 检查 `Initial current |J(t=0)|` 是否接近机器零。
3. 检查 `TB Matrix Quality Diagnostics` 是否正常。
4. 跑一个弱场 scaling 测试。
5. 如果一切正常，再考虑用 ZnO 的已知 HHG 文献参数做更接近文献的对比。

## 11. 仍需注意的事项

1. **当前 SBE 是 2D 求解器。**  
   ZnO 是 3D bulk 体系，所以当前测试只能作为 `kz=0` 切片健康检查。

2. **没有 orbital projection / PDOS 数据。**  
   当前图没有右侧 `O 2s / O 2p / Zn 3d / Zn 4s` 投影 DOS。若需要复现类似文献图，应额外运行 `projwfc.x` 或生成 fat-band 权重。

3. **`.gnu` 文件为空不是错误。**  
   本次有效 QE path-band 数据是 `ZnO_qe_bands_path.dat`，不是 `ZnO_qe_bands.dat.gnu`。

4. **若后续要做定量 ZnO HHG，可能需要更大模型。**  
   20 WF 已足够用于当前 SBE 健康检查；若要拟合高导带或高阶 HHG，建议测试更宽窗口或更多 WF。

## 12. ZnO HHG 文献参照点

这套 ZnO 计算最适合用作“求解器健康检查 + 定性文献对照”，而不是完整复现 bulk ZnO 实验谱。

当前最有用的参照文献：

1. Ghimire et al., *Observation of high-order harmonic generation in a bulk crystal*, Nature Physics 7, 138-141 (2011), DOI: `10.1038/nphys1847`。  
   这是 bulk ZnO HHG 的经典实验。核心事实：ZnO 在长波长 few-cycle 强场下产生超过带边的高阶谐波，谱呈非微扰特征，cutoff 随驱动电场近似线性增加。

2. Vampa et al., *Linking high harmonics from gases and solids*, Nature 522, 462-464 (2015), DOI: `10.1038/nature14517`。  
   该文把 solid ZnO HHG 和气体 recollision 图像联系起来，实验使用 mid-IR 驱动并讨论电子-空穴复碰机制。文中给出的强场尺度约为 `0.25 V/Angstrom`，对应电场约 `0.00486 a.u.`，强度约 `8.3e11 W/cm^2`。

3. Yue and Gaarde, *Structure- and laser-gauges for the semiconductor Bloch equations in high-harmonic generation in solids*, arXiv:2003.02961。  
   该文直接讨论 SBE 的 length/velocity gauge、截断与结构规范问题，并用 1D ZnO 模型作为示例。它支持当前路线中优先用 `lg_cov` 做带截断和退相干计算的判断。

4. Jiang et al., *Crystal Symmetry and Polarization of High-order Harmonics in ZnO*, arXiv:1908.04161。  
   该文强调 ZnO HHG 的奇偶阶、偏振和晶体取向依赖主要受晶体对称性控制。若后续要测试偶次谐波和偏振响应，这篇比 Ghimire 更适合作为对称性参照。

对当前代码最合理的对比指标：

- 初态电流：`J(t=0)` 应接近机器零。
- 弱场标度：低阶谐波随场强呈正确的微扰趋势。
- 场强趋势：增大强度后，高阶谱尾应增强，cutoff 应向高阶移动。
- 对称性趋势：wurtzite ZnO 非中心对称，因此偶次谐波允许出现；但当前 2D `kz=0` 切片不能直接当作完整 bulk 对称响应。

不建议当前阶段强行对比：

- 绝对谐波强度；
- bulk ZnO 的完整 cutoff 阶数；
- 与实验谱的一一对应峰高；
- 取向依赖的定量结果。

原因是当前 SBE 仍是 2D `kz=0` 切片，且 20-WF 模型在高导带窗口顶端仍有偏差。

## 13. 已完成的本地 ZnO SBE 健康检查

本地使用当前 `Quantum-light` 源码重新编译得到 `hhg_sbe`，然后运行了三个 `lg_cov` ZnO 切片测试：

```text
local_sbe_runs/zno_lgcov_k10_health
local_sbe_runs/zno_lgcov_k20_health
local_sbe_runs/zno_lgcov_k40_health
```

共同参数：

```text
wannier_tb_file = '../../output/ZnO_tb.dat'
SOC = 0
E_fermi_eV = 7.70
nv_orig = 18
nb_start = 1
nb_end = 20
gauge_method = 'lg_cov'
wvl_nm = 3200.0
intensity_Wcm2 = 2.0e11
ncyc = 2.0
T2_fs = 1.0e30
dt = 0.50 a.u.
```

运行结果：

| case | k grid | wall time | `|J(t=0)|` | 结论 |
|---|---:|---:|---:|---|
| `zno_lgcov_k10_health` | `10×10` | 16.30 s | `3.68e-17` | 跑通，初态电流接近机器零 |
| `zno_lgcov_k20_health` | `20×20` | 53.18 s | `6.97e-18` | 跑通，初态电流更干净 |
| `zno_lgcov_k40_health` | `40×40` | 201.02 s | `6.38e-19` | 跑通，可作为当前本地健康检查基准 |

TB 质量诊断（以 40×40 为例）：

```text
H(k)   max|H-H+|=  8.05E-16  avg rel_frob=  2.82E-16
r_x(k) max|r-r+|=  2.63E-01  avg rel_frob=  3.01E-02
r_y(k) max|r-r+|=  2.25E-01  avg rel_frob=  5.01E-02
r_z(k) max|r-r+|=  1.70E-01  avg rel_frob=  1.20E-02
v_x(k) max|v-v+|=  3.95E-02  avg rel_frob=  6.14E-02
v_y(k) max|v-v+|=  3.65E-02  avg rel_frob=  5.78E-02
v_z(k) max|v-v+|=  3.84E-02  avg rel_frob=  7.87E-02
```

解释：

- `H(k)` 厄米性非常好，说明 TB Hamiltonian 读入正常。
- `J(t=0)` 随 k 网格加密进一步接近零，说明占据带数 `nv_orig=18` 和初态填充逻辑是正确的。
- `r(k)` 与 `v(k)` 的 raw anti-Hermitian 诊断比 CrI3 大。代码在实际传播前会对 `D/P` 做 Hermitize，因此健康检查可以继续；但这说明当前 20-WF ZnO 模型不宜直接用于文献级定量 HHG。

生成的本地 SBE 谱图：

```text
local_sbe_runs/ZnO_lgcov_k10_k20_k40_HHG_health.png
```

当前图像判断：

- 低阶谐波可以用于健康检查。
- 高阶谱仍有明显 k 网格敏感性。
- 若要做更接近文献的 ZnO 对比，至少需要使用 `40×40` 以上，并最好扩展到真正的 3D k 网格。

## 14. 下一步建议

短期建议：

1. 在本地继续跑 ZnO 弱场 scaling：
   - `2e11`
   - `5e10`
   - `1.25e10`
   检查 H1/H2/H3 的场强指数是否合理。

2. 跑 ZnO 强场趋势测试：
   - `2e11`
   - `4e11`
   - `8e11`
   对应接近 Vampa 2015 的 `0.25 V/Angstrom` 强场量级。目标不是复现实验谱，而是确认高阶尾部随场增强。

3. 保持 `40×40` 作为本地 ZnO 健康检查基准。20-WF 模型很小，本地可承受。

中期建议：

1. 如果要对 ZnO 文献做更认真对比，优先考虑扩展 SBE 到 3D `nkx×nky×nkz`。
2. 如果继续用 2D 切片，则报告中必须明确写成 “ZnO `kz=0` slice benchmark”，不能写成 full bulk ZnO。
3. 若要提高高导带与光学矩阵元质量，可重做 `num_wann=26` 的 ZnO Wannier 模型，加入 Zn:p 投影。

