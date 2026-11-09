# 层 A0 代码执行记录：Gaussian‑Q 采样、复数谱与 ICS/CS（2026-07-29）

> **对应理论**：`量子光驱动HHG与输出光子统计_完整理论与计算方法_20260729`（尤其第四部分 §11–§15）  
> **本记录性质**：Git 工作区实际改动日志；不重复公式推导。  
> **范围**：层 A 校订（A0.1–A0.4），**尚未**进入两时关联 / \(g^{(2)}\) / 压缩。

---

## 0. 一句话结论

已按理论笔记完成层 A0 的最小可运行切片：**历史“BSV”采样改名并保留**、**真正 Gaussian Husimi‑Q 采样（coherent / SV / …）**、**复数 \(J_x(\omega),J_y(\omega)\) 导出**、系综 **ICS / CS / classical_trajectory_variance**；单元测试 Q0 门禁通过，主程序 `hhg_sbe` 编译通过。

---

## 1. 改动文件清单

| 文件 | 作用 |
|------|------|
| `src/mod_quantum_light.f90` | 重写：`state_type`、指数基准、Gaussian \(Q\)、Q0 矩报告 |
| `src/mod_hhg.f90` | `compute_complex_current_spectrum`、`write_hhg_complex`、`write_hhg_ics_cs` |
| `src/mod_params.f90` | 扩展 `&bsv` / `&output` namelist |
| `src/main.f90` | 统一 `qlight_sample_drive`；ICS/CS 累计；可选复数输出 |
| `src/mod_ensemble.f90` | `S_quant` → `classical_trajectory_variance`；接新采样器 |
| `tests/test_qlight_sampling.f90` | 指数分布 + Q0（coherent / SV）验收 |
| 本文 | 工作记录 |

---

## 2. 命名与物理边界（必须遵守）

| 旧称呼 | 新称呼 / 含义 |
|--------|----------------|
| `qlight_sample_bsv` / “BSV 采样” | **`random_phase_exponential`**（圆对称指数强度 + 均匀相位）。保留作历史基准；**不是**固定压缩角单模 SV。 |
| `S_quant` | **`classical_trajectory_variance` = ICS − CS**。经典轨迹方差，**不是**量子源噪声。 |
| `HHG_bsv.dat` | 仍写文件名以兼容旧脚本，内容明确为 **ICS 平均谱（层 A）**。 |

硬边界（来自理论总目录）：

\[
\text{Husimi-}Q\text{ 平均谱}
\neq
\text{完整输出量子态}.
\]

本阶段**禁止**把轨迹矩命名为 quantum \(g^{(2)}\)、squeezing、entanglement。

---

## 3. 接口用法

### 3.1 `&bsv` namelist（可选字段均有默认值，旧输入仍可跑）

```fortran
&bsv
  bsv_enabled = .true.
  bsv_n_samples = 100
  bsv_mean_intensity = 1.0e11   ! I_bar; 驱动均值强度目标 = 2*I_bar
  bsv_seed = 42
  bsv_state_type = 'random_phase_exponential'
  ! 或: 'coherent' | 'thermal' | 'squeezed_vacuum' | 'displaced_squeezed'
  bsv_squeeze_r = 0.0
  bsv_squeeze_theta_deg = 0.0
  bsv_alpha0_abs = 0.0
  bsv_alpha0_phase_deg = 0.0
  bsv_thermal_nbar = 0.0
  bsv_save_complex = .false.
  bsv_save_ics_cs = .true.
/
```

别名：`bsv` / `rpe` → `random_phase_exponential`；`sv` → `squeezed_vacuum`。

### 3.2 经典单轨迹复数谱

```fortran
&output
  save_complex_hhg = .true.
/
```

产出：`HHG_complex.dat`（Hann 窗 FFT 的 \(\mathrm{Re}/\mathrm{Im}\,J_{x,y}\) + 与 `HHG.dat` 同 convention 的强度列）。

### 3.3 系综产出

| 文件 | 内容 |
|------|------|
| `HHG_bsv.dat` | ICS 平均 + stderr（兼容旧名） |
| `HHG_ics_cs.dat` | ICS / CS / classical_trajectory_variance |
| `HHG_complex_sample001.dat` | 仅当 `bsv_save_complex=.true.`：第 1 条轨迹复数谱 |

Gaussian \(Q\) 路径将 \(\alpha\) 映射为

\[
I = I_{\mathrm{mean}}\frac{|\alpha|^2}{\mathbb E_Q[|\alpha|^2]},
\quad
I_{\mathrm{mean}}=2I_{\mathrm{bar}},
\quad
\mathbb E_Q[|\alpha|^2]=\langle\hat n\rangle+1.
\]

压缩椭圆转角使用 \(\theta_s/2\)（理论式 2.23 / §12.1）。

---

## 4. 验收结果（本机 2026-07-29）

### 4.1 单元测试

```text
cd tests
# PATH 需含 C:\msys64\mingw64\bin
gfortran -O2 -std=f2008 -o test_qlight_sampling.exe \
  ../src/mod_quantum_light.f90 test_qlight_sampling.f90
./test_qlight_sampling.exe
```

结果：

| 门 | 结果 |
|----|------|
| [1] random_phase_exponential \(\langle I\rangle\) / Var / \(\phi\) | **PASS** |
| [2a] Q0 coherent（\(d,V_Q,N\)） | **PASS** |
| [2b] Q0 squeezed_vacuum（含 \(M\)） | **PASS** |
| [3] legacy `qlight_sample_bsv` 别名 | 可用 |

### 4.2 主程序编译

```text
mingw32-make clean && mingw32-make
→ hhg_sbe.exe 链接成功
```

（`mod_sbe.f90` 既有 maybe-uninitialized 警告，与本次改动无关。）

---

## 5. 与理论阶段门对照

| 理论阶段 | 本记录状态 |
|----------|------------|
| Q0 输入态矩 | ✅ 单模 coherent / SV 测试通过 |
| A0 命名 + \(Q\) 采样 + 复数 \(J_n\) + ICS/CS | ✅ 代码落地 |
| A0 ±N 共享节点 + 确定性求积 | ✅ manifest / GH3·5·7 / exponential_quad / from_file |
| hBN 端到端验收 | ✅ `examples/hBN_model_tb/a0_e2e/a0_e2e_report.md` OVERALL PASS |
| A1 classical-mixture \(g^{(2)}_{nm}\) 跨谐波矩阵 | ⏸ 仅有对角 ICS/CS；\(A_{nm},B_{nm},F_{nm}\) 未存 |
| B0 小系统电流关联 | ❌ 未开始 |
| 精确 \(g^{(2)}\) / 压缩 | ❌ **禁止**用当前轨迹矩宣称 |

---

## 8. A0.5 端到端补齐与服务器探索（2026-08-02）

### 8.1 代码审查结论（对照场外 AI）

| 问题 | 结论 |
|------|------|
| IDE `mod_wannier.mod` 找不到 | **假阳性**：模块输出在 `mod/`，编译正常 |
| 仅 Q0 单元测试、无端到端 | ✅ 已用 hBN 完成经典复数回归 / 相干极限 / SV 方向 smoke |
| 无 ±N 共享节点 | ✅ `nodes_manifest` + `from_file`；±N 读同一清单 |
| 等平均强度映射 | **保留并明示**：\(I=I_{\rm mean}|\alpha|^2/\mathbb E_Q|\alpha|^2\)，非实验 BSV 能量标定 |
| 生产模型 | 按用户修订：**M88-full@k50**（含 H10），探索先 **M88-full@k20** |

### 8.2 新增接口

- `bsv_sampling_mode = mc | gauss_hermite | exponential_quad | from_file`
- `bsv_gh_order = 3|5|7`；`bsv_nodes_file`；`bsv_harmonics='2,5,7,9,10'`
- 输出：`nodes_manifest.dat`、`HHG_nodes_modes.dat`、加权 `HHG_ics_cs.dat`

### 8.3 hBN 验收摘要

见 `examples/hBN_model_tb/a0_e2e/a0_e2e_report.md`：

1. `HHG.dat` vs `HHG_complex` 强度：相对误差 `0`（PASS）
2. coherent GH3，⟨I⟩对齐经典：H1/H3/H5 ICS/经典 ≈ `1.001 / 1.011 / 0.987`（PASS）
3. SV \(r=0.8\)，\(\theta_s=0^\circ/180^\circ\)：ICS≥CS，节点 \((I,\phi)\) 分布改变（PASS）

### 8.4 服务器 part_1 作业

- 作业号：**27674**（array 0–23，%6）
- 配置：M88-full，k20，GH3，\(r\in\{0.5,1.5,2.5\}\)、\(\theta_s\in\{0,90,180,270\}\)、±N 共享节点
- 脚本：`deploy/a0_layerA_m88full/`
- 完成后：`tools/analysis/combine_pmN_nodes.py` 算 \(J_o\)
- 下一阶段：高场节点 c20/full 抽查 → **M88-full@k50** 生产（`sbatch_prod_k50.sh.template`）

### 8.5 仍未做

1. A1 跨谐波 \(A_{nm},B_{nm},F_{nm}\)
2. 探索完成后的自动 down-select → k50 生产提交
3. B0 小系统两时关联

---

## 9. 给后续 Agent 的最短指令

> A0.5 已落地且 hBN e2e PASS；服务器探索作业 **27674** 在跑。完成后配对 ±N → 选 2–3 个 \((r,\theta)\) → 提交 M88-full@k50 生产。**不要**把 ICS−CS 写成量子压缩。H10 走 M88-full，不用 c20。
