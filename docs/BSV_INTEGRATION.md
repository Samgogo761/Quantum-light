# BSV 量子光驱动 HHG —— Wannier90 + Fortran SBE 求解器集成指南

本仓库提供把已有的经典 SBE 求解器升级成 **BSV（Bright Squeezed Vacuum）量子光驱动 HHG** 所需的全部**新增代码**和**补丁片段**。

理论依据：Nature Physics 2023 Supplementary 公式 1.32 / II.7。在随机相位近似下，BSV 的 Husimi-Q 分布在强度轴上退化为指数分布

$$Q_{BSV}(I) \;=\; \tfrac{1}{2\bar I}\,\exp\!\bigl(-I/2\bar I\bigr)$$

因此算法只是**在经典 SBE 外包一层蒙特卡洛循环**——每条轨迹从指数分布抽一个强度、从均匀分布抽一个相位，独立跑 RK4，然后对 $|J(\omega)|^2$ 做系综平均。经典路径在 `enabled=.false.` 下字节级不变。

---

## 1. 仓库结构

```
src/
  mod_quantum_light.f90   # 新增：BSV 抽样器（独立可编译）
  mod_ensemble.f90        # 新增：MC 主循环 + 累加器 + checkpoint
tests/
  test_qlight_sampling.f90  # 独立单元测试（只依赖 mod_quantum_light）
  Makefile                  # gfortran 构建
patches/
  mod_laser_snippet.f90   # 参考实现：calc_laser2d_sample / calc_laser2d wrapper
  mod_sbes_snippet.f90    # 参考实现：init_density_matrix / sbe_run_single_trajectory
  main_snippet.f90        # 参考实现：main.f90 分叉 + read_input 扩展
examples/
  wannier_sbe_input.example.txt  # 新增 &quantum_light namelist 组
docs/
  BSV_INTEGRATION.md      # 本文档
```

`src/` 下两个 `.f90` 可以**直接拷到你的工程里**。`patches/` 下三个 `.f90` 是**参考片段**——因为你的私有求解器我看不到，需要你按片段对位替换/合并到自己的 `mod_laser.f90`、`mod_sbes.f90`、`main.f90` 里。

---

## 2. 集成步骤

### 第 1 步：落盘两个新模块

把 `src/mod_quantum_light.f90` 和 `src/mod_ensemble.f90` 放进你工程的 `src/` 目录，加进编译规则（Makefile / CMakeLists）。依赖关系：

```
mod_quantum_light   <- (自包含)
mod_ensemble        <- mod_quantum_light, mod_laser, mod_sbes
```

### 第 2 步：先跑抽样器单元测试

**在动任何求解器代码之前**先验证抽样器本身：

```bash
cd tests/
make
./test_qlight_sampling
```

预期输出：

```
<I>          ≈ 2 * I_bar       （相对误差 < 0.5%）
Var(I)       ≈ (2 * I_bar)^2   （相对误差 < 1%）
phi 直方图   平坦
RESULT : PASS
```

这对应 plan 的验证方案 §1。

### 第 3 步：按 `patches/mod_laser_snippet.f90` 改造 `mod_laser.f90`

**核心是把写死 $\cos^2$ 包络的 `calc_laser2d()` 拆成：**

- `calc_laser2d_sample(E_peak_SI, phi_0)` —— 真正的构场函数，接收绝对峰值电场（SI）和载波相位。
- `calc_laser2d()` —— 经典 wrapper，计算出经典 $E_0$ 后调用 `calc_laser2d_sample(E_0, 0.0_dp)`。

**经典回归测试（plan §2）**：改造后在 `enabled=.false.` 下重跑，HHG 谱应与改造前逐点一致到 RK4 误差量级。

### 第 4 步：按 `patches/mod_sbes_snippet.f90` 改造 `mod_sbes.f90`

新增两个 public 子例程：

- `init_density_matrix()` —— 把 `rho_H` 重置为 $|v\rangle\langle v|$。
- `sbe_run_single_trajectory(Jt, Jw)` —— 无 IO 版本，复用原有 `sbe_2d_propagation_ht`，但通过 module 级 logical `single_traj_mode=.true.` 跳过所有 `open/write HHG/...` 调用，直接把 $J(t)$ 和 FFT 后的 $J(\omega)$ 从参数传回。

**你需要在 `sbe_2d_propagation_ht` 里所有文件 IO 处加 `if (.not. single_traj_mode) then ... end if`**（或类似的 gate），保证 MC 循环里数千条轨迹不会彼此覆盖 `HHG/` 目录。

### 第 5 步：按 `patches/main_snippet.f90` 改造 `main.f90`

入口处分叉：

```fortran
if (.not. qp%enabled) then
  call calc_laser2d()       ! 经典路径
  call calc_sbes(...)
else
  call run_mc_ensemble(qp, nw = n_w_hhg, nt = n_t, prefix = 'HHG/bsv_spectrum')
end if
```

并在 `read_input` 中加一段 `namelist /quantum_light/ enabled, I_bar, n_samples, seed`，缺省值由 `qlight_params_t` 提供，因此**老的输入文件不加这个 namelist 也不会报错**。

### 第 6 步：往 `wannier_sbe_input.txt` 追加新 namelist

```
&quantum_light
  enabled   = .true.
  I_bar     = 2.0e11     ! W/cm^2 —— 这是**平均强度**，不是峰值
  n_samples = 500
  seed      = 42
/
```

完整示例见 `examples/wannier_sbe_input.example.txt`。

---

## 3. 输出与分析

`run_mc_ensemble` 在每 50 条轨迹后写一次 checkpoint、跑完后写最终文件，统一格式：

```
# BSV ensemble checkpoint, N = <n>
# iw    S_total         S_coh           S_quant         sigma
```

四列物理含义：

| 列 | 公式 | 物理含义 |
|---|---|---|
| `S_total` | $\frac{1}{N}\sum_i \lvert J_i(\omega)\rvert^2$ | 非相干平均谱，就是你要测的 HHG 输出 |
| `S_coh`   | $\bigl\lvert\frac{1}{N}\sum_i J_i(\omega)\bigr\rvert^2$ | 相干部分，真空场下对应"经典极限" |
| `S_quant` | `S_total - S_coh` | 量子涨落的纯贡献 |
| `sigma`   | Welford 蒙卡标准差 | 每频点的 1-σ 误差棒 |

---

## 4. 验证方案（落实 plan §2 – §6）

| 检查 | 操作 | 预期 |
|---|---|---|
| §1 抽样器 | `make -C tests && ./tests/test_qlight_sampling` | `<I>=2Ī`、`Var=(2Ī)²`、φ 平坦 |
| §2 经典回归 | `enabled=.false.` 重跑老算例 | HHG 逐点 ≈ 改造前（RK4 量级） |
| §3 极限 | `n_samples=1`，手动固定 $I=2\bar I$, $\varphi=0$ | 谱 ≈ 经典谱 |
| §4 收敛性 | 扫 `n_samples = 50/100/200/500/1000` | 低阶峰 $\propto 1/\sqrt{N}$ 收敛，`sigma` 单调下降 |
| §5 物理 | 对比相同 $2\bar I$ 的经典谱 vs BSV 谱 | Cutoff 外移 / 阈值降低（Nature Phys 2023 Fig. IV.2） |
| §6 工程 | 任意杀任务后 checkpoint 重启 | 最终谱与不中断版一致；验证无内存泄漏 |

---

## 5. v2 路线（已在 plan 中列出）

- OpenMP 并行：外层 `do i = 1, n_samples` 加 `!$OMP PARALLEL DO REDUCTION(+:...)`；需要线程私有 `rho` / 线程级 RNG。
- Importance sampling：对指数长尾用 $I\cdot p(I)$ 加权，压缩方差。
- 双色 / 多模 BSV（TMSV）。
- Fixed-CEP BSV：退回二维 Wigner 抽样。

---

## 6. 数值与工程注意事项

1. **数值尾部**：`qlight_sample_bsv` 里对均匀数 `u` 做了 `[eps, 1-eps]` 的 clamp，避免 `log(0)` 产生 `NaN`。即便如此，`I` 的指数分布长尾依然很厚——建议在分析脚本里把 `I > 50*I_bar` 的离群样本单独标出。
2. **每条轨迹的临时数组**（`rho`、`Jt`、`Jw`）必须在轨迹结束前 `deallocate`；`mod_ensemble` 里的 `Jt_i/Jw_i` 已经处理好，你在 `sbe_run_single_trajectory` 内若分配了额外工作数组也要一并释放。
3. **Ī 是平均强度，不是峰值强度**：样本均值 $\langle I\rangle = 2\bar I$。写入 namelist 时别跟经典 `intensity`（峰值）混淆。
4. **单位**：`I_bar` 约定为 `W/cm^2`；`calc_laser2d_sample` 接收的是 `V/m` 的 SI 峰值电场，转换在 `mod_ensemble` 内部完成。
5. **CEP vs 随机相位**：这版实现是"**随机 CEP**"——每条轨迹独立随机，系综平均后 $\langle E(t)\rangle=0$ 是正常的，物理观测量一定要放在 $|J(\omega)|^2$ 层面平均。
