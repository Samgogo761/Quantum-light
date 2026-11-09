# 线 Q 物理冻结规范（Q_THEORY_FROZEN_SPEC）

日期：2026-09-03。物理定义的唯一裁判是 `pt_afm_toy.py` 中的 `FROZEN_SPEC` 与 `PT_AFM_TOY.md` 的 \(H_0\)。
本文件的键值块由 `test_q_frozen_spec.py` 与代码逐项比对；任何 CPU / GPU backend 必须通过 `assert_frozen_spec`。

```text
line = Q_THEORY
model = current_4band_pt_afm_toy
gauge_or_representation = Houston_characteristics
k_evolution = k0 + A(t)
equation = i drho/dt = [H(k(t)), rho]
integrator = midpoint_unitary_exponential
dephasing = none
open_system = False
current_operator = -dH/dkx
wick_equal_time = wick.equal_time_wick
wick_two_time = wick.two_time_wick_greater
two_time_driver = b1_dev_closed_wick.closed_wick_reference
dtype = complex128
floating_point = FP64
k_points = current_existing_small_grid
k_points_are_independent = True
quantum_light_ensemble = none
pump = classical_coherent_prescribed_field
absolute_k_normalization_validated = False
k_normalization_provenance = discrete_weight_squared
material_interpretation = False
cri3_claim = False
current_4band_jo_status = STRUCTURAL_ZERO_CANDIDATE
jo_h2_hard_test_enabled = False
model_role = zero_response_numerical_harness
initial_state = two_lowest_bands_per_k_slater
light_matter_coupling = peierls_k_shift
propagation_coordinate = houston_characteristic
default_run_toy_a_rule = sampled_A_arithmetic_mean
h2_label = finite_pulse_background
```

`model_version` 取 `pt_afm_toy.py` 内容的 sha256 前 16 位（文件尚未入 git），由 `model_version()` 给出，写进每个结果的 metadata。

`default_run_toy_a_rule` 是旧回归：取样 \(A\) 的算术平均。新 harness 用 `evaluate_A_at_t_mid`。初态是逐 \(k\) 最低两带 Slater，**不是**全局半填充。`h2_label=finite_pulse_background`：有限脉冲在 \(2\omega\) 的谱权重，不是磁性 H2。

`light_matter_coupling` 与 `propagation_coordinate` 拆开记录 Peierls \(k\)-shift 与 Houston 特征线；**方程未改**。

## 方程（与 `PT_AFM_TOY.md` 一致，不得改）

\[
\mathbf k(t)=\mathbf k_0+\mathbf A(t),\qquad
i\partial_t\rho_{\mathbf k}=[H(\mathbf k(t)),\rho_{\mathbf k}],\qquad
j_x=-\partial H/\partial k_x .
\]

\[
U_n=\exp[-iH(\mathbf k_0+A(t_{n+1/2}))\Delta t],\qquad
\rho_{n+1}=U_n\rho_nU_n^\dagger .
\]

\[
C^>_{AB}(t,t')=\sum_{\mathbf k}w_{\mathbf k}^2\operatorname{Tr}\bigl[A_{\mathbf k}(t)U_{\mathbf k}(t,t')(I-\rho_{\mathbf k}(t'))B_{\mathbf k}(t')\rho_{\mathbf k}(t')U_{\mathbf k}^\dagger(t,t')\bigr],\quad t\ge t'.
\]

\(w_{\mathbf k}=1/N_k\)。\(w_{\mathbf k}^2\) 是 B1-dev 已声明离散算符的代数结果，**不是**连续 BZ 归一化验收。

## backend 不得暴露的旋钮

`T2` / `dephasing` / `gauge` / `integrator` / `dipole_mode` / `lg_cov` / `dtype` / `hamiltonian`。这些只能来自本规范。

## 允许变化的

```text
same physics, same equations, same parameters, same integration rule, same dtype
different software backend
```

## 与材料线的隔离

材料生产 SBE（`lg_cov`、112/88 带、生产 \(T_2\)）不在本规范内。GPU backend 代码放在 `tools/q_backend/`，与 `tools/b0_small_model/`（物理 reference）和材料 CPU 代码分开。
