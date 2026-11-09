# 线 Q：4-band 零响应 harness 资格关闭（2026-09-07）

```text
FOUR_BAND_HARNESS_STATUS = CLOSED
MODEL_ROLE               = zero_response_numerical_harness
INITIAL_STATE            = two_lowest_bands_per_k_slater
H2_LABEL                 = finite_pulse_background
JO_EXPECTATION           = STRUCTURAL_ZERO  (PASS, not FAIL)
EIGHT_BAND               = HOLD_FOR_HUMAN
QUANTUM_LIGHT            = not_started
G2                       = not_computed
```

不是 CrI\(_3\)。不是论文物理模型。不改总账。本文件关闭 4-band 资格考试；**不**批准 8-band Hamiltonian。

---

## 0. 对外部回复的审校（先裁定，再执行）

外部回复把工程验收与物理模型批准分开，这一条收下。

收下并已执行：

- 当前 4-band 是 layer \(\times\) spin 的零响应 harness，不是论文模型。
- \(V=\tau_x s_z\) 给出同-\(k\) 域映射，故实验室 \(J_o\simeq 0\) 是结构性结果；改 \(\mu\)/强度/CEP 救不活。
- \(P\neq V\neq W\)。\(W=s_z\) 是单畴 \(\mathbf k\leftrightarrow-\mathbf k\) 的内部幺正，不是晶体反演 \(P=\tau_x\)。
- 初态正式名 `two_lowest_bands_per_k_slater`，不再叫全局半填充绝缘基态。
- 旧积分器 `sampled_A_arithmetic_mean` 冻结为 legacy；harness 用真正的 \(A((t_n+t_{n+1})/2)\)。
- \(J_o\simeq 0\) 为 PASS。H2 只做 H1/H2/H3 背景核验。Wick 只核等时。然后停止。
- 8-band 暂不编码；它只是第一物理候选，不是已定主模型。

精确修正（外部方向对，措辞收紧）：

1. **不要写 \(H2=\) numerical error。** 有限 \(\sin^2\) 包络不严格满足 \(A(t+T/2)=-A(t)\)，\(2\omega\) 处可以混合有限脉冲的真实 transient 谱权重、开关、窗、离散与 \(k\) 网格。标签是 `finite_pulse_background`。本仓库用代数 \(W\) 核验了 \(WH_N(\mathbf k)W^\dagger=H_N(-\mathbf k)\) 与 \(Wj_x(\mathbf k)W^\dagger=-j_x(-\mathbf k)\)；**没有**单独证明无限周期半周期电流定理。更长脉冲上 H2/H1 下降，与“非本征背景”一致，仍不是磁性偶次 HHG。
2. **A1/B2 不再问用户。** 这是 harness 的命名与数值积分器，不改变“我们在研究什么物理”。8-band 的 Hilbert 空间、\(H_R\)、\(T_\perp\)、\(P\) 表示仍必须人审。
3. Zhou 等是层状 Kane–Mele 的 BPVE/SHG，不是强场量子 HHG 平台。把 PT-AFM \(\times\) 强场 HHG \(\times\) 量子光接起来是本线要自己做的新问题。

未做（硬禁止）：改 \(H_0\)、8-band 代码、H5–H11、量子光、GPU kernel、\(g^{(2)}\)、修 \(C_{oo}\) 当物理通道、改总账。

---

## 1. 冻结的 4-band 事实（资格考试对象，不再调参）

基底 \((1\uparrow,1\downarrow,2\uparrow,2\downarrow)\)。

\[
H_0=\varepsilon\tau_0+t_\perp\tau_x+\lambda\tau_z(\sin k_y s_x-\sin k_x s_y)+\Delta_N\tau_z s_z,
\]

默认 \(t=1,t_\perp=0.3,\lambda=0.4,\Delta_N=\pm0.5\)。Houston：\(\mathbf k=\mathbf k_0+\mathbf A(t)\)，\(j_x=-\partial_{k_x}H\)。

\[
j_x=-2t\sin k_x\,I_4+\lambda\tau_z\cos k_x\,s_y.
\]

初态：每个 \(\mathbf k\) 占 \(H_0\) 最低两本征态。脉冲：\(A=(e_0/\omega)\sin^2(\pi t/T_p)\cos\omega t\)。\(e_0\) 不是 \(\max|E|\)。

---

## 2. 六个 Gate（`test_q_harness_gates.py`，全部 PASS）

开发回归：b0 9/9、b1-dev 6/6、`test_pt_afm_toy`、`test_pt_afm_two_time`、`test_q_frozen_spec`、`test_q_backend` 同步 PASS。

默认探针：`nk1d=4`，`ncyc=4`，`nt_per_cycle=24`，`e0=0.08`，`omega=0.20`，harness 规则 `evaluate_A_at_t_mid`。

| Gate | 检查 | 本机数字 |
|---|---|---|
| 1 矩阵/对称 | \(P,T,PT\)；\(V=\tau_x s_z\)；\(W=s_z\)；三者互不相等 | \(V\) 残差 \(0\)；\(W\) 残差 \(\sim10^{-16}\)；\([V,W]=0\) |
| 2 电流 | 解析 \(j_x\) 对拍 \(-\partial_{k_x}H\) 有限差分；与 \(\Delta_N\) 无关 | 差分 \(<10^{-7}\)；解析一致到 \(10^{-15}\) |
| 3 传播 | \(U^\dagger U=I\)，\(\rho^\dagger=\rho\)，\(\mathrm{Tr}\rho\) 守恒，\(0\le\lambda_i(\rho)\le 1\)；`nt_per_cycle` = 24/48/96 时域收敛 | 幺正 \(<10^{-12}\)；细网格相对误差 \(<5\%\) |
| 4 域零 | legacy 与 harness 两条规则 \(\max_t\|J_+-J_-\|\) | \(1.9\times10^{-15}\) / \(2.4\times10^{-15}\)（PASS） |
| 5 有限脉冲谱 | 只报 H1/H2/H3；标 `finite_pulse_background` | Hanning：H2/H1 \(\approx 4.05\times10^{-3}\)；`ncyc=8` 降到 \(\approx 1.8\times10^{-5}\)；H2 对窗敏感；\(J_o\) 的 H1–H3 在地板 |
| 6 Wick | 等时 \(C_{JJ}=\mathrm{Tr}[j(I-\rho)j\rho]\) 与 `wick.equal_time_wick` 及两时 \(t=t'\) | 残差 \(<10^{-12}\) |

\(\max|E|\approx 0.0773\neq e_0=0.08\)，两套场标度继续分开记录。

---

## 3. 关闭之后 4-band 不再做什么

不继续 H5–H11。不接量子光。不修 \(C_{oo}\) 当磁性通道。不上 GPU。不为做出 H2 改 \(H_0\)。

`run_toy` 默认仍走 legacy \(A\) 规则，保证旧回归字节级路径；新资格考试走 harness 中点。`c_oo_*` 仍是遗留全 \(C_{JJ}\) 字段。

---

## 4. 停在这里：下一轮只讨论 8-band 是什么模型

8-band bilayer honeycomb / Kane–Mele 是**目前最值得认真审核的第一物理候选**，不是已定主模型。增加带数本身不保证消灭新的 \(V\)/\(W\)，也不保证出现磁性 H2。

下列问题会改变“我们究竟在研究什么物理”，需要本人判断后再编码：

1. Hilbert 空间是否就是层 \(\times\) A/B \(\times\) 自旋？
2. Rashba \(H_R\) 的空间反演性质、是否依赖层/子晶格？
3. \(T_\perp\) 对应哪种堆垛（AA / AB / rhombohedral）？
4. 晶体 \(P\) 在该基底上的表示（\(\tau_x\)、\(\tau_x\sigma_x\)、或带 \(k\) 相位）？
5. 真正要研究的谐波与关联对象（选择定则、inter/intraband、\(C_{eo}\) 等）——这是 8-band 经典验收之后的事，不在 4-band 上预演。

在把实空间模型推到 Bloch \(H(\mathbf k)\)、并把每一个对称性（含隐藏幺等）写清楚之前，不写 8-band 代码。
