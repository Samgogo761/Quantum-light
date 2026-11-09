# PT-AFM 4-band toy（线 Q，第一阶段）

```text
label = PT_AFM_TOY_CANDIDATE
status = development_only / not_cri3
not_a_physics_result = true
not_quantum_output_g2 = true
H2_DYNAMIC_SYMMETRY_ALLOWED
```

本文件写死 \(H_0\)、\(P/T/PT\)、\(U_P\)、Houston 规范与电流。
**不是** CrI\(_3\)。禁止把数字写进总账当材料结果。

## 1. 基底与默认参数

单胞 4 轨道：层 \(\tau\) ⊗ 自旋 \(s\)，顺序

\[
|1\uparrow\rangle,\;|1\downarrow\rangle,\;|2\uparrow\rangle,\;|2\downarrow\rangle.
\]

\(\tau_i=\sigma_i\otimes 1_s\)，\(s_i=1_\tau\otimes\sigma_i\)。

| 符号 | 默认 | 含义 |
|---|---:|---|
| \(t\) | 1 | 层内近邻，能量单位 |
| \(t_\perp\) | 0.30 | 层间 |
| \(\lambda\) | 0.40 | 层反号 Rashba |
| \(\Delta_N\) | \(\pm 0.50\) | Néel；只取这两个给定值 |
| \(n_{\mathrm{occ}}\) | 2 | 每个 \(k\) 占据最低两本征态（`two_lowest_bands_per_k_slater`），**不是**全 BZ 共同 \(\mu\) 半填充 |
| \(E\parallel x\) | — | 第一阶段不扫偏振 |

本文件中的 4-band 角色（2026-09-07）：`zero_response_numerical_harness`。不再作为论文物理模型调参。

\[
\varepsilon(\mathbf k)=-2t(\cos k_x+\cos k_y).
\]

## 2. 写死的 \(H_0\)

\[
H_0(\mathbf k)
=\varepsilon(\mathbf k)\,\tau_0
+t_\perp\tau_x
+\lambda\tau_z\bigl(\sin k_y\,s_x-\sin k_x\,s_y\bigr)
+\Delta_N\tau_z s_z.
\]

第一阶段不得改这个形式，只许改上表数值。

\[
P=\tau_x,\qquad T=is_y K,\qquad U_P=\tau_x.
\]

- \(\Delta_N=0\)：\(P\) 与 \(T\) 分别成立。
- \(\Delta_N\neq 0\)：\(P,T\) 分别破缺，\(PT\) 保持。

代码核验（含 \(\mathbf k\to-\mathbf k\)）：

\[
P H(+\Delta_N,\mathbf k)P^{-1}=H(-\Delta_N,-\mathbf k),
\qquad
T H(+\Delta_N,\mathbf k)T^{-1}=H(-\Delta_N,-\mathbf k),
\qquad
(PT)H(+\Delta_N,\mathbf k)(PT)^{-1}=H(+\Delta_N,\mathbf k).
\]

## 3. 规范与电流（只许 Houston）

\[
\mathbf E(t)=E_x(t)\hat x,
\qquad
\mathbf k(t)=\mathbf k_0+\mathbf A(t),
\qquad
E_x=-\partial_t A_x,
\qquad
j_x=-\partial H/\partial k_x
\]

（玩具单位 \(e=\hbar=1\)）。禁止另接一套 \(er\cdot E\) 而不改 \(j\)。

\(\pm N\) 感受**同一实验室** \(\mathbf A(t)\)。公共基底只对轨道/层做 \(U_P\)，并要求配对 \(\mathbf k\leftrightarrow-\mathbf k\)：

\[
\widetilde J_x(-N;\mathbf k)
=U_P^\dagger J_x\bigl(-N;-\mathbf k\bigr)U_P.
\]

未做该映射就对 \(\pm N\) 矩阵作差 → `ValueError`。

\[
J_e=\frac{J(+N)+\widetilde J(-N)}{2},\qquad
J_o=\frac{J(+N)-\widetilde J(-N)}{2}.
\]

\(C_{ee},C_{eo},C_{oe},C_{oo}\) 全部用这组算符 + `wick.equal_time_wick` / `two_time_wick_greater`。

## 4. \(C_{oo}\) 宇称（禁止写错）

\(C_{oo}=\langle\Delta J_o\Delta J_o\rangle\) 是磁性通道的涨落强度，但是 **N-偶**。
\(C_{oo}(+N)\approx C_{oo}(-N)\) **不是失败**。

随磁畴反号的是

\[
C_{\mathrm{odd}}^{(N)}=\frac{C(+N)-C(-N)}{2}=C_{eo}+C_{oe}.
\]

禁止把 \((C(+N)-C(-N))/2\) 叫做 \(C_{oo}\)。

## 5. 为何偶次 \(J_o\) 不被偶然对称性禁掉

`H2_DYNAMIC_SYMMETRY_ALLOWED`

静电极限下，空间反演把偶次谐波从反演偶的电流里禁掉。本玩具在 \(\Delta_N\neq 0\) 时 **\(P\) 已破缺**，所以磁性（N-奇）偶次通道 \(J_o(2\omega)\) 原则上允许。还要排除三个会把 H2 误杀的偶然对称：

1. **\(\lambda=0\)**：\(s_z\) 守恒，\(j_x\propto\tau_0\) 与 \(\Delta_N\) 脱钩，\(J_o\equiv 0\)。默认 \(\lambda=0.4\neq 0\)，层反号 Rashba 把 Néel 写进横向电流。
2. **竖直镜面 \(M_x\)**（\(k_x\to-k_x\) 且不转自旋）：Rashba 项 \(-\sin k_x s_y\) 变号，\(H_0\) 不是 \(M_x\) 偶的。因此不存在「\(E_x\) + 镜面」型动力学偶次禁戒。
3. **单独的 \(C_{4z}\)**：\(E\parallel x\) 已经破缺 \(C_4\)；\(C_{2z}\) 把 \(E\to-E\)，等价于线偏振的半周期平移，**不能**额外杀掉 N-奇偶次（与双层 AFM 中 \(J_o\) 承载偶次谐波的材料结论同方向，但本玩具仍只是 candidate）。

因此：\(\Delta_N\neq 0\)、\(\lambda\neq 0\)、\(E\parallel x\) 时，动态对称性 **允许** \(J_o(2\omega_0)\neq 0\)，没有镜面/ \(C_4\) 型禁戒。

**第一阶段数值却看到 \(J_o\equiv 0\)。** 不是 \(U_P\) 漏做，也不是 \(k\) 网格不成对。对角化给出偶然对称：

\[
\mathrm{spec}\,H_0(+\Delta_N,\mathbf k)=\mathrm{spec}\,H_0(-\Delta_N,\mathbf k)
\quad\text{对每个 }\mathbf k.
\]

Houston 的 \(j_x=-\partial H/\partial k_x\) **不含** \(\Delta_N\)（Néel 项与 \(k\) 无关）。于是每个 \(\mathbf k\) 上 \(\langle j_x\rangle_{+N}=\langle j_x\rangle_{-N}\)，BZ 求和后 Néel-奇电流低于 \(10^{-14}\)。偶次漏在 \(J_e\) 里。这正是备忘说的「先查偶然对称性，不要先改 \(H_0\) 花样」。

**`Jo_H2 != 0` 硬测试因此不启用。** 第一阶段要保住的是宇称课：等时 \(C_{JJ}(+N)\approx C_{JJ}(-N)\)（N-偶），以及未映射差分被拒绝。打破等谱要改 \(H_0\) 形式，留给以后。

### 5.1 旧期望的状态（2026-09-03）

```text
SUPERSEDED_PHYSICS_EXPECTATION:
  备忘 20260831 §7 第 7 条「C_eo+C_oe 非无理地为零（N-奇门）」
  备忘 20260831 §9 汇合条件「J_o(2ω0) 允许且非零」
CURRENT_4BAND_JO_STATUS = STRUCTURAL_ZERO_CANDIDATE
```

历史文字不删。新的验收原则：CPU reference 给出 \(J_o\approx 0\)，任何 backend 也必须给出 \(J_o\approx 0\)；禁止为了做出 H2 改 \(H_0\)。

### 5.2 等谱的代数来源（事实，不作物理解释）

\(k\) 无关的幺正 \(V=\tau_x s_z\)（\(V^2=I\)）满足

\[
V\,H_0(+\Delta_N,\mathbf k)\,V^\dagger=H_0(-\Delta_N,\mathbf k)\quad\forall\mathbf k,\qquad
V\,j_x\,V^\dagger=j_x .
\]

代码核验残差为 0。同一实验室 \(\mathbf A(t)\) 下 \(\rho_-(\mathbf k,t)=V\rho_+(\mathbf k,t)V^\dagger\)（残差 \(10^{-14}\)）。因此凡由 \(j_x\) 构成的可观测量（平均电流、\(C^>_{JJ}(t,t')\)）在 \(\pm N\) 下逐点相等，与 \(k\) 网格、\(U_P\) 映射无关。**这不是数值误差。**

算符级 \(\hat J_o\) 取决于共同基底映射 \(\mathcal U_N\)（理论 (10.4)）的选择：

| 映射 | \(\max\|\hat j_o\|_F\) | \(\tilde\rho_-\) vs \(\rho_+\) | 备注 |
|---|---:|---:|---|
| \(U_P=\tau_x\)，\(\mathbf k\to-\mathbf k\) | 4.0 | 差 0.21 | \(A=0\) 时 \(\hat j_o=j_x\)；与 \(\Delta_N\) 无关，不是 Néel-奇 |
| \(V=\tau_x s_z\)，同 \(\mathbf k\) | 0 | 差 \(10^{-14}\) | \(\hat J_o\equiv 0\)，\(C_{oo}=C_{eo}=C_{oe}\equiv 0\) |

`jo_case_report` 给 `CASE_C_map_dependent`。物理 \(\mathcal U_N\) 仍待审；**当前 4-band 不再为解开 \(J_o\) 而改 \(H_0\)。**

另有单畴隐藏映射 \(W=s_z\)（不是 \(P\)，不是 \(V\)）：

\[
W H_N(\mathbf k)W^\dagger=H_N(-\mathbf k),\qquad W j_x(\mathbf k)W^\dagger=-j_x(-\mathbf k).
\]

\[
P\neq V\neq W.
\]

理想无限周期且 \(A(t+T/2)=-A(t)\) 时，\(W\) 倾向禁掉单畴偶次。实际 \(\sin^2\) 包络不严格满足半周期平移，\(J_e\) 的 \(2\omega\) 成分标为 `finite_pulse_background`，**不要**写成 numerical error 或 magnetic H2。

`H2_DYNAMIC_SYMMETRY_ALLOWED` 对**当前这个** \(H_0\) 是 SUPERSEDED（\(V\) 与 \(W\) 已说明）；保留字符串只为历史测试 `h2_symmetry_documented()`。

## 6. 明确拒绝

生产 \(T_2\)、开系统、QRT 当主结果、输出 \(g^{(2)}\)、Husimi-\(Q\)、`lg_cov` 网格、把玩具拟合到 CrI\(_3\)。
`absolute_k_normalization_validated=false` 直到有限体积约定验收。

## 7. 两时 \(C^>_{JJ}(t,t')\)（2026-09-03 起）

`two_time_cjj` 通过 `b1_dev_closed_wick.closed_wick_reference` 调 `wick.two_time_wick_greater`（式 7.10），对全部 \(t'\) 给出下三角 \(C^>(t,t')\)。门禁见 `test_pt_afm_two_time.py`：等时极限 \(<10^{-12}\)；\(C^>_{AB}(t,t')=[C^>_{BA}(t',t)]^*\)（\(A\ne B\)，\(t<t'\) 用同一式的 Heisenberg 形式）；\(0\le\rho\le I\)；幺正；(10.6)/(10.7) 在各自态中为恒等式。冻结规范见 `Q_THEORY_FROZEN_SPEC.md`。

## 8. 4-band harness 关闭（2026-09-07）

`test_q_harness_gates.py`：P/T/PT/\(V\)/\(W\)；\(j_x=-2t\sin k_x I+\lambda\tau_z\cos k_x s_y\) 有限差分；legacy 与 `A(t_{n+1/2})` 传播守恒；\(J_o\simeq0\) 为 PASS；H1/H2/H3 有限脉冲背景；等时 Wick。此后 4-band 停止：无 H5–H11、无量子光、无 GPU、不修 \(C_{oo}\) 当物理通道。8-band 仍 HOLD。
