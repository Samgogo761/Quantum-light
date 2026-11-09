# B1-dev 闭系统 Wick 接口（开发态）

状态：

```text
B1_DEV_INTERFACE = DEVELOPMENT_ONLY_PASS（解析 fixture 通过后）
M88_C20_MATERIAL_BACKEND = NOT_CONNECTED
QUANTUM_OUTPUT_G2 = NOT_COMPUTED
```

## 这一步完成了什么

`b1_dev_closed_wick.py` 把 B0 已验收的 Wick 式 (7.10) 扩成带权的多块接口。若

\[
\hat J_A(t)=\sum_k w_k\,\hat c_k^\dagger A_k(t)\hat c_k,
\]

且不同特征线块相互独立，则

\[
C^>_{AB}(t,t')=
\sum_k w_k^2\operatorname{Tr}
\left[
A_k(t)U_k(t,t')(I-\rho_k(t'))B_k(t')\rho_k(t')U_k^\dagger(t,t')
\right].
\]

这里的 \(w_k^2\) 是在**已声明的离散算符**
\(J=\sum_k w_kj_k\) 下的代数结果。它还不是 CrI₃ 连续 BZ/有限体积归一化的验收结论：
正则反对易关系的离散化可能把一个 k-cell 因子移入块关联。因此 material backend 还必须给出
`k_normalization_provenance`；当前输出固定标记
`absolute_k_normalization_validated=false`。

代码支持选定的若干 (t') 行，避免开发阶段默认生成完整 (N_t^2) 张量；同时硬核
幺正性、(0\le\rho\le I)、Hermitian 电流算符、权重归一和等时 Pauli 极限。

严格的 M88-c20 contract 固定为 78 个 active Bloch bands（58 占据 + 20 导带），并要求
算符、传播子与 k 归一化 provenance 非空。所有输出都带：

```text
label = B1_DEV_CLOSED_WICK_CANDIDATE
status = development_only / material_backend_not_validated
not_a_physics_result = true
not_quantum_output_g2 = true
```

## 这一步没有完成什么

生产 `lg_cov` 在离散 k 网格上通过协变有限差分耦合相邻 k 点；当前求解器还没有导出与
本接口等价的独立特征线单粒子传播子。因此接口**主动拒绝**把 `lg_cov_grid` 数据直接冒充
`independent_k_characteristics`。下一步必须先实现并验证下列二者之一：

1. 从同一闭系统方程导出特征线传播子 (U_k(t,t'))，并与无退相干的 1RDM 传播逐时对拍；
2. 建立保持 k 网格耦合的矩阵自由 Green-function/Wick 后端。

在这以前，尚未生成 M88-c20 材料双时关联。生产瞬时本征基 (T_2) 映射被硬拒绝；本模块
不使用 QRT，也不计算输出光子的 (g^{(2)}\)、(N_{nm}\)、(M_{nm}\) 或输入—物质交叉项。
H2 的 (J_o) 仍只是后续 B1-anchor 的首个目标通道，不是本 fixture 的结果。

## 测试

```bash
python tools/b0_small_model/test_b1_dev_closed_wick.py
```

测试覆盖：式 (7.10) 直接对拍、等时 Pauli 极限、BZ 权重平方、78/58 contract、拒绝生产
`T2`、拒绝 `lg_cov_grid` 假接入、拒绝非幺正传播子与非法费米 1RDM。
