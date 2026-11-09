# 量子光 CrI3 理论主线与阶段汇报文献摘要（2026-06-24）

本文用于更新阶段汇报 PPTX/HTML 的理论部分，强调现有文献如何支撑“量子光驱动双层 AFM-CrI3 HHG”的主线。

## 一句话主线

双层 AFM-CrI3 的磁群对称性把 Néel 矢量的奇偶性写入非线性电流；量子光驱动时，输出谐波光不仅有平均谱，还包含由双时电流涨落决定的压缩、纠缠和 QFI，从而可能形成磁序的量子增强读出。

## 研究链条

```text
磁群对称性
  -> 量子几何矩阵元
  -> 双时电流关联(QRT)
  -> 玻色配对核(Bogoliubov)
  -> 辛谱/QFI
  -> 经典矩问题不可复现
```

## 汇报中建议新增或强化的内容

1. **不是重新发明 CrI3 HHG。** Liu 2022 等已有双层 CrI3 经典 HHG；本课题增量是量子光驱动、N 宇称分辨、输出光量子统计和传感判据。
2. **SBE 求解器可靠性要和文献标准对齐。** Yue & Gaarde 负责规范和基础流程，Thümmler 负责 Wannier gauge/dephasing 风险，当前 CrI3/hBN 验证负责数值自洽证据。
3. **Tier 2 不能只说 BSV 平均谱。** 要转向 `C_conn(t,t')`，因为 Ilin 2026、Stammer 2025 等近作说明输出光量子统计由 current-current fluctuation 决定。
4. **N 宇称四通道有群论基础。** Xiao 2023 给磁有序材料 SHG 分类，Gao 2023 给 PT-AFM 中 quantum metric/非互易的输运对偶。
5. **量子优势判据要严谨。** Gaussian covariance/QFI 使用 Weedbrook/Adesso/Even-Odd QFI；经典不可复现使用 Ryl-Sperling-Vogel 的 characteristic function/moment 判据，并与 González-Monge/Feist 的“可由经典统计平均复现”边界区分。

## PPTX 建议插入的 5 页内容

### Slide add-1：文献定位：我们站在 Liu 2022 之后

左：已有工作（CrI3 经典 HHG、堆叠/磁构型选择定则）。

右：本课题增量（量子光、N 宇称双时关联、QFI/压缩/纠缠）。

### Slide add-2：SBE 方法学依据

用 Yue & Gaarde + Thümmler 对齐：为什么选择 lg_cov、为什么 velocity 截断失效、退相干要谨慎处理。

### Slide add-3：从平均谱到双时关联

公式：`J_a(t)=Tr[P_a rho(t)]`；`C_ab(t,t')=<Delta J_a(t) Delta J_b(t')>`。

说明 BSV 平均谱只是第一层，量子统计需要 `C_conn`。

### Slide add-4：N 宇称四通道

定义 `J_e/o=(J(N)±J(-N))/2`，再推广到 `C_ee,C_oo,C_eo,C_oe`。

强调 Xiao 2023 的磁群 SHG 分类是理论底座。

### Slide add-5：量子优势判据

从 `C_conn` 到 covariance matrix `Sigma`，再到 squeezing/log-negativity/QFI；最后用 characteristic function/moment positivity 区分 classical mixture 与 genuine quantum output。

## 目前不建议在汇报里过度承诺的点

- 暂不要声称已经证明 BSV 一定给出量子优势；目前应说已经建立 classical benchmark 和代码验证，下一步计算双时关联与输出 covariance。
- 暂不要把 `nb1-104` 描述为所有高阶都完全收敛；应限定为 H1-H11/12 的经济配置，高阶 cutoff 区仍以 full112 或误差条为准。
- 暂不要把 hBN 模型验证说成第一性原理复现；更准确是解析二带模型/文献参数复现，用于 SBE 数值行为验证。
