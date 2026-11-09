# P0–P2 图数据映射（尚未制作正式图）

本表只定义后续 Origin 图的唯一数据入口。Python 未生成装饰性 PNG，也不以预览图替代验收。

## 图 1：模型与数值基准

- 模型参数：`configs/protocol_v1.json`；报告：`reports/P0_protocol.md`。
- 方法测试：`results/bias_v1/spectrum_method_validation.json`。
- 代表谱：`bias_p1_F0.000_I1_phi000_r02/spectrum_t100_eta0p1_dE0p025_mask2to4.csv` 与 F1 同名文件；横轴 `E_corr_eV`，纵轴 `S_per_eV`，标注 `eta_eV=0.1`、`E*=0.7170816 eV`。
- 收敛：`results/bias_v1/convergence.csv`；η=0.05 对照：`eta_control_t100_eta0p05_dE0p025_mask2to4.json`。

## 图 2：偏压绝对谱（先导）

- 四条件谱：`bias_p2_F{0,1}.000_I{1,4}_phi{000,090,180,270}_r02/spectrum_t100_eta0p1_dE0p025_mask2to4.csv`。
- 60 条谱的汇总：`results/bias_v1/spectra_origin.csv`；列为 `E_corr_eV,F_dc_V_per_nm,run_kind,I_apex_W_per_cm2,S_per_eV,eta_eV,K_infinity_eV,run_id`。
- 相位平均不能靠先画单相位再目测；用 `classical_response.csv` 的 `Y_high` 按 `CEP_rad` 等权。

## 图 3：q 与相位稳定性

- `P2_q_table.csv`：四相位 q；`P2_phase_table.csv`：每个相位的 Y_H 与四相位均值比。
- `P2_pilot.json`：`relative_change_two_to_four`、`eight_phase_representative.relative_change_four_to_eight`。
- 误差条不能用误差项平方和伪造；正文列出 dx/dt/η 单项与“未分辨”的判断。

## 图 4：强度贡献（P3 才可制作）

- P3 前不存在 `p_BSV(I)Y_H(I,F)`、累计贡献或 `D(F)`；不得用当前两个强度点插值生成。
- 当前可画的只有离散 `Y_H(I,F)`，必须标为 pilot、2 个强度、无统计加权。
