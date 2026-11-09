# Agent 自查报告

日期：2026-11-09。本报告只回答事实，不把「文件已整理」包装成「科学问题已解决」。

## 1. 是否确认最新 HEAD 和工作区状态？

是。实测 HEAD `dadb5cc1c8abc38e6b21af211d0b419c197f318d`，分支
`claude/fix-question-submission-UaGkJ`；归档 `PASS: 1502 original hashes and all package links`。
`active_evidence.json` 有未提交的 `source_freeze` 更新；`configs/continuation_v2/` 有大量未跟踪的
运行配置。两者都不改变 QL-TIP-1D 的产额或误差。手册快照 `5f7acc99` / 1488 已过时，已在
`02_项目现状与证据摘要.md` 写清三次计数差异（1488 → 1489 → 1502）。

## 2. 是否确认每个文件真实存在？

是。`03_代码地图.md` 的 25 个代码/测试文件逐个检查，缺失 0 个；`07_关键结果与误差审计/`
的 27 个结果文件、`08_文献与来源/` 的 4 个记录、`09_环境与复现/` 的 3 个文件均实际存在。
唯一未放入包的是 96 个 NPZ，已在 `raw_state_manifest.csv` 列出。

## 3. 是否保留全部必要数据和历史判定？

是。`decision.md`、`decision_r2.md`、`decision_r3.md` 三版判定全部保留；`R3_statistics.json`、
`node_error_table.csv`、`error_headroom.json`、`acceptance_path_estimate.json`、
`error_component_map.csv`、S4/尾部/相位证据与账本均保留。蓝屏中断 attempt 保留在账本，
未删除、未算作成功 TDSE。

## 4. CSV 是否从真实数据导出？

是。`05_Origin绘图数据/_export_origin_csv.py` 只读 `results/bias_v2/readouts/*/summary.json`、
`runs/*/config.json`、`R3_statistics.json`、`classical_response.csv`、`intensity_contributions.csv`、
`phase_validation.json`、`R1_pair_audit_v5.csv`、`node_error_table.csv`。本次导出：
106 项**生产**共同读出（已排除 box900/dx048/dt005/caphalf/m3-5 等 26 项数值控制，
避免污染 CEP 平均）、22 个 CEP 节点、11 个偏压比、44 条贡献、18 条阈值记录、8 个代表性谱
（其中两个 32μ 谱按行拆为两个 <2 MiB 的分卷，未丢数据点）。

## 5. 是否有缺失的 NPZ、谱 CSV 或日志？

- NPZ：96 个、543,033,360 字节，**未入包、未上传 GitHub**；`raw_state_manifest.csv` 提供路径、字节、SHA256。
- 谱 CSV：共同协议 132 项读出都带 `spectrum.csv`，但包内只放 8 个代表性谱；其余在原仓库 `readouts/`。
- 日志：关键批次日志在原仓库 `logs/`，**未整体入包**；包内以账本与 summary 为准。

## 6. 是否有任何字段是推断或人工补写？

没有编造。以下字段为**明确留空或标注**：

- `dx_nm`：未从配置直接得到，全部留空；
- `box_nm`：由 `xmax_nm - xmin_nm` 计算，属确定性派生；
- DC 端点的 `I_apex_W_cm2`、`box_nm`、`dt_au`：DC run 不在 `runs/` 下，留空；
- `threshold_sensitivity.csv` 的 `error_status`：主阈值标 `gate_evaluated_main_only`，
  辅助阈值标 `not_separately_evaluated`，未用主阈值误差冒充；
- `cep_averaged_response.csv` 的 `direct_4to8_check`：只对 F1/4μ、F1/8μ、F1/32μ 填直接值，
  其余留空并写包络来源。

## 7. 是否运行了 TDSE？

否。本轮只做只读解析、哈希、CSV 导出与文件复制，未启动任何传播或谱读出。

## 8. 是否修改了原始物理结果或传播源码？

否。未改写 `results/`、`data/raw/`、`src/`、`scripts/` 中的任何原始文件；`06_关键代码/`
只是副本。新增内容全部位于 `handover/20261109_student/`。

## 9. 交接包总大小、文件数量、哈希校验是否通过？

见 `MANIFEST.csv` 与 `SHA256SUMS.txt`。生成后统计：文件 `82` 个，
打包前总字节 `9,205,911`；`core_handover.zip` 的实际字节见文件属性；SHA256 校验
`全部通过（生成后逐文件核对）`。

## 10. 师妹是否可以只凭交接包开始阅读和复核？

可以完成：阅读手册与现状、按 `03/04` 核对 run_id 与配置、用 `05` 的 CSV 画图、
用 `07` 的证据复核统计与误差。
**不能**只凭本包：重算能谱或改变读出参数（缺 96 个 NPZ）；启动新的 TDSE（需另行授权与预算）。
