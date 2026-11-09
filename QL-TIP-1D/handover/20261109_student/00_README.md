# QL-TIP-1D 师妹交接包（2026-11-09）

接收者：即将接手课题的研究生。
项目：`Quantum-light/QL-TIP-1D`，一维单活性电子、两偏压、BSV/热光/相干光统计比较。

## 一句话状态

响应库已完成（96 条 TDSE、176 项独立读出，全部 PASS），但最终统计判定为 **`INCOMPLETE`**：
`D = 0.859008`，审计区间 `[0.379399, 1.951856]`，`D.valid = false`。**主物理问题尚未回答**：
既不能认为统计依赖存在，也不能认为不存在。

## 建议阅读顺序

1. `01_研究交接与学习指导手册.md`：从物理问题、公式到数值方法的教材式材料。先读第 1–8 章，再读第 9–13 章。
2. `02_项目现状与证据摘要.md`：已经完成什么、什么未完成、版本差异与不能说的话。
3. `03_代码地图.md`：每个脚本负责什么、输入输出、是否为当前入口、运行风险。
4. `04_数据地图与复现说明.md`：run_id、readout_id、config/summary/state 的对应关系与复现命令。
5. `05_Origin绘图数据/`：直接可画的 CSV；先画强度响应、CEP 平均、偏压比与统计贡献。
6. `08_文献与来源/`：三篇核心文献的 DOI、借鉴点与未复现内容。
7. `09_环境与复现/`：依赖、环境快照与只读复现命令。

第一阶段任务是**理解与独立复核**，不是追加 TDSE。

## 包结构

```text
20261109_student/
├── 00_README.md
├── 01_研究交接与学习指导手册.md
├── 02_项目现状与证据摘要.md
├── 03_代码地图.md
├── 04_数据地图与复现说明.md
├── 05_Origin绘图数据/
│   ├── intensity_response_by_bias.csv
│   ├── cep_averaged_response.csv
│   ├── bias_ratio_by_intensity.csv
│   ├── statistics_contribution_BSV.csv
│   ├── statistics_contribution_thermal.csv
│   ├── threshold_sensitivity.csv
│   └── representative_spectra/（8 个共同协议谱 + _index.csv）
├── 06_关键代码/（22 个脚本的只读副本 + 路径映射）
├── 07_关键结果与误差审计/（27 个结果文件 + raw_state_manifest.csv）
├── 08_文献与来源/
├── 09_环境与复现/
├── MANIFEST.csv
├── SHA256SUMS.txt
├── 10_Agent自查报告.md
└── core_handover.zip
```

## 版本与校验

- 当前 HEAD：`dadb5cc1c8abc38e6b21af211d0b419c197f318d`
- 归档校验：`PASS: 1502 original hashes and all package links`
- 手册快照 HEAD `5f7acc99`（PASS 1488）与中间提交 `a0e00403`（PASS 1489）已过时，**以本包为准**。
- 最新提交 `dadb5cc1` 只含线Q项目结果，不含任何 `QL-TIP-1D/` 文件，不改变本项目科学数据。
- `active_evidence.json` 有未提交的 `source_freeze` 更新；不改变产额或误差。

## 数据缺失（必须知道）

- 96 个 `state_100fs.npz`（共 543,033,360 字节）**未放入本包，也未上传 GitHub**。
- 校验清单：`07_关键结果与误差审计/raw_state_manifest.csv`（run_id、路径、字节、SHA256）。
- 缺 NPZ 仍可：读 summary、重画曲线、复核统计与误差；**不能**重算谱或改读出参数。
- 取回方式：从原工作区 `QL-TIP-1D/data/raw/bias_v2/` 复制，或向负责人索取移动硬盘副本。

## 不能说的话

- 不能说 BSV 已表现出可靠的偏压响应优势；
- 不能说量子效应已证明、热光与 BSV 差异已收敛；
- 不能说高能电子已被认证为再散射电子；
- 不能说当前模型已准确预测真实钨针尖的绝对电子数；
- 不能说项目已达到可发表状态。

## 下一步

先按手册第 34 节完成三阶段只读工作（熟悉与核对 → 响应曲线与贡献分解 → 统计积分方法审查），
再与负责人讨论是否追加计算。不要自行启动大规模节点加密或更换模型参数。
