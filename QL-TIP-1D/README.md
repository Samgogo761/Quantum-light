# QL-TIP-1D

量子光统计调控金属纳米针尖电子发射的一维理论研究。

**当前决定（2026-11-04）：保持一维单电子TDSE加明亮BSV的Husimi强度权重，后续主变量改为静态提取场F_dc。** 目标是验证输入统计是否改变高能电子的偏压响应，完成规模可控的补充研究；没有已成立的BSV新机制或发表保证。

本项目独立于Q-MATH八带磁性研究，共用当前唯一Git仓库和本地归档工具。原始资料不覆盖，无新仓库、无远程推送。

## 当前入口

|目的|文件|
|---|---|
|最新问题、最小计算量与师妹分工|[2026-11-04选题决定](reports/DIRECTION_DECISION_20261104.md)|
|发给另一AI的完整提示词|[独立审查提示词](docs/prompt_for_independent_ai_20261104.md)|
|原三篇论文审计|[literature_audit](docs/literature_audit.md)|
|新增六篇文献的真实性与范围|[literature_update](docs/literature_update_20261104.md)|
|已实现零偏压Hamiltonian及观测量|[model_conventions](docs/model_conventions.md)|
|数值算法与数据格式|[numerical_methods](docs/numerical_methods.md)|
|环境检查|[Stage0](reports/STAGE_0.md)、[机器记录](docs/environment.json)|
|定态、无场和自由波包|[Stage1报告](reports/STAGE_1.md)、[实际结果](results/stage1_20261103_r2/summary.json)|
|800 nm单强度收敛与试探点|[Stage2报告](reports/STAGE_2.md)、[复核结果](results/stage2_audit_20261104/summary.json)|
|独立数值审查|[审查记录](docs/independent_review.md)、[SI单位独立实现](validation/finite_well_si.py)|
|完整文件清单和去重映射|[file_inventory](file_inventory.json)、[CSV清单](data/manifest.csv)|
|外部原始资料与哈希|[references](references/README.md)|

## 已验证与尚未验证

Gate A通过：连续方阱基态−5.5087344341 eV、只有一个束缚态；20 fs无场稳定，FFT/CN自由波包通过解析对照。六项物理测试通过。

旧协议Gate B仅在800 nm、15 fs、I_inc=10¹¹ W/cm²、CEP0、零偏压基准点通过：右侧净转移概率2.73368×10⁻⁸，已测最大单项控制变化0.553%。另外两个强度点保留为未完成加密的试探值，不拟合严格多光子幂律。

尚无1600 nm偏压谱、BSV／热光加权、量子优势或真实W材料预测。ACS补充材料尚未取得；新版DC空间剖面、初态准备、能谱和相位协议需要落实。已有零偏压规范作为v0历史保留，新偏压协议不能悄悄替换旧定义。

## 本地复现

环境：`.venv/Scripts/python.exe`（Python3.12），依赖固定在 [requirements-lock.txt](requirements-lock.txt)。在新的Python3.12环境安装此清单即可复现；当前环境不需要重装。读取PDF/Word的inspect_inputs.py使用Codex已带pypdf/python-docx的运行时，属于资料工具，非科研生产依赖。

```powershell
./QL-TIP-1D/.venv/Scripts/python.exe -m pytest QL-TIP-1D/tests -q
./QL-TIP-1D/.venv/Scripts/python.exe QL-TIP-1D/scripts/run_stage1.py --run-id stage1_my_recheck
./QL-TIP-1D/.venv/Scripts/python.exe QL-TIP-1D/scripts/run_stage2.py --config QL-TIP-1D/configs/stage2_baseline.json --run-id stage2_my_recheck
```

这些是显式复算命令，文档或归档脚本不会自动执行它们。已有run_id拒绝覆盖。每次输出包含配置、实际源文件SHA-256、运行前Git提交和原始CSV/NPZ；需精确重放早期源码时按source_resolution使用冻结副本。

## 保存与收尾

`data/origin/`是Origin可导入数值；`data/raw/`保存完整复数态和坐标，较大数组分片，每文件小于2 MiB。R1的序列化失败与原始代码保留；与成功R2相同字节的副本由file_inventory映射至唯一存储位置。环境、版权全文提取、下载与预览不入Git。

本轮代码/结果冻结后，用明确路径登记到Q-MATH清单，随后manage.py finish创建本地提交。后续修改保留旧代码字节与结果，使用新版本／run_id，不能刷新旧哈希掩盖改动。用户其他研究任务的未提交改动不纳入本项目。
