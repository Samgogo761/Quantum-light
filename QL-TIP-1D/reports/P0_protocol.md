# P0：恢复、协议冻结与代码骨架

日期：2026-11-04。状态：**PASS（仅表示 P0 恢复与协议落实，不表示 P1 能谱或物理结果通过）**。

## 继承与完整性

- 继承计划提交 `6f8798e`，执行任务书 `plans/20261104_solo_v1/EXECUTION_TASKBOOK.md`，首轮范围仅 P0–P2。
- `python tools/qmath/manage.py check`：**PASS，461 项原件哈希及所有包链接**。
- 当前 QL-TIP-1D/Q-MATH 没有本轮开始前的未提交项目改动；根工作区有大量其他研究改动和未跟踪文件，本轮不修改、不清理、不纳入提交。
- 未发现运行中的 `python.exe`；新 `run_id` 均拒绝覆盖。
- 未连接服务器、未上传、未提交 Slurm 作业、未推送远程。

## 基线恢复

使用 `QL-TIP-1D/.venv/Scripts/python.exe` 复跑旧六项测试，结果 **6 passed in 0.76 s**。加入新谱测试后的完整测试为 **11 passed**；P0 日志保存在 `logs/P0_tests_20261104.log`。

旧 `src/ql_tip_1d/core.py` 保持原字节。新增：

- `src/ql_tip_1d/bias_v1.py`：DC 势、场方向、smootherstep、CAP、右侧 mask 和协议工具。
- `src/ql_tip_1d/spectrum_v1.py`：静态 `H_dc` 两次三对角求解、独立小矩阵/自由高斯参考与阈值积分。
- `scripts/run_bias_v1.py`：单条准备或观察运行、概率账本、复数态、谱、日志和源码哈希。
- `scripts/run_solo_v1.py`：显式阶段清单、`--dry-run`、复用和 P2 阻断。
- `tests/test_spectrum_v1.py`：五项独立谱/方向测试。

## 冻结的物理协议

完整机器可读字段见 `configs/protocol_v1.json`。关键值：1600 nm、强度包络 FWHM 25 fs、FE=3（`I_apex=9I_inc`）、`ell=L_dc=20 nm`；平均 apex 强度起值 `mu=1.5e12 W/cm²`，对应 `I_inc=1.6666666666666666e11 W/cm²`。

DC 采用 `E_dc=-F/(1+x/L_dc)^2`、`U_dc=-F L_dc x/(x+L_dc)`（x≥0），金属侧屏蔽。F>0 时电场为负、电子受力向 +x。光场与 DC 势能都在表面连续，相互作用符号为 `+e∫E dx`。

DC 默认从 −160 至 −120 fs 以 `10s³−15s⁴+6s⁵` 渐启，等待至 −100 fs；激光窗 −100 至 +100 fs；保存 +100/+125 fs。只在无场初态归一化一次，准备后不补回 CAP 损失。

谱唯一主方法为右侧 mask 后的静态 `H_dc` 能量窗口滤波；不使用自由 FFT 平移。基准 `eta=0.2 eV`、`dE=0.05 eV`，对照 `eta=0.1 eV`、`dE=0.025 eV`，另有 mask 2–4/3–5 nm 和 100/125 fs 对照。

单位核算：`Up(mu)=0.35854080000000005 eV`，冻结主阈值 **`E_star=2Up(mu)=0.7170816000000001 eV`**。辅助窗为 `E_star±0.2 eV`。该阈值是固定高能窗，不声称普适直接电子截止或再散射认证。

## 数值起点、预算与停止边界

基准网格 `well_cells=24`、`dt=0.1 au`、盒 `[-12,600] nm`、CAP 左 `[-12,-8]`/右 `[500,600] nm`、强度 1 au、三次方。对照 48 单元、0.05 au、右盒/CAP 外移和 CAP 强度减半。生产默认四相位；P1/P2 先导可先用两相位，再补齐四相位，并按要求检查八相位。

总上限 240 条新科研 TDSE、累计 72 h；P1/P2 累计 8 h 未通过则停止扩展。所有传播、谱后处理和测试墙钟累加；单条 90 min、单 worker、峰值工作集 6 GiB。首轮不做 BSV/热光积分、不铺四偏压库、不增加模型变量。

## 文献定义与来源限制

沿用既有 `docs/literature_audit.md` 和 `DIRECTION_DECISION_20261104.md`：有限方阱参数有 ACS 正文依据；近场代表值与 TDSE/Husimi 组合见 Heimerl 2025；`L_dc=20 nm` 是本项目明确有效模型选择，不冒称文献 SI 已证实。ACS SI 此前未成功取得，本轮没有重新无限尝试下载，也不假定原作者 CAP/网格实现。静态能量窗法是本项目待 P1 验证的实现选择。

## 下一阶段许可边界

独立谱测试通过只允许进入 DC 准备和实际相干点验收，不等于 P1 PASS。必须保存并比较 DC-only、准备损失、CAP、谱积分、高能窗及数值对照。P2 由 `results/bias_v1/P1_gate.json` 阻断；该文件不是 PASS 时不得运行 P2。
