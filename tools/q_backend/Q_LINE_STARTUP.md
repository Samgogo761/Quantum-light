# 线 Q 工程启动检查器（2026-09-07）

`q_line_bootstrap.py` 负责登记环境、源码版本和既有开发回归状态；研究路线、物理选择及其批准状态由独立研究文档管理。启动器使用 Python 标准库；回归沿用已有 NumPy 依赖。

所有报告固定记录 `physics_review_status = pending_user_review`。即使全部开发回归通过，也只表示当前既有程序通过既有检查，不批准 Hamiltonian、规范、SBE、退相干、关联定义或会改变物理结论的设计，不表示量子输出、`g^(2)`、压缩、纠缠或 GPU 已完成。4-band harness 关闭见 `docs/04_BSV量子光/线Q_4band_harness资格关闭_20260907.md`；那是零响应软件资格，不是论文模型批准。

## 使用

在仓库中用可用的 Python 运行；本机已提供的 Python 可写为：

```powershell
$qPython = 'C:\Users\26507\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $qPython -B tools/q_backend/q_line_bootstrap.py --help
& $qPython -B tools/q_backend/q_line_bootstrap.py
& $qPython -B tools/q_backend/q_line_bootstrap.py --run-existing-regressions
```

默认 inventory 只查 NumPy 模块是否可发现、安装元信息中的版本，以及列定源码/规范是否可读和 SHA256；不导入 NumPy 或物理模块、不执行测试，每项标为 `not_run`。NumPy 的安装元信息不等于运行时成功导入证明，实际导入随显式回归验证。

`--run-existing-regressions` 依次运行以下已有脚本，每个使用 `[sys.executable, "-B", script_path]`、仓库工作目录和独立子进程：

| 既有脚本 | 工程检查范围 |
|---|---|
| `tools/b0_small_model/test_b0_gates.py` | 已有 Fock / Wick / 显式 GKSL-QRT 小系统参考检查 |
| `tools/b0_small_model/test_b1_dev_closed_wick.py` | 已有闭系统 Wick 开发接口契约；仅内置合成数据 |
| `tools/b0_small_model/test_pt_afm_toy.py` | 已有 4-band 候选模型的对称性与动力学检查 |
| `tools/b0_small_model/test_pt_afm_two_time.py` | 已有两时闭系统 Wick 检查 |
| `tools/b0_small_model/test_q_frozen_spec.py` | 已有规范、代码和结果元数据的一致性 |
| `tools/b0_small_model/test_q_harness_gates.py` | 4-band 零响应 harness 六门；\(J_o\simeq 0\) 为 PASS |
| `tools/q_backend/test_q_backend.py` | CPU 包装一致性和 GPU 占位实现拒绝运行 |

运行这些既有开发回归只是保存旧基础的可复现证据，不代表已经进入新的量子输出研究。没有额外运行 open-Gaussian 测试、参数扫描、性能 profile、安装依赖或联网。未给启动器开放任何物理参数。

## 结果与退出码

默认目录为 `results/q_line/startup_20260907`。若已存在，自动选含 UTC 时间戳及随机后缀的新目录。可通过 `--output-dir results/q_line/<新的目录名>` 指定；必须为仓库 `results/q_line` 内的新子目录，拒绝已存在目录、目录穿越及指向仓库外的链接。

`startup_report.json` 保存 Python/NumPy 信息、执行前后相关源码和规范的 SHA256、逐项状态/耗时、工程状态和待人工物理审核标记。每个实际运行的脚本另存独立 `.stdout.log` 与 `.stderr.log`，不把结果写入既有物理代码或规范。若检测到运行中源码变化，工程状态为 `source_changed_during_run`。

`--timeout-seconds` 只调整单项工程等待上限，默认 300 秒。失败或超时会记录并继续其余回归；用户中断会停止后续回归并保留已取得的报告。尚未运行的条目不会记成通过。

退出码：`0` 表示 inventory 完整或列出的既有回归全部通过，`1` 表示环境/源码不完整或回归失败，`2` 表示 CLI 参数或输出路径错误，`130` 表示用户中断。下游必须同时检查 `mode`、`engineering_status` 和 `test_summary`，不能将 inventory 的退出码 `0` 解读为测试通过。
