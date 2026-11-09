# 线 Q：Gamma12 runner r1 修复验收与服务器交接

2026-10-06。当前 runner 版本：2026-10-06.r1。
cri3 上没有 GPU、服务器仓库也还没有这份 runner。机器实测和第一步 CPU 短测的交接口径见 [线Q_cri3服务器执行交接_20261006.md](线Q_cri3服务器执行交接_20261006.md)。
本记录承接《线Q_Gamma12_runner交接终审与修正单_20261006.md》。该终审提出的三项问题已修复并通过针对性回归；旧 r0 产物仍保留。

## 已完成

1. 正确接回 U0
- prepare 复用冻结的 u0_from_bundle，保留原端口、phi、eta 和已乘一次的 sqrt(dnu)。
- 共享缓存新增 U0_canonical，避免与电子顶点 cache['u0'] 混淆。
- worker 的 u0 是对应列的 canonical 无泵输出行；undriven_raw_u 仍独立保存为电流积分。
- merge 采用 U=U0+delta_U、m=U@V.T，显式保存 U0@V.T。
- merge 会核对缓存中的 U0，不能用无泵电流积分冒充它。

2. 执行版本绑定
- 缓存生成历史与 execution_code_sha256 分开。
- worker/benchmark/validate 加载缓存前校验当前代码；merge 核对每块的执行代码身份。
- 旧 manifest 不会被静默当成 r1；需要新 prepare，或明确记录的兼容后处理。
- 路径键使用相对 POSIX 路径，便于服务器迁移；目标机 prepare 后绑定目标代码。

3. 时间窗绑定
- 每个块保存窗口、绝对起始时间、步长、步数、kernel/vertex 缓存身份。
- merge 不给 --window 时只能自动推断同一个窗口；不同窗口或时间参数必定拒绝。
- compare 也核对身份和列 ID，不再用 driven_raw_u 代替完整 U。

## 验证结果

主报告：
results/q_line/p1_h2_finite_kcv_gamma12_runner_u0_repair_20261006/validated_r1_final/repair_checks.json

| 项目 | 结果 |
|---|---|
| 16 步 fixture 的 prepare/分块/落盘/merge | PASS |
| 实际 CPU/CUDA worker 复数组对照 | PASS |
| GPU 整块与 CPU 分块后的完整 U/U0/delta_U/V | PASS |
| U0 与独立显式端口公式，256 列 | 最大差 6.94e-18 |
| U0 与独立显式端口公式，1024 列 | 最大差 3.93e-17 |
| 漏块/重复块/错误 manifest 拒绝 | PASS |
| 未指定 --window 时混窗拒绝 | PASS |
| 错误步数、执行代码、假 U0 拒绝 | PASS |
| 22 个原 continuous_1024 产物哈希 | 全部未改变 |
| AST 语法检查 | PASS |

runner SHA256：
224086831b4b79d94a8da9bd9b42b228fa293a1a1a991654210ceeef2c004998

第一轮回归脚本漏传 compare 的 --run-root，导致测试驱动退出；未影响算法。修正测试命令后，validated_r1_final 全部通过。validated_r1 是该未完成尝试，不作为验收入口。

## 存量修复，没有重跑 2176 步或全量传播

使用原连续 1024 列缓存和四份已验证 raw worker 产物：三块拼接和一份整块。
新目录：
results/q_line/p1_h2_finite_kcv_gamma12_runner_u0_repair_20261006/validated_r1_final/continuous_reprocessed/

- 只修正 U0 和由其生成的 U、m、U0 V^T；
- 所有原始传播数组、delta_U 和 V 不变；
- 三块合并与整块后处理仍通过：U/delta_U 最大差 6.42e-17，V 2.10e-17，U0 0；
- 元数据明确写 U0_only_postprocessing_no_propagation，不能把旧传播说成由 r1 重新运行；
- 这些只有 1024 个选定入射列和 2176 步，不是完整 Gamma12 的 n,m 或光子物理结论。

旧 r0 的 U0/U/m 接线通过标签撤销；其 raw 传播、CPU/GPU、连续性和性能证据继续保留。
修复前后 m 的差仅作实现诊断，不是增强、磁畴或纠缠信号。

## 当前进度不要倒退

- 两点有限 n,m 基准早已归档：tr n=0.01224071，适用已测输入频段和有限时长，不代表晶体。
- Gamma12 静态输入接入是有限试算资格。
- Gamma12 的共享缓存、分块传播及输出组装已有本机工程回归。
- 全列全脉冲 Gamma12 n,m、BZ 收敛、磁畴差、压缩、纠缠尚未完成。

因此，不再把“完整输出接口尚未实现”和“晶体级物理结果尚未得到”混为一件事；也不要求重新选择电子离散/连续路线。

## 给服务器执行 AI

可以交接 r1。服务器下一步是环境迁移和资源试算，不再重开物理审计。

先做：
1. 核对实际 CPU/GPU、显存、驱动、PyTorch、调度分区；不假定服务器有与本机相同的 GPU。
2. 在新目录重新 prepare 原 16 步 fixture，运行 worker/merge/compare，核对本机参考。不要覆盖 r0 或直接绕过执行哈希拒绝。
3. 用真实混频及暗列、小规模连续段测试合适的列块大小、显存和双路径时间；每列仍带全部 k。
4. 验证目标文件系统落盘/合并；按实际准备、循环、传输、写盘分项预算。
5. 条件满足且资源授权明确后，安排首条有限 Gamma12 +N 八周期加尾；它需自己的物理误差表。

本机旧 44.0212 s 的双路径连续短测仍是有效性能证据，但约 4 h 只是外推。不能把 wrapper 一次性开销随步数重复放大，也不能漏计全量缓存、装配和目标机器的实际开销。

服务器可复用：
- tools/b0_small_model/q8_m0_gamma12_runner_20261006.py
- tools/b0_small_model/q8_m0_gamma12_runner_repair_checks_20261006.py（复现实验需原存量目录；不是长作业入口）
- 原 16 步 backend_fixture.npz
- 本修复报告与 repair_checks.json

worker 必须使用与新 manifest 匹配的代码。若仅调日志/计时也改了文件哈希，先明确更新执行身份并做小对照，不能手改旧 hash 冒充原执行。

## 命令示例

在仓库根目录、目标机器自己的 Python 环境中运行；RUN 必须是新的目录：

python -B tools/b0_small_model/q8_m0_gamma12_runner_20261006.py prepare --run-root RUN --fixture results/q_line/p1_h2_finite_kcv_gamma12_backend_cuda_resume_20261006/backend_fixture.npz

python -B tools/b0_small_model/q8_m0_gamma12_runner_20261006.py worker --run-root RUN --manifest RUN/manifest.json --backend cuda --window start --block-id gpu --column-start 0 --column-stop 256

python -B tools/b0_small_model/q8_m0_gamma12_runner_20261006.py merge --run-root RUN --manifest RUN/manifest.json --window start --require-all --block RUN/blocks/block_gpu_00000000_00000256.json --output RUN/merged_gpu.npz

同目录存在重复对照块时，merge 必须显式 --block 选择一套无重叠结果；不能默认把 CPU/GPU 或分块/整块两份一起合并。

本轮未改变 Hamiltonian、初态、D/gamma、P1、规范、波包或探测定义；没有服务器提交。当前无需新增数学物理人审。
