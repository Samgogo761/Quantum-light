# 线 Q：Gamma12 runner 交接终审与修正单

2026-10-06。只读审查代码和存量产物，未改模型、未重跑传播、未提交服务器。本文件优先于“runner 全部验收、本机无需再做任何测试”的概括。

## 结论

后端传播、连续小列集和分块传播的一致性进展可以保留。新 runner 的完整输出行尚不能验收：它把无泵电流源积分误当成 canonical 的 U0。须先修输出组装，再把修正版交给服务器做环境和资源短测。这是实现错误，不是新的物理选择。

可以同时把本修正单交给服务器执行 AI；它可以核对环境，但不得按当前错误 U0 组装去跑全列长任务。

## 1. P1：U0 定义接错，当前 U、m 和 U0 V^T 不是冻结接口的对象

文件 tools/b0_small_model/q8_m0_gamma12_runner_20261006.py：
- 第 1005–1015 行：driven_u/undriven_u 取自 accumulate 返回的电流源积分，随后 u0=undriven_u。
- 第 1317–1323 行：U=driven_raw_u，m=U @ V.T。
- 第 459、713、1085、1270、1351 行附近，manifest、元数据与验收重复使用同一错误定义。

原冻结接口 tools/b0_small_model/q8_m0_p1_h2_finite_kcv_input_close_20261005.py 的 u0_from_bundle 使用：
- stationary 光子列；
- L/R 端口的直接传播项；
- phi、sqrt(nu)/eta 和已计入一次的 sqrt(dnu)。
它给出的是完整无泵出射行，不是有限时间电流积分。

正确组装仍是：
delta_U = source_u_driven - source_u_undriven
V = source_v_driven - source_v_undriven
U = U0_canonical + delta_U
m = U @ V.T

无泵 source integral 应另存为 source_u_undriven，不能占用 U0 字段。

独立只读重构：取当前 continuous_1024 的同一 A/free_A0、列序、极化、原 H2 phi 和已加权状态，经既有 u0_from_bundle 重建：
- U0 正确部分列的 Frobenius 范数：0.6838146541；
- runner 误标 U0 的范数：0.0511149254；
- 两者最大元素差：0.2880208562；
- 相对 Frobenius 差：1.0025077890；
- 替换到同一 delta_U/V 后，部分列 m 收缩的最大变化：0.0035119661。

以上仅证明接线错误；部分列不是完整输入空间，不能由这些数给光子数、CCR 单位阵或磁畴结论。

direct_patch_vs_merged 的 1e-17 一致性能够复现，但双方用了同一错误定义，所以不能验收 U0 接入。原 CPU/GPU 状态、source_u/source_v、delta_U/V、分块和计时证据并不因此作废。

## 2. P2：代码版本没有绑定到实际执行块

prepare 记录 code_sha256；_read_manifest/_load_manifest_cache 只验证 manifest 自身和缓存文件，worker 不核对当前执行代码，也没有将实际执行代码哈希写入块的身份。

本次确有差异：
manifest 中 runner = bb602088d65e87033f88906721075b10f8e7b4defe2bb380ba3540c4deeaa96c
当前 runner = 2dbe0e7ff8b58bfe58c779d4bae885fa48e69ccd0282f79bcad5453e3981bbf3
其他五个记录文件一致。

已对照的计时补丁不能据此被指为数值错误；缺口是后续不同实现可以持同一 manifest 静默混入一个结果。

修复：分开缓存生成身份与执行代码身份，worker 记录实际 runner/后端/收缩依赖哈希，merge 拒绝未显式接受的混用。缓存生成代码变动与 worker 计时补丁不必一律强制重建电子背景；允许明确、可追溯的兼容迁移，但不能无记录忽略差异。

## 3. P2：不指定 --window 时可以混合不同时间窗

merge 的 --window 默认 None；_validate_block_metadata 仅在非 None 时核对窗口。一个含 start/pump 的 fixture manifest 下，来自两个窗口的不同列可能通过列覆盖检查。

修复：无论用户是否传 --window，都要求所有块的 window、t0、h、nstep 和核/顶点身份一致，并与 manifest 的该窗口一致。显式指定窗口则再核对指定值。增加一个混窗拒绝测试即可。

continuous_1024 本次只有一窗，没有证据表明当前产物发生过混窗；这是服务器分块的防混用缺口。

## 4. 计时和研究状态怎样表述

本次复算确认：
- 两条传播 wall 合计 44.0212196 s；
- worker wrapper 46.6147625 s；
- 1024 列的 patched worker 与 merged 输出行差和原文相符；
- runner AST 语法解析通过；
- 产物 SHA256 与该 worker 元数据一致。

3.95–4.20 h 仍只能叫本机线性参考。wrapper 整体乘步数会放大每块一次的加载/落盘；完整 cache 准备、真实全量装配和目标服务器成本尚需分项计入。无需为这一点否定已测速度，也不能叫可靠配额。

后半段“尚无有资格的有限参考 n,m”“只做了几个频点和八周期输出行”等说法混用了早期阶段。应改为：
1. 两点有限 n,m 基准已在已测频率和有限时间范围内归档：ntpc=1024、tr n=0.01224071，有步长/输入加密/端点及延后时间检查；不是晶体结果。
2. Gamma12 静态输入接入已有有限试算资格；其全输入、全脉冲量子输出尚未运行。
3. runner 连续小列集已算，但新组装的 U0/U/m 有本单所列实现错误，须修；无需重新选择电子离散/连续路线。
4. BZ 收敛、磁畴差、压缩和纠缠仍未获得结论。

参照已存在文件：
docs/04_BSV量子光/线Q_有限nm资格收尾复核_20261005.md
docs/04_BSV量子光/线Q_有限nm尾检查与BZ试算成本_一页_20261005.md
后一个文件中的半移 8 试算建议后来已被 Gamma12 替代；这里只引用其中两点基准和尾检查部分，不能恢复旧网格建议。

## 5. 交给执行 AI 的最小修复任务

A. 接回真正 U0
- 直接复用现有 u0_from_bundle 及其端口、phi、eta、dnu 约定。
- 将其按同一全局列 ID 缓存；保留原 source_u_undriven，但不叫它 U0。
- worker/merge/compare/fixture 验收统一改为 U=U0_canonical+delta_U。
- 对现存 1024 列产物先作新目录后处理，重用已有 delta_U/V，不需要重跑电子传播或 CUDA 长段。
- 独立与 u0_from_bundle 比较 U0，不能只检验 U=U0+delta_U 这个自身定义。部分列不要求完整 CCR=I。
- 报真正 U0 V^T 与 delta_U V^T，旧 U/m 审计标签撤销，保留旧文件。

B. 修执行身份与混窗保护
- 记录、核验实际执行版本；同一输出禁止未声明的版本混用。
- 不提供 --window 时也拒绝混窗和不一致时间参数。
- 做各一个针对性拒绝测试，及一次小 fixture 的落盘合并回归；不重开已过物理审计。

C. 改交接状态
- 将“U0 保留链 PASS”改为修复后有独立比较证据才 PASS。
- 将“本机没有必须继续的测试”改为“只需以上接线/防混用修复及小回归”。
- 删除后半段退回早期未算有限 n,m 的描述。
- 新文档和 JSON 给出旧结果、修正版及兼容关系，全部旧结果保留。

完成 A/B/C 后即可冻结本机工程验收，服务器只做目标环境、短样本迁移回归和资源/落盘/调度验证，再在明确资源授权下安排首条有限 Gamma12 +N 长试算。无需再对 P1=0 理论、初态或通道选择作一轮人审。

本单没有替用户批准长作业，也没有要求新物理决定。
