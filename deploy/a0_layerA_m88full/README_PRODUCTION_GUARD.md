# A0 production-guard v2

`sbatch_explore_k20_production_guard.sh` 是新的非覆盖式 successor。旧的探索脚本、加固 v1 和已经运行的作业 27674 均保持不变。

## 相比 v1 新增的硬门

### 1. 完整构建缓存键

缓存目录为：

```text
.a0_build_cache_v2/<build_key>/hhg_sbe
```

其中 `build_key` 同时包含：

- `Makefile + src/*.f90` 的内容 SHA-256；
- `FC` 及其解析后的路径；
- 编译器版本字符串；
- `FFLAGS`；
- `LDFLAGS`。

因此即使源码相同，只要编译器或编译参数变化，也不会复用旧二进制。

### 2. Python3 节点预检

传播开始前，脚本会调用快照后的：

```text
tools/analysis/validate_qlight_nodes_production.py
```

检查节点格式、有限性、重复 ID、权重、强度、Gaussian-Q 矩和驱动映射。它允许旧 GH3 manifest 中 `6.28318531` 这种由文本舍入产生的 \(2\pi\) 边界值，并按相位模 \(2\pi\) 处理。

任何失败都会在调用 SBE 前非零退出。

### 3. Python3 独立运行后验收

传播完成后，脚本调用快照后的：

```text
tools/analysis/validate_a0_run_strict.py
```

独立验证：

- manifest 每个节点与每个目标谐波恰好出现一次；
- 输出的 `weight, I, phi` 与归档 manifest 一致；
- 从逐节点复电流重建的 ICS/CS 与 `HHG_ics_cs.dat` 一致；
- `ICS >= CS`；
- 节点矩报告通过。

只有 Python 前后双硬门、Bash 输出完整性、NaN/Inf 和行数检查全部通过，才生成 `SUCCESS` 与 `status=PASS`。

## 可复现性输出

每个任务保存：

- Git HEAD 与 dirty 状态；
- source digest、完整 build key 与 binary SHA-256；
- FC 路径、compiler version、FFLAGS/LDFLAGS；
- Python 版本及三个 validator 的快照和 bundle SHA-256；
- nodes、TB、input、template SHA-256；
- 输入、manifest、模板、构建元数据和验证 JSON/日志快照；
- 关键结果的 SHA-256。

## 使用方式

必须选择一个新的输出根目录：

```bash
cd /public/home/wangjs/project/New_SBEs/Quantum-light
mkdir -p runs/a0_m88full_k20_production_guard
cd runs/a0_m88full_k20_production_guard
WORKDIR=$PWD \
OUTROOT=$PWD/output_a0_m88full_k20_production_guard \
NODES_DIR=$PWD/../../deploy/a0_layerA_m88full/nodes_gh3 \
  sbatch ../../deploy/a0_layerA_m88full/sbatch_explore_k20_production_guard.sh
```

该脚本只执行当前数组任务，不会提交 GH5、k50 或 k60 后续作业。探索完成后的顺序仍为：

```text
严格 ±N 配对
→ 候选态 GH5@k20
→ 求积收敛
→ 人工确认 M88-full@k50 生产脚本
→ 代表态 k60 抽查
```

正式生产不得由后处理程序自动调用 `sbatch`。
