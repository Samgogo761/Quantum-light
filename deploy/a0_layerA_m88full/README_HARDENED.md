# A0 M88-full：可复现探索与 k50 生产闸门

本文件补充原 `README.md`。旧作业 27674 已由旧脚本启动，不会被之后的本地修改追溯性改变。重新提交或后续作业应使用：

```text
sbatch_explore_k20_hardened.sh
```

并选择一个全新的 `OUTROOT`，避免把旧文件误认为新结果。

## 1. 新脚本解决什么问题

旧脚本以 `.compile_a0.done` 判断是否复用二进制；源码改变后该标记仍可能存在，从而误用旧程序。加固脚本改为对：

```text
Makefile + src/*.f90
```

求内容 SHA-256，并使用：

```text
.a0_build_cache/<source_sha256>/hhg_sbe
```

只有构建输入完全一致时才复用二进制。所有数组任务共用一个全局编译锁，避免并发执行 `make clean`。

每个输出目录还保存：

- `run_metadata.txt`：Git HEAD/dirty 状态、源码/二进制/节点/TB/input SHA-256、编译器与编译参数；
- `git_status_porcelain.txt`：运行时工作树状态；
- `build_metadata.snapshot.txt`：本次二进制构建记录；
- `nodes_manifest.input.dat`：实际节点清单快照；
- `input.nml` 与 `input_template.snapshot.nml`；
- `output_sha256.txt`：关键输出哈希；
- `run_status.txt` 和 `SUCCESS`：只有输出完整、节点矩 `pass = T`、无 NaN/Inf 且节点行数正确时才生成 PASS。

## 2. 重新提交探索作业

节点生成不变：

```bash
cd /public/home/wangjs/project/New_SBEs/Quantum-light
python3 deploy/a0_layerA_m88full/make_gh_nodes.py \
  --outdir deploy/a0_layerA_m88full/nodes_gh3 \
  --gh-order 3 --I-bar 1.0e11
```

必须使用新的输出根目录：

```bash
mkdir -p runs/a0_m88full_k20_hardened
cd runs/a0_m88full_k20_hardened
WORKDIR=$PWD \
OUTROOT=$PWD/output_a0_m88full_k20_hardened \
NODES_DIR=$PWD/../../deploy/a0_layerA_m88full/nodes_gh3 \
  sbatch ../../deploy/a0_layerA_m88full/sbatch_explore_k20_hardened.sh
```

脚本不会自动提交任何依赖作业。

## 3. GH3 探索后的严格顺序

GH3@k20 只用于探索，不应直接触发 k50 生产：

1. 确认 24/24 个目录均有 `run_status.txt: status=PASS` 与 `SUCCESS`；
2. 使用严格配对工具（约定名称 `tools/analysis/combine_pmN_nodes_strict.py`）核对 ±N 的 manifest SHA-256，以及每一行的 `id, weight, I, phi, order`；
3. 预先定义选态指标，再筛选候选 \((r,\theta_s)\)，避免看到结果后只挑最大的点；
4. 候选态先做 **GH5@k20**，比较 GH3→GH5 的 H2/H5/H7/H9/H10：
   - ICS；
   - CS；
   - 复振幅与相位；
   - 加权磁序奇电流
     \[
     \overline J_{o,n}=\frac12\sum_s w_s\bigl[J_{n,s}(+\mathbf N)-J_{n,s}(-\mathbf N)\bigr].
     \]
5. 只有求积结果通过预先声明的误差门槛后，才生成 M88-full@k50 生产脚本；±N 必须共享同一份 GH5 manifest；
6. 最终至少选一个代表态做 **k60 抽查**。已有收敛证据是 k40→k60；k50 是用户为增加网格密度和说服力选择的生产折中，不是已独立验收的新参考点；
7. 若结论依赖 H9/H10 或高场分布尾部，应选最大权重、主要贡献和代表性高场节点，对材料 parent/reference 再做抽查。

## 4. 禁止自动生产提交

`sbatch_prod_k50_hardened.sh.template` 故意立即退出。自动程序可以：

- 汇总结果；
- 执行严格验收；
- 生成候选生产脚本和下选报告。

但不得自动调用 `sbatch`。正式提交前必须人工确认：选态、GH5 节点文件及哈希、数组映射、TB 路径、`OUTROOT`、资源和 k60 抽查方案。

这套闸门并不否定 M88-full@k50：M88-full 是为了把 H10 纳入层 A 研究而作出的保守模型选择；加固只是在它成为生产结果前补齐求积收敛、可复现性和最终网格锚点。
