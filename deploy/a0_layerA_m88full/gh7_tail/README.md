# GH7 尾部模型门（只写、默认不提交）

分级执行，禁止自动串联：

1. 四个 k20 `+N` 宽轴反节点（本目录 sbatch）
2. 后处理 occupation / 迹 / 边界泄漏（`tools/analysis/gh7_tail_model_gate.py --stage k20`）
3. k20 PASS 后才允许 `APPROVE_GH7_TAIL_K40=1` 的代表点 k40
4. 尾部门全部通过后，再决定全量 GH7 或 \(P\) 共轭降本

提交包装器默认拒绝，除非：

```
GH7_TAIL_SUBMIT=I_UNDERSTAND_STAGED_K20_ONLY
```

节点 ID 由 `gh7_nodes.select_wide_axis_antinode_pair` 从冻结 manifest 选出，不硬编码。
使用完整 49 点 manifest 做 Q0/矩检验；`bsv_propagate_ids` 只跑选中节点，不重归一权重。
