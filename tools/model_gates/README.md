# 模型与磁序闸门（model_gates）

> 整理日期：2026-07-28  
> 逻辑权威：Obsidian `量子光理论核心`（\(U_P\) / \(\varepsilon_P\)）+ `课题整体进展报告_理论与文献审校_20260727`  
> 阶段定位：**经典光 L0 — 磁序验证**；尚未进入 QRT / \(C_{\rm conn}\)。

本目录把两类**可审计闸门**收在一处：

| 子目录/文件 | 用途 |
|-------------|------|
| [`SPEC_epsilon_P.md`](SPEC_epsilon_P.md) | 场外 AI 四步 → 可执行规格（严格按笔记） |
| [`verify_epsilon_P.py`](verify_epsilon_P.py) | 显式构造 \(U_P(\mathbf k)\)，全网格算 \(\varepsilon_P\) |
| [`config_epsilon_P.yml`](config_epsilon_P.yml) | 路径与通过阈值 |
| [`score/`](score/) | **少带模型评分器**（原 `tools/analysis/score_*` 的整理副本） |
| [`examples/`](examples/) | 输出示例占位 |

相关但未迁入（避免破坏旧路径）：

- `tools/analysis/construct_p_conjugate_tb.py` — 由 +N 构造严格 P 共轭 −N TB  
- `tools/analysis/compare_tb_eigenvalues.py` — 谱门 \(E_{-\mathbf N}(\mathbf k)=E_{+\mathbf N}(-\mathbf k)\)  
- `tools/analysis/afm_neel_symmetry_check.py` — HHG 的 \(J_e/J_o\)  

文档入口：`docs/01_AFM选择定则_主线/磁序U_P与epsilon_P闸门_20260728.md`

---

## 推荐执行顺序（经典磁序收口）

```text
1) 结构对称性记录（C2/m / 反演中心）     → 写入 SPEC §A 结果表
2) construct_p_conjugate_tb.py          → 得到 -N TB（若尚无）
3) verify_epsilon_P.py                  → 显式 U_P + ε_P 全网格
4) 既有 ±N HHG / Je/Jo                  → 声明建立在已审计磁畴协变上
5) score/score_reduced_model.py         → 少带静态评分（与磁序正交，可并行）
```

**禁止**：从能谱“搜索”任意幺正矩阵当作 \(U_P\)。  
**禁止**：把 \(\varepsilon_P\) 通过误写成已进入量子光。
