# 磁序 \(U_P\) 与 \(\varepsilon_P\) 闸门（2026-07-28）

> 回应场外 AI 四步方案；严格按 Obsidian `量子光理论核心` 与审校报告执行。  
> 实现与规格：`tools/model_gates/`

## 本地已完成 vs 本闸门补齐

| 场外步骤 | 历史状态 | 本闸门 |
|----------|----------|--------|
| 1 反演中心 / \(C2/m\) | 有近似 \(c\)；缺正式对称性页 | `structure_summary.md` + 人工补 spglib |
| 2 配对表 \(a\to a',\mathbf L_a\) | `construct_p_conjugate_tb` 已有置换+宇称 | `pairing_table.csv` 显式写出 \(\mathbf L_a\) |
| 3 \(\varepsilon_P(\mathbf k)\) | 仅有谱/偶极门 | **`verify_epsilon_P.py` 矩阵 Frobenius** |
| 4 ±N HHG + \(J_e/J_o\) | 已完成（M88-c20 等） | 本闸门 PASS 后可正式背书 |

## 怎么跑

```powershell
cd C:\Users\26507\Documents\New_SBEs\Quantum-light\tools\model_gates

# 自洽：H_{-N} 由公式生成（应接近机器精度）
python verify_epsilon_P.py --config config_epsilon_P.yml --mode pconj --quick

# 审计生产用 -N 文件
python verify_epsilon_P.py --config config_epsilon_P.yml --mode file
```

完整网格去掉 `--quick`（默认 12×12；可在 yml 改）。

## 评分器整理位置

原评分器已副本到：

`tools/model_gates/score/`

与磁序闸门并列，见该目录 README。

## 阶段纪律

仍属**经典光 L0 磁序验证**。不得将 \(\varepsilon_P\) PASS 表述为已进入量子光 / QRT。
