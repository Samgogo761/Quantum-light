# 少带模型评分器（整理副本）

原位置：`tools/analysis/score_reduced_model.py`、`score_config.yml`。  
本目录为 **model_gates 统一入口下的整理副本**；配置路径未改，仍指向 `量子光研究/...`。

## 运行

```powershell
cd C:\Users\26507\Documents\New_SBEs\Quantum-light\tools\model_gates\score
python score_reduced_model.py --config score_config.yml
```

## 文档

- `docs/08_QE与Wannier/CrI3_模型评分与nv12验收_20260715.md`
- `docs/08_QE与Wannier/CrI3_M52_M64_M88验收与评分_20260716.md`

## 与 ε_P 的分工

评分器回答「少带相对 QE/full112 好不好」；  
`../verify_epsilon_P.py` 回答「±N 是否算符级 P 协变」。二者都要过，才能把磁序 + 模型选型写进论文口径。
