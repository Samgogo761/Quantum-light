# 阶段汇报 PPT 更新记录（2026-06-25）

## 输出文件

- 新版 PPTX：`docs/stage_report_literature_enhanced_20260625.pptx`
- 参考旧版：`docs/stage_report_literature_enhanced_20260624.pptx`

## 本次处理

1. 保留旧版前 30 页主体内容。
2. 删除并替换旧版第 31-36 页的文献补充页，避免上一版补充页信息密度过高、版式不齐。
3. 新增第 31-38 页补充章节，补入 ZnO 与 hBN 验证工作，并重新整理文献定位。

## 新增补充页结构

| 页码 | 主题 | 作用 |
|---|---|---|
| 31 | 外部材料 benchmark 放在什么位置 | 解释 ZnO/hBN 不替代 CrI3，而是外部求解器参照 |
| 32 | ZnO QE→Wannier90 数据链 | 展示 ZnO 小体系第一性原理输入链路已跑通 |
| 33 | ZnO SBE health check | 展示 lg_cov 能读入外部 ZnO_tb.dat 并稳定传播 |
| 34 | hBN slab Wannier 为什么暂停 | 解释失败来自 slab 真空态 Wannier 建模，而非 SBE 传播 |
| 35 | hBN 解析二带模型 | 展示最小、可控、可复现的 SBE benchmark |
| 36 | hBN 文献参数 k 网格扫描 | 展示 Yue-Gaarde 类参数下的二带 hBN 谱结构 |
| 37 | 文献定位 | 将 Yue/Gaarde、Liu 2022、Ilin 2026、Ryl-Sperling-Vogel/Paulino 放入本课题逻辑 |
| 38 | 合并后的证据链与下一步 | 汇总 CrI3、ZnO、hBN、BSV/Cconn 的证据边界 |

## 使用的本地图像

- `examples/ZnO_wurtzite_qe_wannier/output/ZnO_qe_vs_wannier_bands_near_gap_pubstyle.png`
- `examples/ZnO_wurtzite_qe_wannier/local_sbe_runs/ZnO_lgcov_k10_k20_k40_HHG_health.png`
- `docs/figures/hBN_benchmark/hbn_model_k_scan_T2_5fs.svg`

## 关键表述边界

- ZnO 当前只作为小体系链路健康检查；由于 `Quantum-light` 仍是二维 `kx-ky` 求解器，ZnO bulk 结果应表述为 `kz=0` 切片，不作为 bulk ZnO HHG 定量复现。
- hBN 第一性原理 slab Wannier 路线暂停，是因为 20 Å 真空层导致低导带混入非局域真空态，Wannier spread 过大；这不是 SBE 求解器错误。
- hBN 解析二带模型是文献式 benchmark，用于验证 `lg_cov` 传播、电流分量输出、谐波谱处理和选择结构，不替代 CrI3 的第一性原理生产模型。
- CrI3 的主线仍是：经典光验证闭环 → BSV pilot → 双时电流关联 `Cconn(t,t')`、协方差矩阵、QFI 与非经典判据。

## QA 记录

- 新版 PPTX 总页数：38。
- 新增页 31-38 已导出预览图到 `.tmp/presentations/stage-report-fix-20260625/qa/`。
- 抽查页：31、32、33、34、35、36、37、38。
- 调整点：第一次生成时按 1920×1080 坐标排版导致右侧裁切；已改为 1280×720 坐标系并重排。
