# 量子光 CrI3 文献整理与阶段汇报更新记录（2026-06-24）

## 本次处理目标

用户下载了一批支撑“量子光驱动双层 AFM-CrI3 HHG”的文献，希望：

1. 整理这些 PDF 与 Zotero/本地路径的对应关系。
2. 给后续调整 Quantum-light SBE 求解器的本地 AI 准备可引用的文献路径索引。
3. 将这些文献支撑补入阶段汇报 PPTX，并检查排版风险。

## Zotero 处理边界

本次没有直接修改 Zotero collection 或 zotero.sqlite。原因是当前可用 Zotero 接口只支持搜索和读取，不支持安全创建/移动 collection；直接改数据库容易破坏 Zotero 状态。

建议在 Zotero GUI 中建立：

- 陆老师/博士课题/量子光HHG-CrI3
- 子分类：SBE方法、磁群与非互易、量子光HHG、Gaussian-QFI-非经典性、开放系统-QRT-FCS

如果希望点击父分类“陆老师”时也能看到核心论文，可把核心论文同时拖入父分类和子分类。Zotero 的 collection 是归类引用，不会复制附件。

## 已生成/更新文件

| 文件 | 用途 |
|---|---|
| docs/literature_local_paths_for_SBE_AI_20260624.md | 文献本地路径索引；含 Downloads 路径、少数 Zotero storage 匹配路径、重复文件提醒、给代码 AI 的阅读优先级。 |
| docs/theory_literature_notes_for_report_20260624.md | 阶段汇报理论摘要；列出研究链条、PPTX 应新增的 5 页内容，以及不宜过度承诺的点。 |
| docs/stage_report_literature_supplement_20260624.pptx | 6 页可编辑文献增强补充页，可单独打开或复制到主汇报。 |
| docs/stage_report_literature_enhanced_20260624.pptx | 在“阶段汇报_可编辑修复版_20260621.pptx”后追加 6 页文献增强页的合并副本；原文件未覆盖。 |
| .tmp/literature_20260624/pdf_inventory.json | 下载 PDF 的页数、DOI/arXiv、标题候选等机器索引。 |
| .tmp/literature_20260624/download_zotero_matches.json | Downloads 与 Zotero storage 的自动匹配结果。 |

## PPTX 检查结果

- 原主汇报副本“阶段汇报_可编辑修复版_20260621.pptx”共 30 页，基本是可编辑文本/对象，不是纯图片版。
- 新合并版“stage_report_literature_enhanced_20260624.pptx”共 36 页。
- 第 31-36 页为新增文献增强页，PPTX XML 直接检查显示中文文本正常。
- artifact-tool inspect 在 Windows 控制台里显示中文会出现乱码，但直接读取 PPTX XML 确认文本正确；因此这是 inspect/控制台显示问题，不是 PPTX 内容损坏。
- 未覆盖原始主 PPTX。若后续发现合并版版式不满意，可直接回退使用“stage_report_literature_supplement_20260624.pptx”单独复制页面。

## 新增文献页内容结构

1. 研究定位：不是重复 CrI3 经典 HHG，而是进入输出光量子统计。
2. 研究链条：磁群对称性 -> 量子几何矩阵元 -> 双时电流关联(QRT) -> 玻色配对核(Bogoliubov) -> 辛谱/QFI -> 经典矩问题不可复现。
3. SBE 求解器应对齐的三类文献标准：Yue & Gaarde、Thümmler、Ilin/Stammer。
4. 从平均电流到双时连通电流关联：J_a(t) 与 C_ab(t,t')。
5. N 宇称分解：从平均电流推广到 C_ee, C_oo, C_eo, C_oe。
6. 量子优势判据：covariance、QFI、特征函数/矩判据。

## 后续建议

1. Zotero 建议手动整理 collection，不建议直接脚本修改数据库。
2. 让另一个本地 AI 修改代码前，优先阅读 literature_local_paths_for_SBE_AI_20260624.md 中的四篇：Thümmler 2025、Yue & Gaarde 2020、Yue & Gaarde 2022、Ilin 2026。
3. 下一步代码方向应聚焦双时连通电流关联 C_conn(t,t')，而不是继续只做 BSV 平均谱。
4. 汇报中应明确：CrI3 classical HHG 已有 Liu 2022 等工作；本课题的新意在量子光、N 宇称分辨输出统计和量子传感判据。
