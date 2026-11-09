# 新增选题文献核验（2026-11-04）

继承 `literature_audit.md`，本次只核对《QL-TIP-1D_选题最终方案与Agent变更指令_20261103.md》新增的五篇来源，以及 Meier 等 2023 年少电子相关论文。核对目标是书目真实性与研究范围，不是完整论文复现或穷尽查新。本轮没有运行科研计算，也没有下载／分析新 Source Data。

六条来源均在出版社或 arXiv 官方页面获得对应记录，题目与 DOI／arXiv 编号相符。以下日期按实际 online publication 区分于卷期日期；不能将“检索到相邻研究”升级为“本题已有／没有完全相同工作”的确定判断。

| 核实的书目与主要作者 | 日期与状态 | 原始来源实际研究了什么 | 对选题的意义与核验边界 |
|---|---|---|---|
| **Attosecond physics in optical near fields**；Jonas Heimerl、Stefan Meier、Anne Herzig 等，末位 Peter Hommelhoff；Nature Physics **21**, 1893–1898。DOI `10.1038/s41567-025-03093-3` | 2025-11-12 在线；2025 年 12 月卷期；正式论文 | 强驱动钨针尖中直接电子的 CEP 分辨低能条纹（NILES）。光学近场梯度使有质动力加速具有亚周期发射时刻依赖，并可用于能量筛选孤立阿秒电子发射段。不是 BSV 偏压扫描研究。[官方全文：Abstract、NILES、Methods](https://www.nature.com/articles/s41567-025-03093-3) | “近场影响电子谱”已有具体先例，但不能据此推断所有近场长度 × BSV 问题均被覆盖。当前主线避免将一般近场谱变形单独当作创新。本轮核对了正文相关段与元数据，未重算 NILES。 |
| **Nonlinear atomic tunnelling boosted by bright squeezed vacuum**；Zhejun Jiang、Shengzhe Pan、Jianqi Chen 等，末位 Jian Wu；Nature **654**, 356–360。DOI `10.1038/s41586-026-10485-9` | 2026-05-20 在线；2026-06-11 卷期；正式论文 | 孤立钠原子的 BSV 隧穿电离；角条纹测量比较峰值光电子动量，并在固定平均脉冲能量下调节光的二阶关联。论文所谓超过 20 倍提升指匹配峰值动量所需脉冲能量之比，不是总电离产额提升 20 倍。[官方原文：Abstract、Quantum-boosted tunnelling、Quantum tunability](https://www.nature.com/articles/s41586-026-10485-9) | 任务书关于固定均值调节统计响应的概括成立；实验涉及多模 BSV，不能直接等同本项目单模明亮极限。作者的“quantum boost”用语不为本项目正权重 TDSE 提供不可经典模拟性证明。直开网页遇到认证重定向，但官方索引正文可检索；未下载 PDF／SI。 |
| **Energy- and Momentum-Resolved Single-Shot Yield Fluctuations in Bright-Squeezed-Vacuum-Driven Atomic Strong-Field Ionization**；Haodong Liu、Xiaoxiao Long、Peizeng Li、Yunquan Liu；Physical Review Letters **137**, 153201。DOI `10.1103/pgkx-8t5c` | **2026-10-07 在线**；卷期日期 2026-10-09；正式论文 | 氙原子的逐发 BSV–光电子符合测量，同时记录单脉冲光强与能量／动量分辨响应。摘要报告：强度条件化谱恢复类似相干光的截止标度；高能、高动量电子更偏重稀有高强度 BSV 脉冲。[APS 官方摘要与日期](https://journals.aps.org/prl/abstract/10.1103/pgkx-8t5c) | “高能电子选择稀有强脉冲”已是直接先例，不能单独成为本项目新结论。潜在增量应是经验证的**偏压依赖差别**。本轮只验证官方摘要、作者与书目；全文显示需订阅，未完整阅读全文或 SI。 |
| **Control of tunneling ionization and subcycle interference dynamics with squeezed coherent light**；Peizeng Li、Xiaoxiao Long、Zijian Lyu、Haodong Liu、Yankun Dou、Ziheng Qin、Peipei Ge、Yunquan Liu；Physical Review Research **8**, 023047。DOI `10.1103/pbc2-3r6m` | 2026-04-15；正式开放获取论文 | 氢原子在正交偏振双频场中的理论研究：800 nm 相干分量与 1600 nm 压缩相干分量；使用量子光学扩展的 quantum-trajectory Monte Carlo，研究电子相位、动量、产额、数分布及亚周期干涉。[APS 官方摘要与元数据](https://journals.aps.org/prresearch/abstract/10.1103/pbc2-3r6m) | 应准确称为“双频原子压缩相干驱动的相邻研究”，不能概括为已完成本项目单色针尖 DSV 控制。支持首稿不扩展到 DSV／两色问题。本轮核验摘要与元数据，未完整核验理论推导／代码。 |
| **Photoelectrons resolve the phase-dependent variance of bright squeezed vacuum**；Haoran Ma、Yihe Wang、Chuan Cheng、Ian Gabalski、Aaron M. Ghrist、Eleanor Weckwerth、Mohammed Th. Hassan、Philip H. Bucksbaum；arXiv **2610.31811v1** | 2026-10-22 18:47:06 UTC 提交；本次页面只有 v1，**预印本**，未发现页面所列期刊接收信息 | 相同颜色的相干光作为相位参考；氙强场电离产额读出沿参考场方向的 BSV 正交分量，噪声在不同动量／电离通道中分布不同。[arXiv 官方摘要与提交历史](https://arxiv.org/abs/2610.31811) | “相位敏感 BSV 光电子读出已有预印本”得到支持；它不是金属针尖偏压实验。不得将预印本写成已发表同行评审结果。本轮核验摘要与元数据，未完整审阅正文。 |
| **Few-electron correlations after ultrafast photoemission from nanometric needle tips**；Stefan Meier、Jonas Heimerl、Peter Hommelhoff；Nature Physics **19**, 1402–1409。DOI `10.1038/s41567-023-02059-7` | 2023-06-22 在线；2023 年 10 月卷期；正式论文 | 激光针尖发射后，动态库仑排斥产生少电子能量反关联；能量筛选得到亚泊松计数。涉及直接／再散射电子及多电子传播，不是 BSV 单电子统计加权。[Nature 官方摘要与图示说明](https://www.nature.com/articles/s41567-023-02059-7) | 不能从本项目单电子发射概率唯一推出电子 Fano 因子或 g²；需要另加发射事件、传播、探测计数模型。当前首稿不报告这些量合理。本轮核验官方摘要、图示说明和元数据，订阅正文未完整阅读。 |

## 本轮结论与未成立结论

新增书目没有发现题目／编号不匹配；应修正 PRL 在线日期为 10 月 7 日，并保留 arXiv 的预印本状态。新增文献使“BSV 增强”“高能尾来自稀有强事件”“近场改变谱”“相位读出”这些泛泛表述更不足以构成独立增量。

将唯一主变量收窄为提取场强 `F_dc`，在相同平均 apex 强度、相同电子 Hamiltonian、同一明确 CEP 协议下比较 coherent 与明亮 BSV 的高能窗响应，是值得先做基准验证的候选问题。**本次检索不证明该组合具有首创性，也不证明 BSV 必然减弱偏压淬灭。** 未检查全文／SI 的项已逐条标出；投稿前仍需全文、引用链和更广关键词查新。

新方案用“高能贡献”代替预设“BSV 保留再散射平台”更稳妥。特别是 PRL 已把高能电子与稀有强脉冲联系起来，因此本项目必须报告绝对 `Y_H`、总产额、高能占比及偏压归一化曲线，证明实际差异超过数值／积分误差；仅给出大比值或重新演示尾部加权不够。
