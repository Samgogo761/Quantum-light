# 线 Q × Q-MATH 来源谱系（R0，只读）

**日期：2026-10-23。本文件只记录哈希、解压位置与缺件。它不是传播批准书，也不构成 16.2 复现声明。**

上一轮因额度用尽中断。本轮补写本谱系与 R1 主报告后停机。未解压覆盖仓库源码；未执行 R2；未新增电子—光子传播。

R1 外审订正与 R1.5 纯数学文档已于 2026-10-23 续写。`PHYSICS_CONTRACT.md` 已补 R1.5 数学-only 允许行，并禁止把热态残差或有限维优化写成无限维正 \(P\) 界。仍未解压覆盖 `tools/`；仍未执行 R2；仍无新传播。独立 16.2 ZIP 本机未找到。

## 0. 取证原则

| 规则 | 执行 |
| --- | --- |
| 16.11 包只解到独立目录 | `docs/04_BSV量子光/qmath_pack_16_11_20261023/` |
| 永不覆盖仓库 `tools/` 源码 | 已遵守 |
| Q-MATH-16.2 源码不在本包 | `16.2_in_pack=False`；`inner_08.zip` 是 16.9 档案 |
| 正 P | 非负 Glauber–Sudarshan \(P\)，不是 stochastic positive-P SDE |
| 输入态 | 位移压缩 \(D(0.5)S(0.3e^{i\theta})|0\rangle\)，不是纯 BSV |
| 外审 ZIP 仓库无副本 | Downloads 原包存在；仓库不复制 zip |
| 未复跑 | 未把 16.10/16.11 验证脚本当作本轮新结论依据 |

指导书原文：`C:\Users\26507\Downloads\数学成果收割与本地研究指导书.md`。
外审交接包：`C:\Users\26507\Downloads\Q-MATH_完整研究接续包_截至16_11.zip`。
定理原文：`docs/04_BSV量子光/qmath_pack_16_11_20261023\item_03.md`。
物理契约：`docs/04_BSV量子光/PHYSICS_CONTRACT.md`。

## 1. 外审 ZIP 与解压目录

| 对象 | SHA256 | 备注 |
| --- | --- | --- |
| Downloads ZIP | `1575d35daec2dc3ecdca0f69fd22d5ee4e0c84fdaebd9a29ca0661ac4e780154` | 存在 |
| 仓库内 ZIP 副本 | — | **不存在**，不得假装仓库自带该包 |
| 解压根目录 | — | `docs/04_BSV量子光/qmath_pack_16_11_20261023/` |

解压产物：`item_00..07`、`original_names.txt`、`zip_manifest.json`、`inner_16_10/`、`inner_16_9/`（由 `inner_08.zip` 再解出 `i9_*`）、`_preview/`。

## 2. 16.11 包顶层条目

| alias | 原名 | bytes | SHA256 |
| --- | --- | --- | --- |
| `item_00.docx` | `QMATH_研究总档案_完整恢复至16_10.docx` | 46889 | `7735ce61bddaf608e16afbabf93ad805a2492901567129b6e13a2b288b5b2d95` |
| `item_01.md` | `QMATH_研究总档案_完整恢复至16_10.md` | 53746 | `c4bd87c96065ea7bbc028994b0a9ec2db73ff9805f4cbf68a9c7206607212cf4` |
| `item_02.zip` | `QMATH_研究总档案_早期补全及16_10研究接续包.zip` | 157702 | `c937ceb81b5a66e68339e64e57c76a58bc1506897daeca01fed432b5e939b4e6` |
| `item_03.md` | `Q-MATH-16.10_二光子异常矩阵的严格经典仿真_能量壳禁戒_磁性张量判据.md` | 17220 | `ccd63a9c0edde898abff6227b4f517c49eaf507e093af4821608811cea4cadcf` |
| `item_04.md` | `Q-MATH-16.11_早期底稿文献AI与能量加权经典界.md` | 12359 | `0b0a567de9ef5f084639654cc21bf510e54291a94d04d4fbdf7a1f2de8cb7104` |
| `item_05.docx` | 同上 docx | 43829 | `76366b661ab28054ab16707883884d629924bc9e58614ab766de3ece47dee014` |
| `item_06.py` | `qmath_16_11_weighted_bound.py` | 1409 | `2af853f18302bd95ebfbb3dc53f8a7c8988a0de33f67ef8d85dce2b7f84586c1` |
| `item_07.json` | `qmath_16_11_weighted_bound_results.json` | 1286 | `f50fe7435411078a23f914ed4fc744b22e44c787899f00765bd839c76ff7a3f7` |

## 3. 内层 16.10 与 16.9

`item_02.zip` 解到 `inner_16_10/`。关键条目：

| alias | 原名 | SHA256 |
| --- | --- | --- |
| `inner_03.md` | 与顶层 `item_03.md` 同文 | `ccd63a9c0edde898abff6227b4f517c49eaf507e093af4821608811cea4cadcf` |
| `inner_04.py` | `qmath_16_10_reproduce.py` | `d14336af3c51677cfa26951e1cc624f0ce324fbb59406f66e97f0d1561007475` |
| `inner_05.json` | `qmath_16_10_verification.json` | `2e307dbc6af89062b4bfdfc96cc34edf8c0f64e7f16ae301121006ff9b2e26f7` |
| `inner_08.zip` | `QMATH_量子光磁性HHG_总档案与16_9接续包.zip` | `c2cf3cd608638920ae59d7b8875e1ec0d7a47b9b2ecddc1b537caa2c01008660` |

`inner_08.zip` **不是** 16.2 源码包。再解到 `inner_16_9/`：

| alias | 备注 | SHA256 |
| --- | --- | --- |
| `i9_01.md` | 与 `inner_07.md` 同文（截至 16.9 总档案） | `2d513bf46a68ba13b19cddcb97ce73f665689a12403bf07555e95bd7d4334c52` |
| `i9_04.py` | `qmath_16_9_verify.py` | `05ce98058e1e85daad09ebe776dad4f3fe5cf2c713a7d6c25e65414f08192fdb` |
| `i9_05.json` | `qmath_16_9_verification.json` | `6117dd92b4100eb8c2f37555a057d78d8081a44471e44823cd92123d0b2badc8` |

`i9_00.docx` 原名在 zip 内为乱码（编码不是 UTF-8），哈希 `baf469a7da2186e8e53086c4832dc3f293d9312e71e9cfbbdd6c7b19f22927a6`。不得根据乱码文件名脑补内容。

## 4. 明确缺件：Q-MATH-16.2

| 声明 | 状态 |
| --- | --- |
| 16.11 包内是否含 16.2 源码 | **否** |
| 外审指名的独立 ZIP `Q-MATH-16.2_真实八带M0二光子核_BZ磁性选择规则与噪声通道审计_20261022.zip` | **本机未找到** |
| 接触核公式是否在指导书/16.10 中被引用 | 是，c-number 形式 \(K^{(2)}=\mathrm{Tr}(j_0\rho_{12})+\mathrm{Tr}[j_1(\rho_1+\rho_2)]+\mathrm{Tr}(j_2 P_0)\) |
| 本轮是否独立复现 16.2 数值核 | **否** |
| 允许写什么 | “公式被引用；源码缺失；定量 \(K^{(2)}\) 未导出” |
| 禁止写什么 | “已复现 16.2”“已从完整 Peierls 核算出 \(\Lambda_2^o\)” |

已搜路径：Downloads、Documents、Desktop、`Documents\8带数学`、`Documents\量子光研究`、本仓库。16.2 仍不在 16.11 包内，独立 ZIP 亦未出现。

指导书还提到早期 Mathematica / Q-MATH-03～16.8 的其他 ZIP。那些包**不在**本次 16.11 解压目录中，本轮未核验其字节。不得把资料库“曾经存在”升级为本机已解压、已复跑。

### 4.1 旁系：Q-MATH-15（另一协议，不得套用）

| 对象 | 状态 |
| --- | --- |
| 路径 | `C:\Users\26507\Downloads\Q-MATH-15_纯数学纠错_经典正P全局证书_共享双k核验_20261022.zip` |
| 大小 / 时间 | 515497 bytes，2026-10-23 11:09 |
| 内容（只读清单，未覆盖 `tools/`） | 约 49 条目；含 23-Fock 仿射证书、QMATH13 M0 H2/H4 kernel |
| 协议 | 纯 BSV、零均值、\(4\pi\) 绕数、单 \(k\)/H2 perp |
| 对本线 | **不是** `Qin_a2_1_4_1` 位移有限门 \(\mathcal J_2^o\)。禁止把 15 的仿射包络移到本观测量，禁止解压覆盖仓库源码。 |

## 5. 仓库模型源码哈希（独立脚本核验）

| 文件 | SHA256 | matches_expected |
| --- | --- | --- |
| `tools/b0_small_model/q8_m0_model.py` | `5d53090cabb60238c6c16a74b92ddc061d5adf2d8081e48b46b9bdf0089261d3` | True |
| `tools/b0_small_model/q8_m0_qinput_dynamics_20261026.py` | `bf6d4e407bf55f1e6ebc3767f1b2bd6075ac7ef564f195abc1217c7d51671f94` | True |
| `tools/b0_small_model/q8_m0_qinput_minimal_20261020.py` | `b60c3b7e53e413e614d4a8ec784956f4d16001a906465314dea963a17ae9bc97` | 无预定对照 |

独立重算脚本已内联 \(a_2\) 偏振，避免依赖本机 runtime 的 scipy，也避免 `import q8_m0_qinput_minimal_20261020`。

## 6. 冻结线 Q JSON（只读）

| 标签 | 路径 | SHA256 |
| --- | --- | --- |
| thermal | `results/q_line/qinput_qin_a2_141_gh5_thermal_20261120/gh5_thermal.json` | `909ca504142c56bc44b634a26f67852802ebed7ee9801b7db094a055cf0e7d88` |
| qed_theta_pi | `results/q_line/qinput_qin_a2_141_qed_theta_pi_20261120/qed_theta_pi.json` | `b2518827ef9ff0b1deddd6f2ceabdfa302927a241a42aee10590554d837c0dfa` |
| mech | `results/q_line/qinput_qin_a2_141_qed_theta_pi_mech_20261120/mech.json` | `dfd4827f7d0d7b48c17bee3635c5267364f434b264382d1a18d693b1e81ee56c` |
| q1a | `results/q_line/qinput_qin_a2_141_q1a_20261120/q1a.json` | `aaf5daf013ee84a70afb1cd29824d9da3d7af0694cc2f6cfcd9611dc895c5c87` |
| gh5 | `results/q_line/qinput_qin_a2_141_gh5_20261120/gh5.json` | `94bf3c0e645b00d12bf490e728a2b5049e4088dfc2cb8d1cd3ee28f8129898ea` |
| gh5_theta_pi | `results/q_line/qinput_qin_a2_141_gh5_theta_pi_20261120/gh5_theta_pi.json` | `95758b9dde46721b554a7f6e80d84619eeaaaede33b579cda371887d319a25c2` |

本轮只读提取 H2 复振幅与 \(\Lambda_{\mathrm{eff}}\)。`qed_theta_pi` 两路 `job_status=timeout_partial`，覆盖约 \(4.757T_0\)：主窗完整，后窗 \([3T_0,5T_0]\) **缺失**。`abs_back=0` 是数据缺口，不是物理零。

## 7. 本轮只读产物

| 文件 | 作用 |
| --- | --- |
| `tools/b0_small_model/qmath_r1_readonly_audit.py` | 哈希与链接审计 |
| `tools/b0_small_model/qmath_r1_extract_h2.py` | 只读提取 H2 |
| `tools/b0_small_model/qmath_r1_compact.py` | 压缩摘要 |
| `tools/b0_small_model/qmath_r1_independent_recompute.py` | 独立重算（内联偏振） |
| `docs/04_BSV量子光/qmath_link_audit.json` | 谱系/哈希机读 |
| `docs/04_BSV量子光/qmath_r1_independent.json` | 独立数字 |
| `docs/04_BSV量子光/qmath_r1_h2_extract.json` | 原始提取 |
| `docs/04_BSV量子光/qmath_r1_recompute.json` | 接触项 FD |
| `docs/04_BSV量子光/qmath_r1_compact.txt` | 文本摘要 |
| `docs/04_BSV量子光/PHYSICS_CONTRACT.md` | 已写完的物理契约 |
| `docs/04_BSV量子光/SOURCES_PROVENANCE.md` | 本文件 |
| `docs/04_BSV量子光/线Q_QMATH数学成果收割_响应算符与正P经典界_R1.md` | R1 主报告 |
| `docs/04_BSV量子光/线Q_QMATH_R1.5_正P经典可达界_20261023.md` | R1.5 纯数学：正 P 可达集与对偶证书 |

R1 订正（2026-10-23）：§6.1 删除虚部大十倍的重复 \(R_2\) 行；§6.2 明确 6.488 为 \(J_x\) 比、2D 比约为 5.80；§7 反例 B 改为固定 \(\alpha=0.5\) 背景加对称稀有大涨落。`qmath_r1_independent.json` **未改**（改了会坏哈希）；正文已注明键名 `amp_ratio_2d` 误标。

## 8. 定理原文定位（不在此复述证明）

- **定理 A**（`item_03.md` §1）：零均值非负 Glauber–Sudarshan \(P\) 能实现异常矩阵 \(M\) 当且仅当 \(\|M\|_*\le E\)。
- **定理 B**（`item_03.md` §2）：最低阶双光子相干转换下，归一化输出图样可被 \(M_{\mathrm{cl}}=tM\) 的正 \(P\) 经典输入整体复制。**禁止用归一化 H2 图样宣称量子优势。**
- 这两条**不直接**覆盖：位移压缩、有限耦合门、非微扰 \(g=0.005\) 的 QED 电流 Fourier。

## 9. 本轮停机边界

R0/R1 订正与 R1.5 纯数学到此结束。禁止：新 QED/GH5、Cn、SV、GH7、extra k、nmax、Floquet、服务器、执行 R2。未经用户明确批准，不得把“下一步建议”自动变成作业，不得把 Q-MATH-15 证书套到本线。
