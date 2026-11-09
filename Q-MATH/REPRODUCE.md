# 复现入口与现有验证

本次未运行任何科研计算。归档脚本仅依赖 Python 3 标准库与 Git；研究程序所需 NumPy/SciPy/Matplotlib 等按原报告另行准备。本次不安装研究依赖、不启动 Wolfram 内核或服务器。

## 先只核验归档

在仓库根目录运行：

```powershell
python tools/qmath/manage.py check
```

该命令逐一核验清单中的 SHA-256 和包引用，不 import 研究模块。`refresh` 在核验通过后生成文件索引，不改变原始报告内容。

## 最新版本的文件布局

```powershell
python tools/qmath/manage.py prepare v16.30
```

这只在 `local_runs/qmath_reproduce/v16.30/` 还原包内路径，包括被去重的模型、几何输入、缓存和结果文件。已有目标目录时拒绝覆盖；整个目录由现有 `.gitignore` 排除。临时复现目录不是新的维护源或 Git 仓库。原包 DOCX、ZIP 副本及排除成员不还原，详见 manifest。

**以下命令是将来另外批准科研复现时的入口，本次没有执行：**

```powershell
Set-Location local_runs/qmath_reproduce/v16.30
$env:OPENBLAS_NUM_THREADS = '1'
python qmath_16_30_bridge_audit.py
python qmath_16_30_output_mode_gram.py
```

原版 bridge 脚本默认只运行 L=9/17/25；`qmath_16_30_clean_results.json` 还保存了 L=41/61/81、γ=.03 等检查点。默认入口**不会**自动重现完整报告所有扫描；不能把小网格执行成功写成全量复现。脚本写入的结果仅发生在临时目录，不覆盖档案。原始程序包含环境/路径约定；本次只保留原字节，没有改写科学实现或证明其运行环境已经就绪。

## 测试与历史入口

下表中的脚本均已保存，属于**原作者的既有验证/复算入口**，不代表本次已经执行。prepare 包名见 INDEX；版本号目录中的文件可能经 manifest 引用其他位置。

| 包 | 原相对入口 | 用途及限制 |
|---|---|---|
| `v05` | `QMATH05_shared/test_qmath05.py`、`unit_test_output.json` | 早期共享光场测试；后续 Q06 截断纠错限制其物理解释 |
| `v15` | `Q-MATH-15/q15_audit_certificate.py`、`q15_duhamel_tail_certificate.py` | 有限保存响应核的经典证书/尾界；不自动证明无限 Fock、共享 BZ 优势 |
| `history-to-16.9` / `history-to-16.10` / `history-to-16.11` | `qmath_16_9_verify.py` / `qmath_16_10_reproduce.py` / `qmath_16_11_weighted_bound.py` | 数学或玩具核核验，不是重新执行 M0 散射 |
| `v16.15` | `sources/QMATH_16_14/Q16_13_original/src_0/q167_verify_saved.py` | 找回的 Q16.7 原始程序依赖；该入口是否具备所需原数组须另核对，不据源码存在声称全包已复现 |
| `v16.16` | `qmath_16_16_invariant_tests.py` | 数学不变量核验，附原结果 JSON |
| `v16.18` | `qmath_16_18_quick_verify.py` | 行列式/逆问题检查，附 partial/final 结果，证据范围按原报告 |
| `v16.24` | `qmath_16_24_regression_tests.py` | 静态 Ward 与多体交叉检查，含约化玩具模型，不能混称完整共享场生产结果 |
| `v16.25` | `Q-MATH-16.25/qmath_16_25_regression_tests.py` | 动态消元回归 |
| `v16.27` | `qmath_16_27_reproduce.py`；`--full` 为全扫描 | 默认小网格验算与全扫描不同；经典因果 χ² 不是量子输出 |
| `v16.29` | `qmath_16_29_positive_P_certificate.py`、`qmath_16_29_modal_occupation_certificate.py`、`qmath_16_29_material_dual_certificate.py` | 显式经典反例、模式占据匹配、代理对偶证书；16.30 收紧五输出解释 |
| `v16.30` | `qmath_16_30_bridge_audit.py`、`qmath_16_30_output_mode_gram.py` | 裸 T 桥接与模式 Gram；BZ/真实输出仍未验收 |

线 Q R1/R1.5 的程序仍在 `tools/b0_small_model/qmath_r1_*.py`，证据 JSON 在 `docs/04_BSV量子光/`。这些脚本含本机绝对路径、读取未入库的大型运行区并可能写摘要，**不能当作跨机器开箱即用的复现入口**。本次已保存既有摘要与限制，未改写原脚本；需要完整线 Q 再分析时应另行准备明确输入清单和独立输出目录。

M0 基础资料的 Wolfram 脚本已保存，但其注释明确表示原提供者当时未运行 Mathematica；没有找到独立成功运行证据时，索引不将它列为“已通过 Wolfram 验证”。
