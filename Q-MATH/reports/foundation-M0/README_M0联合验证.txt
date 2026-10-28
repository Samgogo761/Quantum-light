M0 八带数学重构：生产源 + 独立验证包（2026-10-21）

来源：你上传的三份原始 M0 数据，保留未修改的源码。
模型计算环境：Python>=3.10, numpy, scipy。

1. 在本目录运行：
   python check_m0_uploaded.py
   python M0_independent_equation_crosscheck.py
   python M0_contact_derivative_validation.py
   python M0_influence_exterior_validation.py

2. 用 Mathematica 14+ / Wolfram Language 在本目录运行：
   wolframscript -file M0_Wolfram_independent_validation.wl
   或者 Get["<目录>/M0_Wolfram_independent_validation.wl"]

3. Mathematica 程序预期输出：
   M0_Wolfram_validation_results.json
   PASSED: largest check = ...
   如果失败，将所有输出与错误发回 ChatGPT 或 Cursor。

注意：Mathematica 未在 ChatGPT 当前环境实际运行！只有 Python 项目及数值验证实际运行。
原始 q8_tperp_static_audit.py 默认 SCF 路径指向原仓库结构；验证 Python
明确将用户上传的 SCF 文件传给 build_geometry，源文件本身没有修改。

研究说明：M0_数学重构_联合验收报告_20261021.md
历史理论底稿：研究底稿_八带量子光HHG_数学重构_v1.md

物理边界：电子影响泛函的数学与数值正确性 != G1 canonical output 通过。
本包没有提供任何真实的光子 n、m、压缩、纠缠或磁畴读取结果。
