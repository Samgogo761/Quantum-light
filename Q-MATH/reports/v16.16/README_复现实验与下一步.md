# Q-MATH-16.16 可复核包

本包直接接续 Q-MATH-16.15：使用用户此前上传的 Q-MATH-16.7 原始生产模型 `q167_m0_three_photon_T.py`，以及必需的 `deps/` 模型定义、几何解析、生产 SCF 坐标。原始依赖放在 `src/`；新增脚本在上层，**不修改原始模型源码**。

## 核心文件

- `Q-MATH-16.16_数学定理_能量壳与经典负控.md`：数学推导、文献和结果解释；所有数学公式均为 LaTeX。
- `Q-MATH-16.16_数学定理_能量壳与经典负控.docx`：带 Word 公式及图示的阅读版。
- `qmath_16_16_on_shell_m0.py`：完整 36 维非弹性末态、同一固定宽带输入 u(ω)、严格输出能量壳、三阶 Peierls T 核。
- `qmath_16_16_invariant_tests.py`：原始 q167 弹性振幅 6 组交叉检查，最大相对偏差 ~3.79e-15。
- `qmath_16_16_convergence.py`：改变内部频率 Gauss 节点、虚态展宽 gamma、BZ 网格。
- `qmath_16_16_nonmagnetic_k_control.py`：Néel 交换=0、SOC=0 的负控。
- `qmath_16_16_magnetism_null.py`：不同交换强度的局部能隙对照。
- `qmath_16_16_classical_theorem.wl`：非负 P 混合的 Mathematica 代数验证参考脚本；**未在本轮执行 Mathematica**。
- `*.json`：本轮实际计算的源数据。图仅用来展示当前不收敛警示。

## 最小运行环境与命令

Python 3.11+ 和 NumPy。某些脚本绘图可另用 Matplotlib。建议设置 `OPENBLAS_NUM_THREADS=1`。

```bash
python qmath_16_16_invariant_tests.py
python qmath_16_16_on_shell_m0.py
python qmath_16_16_convergence.py
python qmath_16_16_nonmagnetic_k_control.py
```

注：本轮未做真正 LSZ/入射态着装或 Maxwell 双层边界的绝对辐射定标。结论是**同一宽带输入下最低阶单发射扇区的数学 no-go**和**SOC-free 非弹性核在 M0 中不为零但仍未收敛的数值证据**。原始 BZ 计数网格数值剧烈变化，不能作为可发表的发射率预测。

## Q-MATH-16.17 建议优先顺序

首先解决外腿/近共振重求和、正确的散射波包归一化与 BZ 自适应采样，再研究高阶的真正连接算符 \(a_u^\dagger a_u^3\) 和同波包物理输出 \(S_1(\Omega;u)\)，以便突破全部低阶发射计数只由 \(F_2\) 决定的强 no-go。若收敛后仍只是 \(a_u^2\)，则停止宣传这一分支的磁性量子新优势，转向两条真正独立且占据的 BSV 模或精确光场背作用。
