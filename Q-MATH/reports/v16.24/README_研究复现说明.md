# Q-MATH-16.24 可复算说明

环境 Python 3.11+；NumPy、SciPy、Matplotlib。

## 原始 M0 依赖
`source/deps/q19/reference_q167/` 存放本轮从 Q16.23 已验证接续包提取的原始 M0 / Q16.7 模型与几何源；`source/deps/q19/qmath_16_19_quantum_pump.py` 为原联合量子泵浦电子 bond 构造；`source/deps/qmath_16_22_projected_pump.py` 为正确 Laguerre–Fock Peierls 投影。**修改这些依赖要重新验证哈希和结果**。

## 执行顺序
```
python qmath_16_24_ward_sumrule.py
python qmath_16_24_manybody_crosscheck.py
python qmath_16_24_twist_and_stiffness.py
python qmath_16_24_shared_pair_cavity.py
python qmath_16_24_make_figures.py
```

每步产生对应 JSON；两张 PNG 为报告图。

## 不能宣称的内容
- 没有连续谱 BSV input-output/LSZ/Maxwell HHG 输出；没有纯 BSV 驱动新发现。
- 共享二动量示例为每 k 36→2 状态投影，不是完整 36×36 电子 QED，也不是 gauge-preserving downfolding。
- 共享玩具内部裸光子数不能当成远场发射计数。
- 静态 \(K\) 的有限 BZ 非零误差不表示真实破缺规范不变性；仍需收敛或解析 BZ 平移论证。
