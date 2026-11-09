# Q-MATH-16.26｜复算说明

## 范围

真实 M0 八带 Peierls 几何和所有四条占据带参与的**经典时域电子非线性电流系数**。既不是经典材料线 full112，也不是光子量子化 BSV 的 HHG 输出。有限脉冲二/三阶电流投影不等于无限时单色 \(\chi^{(2)}\)/\(\chi^{(3)}\) 张量。BN 网格仍明显不收敛。

## 运行

从 ZIP 解压后在该目录执行：

```bash
python qmath_16_26_run_verified.py
```

依赖：Python 3.10+、NumPy。源码依赖均在 `source/reference_q167` 下；无需改系统路径。为了结果复现，脚本默认将 OpenBLAS/OpenMP/MKL 线程限制为 1。完成后，`qmath_16_26_verified_results.json` 保存全部回归指标，`qmath_16_26_primary_timeseries.npz` 保存正确模型 `Delta=.8, N=+1, lambda=.1` 的时间轨迹。

如需导入而不是执行，使用 `from qmath_16_26_cubic_audit import propagate`。

## 两个彼此独立的实现

1. 完整 Peierls 指数的直接密度矩阵动力学，使用四幅度 \(\pm a,\pm 2a\) 求场强导数。
2. 从 \(h_0\) 到 \(h_4\) 的解析 Peierls 顶点和逐阶 Liouville 递推；H2 横向电流需要 \(h_{hpp}\)。

**注意：** 有限差分强度必须小。旧试算时直接使用 \(a=0.014\) 带来明显高阶污染（约 40% 相对误差）；正式回归使用 \(a=3\times10^{-4}\)，三阶相对误差约 \(2.4\times10^{-4}\)。不得以旧大强度结果认证算法失败。

## 未完成

- BZ 收敛、脉冲包络/带宽与真实材料弛豫误差预算；
- 真正多频的因果三阶响应 \(\chi^{(3)}(\omega_1,\omega_2,\omega_3)\)；
- 无门控供能的量子 BSV 与物理 Maxwell 出射光；
- 全部正 Glauber–Sudarshan P 经典凸对照；
- 与真实 CrI3 Wannier 线的材料比较。

## 审计基准

- 第四 Peierls 导数 \(h_4\) 和低阶导数的解析/有限差分交叉检查；
- \(N\to-N\) 下横向 H2 翻号、纵向三阶不变；
- \(\Delta=0\) 或 SOC=0，横向 H2 禁戒（SOC=0 的小隙/无隙情况仅作为对称性负控）；
- 有限幅度误差近似随 \(a^2\) 缩小；
- 时间步数加倍后一致；
- 当前 BZ 未收敛，禁止将复谱数值用于物理发现。
