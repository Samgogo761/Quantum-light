# Q-MATH-16.29 独立复算说明

**警告：全部输出是基于真实 M0 的 *classical retarded* χ² 构成的光场异常矩读取代理，不是完成的量子化 HHG 物理出射。**

## 运行

Python 3.10+，NumPy、SciPy、Matplotlib、Pandoc（文档导出）。在含 `q16_27_reference`（上轮原始 M0 必要源程序）的目录中运行：

```bash
python qmath_16_29_same_state_readout.py
python qmath_16_29_positive_P_certificate.py
python qmath_16_29_modal_occupation_certificate.py
python qmath_16_29_material_null.py
python qmath_16_29_material_dual_certificate.py
python make_figures.py
```

脚本依赖 `q16_27_reference/qmath_16_27_retarded_kernel.py` 及其 `source/reference_q167` 的原始依赖；文件结构必须保留。

注意：`qmath_16_29_same_state_readout.py` 的全部网格计算较耗时；它优先复用 `qmath_16_29_kernel_cache.json`。`qmath_16_29_material_null.py` 默认只扫描报告里列出的四个案例；201×201,η=.03 的附加扫描结果已缓存并保存在 JSON，若要单独复跑使用 `compute(201,.03,.8)`。

## 验收

- BSV 两输入模式是独立占据的偶/奇波包，两者正交；自由光能量矩阵包含非零的模式间自由能量矩元。
- 在输出 Ω=.36 时 $K_{eo}=0$，在 Ω=.34,.38 时 $K_{eo}\ne0$；对于经典正 P 输入不允许擅自把 $M_{eo}$ 固定为 0。
- 八项经典 P 混合：总概率为 1、平均场为 0、每模式占据等于指定 BSV 目标（严格模式匹配版本）、两个输出代理复振幅同时匹配。
- 三输出矩阵与频率平坦材料对照矩阵都满秩；不能把满秩称为材料专有物理。
- 五输出 $P_\perp K_{\rm flat}=0$ 数值误差 $<10^{-12}$；双重见证的原始核范数优化与对偶界相差 $<10^{-10}$。
- 窄展宽 BZ 漂移与约 1% 的材料剩余信号必须明示；无量子输出、无 LSZ、无 Maxwell、无实际实验信噪比认证。

## 证据分级

- **严格数学：** 正 P 异常矩的加权核范数约束、模式光子数约束、五输出平坦核消去恒等式、正 P 对偶上界。
- **原始 M0 数值：** 经典定态双频因果核在指定 BZ、η 与频率点的求和；频率形状曲率。
- **合成/算法结果：** 正 P 优化与 Takagi/协方差构造的可复算证书。
- **未完成：** 全量子输入输出散射、光子数统计、有限带宽 H2 模式、远场 Maxwell、光学真实信噪比。
