# Q-MATH-16.27 复算说明

## 从头重现

文件包有 `source/reference_q167/q167_m0_three_photon_T.py` 与依赖的真实 M0 几何/Peierls 键，以及上一轮 Q16.26 已通过的基础代码。

先安装 NumPy；不需要网络：

```bash
python qmath_16_27_reproduce.py
```

这将运行约 9×9 BZ 的双频混合导数与完整 Peierls 四点差分交叉验算，并验证 49×49 定态核。

完整重算所有网格、所有频率和报告中的 Ward 审计：

```bash
python qmath_16_27_reproduce.py --full
```

该命令覆盖 JSON 结果；报告图片由 `make_report.py` 读取保存结果并生成。为避免每次生成 Word 时重跑数值，没有把数值计算嵌入报告脚本。

## 证据边界

1. `qmath_16_27_two_tone.py` 是**有限时域**经典 Peierls 混频；`qmath_16_27_retarded_kernel.py` 是频域绝热开启的完整八带经典因果响应。
2. `eta` 是频域正则化/开启参数，**没有证明**它等于真实材料的微观退相干率。
3. 所有结果未包含量子 BSV 输入态的电子反作用、量子光 Keldysh 收缩、光子 LSZ 和 Maxwell 远场。
4. 单波包 BSV 的因子化限制是给定**弱双光子异常矩线性读出假设**下的线性代数定理，不证明全部量子高阶 HHG。
5. 161→201 BZ 的展宽 0.01 收敛尚未通过；任何“共振新物理”都应暂缓。
6. 原始 Q16.7 源码为依赖；没有改写其材料键表和几何。
