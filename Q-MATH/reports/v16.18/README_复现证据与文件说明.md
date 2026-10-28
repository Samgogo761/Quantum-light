# Q-MATH-16.18 复现与证据边界

本包不是“量子磁性 HHG 新发现”的认证，而是原始生产 M0 的精确费米子 Keldysh 生成泛函、新的逆问题条件定理、已完成的数值审计。

## 快速检查

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python qmath_16_18_quick_verify.py
```

会在原始冻结 M0 上重新计算两组 36 维与行列式交叉验证，检查保存数据中的磁畴及 SOC/交换项负控，验证合成逆问题的 F2/F3 反演。

## 原始 M0 时间演化与汇总

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python qmath_16_18_keldysh_determinant.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python qmath_16_18_causal_rank_and_inverse.py --full
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python qmath_16_18_compile_results.py
```

`--full` 将运行较多 BZ 和差分步长检查。典型机器可能需要更长时间；可先只运行快速检查。`qmath_16_18_causal_partial.json` 包含已经实际完成的 3/5/7 网格样本，供无需重新算电子演化时审计。所有完整结果的合并版在 `qmath_16_18_final_results.json`。

## 方法依赖

原始冻结源：`work/src_0/q167_m0_three_photon_T.py` 及 `work/src_0/deps/q8_m0_model.py`、`q8_tperp_static_audit.py`、`CrI3_scf_soc_a1h_clean.in`，来自本对话用户上传的 Q-MATH-16.7 原 ZIP。这里使用 R0 现象学八带模型和相同的几何、键列表；没有使用真实 CrI3 112 带 Wannier 模型。

新程序：
- `qmath_16_18_keldysh_determinant.py`：精确占据态行列式与完整 36 维传播比对；Keldysh 两分支/有限时间电流测试，低阶 BZ 符号检查。
- `qmath_16_18_causal_rank_and_inverse.py`：真实 M0 有限时间二阶/四阶响应；独立、明确标记 *synthetic* 的弱耦合参数扫描逆问题。
- `qmath_16_18_compile_results.py`：由已存 JSON 复算归一化夹角、条件数与两张结果图。
- `qmath_16_18_quick_verify.py`：关键测试门禁。

## 绝对禁止的推断

- 行列式生成电子连通累积量，不是自动完成 photon LSZ/波导归一化或 Maxwell 辐射模式。
- 当前 C2、C4 是外部 c-number 场下的电流响应，不能自动等同量子 S0、S11。
- 3×3—7×7 BZ 内非共线不代表已收敛。
- 数学示例里 S0/S10/S11 是**人为设定**的复数，非 M0 预测；耦合强度扫描也未建立物理实验可操作性。
- 不将普适 Stieltjes 非经典性条件与普适 Slater 行列式恒等式申报为本课题的新数学发现。
