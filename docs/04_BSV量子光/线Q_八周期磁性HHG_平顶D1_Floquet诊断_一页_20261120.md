# 磁性 HHG：D1 Floquet 诊断（一页）

`20261120`；后处理 2026-10-22。只读旧 Γ48 与 D1 电流。`propagate_called = false`。`inputs_unchanged = true`。`d2 = blocked`。墙时 2.45 s。

**周期驱动不自动给出周期电流。\(R_o\approx\sqrt{2}\) 就是 \(J^o(t)\) 与 \(J^o(t+T_0/2)\) 几乎不相关。优先假设：无耗散 Floquet 相干。只推荐少数 \(k\) 占据诊断；本轮不执行。**

一句话：驱动已经周期化；电子仍是升沿制备的 Floquet 叠加，不是对称适配的周期响应态。

同窗周期 3–5 修正原 D1 混窗（旧 3–5 vs 新 3–9）：

| 量 | 旧 Γ48 3–5 | D1 3–5 | D1 3–9 |
|---|---:|---:|---:|
| 驱动 rel | 0.09417 | \(2.47\times10^{-15}\) | \(3.70\times10^{-15}\) |
| \(R_e\) | 0.19015 | **0.09137**（同窗比 0.480） | 0.09689（混窗比 0.510） |
| \(R_o\) | 1.269 | **1.279** | 1.399 |
| \(\mathrm{rms}\,J^o\) | \(6.01\times10^{-5}\) | \(1.36\times10^{-4}\) | \(1.71\times10^{-4}\) |

中央六周期：\(C=\langle J^o(t),J^o(t+T_0/2)\rangle=0.00926\)，\(r=0.987\)，恒等式 \(R_o=\sqrt{1+r^2-2rC}\) 与直接残差逐位一致。对照 \(\langle J^e(t),-J^e(t+T_0/2)\rangle=0.995\)，总电流映射相关 0.995。偶电流已近似半周期反号；失败在 \(J^o\)。

- H2 奇源逐周期 `rel_std=0.645`，max/min=7.01；H1–H9 全部 `stationary=false`。
- \(J^o\) rect6 非整数功率 84%，主峰 \(\nu/\omega_0=0.333\)；分辨率 \(1/6\)，旁带≠准能差。
- \(\langle n_{\mathrm{exc}}\rangle\) 0.354–0.369，相对漂移 4.2%，\(\pm N\) 到 \(10^{-8}\) 一致。

| 假设 | 本轮 |
|---|---|
| H1 Floquet 相干 | **支持**（不平稳 + \(C\approx0\) + 非整数功率） |
| H2 磁畴总电流映射失败 | **当前不支持**（映射相关 0.995；每畴尚未周期） |
| H3 BZ 差分 | 未证伪，但非解释 D1 所必需；D2 仍封锁 |

只推荐、**不执行**：`floquet_occupation_few_k`。8 个代表 \(k\) × \(\pm N\)，同 2+8+2 协议，存 stroboscopic \(\rho\) 并/或对角化平台 Floquet \(H\)。估时本机 10–60 s。不改 \(H\)、不加 \(T_2\)、不上服务器。不选独立 BZ，不选任意退相干。

![共同窗](../../results/q_line/magnetic_hhg_flat_d1_floquet_diag_20261120/figure_common_window.png)

![逐周期](../../results/q_line/magnetic_hhg_flat_d1_floquet_diag_20261120/figure_cycle_harmonics.png)

停机。请决定是否批准少数 \(k\) 的 Floquet 占据诊断。不自动开 D2、延长平台、Γ192、压缩光、QED、\(T_2\)。

完整报告：[线Q_八周期磁性HHG_平顶D1_Floquet诊断_20261120.md](线Q_八周期磁性HHG_平顶D1_Floquet诊断_20261120.md)
