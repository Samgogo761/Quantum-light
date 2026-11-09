# 量子光多周期磁序响应：Q3 精确 QED（一页）

`20261120`；执行 2026-10-24。协议 `Qin_a2_1_4_1`。`stopped = true`。

**是：精确 QED 确认，仅翻转初始 \(\operatorname{Cov}(X,P)\) 符号，就能显著旋转 Néel-odd H2 复相位。主窗 \(\Delta\phi\approx107^\circ\)，振幅比仅 1.06。与冻结 GH5 复差向量相对偏差 3.36%、相位偏差 1.31°。不是已认证 HHG，不是量子优势，也不是反铁磁特殊性证明。输入仍是 \(\alpha=0.5\) 位移压缩态。**

| 量 | 结果 |
|---|---|
| 分类 | `qed_confirms_covsign_phase_rotation_not_certified_HHG` |
| 资源 | \(\langle X\rangle=1\)，\(\langle n\rangle\approx0.3427\)；\(\operatorname{Var}(X)=\operatorname{Var}(P)\approx1.1855\) |
| \(\operatorname{Cov}(X,P)\) | \(-0.6367\leftrightarrow+0.6367\)（\(n_{\max}=8\) 实际 \(\mp0.6356\)） |
| QED \(J_x^o(\pi/2)\) | \(5.971\times10^{-8}\)，\(109.87^\circ\) |
| QED \(J_x^o(3\pi/2)\) | \(6.316\times10^{-8}\)，\(-143.23^\circ\) |
| 振幅比 | 1.058（不是增强故事） |
| \(\Delta\phi\)（wrapped） | \(106.90^\circ\) |
| \(\delta_\perp\) | \((-3.030-9.396i)\times10^{-8}\)，\(\lvert\delta_\perp\rvert=9.872\times10^{-8}\) |
| QED–GH5 | rel_vec \(3.36\%\)；\(\Delta\phi\) 偏差 \(1.31^\circ\) |
| 四角拟合 \(\lvert C\rvert/\lvert A\rvert\) | QED 0.013；GH5 0.020 |
| 奇/偶 H2 | \(\sim10^{-5}\)；绝对磁性源仍极弱 |
| 覆盖 | Q3a 主窗+后窗完整（\(5.70T_0\)）；Q3b 到 \(6T_0\)。未续算。 |

前/主/后两周期窗的 \(\Delta\phi\) 为 \(98.6^\circ\)、\(106.9^\circ\)、\(104.3^\circ\)。相位旋转可辨识，但不能认证稳定谐波。

**外审问题：** 精确 QED 是否也确认仅翻转初始 X–P 协方差符号，就能显著旋转磁序奇 H2 的复响应相位？

**是（有限时间 QED 确认；非认证 HHG）。**

停机。不自动开第三批、Cn、SV、nmax 或服务器。
