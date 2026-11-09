# 线 Q：C3 静态 PT / Bloch sewing 收尾（2026-09-10）

```text
C3_FROZEN = False
DOWNFOLD  = not_performed
DYNAMICS  = not_started
```

本页结束静态几何补查。常数 intertwiner **不再扩大**。原始坐标保留。`test_q8_tperp_static_audit.py` PASS。

---

## 1. 已更正的数字

| 对象 | \(P\) / \(PT\)（F 范数，原坐标） | 来源 |
|---|---:|---|
| GEO 等幅 | \(1.89\times10^{-6}\) | \(tB\) 的反演像相对 \(bA\) 差 \(\sim 6\times10^{-6}\)（分数坐标） |
| `C3_HERM_KDEP` | \(1.07\times10^{-6}\) | 同一配对误差；**不是零** |
| C1 / C2 / 手写卡通 | 0 | — |

GEO 四条最近层间键本来就是等幅。小残差不是「幅度不等」。

常数 \(U\)：联立方程作用在 \(8\times8\) 复数矩阵（64 元）上，已覆盖所有常数线性组合。GEO 最小奇异值 \(\approx 0.071\)，卡通 \(\approx 0.049\)。不必再枚举 Pauli。这仍不证明 \(J_o\neq 0\)。

---

## 2. 最小反演配对（供你看含义，不是新晶体）

规则：冻结生产 \(tA,tB\) 与 \(c\)；令 \(bA=2c-tB\)、\(bB=2c-tA\)（mod 1）。

| 位点 | \(\lvert\Delta\mathbf r\rvert\) (Å) |
|---|---:|
| \(tA,tB\) | 0 |
| \(bB\) | \(3.3\times10^{-9}\) |
| \(bA\) | \(3.51\times10^{-5}\) |

配对后 GEO / Hermitian-\(k\) 的 \(P\) 与 \(PT\) 降到 0。指定 \(V\) 的结论不变：GEO 仍破 \(V\)（\(B\neq 0\)），Hermitian-\(k\) 仍保 \(V\)。

这只是把已采用的几何 \(P=\tau_x\sigma_x\) 接到机器精度。接受与否是「要不要为精确 PT 微调下层 0.000035 Å」，不是换 stacking。

---

## 3. 完整矩阵 sewing

约定：\(H(\mathbf k+\mathbf G)=U^\dagger H(\mathbf k)U\)，\(U_{ii}=e^{i\mathbf G\cdot\tau_i}\)；\(T(\mathbf k+\mathbf G)=S_t^\dagger T(\mathbf k)S_b\)。抽了 \(\mathbf G=(1,0,0),(0,1,0),(1,1,0)\)。

| 检查 | GEO 原坐标 | 说明 |
|---|---|---|
| \(T\) 用位点 \(\tau\) | \(8\times10^{-17}\) | 键表 Fourier 自洽 |
| 单层 \(h_0\) | \(5\times10^{-15}\) | 上层 NN/NNN |
| \(H_8\)：两层共用 \(h_0\) + 位点 \(\tau\) | \(2.8\times10^{-4}\) | 下层 A/B 相对相位与上层 \(h_0\) 差一个配对误差 |
| 同上，配对后 | \(10^{-14}\) | 与 \(PT\) 同源 |
| 对照：下层用自己的 NN 写 \(h_0\) | \(6\times10^{-15}\) | **不是**本轮模型更改 |
| C1 / C2 的 \(T\) sewing | \(O(0.3)\) | \(k\)-无关块不是键表 Fourier |

C1/C2 不必改成晶格模型。\(k\)-dependent 换基或反幺正相似仍不推出同泵浦电流等价。

---

## 4. 停在这里

未做：冻结 eV、downfold、动力学、Rashba。C3 未冻结。

请你只看一件几何含义：是否接受把 \(bA\) 挪 \(3.5\times10^{-5}\) Å，使拟用几何的 \(P/PT\) 与位点 sewing 同时精确。反幺正与驱动选择定则，等模型和泵浦定了再查。

近隙 Cr-\(d\) / I-\(p\) 证据另页，供「8-band 是什么」用，不在本页拍板。
