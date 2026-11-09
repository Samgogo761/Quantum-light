# 线 Q：PT-AFM 玩具闭系统 Wick 第一阶段

日期：2026-08-31  
标签：`PT_AFM_TOY_CANDIDATE` / `development_only / not_cri3`  
**不是** CrI\(_3\)。**不是** 输出 \(g^{(2)}\)。**不是** Husimi-\(Q\)。  
本文件只记玩具数；**不要**抄进总账。

```text
LINE = Q_THEORY
ENGINE = tools/b0_small_model/pt_afm_toy.py + wick.py (7.6)/(7.10)
GAUGE = Houston k(t)=k0+A(t)
H0    = frozen 4-band (PT_AFM_TOY.md)
Jo_H2_hard_test = OFF
```

## 1. 本轮做了什么

在 `tools/b0_small_model/` 上跑闭系统、半填充 Slater、同一实验室 \(\mathbf A(t)\) 的 \(\pm N\) Houston 演化。电流 \(j_x=-\partial H/\partial k_x\)。\(\pm N\) 差分必须先 \(U_P=\tau_x\) 且 \(\mathbf k\leftrightarrow-\mathbf k\)；未映射直接作差抛 `ValueError`。等时关联用已有 `equal_time_wick`。

平均电流用实验室磁畴差，而不是把 \(P\)-odd 算符塞进 \(+N\) 的 \(\rho\)：

\[
J_e(t)=\frac12\bigl(\langle j_x\rangle_{+N}+\langle j_x\rangle_{-N}\bigr),\qquad
J_o(t)=\frac12\bigl(\langle j_x\rangle_{+N}-\langle j_x\rangle_{-N}\bigr).
\]

第一阶段表里的 \(C_{JJ}\) 是实验室总电流的等时 Wick，不是把 \((C(+N)-C(-N))/2\) 叫做 \(C_{oo}\)。

## 2. 门禁（本机 `py -3`）

| 测试 | 结果 |
|---|---|
| `test_pt_afm_toy.py` | PASS |
| `test_b0_gates.py` | 9/9 PASS |

P / T / PT、\(j_x\) 有限差分、未映射差分拒绝、\(k\) 网格对反演封闭、拒绝生产 \(T_2\)/QRT、Wick 与两时极限残差 \(\sim 10^{-16}\)、\(\rho\) 幺正演化，均过。

## 3. 第一阶段三列表（开发网格）

参数：`nk1d=4`，`ncyc=3`，`nt_per_cycle=16`，`E0=0.08`，\(\omega=0.25\)（玩具单位）。中点 \(t\)。

| 量 | 问的是什么 | \(N\to-N\) | 本轮数值 |
|---|---|---|---|
| \(J_e(\omega)\) | 线偏驱动下偶电流基频在不在 | 偶 | \(\lvert J_e(\omega)\rvert\approx 0.108\) |
| \(J_o(2\omega)\) | 平均磁性偶次 | 奇 | \(\lvert J_o(2\omega)\rvert\sim 10^{-15}\)（硬测试**关**） |
| \(\max_t\lvert J_o(t)\rvert\) | Néel-奇平均电流 | 奇 | \(\sim 10^{-15}\) |
| \(C_{JJ}(t,t)\) | 实验室总电流等时涨落 | 偶 | \(\approx 0.00939\)（中点） |
| \(C_{JJ}(+N)\approx C_{JJ}(-N)\) | N-偶课 | 偶 | 相对残差 \(\approx 3.9\times 10^{-14}\) |
| \(C_{\mathrm{odd}}^{(N)}=(C(+N)-C(-N))/2\) | 随磁畴反号的关联 | 奇 | 中点绝对值 \(\sim 10^{-16}\) |
| Wick 残差 | (7.6) 对两时 \(t=t'\) | — | \(\sim 10^{-16}\) |

`Jo_2omega_hard_test_enabled = False`。`absolute_k_normalization_validated = false`。

## 4. 为何 \(J_o\equiv 0\) 却不改 \(H_0\)

动态对称性文档（`H2_DYNAMIC_SYMMETRY_ALLOWED`）说：\(\Delta_N\neq 0\)、\(\lambda\neq 0\)、\(E\parallel x\) 时 **没有** 镜面/\(C_4\) 型禁戒，\(J_o(2\omega)\) **原则上允许**。

数值上同一 \(\mathbf k\) 的偶然等谱：

\[
\mathrm{spec}\,H_0(+\Delta_N,\mathbf k)=\mathrm{spec}\,H_0(-\Delta_N,\mathbf k).
\]

Néel 项 \(\Delta_N\tau_z s_z\) 与 \(\mathbf k\) 无关，所以 \(j_x\) 不含 \(\Delta_N\)。每个 \(\mathbf k\) 上 \(\langle j_x\rangle_{+N}=\langle j_x\rangle_{-N}\)，BZ 上 Néel-奇电流掉到数值零。偶次漏在 \(J_e\) 里。这不是 \(U_P\) 漏做，也不是网格不成对。

备忘要求：先记偶然对称，**不要**为了把 \(J_o(2\omega)\) 做出来改 \(H_0\) 花样。打破等谱是下一阶段的模型问题，不是本轮交付。

## 5. 明确没做

两时 \(C_{JJ}(t,t')\) 的双频投影、H2–H4 模关联、开系统/\(T_2\)、QRT、输出 \(g^{(2)}\)、Husimi-\(Q\)、把玩具拟合到 CrI\(_3\)。未改 `docs/00_当前状态/` 与 `docs/README.md`。
