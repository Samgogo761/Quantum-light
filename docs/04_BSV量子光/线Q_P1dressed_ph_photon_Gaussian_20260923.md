# 线 Q：P1-dressed 粒子—空穴 + 两端口光子 canonical Gaussian（2026-09-23）

人审结论：批准。不是换研究方向，是补上 1N/1O 查清的理论闭合缺陷。

名字收紧为

\[
\textbf{P1-dressed particle-hole canonical Gaussian theory}
\]

也可读成强经典驱动背景上的 time-dependent RPA / 准玻色切空间理论。

代码：`q8_m0_phgauss.py`、`q8_m0_stage1p_phgauss.py`。  
`STAGE1_STATUS = stage1_phgauss_canonical_main_20260923`。  
`THEORY_MAIN = P1_dressed_particle_hole_photon_canonical_Gaussian`。  
`EXPECTATION_RESOLVENT = RA_xi_resolvent_classical_expectation_feedback_diagnostic`。  
JSON：`results/q_line/stage1p_phgauss_20260923/stage1p_phgauss.json`。

生产 \(n,m,V(\theta)\) 仍被 `apply_resummation_to_output` / `apply_response_to_output` 挡住。不加 \(T_2\)，不启动非线性 fermionic feedback，不做大 BZ。

---

## 物理问题

1N/1O 已经把竞争解释分开：

- Born 阶 \(2\operatorname{Re}A_2+\mathrm{comm}_s\simeq0\) 成立。
- 同一套 \(\mathcal R^A+\xi\) 无穷迭代，\(\Delta_{\mathrm{can}}\propto\lambda^{2.18}\sim O(\eta^4)\)，Neumann \(n=2\) 才出现 \(O(1)\) 缺口。
- 经典期望值 resolvent \((1-\mathcal L)^{-1}\) 数学上可以存在；把它宣布成 quantum input-output map 失败。
- \(\Delta C\) 最大处在最低红外箱、H1、H2–IR，与 \(H2\to IR\to H2\) 同一条路径。

问题因此不是「再给 \(\mathcal R^A+\xi\) 打补丁」，而是：把已经验证的 CJJ/Kubo 结构提升成显式、canonical 的二次量子动力学。

---

## 冻结的理论

### 分层（旧理论重新分类）

| 对象 | 地位 |
|---|---|
| Born/Kubo | 量子一致的微扰 benchmark |
| \(\mathcal R^A+\xi+(1-\mathcal L)^{-1}\) | 经典/期望值反馈诊断，**不再是**量子主框架 |
| P1-dressed p-h + 两端口光子 Gaussian | **下一版量子主框架** |
| \(\Delta C\) | 诊断；禁止解释成缺少的 Langevin bath 再塞回去 |
| \(G_{\mathrm{self}}\) | 消去 photon continuum 后应出现的有效核，**禁止**再当人工反馈输入 |

这仍然只处理：经典 P1 非微扰，加上围绕 P1 轨迹的线性/Gaussian 涨落。Pauli 饱和、大幅量子扰动、非高斯高阶相关都还没进来。

### 共动粒子—空穴

占据 4、空态 4，每个 \(k\) 有 16 个 \(v\to c\) 通道。相互作用绘景里

\[
\hat B_{cv,k}^\dagger\sim\tilde c_{c,k}^\dagger\tilde c_{v,k},
\qquad
[\hat B_{cv,k},\hat B_{c'v',k'}^\dagger]\simeq\delta_{kk'}\delta_{cc'}\delta_{vv'}.
\]

已有 CJJ 顶点

\[
X_{a,k}(t)=V_c^\dagger\tilde j_{a,k}(t)V_v
\]

给出

\[
\delta\hat I_a^{\mathrm{ph}}(t)
=
\sum_{kcv}
\Bigl[
X_{a,k;cv}(t)\hat B_{cv,k}^\dagger
+
X_{a,k;cv}^*(t)\hat B_{cv,k}
\Bigr].
\]

p-h 真空上 \(\langle\delta\hat I_a(t)\delta\hat I_b(t')\rangle\) 就是现有 \(C^>\) Gram。数值实现里顶点带 \(\sqrt{w_k}\)，使得

\[
\langle I_a I_b\rangle=\sum_k w_k\operatorname{Tr}[X_a^\dagger X_b].
\]

\(w_k\) 是 BZ 求积权重，**不是** \(N_{\mathrm{cell}}\)。SI 层继续用已冻结的 \(\eta^2 N_{\mathrm{cell}}=\alpha_{\mathrm{fs}}/A_{\mathrm{uc}}\)。

### 二次 Hamiltonian

围绕经典 \(A_{\mathrm{cl}}\)（P1）展开到量子涨落二次阶：

\[
\hat H_{\mathrm{Q}}(t)
=
\hat H_{\mathrm{EM}}
-
\sum_a
\delta\hat A_a(t)\,\delta\hat I_a^{\mathrm{ph}}(t)
+
\frac12\sum_{ab}
\bar D_{ab}(t)\,\delta\hat A_a(t)\delta\hat A_b(t).
\]

\(\bar D\) 是沿 P1 轨迹的 diamagnetic/contact，它是 Hamiltonian 的一部分，不是事后补进响应。response、noise、contact 来自同一个二次 \(H\)，演化按构造应是 symplectic/canonical。

光子用相互作用绘景：时间在波形 \(F_\mu(t)\) 里，不再另加 \(\omega b^\dagger b\)。两端口自由空间连续模显式保留。禁止同时写 \(A_{\mathrm{self}}=G_{\mathrm{self}}I\) 又让光子和电子耦一次（double count）。若形式上消去光子，应自动恢复原来的 Maxwell \(G_{\mathrm{self}}\)——这是后续验收，不是本轮输入。

### 自洽性门（以后，不是现在）

准玻色近似把 \([\hat B,\hat B^\dagger]\) 换成参考 Slater 轨迹上的 c-number。真正的线性化自洽门是 **p-h 涨落占据是否仍然足够小**，不是 vacuum RMS 对 \(A_{\mathrm{P1}}\)。若每个可用通道被大量占据，才讨论非线性 fermionic feedback。

---

## Prototype 证据

工程选择（网格、模式数、积分器）不需要人审：2 个 \(k\)（\(\Gamma\) 与 \((0.17,0.41)\)）、左右各一个 H2 包、隐式中点 / Cayley。单元测试更粗（`ntpc=8`）；战役 `ntpc=32`。

弱耦合（规定 \(\delta A\)，p-h 左 Riemann 前缀）必须严格回到现有结果，否则停：

- \(C^>\) 与 `c_greater_from_x` 重合。
- \(\delta\mu\) 的 state+contact 与 `predict_delta_mu` 重合，因此 \(A_2,B_2\) 重合。
- \(2\operatorname{Re}A_2+\mathrm{comm}_s\) 与同一网格上的 \(R^A\) Kubo 残差相同（粗时间网格的地板属于旧理论，不属于新缺陷）。

打开光子反馈后：

- \(\xi=(q,p)\)，\(J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}\)，\(K=J\,\mathrm{Hess}\) 满足 \(JK+K^TJ=0\)。
- \(\Phi^T J\Phi=J\)。
- 从 \(\Phi\) 读 \(b_{\mathrm{out}}=Sb_{\mathrm{in}}+Tb_{\mathrm{in}}^\dagger+UB+VB^\dagger\)，则 \(SS^\dagger-TT^\dagger+C_{\mathrm{el}}=I\)，其中 \(C_{\mathrm{el}}=UU^\dagger-VV^\dagger\)。

战役（2 个 \(k\)，`ntpc=32`，\(n_t=257\)，2 个光子振子 + 32 个 p-h 通道，墙时 \(0.27\,\mathrm{s}\)）：

| 检查 | 残差 |
|---|---|
| \(C^>\) vs CJJ Gram | \(0\) |
| \(\delta I^{\mathrm{ph}}\) vs \(R^A\) | \(5.8\times10^{-13}\) |
| \(A_2^{\mathrm{ph}}\) vs \(A_2^{R^A}\) | \(\lvert A_2\rvert=0.48790\) 两侧相同 |
| Kubo \(2\operatorname{Re}A_2+\mathrm{comm}_s\) | \(8.66\times10^{-6}\)，与 \(R^A\) 差 \(8\times10^{-16}\) |
| \(\lVert\Phi^T J\Phi-J\rVert_\infty\) | \(1.69\times10^{-14}\) |
| \(\lVert JK+K^TJ\rVert_\infty\) | \(5.2\times10^{-18}\) |
| \(\lVert SS^\dagger-TT^\dagger+C_{\mathrm{el}}-I\rVert_\infty\) | \(9.96\times10^{-15}\) |
| 对角 I/O | \(1.000000,\;1.000000\) |

单元测试 `test_phgauss_canonical` 已过。生产 \(V\) 仍被挡住。JSON：`results/q_line/stage1p_phgauss_20260923/stage1p_phgauss.json`。

---

## 备选解释（本轮不采用）

1. **继续给 \(\mathcal R^A+\xi\) 打 \(\Delta C\) 补丁。** 1O 已说明缺口从 \(L^2\) 起，且集中在 IR/H1；补丁不是从同一个 \(H\) 来的噪声。
2. **手写 \(G_{\mathrm{self}}\) 再加光子。** Double count；新理论里 \(G_{\mathrm{self}}\) 必须 emerge。
3. **把 vacuum RMS \(\gg A_{\mathrm{P1}}\) 读成线性区已经失败。** RMS 仍只是应力诊断。自洽门是 Pauli depletion，见 [准玻色 Pauli 自洽](线Q_准玻色Pauli自洽_20260923.md)。
4. **现在就算生产 \(V(\theta)\)。** \(G_{\mathrm{self}}\) 再现、BZ、H2 的 \(n,m,V\) 都还没做。

---

## 明确还没做 / 已改顺序

1. 准玻色 Pauli：三层量已计算，\(S=N_{\mathrm{cell}}A_{\mathrm{uc}}\) 待人审。
2. 消去光子后是否数值恢复 \(G_{\mathrm{self}}\)（小 \(k\)，排在 BZ 前）。
3. BZ 收敛。
4. H2 的 \(n,m,V(\theta)\)，以及 BSV。

顺序现为：occupation（本轮）\(\to G_{\mathrm{self}}\) 再现 \(\to\) BZ \(\to n,m,V\)。
