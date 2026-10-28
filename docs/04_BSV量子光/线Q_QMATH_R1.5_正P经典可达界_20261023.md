# 线 Q × Q-MATH R1.5：有限时间磁序奇 H2 的正 P 可达集合与对偶证书

**日期：2026-10-23。研究属性：纯数学推导与既有文件只读核验。不是 R2，不是数值优化，不是新传播。**

R0/R1 原则验收；R1 三处外审订正已写入 `线Q_QMATH数学成果收割_响应算符与正P经典界_R1.md`。本文件回答：当前正 P 可达集合能否被严格界定、哪些结论已有证明、哪些必须依赖未来的响应函数 \(f(z)\)。完成后停机。R2 继续冻结。

输入态是位移压缩 \(D(0.5)S(0.3e^{i\theta})|0\rangle\)，**不是**纯 BSV。本文「正 P」= 非负 Glauber–Sudarshan 测度，**不是** SDE。热态残差 \(\varepsilon_2=0.00171\) **不是**经典界。

---

## 0. 冻结与禁止

沿用 `PHYSICS_CONTRACT.md`。协议 `Qin_a2_1_4_1`，\(g=0.005\)，\(\omega_0=0.05\)，\(n_{\max}=8\)，共享 \(k/-k\)，Peierls 电流含接触项。Fourier

\[
\mathcal J_q=\frac1{t_1-t_0}\int_{t_0}^{t_1}e^{+iq\omega_0 t}J(t)\,dt,
\]

主窗 \([2T_0,4T_0]\)。资源

\[
\alpha=0.5,\qquad n=0.34273260912113385,\qquad v=n-|\alpha|^2=0.09273260912113385.
\]

**禁止**：新 QED/GH5/Cn/SV/GH7、extra \(k\)、\(n_{\max}\)、Floquet、服务器、执行 R2、续算 θ=π 后窗、把未复现的 16.2/15 数值当本线证书。

---

## A. 精确响应算符 \(\mathsf K_2^o\)

固定电子初态 \(\rho_{e,N}\)、Néel \(N=\pm 1\)、共享光子模式、耦合开关 \(s(t)\) 与完整联合哈密顿量。令 \(U_N(t)\) 为该冻结生成元的时间演化算符（含有限门，不必是周期 Floquet）。矢势 \(A=g s(t)X\)，\(X=a+a^\dagger\)。电流算符是 Peierls 电流 \(j_N[A]\)，展开含 \(j_0,j_1,j_2,\ldots\)，不得丢掉接触项。

光子空间上的时间域响应算符（电子迹）：

\[
\mathsf K_N(t)=\mathrm{Tr}_e\Bigl[(\rho_{e,N}\otimes I_{\mathrm{ph}})\,U_N^\dagger(t)\,j_N(t)\,U_N(t)\Bigr].
\]

于是 \(J_N(t)=\mathrm{Tr}(\rho_{\mathrm{in}}\mathsf K_N(t))\)。磁序奇

\[
\mathsf K^o(t)=\frac{\mathsf K_{+1}(t)-\mathsf K_{-1}(t)}{2}.
\]

主窗 H2：

\[
\boxed{
\mathsf K_2^o=\frac1{t_1-t_0}\int_{t_0}^{t_1}e^{+i2\omega_0 t}\,\mathsf K^o(t)\,dt,
\qquad
\mathcal J_2^o[\rho_{\mathrm{in}}]=\mathrm{Tr}(\rho_{\mathrm{in}}\mathsf K_2^o).
}
\]

\(\mathsf K_2^o\) 一般非厄米。此式对任意输入光子态精确，**不需要**弱场。仿射

\[
\mathcal J_2^o\stackrel{?}{\approx}J_{\mathrm{base}}^o+\Lambda M
\]

是额外断言：它要求 \(\mathsf K_2^o\) 在所考察的态族上可被 \(I,a,a^\dagger,a^2,(a^\dagger)^2,a^\dagger a\) 的线性组合近似，而不是恒等式。

电子空间在本模型中有限维，\(j_N\) 有界，\(U_N\) 酉，故每个 \(\mathsf K_N(t)\) 作为光子算符有界，\(\|\mathsf K_2^o\|\le \|j\|\)。精确 \(f\) 因此有界。这与 \(g\) 幂级数截断后的多项式增长是两件不同的事。

---

## B. 固定资源的正 P 可达集合

约定：\(z=x+iy\)，\(d^2z=dx\,dy\)，\(P\) 是 \(\mathbb C\) 上的 Borel 概率测度（**不含** \(1/\pi\)）。相干态归一 \(\langle z|z\rangle=1\)，\(\langle z|a|z\rangle=z\)，\(\langle z|a^\dagger a|z\rangle=|z|^2\)。

Q 符号（不是 \(P\) 符号）

\[
f(z)=\langle z|\mathsf K_2^o|z\rangle.
\]

凡存在非负 \(P\) 使

\[
\rho=\int_{\mathbb C}P(z)\,|z\rangle\langle z|\,d^2z
\]

者，

\[
\mathcal J_{2,\mathrm{cl}}^o=\int_{\mathbb C}P(z)\,f(z)\,d^2z.
\]

资源约束下的经典可达集合

\[
\mathfrak C_{\alpha,n}
=\Bigl\{
\int P f
\;\Big|\;
P\ge 0,\;
\int P=1,\;
\int z\,P=\alpha,\;
\int |z|^2 P=n
\Bigr\}.
\]

当前数值：\(\alpha=0.5\)，\(n=0.34273260912113385\)。位移热态、任意相干态 \(|\alpha\rangle\)、以及 R1 反例 B 的三点离散测度都属于候选输入；它们给出 \(\mathfrak C_{\alpha,n}\) 中的**点**，不是 \(\mathfrak C_{\alpha,n}\) 本身。

因为 \(\int |z|^2 P=n<\infty\)，\(\{P\}\) 依 Markov 不等式在 \(\mathbb C\) 上紧：\(P(|z|>R)\le n/R^2\)。因此可达集的有界性不取决于测度逃逸到无穷，而取决于 \(f\) 是否关于这些测度一致可积。

- 精确 \(\mathsf K_2^o\) 有界 \(\Rightarrow |f|\le\|\mathsf K_2^o\|\) \(\Rightarrow \mathfrak C_{\alpha,n}\) 含于半径 \(\|\mathsf K_2^o\|\) 的圆盘。
- \(g\) 展开的截断多项式若次数 \(>2\)，则在同一 \(\mathfrak C\) 的定义测度类上**不必**一致可积（反例 B：\(\mathbb E|z|^4=v^2/p\to\infty\)）。

单模、非负 \(P\)、固定 \(\langle a\rangle,\langle n\rangle\) 的初等界（R1 已证，此处只引用）：

\[
|M|=\bigl|\langle(\Delta a)^2\rangle\bigr|\le v=0.09273260912113385,
\qquad
1\le\mathrm{Var}(X)\le 1+4v\approx 1.37093.
\]

压缩输入 \(|M|=0.31833\) 超出该集合。这是**输入资源**的非经典性，不是 \(\mathcal J_2^o\) 的经典界。

---

## C. 对偶二次上包络：证书构造，不是已有数值证书

对任意检测方向 \(\varphi\in\mathbb R\)，

\[
\sup_{J\in\mathfrak C_{\alpha,n}}\mathrm{Re}\bigl(e^{-i\varphi}J\bigr)
=\sup_P\int P(z)\,\mathrm{Re}\bigl(e^{-i\varphi}f(z)\bigr)\,d^2z.
\]

若存在实数 \(\lambda_0,\mu\) 与复数 \(\lambda\)，使对**一切** \(z\in\mathbb C\)

\[
\mathrm{Re}\bigl(e^{-i\varphi}f(z)\bigr)
\le
\lambda_0+2\mathrm{Re}(\lambda^* z)+\mu|z|^2,
\]

则资源约束立刻给出可检验的上界：

\[
\boxed{
\mathrm{Re}\bigl(e^{-i\varphi}\mathcal J_{2,\mathrm{cl}}^o\bigr)
\le
\lambda_0+2\mathrm{Re}(\lambda^*\alpha)+\mu n.
}
\]

这是对偶可行解，即**证书**：不需要 \(P\) 的最优，只要二次函数压住 \(\mathrm{Re}(e^{-i\varphi}f)\)。若不等式对某个 \(z\) 失败，该 \((\lambda_0,\lambda,\mu)\) 无效。紧界对应最佳二次上包络（广义矩问题的对偶）。

必要说明：

1. **一个正 P 热态不是完整优化。** 热态只给出 \(\mathfrak C_{\alpha,n}\) 中一点。\(\varepsilon_2=0.00171\) 是热态相对两压缩角等权混合的残差，既不是 \(\mathrm{supp}\,\mathfrak C_{\alpha,n}\) 的半径，也不是求积误差。
2. 若 \(\mathrm{Re}(e^{-i\varphi}f)\) 没有全局二次上界（例如扰动展开中的 \(|z|^3,|z|^4\) 项），本证书方法不能产生有限经典上界。这与反例 B 是同一逻辑。
3. 精确有界的 \(f\) 一定存在（平凡的）常数上界 \(\|\mathsf K_2^o\|\)，但该界对区分压缩输出毫无信息。要得到能与 \(\delta_\theta\mathcal J_2^o\sim 10^{-8}\) 比较的界，必须知道 \(f\) 在 \(|z|=O(1)\) 附近的真实形状。
4. Q-MATH-16.10 定理 B 针对**零均值、最低阶双光子相干、归一化图样**。不得套到本有限门、\(\alpha=0.5\)、绝对复电流 Fourier。定理 B 否决的是「用归一化 H2 相位图样宣称量子优势」的写法，不是本观测量的可达集计算。

**结论：** 证书的数学形式已经建立。数值证书不存在，因为 \(f(z)\) 尚未计算。

---

## D. \(n_{\max}=8\)：有限维优化不得升级为无限维正 P 界

当前 QED 的光子空间是九维。设 \(P_8=\sum_{n=0}^8|n\rangle\langle n|\)。

1. 无限维相干态 \(|z\rangle\) 满足 \(\|P_8|z\rangle\|^2=\sum_{n=0}^8 e^{-|z|^2}|z|^{2n}/n!\)。\(|z|\gtrsim 3\) 时该质量已明显泄漏到 \(n>8\)。反例 B 在 \(p\to 0\) 时恰好进入这一区域。
2. 截断位移 \(P_8 D(z)P_8\) **不是酉**。用截断相干态 \(\propto P_8|z\rangle\) 定义的 \(f_8(z)\) 与无限维 \(f(z)\) 在大振幅处不是同一函数。
3. 九维上 \(f_8(z)\) 在 \(|z|\to\infty\) 会饱和到顶层 Fock 矩阵元附近，因为截断相干态集中于 \(n=8\)。无限维 \(f(z)\) 则跟随大振幅驱动的真实联合动力学，二者不可对易极限。
4. 因此：在九维上对 \(\{P_8|z\rangle\}\) 做正 \(P\) 型优化，至多是**截断模型的经典界**。把它写成「全体无限维正 \(P\) 经典界」是错误的。

若未来要做 R2，必须先声明对象是截断 Hilbert 空间上的界，并单独估计 \(|z|\) 大时投影误差；本轮不执行该优化。

---

## E. \(g^2\) 磁序奇核：\(\Lambda\) 与 \(\widetilde\Lambda\)，以及不存在的统一余项界

### E.1 接触项在内的二阶结构

对 c-number 位移 \(A\)：

\[
H_e=H_0+AV+\tfrac12 A^2 W+\cdots,\qquad
j=j_0+A j_1+\tfrac12 A^2 j_2+\cdots,
\]

\(V=-j_0\cdot e\) 已在 R1 有限差分核验；\(\|j_2\|_F\neq 0\)。量子代入 \(A=g s(t)X\) 后，电流期望的 \(g^2\) 项包含三类贡献，缺一不可：

1. 电子密度矩阵的二阶 Liouville/Keldysh 演化（含 \(V,W\)）；
2. 一次接触 \(\mathrm{Tr}(j_1\rho^{(1)})\)；
3. 二次接触 \(\mathrm{Tr}(j_2\rho^{(0)})\)。

时间非局域核还要卷积耦合开关 \(s(t)\) 与光场二时刻关联。自由入射态（忽略反作用）给出

\[
C^S_\pi(t,t')-C^S_0(t,t')=2\sinh(2r)\cos[\omega_0(t+t')].
\]

联合 QED 中此式不是精确等式。结构上

\[
\delta\mathcal J_2^{o,(2)}=g^2\bigl(\Lambda M+\widetilde\Lambda M^*\bigr)+O(g^2\times\text{反作用修正}).
\]

θ=0,π 的 \(M\) 为实且反号，端点差只看见 \(\Lambda+\widetilde\Lambda\)。割线

\[
\Lambda_{\mathrm{eff}}=\frac{\mathcal J_2^o(\pi)-\mathcal J_2^o(0)}{M_\pi-M_0}
\]

**不是**微观 \(\chi^{(2)}\)。R1 账本（只读，不重算）：

\[
\Lambda_{\mathrm{eff}}^{\mathrm{GH5}}\approx(-1.496+0.460i)\times 10^{-7},
\qquad
\Lambda_{\mathrm{eff}}^{\mathrm{QED}}\approx(-1.490+0.514i)\times 10^{-7}.
\]

两点加一个 \(M=0\) 的热态，不能分开 \(\Lambda,\widetilde\Lambda\)，也不能排除在实 \(M\) 轴上与直线重合的高阶奇函数。

### E.2 统一余项界为什么一般不存在

沿相干输入 \(|z\rangle\)，Dyson 级数第 \(k\) 项含 \(k\) 个 \(X\) 插入，幅度 \(\sim (g|z|)^k\)。\(g^2\) 截断的余项在该射线的首项是 \(O(g^3|z|^3)\)。对 \(\mathfrak C_{\alpha,n}\) 取期望：反例 B 给出

\[
\mathbb E|z|^3 \sim v^{3/2}p^{-1/2}\to\infty\qquad(p\to 0).
\]

故：**不存在**仅以 \(\langle a\rangle,\langle n\rangle\) 为条件、对全体正 \(P\) 一致有效的 \(O(g^3)\) 小余项界。要使余项可证，必须附加至少一条：

- 支撑有界，例如 \(|z|\le R\) 且 \(gR\ll 1\)；
- 固定四阶矩或亚高斯尾；
- 限制在截断 Fock 上并接受那已不是无限维正 \(P\)。

精确 \(\mathsf K_2^o\) 的 \(f\) 有界，不依赖上述附加条件；**把 \(\mathsf K_2^o\) 换成 \(g^2\) 核再谈经典界，才需要这些条件。** 两条路线不可混写。

四阶统计的对照（解析，非传播）：\(\langle(\Delta X)^4\rangle_{\mathrm{th}}=3\cosh^2(2r)\approx 4.216\)，等权混合 \(\tfrac32(e^{-4r}+e^{4r})\approx 5.432\)，相对差 \(+28.8\%\)。GH5 的 H2 几乎重合，只说明**当前观测量对这一对高斯型 kurtosis 差不敏感**，不能把四阶响应系数定为零，也不能在无求积证书时从 \(R_2\sim 10^{-10}\) 提取该系数。

---

## F. Q-MATH-16.2 缺件；Q-MATH-15 不得套用

| 对象 | 本机状态 |
| --- | --- |
| `Q-MATH-16.2_真实八带M0二光子核_BZ磁性选择规则与噪声通道审计_20261022.zip` | **未找到**（Downloads / Documents / Desktop / `Documents\8带数学` / `Documents\量子光研究` / 仓库） |
| 16.2 是否在 16.11 包内 | 否；`inner_08.zip` 是 16.9 |
| `C:\Users\26507\Downloads\Q-MATH-15_纯数学纠错_经典正P全局证书_共享双k核验_20261022.zip` | 找到，515497 bytes，2026-10-23 11:09 |

Q-MATH-15 只读清单（未解压覆盖 `tools/`）：含 23-Fock 仿射证书、QMATH13 M0 H2/H4 kernel。协议是**纯 BSV、零均值、\(4\pi\) 绕数、单 \(k\)/H2 perp**。与本线 `Qin_a2_1_4_1`、\(\alpha=0.5\)、有限门位移电流 Fourier **不是同一观测量**。其中 \(|W_2|\le 0.098+0.25\bar n\) 一类仿射包络**不得**移到 \(\mathcal J_2^o\)。

16.2 接触核公式仅在指导书/16.10 中被引用。定量 \(K^{(2)}\) **未导出**。不得声称已复现 16.2。

---

## 7. 三个问题的直接答案

**当前正 P 可达集合能否被严格界定？**

作为集合：能。它就是 \(\mathfrak C_{\alpha,n}\)，定义不依赖弱场。作为 \(\mathbb C\) 中的具体区域：不能。缺少 \(f(z)\)。对偶二次证书给出**一旦 \(f\) 已知**如何证明上界的方法；本轮没有数值证书。热态残差不能冒充 \(\partial\mathfrak C_{\alpha,n}\)。

**已证什么？**

1. \(\mathcal J_2^o=\mathrm{Tr}(\rho_{\mathrm{in}}\mathsf K_2^o)\) 对冻结联合动力学精确线性。
2. 同资源正 P 的 \(|M|\le v\)、\(\mathrm{Var}(X)\in[1,1+4v]\)；压缩输入在其外。
3. Cauchy–Schwarz：\(\Pr(z\neq 0)\ge|\alpha|^2/n\approx 0.729\)；「大部分真空」不可能。
4. 反例 B：固定 \(\alpha,n\) 的对称稀有大涨落使一切 \(k\ge 3\) 的绝对矩无界；\(g^2\) 截断无统一余项界。
5. 精确 \(\mathsf K_2^o\) 有界 \(\Rightarrow\) 精确 \(f\) 有界 \(\Rightarrow\) 精确可达集有界；这不给出与 \(10^{-8}\) 可比较的半径。
6. θ=0,π 分不开 \(\Lambda,\widetilde\Lambda\)；\(\Lambda_{\mathrm{eff}}\) 是割线。
7. 16.10 定理 B 不覆盖本观测量；禁止用归一化图样宣称优势。
8. \(n_{\max}=8\) 的有限维优化不是无限维正 P 界。

**未证、必须依赖未来的 \(f(z)\) 或附加假设的是什么？**

1. \(\mathfrak C_{\alpha,n}\) 是否含有与 \(\mathcal J_2^o(\theta=0)\) 或 \(\theta=\pi\) 重合的点（经典可复制性）。
2. 最佳二次包络 \((\lambda_0,\lambda,\mu)\) 的存在与数值。
3. 反常关联是否以微观核 \(\Lambda,\widetilde\Lambda\) 主导 H2，高阶余项是否在**实际被使用的态**（而非全体正 P）上可忽略。
4. 截断 \(f_8\) 与无限维 \(f\) 在 \(|z|\sim 1\) 处的差是否小于差向量的 3%。
5. 输出非经典性。输入非经典性已经成立，二者不可互换。

**当前数据是否支持近似仿射的 \(M\) 响应？** 相容，未证明。热态与 mix 的 \(\varepsilon_2=0.00171\) 支持低阶统计对**这一对态**足够；四阶差 28.8% 仍未给出四阶系数的严格界。

**能否从八带对称性与响应核推导出 H2 磁序奇分量？** 弱场偶次宇称：是候选。有限门定量核：否。16.2 源码本机未找到。

**距离严格经典负控还差什么？** 差 \(f(z)\)、差可比较的对偶证书、差截断延拓、差 \(g^2\) 余项在实际态族（而非反例 B 的无界矩族）上的控制。证书方法已经写下来；数值证书没有。

---

## 8. 停机声明

本轮只做了三件事：订正 R1 三处；写出 \(\mathsf K_2^o\)、\(\mathfrak C_{\alpha,n}\) 与对偶二次证书的定义与限度；记录 16.2 缺件与 Q-MATH-15 的协议隔离。

**不声称：** 已认证 HHG、输出纠缠、量子优势、纯 BSV、\(\Lambda_{\mathrm{eff}}=\chi^{(2)}\)、热态残差=经典界、有限维优化=无限维正 P 界、16.2 已复现。

**不执行：** R2、任何新传播、压缩角扫描、\(n_{\max}\) 升级、后窗续算、服务器。

若用户将来另行批准，唯一与本文衔接的数值动作是：在**事先声明截断对象**的前提下计算若干 \(f(z)\) 并尝试二次包络证书。那是新的批准项，不是本文件的作业。

**停机。**
