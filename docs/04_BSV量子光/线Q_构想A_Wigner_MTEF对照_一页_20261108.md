# 构想 A：Wigner/MTEF 半经典对照（一页）

日期：2026-11-08。外审已验收 20261107：结束 \(J_{\mathrm{odd}}\) 数值精度战役。本页只建立**同一冻结 Hamiltonian** 的相空间对应、初态、验证策略与成本。不启动完整轨迹集合，不改 h8，不重开真空/G1/H2，不跑 Cn、压缩角、nmax、HHG。

冻结对象：共享 \(k,-k\) 单模，\(g=0.005\,\mathrm{\AA}^{-1}\)，\(\omega_0=0.05\,t_h\)，\(k=(0.173,0.271)\)，偏振 \(a_2\)，一周期 \(\sin^2\) 升沿（无平台），电子 Slater 与 20261102 相同。光子约定仍是 \(D(\alpha)S(\xi)|0\rangle\)，\(\alpha=0.5\)，\(r=0.3\)，\(\theta=0\)。

本页不是半经典已经重现 \(\Delta J_{\mathrm{odd}}\)，也不是量子优势。

## 精确侧（已冻结，不重算）

\[
\hat H(s)=\omega_0\,\hat n+\hat H_e(k,s\hat X;N)+\hat H_e(-k,s\hat X;N),\qquad \hat X=\hat a+\hat a^\dagger.
\]

Peierls：键按 \(q=g\,\mathbf{e}\cdot\boldsymbol{\delta}\) 分组，\(F_q\otimes\exp(i q s\hat X)\)。有限 Fock 用的是无穷空间位移的 Galerkin，不是 \(\exp(i q\hat X_{\mathrm{trunc}})\)。电流与 \(H\) 同一套键、同一个 \(s\)。

线性反作用恒等式（静态已核）：\(F_k=-g\,\mathbf{e}\cdot\langle\mathbf{J}\rangle\)。平衡 \(k,-k\) 的**线性平均**驱动已消掉；单条有限 \(X\) 轨迹上 \(\langle\mathbf{J}\rangle\) 不必为零。

## 相空间与 Weyl 排序

Peierls 的 \(\hat X\) 真空方差是 1，不是 20261030 里 \(W_{\mathrm{sq}}\) 用的 \((ae^{-i\theta}+a^\dagger e^{i\theta})/2\)（真空方差 \(1/4\)）。两套坐标禁止混用。

\[
\hat X=\hat a+\hat a^\dagger,\qquad \hat P=-i(\hat a-\hat a^\dagger),\qquad [\hat X,\hat P]=2i.
\]

Weyl 符号：\(\hat n\;\longleftrightarrow\; n_W=(X^2+P^2-2)/4\)。于是自由光场

\[
H_{\mathrm{ph}}=\frac{\omega_0}{4}(X^2+P^2-2),\qquad \dot X=\omega_0 P,\qquad \dot P=-\omega_0 X.
\]

\(\exp(i q s\hat X)\) 只是 \(\hat X\) 的函数，Weyl 符号就是 \(\exp(i q s X)\)。MTEF 的近似是：丢掉 Moyal 的 \(\hbar^2\) 项，把电子 Hamiltonian 直接钉在经典 \(X\) 上。这不是把初始 \(\alpha\) 当成固定外场。

## 初态 Wigner（正高斯，无负区）

\(D(\alpha)S(re^{i\theta})|0\rangle\) 的 Wigner 是正高斯。椭圆方向 \(\theta/2\)。\(\theta=0\) 压缩的是 Peierls 的 \(X\)：

\[
\langle X\rangle=2\mathrm{Re}\,\alpha,\quad\langle P\rangle=2\mathrm{Im}\,\alpha,\quad
\mathrm{Var}(X)=e^{-2r},\quad\mathrm{Var}(P)=e^{2r}.
\]

本轮数：\(\langle a\rangle=0.5\) 两者相同；\(\langle n\rangle=0.250\)（Cα）对 \(0.343\)（DS）；\(\mathrm{Var}(X)=1\) 对 \(e^{-0.6}\approx0.5488\)。

**负对照。** 只取 Wigner 均值的一条 Ehrenfest 轨迹，Cα 与 DS 的初始 \((X,P)\) 完全相同，因而整条平均场轨迹相同，\(\Delta J_{\mathrm{odd}}^{\mathrm{MF}}=0\)。若精确量子的 \(\Delta J_{\mathrm{odd}}\) 非零，半经典要能解释它，**必须**靠协方差不同，而不是靠 \(\langle a\rangle\)。

## 自洽 Ehrenfest（一条轨迹）

电子在经典 \(X(t)\) 上保持直积：\(\lvert\psi_k\rangle\otimes\lvert\psi_{-k}\rangle\)（36+36，不是 1296，更不是 11664）。

\[
i\partial_t\lvert\psi_\ell\rangle=\Bigl[\mathrm{diag}(\varepsilon_\ell)+\sum_q F_q^{(\ell)}\,e^{i q s(t) X}\Bigr]\lvert\psi_\ell\rangle,
\]

\[
\dot X=\omega_0 P,\qquad
\dot P=-\omega_0 X+2\,s(t)\,g\,\mathbf{e}\cdot\langle\mathbf{J}\rangle_{\psi,X,s}.
\]

第二式里的 \(2\) 来自 \([\hat X,\hat P]=2i\)，不是任意系数。\(\langle\mathbf{J}\rangle\) 用与量子相同的 Peierls 电流，钉在当前 \(X\)。禁止：抽一次初始电场后冻结；禁止：两条独立光子。

分裂积分（本轮单轨迹采用）：中点 \(s\)，自由振子转 \(\mathrm{d}t/2\)，电子 \(\mathrm{expm}\) 一步并踢 \(P\leftarrow P+\mathrm{d}t\cdot 2sg\,\mathbf{e}\cdot\langle J\rangle\)，再转 \(\mathrm{d}t/2\)。

升沿长度恰好一个光周期：无反作用时 \((X,P)\) 必须回到初值。均值 Ehrenfest 跑完后 \(\Delta P\sim-0.025\)，这就是反作用，不是步长误差。升沿开头两步的踢几乎为 0，因为 \(s=0\) 处 \(k,-k\) 平均电流已被消掉。

均值点上的 Weyl 光子数是 \(n_W(\langle X\rangle,\langle P\rangle)=|\alpha|^2-1/2=-0.25\)，**不是** \(\langle n\rangle=0.25\)。真空宽度要靠 \(m\ge2\) 的节点平均才能回来。不要把单条均值轨迹的负 \(n_W\) 读成积分器坏了。

## \(J_{\mathrm{odd}}\) 与配对求积

观测量与 20261102 相同：\(J_{\mathrm{even/odd}}=[J_{\pm N}\pm J_{\mp N}]/2\)，\(\Delta J_{\mathrm{odd}}=J_{\mathrm{odd}}^{\mathrm{DS}}-J_{\mathrm{odd}}^{\mathrm{C}\alpha}\)。信号 \(\sim10^{-7}\)，独立蒙特卡洛会淹没它。

默认：**标准化平面上的二维 Gauss–Hermite**，仿射到各态协方差。同一 \((\xi,\eta)\) 同时用于 Cα、DS、\(\pm N\)（共同随机数 / 共同节点）。\(\pm N\) 只共享**初值** \((X,P)\)；此后电子与反作用场各自演化。

| 阶 \(m\) | 节点 | 角色 |
|---|---|---|
| 1 | 1 | 平均场负对照，\(\Delta J_{\mathrm{odd}}\) 应 ≈ 0 |
| 3 | 9 | 第一次带涨落的公平对照 |
| 5 / 7 | 25 / 49 | 求积收敛 |

阶 \(m\) 对多项式精确到 \(2m-1\)；方差是 2 次，\(m\ge2\) 已能积准高斯权重下的 \(X^2\)。本轮**只构造节点并核矩，不传播 \(m\ge3\)**。

Cn（同 \(\langle n\rangle\) 相干）仍留作资源匹配，本轮不跑。

## 若半经典与精确不同

先查：求积阶、共同节点是否装对、Weyl/\(X\) 约定是否混用、反作用踢是否关掉、时间网格是否与 20261102 的 \(t=(j+1)\mathrm{d}t\)（\(j\bmod 32=0\) 加终点）对齐。禁止把差额直接写成量子优势或电子–光场纠缠。正 Wigner 没有负区，差不能预先归因于某一种机制。

## 本轮交付与明确不跑

交付：本页方程；静态矩 / Weyl / \(F_k=-g\mathbf{e}\cdot J\)；Gauss–Hermite 节点矩；**一条**均值 Ehrenfest 升沿；成本外推。停机。

不跑：完整节点集、Cn、\(\theta\) 扫描、nmax、真空、G1/H2、HHG、输出光。
