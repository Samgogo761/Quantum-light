# Q-MATH-16.18｜八带 M0 精确 Keldysh 行列式、连通电子响应与磁性量子矩逆问题

**研究日期：2026-10-24**　　**模型：**现象学八带 PT-AFM M0（R0；不是 CrI₃ Wannier 模型）

**证据等级：**严格外代数恒等式 + 原始生产版 M0 与 36 维酉演化对照 + 有限时间因果电子响应扫描 + 一个完全标记为 *synthetic* 的量子光矩读出逆问题。**本轮没有构造完成的光子 LSZ/Maxwell 连通五光子 S 矩阵，也没有认证材料特有的量子 HHG 新机制。**

## 一、问题从何而来：把材料响应与光子统计分开

此前 Q-MATH-16.13—16.17 已建立单一占据入射波包的有效发射算符结构

$$
A_N(\Omega)=\eta S_{0,N}(\Omega)a_u^2+\eta^2\big[S_{10,N}(\Omega)a_u^2+S_{11,N}(\Omega)a_u^\dagger a_u^3\big]+O(\eta^3),
$$

其中 $\eta$ 表示泵浦耦合的平方，另一个谐波输出耦合的共同因子已省略。先前的 $S_{11,N}$ 来源于带展宽的裸五光子 $T$ 系数，受外腿回返、能量壳与 BZ 采样影响。因而「$S_{11}\ne0$」尚未在物理连通散射层次成立。

低平均光子数 $E=0.1$ 的单模纯 BSV 有

$$
F_1=0.1,\quad F_2=3E^2+E=0.13,\quad F_3=15E^3+9E^2=0.105,
\qquad EF_3-F_2^2=-0.0064.
$$

任何正 Glauber–Sudarshan $P$ 态都要求 $EF_3-F_2^2\ge0$，这是标准 Stieltjes/Agarwal–Tara 型非经典性见证。研究的**新问题**不是重新发现该不等式，而是：磁性 M0 的散射能否构成一个稳定、可标定、仅依赖同一 BSV 输入波包的实际测量映射，将 $F_2,F_3$ 独立取出？

## 二、严格定理 A：M0 的电子 Keldysh 影响泛函是占据子空间行列式

M0 的 Peierls 耦合在 c-number 电磁历史 $\mathbf A(t)$ 下保持单电子二次型：

$$
\widehat H_{e,N}[\mathbf A(t)]
=\sum_{\mathbf k,ab}c^\dagger_{\mathbf k a}\,h_{N,ab}(\mathbf k+\mathbf A(t))\,c_{\mathbf k b}.
$$

每个 $\mathbf k$，设 $B_{\mathbf k,N}\in\mathbb C^{8\times 4}$ 的正交列为半填充的四个占据轨道。定义两个 Keldysh 分支的**单粒子**演化 $U_\pm$：

$$
i\dot U_\pm(t)=h_N(\mathbf k+\mathbf A_\pm(t))U_\pm(t),\qquad U_\pm(t_i)=\mathbb 1.
$$

由 Slater 外代数的重叠恒等式，严格有

$$
\boxed{\mathcal Z_{\mathbf k,N}[A_+,A_-]
=\det\big(B^\dagger_{\mathbf k,N}U_-^\dagger U_+B_{\mathbf k,N}\big).}
\tag{18.1}
$$

对整个独立动量扇区 $\mathcal Z_N=\prod_{\mathbf k}\mathcal Z_{\mathbf k,N}$，因而 $\log\mathcal Z_N=\sum_{\mathbf k}\log\mathcal Z_{\mathbf k,N}$。M0 还严格守恒 $s_z$，所以每个 $\mathbf k$ 的四维占据行列式进一步分解为两个 $2\times2$ 自旋子行列式之积。

**证明要点。** 在费米外代数 $\wedge^4\mathbb C^8$ 中，初态是 $b_1\wedge\cdots\wedge b_4$；二次型 Hamiltonian 诱导的多体酉变换恰是 $\wedge^4 U$。两个 Slater 态的内积等于单粒子重叠矩阵的行列式，得到式 (18.1)。这属于标准 Slater/Klich 行列式技术在当前 M0 上的精确实现，**不是新数学定理**。

取 $A_{\mathrm{cl}}=(A_++A_-)/2$、$A_q=A_+-A_-$。在 $A_q=0$ 时，$\mathcal Z=1$；利用函数导数 $\delta\log\det X=\operatorname{Tr}(X^{-1}\delta X)$，有

$$
\frac{\delta\log\mathcal Z_N}{\delta A_q(t)}\bigg|_{A_q=0}
=i\langle j_N(t)\rangle_{A_{\mathrm{cl}}},\qquad
j_N(t)=-\frac{\partial H_{e,N}[A_{\mathrm{cl}}(t)]}{\partial A_{\mathrm{cl}}(t)}.
\tag{18.2}
$$

再对 $A_{\mathrm{cl}}$ 求导即得到因果非线性电子电流响应；若同时对多个 $A_q$ 求导，则生成 Keldysh 连通涨落。因为始终使用完整的 $h_N(\mathbf k+\mathbf A)$，Peierls 混合导数与接触顶点不会被遗漏。

**重要限定：** $\log\mathcal Z$ 消除的是费米子连通展开中的电子断开项；它**不是**自动剔除外部光子自能和回返项的 LSZ 手续。将 $\mathbf A$ 当成真正量子泵浦及谐波输出，还需在 Keldysh 双分支上引入 BSV 高斯光子作用量、正反时间关联、输入输出边界和 Maxwell 端口。对经典电流响应的正 $P$ 平均，不能一般地代替非线性量子光响应，这一点与 Mukamel–Dorfman 的量子非线性光谱理论一致。

## 三、真实 M0 的 36 维独立交叉验证

使用附包中原始 `q167_m0_three_photon_T.py`、`q8_m0_model.py`、真实 M0 几何文件；在两个一般动量/磁畴上施加相互独立的有限时间 Peierls 轨迹，分别求

1. 八带 $U_\pm$ 后的自旋因子化行列式；
2. 完整 $36\times36$ 四电子 Hamiltonian 的每一步精确指数传播，再求 $\langle G|U_-^\dag U_+|G\rangle$。

**实际最大复振幅残差：** $2.2887834\times10^{-15}$。相同正反分支 $A_+=A_-$ 时，$\mathcal Z=1$ 的最大残差 $7.55\times10^{-15}$。

式 (18.2) 还通过独立电流传播检查。采用分段常数时间步，Keldysh 行列式对 $A_q$ 的导数与简单左端点电流求和的差异，在 35、70、140 时间步下依次为 $0.02620,0.01306,0.00652$（同一物理时间区间和一组指定测试场）。误差按 $\Delta t$ 近似减半，原因是左端点电流并不是分段常数演化算符的精确时间积分，**不是式 (18.2) 失效**。

## 四、固定同一输入历史时，真实 M0 的二阶与四阶因果电流响应

令 $A_{\mathrm{cl}}(t)=x f_p(t)\hat e_p$，用同一个 $f_p$ 激发 M0；选择横向电流的两个探测函数 $g_1(t)$、$g_2(t)$，分别近似正弦/余弦分量。定义

$$
C_{2,N}^{(j)}=\frac{d^2}{dx^2}\int dt\,g_j(t)\langle j_{\perp,N}(t;x)\rangle\bigg|_{x=0},
\qquad
C_{4,N}^{(j)}=\frac{d^4}{dx^4}\int dt\,g_j(t)\langle j_{\perp,N}(t;x)\rangle\bigg|_{x=0}.
$$

这些是**真实 M0 的有限时间经典外源因果电子响应**，不是已经量子化输出的 $S_0,S_1$。定义两响应向量的规范化非共线度

$$
\delta_{24}=\frac{|\det\mathcal R|}{\|C_2\|_2\|C_4\|_2},\qquad
\mathcal R=\begin{pmatrix}C_2^{(1)}&C_4^{(1)}\\ C_2^{(2)}&C_4^{(2)}\end{pmatrix}.
\tag{18.3}
$$

实际计算采用两个固定探测时间形状、$\lambda_{SO}=0.1,\Delta=0.8$，得到：

| 半格偏移 BZ | $\delta_{24}$ | 原始列矩阵条件数 | 说明 |
|---|---:|---:|---|
| $3\times3$ | 0.04693 | 1598 | 有限网格秩二 |
| $5\times5$ | 0.04670 | 1766 | 接近共线 |
| $7\times7$ | 0.03711 | 2189 | 尚未动量收敛 |

在 $5\times5$，将有限差分步长从 $0.12,0.08$ 减为 $0.05$，$\delta_{24}$ 分别约为 $0.04386,0.04670,0.04791$。禁戒负控：$\lambda=0$ 或 $\Delta=0$ 的 $C_2,C_4$ 均在舍入/有限差分噪声量级；$N=\pm1$ 两磁畴响应相互翻号，数值残余小于 $3\times10^{-9}$（在显示的测试范围）。

![真实 M0 有限时间二/四阶因果电子响应；不是 H2 量子产额。](qmath_16_18_causal_M0_BZ.png)

**判断：** 这提供了一个材料内部高阶响应时间结构的候选非比例性，但测量反演的条件数很大且 $C_4$ 未完成 BZ/差分误差界；不能把 $\delta_{24}>0$ 解释为已经独立测出 $F_3$。尤其，来自经典外源的 $C_4$ 与真量子五光子连通核 $S_{11}$ 之间还缺少严格的光子 Keldysh 收缩和归一化联系。

## 五、新的协议层次数学修正：谱核满秩并非唯一可辨识方式

先假设有一个**经过物理散射认证**的单模泵浦有效发射算符

$$
\mathcal A(\eta)=\eta\left[S_0O_0+\eta(S_{10}O_0+S_{11}O_1)\right]+O(\eta^3),\qquad
O_0=a^2,\quad O_1=a^\dagger a^3.
$$

对输入光泵浦迹掉并只看一个 H2 输出端口的领先单发射光子计数：

$$
\frac{n_H(\eta)}{\eta^2}
=A_0F_2+\eta\big[B_0F_2+B_1F_3\big]+O(\eta^2),
\quad A_0=|S_0|^2,\ B_0=2\operatorname{Re}(S_0^*S_{10}),\ B_1=2\operatorname{Re}(S_0^*S_{11}).
\tag{18.4}
$$

若能够**物理上改变 $\eta$ 而保持 $S_0,S_{10},S_{11}$、入射波包 $u$ 和入射态不变**，那么同一输出端口在 $\eta_1\ne\eta_2$ 的两次归一化读数，其对 $(F_2,F_3)$ 的线性化测量矩阵为

$$
\mathcal J=\begin{pmatrix}
A_0+\eta_1B_0 & \eta_1B_1\\
A_0+\eta_2B_0 & \eta_2B_1
\end{pmatrix},\qquad
\boxed{\det\mathcal J=A_0B_1(\eta_2-\eta_1).}
\tag{18.5}
$$

**严格推论：** 只要 $S_0\ne0$、$B_1\ne0$ 且 $\eta_1\ne\eta_2$，即使**两个材料输出谱向量严格共线**，完整协议的测量矩阵仍可以满秩。因此，前几轮要求“$S_0$ 与 $S_1$ 谱形非共线”作为所有方案的必需条件过强；它只是在**固定一个耦合条件、仅依靠多个输出谱点**进行逆问题时的结构要求。

此结论存在三个严肃限制：① $B_1=0$（纯正交复相位）时，该强度读数对 $F_3$ 没有线性灵敏度；② $\eta\to0$ 时 $\mathcal J$ 虽数学满秩却严重病态，噪声迅速放大；③ 在固体实验中独立调节光子—材料耦合而又不改变输入波包、电子动力学和输出端口，尚未建立，因此这里是**有条件的数学协议**，不是现成实验方案。

### 完全标记为合成的逆问题单元测试

取 $S_0=1+0.3i,S_{10}=0.2-0.1i,S_{11}=0.35+0.15i$，选择 $E=0.1$ 的 BSV。上述系数**不是 M0 散射结果**。把四个 $\eta=0.001,0.002,0.003,0.004$ 的计数按 $\eta^2$ 归一化，拟合截距和一次项，并允许二次项吸收 $F_4$ 造成的有限耦合修正，得到

$$
F_2^{\mathrm{fit}}=0.1300000000000001,\qquad
F_3^{\mathrm{fit}}=0.104999999999964,
$$

与解析值一致。线性化矩阵条件数约为 **1236**，说明该协议尽管满秩，数值可辨识性仍差。

在相同 $E=0.1,F_2=0.13$ 的经典正 $P$ 输入中，$F_3\ge F_2^2/E=0.169$；取真空/亮相干态混合即可达到该下界。合成计数在 $\eta=0.005,0.01,0.02$ 时，经典下界态相对 BSV 分别相差约 **0.178%、0.355%、0.707%**。实验要可靠区分，就必须控制相应量级的统计噪声与所有材料标定误差；不能只报告理想无噪声拟合成功。

![合成的耦合扫描读出测试与 M0 二/四阶电子响应差分敏感性。](qmath_16_18_inverse_and_rank.png)

## 六、与现有数学及物理文献的具体联系

**费米子行列式与统计生成泛函。** I. Klich, *Full Counting Statistics: An elementary derivation of Levitov's formula* (2002), https://arxiv.org/abs/cond-mat/0209642。式 (18.1) 是相关自由费米子外代数思想在固定占据 M0 上的显式实现。其作用是获得全部电子连通因果核；不等于求出量子化的多光子远场输出。

**多光子散射连通分解。** S. Xu and S. Fan, *Input-output formalism for few-photon transport: A systematic treatment beyond two photons*, Phys. Rev. A **91**, 043845 (2015), https://doi.org/10.1103/PhysRevA.91.043845。其波导/局域体系的 S 矩阵—连通格林函数原理很有帮助，但不能不加修改直接用于二维周期电子、完整 Peierls 顶点和自由空间 Maxwell 模。

**量子光不能简单取经典平均。** S. Mukamel and K. E. Dorfman, *Nonlinear fluctuations and dissipation in matter revealed by quantum light*, Phys. Rev. A **91**, 053844 (2015), https://doi.org/10.1103/PhysRevA.91.053844；K. E. Dorfman, F. Schlawin and S. Mukamel, *Nonlinear optical signals and spectroscopy with quantum light*, Rev. Mod. Phys. **88**, 045008 (2016), https://doi.org/10.1103/RevModPhys.88.045008。强调量子光非线性响应必须保留不同 Keldysh 分支上的量子关联与物质涨落，不可把经典电流四阶导数直接命名为五光子量子输出系数。

**正 $P$ 非经典性与矩问题。** G. S. Agarwal and K. Tara, *Nonclassical character of states exhibiting no squeezing or sub-Poissonian statistics*, Phys. Rev. A **46**, 485 (1992), https://doi.org/10.1103/PhysRevA.46.485。此前 Q16.17 中的 $EF_3-F_2^2$ 是现成矩界的应用。

**已有量子 HHG 与禁戒谐波工作。** Rasputnyi *et al.*, *High harmonic generation by a bright squeezed vacuum*, Nature Physics **20**, 1960–1965 (2024), https://doi.org/10.1038/s41567-024-02659-x；Rivera-Dean *et al.*, *Structured squeezed light allows for high-harmonic generation in classical forbidden geometries*, PRL **135**, 013801 (2025), https://doi.org/10.1103/4hdl-bdwj；Prabhu and Dixit, *Sensing quantum fluctuations via forbidden harmonics in quantum materials*, PRB **114**, L140301 (2026), https://doi.org/10.1103/v886-ypf7。故“压缩光改变 HHG / 激活禁戒谐波 / AFM 允许偶次非线性响应”都不构成独立新颖性。

## 七、Q-MATH-16.19：现在必须检验的真正物理对象

可延伸成以下严谨路径，而不是继续机械增加裸五、七、九光子阶数：

1. **完整 Gaussian-BSV 光子 Keldysh 积分。** 把式 (18.1) 放入两分支光子路径积分/相干态泛函，严格区分光子的因果、Keldysh 和异常两光子关联，生成量子入射泵浦—电子联合响应。不能默认用 BSV 的正 Wigner 分布做经典场平均就等价。
2. **物理输出端口。** 接上 Maxwell 辐射正频模式，分别计算 $\langle b_{\mathrm{out}}\rangle,\langle b^\dagger_{\mathrm{out}}b_{\mathrm{out}}\rangle$ 和需要的多时刻条件关联；剥离输出真空与着装态背景，完成 input–output 归一化。
3. **在固定同一波包下构造真实读出雅可比。** 先取得物理 $S_{0,N},S_{10,N},S_{11,N}$，再比较“多频输出读数”“同波包可调耦合”“输出相位参考”三类协议的 **完整** Fisher 信息/雅可比最小奇异值，而不只看材料核 Gram 秩。
4. **磁性独有性。** $N\mapsto-N$、$\lambda\mapsto0$、$\Delta\mapsto0$、非磁性但能隙匹配的控制、同一物理端口与探测窗负控都需通过。既有奇磁畴 SHG 或通用正 $P$ 界本身不是新材料机制。
5. **数值误差证书。** 当前 $C_2,C_4$ 对 BZ 采样有显著变化，$C_4$ 对差分步长也有敏感性；必须完成大网格、时间步、高阶 Peierls 导数和光子输出积分的独立收敛及误差预算。没有这些，非共线指标不应升格为材料预测。

**最终科研判断：** 本轮从完整 M0 确立了不依赖电子 Hilbert 截断的精确连通电子生成接口，并修正了“只有频谱满秩才可读出 $F_3$”的过强论断。有限时间电子响应显示候选非比例性，但病态且尚未收敛。**尚无验证通过的反铁磁 × 纯量子光 HHG 特有机制。**

## 复现说明

最小环境：Python 3、NumPy、SciPy；图示额外使用 Matplotlib。

```bash
# 在 Q-MATH-16.18 根目录运行
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python qmath_16_18_keldysh_determinant.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python qmath_16_18_causal_rank_and_inverse.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python qmath_16_18_compile_results.py
```

第一、二脚本均包含生产版 M0 的实际时间传播，计算负载取决于 BZ 网格；第三脚本汇总保存的 JSON 与出图。输入源代码位于 `work/src_0`，源于用户提供的 Q-MATH-16.7 ZIP，并非仿造哈密顿量。**合成逆问题测试是纯数学单元测试，必须与真实 M0 时间响应分开引用。**
