# Q-MATH-16.17｜截断 Stieltjes 矩问题、有限阶正 P 完整反例与 M0 磁性量子光读出路线

**研究日期：2026-10-24**  
**证据性质：** 严格单模算符定理、显式有限离散正 P 反例、独立数值验证、数学及现有量子 HHG 文献审计。  
**关键限制：** 此次不是新的一次真实 M0 五/七光子 *connected* S-matrix 计算。M0 原始三/五阶材料系数仍有外腿回返、展宽、BZ 和 Maxwell 输出未完成的门槛。不能用本报告中的普适光学不等式取代材料专项计算。

## 1. 为什么需要从头审视“高 Fock 带来新材料物理”

此前 Q-MATH-16.13/14 已将单一泵浦波包的有限阶二光子耗散—谐波发射算符正规序化为

\[
A_N=\sum_{p=0}^{2}S_{p,N}\,O_p,\qquad O_p=(a_u^\dagger)^p a_u^{p+2}.
\tag{1}
\]

这里 $p=0,1,2$ 分别来自三、五、七个光子顶点（计入发射光子）；$S_{p,N}$ 可为输出频率、电子最终态、偏振和端口上的复向量/核。已有原始 M0 计算确认有限阶 Fock 多项式结构，但这些 $S_{p,N}$ **尚不是已剔除外腿回返的连通物理散射矩阵**。一个最基本的逻辑问题是：即便假定它们已算准且线性独立，输出的哪些关联必然超过所有经典光？本报告回答这一问题。

## 2. 量子光学与 Stieltjes 矩问题的精确对接

令 $E=\langle a_u^\dagger a_u\rangle$，并定义阶乘矩

\[
F_k=\langle(a_u^\dagger)^k a_u^k\rangle,\qquad F_0=1,\ F_1=E.
\tag{2}
\]

对非负 Glauber–Sudarshan $P$ 的任意**经典**单模光，$I=|\alpha|^2\ge0$ 是普通随机强度，故 $F_k=\mathbb E[I^k]$。因此，对任何非负强度上的多项式 $q(I)$，必有

\[
\mathbb E[I q(I)^2]\ge0.
\tag{3}
\]

这等价于每个有限阶 **shifted Hankel** 矩阵的半正定必要条件

\[
H_d^{(1)}=[F_{i+j+1}]_{i,j=0}^{d}\succeq0.\tag{4}
\]

普通 Hankel $H_d^{(0)}=[F_{i+j}]\succeq0$ 也是必要条件。这里使用的是成熟的 Stieltjes 矩问题与 Agarwal–Tara / Shchukin–Vogel 光学非经典矩判据，而非宣称发明了一套新的数学工具。[1–4]

零位移纯单模 BSV 的光子数概率生成函数可由偶数 Fock 分布精确求和：

\[
G(z)=\frac{1}{\sqrt{1+E(1-z^2)}},\qquad F_k=\left.\frac{d^kG}{dz^k}\right|_{z=1}.
\tag{5}
\]

由此得到任意阶递推关系

\[
\boxed{F_{k+1}=E(2k+1)F_k+E k^2F_{k-1}}.\tag{6}
\]

特别地，$F_2=E(3E+1)$，$F_3=3E^2(5E+3)$，$F_4=3E^2(35E^2+30E+3)$。生成函数和递推式同时给出了不依赖有限 Fock 截断的可靠检验。

## 3. 第一个有用的严格界：低占据只需要 $F_3$

取式 (4) 的二阶矩阵：

\[
H_1^{(1)}=\begin{pmatrix}F_1&F_2\\F_2&F_3\end{pmatrix},\qquad
\det H_1^{(1)}=F_1F_3-F_2^2.
\tag{7}
\]

经典正 $P$ 必须有 $F_1 F_3\ge F_2^2$。纯 BSV 则精确给出

\[
\boxed{F_1F_3-F_2^2=E^2(6E^2+3E-1)}.\tag{8}
\]

故当 $0<E<E_*=({\sqrt{33}-3})/12=0.228713553878\ldots$ 时违背所有非负 $P$ 经典强度混合。数值如 $E=0.1$ 时，$F_1=0.1,F_2=0.13,F_3=0.105$，因此 $F_1F_3-F_2^2=-0.0064$。

它可以重写为一个实验友好的非负多项式禁戒：经典态必满足

\[
\big\langle I(I-F_2/F_1)^2\big\rangle\ge0,
\tag{9}
\]

左边在低 $E$ 纯 BSV 中反而为负。这个判据属于已知的矩层级非经典性框架，**不是新的普适光学理论**。

## 4. 更强的严格负结论：$E=0.7$ 时，经典态能匹配全部 $F_0\ldots F_6$

这一数字与我们此前反复使用的 BSV 资源点一致。纯 BSV 的前八个阶乘矩如下：

| $k$ | $F_k$ |
|---:|---:|
| 0 | 1 |
| 1 | 0.700000000 |
| 2 | 2.170000000 |
| 3 | 9.555000000 |
| 4 | 60.490500000 |
| 5 | 488.106150000 |
| 6 | 4817.001105000 |
| 7 | 56134.985035500 |

**构造性反例。** 取真空加三个正强度相干态的混合。离散强度 $I_j$ 与概率 $p_j$ 为：

| $j$ | $I_j$ | $p_j$ |
|---:|---:|---:|
| 0 | 0 | 0.726724827165 |
| 1 | 2.315881892590 | 0.261765387132 |
| 2 | 8.137367724303 | 0.011501957732 |
| 3 | 23.840177520763 | 0.000007827971 |

表中小数为方便排版的近似值，完整计算精度见 JSON。对于三个非零强度，各取一半概率的 $|+\sqrt{I_j}\rangle$ 与 $|-\sqrt{I_j}\rangle$；真空占 $p_0$。因此这个光态有**严格非负 $P$**、零平均电场、和 BSV 完全相同的唯一占据频谱波包 $u$，并保持相同的 $E$ 与总自由光能量。其强度矩 $\sum_j p_j I_j^k$ 在 $k=0,\ldots,6$ 与 BSV 一致，最大相对偏差 $1.5\times10^{-14}$。这个光态**没有**匹配所有 Fock 分布与相干异常矩，因此不推出无限阶完全经典模拟。

**构造算法不是黑箱拟合。** 对倾斜强度测度 $I\,d\mu(I)$，从 $[F_{i+j+1}]_{i,j=0}^2$ 和 $[F_{i+j+2}]_{i,j=0}^2$ 的广义特征值求三个正 Gauss 节点，再由 $F_1,F_2,F_3$ 求倾斜权重；将倾斜权重除以节点得到原强度概率，剩余概率放在 $I=0$。这是截断 Stieltjes/Gauss–Radau 构造，且所有概率都经过显式非负检验。

## 5. 严格的单发射扇区经典模拟定理（直到七光子振幅阶）

令

\[
O_p=(a^\dagger)^p a^{p+2},\qquad p=0,1,2,
\quad A_j=\sum_{p=0}^2 S_{jp}O_p,
\tag{10}
\]

其中 $j$ 是完整全局电子最终态和一个输出谐波光子的正交标签，包括输出频率及模式。物理可观察的领先单发射扇区密度矩阵为

\[
\rho^{(1)}_{jl}\propto\operatorname{Tr}_{\mathrm{pump}}\big(A_j\rho_u A_l^\dagger\big)
=\sum_{p,q}S_{jp}G_{pq}S_{lq}^*,\quad G_{pq}=\langle O_q^\dagger O_p\rangle.
\tag{11}
\]

由玻色对易关系严格正规序化，

\[
O_p^\dagger O_q=\sum_{r=0}^{\min(p,q)}r!\binom pr\binom qr
(a^\dagger)^{p+q+2-r}a^{p+q+2-r}.
\tag{12}
\]

所以

\[
\boxed{G=\begin{pmatrix}
F_2&F_3&F_4\\
F_3&F_4+F_3&F_5+2F_4\\
F_4&F_5+2F_4&F_6+4F_5+2F_4
\end{pmatrix}.}\tag{13}
\]

**严格推论：** 对于一个固定且唯一占据的输入波包，忽略尚未纳入的高于七光子振幅阶与其他输出支，只要两个入射态的 $F_2,\dots,F_6$ 一致，式 (11) 的**整个约化单发射扇区密度矩阵**在任何输出端口及电子最终态上完全一致。这不需要 $S_0,S_1,S_2$ 共线，也不需要其具体磁序对称性。用上节的显式经典态和 $E=0.7$ BSV，二者在此观测层次完全不可区分。

独立计算检验：任取八个复数输出通道和三个独立 $S_p$，上述两种光输入的 8×8 约化密度矩阵相对 Frobenius 差约 $1.2\times10^{-14}$，纯数值舍入误差。**不能扩大到**未发射—发射相干交叉块（那里有 $\langle a^2\rangle,\langle a^\dagger a^3\rangle$）、两个及以上发射光子扇区、输入泵浦条件测量、未投影的多泵浦频率模和完整高阶散射态。

## 6. $E=0.7$ 首次被强度矩判据检测到：需要 $F_7$

对 BSV 的矩列构造 shifted Hankel $H_d^{(1)}$；在 $E=0.7$ 时尺寸 1、2、3 的行列式全部为正，尺寸 4 的行列式为负：

\[
\det [F_{i+j+1}]_{i,j=0}^{3}\approx-221169.571839<0.
\tag{14}
\]

这涉及 $F_7$。更强地，可以显式给出多项式 $q(I)=c_0+c_1I+c_2I^2+c_3I^3$，其中一个归一化选择为

\[
(c_0,c_1,c_2,c_3)\approx(-0.31000253,\ 0.20220197,\ -0.03071672,\ 0.00120515).
\tag{15}
\]

对于这个多项式，BSV 的阶乘矩形式上得出 $\mathbb E[I q(I)^2]\approx-0.005430997456$，而任何正 $P$ 经典光必然非负。式 (15) 的 $I$ 不是量子强度算符的普通函数代入，而是**对应的完全正规序化矩多项式**；负号并不意味着某个正算符的期望为负。

在 $E=0.7$ 的单发射通道 $O_p$ 框架里，$F_7$ 最早可能通过 $p+q+2=7$ 的 $O_2^\dagger O_3$ 干涉进入，因此至少需要九光子顶点产生 $O_3=(a^\dagger)^3a^5$，或者改变检测协议去直接测量更敏感的场关联。考虑到 Q-MATH-16.13/14 的五、七阶 *bare* 系数仍对回返、展宽和 BZ 敏感，**不建议仅为了 $E=0.7$ 的计数认证而继续机械地堆到九阶**。

独立有限阶矩测试显示：$E=0.1,0.2$ 首次被 $F_3$ 检出，$E=0.25,0.3$ 为 $F_5$，$E=0.7$ 为 $F_7$，$E=1$ 为 $F_9$，$E=2$ 为 $F_{13}$。这些数字是给定 shifted Hankel 主子式序列的检测阶数，不声称在全部可能的其他（例如相位敏感）量子见证中最优。

## 7. 与反铁磁 M0、真实 HHG 物理的关系：如何从禁止定理走向可能的新效应

**优先路线 A：低光子数五阶磁性读出。** 改为固定 $0<E<0.22871$ 的**同一个宽带纯单模 BSV** 波包，求经完整连通散射规范化后的三、五光子系数 $S_{0,N}(\Omega)$ 和 $S_{1,N}(\Omega)$。发射计数按散射阶展开，其三阶振幅平方对应 $F_2$，三阶与五阶的干涉项含 $F_3$；但五阶中还存在乘 $F_2$ 的 $c_{5,0}$ 修正，必须通过独立材料核标定或多端口/多谱点可辨识性分析剥离。只有当响应矩阵对 $F_2,F_3$ 满秩且条件数、噪声和 BZ/γ 稳定时，才可能提取经典禁戒 $EF_3-F_2^2<0$。在 $E\to0$ 时发射事件稀少，数学上强的见证未必实验可见。

**关键警告：** 以前的 Q-MATH-16.14 中 $c_{5,1}$ 在原始 M0 的一个 $k$ 点为非零，但其 *bare* 数值及所谓手工外腿剔除差别很大，不能直接把非零当作通过 LSZ 的物理读出系数。恢复连通 $S_{1,N}(\Omega)$ 必须先用 Xu–Fan 的多光子 S 与连通格林函数结构，[5] 联合该模型的 Peierls 接触项作完整处理。

**磁性专属性门槛：** M0 的反演与自旋 $\pi$ 旋转组合对称性此前证明，指定反演匹配偶次弹性相干转换的 $S_p$ 对交换强度与 SOC 奇变。在当前同一端口/同一偏振的严格配对协议下，磁畴翻转通常使 $S_p\mapsto-S_p$，于是**光子计数/上述强度矩见证不随 Néel 畴翻号**。因此不能把该信号称为 Néel-odd photon-counting。要获得真正的畴奇可观察量，需要经过证明的参考干涉（例如仅在检测端加入本振，而非将经典强种子混入 BSV 入射）或多个物理端口的相位关系。还必须做 $\Delta=0$ 非磁性负控和 $\lambda_{\mathrm{SO}}=0$ 自旋群负控。

**与现有论文区别：** 双层 CrI3 磁性 SHG、BSV 驱动固体 HHG、量子光激活禁戒谐波、量子 HHG 输出光子统计、能量守恒 Fock 扇区和条件谐波态非经典性都已有工作[6–10]。本报告没有发现新的 M0 专属物理现象；它提出了一项具体、可证伪且更节约数值资源的实验/计算 **材料读出任务**：低 $E$ 五阶连通响应的 Stieltjes 负值检测与磁性门控是否能同时得到认证。

## 8. 对 Q-MATH-16.18 的明确计算验收

1. **先在模型内算对：** 用用户提供的真实 `q167` 原始 Peierls 顶点，扩展成一个被占据的归一化宽带输入波包，使用同一组光学模式求 $S_0(\Omega)$ 与 $S_1(\Omega)$。与传统三阶散射极限交叉验证，保留所有接触图、Ward 恒等式和归一化。
2. **再在散射上算对：** 不能仅在裸 resolvent 中把两种外腿状态设零；必须通过已规范化的 in/out 态或连通格林函数/S-matrix 处理消除回返，且有 $\gamma\to0$ 以及输出 Maxwell 模式的误差记录。
3. **逆问题闭合：** 计算从 $(F_2,F_3)$ 到可测光子输出集合（频谱、端口、偏振、可用的耦合扫描）的雅可比矩阵，检验第二奇异值、条件数与 BZ 收敛。**没有满秩，就不能宣布材料可以独立读取 $F_3$**。
4. **真正磁性独立：** 固定同一个物理端口区分 $\pm N$，报告 $\Delta=0$、$\lambda=0$ 的负控；与独立检测本振的相位协议严格区分。
5. **经典资源优化：** 对照必须是整个非负 $P$ 凸集合，匹配实际波包、输入能量、$F_1$ 和实验上已测的必要阶乘矩，不能仅和一个相干态比较。理论见证必须明确测得的量和可信误差区间。

**本轮状态：** Stieltjes 阈值已严格证明；$E=0.7$ 的匹配 $F_0..F_6$ 非负 $P$ 已构造并数值验证；$O_0,O_1,O_2$ 的全单发射扇区有限阶 classical-simulation no-go 已证明；M0 的高阶 **connected** 材料核、逆问题与真实磁性非经典量子优势仍处于未验证状态。结论不应超出此证据等级。

## 参考文献与查新定位

[1] G. S. Agarwal and K. Tara, *Nonclassical character of states exhibiting no squeezing or sub-Poissonian statistics*, **Phys. Rev. A 46**, 485 (1992). https://doi.org/10.1103/PhysRevA.46.485

[2] E. Shchukin, Th. Richter and W. Vogel, *Nonclassicality criteria in terms of moments*, **Phys. Rev. A 71**, 011802(R) (2005). https://doi.org/10.1103/PhysRevA.71.011802

[3] E. V. Shchukin and W. Vogel, *Nonclassical moments and their measurement*, **Phys. Rev. A 72**, 043808 (2005). https://doi.org/10.1103/PhysRevA.72.043808

[4] Y.-J. Hu and G.-N. Chen, *A unified treatment for the matrix Stieltjes moment problem*, **Linear Algebra Appl. 380**, 227 (2004). https://doi.org/10.1016/j.laa.2003.10.012

[5] S. Xu and S. Fan, *Input-output formalism for few-photon transport: A systematic treatment beyond two photons*, **Phys. Rev. A 91**, 043845 (2015). https://doi.org/10.1103/PhysRevA.91.043845

[6] K. E. Dorfman, F. Schlawin and S. Mukamel, *Nonlinear optical signals and spectroscopy with quantum light*, **Rev. Mod. Phys. 88**, 045008 (2016). https://doi.org/10.1103/RevModPhys.88.045008

[7] K. Prabhu and G. Dixit, *Sensing quantum fluctuations via forbidden harmonics in quantum materials*, **Phys. Rev. B 114**, L140301 (2026). https://doi.org/10.1103/v886-ypf7

[8] H. Yan et al., *Entanglement and nonclassicality of harmonic modes in solid-state high-harmonic generation*, **Phys. Rev. A**, accepted 31 Aug 2026. https://doi.org/10.1103/k3rf-bhbc

[9] P. Stammer, *Invariant energy subspaces in quantum-optical high-harmonic generation: An information-theoretic approach*, **Phys. Rev. A**, accepted 16 Oct 2026. https://doi.org/10.1103/krr8-m45d

[10] X. Liu et al., *Theory of high-harmonic generation with quantum light*, **Phys. Rev. Research 8**, 023326 (2026). https://doi.org/10.1103/y3gr-tsns

## 复现文件

- `qmath_16_17_stieltjes_moment_audit.py`：BSV 矩递推、广义特征值/正权 Gauss–Radau 经典混合、Hankel 判据、低阶发射 Gram、随机复输出核、既存 M0 五阶数值的来源索引。
- `qmath_16_17_results.json`：全部原始计算数据，包含经典混合、矩匹配残差、负多项式证书和 M0 历史系数敏感性。
- `qmath_16_17_F3_witness.png` 与 `qmath_16_17_minimal_moment_order.png`：数学结果的可视化，不是 HHG 强度预测。

