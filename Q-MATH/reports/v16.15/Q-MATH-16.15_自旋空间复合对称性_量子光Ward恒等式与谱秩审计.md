# Q-MATH-16.15｜自旋–空间复合对称性、纯 BSV 的光学 Ward 恒等式及高 Fock 谱秩审计

**日期：2026-10-24**  
**模型：** 原始 Q-MATH-16.7 / Q-MATH-16.14 生产版八带 M0 的现象学 PT-AFM Hamiltonian。**不是**真实双层 CrI₃ full112/M88 Wannier 或完整 Maxwell 远场模型。  
**结论级别：** 一个与摄动阶数无关的精确对称性命题 + Peierls 顶点 120 组机器检验 + 真实 M0 三/五光子裸散射复算 + 运动学伪秩反例 + 严格可解的「无相干但有计数」量子散射玩具反例。尚无经过 LSZ、BZ、展宽与输出端口全部验收的新材料量子优势。

## 1. 本轮真正获得了什么？

在**当前 M0 的自旋守恒、仅有 Kane–Mele 型 SOC 与 Néel 交换、反演配对、均匀电偶极光学端口**的条件下，可证明：

1. 含 **奇数个电偶极场顶点**的电子 Keldysh 影响泛函导数是 SOC 系数 \(\lambda\) 的奇函数；在 \(\lambda=0\) 时严格为零。三、五、七光子阶的 *弹性* \(2\omega\) 转换均在这一禁戒之内。若再反转 Néel 交换 \(\Delta\)，同一类系数也变号，因此在解析区间内具有 \(\Delta\lambda\) 因子。
2. 对任何入射**偶宇称**泵浦态（纯单波包 BSV 包含在内），如果电子准备态按此对称性协变，且 H2 被测模式具有指定的反演奇宇称，则**完整联合量子演化**满足 \(\beta_H(\lambda)=-\beta_H(-\lambda)\)、\(n_H(\lambda)=n_H(-\lambda)\)。因此 \(\lambda=0\) 时 \(\langle b_H\rangle=0\)，而不能据此推断 \(\langle b_H^\dagger b_H\rangle=0\)。量子「相干亮/暗」与「光子计数亮/暗」是两个不同的选择定则问题。
3. 严格单色泵浦 \(\omega\) + 弹性电子返回 + 固定 H2 偏振/端口的输出 Hilbert 空间**只有一个**能量守恒模式 \(\Omega=2\omega\)。不论五、七阶有多少 Fock 多项式系数，其物理谱方向都共线。把许多次**不同泵浦频率**实验串接成一个向量、算出 Gram 秩大于一，不是单次宽带 BSV 的独立材料谱模证明。

这三个结果中，第一项是**受已知自旋群理论启发的 M0 模型特定选择定则**，并非新创的普适磁光规律。第二、三项明确改变下一轮「量子关联如何从磁序通道中读出」的计算策略。

## 2. 从八带 Hamiltonian 严格起底

原始冻结的电子 Hamiltonian（\(N=+1\) 或 \(-1\) 时，也可以将 \(D=N\Delta\) 作为带符号连续交换参数）写成

\[
H_{D,\lambda}(\mathbf k;\mathbf A)=H_0(\mathbf k+\mathbf A)
+\lambda\,g(\mathbf k+\mathbf A)\,\sigma_zs_z
+D\,\tau_zs_z,
\]

其中 \(H_0\) 包含自旋无关的层内三条 Fourier 键与层间四条 Wilson–Peierls 跃迁键；更准确的嵌入 Bloch 形式和键位移直接以附带冻结源码为准，不能把 \(T_\perp\) 简化成任意无相位的常数。交换强度 \(D\) 对顶点不显含 \(\mathbf A\)，但通过初态与虚中间态能谱/本征矢参与全部非线性响应。

引入

\[
\mathcal P=\tau_x\sigma_x s_0,\qquad R_x=e^{-i\pi s_x/2},
\]

空间反演和自旋 \(\pi\) 旋转分别给出

\[
\mathcal P H_{D,\lambda}(\mathbf k;\mathbf A)\mathcal P^{-1}
 =H_{-D,\lambda}(-\mathbf k;-\mathbf A),\tag{15.1}
\]
\[
R_x H_{D,\lambda}(\mathbf k;\mathbf A)R_x^{-1}
 =H_{-D,-\lambda}(\mathbf k;\mathbf A).\tag{15.2}
\]

其成立原因是 \(R_xs_zR_x^{-1}=-s_z\)，而该 M0 的光学耦合只涉及自旋对角的 Peierls 键；在真实自旋混合 Rashba SOC、自旋翻转跃迁或旋转场的情形，这些恒等式需要重新审计。将 (15.1)(15.2) 合成，定义 \(Q_e=\mathcal P R_x\)，得到

\[
Q_e H_{D,\lambda}(\mathbf k;\mathbf A)Q_e^{-1}
=H_{D,-\lambda}(-\mathbf k;-\mathbf A).\tag{15.3}
\]

这不是把两个磁畴无条件当作同一个**固定右侧远场端口**；(15.3) 必须同时映射 \(\mathbf k\)、入射方向、光学场的宇称与探测模式。

## 3. 全阶 Peierls–Keldysh 选择定则

令 \(V_{D,\lambda}^{(r)}\) 表示对输入/输出电偶极矢势总共求 \(r\) 次导数的 Peierls 顶点。对任意方向组合，有

\[
\mathcal P V_{D,\lambda}^{(r)}(\mathbf k)\mathcal P^{-1}
=(-1)^r V_{-D,\lambda}^{(r)}(-\mathbf k),\qquad
R_xV_{D,\lambda}^{(r)}(\mathbf k)R_x^{-1}
=V_{-D,-\lambda}^{(r)}(\mathbf k).\tag{15.4}
\]

数值不只检查低阶三个顶点，而是对 0–7 阶及不同磁序、SOC、动量共 **120 组**完成独立验证。最大 Frobenius 误差：空间反演 \(1.46\times10^{-16}\)，自旋 \(\pi\) 旋转为浮点零。

对于动量配对、准备态协变的电子 Keldysh 泛函

\[
\Phi_{D,\lambda}[A_+,A_-]
=\mathrm{Tr}_e\{U_{D,\lambda}[A_+]\rho_{D,\lambda}
U_{D,\lambda}[A_-]^\dagger\},
\]

(15.3) 推出

\[
\boxed{\Phi_{D,\lambda}[A_+,A_-]
=\Phi_{D,-\lambda}[-A_+,-A_-].}\tag{15.5}
\]

所以 \(m\) 次场导数 \(\partial^m\Phi|_{A=0}\) 在 \(\lambda\to-\lambda\) 下变成 \((-1)^m\) 倍。若再单独用空间反演 \(\mathcal P\)，奇顶点数的弹性磁性响应也对 \(D\to-D\) 变号。特别是三、五、七光子阶的**弹性单 H2 光子产生**，顶点数分别为 3、5、7，满足

\[
\boxed{c^{(2r+1)}(D,\lambda)
=-c^{(2r+1)}(-D,\lambda)
=-c^{(2r+1)}(D,-\lambda),\quad r\ge1.}\tag{15.6}
\]

在有限体积、有能隙且所研究响应对 \((D,\lambda)\) 解析的局部参数区域，进一步得到 \(c^{(2r+1)}(D,\lambda)=D\lambda F_r(D^2,\lambda^2)\)。**这一因子不提供数值幅度下界，也不能替代频率/BZ/展宽收敛。**

### 对纯 BSV 的更强全量子版本

把每个具有指定反演奇宇称的**偶极光学模式**的光子奇偶算符写成 \(\Pi_\gamma=(-1)^{\sum_j a_j^\dagger a_j+\sum_\mu b_\mu^\dagger b_\mu}\)。它令 \(a_j\mapsto-a_j\)、\(b_\mu\mapsto-b_\mu\)。对全电子+光子 Hamiltonian，\(\mathcal Q=Q_e\Pi_\gamma\) 满足

\[
\mathcal QH_{D,\lambda}\mathcal Q^{-1}=H_{D,-\lambda}.\tag{15.7}
\]

纯 BSV \(S_u(\xi)|0\rangle\) 只含偶数泵浦光子，因此对 \(\Pi_\gamma\) 为偶；H2 初态真空同样为偶。只要电子初态为 \(Q_e\) 协变的绝缘基态或对称平衡态，那么**无需对泵浦光子数截断做摄动展开**，对这个 H2 模有

\[
\boxed{\beta_H(D,\lambda;t)=-\beta_H(D,-\lambda;t),\qquad
n_H(D,\lambda;t)=n_H(D,-\lambda;t).}\tag{15.8}
\]

所以 \(\lambda=0\) 时 \(\beta_H=0\)，但 \(n_H\) 的选择规则只是 SOC 偶性，并不要求 \(n_H=0\)。这不证明 \(n_H(0)>0\) 对真实 M0 一定成立，也不证明输出态非经典。对于同样的 \(\pm\alpha\) 相干态经典混合，只要入射光奇偶不变，(15.8) 同样成立；故它是**量子光可观测量层级的对称性差异**，并非独属于 BSV 的优势。

**端口严格限制：** 上式适用于本模型中以空间反演协变的偶极模式定义的 \(b_H\)。固定单侧远场行波一般在反演下映射到另一侧行波，不能把 (15.8) 原样施加到未经映射的单侧探测器。必须先建立完整双层 Maxwell 出射关系。初态需要与 \(\lambda\to-\lambda\) 的变换协变；被人为准备的自旋选态/外部不对称门控也可能破坏该等式。

## 4. 真实 M0 数值：为什么这项选择定则成立，却还不能宣布新机制

本轮调用用户 ZIP 原始 `q167_m0_three_photon_T.py` 与 `qmath_16_14_math_and_m0.py`，并且特别修复了一个容易遗漏的数值坑：原始 `q167` 模块在 import 时缓存了 `BONDS` Fourier 跃迁键列表。**如果扫描 \(\lambda_{SO}\) 只修改 `q.m.R0.lam_so` 而不同时执行 `q.BONDS=q.bonds()`，实际所用 SOC Peierls 顶点不会更新，扫描无效！** 本轮每次修改参数都显式刷新缓存并恢复冻结值。

在 \(\omega_1=0.16,\omega_2=0.20,\Omega=0.36,\gamma=0.25,D=0.8\) 下，得到 BZ 平均三光子弹性系数模：

| BZ | \(|T_3(\lambda=0)|\) | \(|T_3(0.02)|\) | \(|T_3(0.05)|\) | \(|T_3(0.1)|\) |
|---|---:|---:|---:|---:|
| 5×5 | 1.49×10⁻¹² | 1.5795 | 3.3555 | 4.0353 |
| 9×9 | 1.64×10⁻¹¹ | 1.1976 | 2.8423 | 4.2312 |
| 15×15 | 4.83×10⁻¹² | 1.2877 | 3.0781 | 4.9827 |

在 5、9、15 的三个网格上，\(\lambda\mapsto-\lambda\) 的相对残差 \(\sim10^{-12}\)，\(D\mapsto-D\) 的相对残差约 \(10^{-11}\)，\(D=0\) 与 \(\lambda=0\) 的零值检查也达到类似精度。

五光子 \(c_{5,1}\) 的独立扫描在 \(\omega=0.18,\gamma=0.25,D=0.8\) 下给出

| BZ | \(|c_{5,1}(0)|\) | \(|c_{5,1}(0.05)|\) | \(|c_{5,1}(0.1)|\) | \(\lambda\to-\lambda\) 相对残差 |
|---|---:|---:|---:|---:|
| 5×5 | 2.96×10⁻⁸ | 14,436 | 12,644 | 5.20×10⁻¹² |
| 9×9 | 8.75×10⁻⁸ | 8,733 | 8,403 | 2.18×10⁻¹¹ |

五阶数值只是 **\(i\gamma\) 正则化的裸 \(T\) 系数**，当前仍缺 LSZ/外腿自能消除、完整输出归一化及稳定 BZ 外推。大数不是实际转换概率，且不意味着五阶超过三阶。

进一步做 \(\lambda\downarrow0\) 的斜率网格检验发现：\(|T_3(0.005)/0.005|\) 在 9×9/15×15/25×25/35×35 分别约 60.72/64.98/84.83/114.07；这提示虽然**零点与奇偶性严格**，但响应斜率/幅值在此 \(\gamma\) 与网格下**远未 BZ 收敛**。不可把表格读作可预测的 SOC 增益定量规律。

## 5. 一个容易被遗漏的量子光学反例：\(\beta=0\) 不代表 \(n_H=0\)

我们独立构造严格能量守恒、保持电子–光子联合奇偶对称的**双电子能级玩具散射器**（**不是 M0**）：

\[
H=\omega a^\dagger a+\Omega b^\dagger b+\Delta_e|e\rangle\langle e|
+g\big[b^\dagger a^2|e\rangle\langle g|+ba^{\dagger2}|g\rangle\langle e|\big],
\quad \Delta_e=2\omega-\Omega>0.
\]

取 \(Q_e=|g\rangle\langle g|-|e\rangle\langle e|\)，\(\Pi=(-1)^{n_a+n_b}\)，有 \([H,Q_e\Pi]=0\)。输入 \(|g\rangle\otimes|\mathrm{BSV}\rangle\otimes|0_b\rangle\) 是总奇偶 \(+1\) 本征态。

由于 \(b\) 是奇算符，严格有 \(\langle b(t)\rangle=0\)；但转移到电子奇宇称 \(|e\rangle\) 的输出态允许带一个 \(b\) 光子，短时光子数为

\[
\boxed{n_H(t)=g^2t^2F_2+O(g^4t^4)>0,
\qquad F_2=\langle a^{\dagger2}a^2\rangle.}\tag{15.9}
\]

有限 Fock 空间完整厄米 Hamiltonian 精确指数传播的单元测试：取 \(\omega=.18,\Omega=.27,\Delta_e=.09,g=.04,t=.1\)，对 \(E=0.1,0.7,1.5\)，均得到 \(\beta=0\)、\(n_H>0\)；对于 \(E=0.7\)，\(n_H=3.4710264\times10^{-5}\)，而短时公式为 \(3.4719049\times10^{-5}\)（Fock 截断下 \(F_2=2.16994\)）。此例证明**禁止平均谐波电场**和**禁止谐波光子产生**不是同一命题，不能用经典电流平均代替计数 Keldysh Gram。

对于 M0，是否确实存在该类 SOC-free 电子条件激发与正频输出，仍须按真实电子终态能量壳计算，并消除有限时间伪辐射。这是一个可检验候选，不是已经观测到的 M0 量子 emission。

## 6. 为什么本轮必须撤回「单色扫频证明高 Fock 输出谱非共线」的思路

固定一个严格单色泵浦模 \(\omega\)，电子弹性返回基态，测一个指定输出偏振、传播端口；自主散射能量守恒使 \(\Omega=2\omega\) 唯一。所有高阶 Fock 系数对应同一个输出 ket \(|1_{H,2\omega}\rangle\)，所以

\[
S_j(\Omega)\propto\delta(\Omega-2\omega),\quad \mathrm{rank}\{S_0,S_1,S_2,\dots\}\le1.\tag{15.10}
\]

若 \(S_0\neq0\)，则 \(D_{01}=\|S_0\|^2\|S_1\|^2-|\langle S_0,S_1\rangle|^2=0\)。\(\delta\) 必须理解为规范归一化的离散/波包输出模式，不能取无归一化分布范数。

为了显示误导性，本轮在 M0 原始裸 \(T\) 上做了 **11 个分别独立的单色实验**：改变 \(\omega\in[.14,.22]\)，从 9×9 BZ 取得 \(c_3(\omega),c_{5,1}(\omega)\) 两列数据，并把它们当作向量算夹角，得到

\[
\boxed{\sin\angle(c_3,c_{5,1})\simeq0.194365.}
\]

该数值非零，但它比较的是不同 **入射模式、不同实验**，不是同一个 \(u(\omega)\) 的宽带 BSV 经过 M0 散射后的谱核 rank。它没有提供独立的材料输出方向。**因此这条所谓「单色扫频获得高阶多模」的捷径应永久列为经典负控/伪象，而非新现象证据。**

真正的判据必须固定**同一个**光学波包 \(u\)，取得三、五、七阶的物理散射波包 \(S_0(\Omega;u),S_1(\Omega;u),S_2(\Omega;u)\)，并在同一带宽/探测窗/Maxwell 端口中做非共线性与经典正 P 优化。

## 7. 与旧异常矩定理的连接：什么才是新机制？

前一轮严格得到：

\[
M_j=\langle(a^\dagger)^j a^{j+2}\rangle,
\quad M_0=m,\quad M_1=3Em,
\quad M_2=(15E^2+3E)m,
\]

以及任意经典 \(P\ge0\) 服从 \(|M_j|\le F_{j+1}\)。对相同 \(E,F_2\)，纯 BSV 的五阶超经典条件为 \(E>1/3\)；对匹配 \(F_3\) 的七阶见证则是任何 \(E>0\)。这些结论是普适量子光学，不是新磁性机制。

把本轮 M0 Ward 恒等式代入形式上的输出相干谱

\[
\beta_H(D,\lambda;\Omega)
= M_0S_0(D,\lambda;\Omega)
+M_1S_1(D,\lambda;\Omega)
+M_2S_2(D,\lambda;\Omega)+\cdots,
\]

得到在解析弱 SOC/交换区间，各 \(S_j\) 都具有 \(D\lambda\) 因子。这称为**SOC 门控的非经典关联转导器候选结构**。但若所有 \(S_j\) 在真正固定波包/固定输出协议中共线，材料只给出一个相干幅度系数，\(M_1,M_2\) 不能独立从输出分离出来。即使非共线，之前已经构造了可复制归一化谱形的严格经典正 P 反例，仍须比较绝对响应的正 P 资源优化。

只有 **(a) 真正的物理连通散射核**、**(b) 同一宽带纯 BSV 输入下多个独立输出方向**、**(c) Néel × SOC 特有、去掉 SOC/交换以后受 Ward 禁戒的响应**、**(d) 对固定端口、同能量/F2/F3 资源的经典全局负控超越**，四者同时成立，才可能升级为区别现有论文的特殊机制。

## 8. 与现有文献的严格新颖性边界

- *Probing the antiferromagnetic structure of bilayer by second harmonic generation: A first-principles study*, Phys. Rev. B **106**, 235410 (2022), https://doi.org/10.1103/PhysRevB.106.235410。磁性 SHG 和 SOC 依赖均有先例；不能称「发现反铁磁 SHG 需要 SOC」为普适首创，真实材料和当前简化 M0 对称性也不能混同。
- Koki Mizuno, *High-harmonic spin-current signatures of altermagnetic spin-group symmetry*, Phys. Rev. B **114**, L200403 (2026-10-19), https://doi.org/10.1103/1vc6-c7rs。利用自旋点群和动力学对称性推导无 SOC 情况的谐波选择定则，与本轮思路相关，但对象是交错磁性自旋电流，当前 M0 的 BSV Ward 同一性仍需独立物理验收。
- Liu et al., *Theory of high-harmonic generation with quantum light*, Phys. Rev. Research **8**, 023326 (2026), https://doi.org/10.1103/y3gr-tsns。强调平均偶极、两时涨落与谐波光子统计必须区分。
- Rasputnyi et al., *High harmonic generation by a bright squeezed vacuum*, Nature Physics **20**, 1960–1965 (2024), https://doi.org/10.1038/s41567-024-02659-x。纯 BSV 驱动固体 HHG 已有实验。
- 数学基础：Weyl/CCR、Keldysh 超算符与对称性；Dorfman, Schlawin & Mukamel, Rev. Mod. Phys. **88**, 045008 (2016), https://doi.org/10.1103/RevModPhys.88.045008。\(R_x\) 的精确映射是本模型直接矩阵推导，不是对既有文献结论的臆测替代。

## 9. 下一项 Q-MATH-16.16 的优先验收

应优先从生产 Q16.7 三光子 Peierls 核计算**同一输入宽带 BSV 波包**的弹性/非弹性正频输出散射包络，建立规范的输出模式内积、着装态和电子最终态分类；再用 Q16.13 的高 Fock 正规序多项式作为形式级数的交叉检查。五/七阶**外腿重求和与有限 \(\gamma\) 的正确 LSZ 散射振幅**尚未完成，不能使用裸 \(c_{5,1}\) 的巨大数值直接宣称增强。如果全阶计算代价过高，则先在有解析误差界的精确可解散射器验证正则化/剥离策略，再以生产 M0 重跑。

建议保持的四类负控：\(D=0\)、\(\lambda=0\)、材料裸核替换为常数但保留输出波包、同 \(E,F_2,F_3\) 的最优正 P 输入。BZ 需要独立增加 \(L\) 与降低人工展宽 \(\gamma\)；本轮 \(T_3/\lambda\) 的 9²→35² 漂移已经表明**响应绝对数值尚未收敛**。

**冻结状态：** 已获得 M0 的全阶自旋–空间联合 Ward 恒等式及高阶 Peierls 直接检验，排除了把单色扫频当作固定 BSV 材料谱模的错误；没有通过物理散射与经典资源全部验收的反铁磁量子光 HHG 新发现。

## 10. 复现实验与数据清单

保存脚本：`qmath_16_15_vertices_symmetry.py`（120 组 0–7 阶顶点）；`qmath_16_15_soc_symmetry.py`（三光子 BZ SOC/Néel 正负扫）；`qmath_16_15_five_soc_check.py`（五光子 0/±SOC 对照）；`qmath_16_15_soc_derivative.py`（斜率 L 收敛诊断）；`qmath_16_15_frequency_scan_trap.py`（跨输入扫描伪谱秩）；`qmath_16_15_parity_count_toy.py`（完全酉、能量守恒的计数-相干区分反例）。相应 JSON 与原始日志均存档；包含冻结 Q16.7 代码以及生成源文件的哈希记录。原始文件不修改；全部 SOC 变参运行更新 `q.BONDS=q.bonds()`。
