# 线 Q：Codex → Cursor 交接与下一步执行指令

日期：2026-09-26  
适用范围：当前 `Quantum-light` 工作区的线 Q / M0 / PT-AFM 量子输出研究  
目的：把现状、已验收边界、尚未解决的科学问题和 Cursor 的下一步工作范围交接清楚。

---

## 0. 给 Cursor 的最高优先级说明

这不是重新选择量子理论，也不是继续无限追加 P1=0 工程审计。

已经批准的强驱动理论框架是：

\[
\boxed{
\text{P1 强经典背景}
+\text{输运的粒子—空穴准玻色子}
+\text{显式 photon modes}
+\text{联合二次 Hamiltonian / Gaussian 演化}
}
\]

此前称为 Stage1P。不要重新把“选择 RPA/quasi-boson 框架”列为待决问题。当前真正的衔接问题是：把这个既定框架与规范化输出 photon channel 及 PT-AFM 输出对称性清楚地连接起来。

执行原则：

1. **符合既定研究主线的工程工作可以继续做**：查档、复核公式、核对模型定义、做纯代数残差审计、整理文献和写可复核的短说明，不必每一步都停下来请用户批准。
2. **进入会改变物理问题定义的数学物理选择时必须停下并询问用户**，同时给出几个方案、各自后果及推荐项。不要把建议悄悄写成已冻结的定义。
3. 询问前先把不依赖该选择的材料准备好。尤其不要因为比较协议未定，就停止 M0 已有 \(P/T/PT\) 矩阵关系的只读核查。
4. 不以“验收通过”为由直接启动 P1 数值、生产 \(n,m,V\)、squeezing 或纠缠计算。

---

## 1. 当前研究位置

研究目标的主桥仍是：

\[
\boxed{
\text{材料电流算符}
\longrightarrow
\text{电流涨落/关联}
\longrightarrow
\text{传播场算符}
\longrightarrow
\text{有限带宽输出 photon mode}
}
\]

长期科学问题是 PT-AFM 的磁序、磁畴和对称性怎样改变输出光子模式，包括相位敏感相关、谐波间关联，以及与量子光驱动 HHG 的 \(P\)-分布描述之间的关系。

当前已走到：

\[
\boxed{
\text{P1=0 输出通道规范化已过窄门}
\quad\longrightarrow\quad
\text{PT-AFM 输出模式的磁畴/对称性定义待明确}
}
\]

还没有进入 P1 驱动后的量子输出物理结果阶段。不要把“基础 I/O 合法”说成“P1 下的输出理论、真空、Gaussian 性或纠缠已完成”。

### 当前状态表

| 工作层 | 当前状态 | 边界 |
|---|---|---|
| M0 八带机制模型、P0/P1 经典参考计算 | 已有基线与复核文档 | M0 是受约束的机制模型，不是 CrI\(_3\) 定量拟合；若引用未收敛量，须保留原报告限定 |
| P1=0 full-BZ 静态响应、低频/IR、passivity 与稳定性资格链 | 现有线 Q 文档记录为通过相应审计 | 除非发现具体矛盾，不再扩展同类审计 |
| \(A\)-port 到 SI 归一 photon \(a\)-channel | 2026-09-26 窄门通过 | 对当前均匀 \(q_\parallel=0\) 真空两侧二维电流片、两面内偏振和已审计频点成立 |
| \(C_I=L L^\dagger\) 与 photon output row | 已审计网格/频点通过 | 仅证明 photon row coisometry；不是完整 photon+electron scattering matrix 的 unitary |
| P1=0 candidate in-vacuum \(n=m=0\) | 结构 sanity control | 代码把 creation block 直接设为零；不能把它说成独立 ground-state 计算或生产初态选择 |
| M0 的 \(P/T/PT\)、镜面、\(C_2\) 对输出模式的作用 | 模型中已有候选矩阵和 Hamiltonian 级定义；输出端口/偏振/反幺正映射尚需系统整理 | 可先做矩阵代数审计；不可把数学对称操作自动等同于实验磁畴翻转 |
| Stage1P 联合 quasi-boson + photon 强驱动演化 | 理论框架已批准，尚未与最终输出模式完整衔接 | 不重新选择理论；在物理定义确认前不启动生产运行 |
| \(N\to-N\) 下的 \(\beta,n,m\)、谐波间相关 | 未冻结、未生产 | 先定义比较协议与模式变换，再决定代表模式 |
| squeezing、Pauli、\(V\)、纠缠、生产初态、soft dressing | 未解/未生产 | 不因 P1=0 channel 通过而自动解锁 |

---

## 2. 2026-09-26 规范化通道的已验收内容

### 2.1 photon 归一化与源方向

最新报告给出的 SI 电流到远场 photon channel 关系为：

\[
a_{\rm src}(\omega)
=
i\sqrt{\frac{Z_0}{4\pi\hbar\omega S}}\,
\widetilde I(\omega),
\qquad
\widetilde I(\omega)=\int dt\,e^{+i\omega t}I(t).
\]

在当前冻结的代码单位和两端口定义下：

\[
a(\omega)=
\sqrt{\frac{\omega}{\gamma_{\rm self}}}\,A(\omega).
\]

对所用的 \(\delta I(t)\) 展开及 \(+i\omega t\) Fourier convention，正频率电子源为粒子—空穴湮灭通道 \(B\)，不是 \(B^\dagger\)。报告还给出了有限带宽模式与检测核的表达式；未来实现 packet/mode 时要沿用该定义并核对归一化。

### 2.2 规范化输出行

\[
a_{+,\rm out}
=S_a a_{+,\rm in}+L_a b_{\rm el,in},
\qquad
S_aS_a^\dagger+L_aL_a^\dagger=I_2,
\]

奇通道为

\[
a_{-,\rm out}=-a_{-,\rm in}.
\]

报告中三套网格的已审计样本 photon-row 残差约 \(8.9\times10^{-16}\)，隙下电子通道 rank 为 0、隙上样本 rank 为 2。其准确结论是：**已审计样本上的 photon output row coisometry 通过**。电子 output rows 未构造，因此不得声称完整 extended scattering matrix 已证明 unitary。

### 2.3 真空 control 的准确解释

脚本中 `candidate_vacuum_control()` 明确采用

```python
creation_block = np.zeros_like(l_a)
```

所以 \(n_{\rm out}=m_{\rm out}=0\) 是 annihilation-only stationary channel 假设下的结构 control。它不是独立生成出来的 ground-state covariance，也没有解决 interacting ground state 相对 bare Fock 基底的 soft dressing。

**不要**把下面任一结论写进汇报或摘要：

- “已证明完整 photon+electron scattering matrix 全局 unitary”；
- “已证明真实 interacting ground state 就是 bare/asymptotic vacuum”；
- “已完成 P1 下 photon output covariance”；
- “已完成 Pauli、\(n,m,V\)、squeezing 或纠缠”。

---

## 3. 本地权威材料与阅读顺序

以下路径均相对于仓库根目录：

### 必须先读

1. `docs/04_BSV量子光/线Q_P1零驱动规范化光子电子通道_20260926.md`
2. `tools/b0_small_model/q8_m0_stage1ao_canonical_channels.py`
3. `results/q_line/stage1ao_canonical_channels_20260926/canonical_channels.json`
4. `tools/b0_small_model/q8_m0_model.py`
5. `docs/04_BSV量子光/线Q_M0_v1物理解读一页_20260912.md`
6. `docs/04_BSV量子光/线Q_P0P1交付复核与下一阶段边界_20260914.md`
7. `docs/04_BSV量子光/线Q_P1dressed_ph_photon_Gaussian_20260923.md`

### 理论笔记

Obsidian 路径：

`C:\Users\26507\Documents\Obsidian Vault\量子光HHG\01_理论\01_完整理论_20260729\`

优先阅读：

- `量子光驱动HHG与输出光子统计_完整理论与计算方法_20260729_第三部分.md`，尤其 §10.1–10.7：独立磁畴观测量、共同基底映射、\(C_{oo}\) 宇称和实验操作与数学对称操作的区别；
- 其余部分中电流到输出模式的定义，以及电子源关联、输入态、输出 covariance 的前提。

如果 Cursor 当前会话读不到 Obsidian 路径，先说明这一限制；不要假装已经读过。仓库内有镜像/摘录时，核对其日期与原笔记版本后再用。

### 历史 toy 的边界

- `tools/b0_small_model/PT_AFM_TOY.md`
- `tools/b0_small_model/pt_afm_toy.py`

4-band toy 已关闭，只能作为对称性代数/代码 harness。特别是其中 \(V=\tau_xs_z\) 和 \(W=s_z\) 是该 toy 的隐藏映射，不是可直接挪给 M0 或 CrI\(_3\) 的普适结论。不要为了产生非零 \(J_o\) 或 H2 改写 toy Hamiltonian。

---

## 4. 给 Cursor 的具体执行指令

### 阶段 A：先做无须用户选择的只读核查

这部分属于已有模型定义的工程/代数核验，可以继续做，不用先问用户。

1. 对照 `q8_m0_model.py`、M0 spec 和静态/PT/Bloch-sewing 审计文档，列出 M0 当前的：
   - 基底次序与轨道位置；
   - inversion center 与 embedded Bloch convention；
   - \(P,\Theta,PT\)、镜面、\(C_2\) 的矩阵/反幺正定义；
   - \(H_N(\mathbf k)\) 的各自变换目标，包括 \(N\)、\(\mathbf k\) 和 Bloch sewing。
2. 使用现有模型函数做纯代数残差核对，至少检查 \(N=\pm1\)、BZ 内多个对称相关 \(\mathbf k\) 点：
   - \(P H_N(\mathbf k)P^{-1}\)；
   - \(\Theta H_N(\mathbf k)\Theta^{-1}\)（显式保留 complex conjugation）；
   - \(PT\) 的平方与 Hamiltonian 关系；
   - mirror/\(C_2\) 关系及其当前几何/跃迁条件；
   - \(j_x,j_y\) 的 polar-vector 变换，以及 \(\mathbf k\to-\mathbf k\) 后的分量符号。
3. 检查 \(\partial_{\mathbf k}H\) 与已有 `current_ops` 的定义一致。遇到 embedded Bloch 的 reciprocal sewing 时，明确写出 sewing matrix，不可用“周期边界上显然相同”代替。
4. 输出一份短的审计记录：公式、矩阵、测试点、最大残差、依据文件、哪些关系已在旧报告证明、哪些是本轮新增的数值核对。
5. 若新增测试代码只是在复现上述已冻结模型恒等式，可以直接写最小测试；不要借机重构模型、改参数或更新生产脚本。

阶段 A 的结论只应描述**数学空间/时间对称操作**。不要在这一步把 \(P\) 或 \(T\) 叫成实验上的“翻转磁畴操作”。

### 阶段 B：文献核查，服务于定义而非替代定义

整理一页精简的原始研究文献表，至少覆盖：

1. 双层 CrI\(_3\)/层状反铁磁体的磁畴依赖 SHG 与磁点群选择定则；
2. 量子光/强场 HHG 中输出光子统计、squeezing 或纠缠的原始工作；
3. 周期驱动线性玻色系统、Floquet 输入—输出或 Bogoliubov scattering 的原始理论。

每条文献要记录：完整题目、作者、期刊/年份、DOI/arXiv 标识、具体支持的论点及不能支持的论点。优先引用原始论文和期刊/预印本正文；综述只能用于补背景。不能仅凭一篇 SHG 论文推导本项目的量子 \(m_{\mu\nu}\)，也不能把不同实验几何下的 selection rule 直接移植到 M0。

需要特别区分两种驱动描述：

- 输入态用 \(P(\alpha)\) 表示并对经典振幅采样/平均；
- 保留光场算符、电子—光子联合演化，再计算输出算符相关。

文献表需说明二者在哪些输入态/观测量/近似下可能等价，何时 \(P\)-分布的非正定性或算符相位关联使简单经典统计平均不足。不要先验宣称“所有 \(P\)-分布方法都只是经典方法”或“联合算符路线必然展示量子优势”。

### 阶段 C：在真正的物理选择点停下来问用户

Obsidian §10 已说明，分别求解 \(+N\) 与 \(-N\) 的最终观测量可直接做 domain-even/domain-odd；但若构造同一 Hilbert 空间的

\[
J_e=\frac{J_++\widetilde J_-}{2},\qquad
J_o=\frac{J_+-\widetilde J_-}{2},
\]

就必须指定 \(\widetilde J_-\) 的共同基底映射 \(\mathcal U_N\)。这会改变算符层面的 \(J_e,J_o\) 与 \(C_{ee},C_{eo},C_{oe},C_{oo}\) 的含义，不能由工程实现替用户决定。

完成阶段 A/B 后，Cursor 必须把问题明确交给用户，并至少给以下选项：

**A. 固定实验室几何的磁畴对比**  
两畴保持同一入射方向、偏振、CEP、探测端口和收集模式，只比较最终可观测量 \(\beta,n,m\) 的 \(+N/-N\) 差。优点是最贴近实验；不必先定义跨磁畴共同基底算符。缺点是对称性选择定则需要分别联系两畴解。

**B. 先定义一个数学对称映射**  
明确采用 \(P\)、\(T\) 或其他经 M0 验证的操作，并允许它按定义变换 \(\mathbf k\)、端口、偏振、复共轭等，再推导映射后的 photon mode。优点是对称关系更直接；缺点是所得比较不自动等于固定实验室几何的磁畴翻转。

**C. 两层路线（推荐）**  
先把 A 定义为物理观测协议，再用 B 的 \(P/T/PT\) 关系做解释层和 selection-rule audit。优点是实验定义与数学对称性分开，同时保留二者联系；代价是要明确报告两种比较，不混用 domain odd 和 symmetry transformed。

推荐可以清楚说明，但**不得代替用户确认**。在用户选择前：

- 可以完成已冻结 \(P/T/PT\) 的代数审计与文献调查；
- 可以写出带条件标签的候选变换式；
- 不可以冻结 M0 的跨畴 \(\mathcal U_N\)；
- 不可以最终宣称固定实验室几何下 \(\beta,n,m\) 的必然奇偶性；
- 不可以据此生产挑选 H2/H5/H7/H9 或启动相应输出 covariance 数值。

---

## 5. 用户确认协议后的推导/计算顺序

收到用户对阶段 C 的选择后，再按该选择执行：

1. 固定磁畴比较定义、驱动变换和收集几何。
2. 从已审计的电流变换与电磁场量子化出发，推导 \(a_{\lambda,\pm z}(\omega)\) 的端口/偏振变换。对 \(T\) 的反幺正性质、正频率投影和复共轭必须逐步推，不可口头猜测。
3. 对归一化波包
   \[
   b_\mu=\int_0^\infty d\omega\,\phi_\mu(\omega)a_\lambda(\omega)
   \]
   推导模式变换，说明 \(\phi_\mu\) 是否也要变换。
4. 再推导 \(\beta_\mu\)、\(n_{\mu\nu}\)、\(m_{\mu\nu}\) 的对应关系，明确中心化定义和指标顺序。
5. 将旧笔记的
   \[
   (C_+-C_-)/2=C_{eo}+C_{oe}
   \]
   保留为有共同映射前提的代数恒等式；注意 \(C_{oo}\) 本身是 domain-even，不能命名为 domain-odd signal。
6. 从对称性允许/禁止的结构选择代表模式，再决定是否复用旧候选 H2/H5/H7/H9。旧候选不是新的生产批准。
7. 最后制定 Stage1P 接回输出通道的实施单元测试和小规模数值计划；P1 下不直接照搬 stationary、频率对角的 \(P1=0\) scattering 公式。周期/有限脉冲驱动可能混频并出现 Bogoliubov creation blocks，应由已批准的联合演化框架处理。

---

## 6. 暂停线 / 禁止事项

除非用户后续明确批准或出现有记录的矛盾，当前不要：

- 重开 P1=0 pole、IR、passivity、canonical commutator 的无边界审计；
- 重新选择或替换已经批准的 quasi-boson + explicit photon joint Gaussian 框架；
- 启动大规模 P1 数值、生产 covariance、squeezing、\(V_{\min}\)、Pauli 或纠缠；
- 把 4-band toy 当作 M0/CrI\(_3\) 的物理结论；
- 把 stationary P1=0 的 frequency-diagonal scattering 直接当作 P1 下的最终输入—输出公式；
- 未定义端口与实验几何就写 \(a_\mu(+N)=\pm a_\mu(-N)\)；
- 未冻结共同基底映射就逐矩阵元构造 \(J_e,J_o\)；
- 将“文献中允许”写成“本模型必然非零”；
- 根据先验期待强行选择谐波、改 Hamiltonian 或调参制造目标信号。

允许在此期间继续做的工程工作：文件/结果清点、已定义矩阵恒等式的审计、公式和单位复核、测试、文献核对、可复现的短报告。

---

## 7. 给 Cursor 的可直接执行任务文本

> 请继续 `Quantum-light` 线 Q 工作。先读取本交接文档第 3 节列出的仓库报告、2026-09-26 canonical channel 脚本与 JSON、M0 模型源码，以及 Obsidian 理论第三部分 §10。不要根据旧的总状态文件覆盖 2026-09-26 的线 Q 状态。
>
> 已批准的 P1 强驱动框架是 Stage1P：P1 强经典背景 + 输运 particle-hole quasi-boson + 显式 photon modes + 联合二次/Gaussian 演化。不要重新选择框架。P1=0 的 \(A\to a\) 及 \(C_I=L L^\dagger\) photon output-row 窄门已通过；其残差与适用范围见最新报告。\(n=m=0\) 是 creation block 被设为零的结构 control，不是生产 ground-state 证明；完整 photon+electron scattering unitary 也未构造。不要重开同类 P1=0 审计。
>
> 当前先执行两个不需用户选择的工作：  
> (1) 只读核对 M0 现有 \(P/T/PT\)、mirror/\(C_2\)、电流 polar-vector 变换、embedded Bloch sewing；可补最小纯代数回归测试，但不得改物理模型、参数或生产传播代码。  
> (2) 建立一页原始文献表，覆盖层状 AFM/CrI\(_3\) 磁畴依赖 SHG、量子 HHG 输出统计/纠缠、周期驱动玻色输入输出/Floquet Bogoliubov；对每篇说明其能支持与不能支持的项目结论。
>
> 完成后整理一份短审计记录，不启动 P1 数值、不生产 \(n,m,V\) 或 squeezing/entanglement。随后必须向用户询问磁畴比较协议：A 固定实验室几何直接比较 \(+N/-N\)；B 先采用明确数学 \(P/T/PT\) 映射；C 两层路线（推荐：A 作物理定义、B 作解释）。说明这会决定是否需要共同基底 \(\mathcal U_N\)、端口/偏振变换和 \(\beta,n,m\) 的奇偶关系。用户确认前，不冻结 \(\mathcal U_N\)、不宣称最终输出奇偶律、不按 H2/H5/H7/H9 开始生产计算。
>
> 保持变更范围窄。运行的测试需记录命令、输入文件和最大残差；不要覆盖已有结果或伪造历史运行身份。报告标注清楚哪些是旧结果、哪些是本轮新增验证、哪些仍是条件结论。

---

## 8. 本交接的验收标准

Cursor 的下一轮合格交付应包含：

1. M0 对称矩阵/反幺正算符及 Hamiltonian/current 变换的简明表格；
2. 与 Bloch sewing 和端口方向相容的残差结果，或明确指出尚缺的模型输入；
3. 有出处、有 DOI/arXiv 标识、有“支持/不支持”边界的原始文献表；
4. 一个明确的用户决策问题（A/B/C），说明不选择会阻塞哪些结论；
5. 无大规模 P1 运行、无未经批准的 covariance/纠缠生产结果、无对已批准 Stage1P 框架的重选。

**交接核心：**工程核查继续推进；磁畴比较协议、共同基底映射及其物理解释由用户参与决定。推荐 C，但不替用户冻结。
