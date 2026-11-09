# 规格：显式 \(U_P(\mathbf k)\) 与 \(\varepsilon_P\) 闸门

**日期**：2026-07-28  
**必须遵守**：`量子光理论核心` 中磁畴协变关系；审校报告关于「先 \(U_P\) 再 ±N 相加」的纪律。  
**实现**：同目录 `verify_epsilon_P.py`

---

## 0. 科学目标

证明（或量化破缺）双层 AFM-CrI₃ full112 Wannier 哈密顿量满足

\[
\boxed{
U_P(\mathbf k)\,H_{+\mathbf N}(\mathbf k)\,U_P^\dagger(\mathbf k)
=
H_{-\mathbf N}(-\mathbf k)
}
\tag{1}
\]

其中 \(U_P\) 由**晶体空间反演在 Wannier 基底上的表示**构造，**禁止**从 \(H\) 拟合/搜索幺正矩阵。

相对误差（Frobenius）：

\[
\boxed{
\varepsilon_P(\mathbf k)
=
\frac{
\|H_{-\mathbf N}(-\mathbf k)
-U_P(\mathbf k)H_{+\mathbf N}(\mathbf k)U_P^\dagger(\mathbf k)\|_F}
{\|H_{-\mathbf N}(-\mathbf k)\|_F+\delta}
}
\tag{2}
\]

---

## A. 结构前提（步骤 1）

| 检查项 | 要求 | 记录位置 |
|--------|------|----------|
| 反演中心 \(\mathbf c\)（分数坐标） | 与 `construct_p_conjugate_tb` 一致；量化残差 | 本规格结果表 |
| 非磁点群 | 目标 \(C2/m\)（\(C_{2h}\)）；若破缺，报告最大原子位移 | 同上 |
| +N/−N 原子坐标 | 相同；仅 Cr 层磁矩对调 | 既有 |

**默认中心（历史值，须用实际结构复核）**：

\[
\mathbf c \approx (0.499998583,\; 0.333333067,\; 0.372004766)
\]

磁点群提醒：\(P\) 本身不是单畴对称操作；\(P:+\mathbf N\to-\mathbf N\)，\(\Theta P\) 才保单畴。

---

## B. 轨道反演配对表（步骤 2）

输入：

- `CrI3.win` / `CrI3.wout`（或 `wannier90_centres.xyz`）
- 投影顺序：Cr-\(d\times2_{\rm spin}\) 在前，I-\(p\times2_{\rm spin}\) 在后（full112）

对每个 Wannier 指标 \(a=0,\ldots,N_w-1\)：

1. 原子质心 \(\boldsymbol\tau_a^{\rm atom}\)（分数）  
2. 像 \(\mathbf c_{\rm img}=(2\mathbf c-\boldsymbol\tau_a^{\rm atom})\bmod 1\)  
3. 同种原子最近邻 → 原子配对 \(A\to A'\)  
4. 原子内序号（轨道×自旋）不变 → 置换 \(a\mapsto a'=P(a)\)  
5. 晶格矢量：

\[
-\boldsymbol\tau_a+\mathbf t_{\rm inv}
=\boldsymbol\tau_{a'}+\mathbf L_a,
\quad \mathbf L_a\in\mathbb Z^3
\]

（实现：\(\mathbf L_a=\mathrm{round}\big((2\mathbf c-\boldsymbol\tau_a)-\boldsymbol\tau_{a'}\big)\)，残差须 \(\ll 1\)）

6. 轨道宇称对角元 \(D_P\)：Cr-\(d\)：\(+1\)；I-\(p\)：\(-1\)；自旋 \(s_0\)（反演不翻自旋）

**验收**：

- \(P\circ P=\mathrm{id}\)（对合）  
- 无重复像  
- 原子配对分数距离 \(\max\lesssim 10^{-4}\)（与 2026-06-02 门一致）

---

## C. 构造 \(U_P(\mathbf k)\)（步骤 3 前半）

复合指标 \(a\) 已含自旋—轨道时：

\[
\boxed{
\big[U_P(\mathbf k)\big]_{a'a}
=
e^{2\pi i\,\mathbf k\cdot\mathbf L_a}\,
\eta_a\,
\delta_{a',P(a)}
}
\tag{3}
\]

其中 \(\eta_a=\pm 1\) 为宇称。应验证：

- \(U_P^\dagger U_P=I\)（数值）  
- \(U_P(\mathbf k)\) 在 \(\mathbf k\to\mathbf k+\mathbf G\) 下规范一致（相位惯例固定即可）

**禁止**：用本征矢重叠去“定义” \(U_P\)（那是 sewing matrix 诊断，不是本闸门的构造定义）。Sewing matrix 可作为**可选对照**：

\[
B_P^{mn}(\mathbf k)
=
\big\langle u^{-\mathbf N}_{m,-\mathbf k}\big|\hat P\big|u^{+\mathbf N}_{n,\mathbf k}\big\rangle
\]

仅当需要与 DFT 态对齐时另开任务。

---

## D. \(\varepsilon_P\) 全网格验证（步骤 3）

### D.1 \(H(\mathbf k)\) 定义

Wannier90 约定：

\[
H(\mathbf k)
=
\sum_{\mathbf R}
\frac{e^{2\pi i\mathbf k\cdot\mathbf R}}{N_{\rm degen}(\mathbf R)}
H(\mathbf R)
\]

再厄米化 \(\tfrac12(H+H^\dagger)\)。

### D.2 −N 哈密顿量来源（二选一，须在报告中写明）

| 模式 | 含义 | 用途 |
|------|------|------|
| `pconj` | \(H_{-\mathbf N}\) 由 +N 经实空间 P 共轭公式生成（与 `construct_p_conjugate_tb` 同一公式） | **自洽性**：\(\varepsilon_P\) 应达机器噪声；检验 \(U_P\) 相位/\(\mathbf L_a\) |
| `file` | 读入磁盘上的 −N `*_tb.dat`（P 共轭产物或独立 SCF） | **资产审计**：生产用 −N 是否满足式 (1) |

生产声明必须对 **实际用于 HHG 的 −N TB** 跑 `file` 模式。

### D.3 \(k\) 网格

- 默认：与 SBE 生产一致的均匀网格（如 \(40\times40\times1\)），含 \(\Gamma\)  
- 另报高对称点：\(\Gamma,M,K\) 及对应 \(-\mathbf k\)

### D.4 通过阈值（初值，可调）

| 统计量 | PASS | WARN | FAIL |
|--------|------|------|------|
| \(\mathrm{median}\,\varepsilon_P\) | \(<10^{-10}\) | \(<10^{-6}\) | 否则 |
| \(P_{99}\,\varepsilon_P\) | \(<10^{-8}\) | \(<10^{-4}\) | 否则 |
| \(\max\,\varepsilon_P\) | \(<10^{-6}\) | \(<10^{-3}\) | 否则 |

若 `pconj` 模式 FAIL → \(U_P\)/\(\mathbf L_a\)/宇称实现有 bug。  
若 `pconj` PASS 而 `file` FAIL → 磁盘 −N 不是严格 P 共轭（或读错文件）。

### D.5 附属谱门（兼容旧门，不替代 \(\varepsilon_P\)）

\[
\max_n\big|E_n^{-\mathbf N}(\mathbf k)-E_n^{+\mathbf N}(-\mathbf k)\big|
\quad\text{目标 }<10^{-9}\,\mathrm{eV}
\]

谱门通过而 \(\varepsilon_P\) 失败 ⇒ 存在非平凡规范/相位问题，必须修 \(U_P\)，不能用谱门搪塞。

---

## E. HHG 与 \(J_e/J_o\)（步骤 4）

仅在 D 对**生产 −N 文件** PASS/WARN 可接受后，声明：

\[
J_n^{\rm even/odd}
=
\frac{J_n(+\mathbf N)\pm J_n(-\mathbf N)}{2}
\]

约束：

- **同一实验偏振通道**（当前生产：线偏 \(x\)，同一 CEP/包络/\(T_2\)）  
- −N 必须是经 P 映射的共同 Hilbert 空间对象  
- 不把本步骤延伸为 \(C_{oo}\) 或量子优势

既有 M88-c20 ±N 结果可继续使用，但论文/汇报须引用本闸门报告编号。

---

## F. 输出物（每次运行必交）

```text
outdir/
  pairing_table.csv          # a, a', eta, Lx,Ly,Lz, residual
  structure_summary.md       # c, pair distances, involution
  epsilon_P_kgrid.csv        # kx,ky,kz, eps_P, ||H||_F
  epsilon_P_stats.md         # median/P99/max + PASS|WARN|FAIL
  U_P_check_unitarity.md     # max ||U†U-I||
  optional: eps_P_map.png
```

---

## G. 与评分器的关系

| 工具 | 回答的问题 |
|------|------------|
| `verify_epsilon_P.py` | ±N 是否算符级磁畴协变？ |
| `score/score_reduced_model.py` | 少带模型相对 QE/full112 的静态优劣？ |

二者正交：评分 PASS 不能代替 \(\varepsilon_P\)；\(\varepsilon_P\) PASS 不保证少带 HHG 可用。
