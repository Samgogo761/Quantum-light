# P 共轭 -N TB 构造完成（2026-06-02）

本文件记录"严格路线"第 2 步的完成情况：从 `+N` 的 Wannier TB 直接后处理构造一个
**算符意义上严格反演（P）共轭**的 `-N` TB，用于替换此前两次独立 SCF+Wannier 得到
的、并不严格配对的 `-N`。前置诊断见 `AFM_Neel配对失败诊断_20260602.md`。

## 1. 物理依据

双层 CrI3 的非磁几何近似中心对称，反演中心（分数）约
`c = (0.499998583, 0.333333067, 0.372004766)`。`+N` 与 `-N` 原子坐标完全相同，仅
Cr1/Cr2 的初始磁矩方向对调（`+N`: Cr1 angle1=0, Cr2 angle1=180；`-N` 反之）。

空间反演 P 交换上下层（Cr1<->Cr2 及对应 I），并把各原子自旋一起带走。自旋是轴矢量，
P **不翻转**自旋；Neel 矢量的反转完全来自层交换。因此 P 把 `+N` 基态映到 `-N` 基态。

Wannier 函数在 P 下的映射：
```
1. P 把每个原子 X 配对到其反演像 P(X)。
2. 原子内 Wannier 排列是 (轨道 m) x (自旋)，P 同时保持二者
   （d 偶宇称 -> 同一个 d，符号 +1；p 奇宇称 -> 同一个 p，符号 -1）。
   所以原子内序号不变，整个置换仅由原子配对决定。
3. 宇称符号 eta_a = +1（Cr-d），-1（I-p）。
```

算符变换（Wannier90 存储约定，eta_a^2 = 1）：
```
H_{-N}(R)_{ab} =  eta_a eta_b  H_{+N}(-R)_{P(a)P(b)}
r_{-N}(R)_{ab} = -eta_a eta_b  r_{+N}(-R)_{P(a)P(b)} + 2 c_cart * delta_ab delta_{R,0}
```
位置算符是极矢量（P: r -> 2c - r），故有限 R/非对角部分多一个负号，原胞内对角加
`2c_cart`（把 -N 的 Wannier 中心放到正确的反演位置）。`2c` 仅影响带内/对角位置，对长度
规范电流是全局平移规范，但仍保留以保证中心物理正确。

每个 Wannier 函数的 Wigner-Seitz 平移只产生 k 依赖的对角规范相位 `e^{-ik(L_a-L_b)}`，
不改变本征值与规范不变偶极；且 `ndegen(-R)=ndegen(R)`，故复用 `+N` 的 R 列表/ndegen 严格成立。

> 关键发现：外部 AI 最担心的"同原子多 spinor 的局域酉混合"在本体系**不存在到可测水平**。
> 简单的"原子置换 + 宇称符号 + R->-R"已使本征值匹配到 1e-14，说明 Wannier 函数足够接近
> 纯原子轨道，无需局域酉旋转。

## 2. 脚本

```text
C:\Users\26507\Documents\New_SBEs\Quantum-light\tools\analysis\construct_p_conjugate_tb.py
```

输入：`+N` 的 `CrI3_tb.dat`、`CrI3.win`、`CrI3.wout`。
输出：`-N` 的 `CrI3_tb.dat`（H 块 + 位置块，Wannier90 格式）。

运行命令：
```powershell
& 'C:\Users\26507\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'C:\Users\26507\Documents\New_SBEs\Quantum-light\tools\analysis\construct_p_conjugate_tb.py' `
  --plus-tb   'C:\Users\26507\Documents\量子光研究\wannier\CrI3_tb.dat' `
  --plus-win  'C:\Users\26507\Documents\量子光研究\wannier\CrI3.win' `
  --plus-wout 'C:\Users\26507\Documents\量子光研究\wannier\CrI3.wout' `
  --out       'C:\Users\26507\Documents\量子光研究\CrI3_TB_mN_Pconj\CrI3_tb.dat'
```

## 3. 验证门（全部通过）

```text
[A] 原子配对：max/mean 分数距离 = 4.764e-05 / 2.783e-05；16 原子干净配对
[B] 置换：duplicate=0, non-involutive=0（严格对合）
D1 本征值门  E_-N(k)=E_+N(-k)：worst = 1.33e-14 eV         PASS
D2 厄米性    -N=2.051e-03 Ang，与 +N 基线完全一致（继承）   PASS
D3 Wannier中心 |center_-N(a)-(2c-center_+N(P(a)))| = 0.0    PASS
D4 带间偶极（规范不变）|d^band_-N(k)| vs |d^band_+N(-k)|    worst rel = 4.3e-12  PASS
```

写出文件后用 `compare_tb_eigenvalues.py` 独立回读复核：
```text
反演配对 E_+N(k) vs E_-N(-k)：所有高对称点 ~1e-14 eV   （确认严格 P 共轭）
同 k    E_+N(k) vs E_-N(k)  ：G/M ~1e-14；K 点 5.8e-5~5.9e-2
        —— 这是 +N 磁序真实的反演破缺 E_+N(k)!=E_+N(-k)，不是 bug。
```

输出文件：
```text
C:\Users\26507\Documents\量子光研究\CrI3_TB_mN_Pconj\CrI3_tb.dat   (~1.21 GB)
```
头部（注释/晶格/nwann/nrpts/ndegen）与 R 块结构与 +N 一致，Fortran `read_tb_file` 的
晶格校验可通过；列表定向读取对数值格式宽容。

## 4. 与旧 -N 的对照

```text
旧 -N（独立 SCF+Wannier，C:\...\CrI3_TB-N\CrI3_tb.dat）:
   与 +N 的 Gamma/K/M 本征值差 50-110 meV  -> 非严格配对（被 DFT 误差污染）
新 -N（本脚本 P 共轭构造）:
   与 +N 的 E_+N(k)=E_-N(-k) 差 ~1e-14 eV  -> 严格配对
```

旧 -N 数据保留作为"独立 SCF/Wannier 未配对示例"，不用于最终选择定则证明。

## 5. 下一步

```text
1. 上传新 -N TB 到服务器：
   scp CrI3_TB_mN_Pconj\CrI3_tb.dat
       -> /public/home/wangjs/project/CrI3_TB_mN_Pconj/wannier/CrI3_tb.dat

2. 跑 -N SBE（与 +N 同代码、同参数）：
   TB_FILE=/public/home/wangjs/project/CrI3_TB_mN_Pconj/wannier/CrI3_tb.dat \
   OUTROOT=.../output_afm_pconj_T2_0p5cycle \
   T2_CYCLES=0.5 sbatch --array=1,2 deploy/run_lg_cov_fullvalence_scan.sh

3. 用 afm_neel_symmetry_check.py 重做选择定则。
   预期：若 H2 evenN 泄漏显著下降（leakage < ~0.1），则证实此前 H2 异常来自
   DFT 配对误差，SBE 求解器 + lg_cov + k40 离散化通过真正考验，可进入 BSV。
   若仍异常，才回头审视求解器（概率很低）。
```

注意：此构造把选择定则检查从"对 DFT 的检验"转变为"对 SBE 代码 + lg_cov 规范 + 离散化
精度的检验"，因为 -N 现在是 +N 的精确算符配对。这正是隔离 H2 异常根因所需要的。
