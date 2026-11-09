# A1-H：干净 SCF → 6 点 NSCF（工程已自动门，已批准提交）

隔离目录：`/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf/`

**禁止**写回 `temp_scf_soc`。**禁止**手工修 XML。A1-H PASS **不等于** WINDOW144 HHG 已可信。

## 6 个 \(k\)

| 顺序 | 标签 | \(\mathbf k\) | 角色 | 24×24 |
|---|---|---|---|---|
| 1 | `k_ctrl` | \((0,0)\) | 控制 | 在网格上 |
| 2 | `k_eta` | \((0.10,0.55)\) | \(\Pi r Q\) | 离网格 |
| 3 | `k_occ` | \((0.80,0.05)\) | 边占据 | 离网格 |
| 4 | `k_bg` | \((0.35,0.75)\) | 背景 | 离网格 |
| 5 | `k_{H5}` | \((0.45,0.55)\) | H5 候选 | 离网格 |
| 6 | `k_{H7}` | \((0.45,0.05)\) | H7 候选 | **两坐标都离网格** |

H7 静态分：\((0.50,0)=3.557\)（在母网格，不用）、\((0.45,0)=3.500\)、\((0.45,0.05)=3.442\)（差 1.6%）。选后者做插值测试。

MPI（对齐 4 月 SCF）：16 ranks，**`-nk 2`**（`npool=2`，每池 8 个 G 空间进程）。不要 `-nk 4`：电荷密度按池复制，`ecutrho=600` 时 32092 在 node02 上吃到 ≈239 GiB 后 SIGKILL。不要 `16` + `-nk 6`。

## 自动门

1. 赝势 sha256 必须等于 4 月生产 UPF（`a1h_provenance_targets.json`）。
2. SCF 输入除 `outdir` 外必须与 `inputs/CrI3_scf_soc.in` 一致。
3. SCF 结束后必须 `A1H_SCF_PROVENANCE_GATE=PASS`（总能、Fermi、\(M_{\rm tot}\)、\(M_{\rm abs}\)、Hubbard、JOB DONE、schema=`scf`），否则 **NSCF 不启动**。
4. April 的 `verbosity=low` **没有**逐原子 \(M_{\rm Cr}\)；磁门用 \(M_{\rm tot}\approx0\)、\(M_{\rm abs}=20.42\)。若 QE 打印了 site moment 只记录，不作为硬条件。
5. NSCF 后 \(\Gamma\) vs `CrI3.eig`：`A1H_GAMMA_EIG_GATE`。FAIL 则离网格差 **不能**解释成插值。

脚本：默认走 **part_1** `sbatch_a1h_scf_then_nscf_part1.sh`（31876 part_2 OOM；32092 part_1 `-nk 4` SIGKILL，无 save）。运行时在不改 April 物理输入的前提下覆盖 `disk_io=high` 并允许同作业 restart。未完成的 A1-H 不要解释成插值结论。

## A1-O（MMN/Berry \(r\)，已批准）

同一隔离目录、同一 32293 电荷。**不要**和 A1-H 的 `disk_io=low` 6 点混关。NSCF 必须把波函数写到盘上（`disk_io=high`）。

| 项 | 值 |
|---|---|
| 作业 | `sbatch_a1o_mmn_star_part1.sh`（NSCF `-nk 2`）；pw2wannier90 **禁止** `-nk`（QE 7.2 `pools not implemented`）。32446 NSCF 过、pw2wan 失败；续跑 `sbatch_a1o_pw2wan_only_part1.sh` |
| \(k\) | \(\Gamma\)（方法控制）+\(\eta\)+H5+H7，各 12 个 \(\pm\delta\) 卫星，共 52 点 |
| \(\delta k\) | 晶体坐标 0.002 / 0.005 / 0.010，对称差分 |
| 主线 | `pw2wannier90` `.mmn` → Berry \(r\)；不编译速度矩阵元 |
| 比较 | 孤立带 \(\lvert r_{mn}\rvert\)；近简并 \(\Pi Q\) 块 Frobenius/SVD；占据 \(\Pi\Pi\) 控制 |
| 禁止 | 写回 `temp_scf_soc`；用 \(\partial_k H_{\mathrm{TB}}\) 当 QE 真值；宣布 HHG 机制 |
