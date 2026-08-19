# full112@k20 强度–occupation 扫描（SLURM 29452，2026-08-19）

正式标签：

\[
\boxed{\texttt{FULL112\_K20\_SAFE\_INTENSITY\_BRACKET\_BY\_CEP}}
\]

**不是**全局 \(I_{\rm safe}\)。在固定 CEP 与当前 full112 / k20 / dt=0.35 / T2=0.5 / `lg_cov` / 经典固定场配置下，已测最低强度 \(1.00\times10^{12}\,\mathrm{W/cm^2}\) 已同时 FAIL `POSITIVITY_GATE` 与 `TOP_BUFFER_GATE`。**无 \(I_P\)**，因此 **未形成有效括号**。不自动投 k40 / k50 / 完整 GH7。

机器可读归档：`Isafe_occ_29452_bracket.json`；`sacct` 快照：`Isafe_occ_29452_sacct.txt`。

## 执行核验

| 项 | 结果 |
|---|---|
| SLURM | `29452_[0-5]`，分区 `part_1`，6/6 `COMPLETED` ExitCode `0:0` |
| 墙时 | 37–43 min；节点 node08/09/10/01/02/03；36 CPU / 257173M |
| 模型 | full112、k20、`lg_cov`、`bsv_enabled=false`、`occ_stride=126`、band-resolved occupation |
| binary | `34edab1d…`（29175 pin） |
| +N TB | `66382a51…` |
| HEAD | `d1a14318015378df32fad6bbd7e3c95445263bb3` |
| OMP/MKL | 36/1 |
| 产物根 | `.../output_a0_full112_k20_isafe_occ_d1a143180153` |
| 每例执行 | `SUCCESS` + `run_status=PASS`；occupation 网格 1881600 行 = 42×20×20×112；迹 33600；`TRACE_GATE=PASS` |
| 自动下游 | **无** k40 / k50 / 完整 GH7 |

29403 的 \(I\approx2.813\times10^{12}\) FAIL 作为上端锚点复用，未重跑。

## 两门分报（不可合并）

- `POSITIVITY_GATE`：\(\rho_{nn}\in[-10^{-8},1+10^{-8}]\)
- `TOP_BUFFER_GATE`：顶部八带单态最大占据 \(\le10^{-3}\)，且顶部八带正人口 / 激发人口 \(\le1\%\)

只有两门都过，才可定义该 CEP 下的 overall tested-safe point。本批 **六个点 overall 全部 FAIL**。因此 \(I_{\rm pos,safe}\) 与 \(I_{\rm band,safe}\) 在已测集合中均为空。

## 验收表

CEP 265.307° 为 GH7 角相位；85.307° 与之相差正好 \(\pi\)（反相检查），**不是**广泛 CEP 覆盖。

| \(I\) (W/cm²) | CEP | POS | BUF | TRACE | overall | min \(f\) | max \(f\) | 首次越界 | `global_edge_max` | edge/exc | 112/105 | 边缘峰值 \(t\) |
|---:|---:|:---:|:---:|:---:|:---:|---:|---:|---|---:|---:|---:|---:|
| \(1.00\times10^{12}\) | 265.307° | FAIL | FAIL | PASS | FAIL | \(-1.34\times10^{-5}\) | 1.000046 | \(t=1.07\) fs，band 10，\(f=1.00000002\) | 0.0138 (band 106) | 2.44% | 0.971 | 42.70 fs |
| \(1.632\times10^{12}\) | 265.307° | FAIL | FAIL | PASS | FAIL | \(-1.33\times10^{-5}\) | 1.000039 | 同上量级，band 10 | 0.0281 (band 106) | 2.91% | 0.981 | 42.70 fs |
| \(1.90\times10^{12}\) | 265.307° | FAIL | FAIL | PASS | FAIL | \(-1.28\times10^{-5}\) | 1.000034 | 同上 | 0.0354 (band 106) | 3.42% | 0.980 | 42.70 fs |
| \(2.20\times10^{12}\) | 265.307° | FAIL | FAIL | PASS | FAIL | \(-1.21\times10^{-5}\) | 1.000035 | 同上 | 0.0445 (band 106) | 4.08% | 0.974 | 42.70 fs |
| \(2.50\times10^{12}\) | 265.307° | FAIL | FAIL | PASS | FAIL | \(-1.13\times10^{-5}\) | 1.000034 | 同上 | 0.0542 (band 106) | 4.82% | 0.964 | 42.70 fs |
| \(1.632\times10^{12}\) | 85.307° | FAIL | FAIL | PASS | FAIL | \(-1.73\times10^{-5}\) | 1.000027 | \(t=1.07\) fs，band 2 | 0.0264 (band 110) | 2.93% | 0.980 | 39.47 fs |
| \(2.813\times10^{12}\) | 265.307° | FAIL | FAIL | PASS | FAIL | — | — | 29403 锚点，未重跑 | 0.063–0.067 | — | ~0.95–1.00 | ~39–43 fs |

初态边缘人口均为 0；GH7c 轴末态即边缘人口峰值时刻。网格：6/6 `unique_complete_grid=true`，finite 检查通过，迹相对漂移 \(10^{-11}\)–\(10^{-12}\)。

正定性越界发生在 \(t\approx1\)–\(2\) fs、主要为低带；顶部人口峰值在 \(t\approx28\)–\(43\) fs。两者时间与空间分离，**不能**用越界噪声解释顶部占据，也 **不要**放宽正定性门。bands 105–112 的 112/105 正占据和比 \(\approx0.96\)–\(0.98\)，**无顶部衰减包络**。

GH7c 轴上 `global_edge_max` 与 edge/exc **随 \(I\) 单调变差**；正定性越界次数随 \(I\) 略减。两门仍是两个问题。

## 离散括号（禁止插值）

在 CEP \(\approx265.307^\circ\) 与当前数值配置下：

> 已测最低 FAIL 强度 \(I_F=1.00\times10^{12}\,\mathrm{W/cm^2}\)。没有已测 PASS 点，故 **未形成有效括号** \([I_P,I_F]\)。若安全边界存在，必在尚未测量的更低强度与 \(I_F\) 之间。

反相点 \(I=1.632\times10^{12}\)、CEP \(85.307^\circ\) 同样 FAIL（`global_edge_max` 0.0264 vs 同强度正相 0.0281）。这只说明该强度下反相也不安全，不能写成 CEP 扫描。

## 对 GPT 审查的采纳

1. 85.307° 与 265.307° 相差 \(\pi\)：只报告 `I_safe^{tested}(CEP\approx265.307^\circ)` 及其反相检查；不命名全局安全强度。
2. 正定性与顶部缓冲分门报告；本批两者都从最低点 FAIL。
3. 不拟合阈值；因无 \(I_P\)，写「未形成有效括号」。
4. 扫描脚本当时是未跟踪文件、经 `scp` 进入干净 `d1a1431` worktree。现已补 SHA：服务器上 sbatch/submit 与本地 LF 归一哈希一致（`7b76c40c…` / `64b3b072…`）；每例 `input.nml`、`run_metadata`、`run_status`、`output_sha256.txt`、occupation/`Jt`/`Et` SHA 见 JSON；`sacct` 已归档。`audit_isafe_occ.py` 当时不在服务器；occupation 表由 `.tmp/isafe_occ_remote_summary.py` 在登录节点解析。
5. `Q_NODE_MOMENTS=PASS`；`Q_TO_FIELD_EXPERIMENTAL_CALIBRATION=NOT_DONE`。当前映射是等平均驱动峰值 \(I_s=2I_{\bar{}}|\alpha_s|^2/(\langle n\rangle+1)\)，不是固定模式体积下的 \(\mathcal E_1\) 实验标定。\(\langle n\rangle/(\langle n\rangle+1)=0.973\) 的 2.7% 差值与 GH3→GH5 H2 模长约 3% 同量级，不能说「远小于求积误差」。
6. 探测量是 \(|J_x|^2+|J_y|^2\) 最近单 FFT-bin，即 \(C=I_{xy}\) 的 current-spectrum proxy；代码 FFT 约定 \(e^{-i\omega t}\)。
7. B0 只能标 `B0_SMALL_MODEL_PROTOTYPE=PRELIMINARY_PASS`。裸 `assert` 已改为 `require()`，`python -O` 下 9/9 仍会真正检查；代码仍未提交，也还没有开放费米三方对拍。

## 决策

- \(1.632\times10^{12}\) **明显 FAIL**（两个反相 CEP 都 FAIL，且更低的 \(1.00\times10^{12}\) 也 FAIL）。GH5 极端强度的 **模型空间**需要进一步核验；**不能**凭这两个相位否决整个 GH5（多数弱节点未测 occupation）。
- 没有得到该 CEP 下的 \([I_P,I_F]\) 括号。下一步若继续找括号：同一 pinned binary / k20 / 两门，**向下**扫（例如 \(2\times10^{11}\)、\(5\times10^{11}\)、\(8\times10^{11}\)），而不是往 2.81e12 再插，也不是自动投 k40/k50/完整 GH7。
- 扩 \(>112\) parent 仍是闭合顶部缓冲的另一条主线。

未投任何新作业。

## Provenance SHA-256（LF 归一）

| 对象 | SHA-256 |
|---|---|
| `Isafe_occ_29452_bracket.json` | `bae601ee627069747a2ba580fab45917e40015b7b4ac5f335e74085cfa6e3d0f` |
| `Isafe_occ_29452_sacct.txt` | `f6758891e46ab9fbb6e469c30e9db9266beaa898b4d0d7af0673a64850b2e7d0` |
| `sbatch_full112_isafe_occ_k20.sh` | `7b76c40cfeed4c1a2ccb24f8e591805997d23146557ac6e433ec98c972aafe8f` |
| `submit_isafe_occ_k20.sh` | `64b3b072c2b4033dc29982938e36645fdbfd5ac8f7607896c7aebc9461bb4be7` |
| `tools/analysis/audit_isafe_occ.py` | `1d92939552d32b36a128c0b81f96ceae995d7e261e7b1d69db1b60d03aac4764` |
| `.tmp/isafe_occ_remote_summary.py` | `dda998f2482527ecbf6c1fc25b4e7d39dd2619dbbe54a1bf7b4f438e585a021b` |
