# 线 Q：1600 nm 协议 A 精确 QED 四角 runner 只读审计（2026-11-02）

本轮是外审之后的 **只读代码审计**。没有开工四角，没有提取 A/B/C，没有改冻结参数，没有 git commit / push。

审计对象：

- `tools/b0_small_model/q8_m0_qinput_1600nm_protocolA_qed_fourangle_20261102.py`
- 它 patch 的演化本体 `...qed_fockaudit_20261102.py`（`run_one_job` / `fock_trace` / `propagate_ckpt`）
- Néel：`ground_pair_neel`、`build_shared`、`h8`、`onsite`
- Fourier：GH5 的 `rect_jq` / `h2_jx` / `fit_angle_model` / `even_odd`

独立判断先说清楚：**外审的分阶段设计是对的，但四项“开工前补充检查”不能照单全收。** 其中两项已经在代码里，一项把 Hann 窗写进了并不存在的约定，一项是论文措辞而不是开工缺陷。不要把这 7 条检查扩成第二套科学计划。

---

## 0. 现在实际停在哪里

| 层 | 状态 | 禁抄 |
|---|---|---|
| Q6 24.8 μm 精确 QED | 机制候选已闭合 | 4.98、110.6°、\(\rho=1.456\)、\(\arg(A/B)=177.65^\circ\) 是另一套核 |
| 1600 nm GH5 | 四角 + \(\pi/4\) 留出完成 | 8.778、\(\rho=1.274\)、179.04° **不是**精确 QED |
| 1600 nm 精确 QED 截断 | **仅** `n8_p_q`（\(\theta=\pi/4\)，\(N=+1\)）通过 | Weyl 7.91 \(\neq\) Fock \(\langle n\rangle\) |
| 1600 nm 精确 QED 四角 | **未开** | 不得提取 A/B/C |

`--analyze-only`：0/8。无 `--authorize-stage1` 的 `--mode fourangle` 退出码 3。裸 `--job p_0` 拒绝。结果目录未污染 fockaudit。

`freeze-fit` **仍是 stub**：8 轴全 ok 也只写 `fourangle_fit.json.blocked`，不写 `fourangle_fit.json`，退出码 3。这是已知缺口，**不是现在跑生产的理由**，也不是现在去实现拟合的理由。

---

## 1. 外审四项补充：哪些该收，哪些不该收

| 外审要求 | 代码事实 | 本轮处置 |
|---|---|---|
| 每轴不只 \(P_8\)，还要 \(P_7+P_8\)、\(\langle n\rangle\)、\(\|\psi\|^2-1\)、是否完整/NaN | `fock_trace` **已经**写 `max_top_fock`、`max_top_two`、`max_n_ph`、`min_n_ph`、`end_n_ph`、`max_abs_norm_minus_1`。演化非有限态会抛错，`status=failed`。JSON `allow_nan=False`。闸门仍是 \(P_8<0.05\)，这是报警不是收敛定理 | **收**：契约写全字段。`job_row` 补了 `finite_state` / `covers_requested` / `error` / `n_samples`（原先只在 `job_*.json` 里）。**不**扩 Hilbert 空间 |
| 两磁畴除 Néel 外必须同协议 | `neel` 只进 `electron_ground_state`/`h8` 的交换场 \(\pm N\Delta\)，以及 `onsite_determinant_energy`。SOC \(\lambda g\) **不**乘 Néel。Peierls / 电流分组不吃 `neel`。光子态只随 \(\theta\)。\(k\)、\(\omega\)、\(g\)、1+4+1、\(n_{\max}\)、步数由同一 `JOB_TABLE` 锁死 | **收**：确认配对。**不**新写对称性扫描 |
| 冻 Fourier，且与 GH5 同一套；外审点了 Hann | GH5 主窗是矩形 `[2T0,4T0]` 的 `rect_jq`，约定 \(J_q=\frac{1}{t_1-t_0}\int e^{+iq\omega_0 t}J(t)\,dt\)。`hann_jq` 在 odd 模块里存在，**1600 nm GH5 和四角 runner 都没用** | **不收 Hann**。阶段一本来就不拟合。等真做 freeze-fit 时必须 import 同一套 `q2.rect_jq` / `h2_jx` / `even_odd` |
| 留出通过 \(\neq\) 高阶 \(\Delta n=\pm4,\pm6\) 不存在 | 四角 runner 阶段一禁止拟合；阶段三才允许留出。这是论文表述，不是开工前代码缺陷 | **不改代码**。留出失败再查高阶，不把留出塞回拟合 |

三问仍是待验假设。Q6 的 \(177.65^\circ\) 与 GH5 的 \(179.04^\circ\) **不能**当同一套拟合比接近。

---

## 2. 外审 7 条检查 × 代码位置

### 2.1 两个 Néel 是否只在预定磁性项上翻转

| 位置 | 做什么 |
|---|---|
| `q8_m0_model.h8` | `exch = ±neel * p.delta`；SOC 是 `lam * g`，与 Néel 无关 |
| `q8_m0_qinput_minimal_20261020.onsite` | 对角交换场 `neel * delta * layer_sign * spin_sign` |
| `dyn.onsite_determinant_energy` | 把上述对角写进 36 维 Slater 能量 |
| `sh.build_shared` | `neel` 只用于 onsite；`_accumulate` 的 Peierls \(q\) 组和电流不取 `neel` |
| `odd.ground_pair_neel` | `electron_ground_state(..., neel)`，即 `h8(..., neel)` 的占据 |
| `fa.run_one_job` | 同一 `theta`、同一 `photon_raw("ds", n_max, theta)`；只把 `neel` 传给地面态和 `build_shared` |

风险：SharedModel 动力学的 onsite 是交换场，k 依赖 SOC 形态因子在 `h8` 里；这是 Q6 既有结构，**两边磁畴相同**，不构成协议差。本轮不重开 SOC 扫描。

Néel-odd 定义与 GH5 一致：

```
even_odd(jp, jm) = (jp+jm)/2, (jp-jm)/2
```

即 \(J_{\mathrm{odd}}=(J_{+}-J_{-})/2\)。只算 \(N=+1\) 得不到。

### 2.2 四角输入态、压缩角、频率、g、k、包络、时间步

`JOB_TABLE` 锁死 10 个标签。`fa` 被 patch 的 `OUT`/`PROTOCOL`/`job_spec`，**不会**跑 fockaudit 的 `n8_p_q`。`make_env(0.775)`，`assert_protocol_omega` 禁止 `pf.OMEGA0=0.05`。`g=0.005`，`did_not_scale_g=True`。`K_FRAC=[0.173,0.271]`。`n_steps=11154`。1+4+1。

风险：`--n-steps` 可以覆盖步数。生产命令不得带这个开关。契约已禁止用 200 步冒充生产。

### 2.3 每轴是否独立记录顶层、两层、光子数、范数

是。`fock_trace` 已有；`job_*.json` 展开这些字段。本轮只让 `job_row`/`fourangle_status.json` 把 `finite_state`、`covers_requested`、`error`、`n_samples` 也抄出来。0.05 仍只报警 \(P_8\)。

### 2.4 阶段闸门

| 阶段 | 代码 | 行为 |
|---|---|---|
| 无授权 fourangle | `drive` → `refuse`，exit 3 | 已测 |
| 裸 `--job` | 无 `QED_FOURANGLE_AUTHORIZE` → refuse | 已测 |
| 阶段一 | 禁止拟合；`analyze` 不含 A/B/C | 正确 |
| freeze-fit | 8 轴未齐 refuse；齐了也 `freeze_fit_blocked`，**不写** `fourangle_fit.json` | 已知 stub |
| 阶段二 | 要 `--authorize-stage2` **且** `fourangle_fit.json` 存在 **且** 8 轴 ok | 现在走不通，符合“先冻拟合” |
| 阶段三 | `--mode leaveout` 一律 refuse，直到以后实现 | 正确 |

`job_ok_no_alarm` 比 fockaudit 的 `job_finished_ok` 更严：必须 `status==ok` 且 `fock_alarm_0p05 is False`。`ok_fock_alarm` 不算通过。这是对的。

注意：子进程 `--job` 在 `ok_fock_alarm` / `timeout_partial` 时 exit 0，但父进程 `analyze` 后仍会因报警返回 2、因未齐返回 1。不要用子进程码当四角通过。

### 2.5 Fourier / Hann / 符号

阶段一 **故意不拟合**。约定在 GH5：

- 窗：主 `[2T0,4T0]`，前 `[T0,3T0]`，后 `[3T0,5T0]`
- 矩形，不是 Hann
- \(e^{+iq\omega t}\)
- 归一化除以窗长
- H2 读 `Jx` 的 `q=2` 槽（`h2_jx` 取 `Jx[1]`，因为 `qs=1..6`）
- \(k/-k\) 共享模式；奇通道 `even_odd`

**不要给精确 QED 另做一套 Hann。** 将来 freeze-fit 必须调用同一 `q2` 函数，而不是手写积分。

### 2.6 失败、超时、续跑、覆盖

| 情况 | 行为 | 风险 |
|---|---|---|
| 已 `ok` 且无报警 | 串行跳过，不重跑 | 好 |
| 已 `ok_fock_alarm` | 跳过发射，不调 \(g\) | 好 |
| timeout | ckpt 续跑同一轴 | 好 |
| 最终 JSON 含 NaN | `dump(..., allow_nan=False)` 会失败，job 可能停在 `running`，下次从 ckpt 续 | 可接受；不要改成允许 NaN |
| `refuse()` | 总是覆盖 **同一个** `refused_not_authorized.json` | 小；不碰 `job_*.json`（未知 tag 除外） |
| 已 `ok` 的 job | `run_one_job` 直接返回，不覆盖 | 好 |
| 拷 fockaudit `n8_p_q` | runner 不用那个目录，标签也不同 | 好 |

并发：串行。单轴约 30 min，8 轴约 4 h。禁止 8 路并行。

### 2.7 四角拟合与 \(\pi/4\) 对高阶 \(\Delta n\)

现在没有拟合代码路径。三项截断是以后阶段三的表述问题。留出点不得进拟合。本轮不写 \(\Delta n=\pm4\) 分解器。

---

## 3. 真正的代码缺口（按要不要现在改）

| 缺口 | 现在改？ | 原因 |
|---|---|---|
| `job_row` 没抄 `finite_state` 等 | **已改，最小** | 数据本来就在 job JSON 里，只是分析界面漏了 |
| `freeze-fit` stub | **不改** | 改它就是提取 A/B/C；8 轴还没跑 |
| 时间网格配对检查 | **不改** | 同 `n_steps`、同 `omega0` 由构造保证；有 series 之后再比，不要现在空转 |
| 给四角加 Hann | **禁止** | 外审想当然 |
| 默认 `nmax=10` | **禁止** | 报警后再开对照目录 |
| 把 10 轴审计写成新课题 | **禁止** | 科学问题仍是那三问 |

---

## 4. 本轮结论

1. 闸门 runner 按契约拒绝开工，协议锁定正确。
2. 外审要的截断观测量，演化层已经在记；0.05 仍只是报警。
3. Néel 翻转只动交换场，两磁畴其余协议同源。
4. 与 GH5 比相位时必须用矩形 `rect_jq`，**不要 Hann**。
5. `freeze-fit` 故意未实现。阶段一不需要它。
6. **仍不授权阶段一。** 下一句用户令必须是「授权阶段一 / 开工四角」才会串行 8 轴。

不宣称 1600 nm 精确 QED 机制成立。不把 GH5 8.778 抄过来。不调 \(g\)。
