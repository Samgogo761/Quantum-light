# 1600 nm 精确 QED 四角：`job_row` 最小字段说明（2026-11-02）

只读审计报告写本轮为只读，同时又写 `job_row` / `fourangle_status` 增加了状态字段。外审要求留一份最小 diff，确认没有触及物理演化。本页只做这件事。

## 结论

四角 runner 的 `job_row()` **只读** fockaudit 已经写入 `job_<tag>.json` 的字段，再抄进 `fourangle_status.json`。没有改 `propagate_ckpt`、Peierls Hamiltonian、交换场翻转、Fourier、\(g\)、\(\omega\)、\(n_{\max}\) 或时间步。

新增的只是状态摘要里多抄了四项：

| 字段 | 来源 | 作用 |
|---|---|---|
| `finite_state` | fockaudit `run_one_job` 已写 | 演化是否保持有限态 |
| `covers_requested` | 同上，`completed >= n_steps` | 是否跑满 11154 步 |
| `n_samples` | 同上 | 采样条数 |
| `error` | 同上 | 失败原因；通过轴应为空 |

`fourangle_status.json` 只是把上述摘要按 8+2 轴摊开。无授权时 `--job` 仍拒绝开工。

## 不是什么

- 不是新的物理闸门，也不是 n8 vs n10 收敛证明。
- 不授权提前拟合 A/B/C，不改变失败即停纪律。
- 本页不是开工令；开工令是外审对阶段一的明确授权。
