# tools/q_backend — 线 Q backend 路由层（与材料线 CPU SBE 分离）

```text
PHYSICS      = tools/b0_small_model/pt_afm_toy.py   (唯一裁判)
SPEC         = tools/b0_small_model/Q_THEORY_FROZEN_SPEC.md
THIS_DIR     = 路由 + CPU wrapper + GPU stub + profiling；不含任何方程
GPU_STATUS   = NOT_IMPLEMENTED_UNTIL_CPU_PROFILE
MATERIAL_SBE = 不在此目录；不要把 lg_cov / 112 带 / 生产 T2 接进来
```

| 文件 | 作用 |
|---|---|
| `q_backend.py` | `QBackend` 协议、`CPUReferenceBackend`（直接调用 `pt_afm_toy`）、`GPUBackendStub`、GPU 验收协议 |
| `test_q_backend.py` | wrapper 与 reference 逐位相等；stub 拒绝运行；无禁用旋钮 |
| `profile_q_cpu.py` | 真实 workload 计时与内存 → `results/q_line/profile_q_cpu_<date>.json`，给出 `GPU_PRIORITY` |

运行：

```bash
py -3 tools/q_backend/test_q_backend.py
py -3 tools/q_backend/profile_q_cpu.py
```

将来若写 GPU backend（CuPy / PyTorch complex128 / JAX x64），放在本目录，按 `GPU_ACCEPTANCE_PROTOCOL` 对拍 CPU reference；不改 `tools/b0_small_model/` 的物理。
