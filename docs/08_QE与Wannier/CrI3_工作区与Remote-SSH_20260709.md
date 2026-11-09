# CrI3 工作区与 Remote-SSH（2026-07-09）

---

## 工作区分工

| 工作区 | 用途 |
|--------|------|
| `量子光研究/` | QE/Wannier 输入、tb.dat 副本、pilot_40wf、本地分析脚本 |
| `New_SBEs/Quantum-light`（Git） | SBE 代码、`deploy/`、**`docs/` 权威文档** |
| 服务器 | 计算；路径见下 |

---

## SSH 登录（Windows Cursor / OpenSSH）

```text
本机 → physics@202.119.84.63:66（网关）
     → wangjs@192.168.7.110:22（master01）
```

- 配置：`C:\Users\26507\.ssh\config`，Host 别名 `cri3-login`
- **勿用** ControlMaster（Windows + ProxyJump 会报错）
- Agent 自动化：本机 `id_cri3_ed25519` 公钥已写入两跳 `authorized_keys`

---

## SLURM 分区（2026-07 实测）

| 分区 | 节点 | 规模 | 用途 |
|------|------|------|------|
| **part_1** | node01–10 | 36 核 / ~257 GB | **重活**：full112/nb104 HHG、BSV、大 QE |
| **part_2** | node11–14 | 8 核 / ~64 GB | **轻活**：Wannier、**40-WF 收敛**、pw2wan |

分流细则：[CrI3_part1与part2任务分流_20260714.md](CrI3_part1与part2任务分流_20260714.md)

---

## 服务器路径

```text
/public/home/wangjs/project/CrI3_TB/wannier/              # 112-WF
/public/home/wangjs/project/CrI3_TB/wannier/pilot_40wf/   # 40-WF
/public/home/wangjs/project/CrI3_TB/temp_scf_soc/        # QE nscf save
/public/home/wangjs/project/New_SBEs/Quantum-light/     # SBE + deploy
/public/home/wangjs/software/q-e-qe-7.2/                  # QE + wannier90
```

本地说明副本：`量子光研究/04_工具与环境/Remote-SSH配置说明.md`
