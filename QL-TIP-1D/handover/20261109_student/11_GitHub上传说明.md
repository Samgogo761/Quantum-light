# GitHub 上传说明

## 1. 目标仓库现状（2026-11-09 实测）

- 仓库：`Samgogo761/Metal-Quantum`（截图中的 `Samogo761` 少了一个 `g`）。
- 可见性：**public**。
- 状态：空仓库，默认分支 `main`。
- 本地已有另一个远程：`origin = https://github.com/Samgogo761/Quantum-light.git`，当前分支
  `claude/fix-question-submission-UaGkJ`。

## 2. 必须先处理的风险

`Metal-Quantum` 目前是**公开仓库**。本包包含未发表的研究结果、内部审阅记录与完整代码，
直接推送等于公开发布，可能影响论文新颖性与数据边界。**在改为 private 之前不要 push。**

修改可见性：GitHub 仓库页 → **Settings** → **General** → 页面底部 **Danger Zone** →
**Change repository visibility** → **Change to private** → 按提示输入仓库名确认。

若确实要公开，需要负责人明确决定，并先删除审阅记录、内部路径与任何未发表数值。

## 3. 推荐上传方式（保留提交历史，不建嵌套仓库）

只上传交接子树，不推整个 `Quantum-light` 仓库（后者含大量其他项目文件）。

```bash
# 在 Quantum-light 根目录
git subtree split --prefix=QL-TIP-1D/handover/20261109_student -b handover-20261109
git push https://github.com/Samgogo761/Metal-Quantum.git handover-20261109:main
```

说明：

- `git subtree split` 生成一个只含该目录历史的独立分支，不需要 `git init`，不产生嵌套仓库。
- 第一次 push 后，仓库根目录就是 `00_README.md`、`01_...` 等文件。
- 后续更新：重新 `subtree split` 到新分支（或 `-f` 覆盖同名分支）再 push。
- 需要 GitHub 凭据（HTTPS token 或 SSH key）。本轮**未执行 push**，也未修改远程配置。

## 4. 备选方式（不要历史，最简）

```bash
mkdir Metal-Quantum && cd Metal-Quantum
git init
# 把 QL-TIP-1D/handover/20261109_student/ 的内容复制到此目录
git add .
git commit -m "QL-TIP-1D handover 2026-11-09"
git branch -M main
git remote add origin https://github.com/Samgogo761/Metal-Quantum.git
git push -u origin main
```

缺点：丢失原始仓库的提交历史；不推荐，除非只需要一个静态快照。

## 5. 不要上传的内容

| 内容 | 体积 | 原因 |
|---|---:|---|
| `data/raw/bias_v2/`（96 个 NPZ） | 543,033,360 字节 | 超过 GitHub 建议仓库体积；单文件虽 <100 MB，但总量过大 |
| 整个 `Quantum-light` 仓库 | >1 GB | 含大量其他项目与历史 |
| `results/bias_v2/readouts/` 全量谱 | 约 220 MB | 只放 8 个代表性谱；其余按需从原仓库取 |
| 内部审阅记录 | 小 | 公开仓库会暴露审阅过程；private 仓库可保留 |

NPZ 的校验清单在 `07_关键结果与误差审计/raw_state_manifest.csv`；需要原始波函数时用移动硬盘
或机构网盘传递，不要用 Git LFS（免费配额通常不足以承载 543 MB 的多次修订）。

## 6. 上传后校验

```bash
git ls-remote https://github.com/Samgogo761/Metal-Quantum.git
```

在 GitHub 页面确认：

- 根目录有 `00_README.md`；
- `05_Origin绘图数据/` 下 6 个 CSV 与 `representative_spectra/` 存在；
- `07_关键结果与误差审计/raw_state_manifest.csv` 有 96 行；
- `SHA256SUMS.txt` 可下载，且本地 `sha256sum -c` 通过（Windows 可用 PowerShell 逐行核对）。

## 7. 当前状态

- 交接包已在本地生成并通过 SHA256 自检。
- **未执行任何 push**，未修改 `origin`，未改仓库可见性。
- 推送前需要：① 仓库改为 private（或负责人明确同意公开）；② 确认使用哪个 GitHub 账号与凭据；
  ③ 确认是否需要保留提交历史（决定用第 3 节还是第 4 节）。
