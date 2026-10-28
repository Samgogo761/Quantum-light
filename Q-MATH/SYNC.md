# 最简单的 Git 同步方案

## 2026-10-28 检查结果

| 项目 | 结果 |
|---|---|
| Git 根目录 | 现有 `Quantum-light`，不是新仓库 |
| 整理前 HEAD | `b22dc12` |
| 当前分支 | `claude/fix-question-submission-UaGkJ`，沿用，不擅自改名或切换 |
| upstream | `origin/claude/fix-question-submission-UaGkJ`；本地缓存的整理前 ahead/behind 为 0/0，**本次未 fetch，不能据此保证远端实时相同** |
| origin fetch/push | `https://github.com/Samgogo761/Quantum-light.git` |
| GitHub 可见性 | **public**，不符合私有资料同步要求 |
| GitHub 连接器身份 | `Samgogo761`，GitHub 返回 admin/maintain/pull/push 权限 |
| 本机 Git 传输身份 | 尚未验证；连接器有权限不等于本机 Git 凭据已配置成功 |
| GitHub 默认分支 | `claude/wannier90-fortran-integration-dydlH`，与当前分支不同；不能默认向 main 或默认分支推送 |

本次不创建仓库、不 clone、不改 origin、不改变可见性、不 fetch/pull/push。现有工作区有大量无关未提交改动；本次只提交清单内 Q-MATH 文件，不把整个工作区打包。

## 推荐：一个仓库、一个工作区、一个 Q-MATH 索引

优先在用户确认后将**现有** `Samgogo761/Quantum-light` 改为 private，继续在本工作区管理代码与研究资料。这样保留 Git 历史和既有源码，不产生第二个 Q-MATH 仓库。

可在 GitHub 仓库 Settings → General → Danger Zone → Change repository visibility 中改为 Private。**本次没有执行此操作。** 变为 private 不会撤回此前已被下载的公开内容。

如果该仓库必须公开，则先停在本地版本，另由用户确定私有目标和单一源码归属。不要自动新建第二套相互独立维护的源码；不要向当前 public origin 推送本档案。

## 真正同步前的关口

1. 用户确认精确目标地址 `https://github.com/Samgogo761/Quantum-light.git` 及 private 要求，批准将本轮资料同步到该地址。
2. 重新读取 GitHub 可见性和写权限，要求 private；同时检查 `git remote -v` 的 fetch/push 地址均符合确认值。不要在 URL 中嵌入 token。
3. 验证**本机 Git 实际认证身份**与写权限；不要打印凭据。连接器身份与 Git 传输身份可能不同。
4. 只读检查整个待推分支历史、文件大小和敏感资料。现有忽略规则不会移除历史中已提交的数据；本次新文件清单审查不代表老历史已完成全面秘密扫描。
5. 获准后再 fetch，检查目标分支是否存在、是否分叉；首次推送前展示提交范围。保持当前工作区无关改动，不能盲目 pull、reset 或覆盖远端。
6. 用户确认后向明确分支普通推送；不 force push，不未经确认改变默认分支。当前目标若继续用原分支，推送形式如下：

```powershell
git push -u origin HEAD:refs/heads/claude/fix-question-submission-UaGkJ
```

该命令仅作方案说明，**本次未执行**。将来分支发生变化时必须按当时的实际分支重新确定，不能照抄旧值。

## 本地自动收尾

根目录 `AGENTS.md` 要求每轮完成 Q-MATH 时自动调用 `manage.py finish`，更新索引并提交。它只暂存明确的 Q-MATH 路径，发现已有暂存变更即停止，禁止夹带其他工作。

导入外部新版本：`python tools/qmath/manage.py import <原始包路径>`。新增仓库内文件：`python tools/qmath/manage.py register <明确文件路径...>`。更新研究状态后：

```powershell
python tools/qmath/manage.py finish --message "qmath: 本轮具体成果"
```

没有后台定时器、自动科研计算或无人值守远程同步。远程推送仍需要以上确认。
