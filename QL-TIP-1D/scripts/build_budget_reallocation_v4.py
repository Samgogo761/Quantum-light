"""Build the user-authorized v3 budget reallocation and reopen explicit stage permission.

This script does not run TDSE or post-processing.  It reads the real ledger, the
per-task cost table and the existing evidence pointer, then writes a new audit
revision.  The overall global ceiling (240 new TDSE / 72 science hours) is NOT
changed; only the internal continuation sub-allocation is resized to fit inside
that unchanged global ceiling.
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT / "results/bias_v3_audit/reallocation_20261106"
ACTIVE = PROJECT / "results/bias_v3_audit/r1_20261106_v3/active_evidence.json"
LEDGER = PROJECT / "results/bias_v2/compute_ledger_v2.csv"
COST = PROJECT / "results/bias_v3_audit/r1_20261106_v3/cost_remaining.csv"

USER_AUTHORIZATION = (
    "2026-11-06 user message: '不够额度吗？我换了一个模型，再继续完整v3？"
    "我希望能把所有的任务先完成了再提交报告'"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    ledger = read_csv(LEDGER)
    cost = read_csv(COST)
    passed = {row["run_id"] for row in ledger
              if row["kind"] == "tdse" and row["status"] == "PASS"}
    remaining = [row for row in cost if row["task_id"] not in passed]
    used_hours = sum(float(row["counted_seconds"] or 0.0) for row in ledger) / 3600.0
    new_tdse = sum(1 for row in ledger
                   if row["kind"] == "tdse" and row["status"] != "preflight_failed")
    remaining_hours = sum(float(row["propagation_estimate_s"]) + float(row["readout_estimate_s"])
                          for row in remaining) / 3600.0
    global_baseline = 14.117235553083
    snapshot = {
        "schema": "ql-tip-1d-budget-reallocation-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                            text=True).strip(),
        "ledger_sha256": sha256(LEDGER),
        "cost_table_sha256": sha256(COST),
        "ledger_events": len(ledger),
        "ledger_new_tdse": new_tdse,
        "continuation_hours_used": used_hours,
        "global_hours_used": global_baseline + used_hours,
        "remaining_tasks": len(remaining),
        "remaining_estimate_hours": remaining_hours,
        "projected_continuation_hours": used_hours + remaining_hours,
        "projected_global_hours": global_baseline + used_hours + remaining_hours,
        "projected_continuation_tdse": new_tdse + len(remaining),
        "projected_global_tdse": 48 + new_tdse + len(remaining),
    }
    assert snapshot["projected_global_hours"] < 72.0, "global 72 h ceiling would be crossed"
    assert snapshot["projected_continuation_tdse"] <= 112, "continuation TDSE cap crossed"
    assert snapshot["projected_global_tdse"] <= 240, "global TDSE cap crossed"

    limits = {
        "schema": "ql-tip-1d-budget-limits-v4",
        "authorization": USER_AUTHORIZATION,
        "decision": "CONTINUE_FULL_V3_ALL_TASKS_BEFORE_REPORT",
        "global_tdse_limit": 240,
        "global_hours_limit": 72.0,
        "global_hours_baseline": global_baseline,
        "continuation_tdse_limit": 112,
        "continuation_hours_limit": 56.0,
        "previous_continuation_hours_limit": 40.0,
        "repair_tdse_limit": 12,
        "repair_hours_limit": 8.0,
        "stage_s3_hours_limit": 10.0,
        "previous_stage_s3_hours_limit": 8.0,
        "batch_max_new_runs": 8,
        "event_hard_timeout_s": 5400,
        "unchanged": [
            "single active electron square well and all frozen physics parameters",
            "common readout eta=.05 eV / dE=.0125 eV / mask 2-4 nm / three fixed thresholds",
            "all R1/R2/final statistical error gates and the <=5% target",
            "explicit allowed_task_ids per batch, no bare offsets",
            "per-batch <=8 new TDSE and per-event <=90 min",
            "material sensitivity stays DEFERRED_PRIORITY, no new physical model",
            "no dt=.025 unless the task-book trigger is actually met",
        ],
        "reason": (
            "The stop at S3 partial was caused by the internal 40 h continuation "
            "sub-cap and the 8 h S3 stage cap, not by the global 72 h ceiling and "
            "not by any AI-model quota. After the 14 completed S3 propagations the "
            "remaining 66-task mandatory path is estimated at {:.3f} h; the global "
            "budget still has {:.3f} h left. The continuation sub-cap is resized "
            "inside the unchanged global ceiling."
        ).format(remaining_hours, 72.0 - snapshot["global_hours_used"]),
    }
    (AUDIT / "budget_authorization.json").write_text(
        json.dumps(limits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (AUDIT / "START_STATE_v4.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    old = json.loads(ACTIVE.read_text(encoding="utf-8"))
    (AUDIT / "active_evidence_v3_r1_before_reallocation.json").write_text(
        json.dumps(old, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    new = dict(old)
    new.update({
        "revision": "v3_r2_user_authorized_reallocation_20261106",
        "allowed_stage": "SCOUT_BATCHES",
        "allowed_task_ids": [
            f"bias_v2_lib_F{F}_I20_phi{deg:03d}_r01"
            for F in ["0", "1"] for deg in [0, 90, 180, 270]
        ],
        "budget": limits,
        "s3_status": "CONTINUING_20MU_BATCH_C",
        "previous_s3_status": old.get("s3_status"),
        "note": ("User explicitly authorized completing all v3 tasks before the final "
                 "report. Global 72 h / 240 TDSE ceiling unchanged; continuation "
                 "sub-cap resized to 56 h. Batch C = 20 mu, F0/F1, four CEP."),
    })
    ACTIVE.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = f"""# v3-r2 预算重分配与继续执行许可

日期：2026-11-06。依据：{USER_AUTHORIZATION}

## 1. 先把"额度"说清楚

上一轮停在 `S3 = PARTIAL / BUDGET_INCOMPLETE`，原因**不是** AI 模型额度耗尽，也不是物理失败，
而是 v3 任务书里的**机时治理上限**：续接 40 h 子上限与 S3 阶段 8 h 上限。
更换对话模型不会增加 TDSE 机时；能解锁的是用户明确授权调整这条内部子上限。

## 2. 实测复核

| 量 | 数值 |
|---|---:|
| 账本事件 | {snapshot['ledger_events']} |
| 新增 TDSE（续接） | {snapshot['ledger_new_tdse']} |
| 续接已用机时 | {used_hours:.4f} h |
| 全局已用机时（含历史基线 {global_baseline:.4f} h） | {snapshot['global_hours_used']:.4f} h |
| 全局 72 h 剩余 | {72.0 - snapshot['global_hours_used']:.4f} h |
| 全局 240 条已用 | {48 + new_tdse} |
| 剩余必需任务 | {snapshot['remaining_tasks']} 条 |
| 剩余必需任务估时（分项成本表） | {remaining_hours:.4f} h |
| 完成后预计续接机时 | {snapshot['projected_continuation_hours']:.4f} h |
| 完成后预计全局机时 | {snapshot['projected_global_hours']:.4f} h |
| 完成后预计续接 / 全局 TDSE | {snapshot['projected_continuation_tdse']} / {snapshot['projected_global_tdse']} |

结论：**完整必需路径可以装入原本没有改变的全局 72 h / 240 条上限**，
只是装不进 40 h 的续接子上限和 8 h 的 S3 阶段上限。

## 3. 本次改动（只改治理上限，不改科学门槛）

- 续接机时子上限 40 h → **56 h**；S3 阶段上限 8 h → **10 h**。
- 全局 240 条 / 72 h、续接 112 条、R1 修复 12 条 / 8 h **全部不变**。
- 每批 ≤8 条、单事件 ≤90 min、显式 `allowed_task_ids`、共同读出协议、逐类别误差、
  尾部下限、总 5% 门槛、D 的审计区间判定 **全部不变**。
- 物理模型不变：单活性电子方阱、1600 nm/25 fs、F={{0,1}} V/nm、明亮强压缩极限。
  材料敏感性仍为 `DEFERRED_PRIORITY`；不新增 DFT、多电子、温度、三维。
- `dt=.025` 仍不是默认前置任务，只在任务书第 10 节触发条件成立时考虑一次。

## 4. 状态

`active_evidence.json` 由 `NONE` 改为 `SCOUT_BATCHES`，并锁定批次 C 的 8 个显式任务 ID
（20μ，F0/F1，CEP 0/90/180/270）。旧证据指针已原样保存到本目录，未覆盖历史记录。

本报告不是最终交付报告。按用户要求，S3-C、S4、S5、S6 及统计接口修正全部完成后，
再统一生成 `decision.md` 与最终统计结果。
"""
    (AUDIT / "BUDGET_REALLOCATION.md").write_text(report, encoding="utf-8")
    print(json.dumps({"snapshot": snapshot, "allowed_stage": new["allowed_stage"],
                      "allowed_task_ids": new["allowed_task_ids"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
