"""Fixed-R GH5 scan of arg((A+C)/B) vs SOC. Q6 protocol, product GH5, theta=0/pi.

Imports A1 helpers. Patches A1 OUT/CONTROLS so jobs land here.
Does not call A1.write_reports. Skips soc=0 (A1 s0) and soc=1 (frozen GH5).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import q8_m0_abc_a1_gate_gh5_20261029 as a1
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
from q8_m0_model import R0, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/abc_soc_gh5_scan_20261029"
FIG = OUT / "figures"
DOCS = ROOT / "docs/04_BSV量子光"
DRIVER_WALL_S = 7200.0
ALPHA2 = float(abs(complex(odd.ALPHA)) ** 2)
MABS = float(0.5 * np.sinh(2.0 * odd.R_SQ))

CONTROLS = {
    "soc025": {"delta": float(R0.delta), "lam_so": 0.25 * float(R0.lam_so), "label": "SOC=0.25 R0"},
    "soc050": {"delta": float(R0.delta), "lam_so": 0.50 * float(R0.lam_so), "label": "SOC=0.50 R0"},
    "soc075": {"delta": float(R0.delta), "lam_so": 0.75 * float(R0.lam_so), "label": "SOC=0.75 R0"},
}
JOBS = (
    ("soc025_p_0", "soc025", 1, a1.THETA_0),
    ("soc025_m_0", "soc025", -1, a1.THETA_0),
    ("soc025_p_pi", "soc025", 1, a1.THETA_PI),
    ("soc025_m_pi", "soc025", -1, a1.THETA_PI),
    ("soc050_p_0", "soc050", 1, a1.THETA_0),
    ("soc050_m_0", "soc050", -1, a1.THETA_0),
    ("soc050_p_pi", "soc050", 1, a1.THETA_PI),
    ("soc050_m_pi", "soc050", -1, a1.THETA_PI),
    ("soc075_p_0", "soc075", 1, a1.THETA_0),
    ("soc075_m_0", "soc075", -1, a1.THETA_0),
    ("soc075_p_pi", "soc075", 1, a1.THETA_PI),
    ("soc075_m_pi", "soc075", -1, a1.THETA_PI),
)
JOB_MAP = {tag: (ctrl, neel, theta) for tag, ctrl, neel, theta in JOBS}
BATCHES = (
    ("soc025", ("soc025_p_0", "soc025_m_0", "soc025_p_pi", "soc025_m_pi")),
    ("soc050", ("soc050_p_0", "soc050_m_0", "soc050_p_pi", "soc050_m_pi")),
    ("soc075", ("soc075_p_0", "soc075_m_0", "soc075_p_pi", "soc075_m_pi")),
)


def patch_a1() -> None:
    a1.OUT = OUT
    a1.FIG = FIG
    a1.CONTROLS = CONTROLS
    a1.JOBS = JOBS
    a1.JOB_MAP = JOB_MAP
    a1.BATCHES = BATCHES
    a1.TASK_TAG = "20261029_soc"


patch_a1()


def launch_batch(tags, t0: float) -> dict:
    procs, logs = {}, {}
    for tag in tags:
        if a1.job_finished_ok(tag):
            a1.progress(f"drive skip {tag} already ok")
            continue
        logf = (OUT / f"log_{tag}.txt").open("a", encoding="utf-8")
        logs[tag] = logf
        envp = os.environ.copy()
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            envp[k] = "1"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", tag]
        a1.progress(f"launch {tag}")
        procs[tag] = subprocess.Popen(
            cmd, cwd=str(HERE), env=envp, stdout=logf, stderr=subprocess.STDOUT)
    while True:
        elapsed = time.perf_counter() - t0
        alive = {tag: p for tag, p in procs.items() if p.poll() is None}
        if not alive:
            break
        if elapsed >= DRIVER_WALL_S:
            a1.progress(f"driver wall {DRIVER_WALL_S}s: terminating {list(alive)}")
            for tag, p in alive.items():
                p.terminate()
            time.sleep(2.0)
            for tag, p in alive.items():
                if p.poll() is None:
                    p.kill()
            break
        time.sleep(2.0)
    for fh in logs.values():
        fh.close()
    return {tag: (procs[tag].returncode if tag in procs else 0) for tag in tags}


def wrap180(arg_deg: float) -> float:
    return float(((float(arg_deg) + 180.0) % 360.0) - 180.0)


def pi_dev(arg_deg: float) -> float:
    a = abs(wrap180(arg_deg))
    return float(min(a, abs(a - 180.0)))


def row_from_summary(ctrl: str, row: dict) -> dict:
    px = row["odd_x_proxy"]
    babs = float(px["B_proxy"]["abs"])
    acabs = float(px["A_plus_C_proxy"]["abs"])
    arg_acb = float(np.degrees(np.angle(
        complex(px["A_plus_C_proxy"]["re"], px["A_plus_C_proxy"]["im"])
        / complex(px["B_proxy"]["re"], px["B_proxy"]["im"])
    ))) if babs > 0 else float("nan")
    return {
        "ctrl": ctrl,
        "label": row["label"],
        "lam_so": row["lam_so"],
        "lam_so_frac": float(row["lam_so"] / R0.lam_so),
        "delta": row["delta"],
        "min_gap": row["min_gap"],
        "odd0": row["theta0"]["odd_norm2D"],
        "odd_pi": row["theta_pi"]["odd_norm2D"],
        "amp_ratio_pi_over_0": px["amp_ratio_pi_over_0"],
        "abs_AC_over_B": px["abs_AC_over_B"],
        "arg_AC_over_B_deg": arg_acb,
        "pi_dev_deg": pi_dev(arg_acb) if babs > 0 else None,
        "K_B": babs / ALPHA2,
        "K_M": acabs / MABS,
        "K_M_over_K_B": (acabs / MABS) / max(babs / ALPHA2, 1e-30),
        "B_proxy": px["B_proxy"],
        "A_plus_C_proxy": px["A_plus_C_proxy"],
        "cannot_separate_A_and_C": True,
    }


def analyze() -> dict:
    audits_raw = json.loads((OUT / "ctrl_audit.json").read_text(encoding="utf-8"))
    audits = audits_raw["controls"]
    summaries = {}
    missing = []
    for ctrl, tags in BATCHES:
        if not all(a1.job_finished_ok(tag) for tag in tags):
            missing.extend([tag for tag in tags if not a1.job_finished_ok(tag)])
            continue
        summaries[ctrl] = a1.summarize_ctrl(ctrl, audits[ctrl])
    frozen = a1.load_frozen_gh5()
    rows = {ctrl: row_from_summary(ctrl, summaries[ctrl]) for ctrl in summaries}
    fr_px = frozen.get("odd_x_proxy") if isinstance(frozen, dict) else None
    frozen_row = None
    if fr_px and "B_proxy" in fr_px:
        fake = {
            "label": "SOC=1.00 R0 frozen GH5",
            "lam_so": float(R0.lam_so),
            "delta": float(R0.delta),
            "min_gap": None,
            "theta0": frozen["theta0"],
            "theta_pi": frozen["theta_pi"],
            "odd_x_proxy": fr_px,
        }
        frozen_row = row_from_summary("soc100_frozen", fake)
        frozen_row["lam_so_frac"] = 1.0
        frozen_row["reused_frozen_gh5"] = True
        rows["soc100_frozen"] = frozen_row

    live = [r for r in rows.values() if r.get("pi_dev_deg") is not None]
    span = None
    if len(live) >= 2:
        args = [r["arg_AC_over_B_deg"] for r in live]
        ds = [a1.angdiff_deg(a, args[-1]) for a in args]
        span = float(max(ds) - min(ds))
    max_pi_dev = max((r["pi_dev_deg"] for r in live), default=None)
    km_span = None
    if len(live) >= 2:
        rats = [r["K_M_over_K_B"] for r in live]
        km_span = float(max(rats) - min(rats))

    if span is not None and abs(span) >= 10.0:
        verdict = "arg_ACB_moves_ge_10deg"
    elif max_pi_dev is not None and max_pi_dev >= 10.0:
        verdict = "arg_ACB_off_pi_ge_10deg"
    else:
        verdict = "arg_ACB_pinned_near_pi"

    rec = {
        "task_tag": "20261029_soc",
        "protocol": "Qin_a2_1_4_1",
        "status": "ok" if summaries and not missing else "partial",
        "verdict": verdict,
        "input_state": "D(0.5)S(0.3 exp(i theta))|0>",
        "R": float(MABS / ALPHA2),
        "alpha2": ALPHA2,
        "M_abs": MABS,
        "used_product_GH": True,
        "cannot_separate_A_and_C": True,
        "did_not_call_a1_write_reports": True,
        "did_not_overwrite_a1": True,
        "qed_not_rerun": True,
        "rows": rows,
        "summaries": summaries,
        "frozen_gh5": frozen,
        "missing_jobs": missing,
        "arg_span_deg": span,
        "max_pi_dev_deg": max_pi_dev,
        "KM_over_KB_span": km_span,
        "source_file": Path(__file__).name,
        "source_sha256": sha256_file(Path(__file__)),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    a1.dump(OUT / "soc_gh5_scan.json", rec)
    return rec


def drive() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    audits = a1.collect_audits()
    t0 = time.perf_counter()
    codes = {}
    for ctrl, tags in BATCHES:
        remaining = DRIVER_WALL_S - (time.perf_counter() - t0)
        au = audits["controls"][ctrl]
        if not au["ok"]:
            a1.progress(f"skip {ctrl} audit issues={au['issues']}")
            continue
        if remaining < 60.0:
            a1.progress(f"skip {ctrl}: remaining {remaining:.1f}s < 60")
            break
        a1.progress(f"batch {ctrl} remaining={remaining:.1f}s")
        codes.update(launch_batch(tags, t0))
        if time.perf_counter() - t0 >= DRIVER_WALL_S:
            break
    a1.progress(f"jobs return {codes} driver_wall={time.perf_counter()-t0:.1f}s")
    done = [tag for tag, _, _, _ in JOBS if a1.job_finished_ok(tag)]
    if done:
        rec = analyze()
        a1.progress(f"analyze done verdict={rec.get('verdict')}")
        print(json.dumps({
            "status": rec["status"], "verdict": rec["verdict"],
            "arg_span_deg": rec.get("arg_span_deg"),
            "rows": {k: {
                "arg_AC_over_B_deg": v["arg_AC_over_B_deg"],
                "pi_dev_deg": v["pi_dev_deg"],
                "K_M_over_K_B": v["K_M_over_K_B"],
                "abs_AC_over_B": v["abs_AC_over_B"],
            } for k, v in rec["rows"].items()},
        }, ensure_ascii=False, indent=2))
        return 0 if rec.get("status") in ("ok", "partial") else 1
    rec = {"status": "timeout_partial", "returncodes": codes, "stopped": True}
    a1.dump(OUT / "soc_gh5_scan.json", rec)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    patch_a1()
    if args.audit_only:
        rec = a1.collect_audits()
        print(json.dumps({"all_ok": rec.get("all_ok")}, ensure_ascii=False, indent=2))
        return 0 if rec.get("all_ok") else 3
    if args.job:
        rec = a1.run_one_job(args.job)
        return 0 if rec.get("status") in ("ok", "timeout_partial", "ctrl_abort") else 1
    if args.analyze_only:
        rec = analyze()
        print(json.dumps({"status": rec.get("status"), "verdict": rec.get("verdict")},
                         ensure_ascii=False, indent=2))
        return 0 if rec.get("status") in ("ok", "partial") else 1
    return drive()


if __name__ == "__main__":
    raise SystemExit(main())
