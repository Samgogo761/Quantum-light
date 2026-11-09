#!/usr/bin/env python3
"""Protocol A 1600 nm exact-QED four-angle + pi/4 holdout (gated).

This runner does NOT start production unless --authorize-stage1/2/3 is given.
Stage 1 four-angle jobs are already finished; freeze-fit is postprocess only.
It does NOT reuse the fockaudit n8_p_q axis as four-angle or as a two-domain
holdout. Keep Q6 M0, same k/-k, D(0.5)S(0.3 e^{i theta})|0>, R=1.273,
abstract g=0.005 A^{-1}, 1+4+1, omega0=0.775 explicit, n_max=8.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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

import q8_m0_qinput_1600nm_protocolA_qed_fockaudit_20261102 as fa
from q8_m0_qinput_shared_kk_ds_odd_20261102 import jsonable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102"
PROTOCOL = "Qin_a2_1_4_1_1600nm_protocolA_qed_fourangle"
AUTH_ENV = "QED_FOURANGLE_AUTHORIZE"
N_MAX = 8
N_FULL = fa.N_FULL
OMEGA0_1600 = fa.OMEGA0_1600
FOCK_ALARM = fa.FOCK_ALARM
STAGE2_BUDGET_S = 90 * 60
JOB_WALL_DEFAULT = 3300.0
JOB_WALL_MARGIN_S = 120.0
JOB_WALL_ENV = "QED_FOURANGLE_JOB_WALL_S"
FIT_PATH = OUT / "fourangle_fit.json"
FIT_SHA_PATH = OUT / "fourangle_fit.sha256"
ANALYSIS = OUT / "analysis_endpoint_v1"
STAGE2_BUDGET_PATH = OUT / "stage2_budget.json"
FLOAT_ATOL = 1e-12

THETA_0 = fa.THETA_0
THETA_Q = fa.THETA_Q
THETA_H = fa.THETA_H
THETA_PI = fa.THETA_PI
THETA_T = fa.THETA_T

FOURANGLE_TAGS = ("p_0", "m_0", "p_h", "m_h", "p_pi", "m_pi", "p_t", "m_t")
PI4_TAGS = ("p_q", "m_q")
ALL_TAGS = FOURANGLE_TAGS + PI4_TAGS

JOB_TABLE = {
    "p_0": {"neel": 1, "theta": THETA_0, "n_max": N_MAX, "n_steps": N_FULL, "family": "fourangle"},
    "m_0": {"neel": -1, "theta": THETA_0, "n_max": N_MAX, "n_steps": N_FULL, "family": "fourangle"},
    "p_h": {"neel": 1, "theta": THETA_H, "n_max": N_MAX, "n_steps": N_FULL, "family": "fourangle"},
    "m_h": {"neel": -1, "theta": THETA_H, "n_max": N_MAX, "n_steps": N_FULL, "family": "fourangle"},
    "p_pi": {"neel": 1, "theta": THETA_PI, "n_max": N_MAX, "n_steps": N_FULL, "family": "fourangle"},
    "m_pi": {"neel": -1, "theta": THETA_PI, "n_max": N_MAX, "n_steps": N_FULL, "family": "fourangle"},
    "p_t": {"neel": 1, "theta": THETA_T, "n_max": N_MAX, "n_steps": N_FULL, "family": "fourangle"},
    "m_t": {"neel": -1, "theta": THETA_T, "n_max": N_MAX, "n_steps": N_FULL, "family": "fourangle"},
    "p_q": {"neel": 1, "theta": THETA_Q, "n_max": N_MAX, "n_steps": N_FULL, "family": "pi4"},
    "m_q": {"neel": -1, "theta": THETA_Q, "n_max": N_MAX, "n_steps": N_FULL, "family": "pi4"},
}


def bind_fockaudit_namespace() -> None:
    fa.OUT = OUT
    fa.PROTOCOL = PROTOCOL
    fa.N_FULL = N_FULL

    def job_spec(tag: str) -> dict:
        if tag not in JOB_TABLE:
            raise ValueError(f"unknown fourangle tag {tag}; fockaudit tags are not valid here")
        spec = dict(JOB_TABLE[tag])
        spec["tag"] = tag
        return spec

    fa.job_spec = job_spec


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def progress(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with (OUT / "progress.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def env_authorized(stage: str) -> bool:
    got = os.environ.get(AUTH_ENV, "")
    return got in {stage, "all"}


def refuse(why: str, extra=None) -> dict:
    rec = {
        "status": "refused_not_authorized",
        "why": why,
        "protocol": PROTOCOL,
        "did_not_start_production": True,
        "did_not_extract_ABC": True,
        "did_not_copy_n8_p_q": True,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        rec.update(extra)
    dump(OUT / "refused_not_authorized.json", rec)
    progress(f"REFUSED {why}")
    return rec


def load_job(tag: str):
    bind_fockaudit_namespace()
    return fa.load_job(tag)


def official_n_full(rec) -> int | None:
    if not rec:
        return None
    defn = rec.get("definition") or {}
    if defn.get("n_full") is None:
        return None
    n_full = int(defn["n_full"])
    top = rec.get("n_steps")
    if top is not None and int(top) != n_full:
        return None
    return n_full


def apply_job_wall_from_env() -> None:
    raw = os.environ.get(JOB_WALL_ENV)
    if raw:
        fa.JOB_WALL_S = float(raw)


def job_ok_no_alarm(tag: str) -> bool:
    rec = load_job(tag)
    n_full = official_n_full(rec)
    ledger = (rec or {}).get("ledger") or {}
    series = OUT / f"series_{tag}.npz"
    return bool(
        rec
        and rec.get("status") == "ok"
        and rec.get("fock_alarm_0p05") is False
        and rec.get("finite_state") is True
        and rec.get("covers_requested") is True
        and rec.get("error") is None
        and rec.get("omega0") == OMEGA0_1600
        and abs(float(rec.get("omega0", 0.0)) - OMEGA0_1600) <= FLOAT_ATOL
        and rec.get("n_max") == N_MAX
        and rec.get("used_pf_OMEGA0") is False
        and n_full == N_FULL
        and int(ledger.get("steps_completed") or -1) == N_FULL
        and series.is_file()
    )


def job_row(tag: str):
    rec = load_job(tag)
    if rec is None:
        return {"status": None, "present": False}
    n_full = official_n_full(rec)
    ledger = rec.get("ledger") or {}
    return {
        "present": True,
        "status": rec.get("status"),
        "neel": rec.get("neel"),
        "theta": rec.get("theta"),
        "omega0": rec.get("omega0"),
        "n_max": rec.get("n_max"),
        "n_steps": n_full if n_full is not None else rec.get("n_steps"),
        "n_full": (rec.get("definition") or {}).get("n_full"),
        "steps_completed": ledger.get("steps_completed"),
        "t_reached_over_T0": rec.get("t_reached_over_T0"),
        "max_top_fock": rec.get("max_top_fock"),
        "max_top_two": rec.get("max_top_two"),
        "max_n_ph": rec.get("max_n_ph"),
        "min_n_ph": rec.get("min_n_ph"),
        "end_n_ph": rec.get("end_n_ph"),
        "max_abs_norm_minus_1": rec.get("max_abs_norm_minus_1"),
        "finite_state": rec.get("finite_state"),
        "covers_requested": rec.get("covers_requested"),
        "n_samples": rec.get("n_samples"),
        "error": rec.get("error"),
        "fock_alarm_0p05": rec.get("fock_alarm_0p05"),
        "used_pf_OMEGA0": rec.get("used_pf_OMEGA0"),
        "did_not_scale_g": rec.get("did_not_scale_g"),
        "ok_no_alarm": job_ok_no_alarm(tag),
    }


def snapshot_status_if_needed() -> None:
    src = OUT / "fourangle_status.json"
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    dst = ANALYSIS / "fourangle_status_pre_analyze_snapshot.json"
    if src.is_file() and not dst.is_file():
        shutil.copy2(src, dst)


def validate_fit_frozen() -> dict:
    import q8_m0_qinput_1600nm_protocolA_qed_postprocess_20261102 as pp
    return pp.validate_frozen_fit(FIT_PATH)


def analyze_status() -> dict:
    bind_fockaudit_namespace()
    snapshot_status_if_needed()
    four = {tag: job_row(tag) for tag in FOURANGLE_TAGS}
    pi4 = {tag: job_row(tag) for tag in PI4_TAGS}
    n_four_ok = sum(1 for tag in FOURANGLE_TAGS if job_ok_no_alarm(tag))
    n_alarm = [
        tag for tag in ALL_TAGS
        if (load_job(tag) or {}).get("fock_alarm_0p05") is True
    ]
    fit_chk = validate_fit_frozen() if FIT_PATH.is_file() else {"ok": False, "why": "absent"}
    rec = {
        "status": "analyze",
        "protocol": PROTOCOL,
        "omega0": OMEGA0_1600,
        "used_pf_OMEGA0": False,
        "g": fa.G_ABS,
        "n_max": N_MAX,
        "n_steps": N_FULL,
        "fock_alarm": FOCK_ALARM,
        "did_not_extract_ABC": not bool(fit_chk.get("ok")),
        "did_not_claim_mechanism_under_exact_QED": True,
        "did_not_copy_n8_p_q": True,
        "gh5_is_not_qed": True,
        "stage1_tags": list(FOURANGLE_TAGS),
        "stage2_tags": list(PI4_TAGS),
        "fourangle_jobs": four,
        "pi4_jobs": pi4,
        "n_fourangle_ok_no_alarm": n_four_ok,
        "fourangle_complete": n_four_ok == 8,
        "alarm_tags": n_alarm,
        "fit_frozen": bool(fit_chk.get("ok")),
        "fit_check": {k: fit_chk.get(k) for k in ("ok", "why", "sha256") if k in fit_chk},
        "next": (
            "user must authorize stage 1; do not start production from this analyze"
            if n_four_ok == 0 else
            "resume unfinished fourangle axes after explicit stage-1 authorization"
            if n_four_ok < 8 else
            "fourangle 8/8; run --mode freeze-fit; do not start pi/4"
            if not fit_chk.get("ok") else
            "fit frozen; pi/4 holdout still requires explicit stage-2 authorization"
        ),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "fourangle_status.json", rec)
    progress(
        f"analyze four_ok={n_four_ok}/8 alarm={n_alarm} "
        f"fit_frozen={rec['fit_frozen']}"
    )
    return rec


def exit_code_for(rec) -> int:
    if not rec:
        return 1
    st = rec.get("status")
    if st == "ok" and rec.get("fock_alarm_0p05") is False:
        return 0
    if st == "ok_fock_alarm" or rec.get("fock_alarm_0p05") is True:
        return 2
    if st == "timeout_partial":
        return 4
    if st in ("refused_not_authorized", "freeze_fit_blocked"):
        return 3
    return 1


def worst_exit(tags) -> int:
    codes = [exit_code_for(load_job(tag)) for tag in tags]
    for pref in (2, 4, 3, 1):
        if pref in codes:
            return pref
    return 0 if codes and all(c == 0 for c in codes) else 1


def pi4_preflight() -> str | None:
    missing = [tag for tag in FOURANGLE_TAGS if not job_ok_no_alarm(tag)]
    if missing:
        return f"pi4 refused: fourangle axes not all ok_no_alarm {missing}"
    chk = validate_fit_frozen()
    if not chk.get("ok"):
        return f"pi4 refused: fourangle_fit.json is not a valid freeze ({chk.get('why')})"
    payload = chk.get("payload") or {}
    if payload.get("holdout_used_in_fit") is not False:
        return "pi4 refused: freeze used holdout in fit"
    return None


def load_stage2_budget() -> dict | None:
    if not STAGE2_BUDGET_PATH.is_file():
        return None
    return json.loads(STAGE2_BUDGET_PATH.read_text(encoding="utf-8"))


def init_stage2_budget() -> dict:
    existing = load_stage2_budget()
    if existing and existing.get("deadline"):
        return existing
    now = time.time()
    rec = {
        "started_at": now,
        "deadline": now + STAGE2_BUDGET_S,
        "budget_s": STAGE2_BUDGET_S,
        "job_wall_default_s": JOB_WALL_DEFAULT,
        "margin_s": JOB_WALL_MARGIN_S,
        "note": "wall clock includes pauses; restart does not reset the deadline",
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(STAGE2_BUDGET_PATH, rec)
    return rec


def remaining_job_wall(family: str) -> float:
    if family != "pi4":
        return JOB_WALL_DEFAULT
    bud = load_stage2_budget()
    if not bud:
        return JOB_WALL_DEFAULT
    remain = float(bud["deadline"]) - time.time()
    return max(0.0, min(JOB_WALL_DEFAULT, remain - JOB_WALL_MARGIN_S))


def run_one_job(tag: str, n_steps_override: int | None = None) -> dict:
    bind_fockaudit_namespace()
    if n_steps_override:
        rec = refuse("official entry refuses --n-steps", {"tag": tag})
        rec["status"] = "refused_not_authorized"
        return rec
    if tag not in JOB_TABLE:
        rec = refuse(f"unknown tag {tag}", {"tag": tag})
        rec["status"] = "unknown_tag"
        return rec
    if job_ok_no_alarm(tag):
        progress(f"{tag} already ok_no_alarm; wrapper returns read-only and will not call fockaudit")
        return load_job(tag)
    family = JOB_TABLE[tag]["family"]
    if family == "fourangle" and not env_authorized("stage1"):
        return refuse("direct --job fourangle requires QED_FOURANGLE_AUTHORIZE=stage1",
                      {"tag": tag})
    if family == "pi4":
        why = pi4_preflight()
        if why:
            return refuse(why, {"tag": tag})
        if not env_authorized("stage2"):
            return refuse("direct --job pi4 requires QED_FOURANGLE_AUTHORIZE=stage2",
                          {"tag": tag})
    apply_job_wall_from_env()
    return fa.run_one_job(tag, None)


def launch_serial(tags, authorize_stage: str, n_steps_override: int | None = None) -> dict:
    bind_fockaudit_namespace()
    OUT.mkdir(parents=True, exist_ok=True)
    if n_steps_override:
        rec = refuse("official launch_serial refuses --n-steps")
        rec["stopped"] = True
        return rec
    t0 = time.perf_counter()
    codes = {}
    stopped = None
    for tag in tags:
        if job_ok_no_alarm(tag):
            progress(f"skip {tag} already ok_no_alarm")
            codes[tag] = 0
            continue
        rec = load_job(tag)
        if rec and (rec.get("status") == "ok_fock_alarm" or rec.get("fock_alarm_0p05") is True):
            progress(f"STOP {tag}: fock_alarm; do not retune g or continue")
            codes[tag] = 2
            stopped = tag
            break
        family = JOB_TABLE[tag]["family"]
        wall = remaining_job_wall(family)
        if family == "pi4" and wall <= 0.0:
            progress(f"STOP {tag}: stage-2 wall budget exhausted")
            codes[tag] = 4
            stopped = tag
            break
        envp = os.environ.copy()
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            envp[k] = "1"
        envp[AUTH_ENV] = authorize_stage
        envp[JOB_WALL_ENV] = str(wall)
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", tag]
        progress(f"SERIAL launch {tag} wall={wall:.1f}s")
        logf = (OUT / f"log_{tag}.txt").open("a", encoding="utf-8")
        try:
            proc = subprocess.run(
                cmd, cwd=str(HERE), env=envp, stdout=logf, stderr=subprocess.STDOUT)
            codes[tag] = int(proc.returncode)
        finally:
            logf.close()
        child = load_job(tag)
        verified = exit_code_for(child)
        progress(f"{tag} child_rc={codes[tag]} verified_rc={verified} wall={time.perf_counter() - t0:.1f}s")
        if not job_ok_no_alarm(tag):
            progress(f"STOP after {tag}: not ok_no_alarm; do not launch the next axis")
            stopped = tag
            codes[tag] = verified if verified else 1
            break
    return {
        "returncodes": codes,
        "driver_wall_s": time.perf_counter() - t0,
        "serial": True,
        "stopped_after": stopped,
        "did_not_continue_after_failure": bool(stopped),
    }


def freeze_fit() -> dict:
    rec = analyze_status()
    if not rec.get("fourangle_complete"):
        out = refuse(
            "freeze-fit refused: fourangle 8 axes are not all ok_no_alarm",
            {"analyze": rec, "fourangle_complete": False},
        )
        out["rc"] = 3
        return out
    import q8_m0_qinput_1600nm_protocolA_qed_postprocess_20261102 as pp
    out = pp.freeze_fourangle(write=True)
    progress(
        f"freeze-fit {out.get('status')} sha={out.get('sha256')} "
        f"stage={out.get('stage_status')}"
    )
    return out


def drive(mode: str, authorize_stage1: bool, authorize_stage2: bool,
          authorize_stage3: bool, n_steps: int | None) -> int:
    bind_fockaudit_namespace()
    OUT.mkdir(parents=True, exist_ok=True)
    if n_steps:
        refuse("official entry refuses --n-steps")
        return 3
    if mode == "fourangle":
        if not authorize_stage1:
            refuse("fourangle production requires --authorize-stage1 from an explicit user start order")
            return 3
        os.environ[AUTH_ENV] = "stage1"
        launch = launch_serial(FOURANGLE_TAGS, "stage1", None)
        rec = analyze_status()
        rec["launch"] = launch
        dump(OUT / "fourangle_status.json", rec)
        return worst_exit(FOURANGLE_TAGS)
    if mode == "freeze-fit":
        rec = freeze_fit()
        st = rec.get("status")
        return 0 if st in ("fourangle_fit_frozen", "already_frozen") else 3
    if mode == "pi4":
        if not authorize_stage2:
            refuse("pi4 holdout requires --authorize-stage2")
            return 3
        why = pi4_preflight()
        if why:
            refuse(why)
            return 3
        init_stage2_budget()
        if remaining_job_wall("pi4") <= 0.0:
            refuse("pi4 refused: stage-2 90-minute budget already exhausted")
            return 4
        os.environ[AUTH_ENV] = "stage2"
        launch = launch_serial(PI4_TAGS, "stage2", None)
        rec = analyze_status()
        rec["launch"] = launch
        dump(OUT / "fourangle_status.json", rec)
        return worst_exit(PI4_TAGS)
    if mode == "leaveout":
        if not authorize_stage3:
            refuse("leaveout/ABC comparison requires --authorize-stage3")
            return 3
        why = pi4_preflight()
        if why:
            refuse(why)
            return 3
        missing = [tag for tag in PI4_TAGS if not job_ok_no_alarm(tag)]
        if missing:
            refuse(f"leaveout refused: pi/4 axes not all ok_no_alarm {missing}")
            return 3
        refuse("stage 3 leaveout is gated until batch 2 is authorized after both pi/4 domains pass")
        return 3
    return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("fourangle", "pi4", "freeze-fit", "leaveout"), default="")
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--authorize-stage1", action="store_true")
    ap.add_argument("--authorize-stage2", action="store_true")
    ap.add_argument("--authorize-stage3", action="store_true")
    ap.add_argument("--n-steps", type=int, default=0)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    bind_fockaudit_namespace()
    n_steps = int(args.n_steps) if args.n_steps else None
    if n_steps:
        refuse("official entry refuses --n-steps; do not shrink production n_full")
        return 3
    if args.job:
        rec = run_one_job(args.job, None)
        return exit_code_for(rec)
    if args.analyze_only:
        rec = analyze_status()
        print(json.dumps(jsonable({
            "status": rec.get("status"),
            "n_fourangle_ok_no_alarm": rec.get("n_fourangle_ok_no_alarm"),
            "fourangle_complete": rec.get("fourangle_complete"),
            "alarm_tags": rec.get("alarm_tags"),
            "fit_frozen": rec.get("fit_frozen"),
            "next": rec.get("next"),
            "did_not_extract_ABC": rec.get("did_not_extract_ABC"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("fourangle_complete") and not rec.get("alarm_tags") else 1
    if not args.mode:
        refuse("no mode: this runner does not default-evolve; pass --analyze-only or an authorized --mode")
        return 3
    return drive(args.mode, args.authorize_stage1, args.authorize_stage2,
                 args.authorize_stage3, n_steps)


if __name__ == "__main__":
    raise SystemExit(main())
