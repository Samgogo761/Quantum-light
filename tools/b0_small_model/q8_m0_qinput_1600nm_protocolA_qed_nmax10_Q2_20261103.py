#!/usr/bin/env python3
"""Q2: n_max=10 truncation screen on 1600 nm protocol-A exact QED.

Four production axes only: theta in {0, pi} x Neel = +/-1.
Reuses the fockaudit propagator. Input remains D(0.5)S(0.3 e^{i theta})|0>.
Does not patch photon_raw, does not refit A/B/C, does not write fourangle_fit.json.
Default refuses production. Stage 1 requires --authorize-stage1.
"""
from __future__ import annotations

import argparse
import hashlib
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

import q8_m0_qinput_1600nm_protocolA_qed_fockaudit_20261102 as fa
import q8_m0_qinput_1600nm_protocolA_qed_windowQ1_20261103 as wq1
from q8_m0_qinput_shared_kk_ds_odd_20261102 import jsonable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_1600nm_protocolA_qed_nmax10_Q2_20261103"
NMAX8_DIR = ROOT / "results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102"
FIT_PATH = NMAX8_DIR / "fourangle_fit.json"
FROZEN_SHA = "17a8e37ec3e0ab54096ff4edd1440c5a3f0412b09450c2fd8960cbc00d8e69db"
PROTOCOL = "Qin_a2_1_4_1_1600nm_protocolA_qed_nmax10_Q2"
AUTH_ENV = "QED_Q2NMAX10_AUTHORIZE"
JOB_WALL_ENV = "QED_Q2NMAX10_JOB_WALL_S"
N_MAX = 10
N_FULL = fa.N_FULL
OMEGA0_1600 = fa.OMEGA0_1600
FOCK_ALARM = fa.FOCK_ALARM
G_ABS = fa.G_ABS
JOB_WALL_DEFAULT = 5400.0
FLOAT_ATOL = 1e-12
NORM_ALARM = 1e-10
JOINT_DIM_N10 = 36 * 36 * (N_MAX + 1)
TAGS = ("p_0", "m_0", "p_pi", "m_pi")
PAIRS = (("0", "p_0", "m_0"), ("pi", "p_pi", "m_pi"))

# Frozen n_max=8 main-window H2x odd from windowQ1 (do not re-run nmax=8).
J8_H2X_REF = {
    "0": complex(-1.045687e-5, -5.770e-8),
    "pi": complex(+8.909623e-5, +5.022e-6),
}

JOB_TABLE = {
    "p_0": {"neel": 1, "theta": 0.0, "n_max": N_MAX, "n_steps": N_FULL, "family": "q2nmax10"},
    "m_0": {"neel": -1, "theta": 0.0, "n_max": N_MAX, "n_steps": N_FULL, "family": "q2nmax10"},
    "p_pi": {"neel": 1, "theta": float(np.pi), "n_max": N_MAX, "n_steps": N_FULL, "family": "q2nmax10"},
    "m_pi": {"neel": -1, "theta": float(np.pi), "n_max": N_MAX, "n_steps": N_FULL, "family": "q2nmax10"},
}


def bind_fockaudit_namespace() -> None:
    fa.OUT = OUT
    fa.PROTOCOL = PROTOCOL
    fa.N_FULL = N_FULL
    fa.JOB_WALL_S = JOB_WALL_DEFAULT
    fa.TASK_TAG = "20261103"
    fa.RECORD_DATE = "2026-11-03"

    def job_spec(tag: str) -> dict:
        if tag not in JOB_TABLE:
            raise ValueError(f"unknown Q2 nmax10 tag {tag}; fockaudit/fourangle tags are not valid here")
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def assert_frozen_fit() -> str:
    if not FIT_PATH.is_file():
        raise RuntimeError(f"missing frozen fit {FIT_PATH}")
    got = sha256_file(FIT_PATH)
    if got != FROZEN_SHA:
        raise RuntimeError(f"frozen fit SHA changed: {got}")
    return got


def refuse(why: str, extra=None) -> dict:
    rec = {
        "status": "refused_not_authorized",
        "why": why,
        "protocol": PROTOCOL,
        "did_not_start_production": True,
        "did_not_extract_ABC": True,
        "did_not_write_fourangle_fit": True,
        "did_not_patch_photon_raw": True,
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
    else:
        fa.JOB_WALL_S = JOB_WALL_DEFAULT


def job_ok_no_alarm(tag: str) -> bool:
    rec = load_job(tag)
    n_full = official_n_full(rec)
    ledger = (rec or {}).get("ledger") or {}
    series = OUT / f"series_{tag}.npz"
    norm = (rec or {}).get("max_abs_norm_minus_1")
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
        and (norm is not None and float(norm) < NORM_ALARM)
        and series.is_file()
    )


def job_row(tag: str):
    rec = load_job(tag)
    if rec is None:
        return {"status": None, "present": False}
    n_full = official_n_full(rec)
    ledger = rec.get("ledger") or {}
    dim = (rec.get("definition") or {}).get("joint_dim")
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
        "joint_dim": dim,
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


def analyze_status() -> dict:
    bind_fockaudit_namespace()
    jobs = {tag: job_row(tag) for tag in TAGS}
    n_ok = sum(1 for tag in TAGS if job_ok_no_alarm(tag))
    n_alarm = [tag for tag in TAGS if (load_job(tag) or {}).get("fock_alarm_0p05") is True]
    rec = {
        "status": "analyze",
        "protocol": PROTOCOL,
        "omega0": OMEGA0_1600,
        "used_pf_OMEGA0": False,
        "g": G_ABS,
        "n_max": N_MAX,
        "n_steps": N_FULL,
        "joint_dim_expected": JOINT_DIM_N10,
        "fock_alarm": FOCK_ALARM,
        "did_not_extract_ABC": True,
        "did_not_write_fourangle_fit": True,
        "did_not_patch_photon_raw": True,
        "did_not_copy_n10_p_q": True,
        "stage1_tags": list(TAGS),
        "jobs": jobs,
        "n_ok_no_alarm": n_ok,
        "stage1_complete": n_ok == 4,
        "alarm_tags": n_alarm,
        "next": (
            "user must authorize stage 1; do not start production from this analyze"
            if n_ok == 0 else
            "resume unfinished Q2 axes after explicit stage-1 authorization"
            if n_ok < 4 else
            "4/4 ok_no_alarm; run --mode compare; do not start dt scan or Q3"
        ),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "q2_status.json", rec)
    progress(f"analyze ok={n_ok}/4 alarm={n_alarm}")
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
    if st in ("refused_not_authorized",):
        return 3
    return 1


def worst_exit(tags) -> int:
    codes = [exit_code_for(load_job(tag)) for tag in tags]
    for pref in (2, 4, 3, 1):
        if pref in codes:
            return pref
    return 0 if codes and all(c == 0 for c in codes) else 1


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
    if not env_authorized("stage1"):
        return refuse("direct --job requires QED_Q2NMAX10_AUTHORIZE=stage1", {"tag": tag})
    apply_job_wall_from_env()
    return fa.run_one_job(tag, None)


def launch_serial(tags, authorize_stage: str) -> dict:
    bind_fockaudit_namespace()
    OUT.mkdir(parents=True, exist_ok=True)
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
        envp = os.environ.copy()
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            envp[k] = "1"
        envp[AUTH_ENV] = authorize_stage
        envp[JOB_WALL_ENV] = str(JOB_WALL_DEFAULT)
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", tag]
        progress(f"SERIAL launch {tag} wall={JOB_WALL_DEFAULT:.1f}s")
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


def packed_z(z: complex) -> dict:
    z = complex(z)
    return {
        "re": float(z.real),
        "im": float(z.imag),
        "abs": float(abs(z)),
        "arg_deg": float(np.degrees(np.angle(z))),
    }


def rel_dphi(j10: complex, j8: complex):
    if abs(j8) == 0.0:
        return None, None
    rel = float(abs(j10 - j8) / abs(j8))
    dphi = float(np.degrees(np.abs(np.angle(j10 * np.conj(j8)))))
    return rel, dphi


def classify_pair(rel0, dphi0, relpi, dphipi) -> str:
    def tight(rel, dphi) -> bool:
        return rel is not None and dphi is not None and rel < 0.02 and dphi < 2.0

    def fail(rel, dphi) -> bool:
        return rel is None or dphi is None or rel >= 0.05 or dphi >= 5.0

    if fail(rel0, dphi0) or fail(relpi, dphipi) or not tight(relpi, dphipi):
        return "fail"
    if tight(rel0, dphi0) and tight(relpi, dphipi):
        return "pass"
    return "borderline_theta0_sensitive"


def compare_nmax8() -> dict:
    bind_fockaudit_namespace()
    sha = assert_frozen_fit()
    missing = [tag for tag in TAGS if not job_ok_no_alarm(tag)]
    if missing:
        rec = refuse(f"compare refused: axes not all ok_no_alarm {missing}", {"missing": missing})
        rec["status"] = "compare_blocked"
        rec["fit_sha256"] = sha
        dump(OUT / "compare_nmax8.json", rec)
        return rec
    angles = {}
    read_error = []
    for aname, ptag, mtag in PAIRS:
        t8, jp8, jm8 = wq1.pair_currents(NMAX8_DIR, ptag, mtag)
        t10, jp10, jm10 = wq1.pair_currents(OUT, ptag, mtag)
        row = {"theta": aname}
        for axis in ("x", "y"):
            j8 = wq1.z_channel(t8, jp8, jm8, "main", 2, "odd", axis, method="linear_exact")
            j10 = wq1.z_channel(t10, jp10, jm10, "main", 2, "odd", axis, method="linear_exact")
            rel, dphi = rel_dphi(j10, j8)
            entry = {
                "J8": packed_z(j8),
                "J10": packed_z(j10),
                "rel": rel,
                "dphi_deg": dphi,
            }
            if axis == "x":
                ref = J8_H2X_REF[aname]
                ref_rel = float(abs(j8 - ref) / abs(ref))
                entry["J8_windowQ1_ref"] = packed_z(ref)
                entry["J8_vs_windowQ1_rel"] = ref_rel
                if ref_rel > 0.01:
                    read_error.append({"angle": aname, "axis": "x", "J8_vs_ref_rel": ref_rel})
            row[f"h2_odd_{axis}"] = entry
        angles[aname] = row
    hx0 = angles["0"]["h2_odd_x"]
    hxpi = angles["pi"]["h2_odd_x"]
    if read_error:
        verdict = "read_error_do_not_declare_truncation_failure"
    else:
        verdict = classify_pair(hx0["rel"], hx0["dphi_deg"], hxpi["rel"], hxpi["dphi_deg"])
    rec = {
        "schema": "1600nm_protocolA_qed_Q2_nmax10_compare_v1",
        "status": "compared",
        "protocol": PROTOCOL,
        "fit_sha256": sha,
        "did_not_refit_ABC": True,
        "did_not_write_fourangle_fit": True,
        "did_not_patch_photon_raw": True,
        "did_not_copy_n10_p_q": True,
        "n_max_production": N_MAX,
        "n_max_reference": 8,
        "omega0": OMEGA0_1600,
        "g": G_ABS,
        "window": "main_[2T,4T]",
        "fourier": "linear_exact_rect_not_Hann",
        "gate0": {tag: job_row(tag) for tag in TAGS},
        "angles": angles,
        "primary_channel": "h2_odd_x",
        "h2y_recorded_not_classifying": True,
        "read_error": read_error,
        "verdict": verdict,
        "not_full_ABC_convergence": True,
        "not_quantum_advantage": True,
        "not_CrI3": True,
        "g_is_abstract": True,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "compare_nmax8.json", rec)
    progress(
        f"compare verdict={verdict} "
        f"th0 rel={hx0['rel']} dphi={hx0['dphi_deg']} "
        f"thpi rel={hxpi['rel']} dphi={hxpi['dphi_deg']}"
    )
    return rec


def drive_stage1() -> int:
    bind_fockaudit_namespace()
    OUT.mkdir(parents=True, exist_ok=True)
    assert_frozen_fit()
    os.environ[AUTH_ENV] = "stage1"
    launch = launch_serial(TAGS, "stage1")
    rec = analyze_status()
    rec["launch"] = launch
    dump(OUT / "q2_status.json", rec)
    return worst_exit(TAGS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("stage1", "compare"), default="")
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--authorize-stage1", action="store_true")
    ap.add_argument("--n-steps", type=int, default=0)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    bind_fockaudit_namespace()
    if args.n_steps:
        refuse("official entry refuses --n-steps; do not shrink production n_full")
        return 3
    if args.job:
        rec = run_one_job(args.job, None)
        return exit_code_for(rec)
    if args.analyze_only:
        rec = analyze_status()
        print(json.dumps(jsonable({
            "status": rec.get("status"),
            "n_ok_no_alarm": rec.get("n_ok_no_alarm"),
            "stage1_complete": rec.get("stage1_complete"),
            "alarm_tags": rec.get("alarm_tags"),
            "next": rec.get("next"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("stage1_complete") and not rec.get("alarm_tags") else 1
    if not args.mode:
        refuse("no mode: this runner does not default-evolve; pass --analyze-only or --mode stage1 --authorize-stage1")
        return 3
    if args.mode == "compare":
        rec = compare_nmax8()
        if rec.get("verdict") == "read_error_do_not_declare_truncation_failure":
            return 1
        if rec.get("status") == "compared":
            return 0
        return 3
    if args.mode == "stage1":
        if not args.authorize_stage1:
            refuse("stage1 production requires --authorize-stage1")
            return 3
        return drive_stage1()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
