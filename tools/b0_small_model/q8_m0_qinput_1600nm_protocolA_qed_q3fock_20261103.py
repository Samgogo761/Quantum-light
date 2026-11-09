#!/usr/bin/env python3
"""Q3 minimal Fock-probe exact QED, 1600 nm protocol A (gated).

Four legal pure states x two Neel domains = 8 axes. Reuses the fockaudit
propagator without editing that file. Runtime-patches photon_raw so the
input is NOT D(alpha)S(xi)|0>. Default refuses production.

Does not extract A/B/C, does not start Q2, does not scale g.
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

import q8_m0_qinput_1600nm_protocolA_qed_fockaudit_20261102 as fa
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
from q8_m0_qinput_shared_kk_ds_odd_20261102 import jsonable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_1600nm_protocolA_qed_q3fock_20261103"
PROTOCOL = "Qin_a2_1_4_1_1600nm_protocolA_qed_q3fock"
AUTH_ENV = "QED_Q3FOCK_AUTHORIZE"
JOB_WALL_ENV = "QED_Q3FOCK_JOB_WALL_S"
N_MAX = 8
N_FULL = fa.N_FULL
OMEGA0_1600 = fa.OMEGA0_1600
FOCK_ALARM = fa.FOCK_ALARM
G_ABS = fa.G_ABS
JOB_WALL_DEFAULT = 3300.0
FLOAT_ATOL = 1e-12
PN_ATOL = 1e-12
DUMMY_THETA = 0.0
OPERATOR_ORDER = "Fock probe in {|0>,|2>}; NOT D(alpha)S(xi)|0>"

CURRENT_PROBE = None

STAGE1_TAGS = ("p_0", "m_0", "p_2", "m_2", "p_plus", "m_plus", "p_i", "m_i")
JOB_TABLE = {
    "p_0": {"neel": 1, "probe": "fock0", "label": "|0>"},
    "m_0": {"neel": -1, "probe": "fock0", "label": "|0>"},
    "p_2": {"neel": 1, "probe": "fock2", "label": "|2>"},
    "m_2": {"neel": -1, "probe": "fock2", "label": "|2>"},
    "p_plus": {"neel": 1, "probe": "plus", "label": "(|0>+|2>)/sqrt(2)"},
    "m_plus": {"neel": -1, "probe": "plus", "label": "(|0>+|2>)/sqrt(2)"},
    "p_i": {"neel": 1, "probe": "iplus", "label": "(|0>+i|2>)/sqrt(2)"},
    "m_i": {"neel": -1, "probe": "iplus", "label": "(|0>+i|2>)/sqrt(2)"},
}


def photon_raw_q3(kind: str, n_max: int, theta: float = 0.0):
    del kind, theta
    if CURRENT_PROBE is None:
        raise RuntimeError("CURRENT_PROBE unset; refusing DS fallback")
    c = np.zeros(int(n_max) + 1, dtype=complex)
    s2 = 1.0 / np.sqrt(2.0)
    if CURRENT_PROBE == "fock0":
        c[0] = 1.0
    elif CURRENT_PROBE == "fock2":
        c[2] = 1.0
    elif CURRENT_PROBE == "plus":
        c[0] = s2
        c[2] = s2
    elif CURRENT_PROBE == "iplus":
        c[0] = s2
        c[2] = 1j * s2
    else:
        raise ValueError(CURRENT_PROBE)
    return c


def expected_p_n(probe: str, n_max: int) -> np.ndarray:
    p = np.zeros(int(n_max) + 1, float)
    if probe == "fock0":
        p[0] = 1.0
    elif probe == "fock2":
        p[2] = 1.0
    elif probe in ("plus", "iplus"):
        p[0] = 0.5
        p[2] = 0.5
    else:
        raise ValueError(probe)
    return p


def install_photon_patch() -> None:
    odd.photon_raw = photon_raw_q3
    fa.odd.photon_raw = photon_raw_q3
    if fa.odd is not odd:
        raise RuntimeError("fa.odd is not the shared odd module")


def bind_fockaudit_namespace() -> None:
    fa.OUT = OUT
    fa.PROTOCOL = PROTOCOL
    fa.N_FULL = N_FULL

    def job_spec(tag: str) -> dict:
        if tag not in JOB_TABLE:
            raise ValueError(f"unknown q3fock tag {tag}")
        spec = {
            "tag": tag,
            "neel": int(JOB_TABLE[tag]["neel"]),
            "theta": DUMMY_THETA,
            "n_max": N_MAX,
            "n_steps": N_FULL,
            "family": "q3fock",
            "probe": JOB_TABLE[tag]["probe"],
        }
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
        "did_not_start_Q2": True,
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


def occupation_ok(rec, probe: str) -> bool:
    pn = ((rec or {}).get("photon_unnormalized") or {}).get("p_n")
    if not pn:
        return False
    got = np.asarray(pn, float)
    want = expected_p_n(probe, N_MAX)
    if got.shape != want.shape:
        return False
    if float(np.max(np.abs(got - want))) > PN_ATOL:
        return False
    ds = odd.photon_moments(odd.dyn.displaced_squeezed_coefficients(odd.ALPHA, odd.R_SQ, 0.0, N_MAX))
    if float(np.max(np.abs(got - np.asarray(ds["p_n"], float)))) < 1e-6:
        return False
    return True


def probe_fields_ok(rec, tag: str) -> bool:
    probe = JOB_TABLE[tag]["probe"]
    defn = (rec or {}).get("definition") or {}
    if rec.get("probe") != probe:
        return False
    if defn.get("probe") != probe:
        return False
    if defn.get("operator_order") != OPERATOR_ORDER:
        return False
    if defn.get("not_DS") is not True:
        return False
    return occupation_ok(rec, probe)


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
        and rec.get("did_not_scale_g") is True
        and n_full == N_FULL
        and int(ledger.get("steps_completed") or -1) == N_FULL
        and series.is_file()
        and probe_fields_ok(rec, tag)
    )


def rewrite_probe_record(tag: str, rec: dict | None) -> dict:
    if not rec:
        rec = load_job(tag) or {}
    probe = JOB_TABLE[tag]["probe"]
    defn = dict(rec.get("definition") or {})
    defn["operator_order"] = OPERATOR_ORDER
    defn["probe"] = probe
    defn["probe_label"] = JOB_TABLE[tag]["label"]
    defn["not_DS"] = True
    defn["dummy_theta"] = DUMMY_THETA
    defn["alpha"] = None
    defn["r"] = None
    rec["definition"] = defn
    rec["probe"] = probe
    rec["probe_label"] = JOB_TABLE[tag]["label"]
    rec["not_DS"] = True
    rec["did_not_extract_ABC"] = True
    rec["did_not_start_Q2"] = True
    rec["input_is_legal_pure_state"] = True
    rec["protocol"] = PROTOCOL
    if rec.get("status") in ("ok", "ok_fock_alarm", "timeout_partial", "incomplete", "failed"):
        if not occupation_ok(rec, probe):
            rec["status"] = "failed_wrong_input"
            rec["error"] = rec.get("error") or "photon_unnormalized.p_n is not the Q3 Fock probe"
            rec["finite_state"] = False
            progress(f"{tag} STOP: input occupation is not probe {probe}")
    dump(fa.job_path(tag), rec)
    return rec


def job_row(tag: str):
    rec = load_job(tag)
    if rec is None:
        return {"status": None, "present": False, "probe": JOB_TABLE[tag]["probe"]}
    n_full = official_n_full(rec)
    ledger = rec.get("ledger") or {}
    return {
        "present": True,
        "status": rec.get("status"),
        "neel": rec.get("neel"),
        "probe": rec.get("probe"),
        "probe_label": rec.get("probe_label"),
        "operator_order": (rec.get("definition") or {}).get("operator_order"),
        "not_DS": rec.get("not_DS"),
        "occupation_ok": occupation_ok(rec, JOB_TABLE[tag]["probe"]) if rec.get("photon_unnormalized") else False,
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
        "error": rec.get("error"),
        "fock_alarm_0p05": rec.get("fock_alarm_0p05"),
        "used_pf_OMEGA0": rec.get("used_pf_OMEGA0"),
        "did_not_scale_g": rec.get("did_not_scale_g"),
        "ok_no_alarm": job_ok_no_alarm(tag),
    }


def analyze_status() -> dict:
    bind_fockaudit_namespace()
    jobs = {tag: job_row(tag) for tag in STAGE1_TAGS}
    n_ok = sum(1 for tag in STAGE1_TAGS if job_ok_no_alarm(tag))
    n_alarm = [
        tag for tag in STAGE1_TAGS
        if (load_job(tag) or {}).get("fock_alarm_0p05") is True
    ]
    rec = {
        "status": "analyze",
        "protocol": PROTOCOL,
        "omega0": OMEGA0_1600,
        "g": G_ABS,
        "n_max": N_MAX,
        "n_steps": N_FULL,
        "fock_alarm": FOCK_ALARM,
        "used_pf_OMEGA0": False,
        "did_not_scale_g": True,
        "did_not_extract_ABC": True,
        "did_not_start_Q2": True,
        "gh5_is_not_qed": True,
        "stage1_tags": list(STAGE1_TAGS),
        "jobs": jobs,
        "n_ok_no_alarm": n_ok,
        "stage1_complete": n_ok == 8,
        "alarm_tags": n_alarm,
        "next": (
            "user must authorize stage 1; do not start production from this analyze"
            if n_ok == 0 else
            "resume unfinished q3fock axes after explicit stage-1 authorization"
            if n_ok < 8 else
            "q3fock 8/8; run the analyze script; do not start Q2 or extra probes"
        ),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "q3fock_status.json", rec)
    progress(f"analyze ok={n_ok}/8 alarm={n_alarm}")
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
    if st in ("refused_not_authorized", "failed_wrong_input"):
        return 3
    return 1


def worst_exit(tags) -> int:
    codes = [exit_code_for(load_job(tag)) for tag in tags]
    for pref in (2, 4, 3, 1):
        if pref in codes:
            return pref
    return 0 if codes and all(c == 0 for c in codes) else 1


def apply_job_wall_from_env() -> None:
    raw = os.environ.get(JOB_WALL_ENV)
    if raw:
        fa.JOB_WALL_S = float(raw)


def run_one_job(tag: str) -> dict:
    bind_fockaudit_namespace()
    install_photon_patch()
    if tag not in JOB_TABLE:
        rec = refuse(f"unknown tag {tag}", {"tag": tag})
        rec["status"] = "unknown_tag"
        return rec
    if job_ok_no_alarm(tag):
        progress(f"{tag} already ok_no_alarm; will not re-evolve")
        return load_job(tag)
    if not env_authorized("stage1"):
        return refuse("direct --job requires QED_Q3FOCK_AUTHORIZE=stage1", {"tag": tag})
    global CURRENT_PROBE
    CURRENT_PROBE = JOB_TABLE[tag]["probe"]
    apply_job_wall_from_env()
    rec = fa.run_one_job(tag, None)
    rec = rewrite_probe_record(tag, rec)
    return rec


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
        if rec and (rec.get("status") == "ok_fock_alarm" or rec.get("fock_alarm_0p05") is True
                    or rec.get("status") == "failed_wrong_input"):
            progress(f"STOP {tag}: {rec.get('status')}; do not retune g or continue")
            codes[tag] = 2 if rec.get("fock_alarm_0p05") else 3
            stopped = tag
            break
        envp = os.environ.copy()
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            envp[k] = "1"
        envp[AUTH_ENV] = authorize_stage
        envp[JOB_WALL_ENV] = str(JOB_WALL_DEFAULT)
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", tag]
        progress(f"SERIAL launch {tag} probe={JOB_TABLE[tag]['probe']} neel={JOB_TABLE[tag]['neel']}")
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


def preflight_input() -> dict:
    bind_fockaudit_namespace()
    install_photon_patch()
    hashes = fa.source_hashes()
    bad = fa.hash_abort(hashes)
    global CURRENT_PROBE
    rows = {}
    ds_raw = odd.dyn.displaced_squeezed_coefficients(odd.ALPHA, odd.R_SQ, 0.0, N_MAX)
    ds = odd.photon_moments(ds_raw)
    for probe in ("fock0", "fock2", "plus", "iplus"):
        CURRENT_PROBE = probe
        raw = photon_raw_q3("ignored", N_MAX, 99.0)
        mom = odd.photon_moments(raw)
        want = expected_p_n(probe, N_MAX)
        got = np.asarray(mom["p_n"], float)
        rows[probe] = {
            "p_n": mom["p_n"],
            "mean_n": mom["mean_n"],
            "mean_a": mom["mean_a"],
            "mean_aa": mom["mean_aa"],
            "unnormalized_mass": mom["unnormalized_mass"],
            "matches_expected": bool(np.max(np.abs(got - want)) <= PN_ATOL),
            "distinguishable_from_DS": bool(np.max(np.abs(got - np.asarray(ds["p_n"], float))) > 1e-6),
        }
    CURRENT_PROBE = None
    rec = {
        "status": "preflight_input",
        "protocol": PROTOCOL,
        "did_not_propagate": True,
        "did_not_write_production_jobs": True,
        "hash_abort": bad,
        "hashes_ok": not bad,
        "g": G_ABS,
        "omega0": OMEGA0_1600,
        "n_max": N_MAX,
        "n_full": N_FULL,
        "DS_p_n_reference": ds["p_n"],
        "DS_mean_n": ds["mean_n"],
        "probes": rows,
        "all_probes_ok": all(r["matches_expected"] and r["distinguishable_from_DS"] for r in rows.values()),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    rec["status"] = "preflight_ok" if rec["hashes_ok"] and rec["all_probes_ok"] else "preflight_fail"
    dump(OUT / "preflight_input.json", rec)
    progress(f"preflight {rec['status']} hashes_ok={rec['hashes_ok']} probes_ok={rec['all_probes_ok']}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("stage1",), default="")
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--preflight-input", action="store_true")
    ap.add_argument("--authorize-stage1", action="store_true")
    ap.add_argument("--n-steps", type=int, default=0)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    bind_fockaudit_namespace()
    if args.n_steps:
        refuse("official entry refuses --n-steps; do not shrink production n_full")
        return 3
    if args.preflight_input:
        rec = preflight_input()
        print(json.dumps(jsonable({
            "status": rec.get("status"),
            "hashes_ok": rec.get("hashes_ok"),
            "all_probes_ok": rec.get("all_probes_ok"),
            "hash_abort": rec.get("hash_abort"),
            "probes": {k: {"mean_n": v["mean_n"], "matches_expected": v["matches_expected"],
                           "distinguishable_from_DS": v["distinguishable_from_DS"]}
                       for k, v in rec["probes"].items()},
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") == "preflight_ok" else 1
    if args.job:
        rec = run_one_job(args.job)
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
        refuse("no mode: this runner does not default-evolve; pass --preflight-input, --analyze-only, or authorized --mode stage1")
        return 3
    if args.mode == "stage1":
        if not args.authorize_stage1:
            refuse("stage1 production requires --authorize-stage1")
            return 3
        pre = preflight_input()
        if pre.get("status") != "preflight_ok":
            refuse("stage1 refused: preflight_input failed", {"preflight": pre})
            return 3
        os.environ[AUTH_ENV] = "stage1"
        launch = launch_serial(STAGE1_TAGS, "stage1")
        rec = analyze_status()
        rec["launch"] = launch
        dump(OUT / "q3fock_status.json", rec)
        return worst_exit(STAGE1_TAGS)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
