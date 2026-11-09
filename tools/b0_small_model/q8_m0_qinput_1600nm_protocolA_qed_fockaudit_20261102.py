#!/usr/bin/env python3
"""Protocol A 1600 nm exact-QED Fock truncation audit.

Keep Q6 M0, same k/-k, D(0.5)S(0.3 e^{i theta})|0>, R=1.273,
abstract g=0.005 A^{-1}, 1+4+1 coupling switch. Change only omega0:
pass 0.775 explicitly into build_shared and Envelope.

This runner does NOT extract A/B/C, does NOT launch four-angle QED,
does NOT scale g, and does NOT treat GH5 Weyl occupation as Fock <n>.
First evolution job is n8_p_q (n_max=8, Neel=+1, theta=pi/4).
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

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_qin_a2_141_qed_q3_covsign_20261120 as q3
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_1600nm_protocolA_qed_fockaudit_20261102"
GH5_DIR = ROOT / "results/q_line/qinput_1600nm_protocolA_gh5_20261102"
DOCS = ROOT / "docs/04_BSV量子光"
TASK_TAG = "20261102"
RECORD_DATE = "2026-11-02"
PROTOCOL = "Qin_a2_1_4_1_1600nm_protocolA_qed_fockaudit"
OMEGA0_Q6 = 0.05
OMEGA0_1600 = 0.775
G_ABS = 0.005
ALPHA = float(np.real(odd.ALPHA))
R_SQ = float(odd.R_SQ)
HBAR_EV_FS = 0.6582119568564649
N_FULL = 11154
N_SMOKE = 200
SAMPLE_EVERY = 32
CHECKPOINT_EVERY = 400
DRIVER_WALL_S = 3600.0
JOB_WALL_S = 3300.0
FOCK_ALARM = 0.05
NMAX_REF = 80
THETA_0 = 0.0
THETA_Q = 0.25 * np.pi
THETA_H = 0.5 * np.pi
THETA_PI = float(np.pi)
THETA_T = 1.5 * np.pi
EXPECTED_HASHES = dict(q3.EXPECTED_HASHES)
FROZEN = tuple(HERE / name for name in EXPECTED_HASHES)

jsonable = odd.jsonable


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


def source_hashes() -> dict:
    out = {p.name: sha256_file(p) for p in FROZEN}
    out[Path(__file__).name] = sha256_file(Path(__file__))
    return out


def hash_abort(hashes: dict) -> list:
    bad = []
    for name, expect in EXPECTED_HASHES.items():
        got = hashes.get(name)
        if got != expect:
            bad.append({"file": name, "expected": expect, "got": got})
    return bad


def assert_protocol_omega(omega0: float) -> None:
    if abs(float(omega0) - OMEGA0_1600) > 1e-15:
        raise RuntimeError(f"protocol A requires omega0={OMEGA0_1600}, got {omega0}")
    if abs(float(pf.OMEGA0) - OMEGA0_Q6) > 1e-15:
        raise RuntimeError(f"frozen pf.OMEGA0 changed: {pf.OMEGA0}")
    if abs(float(sh.OMEGA0) - OMEGA0_Q6) > 1e-15:
        raise RuntimeError(f"frozen sh.OMEGA0 changed: {sh.OMEGA0}")
    if abs(float(sh.G_ABS) - G_ABS) > 1e-15:
        raise RuntimeError(f"frozen g changed: {sh.G_ABS}")


def make_env(omega0: float = OMEGA0_1600):
    assert_protocol_omega(omega0)
    return dyn.Envelope(float(omega0), ramp_cycles=1.0, flat_cycles=4.0)


def envelope_meta(env) -> dict:
    t_h = float(getattr(R0, "t_h", 1.0))
    T0 = float(env.period)
    return {
        "omega0": float(env.omega0),
        "t_h_eV": t_h,
        "period_model": T0,
        "period_fs": T0 * HBAR_EV_FS / t_h,
        "t_total_model": float(env.t_total),
        "t_total_fs": float(env.t_total) * HBAR_EV_FS / t_h,
        "ramp_cycles": 1.0,
        "flat_cycles": 4.0,
        "not_25fs_gaussian": True,
        "not_free_space_pulse": True,
        "used_pf_OMEGA0": False,
        "g_abs": G_ABS,
        "g_note": sh.g_normalization_note(2, G_ABS),
        "k_frac": list(sh.K_FRAC),
        "two_hbar_omega_eV": 2.0 * float(env.omega0) * t_h,
        "three_hbar_omega_eV": 3.0 * float(env.omega0) * t_h,
        "local_gap_note_eV": 1.91,
    }


def job_spec(tag: str) -> dict:
    table = {
        "smoke_n8_p_q": {
            "neel": 1, "theta": THETA_Q, "n_max": 8, "n_steps": N_SMOKE, "family": "smoke",
        },
        "n8_p_q": {
            "neel": 1, "theta": THETA_Q, "n_max": 8, "n_steps": N_FULL, "family": "evolve",
        },
        "n8_p_0": {
            "neel": 1, "theta": THETA_0, "n_max": 8, "n_steps": N_FULL, "family": "control",
        },
        "n10_p_q": {
            "neel": 1, "theta": THETA_Q, "n_max": 10, "n_steps": N_FULL, "family": "raise_nmax",
        },
    }
    if tag not in table:
        raise ValueError(tag)
    spec = dict(table[tag])
    spec["tag"] = tag
    return spec


def job_path(tag: str) -> Path:
    return OUT / f"job_{tag}.json"


def load_job(tag: str):
    path = job_path(tag)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def job_finished_ok(tag: str) -> bool:
    rec = load_job(tag)
    return bool(rec and rec.get("status") in ("ok", "ok_fock_alarm"))


def sample_js_for(n_steps: int):
    js = list(range(0, n_steps, SAMPLE_EVERY))
    if not js or js[-1] != n_steps - 1:
        js.append(n_steps - 1)
    return js


def ckpt_paths(tag: str):
    return OUT / f"ckpt_{tag}.npz", OUT / f"ckpt_{tag}_samples.json"


def save_ckpt(tag: str, psi, next_j: int, samples, meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p_npz, p_js = ckpt_paths(tag)
    np.savez_compressed(
        p_npz,
        psi=np.asarray(psi, complex),
        next_j=np.int64(next_j),
        omega0=np.float64(meta["omega0"]),
        n_max=np.int64(meta["n_max"]),
        theta=np.float64(meta["theta"]),
        n_steps=np.int64(meta["n_steps"]),
        neel=np.int64(meta["neel"]),
    )
    p_js.write_text(json.dumps(jsonable(samples), ensure_ascii=False), encoding="utf-8")


def load_ckpt(tag: str, spec: dict):
    p_npz, p_js = ckpt_paths(tag)
    if not p_npz.is_file() or not p_js.is_file():
        return None
    z = np.load(p_npz)
    if "omega0" not in z.files:
        return {"status": "ckpt_missing_meta"}
    got = {
        "omega0": float(z["omega0"]),
        "n_max": int(z["n_max"]),
        "theta": float(z["theta"]),
        "n_steps": int(z["n_steps"]),
        "neel": int(z["neel"]),
    }
    want = {
        "omega0": OMEGA0_1600,
        "n_max": int(spec["n_max"]),
        "theta": float(spec["theta"]),
        "n_steps": int(spec["n_steps"]),
        "neel": int(spec["neel"]),
    }
    for key, val in want.items():
        if abs(float(got[key]) - float(val)) > 1e-12:
            return {"status": "ckpt_contract_mismatch", "got": got, "want": want}
    samples = json.loads(p_js.read_text(encoding="utf-8"))
    return {
        "status": "ok",
        "psi": np.asarray(z["psi"], complex),
        "next_j": int(z["next_j"]),
        "samples": samples,
        "meta": got,
    }


def dedup_samples_by_step(samples):
    out, seen = [], set()
    for rec in list(samples or []):
        st = int(rec.get("step", -1))
        if st in seen:
            continue
        seen.add(st)
        out.append(rec)
    return out


def fock_trace(samples, send) -> dict:
    rows = list(samples or [])
    if send is not None:
        rows = rows + [send]
    if not rows:
        return {
            "max_top_fock": None, "max_next_top_fock": None, "max_n_ph": None,
            "min_n_ph": None, "max_abs_norm_minus_1": None, "n_rows": 0,
        }
    tops = [float(s["top_fock_mass"]) for s in rows]
    nxts = [float(s["p_n"][-2]) for s in rows]
    nphs = [float(s["n_ph"]) for s in rows]
    norms = [float(s["norm"]) for s in rows]
    top2 = [float(s["p_n"][-1] + s["p_n"][-2]) for s in rows]
    return {
        "max_top_fock": float(np.max(tops)),
        "end_top_fock": float(tops[-1]),
        "max_next_top_fock": float(np.max(nxts)),
        "max_top_two": float(np.max(top2)),
        "max_n_ph": float(np.max(nphs)),
        "min_n_ph": float(np.min(nphs)),
        "end_n_ph": float(nphs[-1]),
        "max_abs_norm_minus_1": float(np.max(np.abs(np.asarray(norms) - 1.0))),
        "n_rows": len(rows),
        "p_n_end": [float(x) for x in rows[-1]["p_n"]],
    }


def series_from_samples(samples):
    t = np.array([s["t"] for s in samples], float)
    j = np.array(
        [[s["mean_J"][a]["re"] + 1j * s["mean_J"][a]["im"] for a in range(2)]
         for s in samples], complex)
    n_ph = np.array([s["n_ph"] for s in samples], float)
    top = np.array([s["top_fock_mass"] for s in samples], float)
    nxt = np.array([s["p_n"][-2] for s in samples], float)
    nrm = np.array([s["norm"] for s in samples], float)
    ss = np.array([s["s"] for s in samples], float)
    pn = np.array([s["p_n"] for s in samples], float)
    return t, j, n_ph, top, nxt, nrm, ss, pn


def save_series_npz(tag: str, samples) -> None:
    t, j, n_ph, top, nxt, nrm, ss, pn = series_from_samples(samples)
    np.savez_compressed(
        OUT / f"series_{tag}.npz",
        t=t, Jx=j[:, 0], Jy=j[:, 1], n_ph=n_ph, top_fock=top,
        next_top_fock=nxt, norm=nrm, s=ss, p_n=pn)


def propagate_ckpt(model, psi, t0, t1, env, n_steps, sample_js, clock, tag, meta,
                   start_j: int = 0, samples=None, ckpt_every: int = CHECKPOINT_EVERY):
    dt = (t1 - t0) / n_steps
    sample_set = set(int(j) for j in sample_js)
    samples = dedup_samples_by_step(list(samples or []))
    worst_last = 0.0
    j = int(start_j)
    timed_out = False
    timeout_msg = None
    t_loop0 = time.perf_counter()
    try:
        while j < n_steps:
            clock.check()
            tm = t0 + (j + 0.5) * dt
            s = env.s(tm)
            Ds = dyn.peierls_at(model.q_units, s, model.n_max)
            psi, last = pf.taylor_step(model, psi, s, dt, Ds=Ds)
            worst_last = max(worst_last, last)
            if j in sample_set:
                t_now = t0 + (j + 1) * dt
                obs = pf.snapshot(model, psi, env.s(t_now), Ds=None)
                obs["t"] = t_now
                dhds = pf.dHds_expect(model, psi, env.s(t_now))
                dsdt = env.ds(t_now)
                obs["dHds_expect"] = pf.cpack(dhds)
                obs["dsdt"] = float(dsdt)
                obs["power_dHds_dsdt"] = float((dhds * dsdt).real)
                obs["step"] = j + 1
                samples.append(obs)
            if (j + 1) % 32 == 0 or j == n_steps - 1:
                nrm = float(np.linalg.norm(psi))
                if (not np.isfinite(psi).all()) or nrm > 2.0 or nrm < 0.5:
                    raise RuntimeError(f"norm {nrm} at step {j}")
            j += 1
            if ckpt_every and j % ckpt_every == 0:
                save_ckpt(tag, psi, j, samples, meta)
                progress(f"{tag} ckpt j={j}/{n_steps} wall={clock.elapsed():.1f}s")
    except TimeoutError as exc:
        timed_out = True
        timeout_msg = str(exc)
        save_ckpt(tag, psi, j, samples, meta)
    return psi, {
        "steps_requested": int(n_steps),
        "steps_completed": int(j),
        "dt": float(dt),
        "t0": float(t0),
        "t1_requested": float(t1),
        "t1_reached": float(t0 + j * dt),
        "order": pf.ORDER,
        "max_last_taylor_term": float(worst_last),
        "samples": samples,
        "timed_out": bool(timed_out),
        "timeout_msg": timeout_msg,
        "next_j": int(j),
        "loop_seconds": float(time.perf_counter() - t_loop0),
        "omega0": OMEGA0_1600,
        "used_pf_OMEGA0": False,
    }


def clarify_weyl() -> dict:
    hashes = source_hashes()
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "clarify",
        "gh5_dir": str(GH5_DIR.as_posix()),
        "weyl_is_not_qed_n": True,
        "weyl_formula": "0.25*(X_f^2 + P_f^2) - 0.5 on GH quadrature nodes, not <a^dag a>",
        "do_not_use_as_fock_budget": True,
        "input_nbar": float(ALPHA ** 2 + np.sinh(R_SQ) ** 2),
        "locked_first_evolve": "n8_p_q",
        "source_hashes": hashes,
        "hash_abort": hash_abort(hashes),
        "rows": {},
    }
    if rec["hash_abort"]:
        rec["status"] = "hash_abort"
        dump(OUT / "weyl_is_not_qed_n.json", rec)
        return rec
    for path in sorted(GH5_DIR.glob("job_*.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        tag = job.get("tag", path.stem.replace("job_", "", 1))
        rec["rows"][tag] = {
            "theta": job.get("theta"),
            "neel": job.get("neel"),
            "status": job.get("status"),
            "max_weyl_n": job.get("max_weyl_n"),
            "omega0": job.get("omega0"),
            "family": job.get("family"),
            "n_steps": job.get("n_steps"),
        }
    named = {k: rec["rows"][k]["max_weyl_n"] for k in rec["rows"]
             if rec["rows"][k]["max_weyl_n"] is not None}
    rec["named_max_weyl_n"] = named
    rec["review_point_p_q"] = rec["rows"].get("p_q", {}).get("max_weyl_n")
    rec["note"] = (
        "External review cited 7.91 from p_q. p_h/p_t Weyl are larger (~9.11) "
        "but Weyl is still not a Fock budget; first QED angle remains theta=pi/4."
    )
    dump(OUT / "weyl_is_not_qed_n.json", rec)
    progress(
        f"clarify weyl p_q={rec['review_point_p_q']} "
        f"not_qed_n=True first_job=n8_p_q"
    )
    return rec


def input_audit() -> dict:
    hashes = source_hashes()
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "input",
        "omega0": OMEGA0_1600,
        "omega0_does_not_change_input_fock": True,
        "n_max_propagator": 8,
        "n_max_ref": NMAX_REF,
        "alpha": ALPHA, "r": R_SQ,
        "source_hashes": hashes,
        "hash_abort": hash_abort(hashes),
        "rows": {},
    }
    if rec["hash_abort"]:
        rec["status"] = "hash_abort"
        dump(OUT / "input_fock.json", rec)
        return rec
    for key, th in (("0", THETA_0), ("q", THETA_Q), ("pi", THETA_PI),
                    ("h", THETA_H), ("t", THETA_T)):
        rec["rows"][key] = q3.theta_fock_row(th)
    rec["P_n_ge9_ideal80_q"] = rec["rows"]["q"]["P_n_ge9_ideal80"]
    rec["P_n8_trunc8_q"] = rec["rows"]["q"]["P_n8_trunc8_normalized"]
    rec["input_tail_small"] = bool(rec["P_n_ge9_ideal80_q"] < 1e-4)
    rec["note"] = (
        "Input <n>=0.343 and P(n>8) on the undisplaced-squeezed expansion "
        "do not bound occupation during 1600 nm joint evolution."
    )
    dump(OUT / "input_fock.json", rec)
    progress(
        f"input P(n>=9)|80 at pi/4={rec['P_n_ge9_ideal80_q']:.3e} "
        f"trunc8_top={rec['P_n8_trunc8_q']:.3e}"
    )
    return rec


def run_one_job(tag: str, n_steps_override: int | None = None) -> dict:
    spec = job_spec(tag)
    if n_steps_override:
        spec["n_steps"] = int(n_steps_override)
    n_max = int(spec["n_max"])
    n_steps = int(spec["n_steps"])
    neel = int(spec["neel"])
    theta = float(spec["theta"])
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"tag": tag, "status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(job_path(tag), rec)
        return rec
    prev = load_job(tag)
    if prev and prev.get("status") in ("ok", "ok_fock_alarm"):
        if (prev.get("omega0") != OMEGA0_1600 or prev.get("n_steps") != n_steps
                or prev.get("n_max") != n_max):
            rec = {
                "tag": tag, "status": "finished_contract_mismatch", "stopped": True,
                "have": {k: prev.get(k) for k in ("omega0", "n_steps", "n_max", "status")},
                "want_omega0": OMEGA0_1600, "want_n_steps": n_steps, "want_n_max": n_max,
            }
            dump(job_path(tag), rec)
            return rec
        progress(f"{tag} already {prev['status']}, skip")
        return prev

    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = make_env(OMEGA0_1600)
    if abs(float(env.omega0) - OMEGA0_1600) > 1e-15:
        raise RuntimeError("env silently used Q6 omega0")
    dt = env.t_total / n_steps
    ek, emk, masses = odd.ground_pair_neel(geo, k, neel)
    raw = odd.photon_raw("ds", n_max, float(theta))
    raw_diag = odd.photon_moments(raw)
    xp = q3.fock_xp_moments(raw)
    c = raw / np.sqrt(raw_diag["unnormalized_mass"])
    model = sh.build_shared(geo, R0, k, neel, n_max=n_max, omega0=OMEGA0_1600, g=G_ABS)
    if abs(float(model.omega0) - OMEGA0_1600) > 1e-15:
        raise RuntimeError("build_shared did not keep explicit 1600 nm omega0")
    if int(model.n_max) != n_max:
        raise RuntimeError(f"n_max mismatch {model.n_max} vs {n_max}")
    psi, prod_n = odd.product_normalized(ek, emk, c)
    meta = {
        "omega0": OMEGA0_1600, "n_max": n_max, "theta": theta,
        "n_steps": n_steps, "neel": neel,
    }

    start_j = 0
    samples = []
    resumed = False
    loaded = load_ckpt(tag, spec)
    if loaded is not None and loaded.get("status") == "ok":
        psi, start_j, samples = loaded["psi"], loaded["next_j"], loaded["samples"]
        samples = dedup_samples_by_step(samples)
        resumed = True
        progress(f"{tag} resume next_j={start_j} n_samples={len(samples)}")
    elif loaded is not None and loaded.get("status") != "ok":
        rec = {"tag": tag, "status": loaded.get("status"), "stopped": True, "ckpt": loaded}
        dump(job_path(tag), rec)
        return rec

    dump(job_path(tag), {
        "tag": tag, "status": "running", "theta": theta, "neel": neel,
        "n_max": n_max, "omega0": OMEGA0_1600, "n_steps": n_steps,
        "resumed": resumed, "next_j": start_j,
        "utc": datetime.now(timezone.utc).isoformat(),
        "did_not_extract_ABC": True,
    })
    s0 = pf.snapshot(model, psi, env.s(start_j * dt))
    if resumed and prev and prev.get("initial"):
        s0 = prev["initial"]
    progress(
        f"{tag} start n_max={n_max} theta={theta:.6f} neel={neel} "
        f"omega0={OMEGA0_1600} dim={model.dim} j0={start_j}/{n_steps} "
        f"used_pf_OMEGA0=False"
    )
    clock = dyn.WallClock(JOB_WALL_S)
    finite_ok = True
    err = None
    try:
        psi, led = propagate_ckpt(
            model, psi, 0.0, env.t_total, env, n_steps, sample_js_for(n_steps),
            clock, tag, meta, start_j=start_j, samples=samples)
    except Exception as exc:
        finite_ok = False
        err = f"{type(exc).__name__}: {exc}"
        led = {
            "steps_requested": n_steps, "steps_completed": start_j, "dt": float(dt),
            "samples": samples, "timed_out": False, "timeout_msg": None,
            "max_last_taylor_term": None, "next_j": start_j,
            "t1_reached": start_j * dt, "loop_seconds": clock.elapsed(),
            "omega0": OMEGA0_1600, "used_pf_OMEGA0": False,
        }
        save_ckpt(tag, psi, start_j, samples, meta)
        progress(f"{tag} exception {err}")

    wall = clock.elapsed()
    samples = led.get("samples") or []
    if samples:
        save_series_npz(tag, samples)
        save_ckpt(tag, psi, led.get("next_j", start_j), samples, meta)
    send = samples[-1] if samples else pf.snapshot(model, psi, env.s(led.get("t1_reached", 0.0)))
    trace = fock_trace(samples, send)
    fock_alarm = bool(trace["max_top_fock"] is not None and trace["max_top_fock"] > FOCK_ALARM)
    timed_out = bool(led.get("timed_out"))
    completed = int(led.get("steps_completed") or 0)
    t_reach = float(led.get("t1_reached") or 0.0)
    t_over = t_reach / env.period
    if not finite_ok:
        status = "failed"
    elif timed_out:
        status = "timeout_partial"
    elif completed < n_steps:
        status = "incomplete"
    elif fock_alarm:
        status = "ok_fock_alarm"
    else:
        status = "ok"

    report = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "tag": tag, "neel": neel, "theta": theta, "status": status,
        "family": spec["family"], "resumed": resumed,
        "resume_from_next_j": int(start_j) if resumed else 0,
        "finite_state": finite_ok, "error": err,
        "fock_alarm_0p05": fock_alarm, "source_hashes": hashes,
        "n_max": n_max, "omega0": OMEGA0_1600, "g": G_ABS,
        "used_pf_OMEGA0": False,
        "did_not_scale_g": True,
        "did_not_extract_ABC": True,
        "gh5_is_not_cohen_mtef": True,
        "not_macroscopic_BSV": True,
        "weyl_was_not_used_as_n": True,
        "definition": {
            "n_max": n_max, "g": G_ABS, "omega0": OMEGA0_1600,
            "alpha": ALPHA, "r": R_SQ, "theta": theta,
            "operator_order": "D(alpha)S(xi)|0>",
            "not_BSV": True,
            "k_fractional": list(sh.K_FRAC),
            "n_full": n_steps, "dt": float(dt),
            "sample_every": SAMPLE_EVERY,
            "period": env.period, "t_total": env.t_total,
            "plateau": list(env.plateau),
            "s_is_coupling_switch_not_free_space_pulse": True,
            "joint_dim": int(model.dim),
        },
        "envelope": envelope_meta(env),
        "electron": masses,
        "photon_unnormalized": raw_diag,
        "photon_xp_input": xp,
        "product_norm_before_renorm": prod_n,
        "initial": s0, "endpoint": send,
        "ledger": {k: led[k] for k in led if k != "samples"},
        **trace,
        "t_reached": t_reach, "t_reached_over_T0": t_over,
        "covers_requested": bool(completed >= n_steps),
        "wall_seconds": wall,
        "seconds_per_step": (wall / max(completed - start_j, 1)),
        "n_samples": len(samples),
        "series_npz": str((OUT / f"series_{tag}.npz").as_posix()),
        "integer_Fourier_is_not_certified_HHG": True,
        "did_not_auto_raise_nmax": True,
        "mechanism_not_claimed_under_exact_QED": True,
    }
    dump(job_path(tag), report)
    progress(
        f"{tag} {status} steps={completed}/{n_steps} t/T0={t_over:.3f} "
        f"n_ph={trace.get('max_n_ph')} top={trace.get('max_top_fock')} "
        f"omega0={OMEGA0_1600} wall={wall:.1f}s"
    )
    return report


def launch_tags(tags, n_steps_override: int | None = None) -> dict:
    t0 = time.perf_counter()
    procs, logs = {}, {}
    for tag in tags:
        if job_finished_ok(tag):
            progress(f"drive skip {tag} already ok")
            continue
        logf = (OUT / f"log_{tag}.txt").open("a", encoding="utf-8")
        logs[tag] = logf
        envp = os.environ.copy()
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            envp[k] = "1"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", tag]
        if n_steps_override:
            cmd.extend(["--n-steps", str(int(n_steps_override))])
        progress(f"launch {' '.join(cmd[-4:])}")
        procs[tag] = subprocess.Popen(
            cmd, cwd=str(HERE), env=envp, stdout=logf, stderr=subprocess.STDOUT)
    while True:
        elapsed = time.perf_counter() - t0
        alive = {tag: p for tag, p in procs.items() if p.poll() is None}
        if not alive:
            break
        if elapsed >= DRIVER_WALL_S:
            progress(f"driver wall {DRIVER_WALL_S}s: terminating {list(alive)}")
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
    return {
        "returncodes": {tag: (procs[tag].returncode if tag in procs else 0) for tag in tags},
        "driver_wall_s": time.perf_counter() - t0,
    }


def analyze(tags=None) -> dict:
    tags = list(tags or ["smoke_n8_p_q", "n8_p_q"])
    jobs = {tag: load_job(tag) for tag in tags}
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "analyze",
        "omega0": OMEGA0_1600,
        "used_pf_OMEGA0": False,
        "did_not_extract_ABC": True,
        "did_not_claim_mechanism_under_exact_QED": True,
        "gh5_is_not_cohen_mtef": True,
        "weyl_is_not_qed_n": True,
        "fock_alarm": FOCK_ALARM,
        "jobs": {tag: None if jobs[tag] is None else {
            "status": jobs[tag].get("status"),
            "omega0": jobs[tag].get("omega0"),
            "n_max": jobs[tag].get("n_max"),
            "n_steps": jobs[tag].get("n_steps"),
            "max_top_fock": jobs[tag].get("max_top_fock"),
            "max_n_ph": jobs[tag].get("max_n_ph"),
            "end_n_ph": jobs[tag].get("end_n_ph"),
            "max_abs_norm_minus_1": jobs[tag].get("max_abs_norm_minus_1"),
            "t_reached_over_T0": jobs[tag].get("t_reached_over_T0"),
            "seconds_per_step": jobs[tag].get("seconds_per_step"),
            "used_pf_OMEGA0": jobs[tag].get("used_pf_OMEGA0"),
            "fock_alarm_0p05": jobs[tag].get("fock_alarm_0p05"),
        } for tag in tags},
    }
    smoke = jobs.get("smoke_n8_p_q") or {}
    evolve = jobs.get("n8_p_q") or {}
    rec["smoke_omega_ok"] = bool(
        smoke.get("omega0") == OMEGA0_1600 and smoke.get("used_pf_OMEGA0") is False
        and smoke.get("status") in ("ok", "ok_fock_alarm"))
    rec["n8_complete"] = bool(evolve.get("status") in ("ok", "ok_fock_alarm"))
    rec["n8_truncation_pass"] = bool(
        evolve.get("status") == "ok" and evolve.get("fock_alarm_0p05") is False)
    rec["n8_truncation_fail"] = bool(evolve.get("status") == "ok_fock_alarm")
    if rec["n8_truncation_pass"]:
        rec["verdict"] = "n8_fock_truncation_pass_theta_pi4_only"
        rec["next"] = "exact QED four-angle + pi/4 still NOT authorized until this pass is frozen"
    elif rec["n8_truncation_fail"]:
        rec["verdict"] = "n8_fock_truncation_fail_do_not_extract_ABC"
        rec["next"] = "n10_p_q comparison; do not retune g"
    elif evolve.get("status") in ("timeout_partial", "running", "incomplete"):
        rec["verdict"] = "n8_p_q_incomplete_resume"
        rec["next"] = "resume n8_p_q; do not open four-angle QED"
    elif rec["smoke_omega_ok"]:
        rec["verdict"] = "smoke_omega_ok_evolve_not_done"
        rec["next"] = "run --mode evolve n8_p_q"
    else:
        rec["verdict"] = "audit_incomplete"
        rec["next"] = "clarify/input/smoke before claiming anything"
    dump(OUT / "fockaudit.json", rec)
    progress(f"analyze verdict={rec['verdict']} n8_pass={rec['n8_truncation_pass']}")
    return rec


def drive(mode: str, n_steps: int | None) -> int:
    hashes = source_hashes()
    if hash_abort(hashes):
        dump(OUT / "fockaudit.json", {"status": "hash_abort", "stopped": True})
        progress("hash abort, no propagation")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "clarify":
        rec = clarify_weyl()
        return 0 if rec.get("status") == "clarify" else 3
    if mode == "input":
        rec = input_audit()
        return 0 if rec.get("status") == "input" else 3
    if mode == "smoke":
        clarify_weyl()
        input_audit()
        launch = launch_tags(["smoke_n8_p_q"], n_steps)
        progress(f"smoke return {launch['returncodes']}")
        rec = analyze(["smoke_n8_p_q"])
        return 0 if rec.get("smoke_omega_ok") else 1
    if mode == "evolve":
        smoke = load_job("smoke_n8_p_q")
        if not smoke or smoke.get("omega0") != OMEGA0_1600:
            progress("refuse evolve: smoke_n8_p_q did not certify omega0=0.775")
            dump(OUT / "fockaudit.json", {
                "status": "evolve_blocked_no_smoke_omega", "stopped": True,
            })
            return 3
        launch = launch_tags(["n8_p_q"], n_steps)
        progress(f"evolve return {launch['returncodes']}")
        rec = analyze(["smoke_n8_p_q", "n8_p_q"])
        if rec.get("n8_complete"):
            return 0
        return 1
    return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("clarify", "input", "smoke", "evolve"), default="")
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--n-steps", type=int, default=0)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    n_steps = int(args.n_steps) if args.n_steps else None
    if args.job:
        rec = run_one_job(args.job, n_steps)
        return 0 if rec.get("status") in ("ok", "ok_fock_alarm", "timeout_partial") else 1
    if args.analyze_only:
        rec = analyze()
        print(json.dumps(jsonable({
            "status": rec.get("status"), "verdict": rec.get("verdict"),
            "jobs": rec.get("jobs"),
        }), ensure_ascii=False, indent=2))
        return 0
    if args.mode:
        return drive(args.mode, n_steps)
    rc = drive("clarify", None)
    if rc:
        return rc
    rc = drive("input", None)
    if rc:
        return rc
    rc = drive("smoke", n_steps)
    if rc:
        return rc
    return drive("evolve", n_steps)


if __name__ == "__main__":
    raise SystemExit(main())
