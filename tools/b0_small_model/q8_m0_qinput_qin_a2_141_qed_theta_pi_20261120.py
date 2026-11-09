#!/usr/bin/env python3
"""Qin_a2_1_4_1 two-way exact QED: DS(theta=pi), N=+/-1.

Frozen read-only: Q1a theta=0 four-way QED, GH5 theta=0, GH5 theta=pi.
Does not rerun those archives. Does not start Cn, SV, G1/H2, extra k,
nmax scan, D2/Floquet, or server. No automatic second 900 s wave.

Preflight checks nmax=8 truncation of D(alpha)S(xi)|0> before propagate.
Integer Fourier is not certified HHG. Same nbar is not full nonclassicality.
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
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_qin_a2_141_qed_theta_pi_20261120"
FIG = OUT / "figures"
Q1A_DIR = ROOT / "results/q_line/qinput_qin_a2_141_q1a_20261120"
GH5_DIR = ROOT / "results/q_line/qinput_qin_a2_141_gh5_theta_pi_20261120"
DOCS = ROOT / "docs/04_BSV量子光"
TASK_TAG = "20261120"
RECORD_DATE = "2026-10-22"
PROTOCOL = "Qin_a2_1_4_1"
MU = -0.009010620372198276
NMAX = odd.NMAX
NMAX_REF = 80
ALPHA = odd.ALPHA
R_SQ = odd.R_SQ
THETA_0 = 0.0
THETA_PI = float(np.pi)
N_FULL = 11154
SAMPLE_EVERY = 32
CHECKPOINT_EVERY = 400
DRIVER_WALL_S = 900.0
JOB_WALL_S = 870.0
FOCK_ALARM = 0.05
CLASSICAL_A_STAR = 0.8
REL_MARK = 0.20
DCOMPLEX_MARK = 0.50
QED_GH5_REL = 0.20
QED_GH5_DPHI_DEG = 20.0
JOBS = (
    ("ds_p", "ds", 1),
    ("ds_m", "ds", -1),
)
JOB_MAP = {tag: (kind, neel) for tag, kind, neel in JOBS}
EXPECTED_HASHES = {
    "q8_m0_qinput_dynamics_20261026.py":
        "bf6d4e407bf55f1e6ebc3767f1b2bd6075ac7ef564f195abc1217c7d51671f94",
    "q8_m0_qinput_minimal_20261020.py":
        "b60c3b7e53e413e614d4a8ec784956f4d16001a906465314dea963a17ae9bc97",
    "q8_m0_qinput_shared_kk_static_20261029.py":
        "872347c94656ef74d2d0e3cce961e8fb1f1b47d30610bf0bb534b9ad80925345",
    "q8_m0_qinput_shared_kk_dyn_preflight_20261030.py":
        "1640a0598b9731cd9468b7156ff632401f245438af8fc8d0f8e13cd726d23540",
    "q8_m0_qinput_shared_kk_ds_odd_20261102.py":
        "cfc6d8c5dee162ab53e5d2c0344fc3b35b03cd77621fbfe8eb963069ecb7c41c",
    "q8_m0_model.py":
        "5d53090cabb60238c6c16a74b92ddc061d5adf2d8081e48b46b9bdf0089261d3",
}
FROZEN = tuple(HERE / name for name in EXPECTED_HASHES)


def jsonable(x):
    return odd.jsonable(x)


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


def cz(pack) -> complex:
    return complex(pack["re"], pack["im"])


def pack_c(z):
    z = np.asarray(z, complex)
    if z.ndim == 0:
        return pf.cpack(complex(z))
    return [pf.cpack(complex(v)) for v in z]


def wrap(phi: float) -> float:
    return float((phi + np.pi) % (2.0 * np.pi) - np.pi)


def angdiff(a: float, b: float) -> float:
    return wrap(float(a) - float(b))


def a_star_mean(geo) -> float:
    return 2.0 * float(geo.a_nn_mean_ang) * float(pf.G_ABS) * abs(complex(ALPHA))


def fock_xp_moments(c):
    """X=a+a†, P=-i(a-a†). Vacuum Var(X)=1. mean_X = 2 Re <a>."""
    c = np.asarray(c, complex)
    mass = float(np.vdot(c, c).real)
    cn = c / np.sqrt(mass)
    n = np.arange(len(cn), dtype=float)
    mean_n = float(np.vdot(cn, n * cn).real)
    mean_a = 0.0j
    mean_aa = 0.0j
    for i in range(len(cn) - 1):
        mean_a += np.conj(cn[i]) * cn[i + 1] * np.sqrt(i + 1)
    for i in range(len(cn) - 2):
        mean_aa += np.conj(cn[i]) * cn[i + 2] * np.sqrt((i + 1) * (i + 2))
    mean_X = 2.0 * mean_a.real
    mean_P = 2.0 * mean_a.imag
    mean_X2 = 2.0 * mean_aa.real + 2.0 * mean_n + 1.0
    mean_P2 = -2.0 * mean_aa.real + 2.0 * mean_n + 1.0
    p = np.abs(cn) ** 2
    return {
        "unnormalized_mass": mass,
        "mean_a": pf.cpack(mean_a),
        "mean_n": mean_n,
        "mean_X": float(mean_X),
        "mean_P": float(mean_P),
        "var_X": float(mean_X2 - mean_X ** 2),
        "var_P": float(mean_P2 - mean_P ** 2),
        "p_n": [float(x) for x in p],
        "p_top": float(p[-1]) if len(p) else 0.0,
    }


def analytic_xp(theta: float) -> dict:
    a = complex(ALPHA)
    r = float(R_SQ)
    th = float(theta)
    mx, mp = 2.0 * a.real, 2.0 * a.imag
    if abs(th) < 1e-15:
        vx, vp = float(np.exp(-2.0 * r)), float(np.exp(2.0 * r))
        axis = "X_squeezed"
    elif abs(th - np.pi) < 1e-12:
        vx, vp = float(np.exp(2.0 * r)), float(np.exp(-2.0 * r))
        axis = "P_squeezed"
    else:
        raise ValueError("theta must be 0 or pi")
    nbar = abs(a) ** 2 + np.sinh(r) ** 2
    return {
        "alpha": pf.cpack(a), "r": r, "theta": th,
        "mean_X": mx, "mean_P": mp, "var_X": vx, "var_P": vp,
        "mean_n_analytic": float(nbar), "squeeze_axis": axis,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
        "free_field_Var_X_t": "cosh(2r)-sinh(2r) cos(2 omega0 t - theta)",
        "Var_X_t0_is_not_constant_along_the_pulse": True,
    }


def pn_ref(c_ref, ncut: int):
    p = np.abs(np.asarray(c_ref, complex)) ** 2
    p = p / p.sum()
    p_ncut = float(p[ncut]) if ncut < len(p) else 0.0
    p_ge = float(p[ncut + 1:].sum()) if ncut + 1 < len(p) else 0.0
    return p_ncut, p_ge


def theta_fock_row(theta: float) -> dict:
    raw8 = odd.photon_raw("ds", NMAX, float(theta))
    raw80 = odd.photon_raw("ds", NMAX_REF, float(theta))
    m8 = fock_xp_moments(raw8)
    m80 = fock_xp_moments(raw80)
    ana = analytic_xp(theta)
    p8, pge9 = pn_ref(raw80, NMAX)
    return {
        "theta": float(theta),
        "analytic": ana,
        "nmax8_used_by_propagator": m8,
        "nmax80_ideal": m80,
        "P_n8_ideal80": p8,
        "P_n_ge9_ideal80": pge9,
        "P_n8_trunc8_normalized": m8["p_top"],
        "d_mean_X_8_vs_1": abs(m8["mean_X"] - 1.0),
        "d_mean_X_8_vs_ideal": abs(m8["mean_X"] - ana["mean_X"]),
        "d_mean_n_8_vs_ideal": abs(m8["mean_n"] - ana["mean_n_analytic"]),
        "rel_var_X_8_vs_ideal": abs(m8["var_X"] / ana["var_X"] - 1.0),
        "d_mean_P_8_vs_0": abs(m8["mean_P"] - 0.0),
    }


def preflight() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "preflight", "source_hashes": hashes, "hash_abort": bad,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
        "n_max_propagator": NMAX, "n_max_ref": NMAX_REF,
        "did_not_raise_nmax": True, "qed_not_started": True,
        "P_n_ge9_is_not_abort_by_itself": True,
        "no_fock_alarm_is_not_1e-8_error_bar": True,
        "stopped": False,
    }
    if bad:
        rec["status"] = "hash_abort"
        rec["stopped"] = True
        dump(OUT / "preflight.json", rec)
        return rec
    row0 = theta_fock_row(THETA_0)
    rowp = theta_fock_row(THETA_PI)
    rec["theta0"] = row0
    rec["theta_pi"] = rowp
    m0, mp = row0["nmax8_used_by_propagator"], rowp["nmax8_used_by_propagator"]
    rec["compare_nmax8"] = {
        "d_mean_X_pi_minus_0": abs(mp["mean_X"] - m0["mean_X"]),
        "rel_d_mean_n_pi_minus_0": abs(mp["mean_n"] - m0["mean_n"]) / max(abs(m0["mean_n"]), 1e-30),
        "mean_X_pi": mp["mean_X"],
        "mean_n_pi": mp["mean_n"],
        "var_X_pi": mp["var_X"],
        "var_P_pi": mp["var_P"],
        "P_n_ge9_ratio_pi_over_0": (
            rowp["P_n_ge9_ideal80"] / max(row0["P_n_ge9_ideal80"], 1e-30)
        ),
    }
    c = rec["compare_nmax8"]
    abort_reasons = []
    if c["d_mean_X_pi_minus_0"] > 0.02:
        abort_reasons.append("mean_X_pi8_vs_08")
    if c["rel_d_mean_n_pi_minus_0"] > 0.02:
        abort_reasons.append("mean_n_pi8_vs_08")
    if abs(mp["mean_X"] - 1.0) > 0.02:
        abort_reasons.append("mean_X_pi8_vs_1")
    if rowp["rel_var_X_8_vs_ideal"] > 0.10:
        abort_reasons.append("var_X_pi8_vs_ideal")
    rec["abort_reasons"] = abort_reasons
    rec["same_resource_ok"] = not abort_reasons
    rec["external_review_P_n8"] = {"theta0": 1.84e-5, "theta_pi": 1.10e-4}
    rec["external_review_P_n_ge9"] = {"theta0": 2.63e-6, "theta_pi": 5.40e-5}
    rec["note_tail"] = (
        "theta=pi high-Fock tail is typically ~20x theta=0; "
        "P(n>=9)~5e-5 is not an abort. No alarm is not a 1e-8 bound."
    )
    if abort_reasons:
        rec["status"] = "preflight_abort"
        rec["stopped"] = True
        rec["qed_not_started"] = True
    dump(OUT / "preflight.json", rec)
    progress(
        f"preflight ok={rec['same_resource_ok']} "
        f"meanX_pi={mp['mean_X']:.6f} nbar_pi={mp['mean_n']:.6f} "
        f"varX_pi={mp['var_X']:.4f} Pge9_pi={rowp['P_n_ge9_ideal80']:.3e} "
        f"abort={abort_reasons}"
    )
    return rec


def ckpt_paths(tag: str):
    return OUT / f"ckpt_{tag}.npz", OUT / f"ckpt_{tag}_samples.json"


def save_ckpt(tag: str, psi, next_j: int, samples) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p_npz, p_js = ckpt_paths(tag)
    np.savez_compressed(p_npz, psi=np.asarray(psi, complex), next_j=np.int64(next_j))
    p_js.write_text(json.dumps(jsonable(samples), ensure_ascii=False), encoding="utf-8")


def load_ckpt(tag: str):
    p_npz, p_js = ckpt_paths(tag)
    if not p_npz.is_file() or not p_js.is_file():
        return None
    z = np.load(p_npz)
    samples = json.loads(p_js.read_text(encoding="utf-8"))
    return np.asarray(z["psi"], complex), int(z["next_j"]), samples


def dedup_samples_by_step(samples):
    out, seen = [], set()
    for rec in list(samples or []):
        st = int(rec.get("step", -1))
        if st in seen:
            continue
        seen.add(st)
        out.append(rec)
    return out


def load_job(tag: str):
    path = OUT / f"job_{tag}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sample_js_for(n_steps: int):
    js = list(range(0, n_steps, SAMPLE_EVERY))
    if not js or js[-1] != n_steps - 1:
        js.append(n_steps - 1)
    return js


def propagate_ckpt(model, psi, t0, t1, env, n_steps, sample_js, clock, tag,
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
                save_ckpt(tag, psi, j, samples)
                progress(f"{tag} ckpt j={j}/{n_steps} wall={clock.elapsed():.1f}s")
    except TimeoutError as exc:
        timed_out = True
        timeout_msg = str(exc)
        save_ckpt(tag, psi, j, samples)
    work = 0.0
    if len(samples) >= 2:
        for a, b in zip(samples[:-1], samples[1:]):
            work += 0.5 * (a["power_dHds_dsdt"] + b["power_dHds_dsdt"]) * (b["t"] - a["t"])
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
        "work_trap_dHds_dsdt": float(work),
        "timed_out": bool(timed_out),
        "timeout_msg": timeout_msg,
        "next_j": int(j),
        "loop_seconds": float(time.perf_counter() - t_loop0),
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
    return t, j, n_ph, top, nxt, nrm, ss


def save_series_npz(tag: str, samples) -> None:
    t, j, n_ph, top, nxt, nrm, ss = series_from_samples(samples)
    np.savez_compressed(
        OUT / f"series_{tag}.npz",
        t=t, Jx=j[:, 0], Jy=j[:, 1], n_ph=n_ph, top_fock=top,
        next_top_fock=nxt, norm=nrm, s=ss)


def run_one_job(tag: str) -> dict:
    if tag not in JOB_MAP:
        raise ValueError(tag)
    kind, neel = JOB_MAP[tag]
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        report = {
            "tag": tag, "status": "hash_abort", "hash_mismatch": bad,
            "source_hashes": hashes, "stopped": True,
        }
        dump(OUT / f"job_{tag}.json", report)
        return report

    prev = load_job(tag)
    if prev and prev.get("status") in ("ok", "ok_fock_alarm"):
        progress(f"{tag} already {prev['status']}, skip")
        return prev
    if prev and prev.get("status") == "timeout_partial":
        progress(f"{tag} timeout_partial, this round does not auto-resume a second wave")
        return prev

    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=4.0)
    n_steps = N_FULL
    dt = env.t_total / n_steps
    ek, emk, masses = odd.ground_pair_neel(geo, k, neel)
    raw = odd.photon_raw("ds", NMAX, THETA_PI)
    raw_diag = odd.photon_moments(raw)
    xp8 = fock_xp_moments(raw)
    c = raw / np.sqrt(raw_diag["unnormalized_mass"])
    model = sh.build_shared(geo, R0, k, neel, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    psi, prod_n = odd.product_normalized(ek, emk, c)

    start_j = 0
    samples = []
    resumed = False
    want_ckpt = prev is None or prev.get("status") in ("running", "failed", "incomplete")
    if want_ckpt:
        loaded = load_ckpt(tag)
        if loaded is not None:
            psi, start_j, samples = loaded
            samples = dedup_samples_by_step(samples)
            resumed = True
            progress(f"{tag} resume next_j={start_j} n_samples={len(samples)}")

    dump(OUT / f"job_{tag}.json", {
        "tag": tag, "kind": kind, "neel": int(neel), "status": "running",
        "theta": THETA_PI, "resumed": resumed,
        "next_j": start_j, "utc": datetime.now(timezone.utc).isoformat(),
    })
    resume_obs = pf.snapshot(model, psi, env.s(start_j * dt))
    if resumed and prev and prev.get("initial"):
        s0 = prev["initial"]
    else:
        s0 = resume_obs
    progress(f"{tag} start ds theta=pi neel={neel} dim={model.dim} j0={start_j}/{n_steps}")
    clock = dyn.WallClock(JOB_WALL_S)
    finite_ok = True
    err = None
    try:
        psi, led = propagate_ckpt(
            model, psi, 0.0, env.t_total, env, n_steps, sample_js_for(n_steps),
            clock, tag, start_j=start_j, samples=samples)
    except Exception as exc:
        finite_ok = False
        err = f"{type(exc).__name__}: {exc}"
        led = {
            "steps_requested": n_steps, "steps_completed": start_j, "dt": float(dt),
            "samples": samples, "timed_out": False, "timeout_msg": None,
            "max_last_taylor_term": None, "work_trap_dHds_dsdt": None,
            "next_j": start_j, "t1_reached": start_j * dt, "loop_seconds": clock.elapsed(),
        }
        save_ckpt(tag, psi, start_j, samples)
        progress(f"{tag} exception {err}")

    wall = clock.elapsed()
    samples = led.get("samples") or []
    if samples:
        save_series_npz(tag, samples)
        save_ckpt(tag, psi, led.get("next_j", start_j), samples)
    send = samples[-1] if samples else pf.snapshot(model, psi, env.s(led.get("t1_reached", 0.0)))
    tops = [s["top_fock_mass"] for s in samples] + [send["top_fock_mass"]]
    nxts = [s["p_n"][-2] for s in samples] + [send["p_n"][-2]]
    norms = [s["norm"] for s in samples] + [send["norm"]]
    nphs = [s["n_ph"] for s in samples] + [send["n_ph"]]
    max_top = float(np.max(tops))
    max_next = float(np.max(nxts))
    max_dn = float(np.max(np.abs(np.asarray(norms) - 1.0)))
    fock_alarm = bool(max_top > FOCK_ALARM)
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
        "task_tag": TASK_TAG,
        "record_date": RECORD_DATE,
        "protocol": PROTOCOL,
        "tag": tag,
        "kind": kind,
        "neel": int(neel),
        "status": status,
        "resumed": resumed,
        "resume_from_next_j": int(start_j) if resumed else 0,
        "finite_state": finite_ok,
        "error": err,
        "fock_alarm_0p05": fock_alarm,
        "source_hashes": hashes,
        "definition": {
            "n_max": NMAX, "g": pf.G_ABS, "omega0": pf.OMEGA0,
            "alpha": {"re": float(ALPHA.real), "im": float(ALPHA.imag)},
            "r": R_SQ, "theta": THETA_PI,
            "operator_order": "D(alpha)S(xi)|0>",
            "k_fractional": list(sh.K_FRAC),
            "mu_half_filling_documented": MU,
            "n_full": n_steps, "dt": float(dt),
            "sample_every": SAMPLE_EVERY,
            "period": env.period, "t_total": env.t_total,
            "plateau": list(env.plateau),
            "A_star_mean": a_star_mean(geo),
            "classical_A_star": CLASSICAL_A_STAR,
            "s_is_coupling_switch_not_free_space_pulse": True,
        },
        "electron": masses,
        "photon_unnormalized": raw_diag,
        "photon_xp_nmax8": xp8,
        "product_norm_before_renorm": prod_n,
        "initial": s0,
        "endpoint": send,
        "ledger": {k: led[k] for k in led if k != "samples"},
        "max_top_fock": max_top,
        "max_next_top_fock": max_next,
        "max_n_ph": float(np.max(nphs)),
        "min_n_ph": float(np.min(nphs)),
        "max_abs_norm_minus_1": max_dn,
        "t_reached": t_reach,
        "t_reached_over_T0": t_over,
        "covers_5T0": bool(t_over + 1e-9 >= 5.0),
        "covers_4T0": bool(t_over + 1e-9 >= 4.0),
        "covers_6T0": bool(t_over + 1e-9 >= 6.0),
        "wave": 1,
        "wall_seconds": wall,
        "seconds_per_step": (wall / max(completed - start_j, 1)),
        "time_series": pf.pack_samples(samples) if samples else {},
        "n_samples": len(samples),
        "series_npz": str((OUT / f"series_{tag}.npz").as_posix()),
        "integer_Fourier_is_not_certified_HHG": True,
        "no_fock_alarm_is_not_1e-8_error_bar": True,
        "did_not_auto_raise_nmax": True,
    }
    if resumed:
        report["resume_snapshot"] = resume_obs
    dump(OUT / f"job_{tag}.json", report)
    progress(f"{tag} {status} steps={completed}/{n_steps} t/T0={t_over:.3f} "
             f"top={max_top:.3e} wall={wall:.1f}s")
    return report


def window_mask(t, t0, t1):
    return (t >= t0 - 1e-12) & (t <= t1 + 1e-12)


def rect_jq(t, j, omega0, t0, t1, qs):
    t = np.asarray(t, float)
    j = np.asarray(j, complex)
    m = window_mask(t, t0, t1)
    tw, jw = t[m], j[m]
    span = float(t1 - t0)
    out = []
    for q in qs:
        ker = np.exp(1j * float(q) * omega0 * tw)
        slot = np.array([np.trapezoid(ker * jw[:, a], tw) for a in range(j.shape[1])])
        out.append(slot / span)
    return np.asarray(out, complex)


def load_series(tag: str, directory: Path | None = None):
    directory = OUT if directory is None else directory
    path = directory / f"series_{tag}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    z = np.load(path)
    t = np.asarray(z["t"], float)
    j = np.stack([z["Jx"], z["Jy"]], axis=1)
    return t, j


def slot_pack(jq):
    return {
        "Jx": pack_c(jq[:, 0]),
        "Jy": pack_c(jq[:, 1]),
        "abs": [float(np.linalg.norm(v)) for v in jq],
        "phase_x": [float(np.angle(v[0])) for v in jq],
        "phase_y": [float(np.angle(v[1])) for v in jq],
    }


def pack_win(name, t, omega0, T0, w0, w1, arrs, complete: bool):
    slots = {key: slot_pack(rect_jq(t, arr, omega0, w0, w1, np.arange(1, 7, dtype=float)))
             for key, arr in arrs.items()}
    return {
        "name": name, "t0": w0, "t1": w1,
        "t0_over_T0": w0 / T0, "t1_over_T0": w1 / T0,
        "complete": bool(complete),
        "convention": "J_q = 1/(t1-t0) int exp(+i q omega0 t) J(t) dt, rectangular",
        "not_certified_HHG": True,
        "slots": slots,
    }


def h2_jx(slots, key):
    jx = cz(slots[key]["Jx"][1])
    jy = cz(slots[key]["Jy"][1])
    return jx, jy, float(abs(jx)), float(np.angle(jx)), float(slots[key]["abs"][1])


def svg_axes(x0, y0, w, h, xlabel, ylabel, title):
    return (
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fffdf8" '
        f'stroke="#2c3338" stroke-width="1"/>'
        f'<text x="{x0 + w/2:.1f}" y="{y0 - 8:.1f}" text-anchor="middle" '
        f'font-size="13" fill="#1b242c">{title}</text>'
        f'<text x="{x0 + w/2:.1f}" y="{y0 + h + 22:.1f}" text-anchor="middle" '
        f'font-size="11" fill="#243038">{xlabel}</text>'
        f'<text x="{x0 - 28:.1f}" y="{y0 + h/2:.1f}" text-anchor="middle" '
        f'font-size="11" fill="#243038" transform="rotate(-90 {x0-28:.1f} {y0+h/2:.1f})">{ylabel}</text>'
    )


def write_svg(path: Path, body: str, width=920, height=560) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="#f4efe6"/>{body}</svg>\n'
    )
    path.write_text(svg, encoding="utf-8")
    return str(path.as_posix())


def map_xy(z, lim, x0, y0, w, h):
    px = x0 + (z.real / lim * 0.5 + 0.5) * w
    py = y0 + (0.5 - z.imag / lim * 0.5) * h
    return px, py


def arrow(z, lim, x0, y0, w, h, color, label, lx, ly, dash=False):
    cx, cy = x0 + w / 2, y0 + h / 2
    px, py = map_xy(z, lim, x0, y0, w, h)
    dash_attr = ' stroke-dasharray="4 3"' if dash else ""
    return (
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
        f'stroke="{color}" stroke-width="2"{dash_attr}/>'
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>'
        f'<text x="{lx}" y="{ly}" font-size="11" fill="{color}">{label}</text>'
    )


def figure_h2_plane(z0, zpi, zd, zgh0, zghpi, zghd, path):
    zs = [z0, zpi, zd, zgh0, zghpi, zghd]
    lim = max(abs(z) for z in zs) * 1.25 + 1e-30
    x0, y0, w, h = 90, 40, 740, 430
    body = svg_axes(x0, y0, w, h, "Re Jx (H2 odd)", "Im Jx (H2 odd)",
                    "H2 odd Jx: QED θ=0 vs θ=π (GH5 overlay dashed)")
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    body += arrow(z0, lim, x0, y0, w, h, "#4c6b8a", "QED θ=0", 80, 22)
    body += arrow(zpi, lim, x0, y0, w, h, "#c45c26", "QED θ=π", 170, 22)
    body += arrow(zd, lim, x0, y0, w, h, "#2f9e44", "QED δθ", 270, 22)
    body += arrow(zgh0, lim, x0, y0, w, h, "#7a93aa", "GH5 θ=0", 80, 38, dash=True)
    body += arrow(zghpi, lim, x0, y0, w, h, "#d08a62", "GH5 θ=π", 170, 38, dash=True)
    body += arrow(zghd, lim, x0, y0, w, h, "#6fbf7a", "GH5 δθ", 270, 38, dash=True)
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "same alpha, same nbar; integer Fourier not certified HHG; GH5 overlay is prediction</text>"
    )
    return write_svg(path, body)


def figure_windows(rows, path):
    lim = max(abs(z) for _, z, _ in rows) * 1.25 + 1e-30
    x0, y0, w, h = 90, 40, 740, 430
    body = svg_axes(x0, y0, w, h, "Re δθ Jx (H2 odd)", "Im δθ Jx (H2 odd)",
                    "QED H2 δθ three windows (Jx)")
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    lx = 80
    for label, z, color in rows:
        body += arrow(z, lim, x0, y0, w, h, color, label, lx, 36)
        lx += 150
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "delta_complex uses Jx. Magnitude stability is not complex-amplitude convergence.</text>"
    )
    return write_svg(path, body)


def figure_cont(q, y0, ypi, yd, path):
    xmin, xmax = float(q[0]), float(q[-1])
    ys = np.concatenate([y0, ypi, yd])
    ymin, ymax = 0.0, float(np.max(ys)) * 1.12 + 1e-30
    x0, y0p, w, h = 70, 40, 800, 430
    body = svg_axes(x0, y0p, w, h, "Ω / ω0", "|J^o(Ω)|",
                    "QED main-window continuum |J_odd|")

    def poly(xs, ys_, color, sw):
        pts = []
        for x, y in zip(xs, ys_):
            px = x0 + (float(x) - xmin) / (xmax - xmin) * w
            py = y0p + (1.0 - (float(y) - ymin) / max(ymax - ymin, 1e-30)) * h
            pts.append(f"{px:.2f},{py:.2f}")
        return (f'<polyline fill="none" stroke="{color}" stroke-width="{sw}" '
                f'points="{" ".join(pts)}"/>')

    body += poly(q, y0, "#4c6b8a", 1.4)
    body += poly(q, ypi, "#c45c26", 1.4)
    body += poly(q, yd, "#2f9e44", 1.4)
    for n in range(1, 7):
        px = x0 + (n - xmin) / (xmax - xmin) * w
        body += (f'<line x1="{px:.1f}" y1="{y0p}" x2="{px:.1f}" y2="{y0p+h}" '
                 f'stroke="#8a9096" stroke-dasharray="3 4" stroke-width="0.8"/>')
    body += '<text x="80" y="36" font-size="11" fill="#4c6b8a">QED θ=0</text>'
    body += '<text x="160" y="36" font-size="11" fill="#c45c26">QED θ=π</text>'
    body += '<text x="250" y="36" font-size="11" fill="#2f9e44">δθ</text>'
    return write_svg(path, body)


def compact_job(rec):
    if rec is None:
        return None
    led = rec.get("ledger") or {}
    return {
        "status": rec.get("status"),
        "t_reached_over_T0": rec.get("t_reached_over_T0"),
        "steps_completed": led.get("steps_completed"),
        "max_top_fock": rec.get("max_top_fock"),
        "max_next_top_fock": rec.get("max_next_top_fock"),
        "max_abs_norm_minus_1": rec.get("max_abs_norm_minus_1"),
        "max_last_taylor_term": led.get("max_last_taylor_term"),
        "max_n_ph": rec.get("max_n_ph"),
        "fock_alarm_0p05": rec.get("fock_alarm_0p05"),
        "wall_seconds": rec.get("wall_seconds"),
        "seconds_per_step": rec.get("seconds_per_step"),
        "covers_4T0": rec.get("covers_4T0"),
        "covers_5T0": rec.get("covers_5T0"),
        "covers_6T0": rec.get("covers_6T0"),
        "finite_state": rec.get("finite_state"),
        "error": rec.get("error"),
        "n_samples": rec.get("n_samples"),
    }


def load_gh5_h2():
    path = GH5_DIR / "gh5_theta_pi.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    h2 = d["h2_main"]
    return {
        "jx_0": cz(h2["jx_0"]),
        "jx_pi": cz(h2["jx_pi"]),
        "jx_delta": cz(h2["jx_delta"]),
        "abs_jx_0": h2["abs_jx_0"],
        "abs_jx_pi": h2["abs_jx_pi"],
        "abs_delta_jx": h2["abs_delta_jx"],
        "abs_2d_0": h2["abs_2d_0"],
        "abs_2d_pi": h2["abs_2d_pi"],
        "phase_0": h2["phase_0"],
        "phase_pi": h2["phase_pi"],
        "delta_complex": h2["delta_complex"],
        "verdict": d.get("verdict"),
        "peak_q_dJ": d.get("continuum", {}).get("peak_q_dJ"),
    }


def write_reports(rec: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    h2 = rec.get("h2_main") or {}
    pf0 = rec.get("preflight") or {}
    v = rec.get("verdict")
    ans = rec.get("answer")
    cov = rec.get("coverage") or {}
    cmpg = rec.get("qed_vs_gh5") or {}
    full = []
    full.append("# 量子光多周期磁序响应：θ=π 精确 QED 验证\n")
    full.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。`stopped = true`。\n")
    full.append("冻结只读：θ=0 四路 QED、GH5 θ=0、GH5 θ=π。本轮仅新增 DS(α=0.5,r=0.3,θ=π) 的 N=±1。\n")
    full.append(f"**判定：** `{v}`\n")
    full.append(f"{ans}\n")
    full.append("## 一、nmax=8 预检\n")
    if pf0:
        t0 = pf0.get("theta0") or {}
        tp = pf0.get("theta_pi") or {}
        c = pf0.get("compare_nmax8") or {}
        full.append("| 量 | θ=0 nmax=8 | θ=π nmax=8 |")
        full.append("|---|---|---|")
        m0 = t0.get("nmax8_used_by_propagator") or {}
        mp = tp.get("nmax8_used_by_propagator") or {}
        full.append(f"| ⟨X⟩ | {m0.get('mean_X')} | {mp.get('mean_X')} |")
        full.append(f"| ⟨P⟩ | {m0.get('mean_P')} | {mp.get('mean_P')} |")
        full.append(f"| ⟨n⟩ | {m0.get('mean_n')} | {mp.get('mean_n')} |")
        full.append(f"| Var(X) | {m0.get('var_X')} | {mp.get('var_X')} |")
        full.append(f"| Var(P) | {m0.get('var_P')} | {mp.get('var_P')} |")
        full.append(f"| P(n=8) ideal80 | {t0.get('P_n8_ideal80')} | {tp.get('P_n8_ideal80')} |")
        full.append(f"| P(n≥9) ideal80 | {t0.get('P_n_ge9_ideal80')} | {tp.get('P_n_ge9_ideal80')} |")
        full.append(f"\nP(n≥9) 比值约 {c.get('P_n_ge9_ratio_pi_over_0')}。"
                    "该尾部本身不是中止理由。未触发 0.05 警报 ≠ 1e-8 奇电流有截断误差界。\n")
        full.append(f"同资源比较：`same_resource_ok={pf0.get('same_resource_ok')}`；"
                    f"abort={pf0.get('abort_reasons')}\n")
    full.append("## 二、覆盖与数值监测\n")
    full.append(f"- t_min/T0 = {cov.get('t_min_over_T0')}")
    full.append(f"- covers 4/5/6 T0 = {cov.get('covers_4T0')}/{cov.get('covers_5T0')}/{cov.get('covers_6T0')}")
    full.append(f"- job_status = {cov.get('job_status')}")
    full.append(f"- max_top_fock = {cov.get('max_top_fock')}")
    full.append(f"- max_abs_norm_minus_1 = {cov.get('max_abs_norm_minus_1')}\n")
    if h2:
        full.append("## 三、主窗 H2 磁序奇复电流\n")
        full.append("| 量 | QED |")
        full.append("|---|---|")
        full.append(f"| \\(\\lvert\\mathcal J_{{2,x}}^{{o}}(\\theta=0)\\rvert\\) | {h2.get('abs_jx_0')} |")
        full.append(f"| \\(\\lvert\\mathcal J_{{2,x}}^{{o}}(\\theta=\\pi)\\rvert\\) | {h2.get('abs_jx_pi')} |")
        full.append(f"| 振幅比 π/0 | {h2.get('amp_ratio_pi_over_0')} |")
        full.append(f"| Δφ (rad) | {h2.get('dphi_pi_minus_0_rad')} |")
        full.append(f"| \\(\\delta_\\theta\\mathcal J_{{2,x}}^o\\) | {h2.get('delta_jx_re')}+i{h2.get('delta_jx_im')} |")
        full.append(f"| \\(\\lvert\\delta_\\theta J_{{2,x}}\\rvert\\) | {h2.get('abs_delta_jx')} |")
        full.append(f"| rel_to_max | {h2.get('rel_to_max')} |")
        full.append(f"| δ_complex | {h2.get('delta_complex')} |\n")
        full.append("模长稳定 ≠ 复振幅已收敛。两周期窗频率分辨约 0.5 ω₀，整数投影不是已认证 HHG。\n")
    if cmpg:
        full.append("## 四、与 GH5 对照\n")
        full.append("| 量 | 值 |")
        full.append("|---|---|")
        full.append(f"| \\(\\lvert\\delta_{{\\mathrm{{QED}}}}-\\delta_{{\\mathrm{{GH5}}}}\\rvert/\\lvert\\delta_{{\\mathrm{{QED}}}}\\rvert\\) | {cmpg.get('rel_vec')} |")
        full.append(f"| 相位偏差 (deg) | {cmpg.get('dphi_deg')} |")
        full.append(f"| 模长接近且相位符合 | {cmpg.get('agrees_complex')} |\n")
        full.append("不能仅凭差向量模长接近就宣称符合。\n")
    full.append("## 五、物理解释边界\n")
    full.append("- 自由光场 \\(\\mathrm{Var}[X(t)]=\\cosh(2r)-\\sinh(2r)\\cos(2\\omega_0 t-\\theta)\\)。")
    full.append("  不要把 Var(X,t=0) 当成全程不变的方差。")
    full.append("- 同位移、同 ⟨n⟩ 隔离的是压缩角，不是全部非经典性。")
    full.append("- QED 电流变化不能证明输出光压缩或纠缠。")
    full.append("- 不得与 A*=0.8 经典强场谱拼接。")
    full.append("- 未自动启动第二波、服务器、Cn、SV、nmax 扫描。\n")
    figs = rec.get("figures") or {}
    if figs:
        full.append("## 六、图\n")
        for k, pth in figs.items():
            full.append(f"- {k}: `{pth}`")
        full.append("")
    full.append("完成后停机。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q1a_QED_theta_pi_20261120.md").write_text(
        "\n".join(full) + "\n", encoding="utf-8")
    one = []
    one.append("# 量子光多周期磁序响应：θ=π 精确 QED（一页）\n")
    one.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。`stopped = true`。\n")
    one.append(f"**{ans}**\n")
    one.append("| 量 | 结果 |")
    one.append("|---|---|")
    one.append(f"| 覆盖 | t_min/T0={cov.get('t_min_over_T0')} |")
    one.append(f"| 分类 | `{v}` |")
    if h2:
        one.append(f"| QED \\(\\lvert\\delta_\\theta J_{{2,x}}^o\\rvert\\) | {h2.get('abs_delta_jx')} |")
        one.append(f"| rel / δ_complex | {h2.get('rel_to_max')} / {h2.get('delta_complex')} |")
    if cmpg:
        one.append(f"| QED–GH5 rel_vec / Δφ | {cmpg.get('rel_vec')} / {cmpg.get('dphi_deg')}° |")
    one.append("")
    one.append("**外审问题：** 在同位移、同平均光子数的两种压缩输入下，精确 QED 是否也预测 H2 邻域磁序奇复电流的显著压缩角调控？\n")
    one.append(f"**{rec.get('answer_short')}**\n")
    one.append("停机。不自动开第二波、Cn、SV、nmax、G1/H2 或服务器。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q1a_QED_theta_pi_一页_20261120.md").write_text(
        "\n".join(one) + "\n", encoding="utf-8")


def analyze() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(OUT / "qed_theta_pi.json", rec)
        return rec
    jobs = {tag: load_job(tag) for tag, _, _ in JOBS}
    missing = [tag for tag, rec in jobs.items() if rec is None]
    if missing:
        rec = {"status": "missing_jobs", "missing": missing, "stopped": True}
        dump(OUT / "qed_theta_pi.json", rec)
        return rec
    for tag in ("ds_p", "ds_m"):
        if not (Q1A_DIR / f"series_{tag}.npz").is_file():
            rec = {"status": "missing_theta0_qed_archive", "tag": tag, "stopped": True}
            dump(OUT / "qed_theta_pi.json", rec)
            return rec
        if not (OUT / f"series_{tag}.npz").is_file():
            rec = {"status": "missing_theta_pi_series", "tag": tag, "stopped": True}
            dump(OUT / "qed_theta_pi.json", rec)
            return rec
    pf0 = json.loads((OUT / "preflight.json").read_text(encoding="utf-8")) if (OUT / "preflight.json").is_file() else {}
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=4.0)
    T0 = env.period
    omega0 = pf.OMEGA0
    t_pi_p, j_pi_p = load_series("ds_p")
    t_pi_m, j_pi_m = load_series("ds_m")
    t0_p, j0_p = load_series("ds_p", Q1A_DIR)
    t0_m, j0_m = load_series("ds_m", Q1A_DIR)
    t = t_pi_p
    for label, tt in (("pi_m", t_pi_m),):
        n = min(len(tt), len(t))
        if not np.allclose(tt[:n], t[:n], rtol=0, atol=1e-9):
            rec = {"status": "time_grid_mismatch", "tag": label, "stopped": True}
            dump(OUT / "qed_theta_pi.json", rec)
            return rec
    n_pi = min(len(t_pi_p), len(t_pi_m))
    t = t_pi_p[:n_pi]
    even_pi, odd_pi = odd.even_odd(j_pi_p[:n_pi], j_pi_m[:n_pi])
    # Align theta=0 archive onto the pi sample times by nearest match.
    def interp_pair(t_ref, t_src, j_src):
        out = np.empty((len(t_ref), j_src.shape[1]), complex)
        for a in range(j_src.shape[1]):
            out[:, a] = np.interp(t_ref, t_src, j_src[:, a].real) + 1j * np.interp(
                t_ref, t_src, j_src[:, a].imag)
        return out
    j0_p_i = interp_pair(t, t0_p, j0_p)
    j0_m_i = interp_pair(t, t0_m, j0_m)
    even_0, odd_0 = odd.even_odd(j0_p_i, j0_m_i)
    d_odd = odd_pi - odd_0
    d_even = even_pi - even_0
    arrs = {
        "J_odd_0": odd_0, "J_odd_pi": odd_pi, "dJ_odd": d_odd,
        "J_even_0": even_0, "J_even_pi": even_pi, "dJ_even": d_even,
    }
    t_max = float(t[-1])
    t_over = t_max / T0
    t_min_jobs = min(float(jobs[tag].get("t_reached_over_T0") or 0.0) for tag, _, _ in JOBS)
    covers_4 = bool(t_min_jobs + 1e-9 >= 4.0)
    covers_5 = bool(t_min_jobs + 1e-9 >= 5.0)
    covers_6 = bool(t_min_jobs + 1e-9 >= 6.0)
    win_main = (2.0 * T0, 4.0 * T0)
    win_front = (1.0 * T0, 3.0 * T0)
    win_back = (3.0 * T0, 5.0 * T0)
    main_ok = bool(t_max + 1e-9 >= win_main[1])
    front_ok = bool(t_max + 1e-9 >= win_front[1])
    back_ok = bool(t_max + 1e-9 >= win_back[1])
    coverage = {
        "t_min_over_T0": t_min_jobs,
        "t_series_over_T0": t_over,
        "covers_4T0": covers_4,
        "covers_5T0": covers_5,
        "covers_6T0": covers_6,
        "job_status": {tag: jobs[tag]["status"] for tag, _, _ in JOBS},
        "job_wall": {tag: jobs[tag].get("wall_seconds") for tag, _, _ in JOBS},
        "max_top_fock": {tag: jobs[tag].get("max_top_fock") for tag, _, _ in JOBS},
        "max_next_top_fock": {tag: jobs[tag].get("max_next_top_fock") for tag, _, _ in JOBS},
        "max_abs_norm_minus_1": {tag: jobs[tag].get("max_abs_norm_minus_1") for tag, _, _ in JOBS},
        "max_last_taylor": {tag: (jobs[tag].get("ledger") or {}).get("max_last_taylor_term")
                           for tag, _, _ in JOBS},
        "fock_alarm": {tag: jobs[tag].get("fock_alarm_0p05") for tag, _, _ in JOBS},
        "jobs": {tag: compact_job(jobs[tag]) for tag, _, _ in JOBS},
    }
    if not main_ok:
        rec = {
            "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
            "status": "main_window_incomplete_no_spectral_claim",
            "verdict": "main_window_incomplete_no_spectral_claim",
            "answer": (
                "两路尚未覆盖中央两周期主窗 [2T0,4T0]，不作完整谱学判断。"
                "已完成部分已存档。本轮不自动续第二波，不上服务器。"
            ),
            "answer_short": "主窗未完成，无谱学判断",
            "stopped": True, "preflight": pf0, "coverage": coverage,
            "source_hashes": hashes,
        }
        dump(OUT / "qed_theta_pi.json", rec)
        write_reports(rec)
        return rec
    main = pack_win("plateau_center_2T", t, omega0, T0, *win_main, arrs, main_ok)
    front = pack_win("front_2T", t, omega0, T0, *win_front, arrs, front_ok)
    back = pack_win("back_2T", t, omega0, T0, *win_back, arrs, back_ok)
    qs_c = np.linspace(0.0, 6.5, 326)
    cont_0 = rect_jq(t, odd_0, omega0, *win_main, qs_c)
    cont_pi = rect_jq(t, odd_pi, omega0, *win_main, qs_c)
    cont_d = rect_jq(t, d_odd, omega0, *win_main, qs_c)
    abs0 = np.array([float(np.linalg.norm(v)) for v in cont_0], float)
    abspi = np.array([float(np.linalg.norm(v)) for v in cont_pi], float)
    absd = np.array([float(np.linalg.norm(v)) for v in cont_d], float)
    peak_i = int(np.argmax(absd[1:])) + 1
    z0 = h2_jx(main["slots"], "J_odd_0")
    zpi = h2_jx(main["slots"], "J_odd_pi")
    zd = h2_jx(main["slots"], "dJ_odd")
    zf = h2_jx(front["slots"], "dJ_odd") if front_ok else (0j, 0j, 0.0, 0.0, 0.0)
    zb = h2_jx(back["slots"], "dJ_odd") if back_ok else (0j, 0j, 0.0, 0.0, 0.0)
    den = max(z0[2], zpi[2], 1e-30)
    rel = float(zd[2] / den)
    dcomplex = float(abs(zb[0] - zf[0]) / max(zd[2], 1e-30)) if (front_ok and back_ok) else None
    dphi = abs(angdiff(zpi[3], z0[3]))
    amp_ratio = float(zpi[2] / max(z0[2], 1e-30))
    gh = load_gh5_h2()
    rel_vec = float(abs(zd[0] - gh["jx_delta"]) / max(abs(zd[0]), 1e-30))
    dphi_gh = abs(angdiff(np.angle(zd[0]), np.angle(gh["jx_delta"])))
    dphi_gh_deg = float(dphi_gh * 180.0 / np.pi)
    agrees = bool(rel_vec < QED_GH5_REL and dphi_gh_deg < QED_GH5_DPHI_DEG)
    strong = bool(rel >= REL_MARK)
    window_ok = bool(dcomplex is not None and dcomplex < DCOMPLEX_MARK)
    if not covers_4:
        verdict = "main_window_incomplete_no_spectral_claim"
        answer = "主窗覆盖不足，不作完整谱学判断。"
        answer_short = "主窗未完成"
        status = "main_window_incomplete_no_spectral_claim"
    elif strong and window_ok and agrees:
        verdict = "qed_confirms_gh5_squeeze_axis_not_certified_HHG"
        answer = (
            "是：精确 QED 在同位移、同平均光子数下也给出显著且窗口不太散的 H2 磁序奇压缩角调控，"
            "并且复差向量在预设工程标记内与 GH5 相符。这支持压缩轴方向可以调控该观测量，"
            "但不是已认证 HHG，也不是输出纠缠或纯非经典优势。"
        )
        answer_short = "是（QED 支持压缩角调控，非认证 HHG）"
        status = "ok" if covers_6 else ("ok_full_switch_incomplete" if covers_5 else "ok_main_window_no_back")
    elif strong and window_ok and not agrees:
        verdict = "qed_supports_squeeze_axis_but_disagrees_with_gh5"
        answer = (
            "精确 QED 显示显著且窗口不太散的压缩角效应，但与 GH5 的复差向量定量不符"
            "（模长或相位超出标记）。如实报告方法分歧。不自动扫描 nmax/GH7/g/r/α/k。"
        )
        answer_short = "有压缩角效应，但与 GH5 定量不符"
        status = "ok" if covers_6 else ("ok_full_switch_incomplete" if covers_5 else "ok_main_window_no_back")
    elif strong:
        verdict = "qed_theta_effect_window_unstable"
        answer = (
            "精确 QED 主窗 H2 差较大，但前后窗复差向量不够一致。"
            "不能升级为稳定的压缩角调控。不作已认证 HHG 声明。"
        )
        answer_short = "有变化但窗口不稳定"
        status = "ok_window_unstable" if covers_4 else "ok_main_window_no_back"
    else:
        verdict = "qed_does_not_support_gh5_squeeze_axis"
        answer = (
            "否：在同位移、同平均光子数下，精确 QED 未给出超过工程标记的 H2 磁序奇压缩角调控。"
            "如实报告与 GH5 的方法分歧。不自动扫描 nmax、GH7、g、r、α 或更多 k 点。"
        )
        answer_short = "否（QED 不支持该 GH5 压缩角效应）"
        status = "ok_negative"
    FIG.mkdir(parents=True, exist_ok=True)
    f_plane = figure_h2_plane(
        z0[0], zpi[0], zd[0], gh["jx_0"], gh["jx_pi"], gh["jx_delta"],
        FIG / "h2_complex_plane_qed_theta0_vs_pi.svg")
    win_rows = [("front", zf[0], "#4c6b8a"), ("main", zd[0], "#2f9e44")]
    if back_ok:
        win_rows.append(("back", zb[0], "#c45c26"))
    f_win = figure_windows(win_rows, FIG / "h2_dtheta_three_windows_qed.svg")
    f_cont = figure_cont(qs_c, abs0, abspi, absd, FIG / "h2_continuum_qed_theta.svg")
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": status, "stopped": True, "verdict": verdict,
        "answer": answer, "answer_short": answer_short,
        "n_max": NMAX, "n_steps": N_FULL, "dt": env.t_total / N_FULL,
        "theta0_qed_archive": str(Q1A_DIR.as_posix()),
        "theta0_qed_not_rerun": True, "gh5_not_rerun": True,
        "operator_order": "D(alpha)S(xi)|0>",
        "source_hashes": hashes, "hash_abort": [],
        "preflight": pf0, "coverage": coverage,
        "windows": {"main": main, "front": front, "back": back},
        "h2_main": {
            "jx_0": pack_c(z0[0]), "jx_pi": pack_c(zpi[0]), "jx_delta": pack_c(zd[0]),
            "abs_jx_0": z0[2], "abs_jx_pi": zpi[2], "abs_delta_jx": zd[2],
            "abs_2d_0": z0[4], "abs_2d_pi": zpi[4], "abs_2d_delta": zd[4],
            "phase_0": z0[3], "phase_pi": zpi[3], "phase_delta": zd[3],
            "dphi_pi_minus_0_rad": dphi,
            "amp_ratio_pi_over_0": amp_ratio,
            "delta_jx_re": float(zd[0].real), "delta_jx_im": float(zd[0].imag),
            "rel_to_max": rel, "delta_complex": dcomplex,
            "abs_front": zf[2], "phase_front": zf[3],
            "abs_back": zb[2], "phase_back": zb[3],
            "engineering_marks_not_physical_laws": True,
            "rel_mark": REL_MARK, "dcomplex_mark": DCOMPLEX_MARK,
            "note": "rel_to_max and delta_complex use H2 Jx. Magnitude stability is not complex convergence.",
        },
        "qed_vs_gh5": {
            "rel_vec": rel_vec,
            "dphi_rad": dphi_gh,
            "dphi_deg": dphi_gh_deg,
            "agrees_complex": agrees,
            "threshold_rel": QED_GH5_REL,
            "threshold_dphi_deg": QED_GH5_DPHI_DEG,
            "gh5_jx_0": pack_c(gh["jx_0"]),
            "gh5_jx_pi": pack_c(gh["jx_pi"]),
            "gh5_jx_delta": pack_c(gh["jx_delta"]),
            "cannot_judge_by_magnitude_alone": True,
        },
        "continuum": {
            "q": qs_c.tolist(),
            "abs_Jodd_0": abs0.tolist(), "abs_Jodd_pi": abspi.tolist(),
            "abs_dJodd": absd.tolist(), "peak_q_dJ": float(qs_c[peak_i]),
            "peak_abs_dJ": float(absd[peak_i]),
            "integer_Fourier_is_not_certified_HHG": True,
        },
        "figures": {"complex_plane": f_plane, "three_windows": f_win, "continuum": f_cont},
        "physics_boundaries": {
            "Var_X_t0_is_not_constant_along_the_pulse": True,
            "free_field_Var_X_t": "cosh(2r)-sinh(2r) cos(2 omega0 t - theta)",
            "same_alpha_and_nbar_is_not_full_nonclassical_separation": True,
            "integer_Fourier_is_not_certified_HHG": True,
            "QED_current_is_not_output_squeezing_or_entanglement": True,
            "no_classical_Astar_0p8_splice": True,
            "no_fock_alarm_is_not_1e-8_error_bar": True,
            "did_not_auto_second_wave": True,
        },
        "did_not": [
            "rerun theta=0 QED", "rerun GH5", "Cn", "squeezed vacuum",
            "G1", "H2 extra k", "nmax scan", "D2", "Floquet", "T2", "server",
            "auto second 900s wave", "raise nmax",
        ],
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "qed_theta_pi.json", rec)
    write_reports(rec)
    return rec


def should_launch(tag: str) -> bool:
    rec = load_job(tag)
    if rec is None:
        return True
    if rec.get("status") in ("ok", "ok_fock_alarm"):
        return False
    if rec.get("status") == "timeout_partial":
        return False
    return True


def drive() -> int:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        dump(OUT / "qed_theta_pi.json", {"status": "hash_abort", "hash_mismatch": bad, "stopped": True})
        progress("hash abort, no propagation")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pf0 = preflight()
    if pf0.get("status") in ("hash_abort", "preflight_abort") or not pf0.get("same_resource_ok", False):
        rec = {
            "status": pf0.get("status"), "stopped": True, "preflight": pf0,
            "verdict": "preflight_abort_no_qed",
            "answer": "nmax=8 截断严重破坏原定同资源比较，停止 QED。未自动升 nmax。",
            "answer_short": "预检失败，未开 QED",
            "coverage": {},
        }
        dump(OUT / "qed_theta_pi.json", rec)
        write_reports(rec)
        progress("preflight abort, no QED")
        return 3
    t0 = time.perf_counter()
    procs, logs = {}, {}
    for tag, kind, neel in JOBS:
        if not should_launch(tag):
            progress(f"drive skip {tag} status={load_job(tag).get('status')}")
            continue
        logf = (OUT / f"log_{tag}.txt").open("a", encoding="utf-8")
        logs[tag] = logf
        envp = os.environ.copy()
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            envp[k] = "1"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", tag]
        progress(f"launch {tag}")
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
    codes = {tag: (procs[tag].returncode if tag in procs else 0) for tag, _, _ in JOBS}
    progress(f"jobs return {codes} driver_wall={time.perf_counter()-t0:.1f}s")
    recs = {tag: load_job(tag) for tag, _, _ in JOBS}
    t_vals = [
        0.0 if recs[tag] is None else float(recs[tag].get("t_reached_over_T0") or 0.0)
        for tag, _, _ in JOBS
    ]
    t_min = min(t_vals) if t_vals else 0.0
    covers_4 = bool(t_min + 1e-9 >= 4.0)
    if covers_4:
        rec = analyze()
        progress(f"analyze done verdict={rec.get('verdict')}")
        return 0
    timeout_rep = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "timeout_partial_no_analyze",
        "verdict": "main_window_incomplete_no_spectral_claim",
        "answer": "两路尚未覆盖 [2T0,4T0]，不作谱学判断。已安全存档。不自动续第二波，不上服务器。",
        "answer_short": "主窗未完成，无谱学判断",
        "stopped": True, "analyze_ran": False, "preflight": pf0,
        "t_min_over_T0": t_min, "covers_4T0": False,
        "returncodes": codes,
        "job_status": {tag: None if recs[tag] is None else recs[tag].get("status") for tag, _, _ in JOBS},
        "jobs": {tag: compact_job(recs[tag]) for tag, _, _ in JOBS},
        "driver_wall_s": time.perf_counter() - t0,
        "did_not_auto_second_wave": True,
    }
    dump(OUT / "qed_theta_pi.json", timeout_rep)
    write_reports(timeout_rep)
    progress(f"not enough coverage for analyze t_min/T0={t_min:.3f}")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        rec = preflight()
        print(json.dumps(jsonable({
            "status": rec.get("status"),
            "same_resource_ok": rec.get("same_resource_ok"),
            "abort_reasons": rec.get("abort_reasons"),
            "compare_nmax8": rec.get("compare_nmax8"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") == "preflight" and rec.get("same_resource_ok") else 3
    if args.job:
        rec = run_one_job(args.job)
        return 0 if rec.get("status") in ("ok", "ok_fock_alarm", "timeout_partial") else 1
    if args.analyze_only:
        rec = analyze()
        print(json.dumps(jsonable({
            "status": rec.get("status"), "verdict": rec.get("verdict"),
            "answer_short": rec.get("answer_short"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") not in (
            "hash_abort", "missing_jobs", "missing_theta0_qed_archive",
            "missing_theta_pi_series", "time_grid_mismatch",
            "main_window_incomplete_no_spectral_claim") else 1
    return drive()


if __name__ == "__main__":
    raise SystemExit(main())
