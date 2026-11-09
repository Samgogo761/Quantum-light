#!/usr/bin/env python3
"""Qin_a2_1_4_1 two-batch exact QED Q3: DS(theta=pi/2) then DS(theta=3pi/2).

Q3a: theta=pi/2, N=+/-1. Q3b runs only if both Q3a jobs cover [2T0,4T0]
and are numerically OK. Frozen read-only: theta=0 QED, theta=pi QED, four-angle
GH5. No Cn, SV, GH7, extra k, nmax scan, D2/Floquet, R2, or server.
Integer Fourier is not certified HHG. Displaced squeezed, not BSV.
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
Q3A_DIR = ROOT / "results/q_line/qinput_qin_a2_141_qed_q3a_pi2_20261120"
Q3B_DIR = ROOT / "results/q_line/qinput_qin_a2_141_qed_q3b_3pi2_20261120"
MERGE = ROOT / "results/q_line/qinput_qin_a2_141_qed_q3_covsign_20261120"
Q1A_DIR = ROOT / "results/q_line/qinput_qin_a2_141_q1a_20261120"
THETA_PI_DIR = ROOT / "results/q_line/qinput_qin_a2_141_qed_theta_pi_20261120"
GH5_DIR = ROOT / "results/q_line/qinput_qin_a2_141_gh5_q2_covsign_20261120"
DOCS = ROOT / "docs/04_BSV量子光"
TASK_TAG = "20261120"
RECORD_DATE = "2026-10-24"
PROTOCOL = "Qin_a2_1_4_1"
MU = -0.009010620372198276
NMAX = odd.NMAX
NMAX_REF = 80
ALPHA = odd.ALPHA
R_SQ = odd.R_SQ
THETA_0 = 0.0
THETA_H = 0.5 * np.pi
THETA_PI = float(np.pi)
THETA_T = 1.5 * np.pi
N_FULL = 11154
SAMPLE_EVERY = 32
CHECKPOINT_EVERY = 400
DRIVER_WALL_S = 900.0
JOB_WALL_S = 870.0
FOCK_ALARM = 0.05
CLASSICAL_A_STAR = 0.8
QED_GH5_REL = 0.20
QED_GH5_DPHI_DEG = 20.0
PHASE_MARK_DEG = 20.0
JOBS = (("ds_p", "ds", 1), ("ds_m", "ds", -1))
JOB_MAP = {tag: (kind, neel) for tag, kind, neel in JOBS}
OK_STATUSES = ("ok", "ok_fock_alarm", "timeout_partial")
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

OUT = MERGE
FIG = MERGE / "figures"
BATCH_NAME = None
BATCH_THETA = None


def set_batch(name: str) -> None:
    global OUT, FIG, BATCH_NAME, BATCH_THETA
    if name == "q3a":
        OUT = Q3A_DIR
        BATCH_THETA = float(THETA_H)
    elif name == "q3b":
        OUT = Q3B_DIR
        BATCH_THETA = float(THETA_T)
    else:
        raise ValueError(f"batch must be q3a or q3b, got {name!r}")
    BATCH_NAME = name
    FIG = OUT / "figures"


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
    MERGE.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with (OUT / "progress.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if OUT != MERGE:
        with (MERGE / "progress.txt").open("a", encoding="utf-8") as fh:
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
    """X=a+a†, P=-i(a-a†). Cov(X,P)=2 Im <(Δa)^2>. Vacuum Var(X)=1."""
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
    Mm = mean_aa - mean_a ** 2
    cov = 2.0 * float(np.asarray(Mm).imag)
    p = np.abs(cn) ** 2
    return {
        "unnormalized_mass": mass,
        "mean_a": pf.cpack(mean_a),
        "mean_aa": pf.cpack(mean_aa),
        "mean_n": mean_n,
        "mean_X": float(mean_X),
        "mean_P": float(mean_P),
        "var_X": float(mean_X2 - mean_X ** 2),
        "var_P": float(mean_P2 - mean_P ** 2),
        "cov_XP": float(cov),
        "M": pf.cpack(complex(Mm)),
        "p_n": [float(x) for x in p],
        "p_top": float(p[-1]) if len(p) else 0.0,
    }


def analytic_xp(theta: float) -> dict:
    a = complex(ALPHA)
    r = float(R_SQ)
    th = float(theta)
    mx, mp = 2.0 * a.real, 2.0 * a.imag
    c2, s2 = float(np.cosh(2.0 * r)), float(np.sinh(2.0 * r))
    vx = c2 - s2 * np.cos(th)
    vp = c2 + s2 * np.cos(th)
    cxy = -s2 * np.sin(th)
    nbar = abs(a) ** 2 + np.sinh(r) ** 2
    Mm = -0.5 * np.exp(1j * th) * s2
    return {
        "alpha": pf.cpack(a), "r": r, "theta": th,
        "mean_X": mx, "mean_P": mp, "var_X": float(vx), "var_P": float(vp),
        "cov_XP": float(cxy), "mean_n_analytic": float(nbar),
        "M": pf.cpack(complex(Mm)),
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
        "free_field_cov_diff": "-2 sinh(2r) sin[omega0(t+t')]",
        "not_BSV": True,
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
        "d_mean_n_8_vs_ideal": abs(m8["mean_n"] - ana["mean_n_analytic"]),
        "rel_d_mean_n_8_vs_ideal": abs(m8["mean_n"] - ana["mean_n_analytic"]) / max(abs(ana["mean_n_analytic"]), 1e-30),
        "rel_var_X_8_vs_ideal": abs(m8["var_X"] / ana["var_X"] - 1.0),
        "rel_var_P_8_vs_ideal": abs(m8["var_P"] / ana["var_P"] - 1.0),
        "d_cov_8_vs_ideal": abs(m8["cov_XP"] - ana["cov_XP"]),
        "rel_cov_8_vs_ideal": abs(m8["cov_XP"] - ana["cov_XP"]) / max(abs(ana["cov_XP"]), 1e-30),
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
        "not_BSV": True, "stopped": False,
    }
    MERGE.mkdir(parents=True, exist_ok=True)
    if bad:
        rec["status"] = "hash_abort"
        rec["stopped"] = True
        dump(MERGE / "preflight.json", rec)
        return rec
    row_h = theta_fock_row(THETA_H)
    row_t = theta_fock_row(THETA_T)
    rec["theta_h"] = row_h
    rec["theta_t"] = row_t
    mh = row_h["nmax8_used_by_propagator"]
    mt = row_t["nmax8_used_by_propagator"]
    rec["compare_nmax8"] = {
        "d_mean_X": abs(mt["mean_X"] - mh["mean_X"]),
        "rel_d_mean_n": abs(mt["mean_n"] - mh["mean_n"]) / max(abs(mh["mean_n"]), 1e-30),
        "d_var_X": abs(mt["var_X"] - mh["var_X"]),
        "d_var_P": abs(mt["var_P"] - mh["var_P"]),
        "cov_h": mh["cov_XP"],
        "cov_t": mt["cov_XP"],
        "cov_sum": mh["cov_XP"] + mt["cov_XP"],
        "mean_X_h": mh["mean_X"], "mean_n_h": mh["mean_n"],
        "var_X_h": mh["var_X"], "var_P_h": mh["var_P"],
        "P_n_ge9_h": row_h["P_n_ge9_ideal80"],
        "P_n_ge9_t": row_t["P_n_ge9_ideal80"],
    }
    c = rec["compare_nmax8"]
    abort_reasons = []
    if c["d_mean_X"] > 0.02:
        abort_reasons.append("mean_X_h8_vs_t8")
    if c["rel_d_mean_n"] > 0.02:
        abort_reasons.append("mean_n_h8_vs_t8")
    if abs(mh["mean_X"] - 1.0) > 0.02:
        abort_reasons.append("mean_X_h8_vs_1")
    if abs(mt["mean_X"] - 1.0) > 0.02:
        abort_reasons.append("mean_X_t8_vs_1")
    if row_h["rel_d_mean_n_8_vs_ideal"] > 0.02:
        abort_reasons.append("mean_n_h8_vs_ideal")
    if row_t["rel_d_mean_n_8_vs_ideal"] > 0.02:
        abort_reasons.append("mean_n_t8_vs_ideal")
    if row_h["rel_var_X_8_vs_ideal"] > 0.10:
        abort_reasons.append("var_X_h8_vs_ideal")
    if row_t["rel_var_X_8_vs_ideal"] > 0.10:
        abort_reasons.append("var_X_t8_vs_ideal")
    if abs(mh["var_X"] - mh["var_P"]) > 0.05:
        abort_reasons.append("varX_neq_varP_h")
    if abs(mt["var_X"] - mt["var_P"]) > 0.05:
        abort_reasons.append("varX_neq_varP_t")
    if mh["cov_XP"] * mt["cov_XP"] >= 0.0:
        abort_reasons.append("cov_sign_not_flipped")
    if abs(c["cov_sum"]) > 0.02:
        abort_reasons.append("cov_abs_mismatch")
    rec["abort_reasons"] = abort_reasons
    rec["same_resource_ok"] = not abort_reasons
    rec["note_tail"] = (
        "P(n>=9) is recorded, not an abort. No 0.05 alarm is not a 1e-8 bound. "
        "Abort only if nmax=8 destroys the same-resource comparison."
    )
    if abort_reasons:
        rec["status"] = "preflight_abort"
        rec["stopped"] = True
        rec["qed_not_started"] = True
    dump(MERGE / "preflight.json", rec)
    progress(
        f"preflight ok={rec['same_resource_ok']} "
        f"meanX_h={mh['mean_X']:.6f} nbar_h={mh['mean_n']:.6f} "
        f"cov_h={mh['cov_XP']:.6f} cov_t={mt['cov_XP']:.6f} abort={abort_reasons}"
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


def load_job(tag: str, directory: Path | None = None):
    directory = OUT if directory is None else directory
    path = directory / f"job_{tag}.json"
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
                progress(f"{BATCH_NAME}:{tag} ckpt j={j}/{n_steps} wall={clock.elapsed():.1f}s")
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
    if BATCH_THETA is None:
        raise RuntimeError("set_batch before --job")
    kind, neel = JOB_MAP[tag]
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        report = {
            "tag": tag, "status": "hash_abort", "hash_mismatch": bad,
            "source_hashes": hashes, "stopped": True, "batch": BATCH_NAME,
        }
        dump(OUT / f"job_{tag}.json", report)
        return report

    prev = load_job(tag)
    if prev and prev.get("status") in ("ok", "ok_fock_alarm"):
        progress(f"{BATCH_NAME}:{tag} already {prev['status']}, skip")
        return prev
    if prev and prev.get("status") == "timeout_partial":
        progress(f"{BATCH_NAME}:{tag} timeout_partial, this round does not auto-resume")
        return prev

    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=4.0)
    n_steps = N_FULL
    dt = env.t_total / n_steps
    ek, emk, masses = odd.ground_pair_neel(geo, k, neel)
    raw = odd.photon_raw("ds", NMAX, float(BATCH_THETA))
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
            progress(f"{BATCH_NAME}:{tag} resume next_j={start_j} n_samples={len(samples)}")

    dump(OUT / f"job_{tag}.json", {
        "tag": tag, "kind": kind, "neel": int(neel), "status": "running",
        "theta": float(BATCH_THETA), "batch": BATCH_NAME, "resumed": resumed,
        "next_j": start_j, "utc": datetime.now(timezone.utc).isoformat(),
    })
    resume_obs = pf.snapshot(model, psi, env.s(start_j * dt))
    if resumed and prev and prev.get("initial"):
        s0 = prev["initial"]
    else:
        s0 = resume_obs
    progress(f"{BATCH_NAME}:{tag} start ds theta={BATCH_THETA:.6f} neel={neel} "
             f"dim={model.dim} j0={start_j}/{n_steps}")
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
        progress(f"{BATCH_NAME}:{tag} exception {err}")

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
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "tag": tag, "kind": kind, "neel": int(neel), "status": status,
        "batch": BATCH_NAME, "resumed": resumed,
        "resume_from_next_j": int(start_j) if resumed else 0,
        "finite_state": finite_ok, "error": err,
        "fock_alarm_0p05": fock_alarm, "source_hashes": hashes,
        "definition": {
            "n_max": NMAX, "g": pf.G_ABS, "omega0": pf.OMEGA0,
            "alpha": {"re": float(ALPHA.real), "im": float(ALPHA.imag)},
            "r": R_SQ, "theta": float(BATCH_THETA),
            "operator_order": "D(alpha)S(xi)|0>",
            "not_BSV": True,
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
        "initial": s0, "endpoint": send,
        "ledger": {k: led[k] for k in led if k != "samples"},
        "max_top_fock": max_top, "max_next_top_fock": max_next,
        "max_n_ph": float(np.max(nphs)), "min_n_ph": float(np.min(nphs)),
        "max_abs_norm_minus_1": max_dn,
        "t_reached": t_reach, "t_reached_over_T0": t_over,
        "covers_5T0": bool(t_over + 1e-9 >= 5.0),
        "covers_4T0": bool(t_over + 1e-9 >= 4.0),
        "covers_6T0": bool(t_over + 1e-9 >= 6.0),
        "wave": 1, "wall_seconds": wall,
        "seconds_per_step": (wall / max(completed - start_j, 1)),
        "time_series": pf.pack_samples(samples) if samples else {},
        "n_samples": len(samples),
        "series_npz": str((OUT / f"series_{tag}.npz").as_posix()),
        "integer_Fourier_is_not_certified_HHG": True,
        "did_not_auto_raise_nmax": True,
        "did_not_auto_second_wave": True,
    }
    if resumed:
        report["resume_snapshot"] = resume_obs
    dump(OUT / f"job_{tag}.json", report)
    progress(f"{BATCH_NAME}:{tag} {status} steps={completed}/{n_steps} t/T0={t_over:.3f} "
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


def load_series(tag: str, directory: Path):
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


def h2_from_slots(slots, key):
    jx = cz(slots[key]["Jx"][1])
    jy = cz(slots[key]["Jy"][1])
    return {
        "jx": jx, "jy": jy, "abs_jx": float(abs(jx)),
        "phase_x": float(np.angle(jx)), "abs_2d": float(slots[key]["abs"][1]),
    }


def svg_axes(x0, y0, w, h, xlabel, ylabel, title):
    return (
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fffdf8" '
        f'stroke="#2c3338" stroke-width="1"/>'
        f'<text x="{x0 + w/2:.1f}" y="{y0 - 10:.1f}" text-anchor="middle" '
        f'font-size="13" fill="#1b242c">{title}</text>'
        f'<text x="{x0 + w/2:.1f}" y="{y0 + h + 22:.1f}" text-anchor="middle" '
        f'font-size="11" fill="#243038">{xlabel}</text>'
        f'<text x="{x0 - 28:.1f}" y="{y0 + h/2:.1f}" text-anchor="middle" '
        f'font-size="11" fill="#243038" transform="rotate(-90 {x0-28:.1f} {y0+h/2:.1f})">{ylabel}</text>'
    )


def write_svg(path: Path, body: str, width=960, height=580) -> str:
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


def compact_job(rec):
    if rec is None:
        return None
    led = rec.get("ledger") or {}
    return {
        "status": rec.get("status"),
        "theta": rec.get("definition", {}).get("theta", rec.get("theta")),
        "t_reached_over_T0": rec.get("t_reached_over_T0"),
        "steps_completed": led.get("steps_completed"),
        "max_top_fock": rec.get("max_top_fock"),
        "max_next_top_fock": rec.get("max_next_top_fock"),
        "max_abs_norm_minus_1": rec.get("max_abs_norm_minus_1"),
        "max_last_taylor_term": led.get("max_last_taylor_term"),
        "max_n_ph": rec.get("max_n_ph"),
        "fock_alarm_0p05": rec.get("fock_alarm_0p05"),
        "wall_seconds": rec.get("wall_seconds"),
        "covers_4T0": rec.get("covers_4T0"),
        "covers_5T0": rec.get("covers_5T0"),
        "covers_6T0": rec.get("covers_6T0"),
        "finite_state": rec.get("finite_state"),
        "error": rec.get("error"),
        "n_samples": rec.get("n_samples"),
    }


def batch_gate(directory: Path) -> dict:
    jobs = {tag: load_job(tag, directory) for tag, _, _ in JOBS}
    missing = [tag for tag, rec in jobs.items() if rec is None]
    rows = []
    for tag, rec in jobs.items():
        if rec is None:
            rows.append({"tag": tag, "ok": False, "reason": "missing"})
            continue
        st = rec.get("status")
        cov4 = bool(rec.get("covers_4T0"))
        finite = rec.get("finite_state", True)
        ok = st in OK_STATUSES and cov4 and finite and st != "failed"
        if st == "timeout_partial" and not cov4:
            ok = False
        rows.append({
            "tag": tag, "ok": ok, "status": st, "covers_4T0": cov4,
            "t_reached_over_T0": rec.get("t_reached_over_T0"),
            "finite_state": finite,
        })
    ok = (not missing) and all(r["ok"] for r in rows)
    return {"ok": ok, "missing": missing, "rows": rows, "jobs": {k: compact_job(v) for k, v in jobs.items()}}


def should_launch(tag: str) -> bool:
    rec = load_job(tag)
    if rec is None:
        return True
    if rec.get("status") in ("ok", "ok_fock_alarm", "timeout_partial"):
        return False
    return True


def drive_batch(name: str) -> dict:
    set_batch(name)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    procs, logs = {}, {}
    for tag, _, _ in JOBS:
        if not should_launch(tag):
            progress(f"drive skip {name}:{tag} status={load_job(tag).get('status')}")
            continue
        logf = (OUT / f"log_{tag}.txt").open("a", encoding="utf-8")
        logs[tag] = logf
        envp = os.environ.copy()
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            envp[k] = "1"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--batch", name, "--job", tag]
        progress(f"launch {name}:{tag}")
        procs[tag] = subprocess.Popen(
            cmd, cwd=str(HERE), env=envp, stdout=logf, stderr=subprocess.STDOUT)
    while True:
        elapsed = time.perf_counter() - t0
        alive = {tag: p for tag, p in procs.items() if p.poll() is None}
        if not alive:
            break
        if elapsed >= DRIVER_WALL_S:
            progress(f"{name} driver wall {DRIVER_WALL_S}s: terminating {list(alive)}")
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
    gate = batch_gate(OUT)
    gate["batch"] = name
    gate["returncodes"] = codes
    gate["driver_wall_s"] = time.perf_counter() - t0
    dump(OUT / "batch_gate.json", gate)
    progress(f"{name} gate_ok={gate['ok']} codes={codes} wall={gate['driver_wall_s']:.1f}s")
    return gate


def load_gh5():
    path = GH5_DIR / "gh5_q2_covsign.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    h2 = d["h2_main"]
    fit = d["angle_fit"]
    return {
        "jx_0": cz(h2["jx_0"]), "jx_h": cz(h2["jx_h"]),
        "jx_pi": cz(h2["jx_pi"]), "jx_t": cz(h2["jx_t"]),
        "jx_delta": cz(h2["jx_delta"]),
        "abs_delta": h2["abs_delta_jx"],
        "dphi_t_minus_h_rad": h2["dphi_t_minus_h_rad"],
        "abs_even_h": h2.get("abs_even_jx_h"),
        "abs_even_t": h2.get("abs_even_jx_t"),
        "B": cz(fit["B"]), "A": cz(fit["A"]), "C": cz(fit["C"]),
        "rel_resid": fit["rel_resid"],
        "verdict": d.get("verdict"),
    }


def load_angle_eo(directory: Path):
    t_p, j_p = load_series("ds_p", directory)
    t_m, j_m = load_series("ds_m", directory)
    n = min(len(t_p), len(t_m))
    if not np.allclose(t_p[:n], t_m[:n], rtol=0, atol=1e-9):
        raise RuntimeError(f"time-grid mismatch in {directory}")
    even, oddj = odd.even_odd(j_p[:n], j_m[:n])
    return t_p[:n], even, oddj


def analyze_one_angle(name, directory, T0, omega0):
    t, even, oddj = load_angle_eo(directory)
    t_max = float(t[-1])
    arrs = {"J_odd": oddj, "J_even": even}
    win_main = (2.0 * T0, 4.0 * T0)
    win_front = (1.0 * T0, 3.0 * T0)
    win_back = (3.0 * T0, 5.0 * T0)
    main_ok = bool(t_max + 1e-9 >= win_main[1])
    front_ok = bool(t_max + 1e-9 >= win_front[1])
    back_ok = bool(t_max + 1e-9 >= win_back[1])
    if not main_ok:
        return {
            "name": name, "directory": str(directory.as_posix()),
            "t_over_T0": t_max / T0, "main_ok": False,
            "front_ok": front_ok, "back_ok": back_ok, "unavailable": True,
        }
    main = pack_win("plateau_center_2T", t, omega0, T0, *win_main, arrs, True)
    front = pack_win("front_2T", t, omega0, T0, *win_front, arrs, front_ok) if front_ok else None
    back = pack_win("back_2T", t, omega0, T0, *win_back, arrs, back_ok) if back_ok else None
    h2o = h2_from_slots(main["slots"], "J_odd")
    h2e = h2_from_slots(main["slots"], "J_even")
    qs_c = np.linspace(0.0, 6.5, 326)
    cont = rect_jq(t, oddj, omega0, *win_main, qs_c)
    abs_c = np.array([float(np.linalg.norm(v)) for v in cont], float)
    h16 = []
    for iq, q in enumerate(range(1, 7)):
        jx = cz(main["slots"]["J_odd"]["Jx"][iq])
        h16.append({"q": q, "jx": pack_c(jx), "abs_jx": float(abs(jx)), "abs_2d": float(main["slots"]["J_odd"]["abs"][iq])})
    return {
        "name": name, "directory": str(directory.as_posix()),
        "t_over_T0": t_max / T0, "main_ok": True,
        "front_ok": front_ok, "back_ok": back_ok, "unavailable": False,
        "windows": {"main": main, "front": front, "back": back},
        "h2_odd": {
            "jx": pack_c(h2o["jx"]), "jy": pack_c(h2o["jy"]),
            "abs_jx": h2o["abs_jx"], "phase_x": h2o["phase_x"], "abs_2d": h2o["abs_2d"],
        },
        "h2_even": {
            "jx": pack_c(h2e["jx"]), "abs_jx": h2e["abs_jx"], "phase_x": h2e["phase_x"],
        },
        "h1_h6_odd_jx": h16,
        "continuum_q": qs_c.tolist(),
        "continuum_abs": abs_c.tolist(),
        "h2_odd_c": h2o["jx"],
        "h2_front_c": h2_from_slots(front["slots"], "J_odd")["jx"] if front else None,
        "h2_back_c": h2_from_slots(back["slots"], "J_odd")["jx"] if back else None,
    }


def four_fit(j0, jh, jpi, jt):
    B = 0.25 * (j0 + jh + jpi + jt)
    A = 0.25 * (j0 - 1j * jh - jpi + 1j * jt)
    C = 0.25 * (j0 + 1j * jh - jpi - 1j * jt)
    thetas = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)
    js = (j0, jh, jpi, jt)
    pred = [B + A * np.exp(1j * th) + C * np.exp(-1j * th) for th in thetas]
    resid = [pred[i] - js[i] for i in range(4)]
    den = max(abs(z) for z in js)
    rel = float(max(abs(z) for z in resid) / max(den, 1e-30))
    rms = float(np.sqrt(np.mean([abs(z) ** 2 for z in resid])) / max(den, 1e-30))
    return {
        "B": pack_c(B), "A": pack_c(A), "C": pack_c(C),
        "abs_B": float(abs(B)), "abs_A": float(abs(A)), "abs_C": float(abs(C)),
        "abs_C_over_A": float(abs(C) / max(abs(A), 1e-30)),
        "rel_resid_max": rel, "rel_resid_rms": rms,
        "aliasing_warning": "four equally spaced angles cannot separate e^{i5 theta} from e^{i theta}",
        "not_microscopic_chi2": True,
        "_B": B, "_A": A, "_C": C,
    }


def figure_four_angle(qed, gh, path):
    labels = ("0", "π/2", "π", "3π/2")
    zq = [qed[k] for k in ("0", "h", "pi", "t")]
    zg = [gh[k] for k in ("0", "h", "pi", "t")]
    lim = max(abs(z) for z in zq + zg) * 1.30 + 1e-30
    x0, y0, w, h = 90, 50, 760, 430
    body = svg_axes(x0, y0, w, h, "Re Jx (H2 odd)", "Im Jx (H2 odd)",
                    "Four-angle H2 Néel-odd Jx: QED vs GH5")
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    cols = ("#4c6b8a", "#2f9e44", "#c45c26", "#6b4c9a")

    def poly(zs, color, dash=False):
        pts = []
        for z in list(zs) + [zs[0]]:
            px, py = map_xy(z, lim, x0, y0, w, h)
            pts.append(f"{px:.1f},{py:.1f}")
        dsh = ' stroke-dasharray="5 4"' if dash else ""
        return f'<polyline fill="none" stroke="{color}" stroke-width="1.6"{dsh} points="{" ".join(pts)}"/>'

    body += poly(zq, "#1b242c", dash=False)
    body += poly(zg, "#8a9096", dash=True)
    for i, (zq_i, zg_i, lab, col) in enumerate(zip(zq, zg, labels, cols)):
        px, py = map_xy(zq_i, lim, x0, y0, w, h)
        gx, gy = map_xy(zg_i, lim, x0, y0, w, h)
        body += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{col}"/>'
        body += (f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="5.5" fill="none" '
                 f'stroke="{col}" stroke-width="1.6" stroke-dasharray="3 2"/>')
        body += f'<text x="{px+8:.1f}" y="{py-8:.1f}" font-size="11" fill="{col}">QED θ={lab}</text>'
    body += '<text x="90" y="36" font-size="11" fill="#1b242c">solid = QED; dashed open = GH5 (a priori, not rerun)</text>'
    body += ('<text x="480" y="568" font-size="10" fill="#5c5148">'
             "same α, nbar, Var(X), Var(P) at π/2 vs 3π/2; integer Fourier ≠ certified HHG</text>")
    return write_svg(path, body)


def figure_delta(zd_qed, zd_gh, path):
    lim = max(abs(zd_qed), abs(zd_gh)) * 1.30 + 1e-30
    x0, y0, w, h = 90, 50, 760, 430
    body = svg_axes(x0, y0, w, h, "Re δ⊥ Jx", "Im δ⊥ Jx",
                    "δ⊥ = J(3π/2)−J(π/2), H2 odd Jx")
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    for z, col, lab, dash in (
        (zd_qed, "#2f9e44", "QED δ⊥", False),
        (zd_gh, "#8a9096", "GH5 δ⊥ (frozen)", True),
    ):
        px, py = map_xy(z, lim, x0, y0, w, h)
        dsh = ' stroke-dasharray="5 4"' if dash else ""
        body += (f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
                 f'stroke="{col}" stroke-width="2.2"{dsh}/>')
        body += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}"/>'
        body += f'<text x="{px+8:.1f}" y="{py-8:.1f}" font-size="12" fill="{col}">{lab}</text>'
    return write_svg(path, body)


def figure_cont(q, yh, yt, yd, path):
    xmin, xmax = float(q[0]), float(q[-1])
    ys = np.concatenate([yh, yt, yd])
    ymin, ymax = 0.0, float(np.max(ys)) * 1.12 + 1e-30
    x0, y0p, w, h = 70, 40, 820, 430
    body = svg_axes(x0, y0p, w, h, "Ω / ω0", "|J^o(Ω)|",
                    "QED main-window continuum |J_odd|, π/2 vs 3π/2")

    def poly(xs, ys_, color, sw):
        pts = []
        for x, y in zip(xs, ys_):
            px = x0 + (float(x) - xmin) / (xmax - xmin) * w
            py = y0p + (1.0 - (float(y) - ymin) / max(ymax - ymin, 1e-30)) * h
            pts.append(f"{px:.2f},{py:.2f}")
        return (f'<polyline fill="none" stroke="{color}" stroke-width="{sw}" '
                f'points="{" ".join(pts)}"/>')

    body += poly(q, yh, "#2f9e44", 1.5)
    body += poly(q, yt, "#6b4c9a", 1.5)
    body += poly(q, yd, "#c45c26", 1.5)
    for n in range(1, 7):
        px = x0 + (n - xmin) / (xmax - xmin) * w
        body += (f'<line x1="{px:.1f}" y1="{y0p}" x2="{px:.1f}" y2="{y0p+h}" '
                 f'stroke="#8a9096" stroke-dasharray="3 4" stroke-width="0.8"/>')
    body += '<text x="80" y="36" font-size="11" fill="#2f9e44">θ=π/2</text>'
    body += '<text x="160" y="36" font-size="11" fill="#6b4c9a">θ=3π/2</text>'
    body += '<text x="260" y="36" font-size="11" fill="#c45c26">|δ⊥|</text>'
    return write_svg(path, body, width=960, height=560)


def fmt_c(z, scale=1e8):
    z = complex(z)
    return f"({z.real*scale:+.3f}{z.imag*scale:+.3f}i)×10^{{{int(np.log10(scale))}}}"


def write_reports(rec: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    h2 = rec.get("h2_main") or {}
    cov = rec.get("coverage") or {}
    cmpg = rec.get("qed_vs_gh5") or {}
    fit = rec.get("qed_angle_fit") or {}
    v = rec.get("verdict")
    ans = rec.get("answer")
    full = []
    full.append("# 量子光多周期磁序响应：Q3 精确 QED 正交协方差验证\n")
    full.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。`stopped = true`。\n")
    full.append("冻结只读：θ=0 QED、θ=π QED、四角 GH5。本轮仅新增 θ=π/2 与 3π/2 的 N=±1。位移压缩态，不是纯 BSV。\n")
    full.append(f"**判定：** `{v}`\n")
    full.append(f"{ans}\n")
    full.append("## 一、覆盖\n")
    full.append(f"- Q3a gate = {cov.get('q3a_ok')}")
    full.append(f"- Q3b gate = {cov.get('q3b_ok')}")
    full.append(f"- Q3a t/T0 = {cov.get('q3a_t')}")
    full.append(f"- Q3b t/T0 = {cov.get('q3b_t')}")
    full.append(f"- job_status = {cov.get('job_status')}\n")
    if h2:
        full.append("## 二、主窗 H2 磁序奇 Jx\n")
        full.append("| θ | QED Jx | |J| | arg |")
        full.append("|---|---|---|---|")
        for lab, key in (("0 (RO)", "0"), ("π/2", "h"), ("π (RO)", "pi"), ("3π/2", "t")):
            full.append(
                f"| {lab} | {h2.get(f'jx_{key}_re')}+i{h2.get(f'jx_{key}_im')} | "
                f"{h2.get(f'abs_{key}')} | {h2.get(f'phase_{key}_deg')}° |"
            )
        full.append("")
        full.append(f"- δ⊥ = {h2.get('delta_re')}+i{h2.get('delta_im')}")
        full.append(f"- |δ⊥| = {h2.get('abs_delta')}")
        full.append(f"- 振幅比 3π/2 ÷ π/2 = {h2.get('amp_ratio_t_over_h')}")
        full.append(f"- Δφ (3π/2 − π/2) = {h2.get('dphi_deg')}°")
        full.append(f"- 偶电流 |J_even,x(H2)| π/2, 3π/2 = {h2.get('abs_even_h')}, {h2.get('abs_even_t')}")
        full.append("- 整数 Fourier 不是已认证 HHG。磁序奇分量仍远小于偶背景。\n")
    if cmpg:
        full.append("## 三、与冻结 GH5 对照\n")
        full.append("| 量 | 值 |")
        full.append("|---|---|")
        full.append(f"| GH5 δ⊥ (事前) | {cmpg.get('gh5_delta')} |")
        full.append(f"| rel_vec | {cmpg.get('rel_vec')} |")
        full.append(f"| 相位偏差 (deg) | {cmpg.get('dphi_deg')} |")
        full.append(f"| 工程标记内 | {cmpg.get('agrees_complex')} |")
        full.append("20% / 20° 是预先声明的工程比较阈值，不是物理定律。\n")
    if fit:
        full.append("## 四、四点角向拟合 B + A e^{iθ} + C e^{-iθ}\n")
        full.append(f"- B = {fit.get('B')}")
        full.append(f"- A = {fit.get('A')}")
        full.append(f"- C = {fit.get('C')}")
        full.append(f"- |C|/|A| = {fit.get('abs_C_over_A')}")
        full.append(f"- 相对残差 = {fit.get('rel_resid_max')}")
        full.append("四个等间隔角存在高阶混叠。近圆拟合不能证明响应仅由 M 或微观 χ(2) 决定。\n")
    h16 = rec.get("h1_h6_delta") or []
    if h16:
        full.append("## 五、H1–H6 差向量模长（冻结列表）\n")
        full.append("| q | |δ⊥ Jx| |")
        full.append("|---|---|")
        for row in h16:
            full.append(f"| {row['q']} | {row['abs_jx']} |")
        full.append("")
    full.append("## 六、边界\n")
    full.append("- 输入是 D(α)S(ξ)|0⟩，α=0.5，不是纯压缩真空。")
    full.append("- 初始 Var(X)=Var(P) 相同，不表示全程 X(t) 方差相同。")
    full.append("- 未证明经典正 P 不可复制，也未证明反铁磁特殊性相对非磁介质。")
    full.append("- 未自动开第三批、服务器、Cn、SV、nmax 或更多压缩角。\n")
    figs = rec.get("figures") or {}
    if figs:
        full.append("## 七、图\n")
        for k, pth in figs.items():
            full.append(f"- {k}: `{pth}`")
        full.append("")
    full.append("完成后停机。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q3_QED_covsign_20261120.md").write_text(
        "\n".join(full) + "\n", encoding="utf-8")
    one = []
    one.append("# 量子光多周期磁序响应：Q3 精确 QED（一页）\n")
    one.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。`stopped = true`。\n")
    one.append(f"**{ans}**\n")
    one.append("| 量 | 结果 |")
    one.append("|---|---|")
    one.append(f"| 分类 | `{v}` |")
    one.append(f"| Q3a/Q3b 覆盖 | {cov.get('q3a_ok')}/{cov.get('q3b_ok')} |")
    if h2:
        one.append(f"| QED Δφ (3π/2−π/2) | {h2.get('dphi_deg')}° |")
        one.append(f"| QED 振幅比 | {h2.get('amp_ratio_t_over_h')} |")
        one.append(f"| QED |δ⊥| | {h2.get('abs_delta')} |")
    if cmpg:
        one.append(f"| QED–GH5 rel_vec / Δφ | {cmpg.get('rel_vec')} / {cmpg.get('dphi_deg')}° |")
    one.append("")
    one.append("**外审问题：** 精确 QED 是否也确认仅翻转初始 X–P 协方差符号，就能显著旋转磁序奇 H2 的复响应相位？\n")
    one.append(f"**{rec.get('answer_short')}**\n")
    one.append("停机。不自动开第三批、Cn、SV、nmax 或服务器。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q3_QED_covsign_一页_20261120.md").write_text(
        "\n".join(one) + "\n", encoding="utf-8")


def analyze() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    MERGE.mkdir(parents=True, exist_ok=True)
    FIG = MERGE / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    if bad:
        rec = {"status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(MERGE / "qed_q3_covsign.json", rec)
        return rec
    pf0 = json.loads((MERGE / "preflight.json").read_text(encoding="utf-8")) if (MERGE / "preflight.json").is_file() else {}
    g3a = batch_gate(Q3A_DIR)
    g3b = batch_gate(Q3B_DIR)
    coverage = {
        "q3a_ok": g3a["ok"], "q3b_ok": g3b["ok"],
        "q3a_t": {tag: (g3a["jobs"][tag] or {}).get("t_reached_over_T0") for tag, _, _ in JOBS},
        "q3b_t": {tag: (g3b["jobs"][tag] or {}).get("t_reached_over_T0") for tag, _, _ in JOBS},
        "job_status": {
            "q3a": {tag: (g3a["jobs"][tag] or {}).get("status") for tag, _, _ in JOBS},
            "q3b": {tag: (g3b["jobs"][tag] or {}).get("status") for tag, _, _ in JOBS},
        },
        "q3a_jobs": g3a["jobs"], "q3b_jobs": g3b["jobs"],
        "max_top_fock": {
            "q3a": {tag: (g3a["jobs"][tag] or {}).get("max_top_fock") for tag, _, _ in JOBS},
            "q3b": {tag: (g3b["jobs"][tag] or {}).get("max_top_fock") for tag, _, _ in JOBS},
        },
    }
    if not g3a["ok"]:
        rec = {
            "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
            "status": "q3a_incomplete_stop",
            "verdict": "q3a_incomplete_no_spectral_claim",
            "answer": "Q3a 两路尚未完整覆盖主窗或数值异常，按批准边界停止，不启动 Q3b。已安全存档。",
            "answer_short": "Q3a 主窗未完成，停止",
            "stopped": True, "preflight": pf0, "coverage": coverage,
            "source_hashes": hashes,
        }
        dump(MERGE / "qed_q3_covsign.json", rec)
        write_reports(rec)
        return rec
    if not g3b["ok"]:
        rec = {
            "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
            "status": "q3b_incomplete_stop",
            "verdict": "q3b_incomplete_no_four_angle_claim",
            "answer": "Q3a 已覆盖主窗，但 Q3b 未完成主窗，不能比较协方差符号反转。不自动续算。",
            "answer_short": "Q3b 主窗未完成，无四角判断",
            "stopped": True, "preflight": pf0, "coverage": coverage,
            "source_hashes": hashes,
        }
        dump(MERGE / "qed_q3_covsign.json", rec)
        write_reports(rec)
        return rec
    for tag in ("ds_p", "ds_m"):
        if not (Q1A_DIR / f"series_{tag}.npz").is_file():
            rec = {"status": "missing_theta0_qed_archive", "tag": tag, "stopped": True}
            dump(MERGE / "qed_q3_covsign.json", rec)
            return rec
        if not (THETA_PI_DIR / f"series_{tag}.npz").is_file():
            rec = {"status": "missing_theta_pi_qed_archive", "tag": tag, "stopped": True}
            dump(MERGE / "qed_q3_covsign.json", rec)
            return rec
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=4.0)
    T0 = env.period
    omega0 = pf.OMEGA0
    a0 = analyze_one_angle("theta0", Q1A_DIR, T0, omega0)
    ah = analyze_one_angle("theta_h", Q3A_DIR, T0, omega0)
    api = analyze_one_angle("theta_pi", THETA_PI_DIR, T0, omega0)
    at = analyze_one_angle("theta_t", Q3B_DIR, T0, omega0)
    if not all(x.get("main_ok") for x in (a0, ah, api, at)):
        rec = {
            "status": "main_window_incomplete_no_spectral_claim",
            "verdict": "main_window_incomplete_no_spectral_claim",
            "answer": "至少一个角度的主窗数据不完整，不作谱学判断。",
            "answer_short": "主窗不完整",
            "stopped": True, "coverage": coverage, "angles": {
                "0": {k: a0.get(k) for k in ("main_ok", "t_over_T0")},
                "h": {k: ah.get(k) for k in ("main_ok", "t_over_T0")},
                "pi": {k: api.get(k) for k in ("main_ok", "t_over_T0")},
                "t": {k: at.get(k) for k in ("main_ok", "t_over_T0")},
            },
        }
        dump(MERGE / "qed_q3_covsign.json", rec)
        write_reports(rec)
        return rec
    j0, jh, jpi, jt = a0["h2_odd_c"], ah["h2_odd_c"], api["h2_odd_c"], at["h2_odd_c"]
    zd = jt - jh
    dphi = angdiff(np.angle(jt), np.angle(jh))
    dphi_deg = float(dphi * 180.0 / np.pi)
    amp_ratio = float(abs(jt) / max(abs(jh), 1e-30))
    gh = load_gh5()
    rel_vec = float(abs(zd - gh["jx_delta"]) / max(abs(zd), 1e-30))
    dphi_gh = abs(angdiff(np.angle(zd), np.angle(gh["jx_delta"])))
    dphi_gh_deg = float(dphi_gh * 180.0 / np.pi)
    agrees = bool(rel_vec < QED_GH5_REL and dphi_gh_deg < QED_GH5_DPHI_DEG)
    phase_sig = bool(abs(dphi_deg) >= PHASE_MARK_DEG)
    fit = four_fit(j0, jh, jpi, jt)
    h16 = []
    for iq, q in enumerate(range(1, 7)):
        dh = cz(ah["h1_h6_odd_jx"][iq]["jx"])
        dtv = cz(at["h1_h6_odd_jx"][iq]["jx"])
        dd = dtv - dh
        h16.append({"q": q, "jx": pack_c(dd), "abs_jx": float(abs(dd))})
    abs_h = np.array(ah["continuum_abs"], float)
    abs_t = np.array(at["continuum_abs"], float)
    qg = np.array(ah["continuum_q"], float)
    t2, even_h, odd_h = load_angle_eo(Q3A_DIR)
    _, even_t, odd_t = load_angle_eo(Q3B_DIR)
    n = min(len(t2), len(even_t))
    # independent Fourier already used; continuum difference uses each angle's own spectrum.
    abs_d = np.abs(np.array(at["continuum_abs"]) - np.array(ah["continuum_abs"]))
    # better: Fourier of difference is not available on a shared grid; keep spectral |J| overlay.
    win_note = {
        "front_h": "unavailable" if not ah["front_ok"] else pack_c(ah["h2_front_c"]),
        "front_t": "unavailable" if not at["front_ok"] else pack_c(at["h2_front_c"]),
        "back_h": "unavailable" if not ah["back_ok"] else pack_c(ah["h2_back_c"]),
        "back_t": "unavailable" if not at["back_ok"] else pack_c(at["h2_back_c"]),
        "back_pi_frozen": "unavailable" if not api["back_ok"] else pack_c(api["h2_back_c"]),
    }
    if phase_sig and agrees:
        verdict = "qed_confirms_covsign_phase_rotation_not_certified_HHG"
        answer = (
            "是：精确 QED 在固定平均位移、平均光子数以及相同初始 Var(X)、Var(P) 下，"
            "仅翻转 Cov(X,P) 符号，也给出磁序奇 H2 复相位的显著旋转，并与冻结 GH5 的复差向量"
            "落在预先声明的工程标记内。这支持压缩椭圆相位调控，但不是已认证 HHG，"
            "也不是经典不可复制性或反铁磁特殊性证明。"
        )
        answer_short = "是（QED 确认协方差符号引起的相位旋转，非认证 HHG）"
        status = "ok"
    elif phase_sig and not agrees:
        verdict = "qed_shows_phase_rotation_but_disagrees_with_gh5"
        answer = (
            "精确 QED 显示显著的 H2 相位旋转，但与冻结 GH5 的复差向量超出 20%/20° 工程标记。"
            "如实报告方法分歧。不自动扫描 nmax/GH7/g/r/α/k。"
        )
        answer_short = "有相位旋转，但与 GH5 定量不符"
        status = "ok_disagreement"
    elif agrees and not phase_sig:
        verdict = "qed_agrees_gh5_but_phase_below_mark"
        answer = (
            "QED 与 GH5 复差向量接近，但相位旋转未超过预先声明的 20° 标记。"
            "不能宣称显著相位调控。"
        )
        answer_short = "未达显著相位旋转标记"
        status = "ok_negative"
    else:
        verdict = "qed_does_not_confirm_covsign_phase_rotation"
        answer = (
            "否：在相同初始方差、仅协方差符号相反的两个压缩输入下，精确 QED 未确认"
            "显著的磁序奇 H2 复相位旋转。如实报告。不自动扩大参数。"
        )
        answer_short = "否（QED 未确认该协方差符号相位效应）"
        status = "ok_negative"
    f_plane = figure_four_angle(
        {"0": j0, "h": jh, "pi": jpi, "t": jt},
        {"0": gh["jx_0"], "h": gh["jx_h"], "pi": gh["jx_pi"], "t": gh["jx_t"]},
        FIG / "h2_four_angle_qed_vs_gh5.svg",
    )
    f_delta = figure_delta(zd, gh["jx_delta"], FIG / "h2_delta_perp_qed_vs_gh5.svg")
    f_cont = figure_cont(qg, abs_h, abs_t, abs_d, FIG / "h2_continuum_qed_covsign.svg")
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": status, "stopped": True, "verdict": verdict,
        "answer": answer, "answer_short": answer_short,
        "n_max": NMAX, "n_steps": N_FULL, "dt": env.t_total / N_FULL,
        "operator_order": "D(alpha)S(xi)|0>", "not_BSV": True,
        "frozen_not_rerun": True, "gh5_not_rerun": True,
        "source_hashes": hashes, "preflight": pf0, "coverage": coverage,
        "angles": {
            "0": {k: a0[k] for k in a0 if not k.endswith("_c") and k not in ("windows", "continuum_q", "continuum_abs")},
            "h": {k: ah[k] for k in ah if not k.endswith("_c") and k not in ("windows", "continuum_q", "continuum_abs")},
            "pi": {k: api[k] for k in api if not k.endswith("_c") and k not in ("windows", "continuum_q", "continuum_abs")},
            "t": {k: at[k] for k in at if not k.endswith("_c") and k not in ("windows", "continuum_q", "continuum_abs")},
        },
        "h2_main": {
            "jx_0": pack_c(j0), "jx_h": pack_c(jh), "jx_pi": pack_c(jpi), "jx_t": pack_c(jt),
            "jx_0_re": float(j0.real), "jx_0_im": float(j0.imag),
            "jx_h_re": float(jh.real), "jx_h_im": float(jh.imag),
            "jx_pi_re": float(jpi.real), "jx_pi_im": float(jpi.imag),
            "jx_t_re": float(jt.real), "jx_t_im": float(jt.imag),
            "abs_0": float(abs(j0)), "abs_h": float(abs(jh)),
            "abs_pi": float(abs(jpi)), "abs_t": float(abs(jt)),
            "phase_0_deg": float(np.angle(j0) * 180 / np.pi),
            "phase_h_deg": float(np.angle(jh) * 180 / np.pi),
            "phase_pi_deg": float(np.angle(jpi) * 180 / np.pi),
            "phase_t_deg": float(np.angle(jt) * 180 / np.pi),
            "delta": pack_c(zd), "delta_re": float(zd.real), "delta_im": float(zd.imag),
            "abs_delta": float(abs(zd)),
            "amp_ratio_t_over_h": amp_ratio,
            "dphi_rad": dphi, "dphi_deg": dphi_deg,
            "abs_even_h": ah["h2_even"]["abs_jx"],
            "abs_even_t": at["h2_even"]["abs_jx"],
            "odd_over_even_h": float(abs(jh) / max(ah["h2_even"]["abs_jx"], 1e-30)),
            "engineering_marks_not_physical_laws": True,
            "windows": win_note,
            "integer_Fourier_is_not_certified_HHG": True,
        },
        "qed_vs_gh5": {
            "rel_vec": rel_vec, "dphi_rad": dphi_gh, "dphi_deg": dphi_gh_deg,
            "agrees_complex": agrees,
            "threshold_rel": QED_GH5_REL, "threshold_dphi_deg": QED_GH5_DPHI_DEG,
            "gh5_delta": pack_c(gh["jx_delta"]),
            "gh5_abs_delta": gh["abs_delta"],
            "gh5_dphi_deg": float(gh["dphi_t_minus_h_rad"] * 180 / np.pi),
            "cannot_judge_by_magnitude_alone": True,
        },
        "qed_angle_fit": {k: fit[k] for k in fit if not k.startswith("_")},
        "gh5_angle_fit": {
            "B": pack_c(gh["B"]), "A": pack_c(gh["A"]), "C": pack_c(gh["C"]),
            "rel_resid": gh["rel_resid"],
        },
        "h1_h6_delta": h16,
        "continuum": {
            "q": qg.tolist(), "abs_h": abs_h.tolist(), "abs_t": abs_t.tolist(),
            "abs_diff_of_moduli": abs_d.tolist(),
            "integer_Fourier_is_not_certified_HHG": True,
        },
        "figures": {
            "four_angle": f_plane, "delta_perp": f_delta, "continuum": f_cont,
        },
        "physics_boundaries": {
            "initial_equal_variance_is_not_equal_Var_X_of_t": True,
            "not_BSV": True,
            "integer_Fourier_is_not_certified_HHG": True,
            "no_classical_unsimulability_claim": True,
            "no_AFM_specificity_vs_nonmagnetic_claim": True,
            "four_angle_aliasing": True,
            "did_not_auto_third_wave": True,
        },
        "did_not": [
            "rerun theta=0 QED", "rerun theta=pi QED", "rerun GH5",
            "Cn", "pure BSV", "GH7", "extra k", "nmax scan",
            "D2", "Floquet", "R2", "server", "auto third batch",
        ],
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(MERGE / "qed_q3_covsign.json", rec)
    write_reports(rec)
    return rec


def drive() -> int:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    MERGE.mkdir(parents=True, exist_ok=True)
    if bad:
        dump(MERGE / "qed_q3_covsign.json", {"status": "hash_abort", "hash_mismatch": bad, "stopped": True})
        progress("hash abort, no propagation")
        return 3
    pf0 = preflight()
    if pf0.get("status") in ("hash_abort", "preflight_abort") or not pf0.get("same_resource_ok", False):
        rec = {
            "status": pf0.get("status"), "stopped": True, "preflight": pf0,
            "verdict": "preflight_abort_no_qed",
            "answer": "nmax=8 截断严重破坏原定同资源比较，停止 QED。未自动升 nmax。",
            "answer_short": "预检失败，未开 QED",
            "coverage": {},
        }
        dump(MERGE / "qed_q3_covsign.json", rec)
        write_reports(rec)
        progress("preflight abort, no QED")
        return 3
    g3a = drive_batch("q3a")
    if not g3a.get("ok"):
        rec = {
            "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
            "status": "q3a_incomplete_stop",
            "verdict": "q3a_incomplete_no_spectral_claim",
            "answer": "Q3a 两路未完整覆盖 [2T0,4T0] 或数值异常。按批准边界停止，不启动 Q3b，不上服务器。",
            "answer_short": "Q3a 未完成，停止",
            "stopped": True, "analyze_ran": False, "preflight": pf0,
            "coverage": {"q3a_ok": False, "q3b_ok": False, "q3a_gate": g3a},
            "did_not_start_q3b": True,
        }
        dump(MERGE / "qed_q3_covsign.json", rec)
        write_reports(rec)
        progress("Q3a incomplete, stop before Q3b")
        return 1
    g3b = drive_batch("q3b")
    rec = analyze()
    progress(f"analyze done verdict={rec.get('verdict')}")
    return 0 if rec.get("status") not in (
        "hash_abort", "q3a_incomplete_stop", "q3b_incomplete_stop",
        "main_window_incomplete_no_spectral_claim") else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=str, default="")
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args(argv)
    MERGE.mkdir(parents=True, exist_ok=True)
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
        if not args.batch:
            raise SystemExit("--job requires --batch q3a|q3b")
        set_batch(args.batch)
        rec = run_one_job(args.job)
        return 0 if rec.get("status") in ("ok", "ok_fock_alarm", "timeout_partial") else 1
    if args.batch and not args.job and not args.analyze_only:
        hashes = source_hashes()
        if hash_abort(hashes):
            return 3
        gate = drive_batch(args.batch)
        return 0 if gate.get("ok") else 1
    if args.analyze_only:
        rec = analyze()
        print(json.dumps(jsonable({
            "status": rec.get("status"), "verdict": rec.get("verdict"),
            "answer_short": rec.get("answer_short"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") not in (
            "hash_abort", "q3a_incomplete_stop", "q3b_incomplete_stop",
            "missing_theta0_qed_archive", "missing_theta_pi_qed_archive",
            "main_window_incomplete_no_spectral_claim") else 1
    return drive()


if __name__ == "__main__":
    raise SystemExit(main())
