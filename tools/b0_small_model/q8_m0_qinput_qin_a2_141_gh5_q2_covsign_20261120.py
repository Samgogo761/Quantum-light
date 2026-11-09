#!/usr/bin/env python3
"""Qin_a2_1_4_1 GH5/MTEF Q2: squeeze-angle covariance-sign test.

New jobs only: DS(alpha=0.5, r=0.3, theta=pi/2 and 3pi/2), N=+/-1 (four paths).
Frozen theta=0 / theta=pi GH5 and all QED are read-only. No new QED, Cn, SV,
GH7, extra k, nmax, D2, Floquet, R2, or server.

These two inputs share <X>, <P>, <n>, Var(X), Var(P); only Cov(X,P) flips.
Must use correlated 2D Gauss-Hermite. Diagonal product GH is the wrong state.
Do not call mp.analytic_xp('ds') / gauss_hermite_nodes('ds').
Integer Fourier is not certified HHG. GH5 is not exact QED.
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
from numpy.polynomial.hermite import hermgauss

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_mtef_map_20261108 as mp
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_qin_a2_141_gh5_q2_covsign_20261120"
FIG = OUT / "figures"
THETA0_DIR = ROOT / "results/q_line/qinput_qin_a2_141_gh5_20261120"
THETAPI_DIR = ROOT / "results/q_line/qinput_qin_a2_141_gh5_theta_pi_20261120"
DOCS = ROOT / "docs/04_BSV量子光"
TASK_TAG = "20261120"
RECORD_DATE = "2026-10-23"
PROTOCOL = "Qin_a2_1_4_1"
ORDER = 5
N_NODES = ORDER * ORDER
N_FULL = 11154
SAMPLE_EVERY = 32
NMAX_FOCK_REF = 80
DRIVER_WALL_S = 600.0
JOB_WALL_S = 570.0
ALPHA = odd.ALPHA
R_SQ = odd.R_SQ
THETA_0 = 0.0
THETA_H = 0.5 * np.pi
THETA_PI = float(np.pi)
THETA_T = 1.5 * np.pi
JOBS = (
    ("ds_p_h", "ds", 1, THETA_H),
    ("ds_m_h", "ds", -1, THETA_H),
    ("ds_p_t", "ds", 1, THETA_T),
    ("ds_m_t", "ds", -1, THETA_T),
)
JOB_MAP = {tag: (kind, neel, theta) for tag, kind, neel, theta in JOBS}
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
    "q8_m0_qinput_shared_kk_mtef_map_20261108.py":
        "cc69ae6aa963cb4ac8af9bcc308c24950c6db01dc9b94ca6a971253f4bf1b34c",
    "q8_m0_qinput_shared_kk_mtef_compare_20261109.py":
        "391a75b37470e2670d9e782bf580191fa9c67d786515fd17df5f2569b192de19",
    "q8_m0_qinput_shared_kk_mtef_gh5_20261110.py":
        "3af3c05b33c304238b252d601124a2f6016e162b234aaa1821221e7aac4a3c14",
}
FROZEN = tuple(HERE / name for name in EXPECTED_HASHES)
REL_MARK = 0.20
DCOMPLEX_MARK = 0.50


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


def sample_js_for(n_steps: int):
    js = list(range(0, n_steps, SAMPLE_EVERY))
    if not js or js[-1] != n_steps - 1:
        js.append(n_steps - 1)
    return js


def make_env():
    return dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=4.0)


def make_bundle():
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    models, grounds = {}, {}
    for neel in (1, -1):
        models[neel] = sh.build_shared(
            geo, R0, k, neel, n_max=0, omega0=pf.OMEGA0, g=pf.G_ABS)
        grounds[neel] = odd.ground_pair_neel(geo, k, neel)
    env = make_env()
    return geo, k, models, grounds, env


def anomalous_M(var_x: float, var_p: float, cov_xp: float) -> complex:
    return 0.25 * ((var_x - var_p) + 2.0j * cov_xp)


def analytic_xp_theta(theta: float) -> dict:
    """Peierls X=a+a†, P=-i(a-a†). Vacuum Var(X)=1. Valid for any squeeze angle."""
    a = complex(ALPHA)
    mx, mp = 2.0 * a.real, 2.0 * a.imag
    r = float(R_SQ)
    th = float(theta)
    c2, s2 = float(np.cosh(2.0 * r)), float(np.sinh(2.0 * r))
    vx = c2 - s2 * np.cos(th)
    vp = c2 + s2 * np.cos(th)
    cxy = -s2 * np.sin(th)
    nbar = abs(a) ** 2 + np.sinh(r) ** 2
    Mm = -0.5 * np.exp(1j * th) * s2
    product_ok = abs(cxy) < 1e-14
    return {
        "kind": "ds", "alpha": pf.cpack(a), "r": r, "theta": th,
        "mean_X": mx, "mean_P": mp, "var_X": float(vx), "var_P": float(vp),
        "cov_XP": float(cxy), "mean_n_analytic": float(nbar),
        "mean_n_Weyl": 0.25 * (mx * mx + mp * mp + vx + vp) - 0.5,
        "M": pf.cpack(complex(Mm)),
        "M_from_xp": pf.cpack(anomalous_M(vx, vp, cxy)),
        "product_GH_valid": bool(product_ok),
        "requires_correlated_GH": bool(not product_ok),
        "did_not_call_mp_analytic_xp_ds": True,
    }


def spd_sqrt(V: np.ndarray) -> np.ndarray:
    w, u = np.linalg.eigh(np.asarray(V, float))
    if np.min(w) <= 0.0:
        raise ValueError(f"covariance not SPD, eigenvalues={w}")
    return (u * np.sqrt(w)) @ u.T


def gh5_nodes(theta: float) -> dict:
    """Affine-map the same (xi, eta) CRN through the full 2x2 covariance square root."""
    xi, w1 = hermgauss(ORDER)
    mom = analytic_xp_theta(theta)
    if abs(mom["cov_XP"]) < 1e-10:
        raise ValueError("Q2 nodes require nonzero Cov(X,P); diagonal GH is the wrong state")
    V = np.array([[mom["var_X"], mom["cov_XP"]], [mom["cov_XP"], mom["var_P"]]], float)
    S = spd_sqrt(V)
    mu = np.array([mom["mean_X"], mom["mean_P"]], float)
    nodes = []
    for i, x in enumerate(xi):
        for j, y in enumerate(xi):
            wt = float(w1[i] * w1[j] / np.pi)
            vec = mu + S @ (np.sqrt(2.0) * np.array([float(x), float(y)]))
            nodes.append({
                "xi": float(x), "eta": float(y), "weight": wt,
                "X": float(vec[0]), "P": float(vec[1]),
            })
    wsum = float(sum(n["weight"] for n in nodes))
    mx = sum(n["weight"] * n["X"] for n in nodes)
    mpv = sum(n["weight"] * n["P"] for n in nodes)
    vx = sum(n["weight"] * n["X"] ** 2 for n in nodes) - mx * mx
    vp = sum(n["weight"] * n["P"] ** 2 for n in nodes) - mpv * mpv
    cxy = sum(n["weight"] * n["X"] * n["P"] for n in nodes) - mx * mpv
    return {
        "kind": "ds", "theta": float(theta), "order": ORDER, "n_nodes": len(nodes),
        "weight_sum": wsum, "quad_mean_X": float(mx), "quad_mean_P": float(mpv),
        "quad_var_X": float(vx), "quad_var_P": float(vp), "quad_cov_XP": float(cxy),
        "target_mean_X": mom["mean_X"], "target_mean_P": mom["mean_P"],
        "target_var_X": mom["var_X"], "target_var_P": mom["var_P"],
        "target_cov_XP": mom["cov_XP"],
        "sqrtV": S.tolist(), "used_correlated_GH": True, "used_product_GH": False,
        "nodes": nodes, "analytic": mom,
        "did_not_call_mp_gauss_hermite_nodes_ds": True,
    }


def fock_xp_full(c) -> dict:
    """mp.fock_xp_moments plus Cov(X,P) and M=< (Delta a)^2 >. Local, does not edit frozen code."""
    base = mp.fock_xp_moments(c)
    cn = np.asarray(c, complex)
    mass = float(np.vdot(cn, cn).real)
    cn = cn / np.sqrt(mass)
    mean_a = 0.0j
    mean_aa = 0.0j
    for i in range(len(cn) - 1):
        mean_a += np.conj(cn[i]) * cn[i + 1] * np.sqrt(i + 1)
    for i in range(len(cn) - 2):
        mean_aa += np.conj(cn[i]) * cn[i + 2] * np.sqrt((i + 1) * (i + 2))
    Mm = mean_aa - mean_a ** 2
    cov = 2.0 * float(np.asarray(Mm).imag)
    base["cov_XP"] = cov
    base["M"] = pf.cpack(complex(Mm))
    base["mean_aa"] = pf.cpack(complex(mean_aa))
    return base


def one_traj(model, ek0, emk0, X0, P0, env, n_steps, js, clock):
    ek, emk = np.array(ek0, complex), np.array(emk0, complex)
    X, P = float(X0), float(P0)
    dt = env.t_total / n_steps
    want = set(int(j) for j in js)
    j_rows, t_rows = [], []
    max_n = 0.0
    for j in range(n_steps):
        if clock is not None:
            clock.check()
        s = env.s((j + 0.5) * dt)
        ek, emk, X, P, _ = mp.step_ehrenfest(ek, emk, X, P, model, s, dt, kick=True)
        nk = abs(np.linalg.norm(ek) - 1.0)
        nm = abs(np.linalg.norm(emk) - 1.0)
        max_n = max(max_n, nk, nm)
        if j in want:
            t = (j + 1) * dt
            jvec = mp.j_total(ek, emk, model, env.s(t), X)
            j_rows.append(jvec)
            t_rows.append(t)
    return np.asarray(j_rows, complex), np.asarray(t_rows, float), max_n, X, P


def ckpt_path(tag: str) -> Path:
    return OUT / f"ckpt_{tag}.npz"


def job_path(tag: str) -> Path:
    return OUT / f"job_{tag}.json"


def series_path(tag: str) -> Path:
    return OUT / f"series_{tag}.npz"


def save_ckpt(tag, kind, neel, theta, n_done, n_nodes, acc, wsum, max_n, t_arr) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc = np.asarray(acc, complex)
    np.savez_compressed(
        ckpt_path(tag),
        acc_re=acc.real, acc_im=acc.imag, wsum=float(wsum),
        n_done=int(n_done), n_nodes=int(n_nodes), max_n=float(max_n),
        t=np.asarray(t_arr, float), kind=str(kind), neel=int(neel), theta=float(theta),
    )


def load_ckpt(tag: str):
    p = ckpt_path(tag)
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    acc = np.asarray(z["acc_re"], float) + 1j * np.asarray(z["acc_im"], float)
    return {
        "n_done": int(z["n_done"]), "n_nodes": int(z["n_nodes"]),
        "acc": acc, "wsum": float(z["wsum"]), "max_n": float(z["max_n"]),
        "t": np.asarray(z["t"], float), "theta": float(z["theta"]) if "theta" in z.files else None,
    }


def load_job(tag: str):
    p = job_path(tag)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def job_finished_ok(tag: str) -> bool:
    rec = load_job(tag)
    return bool(rec and rec.get("status") == "ok" and series_path(tag).is_file())


def save_series(tag: str, t, j) -> None:
    j = np.asarray(j, complex)
    np.savez_compressed(series_path(tag), t=np.asarray(t, float), Jx=j[:, 0], Jy=j[:, 1])


def load_series(tag: str, directory: Path | None = None):
    d = OUT if directory is None else directory
    z = np.load(d / f"series_{tag}.npz")
    t = np.asarray(z["t"], float)
    j = np.stack([z["Jx"], z["Jy"]], axis=1)
    return t, j


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


def resource_row(theta: float) -> dict:
    ana = analytic_xp_theta(theta)
    raw = dyn.displaced_squeezed_coefficients(ALPHA, R_SQ, float(theta), NMAX_FOCK_REF)
    fock = fock_xp_full(raw)
    gh = None
    gh_err = None
    try:
        gh = gh5_nodes(theta)
    except ValueError as exc:
        gh_err = str(exc)
    row = {
        "theta": float(theta),
        "analytic": ana,
        "fock_nmax": NMAX_FOCK_REF,
        "fock": fock,
        "abs_d_mean_X": abs(fock["mean_X"] - ana["mean_X"]),
        "abs_d_mean_P": abs(fock["mean_P"] - ana["mean_P"]),
        "abs_d_var_X": abs(fock["var_X"] - ana["var_X"]),
        "abs_d_var_P": abs(fock["var_P"] - ana["var_P"]),
        "abs_d_cov_XP": abs(fock["cov_XP"] - ana["cov_XP"]),
        "abs_d_nbar": abs(fock["mean_n"] - ana["mean_n_analytic"]),
        "gh_error": gh_err,
    }
    if gh is not None:
        row.update({
            "gh5_quad_mean_X": gh["quad_mean_X"],
            "gh5_quad_mean_P": gh["quad_mean_P"],
            "gh5_quad_var_X": gh["quad_var_X"],
            "gh5_quad_var_P": gh["quad_var_P"],
            "gh5_quad_cov_XP": gh["quad_cov_XP"],
            "gh5_weight_sum": gh["weight_sum"],
            "gh5_used_correlated_GH": True,
            "abs_d_quad_cov": abs(gh["quad_cov_XP"] - ana["cov_XP"]),
            "abs_d_quad_var_X": abs(gh["quad_var_X"] - ana["var_X"]),
            "abs_d_quad_var_P": abs(gh["quad_var_P"] - ana["var_P"]),
        })
    return row


def row_ok(row: dict, need_corr: bool) -> tuple[bool, list]:
    reasons = []
    ana = row["analytic"]
    if abs(ana["mean_X"] - 1.0) > 1e-12:
        reasons.append("mean_X")
    if abs(ana["mean_P"]) > 1e-12:
        reasons.append("mean_P")
    if row["abs_d_mean_X"] > 1e-10 or row["abs_d_mean_P"] > 1e-10:
        reasons.append("fock_mean")
    if row["abs_d_var_X"] > 1e-8 or row["abs_d_var_P"] > 1e-8:
        reasons.append("fock_var")
    if row["abs_d_cov_XP"] > 1e-8:
        reasons.append("fock_cov")
    if row["abs_d_nbar"] > 1e-8:
        reasons.append("fock_nbar")
    if need_corr:
        if abs(ana["cov_XP"]) < 0.5:
            reasons.append("analytic_cov_too_small")
        if row.get("gh_error"):
            reasons.append("gh_error:" + str(row["gh_error"]))
        elif abs(row.get("gh5_quad_cov_XP", 0.0)) < 0.5:
            reasons.append("quad_cov_too_small_diagonal_GH")
        elif row.get("abs_d_quad_cov", 1.0) > 1e-10:
            reasons.append("quad_cov_mismatch")
        if abs(ana["var_X"] - ana["var_P"]) > 1e-10:
            reasons.append("varX_varP_not_equal")
    return (len(reasons) == 0), reasons


def collect_resources() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "resources", "source_hashes": hashes, "hash_abort": bad,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
        "alpha": pf.cpack(complex(ALPHA)), "r": float(R_SQ),
        "theta_h": THETA_H, "theta_t": THETA_T,
        "did_not_call_mp_analytic_xp_ds": True,
        "stopped": True,
    }
    if bad:
        rec["status"] = "hash_abort"
        dump(OUT / "resources.json", rec)
        return rec
    rec["theta0_ref"] = analytic_xp_theta(THETA_0)
    rec["theta_pi_ref"] = analytic_xp_theta(THETA_PI)
    rec["theta_h"] = resource_row(THETA_H)
    rec["theta_t"] = resource_row(THETA_T)
    ah, at = rec["theta_h"]["analytic"], rec["theta_t"]["analytic"]
    rec["same_mean_X"] = bool(abs(ah["mean_X"] - at["mean_X"]) < 1e-14)
    rec["same_mean_P"] = bool(abs(ah["mean_P"] - at["mean_P"]) < 1e-14)
    rec["same_nbar"] = bool(abs(ah["mean_n_analytic"] - at["mean_n_analytic"]) < 1e-14)
    rec["same_var_X"] = bool(abs(ah["var_X"] - at["var_X"]) < 1e-12)
    rec["same_var_P"] = bool(abs(ah["var_P"] - at["var_P"]) < 1e-12)
    rec["cov_sign_flipped"] = bool(abs(ah["cov_XP"] + at["cov_XP"]) < 1e-12 and abs(ah["cov_XP"]) > 0.5)
    ok_h, why_h = row_ok(rec["theta_h"], True)
    ok_t, why_t = row_ok(rec["theta_t"], True)
    rec["theta_h_ok"] = ok_h
    rec["theta_t_ok"] = ok_t
    rec["theta_h_fail"] = why_h
    rec["theta_t_fail"] = why_t
    rec["resource_match_ok"] = bool(
        rec["same_mean_X"] and rec["same_mean_P"] and rec["same_nbar"]
        and rec["same_var_X"] and rec["same_var_P"] and rec["cov_sign_flipped"]
        and ok_h and ok_t
    )
    dump(OUT / "resources.json", rec)
    progress(
        f"resources match={rec['resource_match_ok']} "
        f"nbar={ah['mean_n_analytic']:.6f} "
        f"varX={ah['var_X']:.6f} cov_h={ah['cov_XP']:.6f} cov_t={at['cov_XP']:.6f} "
        f"fail_h={why_h} fail_t={why_t}"
    )
    return rec


def run_one_job(tag: str) -> dict:
    if tag not in JOB_MAP:
        raise ValueError(tag)
    kind, neel, theta = JOB_MAP[tag]
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"tag": tag, "status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(job_path(tag), rec)
        return rec
    if job_finished_ok(tag):
        rec = load_job(tag)
        progress(f"{tag} already complete n_run={rec.get('n_run')}")
        return rec
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, models, grounds, env = make_bundle()
    gh = gh5_nodes(theta)
    nodes = gh["nodes"]
    if abs(gh["quad_cov_XP"]) < 0.5:
        rec = {"tag": tag, "status": "diag_cov_abort", "quad_cov_XP": gh["quad_cov_XP"], "stopped": True}
        dump(job_path(tag), rec)
        return rec
    ek0, emk0, mass = grounds[neel]
    model = models[neel]
    js = sample_js_for(N_FULL)
    ck = load_ckpt(tag)
    i0, acc, wsum, max_n, t_arr = 0, None, 0.0, 0.0, None
    if ck is not None:
        if ck.get("theta") is not None and abs(float(ck["theta"]) - float(theta)) > 1e-12:
            rec = {"tag": tag, "status": "ckpt_theta_mismatch", "stopped": True,
                   "ckpt_theta": ck.get("theta"), "job_theta": float(theta)}
            dump(job_path(tag), rec)
            return rec
        i0, acc, wsum, max_n, t_arr = ck["n_done"], ck["acc"], ck["wsum"], ck["max_n"], ck["t"]
        progress(f"{tag} resume n_done={i0}/{len(nodes)}")
    clock = dyn.WallClock(JOB_WALL_S)
    progress(f"{tag} start kind={kind} neel={neel} theta={theta:.6f} "
             f"cov={gh['quad_cov_XP']:.6f} nodes={len(nodes)} n_steps={N_FULL}")
    timed_out = False
    timeout_msg = None
    try:
        for i, node in enumerate(nodes):
            if i < i0:
                continue
            j_rows, t_rows, nrm, _, _ = one_traj(
                model, ek0, emk0, node["X"], node["P"], env, N_FULL, js, clock)
            w = float(node["weight"])
            acc = w * j_rows if acc is None else acc + w * j_rows
            wsum += w
            max_n = max(max_n, nrm)
            t_arr = t_rows
            save_ckpt(tag, kind, neel, theta, i + 1, len(nodes), acc, wsum, max_n, t_arr)
            if (i + 1) == len(nodes) or (i + 1) % 5 == 0:
                progress(f"{tag} node {i + 1}/{len(nodes)} wall={clock.elapsed():.1f}s")
    except TimeoutError as exc:
        timed_out = True
        timeout_msg = str(exc)
        progress(f"{tag} timeout {timeout_msg}")
    n_done = 0 if acc is None else int(load_ckpt(tag)["n_done"] if ckpt_path(tag).is_file() else 0)
    if timed_out or n_done < len(nodes):
        rec = {
            "tag": tag, "kind": kind, "neel": int(neel), "order": ORDER, "theta": float(theta),
            "status": "timeout_partial", "n_nodes": len(nodes), "n_done": n_done,
            "weight_sum": wsum, "max_abs_norm_minus_1": float(max_n),
            "wall_s": clock.elapsed(), "timeout_msg": timeout_msg,
            "n_max": 0, "protocol": PROTOCOL, "n_steps": N_FULL,
            "dt": env.t_total / N_FULL, "stopped": True, "electron": mass,
            "quad_cov_XP": gh["quad_cov_XP"],
        }
        dump(job_path(tag), rec)
        return rec
    avg = acc / (wsum + 1e-30)
    save_series(tag, t_arr, avg)
    rec = {
        "tag": tag, "kind": kind, "neel": int(neel), "order": ORDER, "theta": float(theta),
        "status": "ok", "n_nodes": len(nodes), "n_run": len(nodes),
        "n_done": len(nodes), "weight_sum": wsum,
        "max_abs_norm_minus_1": float(max_n),
        "wall_s": clock.elapsed(), "n_max": 0, "protocol": PROTOCOL,
        "n_steps": N_FULL, "dt": env.t_total / N_FULL,
        "n_samples": int(len(t_arr)), "t0": float(t_arr[0]), "t_end": float(t_arr[-1]),
        "electron": mass, "stopped": True,
        "wigner": analytic_xp_theta(theta),
        "quad_cov_XP": gh["quad_cov_XP"],
        "used_correlated_GH": True,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
        "ds_not_rescaled_coherent": True,
    }
    dump(job_path(tag), rec)
    progress(f"{tag} done wall={clock.elapsed():.1f}s")
    return rec


def slot_pack(jq):
    return {
        "Jx": pack_c(jq[:, 0]),
        "Jy": pack_c(jq[:, 1]),
        "abs": [float(np.linalg.norm(v)) for v in jq],
        "phase_x": [float(np.angle(v[0])) for v in jq],
        "phase_y": [float(np.angle(v[1])) for v in jq],
    }


def pack_win(name, t, omega0, T0, w0, w1, arrs):
    slots = {key: slot_pack(rect_jq(t, arr, omega0, w0, w1, np.arange(1, 7, dtype=float)))
             for key, arr in arrs.items()}
    return {
        "name": name, "t0": w0, "t1": w1,
        "t0_over_T0": w0 / T0, "t1_over_T0": w1 / T0,
        "complete": True,
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


def arrow(z, lim, x0, y0, w, h, color, label, lx, ly):
    cx, cy = x0 + w / 2, y0 + h / 2
    px, py = map_xy(z, lim, x0, y0, w, h)
    return (
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
        f'stroke="{color}" stroke-width="2"/>'
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>'
        f'<text x="{lx}" y="{ly}" font-size="11" fill="{color}">{label}</text>'
    )


def figure_four_plane(zs, path):
    lim = max(abs(z) for _, z, _ in zs) * 1.25 + 1e-30
    x0, y0, w, h = 90, 40, 740, 430
    body = svg_axes(x0, y0, w, h, "Re Jx (H2 odd)", "Im Jx (H2 odd)",
                    "H2 odd Jx vs squeeze angle (GH5)")
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    lx = 80
    for label, z, color in zs:
        body += arrow(z, lim, x0, y0, w, h, color, label, lx, 36)
        lx += 130
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "same alpha, nbar; pi/2 and 3pi/2 share Var(X),Var(P); GH5 not QED; not certified HHG</text>"
    )
    return write_svg(path, body)


def figure_windows(rows, path, title):
    lim = max(abs(z) for _, z, _ in rows) * 1.25 + 1e-30
    x0, y0, w, h = 90, 40, 740, 430
    body = svg_axes(x0, y0, w, h, "Re δ⊥ Jx (H2 odd)", "Im δ⊥ Jx (H2 odd)", title)
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    lx = 80
    for label, z, color in rows:
        body += arrow(z, lim, x0, y0, w, h, color, label, lx, 36)
        lx += 150
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "delta_complex uses Jx. Engineering marks are not physical laws.</text>"
    )
    return write_svg(path, body)


def figure_cont(q, series, path):
    xmin, xmax = float(q[0]), float(q[-1])
    ys = np.concatenate([y for _, y, _ in series])
    ymin, ymax = 0.0, float(np.max(ys)) * 1.12 + 1e-30
    x0, y0p, w, h = 70, 40, 800, 430
    body = svg_axes(x0, y0p, w, h, "Ω / ω0", "|J^o(Ω)|",
                    "Main-window continuum |J_odd|")

    def poly(xs, ys_, color, sw):
        pts = []
        for x, y in zip(xs, ys_):
            px = x0 + (float(x) - xmin) / (xmax - xmin) * w
            py = y0p + (1.0 - (float(y) - ymin) / max(ymax - ymin, 1e-30)) * h
            pts.append(f"{px:.2f},{py:.2f}")
        return (f'<polyline fill="none" stroke="{color}" stroke-width="{sw}" '
                f'points="{" ".join(pts)}"/>')

    lx = 80
    for label, y, color in series:
        body += poly(q, y, color, 1.4)
        body += f'<text x="{lx}" y="36" font-size="11" fill="{color}">{label}</text>'
        lx += 90
    for n in range(1, 7):
        px = x0 + (n - xmin) / (xmax - xmin) * w
        body += (f'<line x1="{px:.1f}" y1="{y0p}" x2="{px:.1f}" y2="{y0p+h}" '
                 f'stroke="#8a9096" stroke-dasharray="3 4" stroke-width="0.8"/>')
    return write_svg(path, body)


def fit_angle_model(thetas, zs):
    thetas = np.asarray(thetas, float)
    zs = np.asarray(zs, complex)
    X = np.column_stack([np.ones(len(thetas)), np.exp(1j * thetas), np.exp(-1j * thetas)])
    coef, _, _, _ = np.linalg.lstsq(X, zs, rcond=None)
    fit = X @ coef
    resid = zs - fit
    den = float(np.linalg.norm(zs))
    return {
        "B": pack_c(coef[0]), "A": pack_c(coef[1]), "C": pack_c(coef[2]),
        "fit": pack_c(fit), "resid": pack_c(resid),
        "rel_resid": float(np.linalg.norm(resid) / max(den, 1e-30)),
        "note": "four-point least squares for B+A e^{iθ}+C e^{-iθ}; not a microscopic χ(2)",
        "cannot_exclude_higher_theta_harmonics": True,
    }


def write_reports(rec: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    h2 = rec["h2_main"]
    res = rec["resources"]
    ah = res["theta_h"]["analytic"]
    at = res["theta_t"]["analytic"]
    v = rec["verdict"]
    ans = rec["answer"]
    full = []
    full.append("# 量子光多周期磁序响应：Q2 正交协方差符号反转（GH5/MTEF）\n")
    full.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。")
    full.append("冻结 θ=0、θ=π 的 GH5 与全部精确 QED，只读不重算。本轮仅四路新 GH5：")
    full.append("DS(α=0.5, r=0.3, θ=π/2 与 3π/2)，各 N=±1。")
    full.append("算符顺序 D(α)S(ξ)|0⟩。相关二维 GH5，禁止对角乘积节点。`stopped = true`。\n")
    full.append(f"**结论：{ans}**\n")
    full.append("## 输入资源（固定位移、光子数，且 Var(X)=Var(P)）\n")
    full.append("| 量 | θ=π/2 | θ=3π/2 |\n|---|---|---|")
    full.append(f"| ⟨X⟩ | {ah['mean_X']:.6f} | {at['mean_X']:.6f} |")
    full.append(f"| ⟨P⟩ | {ah['mean_P']:.6f} | {at['mean_P']:.6f} |")
    full.append(f"| ⟨n⟩ | {ah['mean_n_analytic']:.8f} | {at['mean_n_analytic']:.8f} |")
    full.append(f"| Var(X) | {ah['var_X']:.6f} | {at['var_X']:.6f} |")
    full.append(f"| Var(P) | {ah['var_P']:.6f} | {at['var_P']:.6f} |")
    full.append(f"| Cov(X,P) | {ah['cov_XP']:.6f} | {at['cov_XP']:.6f} |")
    full.append(f"| 相关 GH | `{res['resource_match_ok']}` | 符号相反的非零协方差 |\n")
    full.append("θ=0、θ=π 只读对照的 Var(X) 对调，Cov=0；本轮两态的初始 Var(X) 完全相同。\n")
    full.append("## 主窗 H2 磁序奇复源（Jx）\n")
    full.append("| θ | Jx | |Jx| | arg | 2D |J| |\n|---|---|---|---|---|")
    for key, lab in (("0", "0"), ("h", "π/2"), ("pi", "π"), ("t", "3π/2")):
        full.append(
            f"| {lab} | {h2[f'jx_{key}_re']:.3e}{h2[f'jx_{key}_im']:+.3e}i | "
            f"{h2[f'abs_jx_{key}']:.3e} | {h2[f'phase_{key}']:.3f} rad | {h2[f'abs_2d_{key}']:.3e} |"
        )
    full.append(
        f"\n\\(\\delta_\\perp\\mathcal J_{{2,x}}^o="
        f"{h2['delta_jx_re']:.3e}{h2['delta_jx_im']:+.3e}i\\)，"
        f"模长 {h2['abs_delta_jx']:.3e}。"
        f" rel_to_max={h2['rel_to_max']:.3f}；δ_complex={h2['delta_complex']:.3f}。\n"
    )
    full.append("| 窗 | |δ⊥ J_{2,x}| | arg |\n|---|---|---|")
    full.append(f"| 前 [1,3]T0 | {h2['abs_front']:.3e} | {h2['phase_front']:.3f} rad |")
    full.append(f"| 主 [2,4]T0 | {h2['abs_delta_jx']:.3e} | {h2['phase_delta']:.3f} rad |")
    full.append(f"| 后 [3,5]T0 | {h2['abs_back']:.3e} | {h2['phase_back']:.3f} rad |\n")
    full.append("## 偶/奇绝对幅度（主窗 H2，Jx）\n")
    full.append("| θ | |J_even,x| | |J_odd,x| | |odd|/|even| |\n|---|---|---|---|")
    for key, lab in (("0", "0"), ("h", "π/2"), ("pi", "π"), ("t", "3π/2")):
        ev = h2[f"abs_even_jx_{key}"]
        odv = h2[f"abs_jx_{key}"]
        full.append(f"| {lab} | {ev:.3e} | {odv:.3e} | {odv / max(ev, 1e-30):.3e} |")
    full.append("\n磁序奇分量远弱于偶分量；相对调控不能写成巨大的实际磁性输出。\n")
    fit = rec["angle_fit"]
    full.append("## 四点角向结构（初步测试，不是 χ⁽²⁾）\n")
    full.append("拟合 \\(J(\\theta)\\approx B+A e^{i\\theta}+C e^{-i\\theta}\\)，"
                f"相对残差 {fit['rel_resid']:.3f}。四点不能排除更高 θ 谐波混叠。\n")
    full.append("工程分类标记（不是物理定律）："
                f"`rel≥{REL_MARK}` 且 `δ_complex<{DCOMPLEX_MARK}` 记为明显且窗口不太散。\n")
    full.append(f"verdict `{v}`。\n")
    full.append("## 解释边界\n")
    full.append("- 本轮是 GH5/MTEF，不是精确 QED。")
    full.append("- 自由场 \\(C^S_{\\pi/2}-C^S_{3\\pi/2}=-2\\sinh(2r)\\sin[\\omega_0(t+t')]\\) 只是输入解释，不是已证明的电子响应核。")
    full.append("- 输入仍是位移压缩态 α=0.5，不是纯 BSV。")
    full.append("- 两周期矩形 Fourier ≠ 已认证 HHG。未改选 H1–H6 以外的阶次。")
    full.append("- 未自动启动精确 QED、Cn、SV、GH7、额外 k 或服务器。\n")
    full.append("完成后停机。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q2_GH5_covsign_20261120.md").write_text(
        "\n".join(full) + "\n", encoding="utf-8")
    one = []
    one.append("# 量子光多周期磁序响应：Q2 协方差符号（一页）\n")
    one.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。冻结 θ=0/π GH5 与 QED。`stopped = true`。\n")
    one.append(f"**{ans}**\n")
    one.append("| 量 | 结果 |\n|---|---|")
    one.append(f"| 资源 | ⟨X⟩=1, ⟨n⟩={ah['mean_n_analytic']:.4f}；Var(X)=Var(P)={ah['var_X']:.4f} |")
    one.append(f"| Cov(X,P) | {ah['cov_XP']:.4f} ↔ {at['cov_XP']:.4f} |")
    one.append(f"| 主窗 \\(\\lvert\\delta_\\perp J_{{2,x}}^o\\rvert\\) | {h2['abs_delta_jx']:.3e} |")
    one.append(f"| rel_to_max / δ_complex | {h2['rel_to_max']:.3f} / {h2['delta_complex']:.3f} |")
    one.append(f"| 分类 | `{v}` |")
    one.append("| 新 QED / 服务器 | 现在不启动 |\n")
    one.append("**问题：** 在相同平均场、相同平均光子数、相同 Var(X) 与 Var(P) 下，仅翻转 Cov(X,P) 符号，"
               "GH5 中的 Néel-odd H2 复电流是否明显不同？\n")
    one.append(f"**{rec['answer_short']}**\n")
    one.append("停机。不自动开精确 QED。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q2_GH5_covsign_一页_20261120.md").write_text(
        "\n".join(one) + "\n", encoding="utf-8")


def analyze() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(OUT / "gh5_q2_covsign.json", rec)
        return rec
    missing = [tag for tag, _, _, _ in JOBS if not job_finished_ok(tag)]
    if missing:
        rec = {"status": "missing_jobs", "missing": missing, "stopped": True,
               "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                              for tag, _, _, _ in JOBS}}
        dump(OUT / "gh5_q2_covsign.json", rec)
        return rec
    for directory, tag in ((THETA0_DIR, "ds_p"), (THETA0_DIR, "ds_m"),
                           (THETAPI_DIR, "ds_p"), (THETAPI_DIR, "ds_m")):
        if not (directory / f"series_{tag}.npz").is_file():
            rec = {"status": "missing_frozen_archive", "dir": str(directory), "tag": tag, "stopped": True}
            dump(OUT / "gh5_q2_covsign.json", rec)
            return rec
    res = collect_resources()
    if not res.get("resource_match_ok"):
        rec = {"status": "resource_mismatch", "resources": res, "stopped": True}
        dump(OUT / "gh5_q2_covsign.json", rec)
        return rec
    env = make_env()
    T0 = env.period
    omega0 = pf.OMEGA0
    t_h_p, j_h_p = load_series("ds_p_h")
    t_h_m, j_h_m = load_series("ds_m_h")
    t_t_p, j_t_p = load_series("ds_p_t")
    t_t_m, j_t_m = load_series("ds_m_t")
    t0_p, j0_p = load_series("ds_p", THETA0_DIR)
    t0_m, j0_m = load_series("ds_m", THETA0_DIR)
    tpi_p, jpi_p = load_series("ds_p", THETAPI_DIR)
    tpi_m, jpi_m = load_series("ds_m", THETAPI_DIR)
    t = t_h_p
    for label, tt in (("h_m", t_h_m), ("t_p", t_t_p), ("t_m", t_t_m),
                      ("0_p", t0_p), ("0_m", t0_m), ("pi_p", tpi_p), ("pi_m", tpi_m)):
        if not np.allclose(tt, t, rtol=0, atol=1e-9):
            rec = {"status": "time_grid_mismatch", "tag": label, "stopped": True}
            dump(OUT / "gh5_q2_covsign.json", rec)
            return rec
    even_h, odd_h = odd.even_odd(j_h_p, j_h_m)
    even_t, odd_t = odd.even_odd(j_t_p, j_t_m)
    even_0, odd_0 = odd.even_odd(j0_p, j0_m)
    even_pi, odd_pi = odd.even_odd(jpi_p, jpi_m)
    d_odd = odd_t - odd_h
    d_even = even_t - even_h
    arrs = {
        "J_odd_0": odd_0, "J_odd_h": odd_h, "J_odd_pi": odd_pi, "J_odd_t": odd_t,
        "dJ_odd": d_odd, "J_even_0": even_0, "J_even_h": even_h,
        "J_even_pi": even_pi, "J_even_t": even_t, "dJ_even": d_even,
    }
    win_main = (2.0 * T0, 4.0 * T0)
    win_front = (1.0 * T0, 3.0 * T0)
    win_back = (3.0 * T0, 5.0 * T0)
    main = pack_win("plateau_center_2T", t, omega0, T0, *win_main, arrs)
    front = pack_win("front_2T", t, omega0, T0, *win_front, arrs)
    back = pack_win("back_2T", t, omega0, T0, *win_back, arrs)
    qs_c = np.linspace(0.0, 6.5, 326)
    cont_h = rect_jq(t, odd_h, omega0, *win_main, qs_c)
    cont_t = rect_jq(t, odd_t, omega0, *win_main, qs_c)
    cont_d = rect_jq(t, d_odd, omega0, *win_main, qs_c)
    absh = np.array([float(np.linalg.norm(v)) for v in cont_h], float)
    abst = np.array([float(np.linalg.norm(v)) for v in cont_t], float)
    absd = np.array([float(np.linalg.norm(v)) for v in cont_d], float)
    peak_i = int(np.argmax(absd[1:])) + 1

    def take(winobj, key):
        return h2_jx(winobj["slots"], key)

    z0 = take(main, "J_odd_0")
    zh = take(main, "J_odd_h")
    zpi = take(main, "J_odd_pi")
    zt = take(main, "J_odd_t")
    zd = take(main, "dJ_odd")
    zf = take(front, "dJ_odd")
    zb = take(back, "dJ_odd")
    e0 = take(main, "J_even_0")
    eh = take(main, "J_even_h")
    epi = take(main, "J_even_pi")
    et = take(main, "J_even_t")
    den = max(zh[2], zt[2], 1e-30)
    rel = float(zd[2] / den)
    dcomplex = float(abs(zb[0] - zf[0]) / max(zd[2], 1e-30))
    dphi = abs(angdiff(zt[3], zh[3]))
    z0pi = zpi[0] - z0[0]
    fit = fit_angle_model([THETA_0, THETA_H, THETA_PI, THETA_T],
                          [z0[0], zh[0], zpi[0], zt[0]])
    if rel >= REL_MARK and dcomplex < DCOMPLEX_MARK:
        verdict = "covsign_controls_h2_odd_in_GH5_not_qed"
        answer = (
            "是（仅限当前 GH5/MTEF）：在相同平均位移、相同平均光子数、相同 Var(X) 与 Var(P) 下，"
            "仅翻转 Cov(X,P) 符号，主窗 H2 磁序奇 Jx 出现可辨识且窗口不太散的复响应变化。"
            "这说明当前响应不只读取初始 X 方向涨落大小。这不是精确 QED，也不是已认证 HHG。"
        )
        answer_short = "是（GH5 对 Cov 符号敏感，非 QED、非认证 HHG）"
    elif rel >= REL_MARK:
        verdict = "h2_covsign_change_without_window_consistency"
        answer = (
            "主窗 H2 磁序奇 Jx 在 θ=π/2 与 3π/2 之间有明显模长差，但前后窗复差向量不够一致。"
            "不能把该差异升级为稳定的协方差符号调控。接受窗口不稳，不改参数，不自动开 QED。"
        )
        answer_short = "有变化但窗口不一致"
    else:
        verdict = "covsign_h2_response_weak_or_negative"
        answer = (
            "否：在相同 Var(X)、Var(P) 下仅翻转 Cov(X,P) 符号，未能在 GH5 中给出"
            "超过工程标记的、窗口稳健的 H2 磁序奇复电流调控。"
            "这与主要读取 cosθ 型涨落（Var X/P 对调）的图像相容，或存在模型响应对称性。"
            "如实接受该结果。不改 r/g/α/k，不自动开 QED。"
        )
        answer_short = "否（弱或负结果：可能主要读 Var(X)/Var(P)）"
    FIG.mkdir(parents=True, exist_ok=True)
    f_plane = figure_four_plane(
        [("θ=0", z0[0], "#4c6b8a"), ("θ=π/2", zh[0], "#7a4ea3"),
         ("θ=π", zpi[0], "#c45c26"), ("θ=3π/2", zt[0], "#2f9e44"),
         ("δ⊥", zd[0], "#c9a24a")],
        FIG / "h2_complex_plane_four_angles.svg",
    )
    f_win = figure_windows(
        [("front", zf[0], "#4c6b8a"), ("main", zd[0], "#2f9e44"), ("back", zb[0], "#c45c26")],
        FIG / "h2_dperp_three_windows.svg",
        "H2 δ⊥ three windows (Jx)",
    )
    f_cont = figure_cont(
        qs_c,
        [("θ=π/2", absh, "#7a4ea3"), ("θ=3π/2", abst, "#2f9e44"), ("δ⊥", absd, "#c9a24a")],
        FIG / "h2_continuum_covsign.svg",
    )
    jobs = {tag: load_job(tag) for tag, _, _, _ in JOBS}
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "ok", "stopped": True, "verdict": verdict,
        "answer": answer, "answer_short": answer_short,
        "n_max": 0, "order": ORDER, "n_nodes": N_NODES, "n_steps": N_FULL,
        "dt": env.t_total / N_FULL,
        "theta0_archive": str(THETA0_DIR.as_posix()),
        "thetapi_archive": str(THETAPI_DIR.as_posix()),
        "frozen_not_rerun": True, "qed_not_rerun": True,
        "operator_order": "D(alpha)S(xi)|0>",
        "used_correlated_GH": True,
        "source_hashes": hashes, "hash_abort": [],
        "resources": res, "windows": {"main": main, "front": front, "back": back},
        "h2_main": {
            "jx_0": pack_c(z0[0]), "jx_h": pack_c(zh[0]),
            "jx_pi": pack_c(zpi[0]), "jx_t": pack_c(zt[0]), "jx_delta": pack_c(zd[0]),
            "jx_0_re": float(z0[0].real), "jx_0_im": float(z0[0].imag),
            "jx_h_re": float(zh[0].real), "jx_h_im": float(zh[0].imag),
            "jx_pi_re": float(zpi[0].real), "jx_pi_im": float(zpi[0].imag),
            "jx_t_re": float(zt[0].real), "jx_t_im": float(zt[0].imag),
            "abs_jx_0": z0[2], "abs_jx_h": zh[2], "abs_jx_pi": zpi[2], "abs_jx_t": zt[2],
            "abs_delta_jx": zd[2],
            "abs_2d_0": z0[4], "abs_2d_h": zh[4], "abs_2d_pi": zpi[4], "abs_2d_t": zt[4],
            "abs_2d_delta": zd[4],
            "phase_0": z0[3], "phase_h": zh[3], "phase_pi": zpi[3], "phase_t": zt[3],
            "phase_delta": zd[3],
            "dphi_t_minus_h_rad": dphi,
            "delta_jx_re": float(zd[0].real), "delta_jx_im": float(zd[0].imag),
            "abs_delta_0_pi": float(abs(z0pi)),
            "rel_to_max": rel, "delta_complex": dcomplex,
            "abs_front": zf[2], "phase_front": zf[3],
            "abs_back": zb[2], "phase_back": zb[3],
            "abs_even_jx_0": e0[2], "abs_even_jx_h": eh[2],
            "abs_even_jx_pi": epi[2], "abs_even_jx_t": et[2],
            "engineering_marks_not_physical_laws": True,
            "rel_mark": REL_MARK, "dcomplex_mark": DCOMPLEX_MARK,
            "note": "rel_to_max and delta_complex use H2 Jx, not 2D |J|.",
        },
        "angle_fit": fit,
        "continuum": {
            "q": qs_c.tolist(),
            "abs_Jodd_h": absh.tolist(), "abs_Jodd_t": abst.tolist(),
            "abs_dJodd": absd.tolist(), "peak_q_dJ": float(qs_c[peak_i]),
            "peak_abs_dJ": float(absd[peak_i]),
        },
        "jobs": {tag: {k: jobs[tag].get(k) for k in
                       ("status", "n_run", "n_done", "wall_s", "max_abs_norm_minus_1",
                        "weight_sum", "kind", "neel", "theta", "quad_cov_XP")}
                 for tag, _, _, _ in JOBS},
        "figures": {"four_angles": f_plane, "three_windows": f_win, "continuum": f_cont},
        "physics_boundaries": {
            "GH5_is_not_exact_QED": True,
            "free_field_covariance_is_not_proven_response_kernel": True,
            "displaced_squeezed_not_pure_BSV": True,
            "integer_Fourier_is_not_certified_HHG": True,
            "did_not_retarget_away_from_H2": True,
            "four_point_fit_is_not_chi2": True,
        },
        "did_not": [
            "new QED", "rerun theta=0 GH5", "rerun theta=pi GH5",
            "Cn", "squeezed vacuum", "GH7", "extra k", "nmax scan",
            "D2", "Floquet", "R2 positive-P optimization", "server",
            "increase r/g/alpha",
        ],
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "gh5_q2_covsign.json", rec)
    write_reports(rec)
    return rec


def drive() -> int:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        dump(OUT / "gh5_q2_covsign.json", {"status": "hash_abort", "hash_mismatch": bad, "stopped": True})
        progress("hash abort, no propagation")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    res = collect_resources()
    if not res.get("resource_match_ok"):
        dump(OUT / "gh5_q2_covsign.json", {"status": "resource_mismatch", "resources": res, "stopped": True})
        progress("resource mismatch, no propagation")
        return 3
    t0 = time.perf_counter()
    procs, logs = {}, {}
    for tag, kind, neel, theta in JOBS:
        if job_finished_ok(tag):
            progress(f"drive skip {tag} already ok")
            continue
        logf = (OUT / f"log_{tag}.txt").open("a", encoding="utf-8")
        logs[tag] = logf
        envp = os.environ.copy()
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            envp[k] = "1"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", tag]
        progress(f"launch {tag} theta={theta:.6f}")
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
    codes = {tag: (procs[tag].returncode if tag in procs else 0) for tag, _, _, _ in JOBS}
    progress(f"jobs return {codes} driver_wall={time.perf_counter()-t0:.1f}s")
    if all(job_finished_ok(tag) for tag, _, _, _ in JOBS):
        rec = analyze()
        progress(f"analyze done verdict={rec.get('verdict')}")
        return 0
    rec = {
        "status": "timeout_partial", "stopped": True,
        "driver_wall_s": time.perf_counter() - t0, "returncodes": codes,
        "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                       for tag, _, _, _ in JOBS},
        "jobs_n_done": {tag: None if load_job(tag) is None else load_job(tag).get("n_done")
                        for tag, _, _, _ in JOBS},
        "did_not_analyze_incomplete": True, "qed_not_rerun": True,
    }
    dump(OUT / "gh5_q2_covsign.json", rec)
    progress("incomplete, no full analyze")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--resources-only", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.resources_only:
        rec = collect_resources()
        print(json.dumps(jsonable({
            "status": rec.get("status"),
            "resource_match_ok": rec.get("resource_match_ok"),
            "same_nbar": rec.get("same_nbar"),
            "same_var_X": rec.get("same_var_X"),
            "cov_sign_flipped": rec.get("cov_sign_flipped"),
            "theta_h_fail": rec.get("theta_h_fail"),
            "theta_t_fail": rec.get("theta_t_fail"),
            "cov_h": None if "theta_h" not in rec else rec["theta_h"]["analytic"]["cov_XP"],
            "cov_t": None if "theta_t" not in rec else rec["theta_t"]["analytic"]["cov_XP"],
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") == "resources" and rec.get("resource_match_ok") else 3
    if args.job:
        rec = run_one_job(args.job)
        return 0 if rec.get("status") in ("ok", "timeout_partial") else 1
    if args.analyze_only:
        rec = analyze()
        print(json.dumps(jsonable({
            "status": rec.get("status"), "verdict": rec.get("verdict"),
            "answer_short": rec.get("answer_short"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") == "ok" else 1
    return drive()


if __name__ == "__main__":
    raise SystemExit(main())
