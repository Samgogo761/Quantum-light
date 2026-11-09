#!/usr/bin/env python3
"""Qin_a2_1_4_1 GH5/MTEF Q5: four-angle trajectories at eta=0.5 and 1.5.

New jobs only: eta=0.5/1.5, theta=0/pi, N=+/-1 (eight GH5 paths).
Frozen: eta=1 four-angle GH5/QED; Q4 eta=0.5/1.5 at theta=pi/2, 3pi/2.
theta=0/pi are diagonal product GH (Cov=0, swapped Var X/P). Do not reuse
Q4 diag_cov_abort. No new QED, extra eta/theta, Cn, BSV, GH7, k, nmax, D2, R2.

Test strict J=F(eta)G(theta) versus weak J=B(eta)+F(eta)G(theta).
Four equal angles alias e^{i5theta} with e^{itheta}. Not certified HHG.
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
import q8_m0_qinput_qin_a2_141_gh5_q2_covsign_20261120 as q2
import q8_m0_qinput_qin_a2_141_gh5_q4_eta_20261120 as q4
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
from q8_m0_model import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_qin_a2_141_gh5_q5_fourangle_20261120"
FIG = OUT / "figures"
THETA0_DIR = ROOT / "results/q_line/qinput_qin_a2_141_gh5_20261120"
THETAPI_DIR = ROOT / "results/q_line/qinput_qin_a2_141_gh5_theta_pi_20261120"
Q2_DIR = ROOT / "results/q_line/qinput_qin_a2_141_gh5_q2_covsign_20261120"
Q4_DIR = ROOT / "results/q_line/qinput_qin_a2_141_gh5_q4_eta_20261120"
DOCS = ROOT / "docs/04_BSV量子光"
TASK_TAG = "20261120"
RECORD_DATE = "2026-10-24"
PROTOCOL = "Qin_a2_1_4_1"
ORDER = q2.ORDER
N_NODES = q2.N_NODES
N_FULL = q2.N_FULL
SAMPLE_EVERY = q2.SAMPLE_EVERY
DRIVER_WALL_S = 600.0
JOB_WALL_S = 560.0
THETA_0 = 0.0
THETA_H = 0.5 * np.pi
THETA_PI = float(np.pi)
THETA_T = 1.5 * np.pi
THETAS = (THETA_0, THETA_H, THETA_PI, THETA_T)
THETA_KEYS = ("0", "h", "pi", "t")
ETA_NEW = (0.5, 1.5)
ETA_ALL = (0.5, 1.0, 1.5)
NBAR_TARGET = 0.25 + float(np.sinh(0.3) ** 2)
STRICT_REL_MARK = 0.20
WEAK_REL_MARK = 0.10
ETA_COLOR = {0.5: "#7a4ea3", 1.0: "#2f9e44", 1.5: "#c45c26"}
JOBS = (
    ("e05_p_0", 0.5, 1, THETA_0),
    ("e05_m_0", 0.5, -1, THETA_0),
    ("e05_p_pi", 0.5, 1, THETA_PI),
    ("e05_m_pi", 0.5, -1, THETA_PI),
    ("e15_p_0", 1.5, 1, THETA_0),
    ("e15_m_0", 1.5, -1, THETA_0),
    ("e15_p_pi", 1.5, 1, THETA_PI),
    ("e15_m_pi", 1.5, -1, THETA_PI),
)
JOB_MAP = {tag: (eta, neel, theta) for tag, eta, neel, theta in JOBS}
BATCHES = (
    (0.5, ("e05_p_0", "e05_m_0", "e05_p_pi", "e05_m_pi")),
    (1.5, ("e15_p_0", "e15_m_0", "e15_p_pi", "e15_m_pi")),
)
EXPECTED_HASHES = dict(q2.EXPECTED_HASHES)
FROZEN = tuple(HERE / name for name in EXPECTED_HASHES)
jsonable = q2.jsonable
cz = q2.cz
pack_c = q2.pack_c
wrap = q2.wrap
angdiff = q2.angdiff
sample_js_for = q2.sample_js_for
analytic_xp_theta = q2.analytic_xp_theta
one_traj = q2.one_traj
rect_jq = q2.rect_jq
slot_pack = q2.slot_pack
h2_jx = q2.h2_jx
fit_angle_model = q2.fit_angle_model
pack_win = q2.pack_win


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


def gh5_nodes_diag(theta: float) -> dict:
    """Product GH for theta=0/pi. Abort if covariance is not diagonal or vars equal."""
    mom = analytic_xp_theta(theta)
    if abs(mom["cov_XP"]) > 1e-8:
        raise ValueError(f"diag GH requires Cov~0, got {mom['cov_XP']}")
    if abs(mom["var_X"] - mom["var_P"]) < 0.5:
        raise ValueError("diag GH requires |VarX-VarP|>=0.5")
    xi, w1 = hermgauss(ORDER)
    sx, sp = np.sqrt(mom["var_X"]), np.sqrt(mom["var_P"])
    nodes = []
    for i, x in enumerate(xi):
        for j, y in enumerate(xi):
            wt = float(w1[i] * w1[j] / np.pi)
            nodes.append({
                "xi": float(x), "eta": float(y), "weight": wt,
                "X": mom["mean_X"] + sx * np.sqrt(2.0) * float(x),
                "P": mom["mean_P"] + sp * np.sqrt(2.0) * float(y),
            })
    wsum = float(sum(n["weight"] for n in nodes))
    mx = sum(n["weight"] * n["X"] for n in nodes)
    mpv = sum(n["weight"] * n["P"] for n in nodes)
    vx = sum(n["weight"] * n["X"] ** 2 for n in nodes) - mx * mx
    vp = sum(n["weight"] * n["P"] ** 2 for n in nodes) - mpv * mpv
    cxy = sum(n["weight"] * n["X"] * n["P"] for n in nodes) - mx * mpv
    if abs(cxy) > 1e-8:
        raise ValueError(f"product GH produced cov={cxy}")
    return {
        "kind": "ds", "theta": float(theta), "order": ORDER, "n_nodes": len(nodes),
        "weight_sum": wsum, "quad_mean_X": float(mx), "quad_mean_P": float(mpv),
        "quad_var_X": float(vx), "quad_var_P": float(vp), "quad_cov_XP": float(cxy),
        "target_mean_X": mom["mean_X"], "target_mean_P": mom["mean_P"],
        "target_var_X": mom["var_X"], "target_var_P": mom["var_P"],
        "target_cov_XP": mom["cov_XP"],
        "used_correlated_GH": False, "used_product_GH": True,
        "nodes": nodes, "analytic": mom,
        "did_not_call_q2_gh5_nodes": True,
    }


def resource_row(theta: float) -> dict:
    ana = analytic_xp_theta(theta)
    raw = dyn.displaced_squeezed_coefficients(q2.ALPHA, q2.R_SQ, float(theta), q2.NMAX_FOCK_REF)
    fock = q2.fock_xp_full(raw)
    gh_err = None
    gh = None
    try:
        gh = gh5_nodes_diag(theta)
    except ValueError as exc:
        gh_err = str(exc)
    row = {
        "theta": float(theta), "analytic": ana, "fock_nmax": q2.NMAX_FOCK_REF,
        "fock": fock, "gh_error": gh_err,
        "abs_d_mean_X": abs(fock["mean_X"] - ana["mean_X"]),
        "abs_d_mean_P": abs(fock["mean_P"] - ana["mean_P"]),
        "abs_d_var_X": abs(fock["var_X"] - ana["var_X"]),
        "abs_d_var_P": abs(fock["var_P"] - ana["var_P"]),
        "abs_d_cov_XP": abs(fock["cov_XP"] - ana["cov_XP"]),
        "abs_d_nbar": abs(fock["mean_n"] - ana["mean_n_analytic"]),
    }
    if gh is not None:
        row.update({
            "gh5_quad_mean_X": gh["quad_mean_X"], "gh5_quad_mean_P": gh["quad_mean_P"],
            "gh5_quad_var_X": gh["quad_var_X"], "gh5_quad_var_P": gh["quad_var_P"],
            "gh5_quad_cov_XP": gh["quad_cov_XP"], "gh5_weight_sum": gh["weight_sum"],
            "gh5_used_product_GH": True, "gh5_used_correlated_GH": False,
            "abs_d_quad_cov": abs(gh["quad_cov_XP"] - ana["cov_XP"]),
            "abs_d_quad_var_X": abs(gh["quad_var_X"] - ana["var_X"]),
            "abs_d_quad_var_P": abs(gh["quad_var_P"] - ana["var_P"]),
        })
    return row


def row_ok(row: dict) -> tuple[bool, list]:
    reasons = []
    ana = row["analytic"]
    th = float(row["theta"])
    if abs(ana["mean_X"] - 1.0) > 1e-12:
        reasons.append("mean_X")
    if abs(ana["mean_P"]) > 1e-12:
        reasons.append("mean_P")
    if abs(ana["mean_n_analytic"] - NBAR_TARGET) > 1e-8:
        reasons.append("nbar")
    if abs(ana["cov_XP"]) > 1e-10:
        reasons.append("analytic_cov_nonzero")
    if abs(ana["var_X"] - ana["var_P"]) < 0.5:
        reasons.append("varX_varP_too_close")
    if abs(th) < 1e-12:
        if not (ana["var_X"] < 1.0 < ana["var_P"]):
            reasons.append("theta0_var_order")
    elif abs(th - np.pi) < 1e-12:
        if not (ana["var_P"] < 1.0 < ana["var_X"]):
            reasons.append("thetapi_var_order")
    else:
        reasons.append("theta_not_diag")
    if row["abs_d_mean_X"] > 1e-10 or row["abs_d_mean_P"] > 1e-10:
        reasons.append("fock_mean")
    if row["abs_d_var_X"] > 1e-8 or row["abs_d_var_P"] > 1e-8:
        reasons.append("fock_var")
    if row["abs_d_cov_XP"] > 1e-8:
        reasons.append("fock_cov")
    if row["abs_d_nbar"] > 1e-8:
        reasons.append("fock_nbar")
    if row.get("gh_error"):
        reasons.append("gh_error:" + str(row["gh_error"]))
    else:
        if abs(row.get("gh5_quad_cov_XP", 1.0)) > 1e-8:
            reasons.append("quad_cov_nonzero")
        if abs(row.get("gh5_quad_var_X", 0.0) - row.get("gh5_quad_var_P", 0.0)) < 0.5:
            reasons.append("quad_var_too_close")
    return (len(reasons) == 0), reasons


def collect_resources() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    rows = {0.0: resource_row(THETA_0), np.pi: resource_row(THETA_PI)}
    ok0, r0 = row_ok(rows[0.0])
    okpi, rpi = row_ok(rows[np.pi])
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "resources", "source_hashes": hashes, "hash_abort": bad,
        "nbar_target": NBAR_TARGET, "theta_0": rows[0.0], "theta_pi": rows[np.pi],
        "theta0_ok": ok0, "theta0_reasons": r0, "thetapi_ok": okpi, "thetapi_reasons": rpi,
        "resource_match_ok": bool(ok0 and okpi and not bad),
        "used_product_GH": True, "did_not_call_q2_collect_resources": True,
        "did_not_call_q4_collect_resources": True, "stopped": True,
        "input_is_displaced_squeezed_not_pure_BSV": True,
    }
    dump(OUT / "resources.json", rec)
    return rec


def collect_eta_audits() -> dict:
    hashes = source_hashes()
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "status": "eta_audit",
        "source_hashes": hashes, "hash_abort": hash_abort(hashes),
        "etas": {}, "did_not_call_q4_collect_eta_audits": True, "stopped": True,
    }
    all_ok = True
    for eta in ETA_ALL:
        au = q4.eta_audit(eta)
        rec["etas"][f"{eta:g}"] = au
        all_ok = all_ok and bool(au["ok"])
    rec["all_ok"] = all_ok
    dump(OUT / "eta_audit.json", rec)
    return rec


def ckpt_path(tag: str) -> Path:
    return OUT / f"ckpt_{tag}.npz"


def job_path(tag: str) -> Path:
    return OUT / f"job_{tag}.json"


def series_path(tag: str) -> Path:
    return OUT / f"series_{tag}.npz"


def save_ckpt(tag, eta, neel, theta, n_done, n_nodes, acc, wsum, max_n, t_arr) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc = np.asarray(acc, complex)
    np.savez_compressed(
        ckpt_path(tag),
        acc_re=acc.real, acc_im=acc.imag, wsum=float(wsum),
        n_done=int(n_done), n_nodes=int(n_nodes), max_n=float(max_n),
        t=np.asarray(t_arr, float), eta=float(eta), neel=int(neel), theta=float(theta),
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
        "t": np.asarray(z["t"], float),
        "eta": float(z["eta"]) if "eta" in z.files else None,
        "theta": float(z["theta"]) if "theta" in z.files else None,
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


def load_series_any(tag: str, directory: Path):
    return q2.load_series(tag, directory)


def eta_jobs_ok(eta: float) -> bool:
    tags = [tag for tag, e, _, _ in JOBS if abs(e - eta) < 1e-12]
    return all(job_finished_ok(tag) for tag in tags)


def run_one_job(tag: str) -> dict:
    if tag not in JOB_MAP:
        raise ValueError(tag)
    eta, neel, theta = JOB_MAP[tag]
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"tag": tag, "status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(job_path(tag), rec)
        return rec
    audit = q4.eta_audit(eta)
    if not audit["ok"]:
        rec = {"tag": tag, "status": "eta_abort", "eta_audit": audit, "stopped": True}
        dump(job_path(tag), rec)
        progress(f"{tag} eta abort issues={audit['issues']}")
        return rec
    if job_finished_ok(tag):
        rec = load_job(tag)
        progress(f"{tag} already complete n_run={rec.get('n_run')}")
        return rec
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        gh = gh5_nodes_diag(theta)
    except ValueError as exc:
        rec = {"tag": tag, "status": "diag_gh_abort", "error": str(exc), "stopped": True}
        dump(job_path(tag), rec)
        return rec
    if abs(gh["quad_cov_XP"]) > 1e-8:
        rec = {"tag": tag, "status": "diag_cov_nonzero_abort",
               "quad_cov_XP": gh["quad_cov_XP"], "stopped": True}
        dump(job_path(tag), rec)
        return rec
    if abs(gh["quad_var_X"] - gh["quad_var_P"]) < 0.5:
        rec = {"tag": tag, "status": "equal_var_abort",
               "quad_var_X": gh["quad_var_X"], "quad_var_P": gh["quad_var_P"], "stopped": True}
        dump(job_path(tag), rec)
        return rec
    _, _, p, models, grounds, env = q4.make_bundle(eta)
    ek0, emk0, mass = grounds[neel]
    model = models[neel]
    nodes = gh["nodes"]
    js = sample_js_for(N_FULL)
    ck = load_ckpt(tag)
    i0, acc, wsum, max_n, t_arr = 0, None, 0.0, 0.0, None
    if ck is not None:
        if ck.get("eta") is not None and abs(float(ck["eta"]) - float(eta)) > 1e-12:
            rec = {"tag": tag, "status": "ckpt_eta_mismatch", "stopped": True}
            dump(job_path(tag), rec)
            return rec
        if ck.get("theta") is not None and abs(float(ck["theta"]) - float(theta)) > 1e-12:
            rec = {"tag": tag, "status": "ckpt_theta_mismatch", "stopped": True}
            dump(job_path(tag), rec)
            return rec
        i0, acc, wsum, max_n, t_arr = ck["n_done"], ck["acc"], ck["wsum"], ck["max_n"], ck["t"]
        progress(f"{tag} resume n_done={i0}/{len(nodes)}")
    clock = dyn.WallClock(JOB_WALL_S)
    progress(f"{tag} start eta={eta:g} neel={neel} theta={theta:.6f} "
             f"delta={p.delta:.4f} cov={gh['quad_cov_XP']:.3e} nodes={len(nodes)}")
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
            save_ckpt(tag, eta, neel, theta, i + 1, len(nodes), acc, wsum, max_n, t_arr)
            if (i + 1) == len(nodes) or (i + 1) % 5 == 0:
                progress(f"{tag} node {i + 1}/{len(nodes)} wall={clock.elapsed():.1f}s")
    except TimeoutError as exc:
        timed_out = True
        timeout_msg = str(exc)
        progress(f"{tag} timeout {timeout_msg}")
    n_done = 0 if acc is None else int(load_ckpt(tag)["n_done"] if ckpt_path(tag).is_file() else 0)
    if timed_out or n_done < len(nodes):
        rec = {
            "tag": tag, "eta": float(eta), "neel": int(neel), "theta": float(theta),
            "status": "timeout_partial", "n_nodes": len(nodes), "n_done": n_done,
            "weight_sum": wsum, "max_abs_norm_minus_1": float(max_n),
            "wall_s": clock.elapsed(), "timeout_msg": timeout_msg,
            "n_max": 0, "protocol": PROTOCOL, "n_steps": N_FULL,
            "dt": env.t_total / N_FULL, "stopped": True, "electron": mass,
            "quad_cov_XP": gh["quad_cov_XP"], "delta": float(p.delta),
            "used_product_GH": True,
        }
        dump(job_path(tag), rec)
        return rec
    avg = acc / (wsum + 1e-30)
    save_series(tag, t_arr, avg)
    rec = {
        "tag": tag, "eta": float(eta), "neel": int(neel), "order": ORDER, "theta": float(theta),
        "status": "ok", "n_nodes": len(nodes), "n_run": len(nodes),
        "n_done": len(nodes), "weight_sum": wsum,
        "max_abs_norm_minus_1": float(max_n),
        "wall_s": clock.elapsed(), "n_max": 0, "protocol": PROTOCOL,
        "n_steps": N_FULL, "dt": env.t_total / N_FULL,
        "n_samples": int(len(t_arr)), "t0": float(t_arr[0]), "t_end": float(t_arr[-1]),
        "electron": mass, "stopped": True, "delta": float(p.delta),
        "wigner": analytic_xp_theta(theta),
        "quad_cov_XP": gh["quad_cov_XP"],
        "used_correlated_GH": False, "used_product_GH": True,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
        "did_not_reuse_eta1_ground": True,
        "input_is_displaced_squeezed_not_pure_BSV": True,
    }
    dump(job_path(tag), rec)
    progress(f"{tag} done wall={clock.elapsed():.1f}s")
    return rec


def frozen_paths():
    return {
        "eta1_0": [(THETA0_DIR, "ds_p"), (THETA0_DIR, "ds_m")],
        "eta1_pi": [(THETAPI_DIR, "ds_p"), (THETAPI_DIR, "ds_m")],
        "eta1_h": [(Q2_DIR, "ds_p_h"), (Q2_DIR, "ds_m_h")],
        "eta1_t": [(Q2_DIR, "ds_p_t"), (Q2_DIR, "ds_m_t")],
        "e05_h": [(Q4_DIR, "e05_p_h"), (Q4_DIR, "e05_m_h")],
        "e05_t": [(Q4_DIR, "e05_p_t"), (Q4_DIR, "e05_m_t")],
        "e15_h": [(Q4_DIR, "e15_p_h"), (Q4_DIR, "e15_m_h")],
        "e15_t": [(Q4_DIR, "e15_p_t"), (Q4_DIR, "e15_m_t")],
    }


def missing_frozen() -> list:
    miss = []
    for _, pairs in frozen_paths().items():
        for directory, tag in pairs:
            if not (directory / f"series_{tag}.npz").is_file():
                miss.append({"dir": str(directory), "tag": tag})
    return miss


def load_pair(directory: Path, tag_p: str, tag_m: str):
    t_p, j_p = load_series_any(tag_p, directory)
    t_m, j_m = load_series_any(tag_m, directory)
    return t_p, t_m, j_p, j_m


def load_eta_four(eta: float):
    if abs(eta - 1.0) < 1e-12:
        t0p, t0m, j0p, j0m = load_pair(THETA0_DIR, "ds_p", "ds_m")
        thp, thm, jhp, jhm = load_pair(Q2_DIR, "ds_p_h", "ds_m_h")
        tpp, tpm, jpp, jpm = load_pair(THETAPI_DIR, "ds_p", "ds_m")
        ttp, ttm, jtp, jtm = load_pair(Q2_DIR, "ds_p_t", "ds_m_t")
        src = "frozen_eta1"
    else:
        pref = "e05" if abs(eta - 0.5) < 1e-12 else "e15"
        t0p, t0m, j0p, j0m = load_pair(OUT, f"{pref}_p_0", f"{pref}_m_0")
        thp, thm, jhp, jhm = load_pair(Q4_DIR, f"{pref}_p_h", f"{pref}_m_h")
        tpp, tpm, jpp, jpm = load_pair(OUT, f"{pref}_p_pi", f"{pref}_m_pi")
        ttp, ttm, jtp, jtm = load_pair(Q4_DIR, f"{pref}_p_t", f"{pref}_m_t")
        src = "mixed_q4_frozen_plus_q5"
    times = [("0p", t0p), ("0m", t0m), ("hp", thp), ("hm", thm),
             ("pip", tpp), ("pim", tpm), ("tp", ttp), ("tm", ttm)]
    t = t0p
    for lab, tt in times:
        if not np.allclose(tt, t, rtol=0, atol=1e-9):
            raise ValueError(f"time_grid_mismatch eta={eta} {lab}")
    even_odd = {}
    for key, jp, jm in (("0", j0p, j0m), ("h", jhp, jhm), ("pi", jpp, jpm), ("t", jtp, jtm)):
        ev, od = odd.even_odd(jp, jm)
        even_odd[key] = (ev, od)
    return {"eta": float(eta), "source": src, "t": t, "even_odd": even_odd}


def summarize_eta(pack, env) -> dict:
    t = pack["t"]
    T0 = env.period
    omega0 = pf.OMEGA0
    arrs = {}
    for key in THETA_KEYS:
        arrs[f"J_odd_{key}"] = pack["even_odd"][key][1]
        arrs[f"J_even_{key}"] = pack["even_odd"][key][0]
    arrs["dJ_odd"] = pack["even_odd"]["t"][1] - pack["even_odd"]["h"][1]
    arrs["dJ_even"] = pack["even_odd"]["t"][0] - pack["even_odd"]["h"][0]
    main = pack_win("plateau_center_2T", t, omega0, T0, 2.0 * T0, 4.0 * T0, arrs)
    front = pack_win("front_2T", t, omega0, T0, 1.0 * T0, 3.0 * T0, arrs)
    back = pack_win("back_2T", t, omega0, T0, 3.0 * T0, 5.0 * T0, arrs)

    def take(winobj, key):
        return h2_jx(winobj["slots"], key)

    odd_z, even_z = [], []
    odd_pack, even_pack = {}, {}
    for key in THETA_KEYS:
        zo = take(main, f"J_odd_{key}")
        ze = take(main, f"J_even_{key}")
        odd_z.append(zo[0])
        even_z.append(ze[0])
        odd_pack[key] = {"jx": pack_c(zo[0]), "abs": zo[2], "phase": zo[3]}
        even_pack[key] = {"jx": pack_c(ze[0]), "abs": ze[2], "phase": ze[3]}
    odd_z = np.asarray(odd_z, complex)
    even_z = np.asarray(even_z, complex)
    fit_o = fit_angle_model(list(THETAS), odd_z)
    fit_e = fit_angle_model(list(THETAS), even_z)
    Ao, Bo, Co = cz(fit_o["A"]), cz(fit_o["B"]), cz(fit_o["C"])
    Ae, Be, Ce = cz(fit_e["A"]), cz(fit_e["B"]), cz(fit_e["C"])
    q_odd = odd_z[3] / odd_z[1]
    q_even = even_z[3] / even_z[1]
    d_odd = odd_z[3] - odd_z[1]
    d_even = even_z[3] - even_z[1]
    zf = take(front, "dJ_odd")
    zb = take(back, "dJ_odd")
    h1h6 = [{"q": iq + 1, "jx": pack_c(cz(main["slots"]["dJ_odd"]["Jx"][iq])),
             "abs": float(abs(cz(main["slots"]["dJ_odd"]["Jx"][iq])))} for iq in range(6)]
    return {
        "eta": pack["eta"], "source": pack["source"],
        "windows": {"main": main, "front": front, "back": back},
        "odd": odd_pack, "even": even_pack,
        "odd_z": pack_c(odd_z), "even_z": pack_c(even_z),
        "fit_odd": fit_o, "fit_even": fit_e,
        "abs_A_over_B_odd": float(abs(Ao) / max(abs(Bo), 1e-30)),
        "abs_A_over_B_even": float(abs(Ae) / max(abs(Be), 1e-30)),
        "abs_C_over_A_odd": float(abs(Co) / max(abs(Ao), 1e-30)),
        "abs_C_over_A_even": float(abs(Ce) / max(abs(Ae), 1e-30)),
        "A_over_B_odd": pack_c(Ao / Bo if abs(Bo) > 1e-30 else 0.0),
        "Q_odd": pack_c(q_odd), "abs_Q_odd": float(abs(q_odd)),
        "arg_Q_odd": float(np.angle(q_odd)),
        "Q_even": pack_c(q_even), "abs_Q_even": float(abs(q_even)),
        "arg_Q_even": float(np.angle(q_even)),
        "delta_odd": pack_c(d_odd), "abs_delta_odd": float(abs(d_odd)),
        "phase_delta_odd": float(np.angle(d_odd)),
        "delta_even": pack_c(d_even), "abs_delta_even": float(abs(d_even)),
        "abs_front_dodd": zf[2], "phase_front_dodd": zf[3],
        "abs_back_dodd": zb[2], "phase_back_dodd": zb[3],
        "h1h6_dodd_jx": h1h6,
        "cannot_exclude_higher_theta_harmonics": True,
        "not_certified_HHG": True,
    }


def factor_strict(z_ref, z_eta) -> dict:
    z_ref = np.asarray(z_ref, complex)
    z_eta = np.asarray(z_eta, complex)
    den = np.vdot(z_ref, z_ref)
    F = np.vdot(z_ref, z_eta) / den if abs(den) > 0 else 0.0j
    pred = F * z_ref
    resid = z_eta - pred
    rel = float(np.linalg.norm(resid) / max(float(np.linalg.norm(z_eta)), 1e-30))
    per = [float(abs(resid[k]) / max(abs(z_eta[k]), 1e-30)) for k in range(len(z_eta))]
    abs_err = [float(abs(resid[k])) for k in range(len(z_eta))]
    return {
        "kind": "strict_J=F*G_with_G_from_eta1",
        "F": pack_c(F), "abs_F": float(abs(F)), "arg_F": float(np.angle(F)),
        "pred": pack_c(pred), "resid": pack_c(resid), "rel_resid": rel,
        "per_angle_rel": per, "per_angle_abs": abs_err,
        "engineering_mark": STRICT_REL_MARK,
        "below_mark": bool(rel < STRICT_REL_MARK),
        "not_a_physical_law": True,
    }


def factor_weak(z_ref, z_eta) -> dict:
    z_ref = np.asarray(z_ref, complex)
    z_eta = np.asarray(z_eta, complex)
    X = np.column_stack([np.ones(len(z_ref)), z_ref])
    coef, _, _, _ = np.linalg.lstsq(X, z_eta, rcond=None)
    B, F = complex(coef[0]), complex(coef[1])
    pred = X @ coef
    resid = z_eta - pred
    rel = float(np.linalg.norm(resid) / max(float(np.linalg.norm(z_eta)), 1e-30))
    per = [float(abs(resid[k]) / max(abs(z_eta[k]), 1e-30)) for k in range(len(z_eta))]
    abs_err = [float(abs(resid[k])) for k in range(len(z_eta))]
    return {
        "kind": "weak_J=B+F*G_with_G_from_eta1",
        "B": pack_c(B), "F": pack_c(F), "abs_F": float(abs(F)), "arg_F": float(np.angle(F)),
        "pred": pack_c(pred), "resid": pack_c(resid), "rel_resid": rel,
        "per_angle_rel": per, "per_angle_abs": abs_err,
        "engineering_mark": WEAK_REL_MARK,
        "below_mark": bool(rel < WEAK_REL_MARK),
        "not_a_physical_law": True,
        "not_the_same_as_strict": True,
    }


def normalize_shape(zs, B, A):
    zs = np.asarray(zs, complex)
    scale = abs(A) if abs(A) > 0 else 1.0
    return (zs - B) / scale * np.exp(-1j * np.angle(A))


def figure_four_plane(zs, path, title):
    lim = max(abs(z) for z in zs) * 1.28 + 1e-30
    x0, y0, w, h = 90, 40, 740, 430
    body = q2.svg_axes(x0, y0, w, h, "Re Jx (H2 odd)", "Im Jx (H2 odd)", title)
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    labels = ["θ=0", "θ=π/2", "θ=π", "θ=3π/2"]
    colors = ["#4c6b8a", "#7a4ea3", "#c45c26", "#2f9e44"]
    lx = 70
    for lab, z, col in zip(labels, zs, colors):
        body += q2.arrow(z, lim, x0, y0, w, h, col, lab, lx, 36)
        lx += 130
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "GH5/MTEF; four equal angles alias e^{i5θ}; not certified HHG</text>"
    )
    return q2.write_svg(path, body)


def figure_normalized(shapes, path):
    x0, y0, w, h = 90, 40, 740, 430
    body = q2.svg_axes(x0, y0, w, h, "Re (z-B)/|A| e^{-i arg A}", "Im",
                       "Normalized odd H2 angular shape")
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    lim = 2.2
    pts = []
    for ang in np.linspace(0, 2 * np.pi, 97):
        z = np.exp(1j * ang)
        px, py = q2.map_xy(z, lim, x0, y0, w, h)
        pts.append(f"{px:.1f},{py:.1f}")
    body += f'<polyline fill="none" stroke="#c9a24a" stroke-dasharray="4 3" points="{" ".join(pts)}"/>'
    lx = 70
    for eta, zs, col in shapes:
        poly = []
        for z in list(zs) + [zs[0]]:
            px, py = q2.map_xy(z, lim, x0, y0, w, h)
            poly.append(f"{px:.1f},{py:.1f}")
            body += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{col}"/>'
        body += f'<polyline fill="none" stroke="{col}" stroke-width="2" points="{" ".join(poly)}"/>'
        body += f'<text x="{lx}" y="36" font-size="11" fill="{col}">η={eta:g}</text>'
        lx += 90
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "shape after removing B and rotating/scaling by A; dashed = unit e^{iθ}</text>"
    )
    return q2.write_svg(path, body)


def cfmt(z, prec=3):
    z = complex(z)
    return f"{z.real:.{prec}e}{z.imag:+.{prec}e}i"


def write_reports(rec: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    v = rec["verdict"]
    ans = rec["answer"]
    full = []
    full.append("# 量子光多周期磁序响应：Q5 三交换强度四角 GH5 复轨迹\n")
    full.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。")
    full.append("冻结 η=1 四角 GH5/QED，以及 Q4 的 η=0.5、1.5 在 θ=π/2、3π/2。")
    full.append("本轮仅补齐 η=0.5、1.5 的 θ=0、π，各 N=±1，共八路新 GH5。")
    full.append("θ=0/π 使用对角乘积 GH。`stopped = true`。输入是位移压缩态，不是纯 BSV。\n")
    full.append(f"**结论：{ans}**\n")
    full.append("## 控制与资源\n")
    full.append("α=0.5，r=0.3，⟨X⟩=1，⟨P⟩=0，⟨n⟩≈0.342733。")
    full.append("θ=0：Var(X)=e^{-0.6}<1<Var(P)；θ=π 对调；Cov(X,P)=0。")
    full.append("η 只缩放 `neel * η * R0.delta`。filling=4 按 η 重建。\n")
    full.append("| η | δ | 能隙(k,+N) | 审计 |\n|---|---|---|---|")
    for key in ("0.5", "1", "1.5"):
        au = rec["eta_audit"]["etas"][key]
        gap = au["bands"][0]["gap"]
        full.append(f"| {key} | {au['delta']:.4f} | {gap:.4f} | `{au['ok']}` {au['issues']} |")
    full.append("\nη=1.5 能隙 0.362 仍大于 2ω0=0.10，不能预设为双光子共振；它是能带重整化。\n")
    full.append("## 主窗 H2 磁序奇四角\n")
    full.append("| η | θ | Jx | |J| | arg(deg) |\n|---|---|---|---|---|")
    labs = {"0": "0", "h": "π/2", "pi": "π", "t": "3π/2"}
    for eta in ETA_ALL:
        s = rec["by_eta"][f"{eta:g}"]
        for key in THETA_KEYS:
            z = cz(s["odd"][key]["jx"])
            full.append(
                f"| {eta:g} | {labs[key]} | {cfmt(z)} | {s['odd'][key]['abs']:.3e} | "
                f"{s['odd'][key]['phase']*180/np.pi:.1f} |"
            )
    full.append("\n## 角向拟合 B+A e^{iθ}+C e^{-iθ}（四点，存在 e^{i5θ} 混叠）\n")
    full.append("| η | B | A | |C/A| | |A/B| | 拟合相对残差 |\n|---|---|---|---|---|---|")
    for eta in ETA_ALL:
        s = rec["by_eta"][f"{eta:g}"]
        full.append(
            f"| {eta:g} | {cfmt(cz(s['fit_odd']['B']))} | {cfmt(cz(s['fit_odd']['A']))} | "
            f"{s['abs_C_over_A_odd']:.3f} | {s['abs_A_over_B_odd']:.3f} | {s['fit_odd']['rel_resid']:.4f} |"
        )
    full.append("\n## Q(η)=J(3π/2)/J(π/2) 磁序奇\n")
    full.append("| η | |Q| | arg(deg) |\n|---|---|---|")
    for eta in ETA_ALL:
        s = rec["by_eta"][f"{eta:g}"]
        full.append(f"| {eta:g} | {s['abs_Q_odd']:.3f} | {s['arg_Q_odd']*180/np.pi:.2f} |")
    full.append("\n## 因子化残差（相对 η=1 四角样本，不是 χ(2)）\n")
    full.append("严格：J(η)=F G，G 取 η=1 的四角向量。弱：J(η)=B+F G，B 独立。")
    full.append(f"工程标记严格相对残差<{STRICT_REL_MARK}、弱<{WEAK_REL_MARK}，不是物理定律。\n")
    full.append("| η | 通道 | 类型 | F | |F| | arg(deg) | 相对残差 | 各角相对误差 |\n|---|---|---|---|---|---|---|---|")
    for eta in (0.5, 1.5):
        fac = rec["factor"][f"{eta:g}"]
        for ch in ("odd", "even"):
            for kind in ("strict", "weak"):
                row = fac[ch][kind]
                F = cz(row["F"])
                per = ",".join(f"{x:.3f}" for x in row["per_angle_rel"])
                full.append(
                    f"| {eta:g} | {ch} | {kind} | {cfmt(F)} | {row['abs_F']:.3e} | "
                    f"{row['arg_F']*180/np.pi:.1f} | {row['rel_resid']:.4f} | {per} |"
                )
    full.append("\n各角绝对残差见 JSON。四点等间隔角向 Fourier 不能认证唯一角向定律。\n")
    full.append("## 偶通道对照\n")
    full.append("| η | |A_o/B_o| | |A_e/B_e| | |δ⊥_odd| | |δ⊥_even| |\n|---|---|---|---|---|")
    for eta in ETA_ALL:
        s = rec["by_eta"][f"{eta:g}"]
        full.append(
            f"| {eta:g} | {s['abs_A_over_B_odd']:.3f} | {s['abs_A_over_B_even']:.5f} | "
            f"{s['abs_delta_odd']:.3e} | {s['abs_delta_even']:.3e} |"
        )
    full.append("\n归一化角向选择性不是绝对磁性增强，也不是新选择定则。\n")
    full.append("## 窗口敏感性（磁序奇 δ⊥，主/前/后）\n")
    full.append("| η | |δ⊥|_main | |δ⊥|_front | |δ⊥|_back |\n|---|---|---|---|")
    for eta in ETA_ALL:
        s = rec["by_eta"][f"{eta:g}"]
        full.append(
            f"| {eta:g} | {s['abs_delta_odd']:.3e} | {s['abs_front_dodd']:.3e} | {s['abs_back_dodd']:.3e} |"
        )
    full.append("\n## 解释边界\n")
    full.append("- 本轮只有 GH5/MTEF，没有新的精确 QED。")
    full.append("- 证伪 δ⊥∝η 不等于证伪 J(η,θ)≈F(η)G(θ) 或 B(η)+F(η)G(θ)。二者必须分开。")
    full.append("- η=1.5 的大幅值变化伴随能隙下降，优先视为谱结构重整化。")
    full.append("- 两周期矩形 Fourier ≠ 已认证 HHG。共享 k/−k，不是完整 BZ。")
    full.append("- 未启动精确 QED、额外 η/θ、Cn、nmax、服务器。\n")
    full.append("完成后停机。不自动申请 QED。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q5_GH5_fourangle_20261120.md").write_text(
        "\n".join(full) + "\n", encoding="utf-8")
    one = []
    one.append("# 量子光多周期磁序响应：Q5 四角轨迹（一页）\n")
    one.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。`stopped = true`。\n")
    one.append(f"**{ans}**\n")
    one.append("| 量 | 结果 |\n|---|---|")
    one.append("| 新作业 | η=0.5、1.5 × θ=0、π × N=±1 |\n")
    one.append(f"| 分类 | `{v}` |")
    fo = rec["factor"]
    one.append(
        f"| 严格相对残差 0.5/1.5 (odd) | {fo['0.5']['odd']['strict']['rel_resid']:.4f} / "
        f"{fo['1.5']['odd']['strict']['rel_resid']:.4f} |"
    )
    one.append(
        f"| 弱相对残差 0.5/1.5 (odd) | {fo['0.5']['odd']['weak']['rel_resid']:.4f} / "
        f"{fo['1.5']['odd']['weak']['rel_resid']:.4f} |"
    )
    one.append("| 新 QED | 现在不启动 |\n")
    one.append("**问题：** 改变反铁磁交换强度时，压缩角响应是同一种复轨迹被材料系数放大/旋转，")
    one.append("还是角向结构本身发生了不可用共同复比例描述的变化？\n")
    one.append(f"**{rec['answer_short']}**\n")
    one.append("停机。不自动开精确 QED。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q5_GH5_fourangle_一页_20261120.md").write_text(
        "\n".join(one) + "\n", encoding="utf-8")


def analyze() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(OUT / "gh5_q5_fourangle.json", rec)
        return rec
    miss = missing_frozen()
    if miss:
        rec = {"status": "missing_frozen_archive", "missing": miss, "stopped": True}
        dump(OUT / "gh5_q5_fourangle.json", rec)
        return rec
    missing_jobs = [tag for tag, _, _, _ in JOBS if not job_finished_ok(tag)]
    if missing_jobs:
        rec = {"status": "missing_jobs", "missing": missing_jobs, "stopped": True,
               "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                              for tag, _, _, _ in JOBS}}
        dump(OUT / "gh5_q5_fourangle.json", rec)
        return rec
    res = collect_resources()
    if not res.get("resource_match_ok"):
        rec = {"status": "resource_mismatch", "resources": res, "stopped": True}
        dump(OUT / "gh5_q5_fourangle.json", rec)
        return rec
    audits = collect_eta_audits()
    env = q2.make_env()
    by_eta = {}
    packs = {}
    for eta in ETA_ALL:
        pack = load_eta_four(eta)
        packs[eta] = pack
        by_eta[f"{eta:g}"] = summarize_eta(pack, env)
    z1_o = np.asarray([cz(z) for z in by_eta["1"]["odd_z"]], complex)
    z1_e = np.asarray([cz(z) for z in by_eta["1"]["even_z"]], complex)
    factor = {}
    for eta in (0.5, 1.5):
        zo = np.asarray([cz(z) for z in by_eta[f"{eta:g}"]["odd_z"]], complex)
        ze = np.asarray([cz(z) for z in by_eta[f"{eta:g}"]["even_z"]], complex)
        factor[f"{eta:g}"] = {
            "odd": {"strict": factor_strict(z1_o, zo), "weak": factor_weak(z1_o, zo)},
            "even": {"strict": factor_strict(z1_e, ze), "weak": factor_weak(z1_e, ze)},
        }
    shapes = []
    shape_dist = {}
    for eta in ETA_ALL:
        s = by_eta[f"{eta:g}"]
        B, A = cz(s["fit_odd"]["B"]), cz(s["fit_odd"]["A"])
        zs = np.asarray([cz(z) for z in s["odd_z"]], complex)
        sh = normalize_shape(zs, B, A)
        s["normalized_odd"] = pack_c(sh)
        shapes.append((eta, sh, ETA_COLOR[eta]))
    sh1 = np.asarray([cz(z) for z in by_eta["1"]["normalized_odd"]], complex)
    for eta in (0.5, 1.5):
        shn = np.asarray([cz(z) for z in by_eta[f"{eta:g}"]["normalized_odd"]], complex)
        shape_dist[f"{eta:g}"] = float(np.linalg.norm(shn - sh1) / max(np.linalg.norm(sh1), 1e-30))
    rs05 = factor["0.5"]["odd"]["strict"]["rel_resid"]
    rs15 = factor["1.5"]["odd"]["strict"]["rel_resid"]
    rw05 = factor["0.5"]["odd"]["weak"]["rel_resid"]
    rw15 = factor["1.5"]["odd"]["weak"]["rel_resid"]
    strict_ok = rs05 < STRICT_REL_MARK and rs15 < STRICT_REL_MARK
    weak_ok = rw05 < WEAK_REL_MARK and rw15 < WEAK_REL_MARK
    if strict_ok:
        verdict = "strict_factorization_compatible_within_mark"
        answer = (
            "在当前 GH5 四角样本上，η=0.5 与 1.5 的磁序奇 H2 可用同一个复比例把 η=1 的四角轨迹"
            "映射过去（工程标记内）。这与“量子光统计给出角向结构、材料系数放大/旋转”相容，"
            "但四点存在角向谐波混叠，且 η=1.5 伴随能隙下降，不能称为新的反铁磁选择定则。"
        )
        answer_short = "主要表现为同一种复轨迹被材料系数放大/旋转（严格因子化，工程标记内）"
    elif weak_ok:
        verdict = "weak_factorization_compatible_independent_background"
        answer = (
            "严格 J=F(η)G(θ) 不能在工程标记内成立，但允许独立背景的 J=B(η)+F(η)G(θ)"
            "可以把 η=1 的角向样本映射到 η=0.5、1.5。压缩角相对结构仍较稳定，"
            "磁性交换主要改背景、幅值和整体相位。这不是 χ(2)，也不是新选择定则。"
        )
        answer_short = "角向结构近似可因子化，但需要独立背景 B(η)；不是严格 F(η)G(θ)"
    else:
        verdict = "angular_structure_changes_with_eta"
        answer = (
            "即使允许独立背景，也不能用一个共同复比例把 η=1 的四角轨迹映射到"
            "η=0.5 或 1.5。归一化角向结构随磁性交换改变。这提示更复杂的磁性—统计耦合，"
            "但 η=1.5 同时发生能带重整化，不能自动升级为新型量子磁性选择定则。"
        )
        answer_short = "角向结构本身随 η 变化，不能用共同复比例完全描述"
    FIG.mkdir(parents=True, exist_ok=True)
    figs = {
        "eta0.5": figure_four_plane(
            np.asarray([cz(z) for z in by_eta["0.5"]["odd_z"]], complex),
            FIG / "h2_odd_plane_eta05.svg", "η=0.5 H2 odd Jx (GH5)"),
        "eta1.0": figure_four_plane(
            np.asarray([cz(z) for z in by_eta["1"]["odd_z"]], complex),
            FIG / "h2_odd_plane_eta10.svg", "η=1 H2 odd Jx (GH5, frozen)"),
        "eta1.5": figure_four_plane(
            np.asarray([cz(z) for z in by_eta["1.5"]["odd_z"]], complex),
            FIG / "h2_odd_plane_eta15.svg", "η=1.5 H2 odd Jx (GH5)"),
        "normalized": figure_normalized(shapes, FIG / "h2_odd_normalized_overlay.svg"),
    }
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "ok", "stopped": True, "verdict": verdict,
        "answer": answer, "answer_short": answer_short,
        "source_hashes": hashes, "resources": res, "eta_audit": audits,
        "by_eta": by_eta, "factor": factor, "shape_dist_to_eta1": shape_dist,
        "strict_rel_mark": STRICT_REL_MARK, "weak_rel_mark": WEAK_REL_MARK,
        "strict_ok_odd": strict_ok, "weak_ok_odd": weak_ok,
        "figures": figs, "qed_not_started": True, "no_extra_eta": True,
        "no_extra_theta": True, "no_server": True,
        "input_is_displaced_squeezed_not_pure_BSV": True,
        "not_certified_HHG": True, "four_angle_aliasing": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "gh5_q5_fourangle.json", rec)
    write_reports(rec)
    return rec


def launch_batch(tags, t0: float) -> dict:
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
    return {tag: (procs[tag].returncode if tag in procs else 0) for tag in tags}


def drive() -> int:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        dump(OUT / "gh5_q5_fourangle.json", {"status": "hash_abort", "hash_mismatch": bad, "stopped": True})
        progress("hash abort, no propagation")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    res = collect_resources()
    if not res.get("resource_match_ok"):
        dump(OUT / "gh5_q5_fourangle.json", {"status": "resource_mismatch", "resources": res, "stopped": True})
        progress("resource mismatch, no propagation")
        return 3
    audits = collect_eta_audits()
    t0 = time.perf_counter()
    codes = {}
    for eta, tags in BATCHES:
        remaining = DRIVER_WALL_S - (time.perf_counter() - t0)
        au = audits["etas"][f"{eta:g}"]
        if not au["ok"]:
            progress(f"skip eta={eta:g} audit issues={au['issues']}")
            continue
        if remaining < 60.0:
            progress(f"skip eta={eta:g}: remaining {remaining:.1f}s < 60")
            break
        progress(f"batch eta={eta:g} remaining={remaining:.1f}s")
        codes.update(launch_batch(tags, t0))
        if time.perf_counter() - t0 >= DRIVER_WALL_S:
            progress("driver wall reached after batch")
            break
    progress(f"jobs return {codes} driver_wall={time.perf_counter()-t0:.1f}s")
    if eta_jobs_ok(0.5) and eta_jobs_ok(1.5):
        rec = analyze()
        progress(f"analyze done verdict={rec.get('verdict')}")
        return 0 if rec.get("status") == "ok" else 1
    rec = {
        "status": "timeout_partial", "stopped": True,
        "driver_wall_s": time.perf_counter() - t0, "returncodes": codes,
        "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                       for tag, _, _, _ in JOBS},
        "did_not_analyze_incomplete": True, "qed_not_started": True,
    }
    dump(OUT / "gh5_q5_fourangle.json", rec)
    progress("incomplete, no full analyze")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--resources-only", action="store_true")
    ap.add_argument("--eta-audit-only", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.eta_audit_only:
        rec = collect_eta_audits()
        print(json.dumps(jsonable({
            "status": rec.get("status"), "all_ok": rec.get("all_ok"),
            "etas": {k: {"ok": v["ok"], "delta": v["delta"], "issues": v["issues"],
                         "gap_k_plus": v["bands"][0]["gap"]}
                     for k, v in rec.get("etas", {}).items()},
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") == "eta_audit" and rec.get("all_ok") else 3
    if args.resources_only:
        rec = collect_resources()
        print(json.dumps(jsonable({
            "status": rec.get("status"),
            "resource_match_ok": rec.get("resource_match_ok"),
            "theta0_ok": rec.get("theta0_ok"), "thetapi_ok": rec.get("thetapi_ok"),
            "theta0_reasons": rec.get("theta0_reasons"),
            "thetapi_reasons": rec.get("thetapi_reasons"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") == "resources" and rec.get("resource_match_ok") else 3
    if args.job:
        rec = run_one_job(args.job)
        return 0 if rec.get("status") in ("ok", "timeout_partial", "eta_abort") else 1
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
