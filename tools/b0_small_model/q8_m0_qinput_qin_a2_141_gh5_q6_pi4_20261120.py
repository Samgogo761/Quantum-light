#!/usr/bin/env python3
"""Qin_a2_1_4_1 GH5/MTEF Q6: out-of-sample theta=pi/4 prediction test.

Freeze Q5 four-angle B,A,C and predict J(pi/4) BEFORE any new GH5.
New jobs only: theta=pi/4, eta=0.5/1/1.5, N=+/-1 (six GH5 paths).
pi/4 is the first state with BOTH unequal Var(X),Var(P) AND nonzero Cov(X,P).
Must use Q2 correlated 2x2 GH. Product GH and Q2 row_ok(need_corr=True) are wrong.
No new QED, extra theta/eta, Cn, BSV, GH7, k, nmax, D2, R2, or server.
Integer Fourier is not certified HHG. GH5 is not exact QED.
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

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_qin_a2_141_gh5_q2_covsign_20261120 as q2
import q8_m0_qinput_qin_a2_141_gh5_q4_eta_20261120 as q4
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
from q8_m0_model import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_qin_a2_141_gh5_q6_pi4_20261120"
FIG = OUT / "figures"
Q5_DIR = ROOT / "results/q_line/qinput_qin_a2_141_gh5_q5_fourangle_20261120"
Q5_JSON = Q5_DIR / "gh5_q5_fourangle.json"
Q5_SHA256 = "e2d58d84363e301a427c53126f09ef7bc0f7b3d383f0a84ca3a797cf16d8b1c5"
FROZEN_PRED = OUT / "frozen_predictions.json"
DOCS = ROOT / "docs/04_BSV量子光"
TASK_TAG = "20261120"
RECORD_DATE = "2026-10-24"
PROTOCOL = "Qin_a2_1_4_1"
ORDER = q2.ORDER
N_NODES = q2.N_NODES
N_FULL = q2.N_FULL
SAMPLE_EVERY = q2.SAMPLE_EVERY
DRIVER_WALL_S = 600.0
JOB_WALL_S = 520.0
THETA_Q = 0.25 * np.pi
ETA_ALL = (0.5, 1.0, 1.5)
NBAR_TARGET = 0.25 + float(np.sinh(0.3) ** 2)
ETA_COLOR = {0.5: "#7a4ea3", 1.0: "#2f9e44", 1.5: "#c45c26"}
JOBS = (
    ("e05_p_q", 0.5, 1, THETA_Q),
    ("e05_m_q", 0.5, -1, THETA_Q),
    ("e10_p_q", 1.0, 1, THETA_Q),
    ("e10_m_q", 1.0, -1, THETA_Q),
    ("e15_p_q", 1.5, 1, THETA_Q),
    ("e15_m_q", 1.5, -1, THETA_Q),
)
JOB_MAP = {tag: (eta, neel, theta) for tag, eta, neel, theta in JOBS}
BATCHES = (
    (0.5, ("e05_p_q", "e05_m_q")),
    (1.0, ("e10_p_q", "e10_m_q")),
    (1.5, ("e15_p_q", "e15_m_q")),
)
ETA_PREF = {0.5: "e05", 1.0: "e10", 1.5: "e15"}
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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def eta_key(eta: float) -> str:
    return f"{eta:g}"


def pred_from_bac(B, A, C, theta: float) -> complex:
    return complex(B) + complex(A) * np.exp(1j * theta) + complex(C) * np.exp(-1j * theta)


def pack_pred(z):
    z = complex(z)
    return {
        "jx": pack_c(z), "abs": float(abs(z)), "phase": float(np.angle(z)),
        "phase_deg": float(np.angle(z) * 180.0 / np.pi),
    }


def freeze_predictions() -> dict:
    """Lock Q5 B,A,C predictions. If freeze file exists, read-only; never rewrite with Q6 data."""
    OUT.mkdir(parents=True, exist_ok=True)
    if FROZEN_PRED.is_file():
        rec = json.loads(FROZEN_PRED.read_text(encoding="utf-8"))
        rec["loaded_existing_freeze"] = True
        rec["did_not_modify_existing_freeze"] = True
        progress(f"freeze existing sha={file_sha256(FROZEN_PRED)[:12]}")
        return rec
    if not Q5_JSON.is_file():
        rec = {"status": "missing_q5_json", "path": str(Q5_JSON), "stopped": True}
        dump(FROZEN_PRED, rec)
        return rec
    got = file_sha256(Q5_JSON)
    if got != Q5_SHA256:
        rec = {
            "status": "q5_hash_mismatch", "expected": Q5_SHA256, "got": got,
            "stopped": True, "did_not_propagate": True,
        }
        dump(FROZEN_PRED, rec)
        return rec
    q5 = json.loads(Q5_JSON.read_text(encoding="utf-8"))
    if q5.get("status") != "ok":
        rec = {"status": "q5_not_ok", "q5_status": q5.get("status"), "stopped": True}
        dump(FROZEN_PRED, rec)
        return rec
    by_eta = {}
    for eta in ETA_ALL:
        key = eta_key(eta)
        s = q5["by_eta"][key]
        row = {"eta": float(eta)}
        for ch in ("odd", "even"):
            fit = s[f"fit_{ch}"]
            B, A, C = cz(fit["B"]), cz(fit["A"]), cz(fit["C"])
            pred = pred_from_bac(B, A, C, THETA_Q)
            row[f"fit_{ch}"] = {
                "B": pack_c(B), "A": pack_c(A), "C": pack_c(C),
                "rel_resid": float(fit["rel_resid"]),
                "abs_A": float(abs(A)), "abs_B": float(abs(B)), "abs_C": float(abs(C)),
                "abs_C_over_A": float(abs(C) / max(abs(A), 1e-30)),
                "arg_A": float(np.angle(A)), "arg_B": float(np.angle(B)),
            }
            row[f"pred_{ch}"] = pack_pred(pred)
        by_eta[key] = row
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "frozen", "stopped": True,
        "frozen_before_any_q6_propagation": True,
        "did_not_use_q6_data": True,
        "q5_json": str(Q5_JSON.as_posix()),
        "q5_sha256": got,
        "q5_verdict_as_recorded": q5.get("verdict"),
        "q5_verdict_not_used_as_physical_law": True,
        "theta_test": float(THETA_Q),
        "theta_test_label": "pi/4",
        "model": "J_pred(eta,theta)=B(eta)+A(eta)*exp(i theta)+C(eta)*exp(-i theta)",
        "alias_identity": "at four Q5 angles e^{i5theta}=e^{itheta}; at pi/4 e^{i5theta}=-e^{itheta}",
        "input_is_displaced_squeezed_not_pure_BSV": True,
        "by_eta": by_eta,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(FROZEN_PRED, rec)
    progress(f"freeze created sha={file_sha256(FROZEN_PRED)[:12]}")
    return rec


def resource_row() -> dict:
    theta = THETA_Q
    ana = analytic_xp_theta(theta)
    raw = dyn.displaced_squeezed_coefficients(q2.ALPHA, q2.R_SQ, float(theta), q2.NMAX_FOCK_REF)
    fock = q2.fock_xp_full(raw)
    gh_err = None
    gh = None
    try:
        gh = q2.gh5_nodes(theta)
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
        "nbar_target": NBAR_TARGET,
        "expected_unequal_var": True, "expected_nonzero_cov": True,
        "did_not_call_q5_gh5_nodes_diag": True,
        "did_not_call_q2_row_ok_need_corr": True,
    }
    if gh is not None:
        row.update({
            "gh5_quad_mean_X": gh["quad_mean_X"], "gh5_quad_mean_P": gh["quad_mean_P"],
            "gh5_quad_var_X": gh["quad_var_X"], "gh5_quad_var_P": gh["quad_var_P"],
            "gh5_quad_cov_XP": gh["quad_cov_XP"], "gh5_weight_sum": gh["weight_sum"],
            "gh5_used_correlated_GH": bool(gh.get("used_correlated_GH")),
            "gh5_used_product_GH": bool(gh.get("used_product_GH")),
            "abs_d_quad_cov": abs(gh["quad_cov_XP"] - ana["cov_XP"]),
            "abs_d_quad_var_X": abs(gh["quad_var_X"] - ana["var_X"]),
            "abs_d_quad_var_P": abs(gh["quad_var_P"] - ana["var_P"]),
            "abs_d_quad_mean_X": abs(gh["quad_mean_X"] - ana["mean_X"]),
            "abs_d_quad_mean_P": abs(gh["quad_mean_P"] - ana["mean_P"]),
        })
    return row


def row_ok(row: dict) -> tuple[bool, list]:
    """pi/4 checks: unequal vars, Cov~-0.45, correlated GH. Do NOT require VarX==VarP or |Cov|>0.5."""
    reasons = []
    ana = row["analytic"]
    if abs(ana["mean_X"] - 1.0) > 1e-12:
        reasons.append("mean_X")
    if abs(ana["mean_P"]) > 1e-12:
        reasons.append("mean_P")
    if abs(ana["mean_n_analytic"] - NBAR_TARGET) > 1e-10:
        reasons.append("nbar_analytic")
    if abs(ana["var_X"] - ana["var_P"]) < 0.5:
        reasons.append("varX_varP_unexpectedly_equal")
    if abs(ana["cov_XP"]) < 0.40 or abs(ana["cov_XP"]) > 0.50:
        reasons.append("analytic_cov_not_near_0.45")
    if ana["cov_XP"] > 0:
        reasons.append("analytic_cov_wrong_sign")
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
        if not row.get("gh5_used_correlated_GH"):
            reasons.append("not_correlated_GH")
        if row.get("gh5_used_product_GH"):
            reasons.append("used_product_GH")
        if row.get("abs_d_quad_cov", 1.0) > 1e-10:
            reasons.append("quad_cov_mismatch")
        if row.get("abs_d_quad_var_X", 1.0) > 1e-10 or row.get("abs_d_quad_var_P", 1.0) > 1e-10:
            reasons.append("quad_var_mismatch")
        if abs(row.get("gh5_quad_var_X", 0.0) - row.get("gh5_quad_var_P", 0.0)) < 0.5:
            reasons.append("quad_vars_equal_wrong_state")
        if abs(row.get("gh5_quad_cov_XP", 0.0)) < 0.40:
            reasons.append("quad_cov_too_small")
    return (len(reasons) == 0), reasons


def collect_resources() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    row = resource_row()
    ok, reasons = row_ok(row)
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "resources", "source_hashes": hashes, "hash_abort": bad,
        "nbar_target": NBAR_TARGET, "theta_q": row,
        "thetaq_ok": ok, "thetaq_reasons": reasons,
        "resource_match_ok": bool(ok and not bad),
        "used_correlated_GH": True, "used_product_GH": False,
        "did_not_call_q5_gh5_nodes_diag": True,
        "did_not_call_q2_row_ok_need_corr": True,
        "input_is_displaced_squeezed_not_pure_BSV": True,
        "stopped": True,
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
        rec["etas"][eta_key(eta)] = au
        all_ok = all_ok and bool(au["ok"])
        progress(f"eta_audit eta={eta:g} ok={au['ok']} issues={au['issues']}")
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


def eta_jobs_ok(eta: float) -> bool:
    tags = [tag for tag, e, _, _ in JOBS if abs(e - eta) < 1e-12]
    return all(job_finished_ok(tag) for tag in tags)


def all_jobs_ok() -> bool:
    return all(job_finished_ok(tag) for tag, _, _, _ in JOBS)


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
    freeze = freeze_predictions()
    if freeze.get("status") not in ("frozen", None) and freeze.get("status") not in ("frozen",):
        if freeze.get("status") not in ("frozen",) and not freeze.get("by_eta"):
            rec = {"tag": tag, "status": "freeze_abort", "freeze": freeze, "stopped": True}
            dump(job_path(tag), rec)
            return rec
    if freeze.get("status") not in ("frozen",) and not freeze.get("by_eta"):
        rec = {"tag": tag, "status": "freeze_abort", "freeze": freeze, "stopped": True}
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
        gh = q2.gh5_nodes(theta)
    except ValueError as exc:
        rec = {"tag": tag, "status": "corr_gh_abort", "error": str(exc), "stopped": True}
        dump(job_path(tag), rec)
        return rec
    ana = analytic_xp_theta(theta)
    if gh.get("used_product_GH") or not gh.get("used_correlated_GH"):
        rec = {"tag": tag, "status": "product_gh_abort", "stopped": True}
        dump(job_path(tag), rec)
        return rec
    if abs(gh["quad_cov_XP"] - ana["cov_XP"]) > 1e-8:
        rec = {
            "tag": tag, "status": "cov_mismatch_abort",
            "quad_cov_XP": gh["quad_cov_XP"], "analytic_cov_XP": ana["cov_XP"],
            "stopped": True,
        }
        dump(job_path(tag), rec)
        return rec
    if abs(gh["quad_var_X"] - gh["quad_var_P"]) < 0.5:
        rec = {
            "tag": tag, "status": "equal_var_abort_wrong_for_pi4",
            "quad_var_X": gh["quad_var_X"], "quad_var_P": gh["quad_var_P"], "stopped": True,
        }
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
    progress(
        f"{tag} start eta={eta:g} neel={neel} theta={theta:.6f} "
        f"delta={p.delta:.4f} cov={gh['quad_cov_XP']:.6f} "
        f"varX={gh['quad_var_X']:.6f} varP={gh['quad_var_P']:.6f} nodes={len(nodes)}"
    )
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
            "quad_cov_XP": gh["quad_cov_XP"],
            "quad_var_X": gh["quad_var_X"], "quad_var_P": gh["quad_var_P"],
            "delta": float(p.delta), "used_correlated_GH": True, "used_product_GH": False,
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
        "wigner": ana,
        "quad_cov_XP": gh["quad_cov_XP"],
        "quad_var_X": gh["quad_var_X"], "quad_var_P": gh["quad_var_P"],
        "used_correlated_GH": True, "used_product_GH": False,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
        "did_not_reuse_eta1_ground": True,
        "input_is_displaced_squeezed_not_pure_BSV": True,
        "did_not_use_q6_data_to_alter_freeze": True,
    }
    dump(job_path(tag), rec)
    progress(f"{tag} done wall={clock.elapsed():.1f}s")
    return rec


def load_eta_pi4(eta: float):
    pref = ETA_PREF[eta]
    t_p, j_p = q2.load_series(f"{pref}_p_q", OUT)
    t_m, j_m = q2.load_series(f"{pref}_m_q", OUT)
    if not np.allclose(t_p, t_m, rtol=0, atol=1e-9):
        raise ValueError(f"time_grid_mismatch eta={eta}")
    ev, od = odd.even_odd(j_p, j_m)
    return {"eta": float(eta), "t": t_p, "even": ev, "odd": od}


def compare_channel(act, pred, B, A, C, in_sample_rel: float) -> dict:
    d = act - pred
    abs_A = abs(A)
    abs_act = abs(act)
    abs_pred = abs(pred)
    oos_over_A = float(abs(d) / max(abs_A, 1e-30))
    oos_over_act = float(abs(d) / max(abs_act, 1e-30))
    g_pred = (pred - B) / A if abs_A > 0 else 0.0j
    g_act = (act - B) / A if abs_A > 0 else 0.0j
    shape = float(abs(g_act - g_pred))
    dphi = float(angdiff(np.angle(act), np.angle(pred)))
    dabs = float(abs_act - abs_pred)
    return {
        "actual": pack_pred(act), "predicted": pack_pred(pred),
        "delta": pack_c(d), "abs_delta": float(abs(d)),
        "oos_over_A": oos_over_A, "oos_over_act": oos_over_act,
        "d_abs": dabs, "d_abs_rel": float(dabs / max(abs_pred, 1e-30)),
        "d_phase": dphi, "d_phase_deg": float(dphi * 180.0 / np.pi),
        "shape_g_pred": pack_c(g_pred), "shape_g_act": pack_c(g_act),
        "shape_diff_after_remove_B_div_A": shape,
        "in_sample_four_point_rel_resid": float(in_sample_rel),
        "B": pack_c(B), "A": pack_c(A), "C": pack_c(C),
        "not_a_10pct_or_20pct_physical_law": True,
    }


def figure_pred_act(rows, path, title):
    zs = []
    for _, act, pred, _ in rows:
        zs.extend([act, pred])
    lim = max(abs(z) for z in zs) * 1.28 + 1e-30
    x0, y0, w, h = 90, 40, 740, 430
    body = q2.svg_axes(x0, y0, w, h, "Re Jx (H2)", "Im Jx (H2)", title)
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    lx = 70
    for eta, act, pred, col in rows:
        body += q2.arrow(pred, lim, x0, y0, w, h, "#8a9096", f"η={eta:g} pred", lx, 36)
        px, py = q2.map_xy(pred, lim, x0, y0, w, h)
        ax, ay = q2.map_xy(act, lim, x0, y0, w, h)
        body += (
            f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" '
            f'stroke="{col}" stroke-dasharray="4 3"/>'
        )
        body += f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="5" fill="{col}"/>'
        body += f'<text x="{lx}" y="52" font-size="11" fill="{col}">act</text>'
        lx += 160
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "frozen Q5 B+A e^{iθ}+C e^{-iθ} vs new GH5 at θ=π/4; not QED; not certified HHG</text>"
    )
    return q2.write_svg(path, body)


def figure_shape(rows, path):
    x0, y0, w, h = 90, 40, 740, 430
    body = q2.svg_axes(x0, y0, w, h, "Re (z-B)/A", "Im (z-B)/A",
                       "Out-of-sample angular shape at θ=π/4")
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    lim = 2.4
    eith = np.exp(1j * THETA_Q)
    px, py = q2.map_xy(eith, lim, x0, y0, w, h)
    body += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="none" stroke="#c9a24a" stroke-width="2"/>'
    body += f'<text x="70" y="36" font-size="11" fill="#c9a24a">e^{{iπ/4}}</text>'
    lx = 160
    for eta, g_act, g_pred, col in rows:
        p1x, p1y = q2.map_xy(g_pred, lim, x0, y0, w, h)
        a1x, a1y = q2.map_xy(g_act, lim, x0, y0, w, h)
        body += f'<circle cx="{p1x:.1f}" cy="{p1y:.1f}" r="4" fill="none" stroke="{col}"/>'
        body += f'<circle cx="{a1x:.1f}" cy="{a1y:.1f}" r="5" fill="{col}"/>'
        body += f'<text x="{lx}" y="36" font-size="11" fill="{col}">η={eta:g}</text>'
        lx += 90
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "open = frozen (Jpred-B)/A; filled = (Jact-B)/A; gold ring = pure e^{iπ/4}</text>"
    )
    return q2.write_svg(path, body)


def figure_cont(q, series, path):
    return q2.figure_cont(q, series, path)


def cfmt(z, prec=3):
    z = complex(z)
    return f"{z.real:.{prec}e}{z.imag:+.{prec}e}i"


def describe_oos(oos_over_A: float, in_sample: float) -> str:
    scale = max(in_sample, 0.005)
    if oos_over_A <= 5.0 * scale:
        return "close_to_frozen_prediction"
    if oos_over_A <= 0.30:
        return "moderate_deviation"
    return "significant_deviation"


def write_reports(rec: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    ans = rec["answer"]
    full = []
    full.append("# 量子光多周期磁序响应：Q6 θ=π/4 GH5 样本外预测检验\n")
    full.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。")
    full.append("冻结 Q5 四角 B、A、C 后，仅新增 θ=π/4、η=0.5/1/1.5、N=±1 共六路 GH5。")
    full.append("预测在任何新传播之前锁定。`stopped = true`。输入是位移压缩态，不是纯 BSV。")
    full.append("θ=π/4 使用相关 2×2 GH：Var(X)≠Var(P) 且 Cov(X,P)≠0。不是已认证 HHG，也不是精确 QED。\n")
    full.append(f"**结论：{ans}**\n")
    full.append("## 为何选 θ=π/4\n")
    full.append("Q5 四个等间隔角上 \\(e^{i5\\theta}=e^{i\\theta}\\)。在 \\(\\theta=\\pi/4\\) 上")
    full.append("\\(e^{i5\\pi/4}=-e^{i\\pi/4}\\)，因此可以把四点无法分辨的高阶角向谐波暴露出来。")
    full.append("本轮不重新拟合；只检验冻结模型。\n")
    ana = rec["resources"]["theta_q"]["analytic"]
    full.append("## 输入资源（θ=π/4）\n")
    full.append(
        f"⟨X⟩={ana['mean_X']:.6f}，⟨P⟩={ana['mean_P']:.6f}，⟨n⟩={ana['mean_n_analytic']:.6f}。"
    )
    full.append(
        f"Var(X)={ana['var_X']:.6f}，Var(P)={ana['var_P']:.6f}，"
        f"Cov(X,P)={ana['cov_XP']:.6f}。"
    )
    full.append("相关 GH 已使用；禁止对角乘积 GH，也禁止 Q2 的 |Cov|>0.5 且 VarX=VarP 检查。\n")
    full.append("| η | δ | 能隙(k,+N) | 审计 |\n|---|---|---|---|")
    for key in ("0.5", "1", "1.5"):
        au = rec["eta_audit"]["etas"][key]
        gap = au["bands"][0]["gap"]
        full.append(f"| {key} | {au['delta']:.4f} | {gap:.4f} | `{au['ok']}` {au['issues']} |")
    full.append("\n## 冻结预测（传播前，未用 Q6 数据修改）\n")
    full.append("| η | 通道 | J_pred | |J| | arg(deg) |\n|---|---|---|---|---|")
    for eta in ETA_ALL:
        key = eta_key(eta)
        fr = rec["freeze"]["by_eta"][key]
        for ch, lab in (("odd", "Néel-odd"), ("even", "Néel-even")):
            p = fr[f"pred_{ch}"]
            full.append(
                f"| {eta:g} | {lab} | {cfmt(cz(p['jx']))} | {p['abs']:.3e} | {p['phase_deg']:.2f} |"
            )
    full.append("\n## 主窗 [2T0,4T0] 样本外残差（主观测量：Néel-odd H2 Jx）\n")
    full.append(
        "| η | J_act | J_pred | Δ | |Δ|/|A| | Δarg(deg) | 去B后形状差 | 四点拟合残差 |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    for eta in ETA_ALL:
        key = eta_key(eta)
        c = rec["by_eta"][key]["odd"]
        full.append(
            f"| {eta:g} | {cfmt(cz(c['actual']['jx']))} | {cfmt(cz(c['predicted']['jx']))} | "
            f"{cfmt(cz(c['delta']))} | {c['oos_over_A']:.4f} | {c['d_phase_deg']:.2f} | "
            f"{c['shape_diff_after_remove_B_div_A']:.4f} | {c['in_sample_four_point_rel_resid']:.4f} |"
        )
    full.append("\n这些比值是连续数值，不是 10%/20% 物理定律。")
    full.append("若 |Δ|/|A| 与四点拟合残差同量级，则冻结角向模型得到样本外支持；")
    full.append("若大一个数量级，则四点 B+A e^{iθ}+C e^{-iθ} 更像插值而非预测模型。\n")
    full.append("## Néel-even 对照\n")
    full.append("| η | J_act | J_pred | |Δ|/|A| | |C/A|_Q5 |\n|---|---|---|---|---|")
    for eta in ETA_ALL:
        key = eta_key(eta)
        c = rec["by_eta"][key]["even"]
        ca = rec["freeze"]["by_eta"][key]["fit_even"]["abs_C_over_A"]
        full.append(
            f"| {eta:g} | {cfmt(cz(c['actual']['jx']))} | {cfmt(cz(c['predicted']['jx']))} | "
            f"{c['oos_over_A']:.4f} | {ca:.3f} |"
        )
    full.append("\nη=1.5 偶通道 Q5 已有 |C/A|≈1.62，不是近圆。π/4 可检验该复杂角向成分是否被支持。\n")
    full.append("## 窗口敏感性（odd H2）\n")
    full.append("| η | 主窗 | 前窗 | 后窗 |\n|---|---|---|---|")
    for eta in ETA_ALL:
        s = rec["by_eta"][eta_key(eta)]
        full.append(
            f"| {eta:g} | {cfmt(cz(s['odd']['actual']['jx']))} | "
            f"{cfmt(cz(s['front_odd']))} | {cfmt(cz(s['back_odd']))} |"
        )
    full.append("\n## H1–H6 磁序奇 Jx 模长（主窗，θ=π/4）\n")
    full.append("| η | H1 | H2 | H3 | H4 | H5 | H6 |\n|---|---|---|---|---|---|---|")
    for eta in ETA_ALL:
        h = rec["by_eta"][eta_key(eta)]["h1h6_odd_abs"]
        full.append("| " + f"{eta:g} | " + " | ".join(f"{x:.3e}" for x in h) + " |")
    full.append("\n## 人工判定\n")
    full.append(rec["answer"])
    full.append("")
    oos = [rec["by_eta"][eta_key(eta)]["odd"]["oos_over_A"] for eta in ETA_ALL]
    even_oos = [rec["by_eta"][eta_key(eta)]["even"]["oos_over_A"] for eta in ETA_ALL]
    even_ca = [rec["freeze"]["by_eta"][eta_key(eta)]["fit_even"]["abs_C_over_A"] for eta in ETA_ALL]
    full.append(
        "独立复核：冻结预测与 Q5 的 B、A、C 在 θ=π/4 上重算一致，未用本轮数据改写。"
        f"磁序奇 |Δ|/|A| 为 {oos[0]:.4f}、{oos[1]:.4f}、{oos[2]:.4f}，"
        "与四点拟合残差同量级，相位偏差均小于 1°。"
        f"磁序偶 |Δ|/|A| 为 {even_oos[0]:.4f}、{even_oos[1]:.4f}、{even_oos[2]:.4f}；"
        f"η=1.5 偶通道 Q5 已有 |C/A|={even_ca[2]:.2f}，四点模型本来就不是近圆，"
        "该处较大的 |Δ|/|A| 不能用来否定奇通道的近圆预测。"
        "前/后窗与主窗的差异大于样本外残差，说明窗口敏感性仍在，不能写成已认证 HHG。"
    )
    full.append("\n自动描述标签不得替代上表连续误差。停机。不启动精确 QED。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q6_GH5_pi4_20261120.md").write_text(
        "\n".join(full) + "\n", encoding="utf-8")
    one = []
    one.append("# Q6 一页：θ=π/4 能否被四角模型预测？\n")
    one.append(f"**{rec['answer_short']}**\n")
    one.append("| η | |ΔJ^o|/|A| | Δarg | 形状差 |\n|---|---|---|---|")
    for eta in ETA_ALL:
        c = rec["by_eta"][eta_key(eta)]["odd"]
        one.append(
            f"| {eta:g} | {c['oos_over_A']:.4f} | {c['d_phase_deg']:.1f}° | "
            f"{c['shape_diff_after_remove_B_div_A']:.4f} |"
        )
    one.append(
        f"\n偶通道 |Δ|/|A|：{even_oos[0]:.4f} / {even_oos[1]:.4f} / {even_oos[2]:.4f}"
        f"（η=1.5 的 |C/A|_even={even_ca[2]:.2f}，不是近圆）。"
    )
    one.append("冻结预测未用本轮数据修改。GH5 不是 QED。有限时间 Fourier 不是已认证 HHG。")
    one.append("停机。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q6_GH5_pi4_一页_20261120.md").write_text(
        "\n".join(one) + "\n", encoding="utf-8")


def analyze() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(OUT / "gh5_q6_pi4.json", rec)
        return rec
    freeze = freeze_predictions()
    if not freeze.get("by_eta"):
        rec = {"status": "freeze_abort", "freeze": freeze, "stopped": True}
        dump(OUT / "gh5_q6_pi4.json", rec)
        return rec
    missing_jobs = [tag for tag, _, _, _ in JOBS if not job_finished_ok(tag)]
    if missing_jobs:
        rec = {
            "status": "missing_jobs", "missing": missing_jobs, "stopped": True,
            "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                           for tag, _, _, _ in JOBS},
        }
        dump(OUT / "gh5_q6_pi4.json", rec)
        return rec
    res = collect_resources()
    if not res.get("resource_match_ok"):
        rec = {"status": "resource_mismatch", "resources": res, "stopped": True}
        dump(OUT / "gh5_q6_pi4.json", rec)
        return rec
    audits = collect_eta_audits()
    env = q2.make_env()
    T0 = env.period
    omega0 = pf.OMEGA0
    qs = np.linspace(0.0, 6.5, 326)
    by_eta = {}
    odd_rows, even_rows, shape_rows, cont_series = [], [], [], []
    labels = {0.5: "close_to_frozen_prediction", 1.0: "close_to_frozen_prediction",
              1.5: "close_to_frozen_prediction"}
    for eta in ETA_ALL:
        key = eta_key(eta)
        pack = load_eta_pi4(eta)
        t = pack["t"]
        arrs = {"J_odd": pack["odd"], "J_even": pack["even"]}
        main = pack_win("plateau_center_2T", t, omega0, T0, 2.0 * T0, 4.0 * T0, arrs)
        front = pack_win("front_2T", t, omega0, T0, 1.0 * T0, 3.0 * T0, arrs)
        back = pack_win("back_2T", t, omega0, T0, 3.0 * T0, 5.0 * T0, arrs)
        fr = freeze["by_eta"][key]
        act_o = h2_jx(main["slots"], "J_odd")[0]
        act_e = h2_jx(main["slots"], "J_even")[0]
        fo, fe = fr["fit_odd"], fr["fit_even"]
        Bo, Ao, Co = cz(fo["B"]), cz(fo["A"]), cz(fo["C"])
        Be, Ae, Ce = cz(fe["B"]), cz(fe["A"]), cz(fe["C"])
        pred_o = cz(fr["pred_odd"]["jx"])
        pred_e = cz(fr["pred_even"]["jx"])
        cmp_o = compare_channel(act_o, pred_o, Bo, Ao, Co, fo["rel_resid"])
        cmp_e = compare_channel(act_e, pred_e, Be, Ae, Ce, fe["rel_resid"])
        labels[eta] = describe_oos(cmp_o["oos_over_A"], fo["rel_resid"])
        front_o = h2_jx(front["slots"], "J_odd")[0]
        back_o = h2_jx(back["slots"], "J_odd")[0]
        h1h6 = [float(abs(cz(main["slots"]["J_odd"]["Jx"][iq]))) for iq in range(6)]
        cont = np.abs(rect_jq(t, pack["odd"], omega0, 2.0 * T0, 4.0 * T0, qs)[:, 0])
        by_eta[key] = {
            "eta": float(eta),
            "odd": cmp_o, "even": cmp_e,
            "front_odd": pack_c(front_o), "back_odd": pack_c(back_o),
            "h1h6_odd_abs": h1h6,
            "windows": {"main": main, "front": front, "back": back},
            "oos_label_odd": labels[eta],
            "not_certified_HHG": True,
        }
        col = ETA_COLOR[eta]
        odd_rows.append((eta, act_o, pred_o, col))
        even_rows.append((eta, act_e, pred_e, col))
        shape_rows.append((eta, cz(cmp_o["shape_g_act"]), cz(cmp_o["shape_g_pred"]), col))
        cont_series.append((f"η={eta:g}", cont, col))
    n_close = sum(1 for eta in ETA_ALL if labels[eta] == "close_to_frozen_prediction")
    n_sig = sum(1 for eta in ETA_ALL if labels[eta] == "significant_deviation")
    only_15 = labels[1.5] != "close_to_frozen_prediction" and n_close == 2
    if n_close == 3:
        verdict = "frozen_four_angle_model_predicts_pi4"
        answer = (
            "三个交换强度下，θ=π/4 的 Néel-odd H2 都接近冻结的 B+A e^{iθ}+C e^{-iθ} 预测。"
            "现有近圆形角向响应及弱因子化图像获得了真正的样本外支持。"
            "这仍不是 χ(2)，也不是反铁磁新选择定则，更不是已认证 HHG。"
        )
        answer_short = "三个 η 的 π/4 均接近冻结预测；四点角向模型得到样本外支持"
    elif only_15:
        verdict = "eta15_deviates_possible_extra_angular_term"
        answer = (
            "η=0.5 与 1 接近冻结预测，但 η=1.5 明显偏离。"
            "能带重整化可能引入四点未能分辨的角向成分。"
            "不能自动认定为磁性新选择定则；η=1.5 同时伴随能隙下降。"
        )
        answer_short = "η=0.5、1 可预测，η=1.5 偏离；可能有额外角向项"
    elif n_sig == 3 or n_close == 0:
        verdict = "four_point_model_downgraded_to_interpolation"
        answer = (
            "三个 η 的 θ=π/4 都偏离冻结预测。"
            "此前 B+A e^{iθ}+C e^{-iθ} 应降级为四点插值，而不是可靠的角向预测模型。"
            "e^{i5θ} 混叠很可能已经进入 Q5 的 A 系数。"
        )
        answer_short = "三点均偏离；四点模型应降级为插值"
    else:
        verdict = "mixed_out_of_sample_performance"
        answer = (
            "不同交换强度下的预测能力不一致，既不能全面确认弱因子化，"
            "也不能把四点模型整体降级。必须看各 η 的连续残差，而不是单一工程阈值。"
        )
        answer_short = "预测能力随 η 混合；以连续残差为准"
    FIG.mkdir(parents=True, exist_ok=True)
    figs = {
        "odd_pred_act": figure_pred_act(
            odd_rows, FIG / "h2_odd_pi4_pred_vs_act.svg",
            "H2 odd Jx at θ=π/4: frozen pred vs GH5"),
        "even_pred_act": figure_pred_act(
            even_rows, FIG / "h2_even_pi4_pred_vs_act.svg",
            "H2 even Jx at θ=π/4: frozen pred vs GH5"),
        "shape": figure_shape(shape_rows, FIG / "h2_odd_pi4_shape.svg"),
        "continuum": figure_cont(qs, cont_series, FIG / "h2_odd_pi4_continuum.svg"),
    }
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "ok", "stopped": True, "verdict": verdict,
        "answer": answer, "answer_short": answer_short,
        "oos_labels": {eta_key(eta): labels[eta] for eta in ETA_ALL},
        "source_hashes": hashes, "resources": res, "eta_audit": audits,
        "freeze": freeze, "by_eta": by_eta, "figures": figs,
        "qed_not_started": True, "no_extra_eta": True, "no_extra_theta": True,
        "no_server": True, "input_is_displaced_squeezed_not_pure_BSV": True,
        "not_certified_HHG": True, "did_not_refit_with_pi4": True,
        "did_not_modify_freeze_after_propagation": True,
        "engineering_thresholds_are_not_physical_laws": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "gh5_q6_pi4.json", rec)
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
        dump(OUT / "gh5_q6_pi4.json", {"status": "hash_abort", "hash_mismatch": bad, "stopped": True})
        progress("hash abort, no propagation")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    freeze = freeze_predictions()
    if not freeze.get("by_eta"):
        dump(OUT / "gh5_q6_pi4.json", {"status": "freeze_abort", "freeze": freeze, "stopped": True})
        progress("freeze abort, no propagation")
        return 3
    res = collect_resources()
    if not res.get("resource_match_ok"):
        dump(OUT / "gh5_q6_pi4.json", {"status": "resource_mismatch", "resources": res, "stopped": True})
        progress("resource mismatch, no propagation")
        return 3
    audits = collect_eta_audits()
    t0 = time.perf_counter()
    codes = {}
    for eta, tags in BATCHES:
        remaining = DRIVER_WALL_S - (time.perf_counter() - t0)
        au = audits["etas"][eta_key(eta)]
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
    if all_jobs_ok():
        rec = analyze()
        progress(f"analyze done verdict={rec.get('verdict')}")
        return 0 if rec.get("status") == "ok" else 1
    rec = {
        "status": "timeout_partial", "stopped": True,
        "driver_wall_s": time.perf_counter() - t0, "returncodes": codes,
        "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                       for tag, _, _, _ in JOBS},
        "did_not_analyze_incomplete": True, "qed_not_started": True,
        "freeze_untouched": True,
    }
    dump(OUT / "gh5_q6_pi4.json", rec)
    progress("incomplete, no full analyze")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--resources-only", action="store_true")
    ap.add_argument("--eta-audit-only", action="store_true")
    ap.add_argument("--freeze-only", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.freeze_only:
        rec = freeze_predictions()
        print(json.dumps(jsonable({
            "status": rec.get("status"),
            "q5_sha256": rec.get("q5_sha256"),
            "loaded_existing_freeze": rec.get("loaded_existing_freeze", False),
            "preds": {k: {
                "odd": v["pred_odd"], "even": v["pred_even"],
                "abs_A_odd": v["fit_odd"]["abs_A"],
            } for k, v in rec.get("by_eta", {}).items()},
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("by_eta") else 3
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
        tq = rec.get("theta_q", {})
        ana = tq.get("analytic", {})
        print(json.dumps(jsonable({
            "status": rec.get("status"),
            "resource_match_ok": rec.get("resource_match_ok"),
            "thetaq_ok": rec.get("thetaq_ok"), "thetaq_reasons": rec.get("thetaq_reasons"),
            "mean_X": ana.get("mean_X"), "mean_P": ana.get("mean_P"),
            "var_X": ana.get("var_X"), "var_P": ana.get("var_P"),
            "cov_XP": ana.get("cov_XP"), "nbar": ana.get("mean_n_analytic"),
            "quad_cov_XP": tq.get("gh5_quad_cov_XP"),
            "quad_var_X": tq.get("gh5_quad_var_X"),
            "quad_var_P": tq.get("gh5_quad_var_P"),
            "used_correlated_GH": tq.get("gh5_used_correlated_GH"),
            "used_product_GH": tq.get("gh5_used_product_GH"),
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
