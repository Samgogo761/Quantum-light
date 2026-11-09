#!/usr/bin/env python3
"""Fixed-nbar GH5 cancellation screen, 1600 nm protocol A.

Hold nbar of D(0.5)S(0.3 e^{i theta})|0>, move to R_*=1:
  r_*=0.26102080980486986, |alpha_*|=0.5225316298753016.
Product GH5, cov=0, theta=0/pi only, four parallel jobs.
Reuse protocol-A nodes; do not write the 20261102 OUT.
Two-angle proxy cannot split A vs C. Not QED.
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

import q8_m0_qinput_1600nm_protocolA_gh5_20261102 as proto
import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
from q8_m0_model import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_1600nm_protocolA_gh5_nbar_cancel_20261109"
DOCS = ROOT / "docs/04_BSV量子光"
REPORT = DOCS / "线Q_共同核零假设_冻结复算_20261109.md"
TASK_TAG = "20261109"
RECORD_DATE = "2026-11-09"
PROTOCOL = "Qin_a2_1_4_1_1600nm_protocolA_nbar_cancel"
FROZEN_SHA = "17a8e37ec3e0ab54096ff4edd1440c5a3f0412b09450c2fd8960cbc00d8e69db"
OLD_OUT = ROOT / "results/q_line/qinput_1600nm_protocolA_gh5_20261102"

NBAR_HELD = 0.34273260912113385
R_STAR = 0.26102080980486986
ALPHA_STAR = 0.5225316298753016
R_RATIO_STAR = 1.0
OLD_ALPHA = 0.5
OLD_R = 0.3

OMEGA0_1600 = proto.OMEGA0_1600
G_ABS = proto.G_ABS
N_FULL = proto.N_FULL
ORDER = proto.ORDER
JOB_WALL_S = proto.JOB_WALL_S
DRIVER_WALL_S = proto.DRIVER_WALL_S
THETA_0 = 0.0
THETA_PI = float(np.pi)
TAGS = ("p_0", "m_0", "p_pi", "m_pi")
RHO_TRACK_MARK = 0.02
ARG_TRACK_DEG = 10.0

jsonable = proto.jsonable
pack_c = proto.pack_c
cz = proto.cz
wrap = proto.wrap
sample_js_for = proto.sample_js_for
one_traj = proto.one_traj
rect_jq = proto.rect_jq
make_bundle = proto.make_bundle
envelope_meta = proto.envelope_meta


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


def field_R(alpha: float, r: float) -> float:
    return float(np.sinh(2.0 * r) / (2.0 * alpha * alpha))


def field_nbar(alpha: float, r: float) -> float:
    return float(alpha * alpha + np.sinh(r) ** 2)


def analytic_xp_theta(theta: float, alpha: float, r: float) -> dict:
    a = complex(float(alpha))
    mx, mp = 2.0 * a.real, 2.0 * a.imag
    rr = float(r)
    th = float(theta)
    if abs(th) < 1e-15:
        vx, vp = float(np.exp(-2.0 * rr)), float(np.exp(2.0 * rr))
        axis = "X_squeezed"
    elif abs(th - np.pi) < 1e-12:
        vx, vp = float(np.exp(2.0 * rr)), float(np.exp(-2.0 * rr))
        axis = "P_squeezed"
    else:
        raise ValueError("nbar-cancel ships theta=0 or pi only; product GH requires Cov=0")
    nbar = field_nbar(float(alpha), rr)
    Mm = -0.5 * np.exp(1j * th) * np.sinh(2.0 * rr)
    return {
        "kind": "ds", "alpha": float(alpha), "r": rr, "theta": th,
        "mean_X": mx, "mean_P": mp, "var_X": vx, "var_P": vp,
        "cov_XP": 0.0, "mean_n_analytic": float(nbar),
        "mean_n_Weyl": 0.25 * (mx * mx + mp * mp + vx + vp) - 0.5,
        "M": proto.pack_c(complex(Mm)) if False else {
            "re": float(np.real(Mm)), "im": float(np.imag(Mm)),
        },
        "product_GH_valid": True, "squeeze_axis": axis,
        "R_M_over_alpha2": field_R(float(alpha), rr),
        "did_not_call_proto_analytic_xp_theta": True,
    }


def gh5_nodes(theta: float, alpha: float, r: float) -> dict:
    xi, w1 = hermgauss(ORDER)
    mom = analytic_xp_theta(theta, alpha, r)
    if abs(mom["cov_XP"]) > 1e-10:
        raise ValueError("product GH requires Cov(X,P)=0")
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
    cxp = sum(n["weight"] * n["X"] * n["P"] for n in nodes) - mx * mpv
    return {
        "kind": "ds", "theta": float(theta), "order": ORDER, "n_nodes": len(nodes),
        "weight_sum": wsum, "quad_mean_X": float(mx), "quad_mean_P": float(mpv),
        "quad_var_X": float(vx), "quad_var_P": float(vp), "quad_cov_XP": float(cxp),
        "target_mean_X": mom["mean_X"], "target_mean_P": mom["mean_P"],
        "target_var_X": mom["var_X"], "target_var_P": mom["var_P"],
        "target_cov_XP": 0.0, "nodes": nodes, "analytic": mom,
        "used_product_GH": True, "used_correlated_GH": False,
        "did_not_call_proto_gh5_nodes": True,
        "alpha": float(alpha), "r": float(r),
    }


def two_angle_proxy(z0, zpi) -> dict:
    z0, zpi = complex(z0), complex(zpi)
    B = 0.5 * (z0 + zpi)
    AC = 0.5 * (z0 - zpi)
    arg = float(np.degrees(wrap(np.angle(AC) - np.angle(B)))) if abs(B) > 0 else None
    dist = None if arg is None else float(abs(abs(arg) - 180.0))
    return {
        "B_proxy": pack_c(B),
        "A_plus_C_proxy": pack_c(AC),
        "abs_AC_over_B": float(abs(AC) / max(abs(B), 1e-30)),
        "J0": pack_c(z0), "Jpi": pack_c(zpi),
        "amp_ratio_pi_over_0": float(abs(zpi) / max(abs(z0), 1e-30)),
        "arg_AC_over_B_deg": arg,
        "arg_distance_to_pi_deg": dist,
        "cannot_separate_A_and_C": True,
    }


def job_spec(tag: str) -> dict:
    table = {
        "p_0": {"neel": 1, "theta": THETA_0},
        "m_0": {"neel": -1, "theta": THETA_0},
        "p_pi": {"neel": 1, "theta": THETA_PI},
        "m_pi": {"neel": -1, "theta": THETA_PI},
    }
    if tag not in table:
        raise ValueError(tag)
    spec = dict(table[tag])
    spec["tag"] = tag
    spec["alpha"] = ALPHA_STAR
    spec["r"] = R_STAR
    spec["n_steps"] = N_FULL
    return spec


def ckpt_path(tag: str) -> Path:
    return OUT / f"ckpt_{tag}.npz"


def job_path(tag: str) -> Path:
    return OUT / f"job_{tag}.json"


def series_path(tag: str) -> Path:
    return OUT / f"series_{tag}.npz"


def save_ckpt(tag, neel, theta, n_done, n_nodes, acc, wsum, max_n, t_arr) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc = np.asarray(acc, complex)
    np.savez_compressed(
        ckpt_path(tag),
        acc_re=acc.real, acc_im=acc.imag, wsum=float(wsum),
        n_done=int(n_done), n_nodes=int(n_nodes), max_n=float(max_n),
        t=np.asarray(t_arr, float), neel=int(neel), theta=float(theta),
        n_steps=int(N_FULL), omega0=float(OMEGA0_1600),
        alpha=float(ALPHA_STAR), r=float(R_STAR),
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
        "theta": float(z["theta"]) if "theta" in z.files else None,
        "n_steps": int(z["n_steps"]) if "n_steps" in z.files else None,
        "omega0": float(z["omega0"]) if "omega0" in z.files else None,
        "alpha": float(z["alpha"]) if "alpha" in z.files else None,
        "r": float(z["r"]) if "r" in z.files else None,
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


def load_series(tag: str):
    z = np.load(series_path(tag))
    t = np.asarray(z["t"], float)
    j = np.stack([z["Jx"], z["Jy"]], axis=1)
    return t, j


def ckpt_mismatch(ck, theta) -> str | None:
    if ck is None:
        return None
    if ck.get("theta") is None or abs(float(ck["theta"]) - float(theta)) > 1e-12:
        return "ckpt_theta_mismatch"
    if ck.get("n_steps") is None or int(ck["n_steps"]) != int(N_FULL):
        return "ckpt_n_steps_mismatch"
    if ck.get("omega0") is None or abs(float(ck["omega0"]) - OMEGA0_1600) > 1e-12:
        return "ckpt_omega0_mismatch"
    if ck.get("alpha") is None or abs(float(ck["alpha"]) - ALPHA_STAR) > 1e-12:
        return "ckpt_alpha_mismatch"
    if ck.get("r") is None or abs(float(ck["r"]) - R_STAR) > 1e-12:
        return "ckpt_r_mismatch"
    return None


def source_hashes() -> dict:
    out = proto.source_hashes()
    out[Path(__file__).name] = sha256_file(Path(__file__))
    return out


def working_point_ok() -> dict:
    nbar = field_nbar(ALPHA_STAR, R_STAR)
    ratio = field_R(ALPHA_STAR, R_STAR)
    old_nbar = field_nbar(OLD_ALPHA, OLD_R)
    rec = {
        "alpha_star": ALPHA_STAR, "r_star": R_STAR, "R_star": ratio,
        "nbar_star": nbar, "nbar_held": NBAR_HELD, "old_nbar": old_nbar,
        "nbar_match": bool(abs(nbar - NBAR_HELD) < 1e-12),
        "old_nbar_match": bool(abs(old_nbar - NBAR_HELD) < 1e-12),
        "R_is_one": bool(abs(ratio - 1.0) < 1e-10),
        "did_not_use_proto_ALPHA": True,
        "did_not_write_20261102": True,
    }
    return rec


def collect_resources() -> dict:
    hashes = source_hashes()
    bad = proto.hash_abort(hashes)
    env = proto.make_env()
    wp = working_point_ok()
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "resources", "source_hashes": hashes, "hash_abort": bad,
        "inherited_freeze_sha256": FROZEN_SHA,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
        "alpha": ALPHA_STAR, "r": R_STAR, "R_target": R_RATIO_STAR,
        "nbar": NBAR_HELD, "omega0": OMEGA0_1600, "g": G_ABS,
        "n_max_electronic_fock": 0, "n_full": N_FULL,
        "gh5_is_not_cohen_mtef": True, "gh5_is_not_qed": True,
        "envelope": envelope_meta(env),
        "working_point": wp,
        "old_out_untouched": str(OLD_OUT),
        "stopped": True,
    }
    if bad:
        rec["status"] = "hash_abort"
        dump(OUT / "resources.json", rec)
        return rec
    rows = {}
    for key, th in (("0", THETA_0), ("pi", THETA_PI)):
        gh = gh5_nodes(th, ALPHA_STAR, R_STAR)
        ana = gh["analytic"]
        rows[key] = {
            "analytic": ana,
            "gh5_quad_mean_X": gh["quad_mean_X"],
            "gh5_quad_mean_P": gh["quad_mean_P"],
            "gh5_quad_var_X": gh["quad_var_X"],
            "gh5_quad_var_P": gh["quad_var_P"],
            "gh5_quad_cov_XP": gh["quad_cov_XP"],
            "gh5_weight_sum": gh["weight_sum"],
            "gh5_n_nodes": gh["n_nodes"],
            "used_product_GH": gh["used_product_GH"],
            "abs_d_quad_cov": abs(gh["quad_cov_XP"]),
            "abs_d_nbar": abs(ana["mean_n_analytic"] - NBAR_HELD),
        }
    rec["rows"] = rows
    rec["product_at_0_pi"] = bool(rows["0"]["used_product_GH"] and rows["pi"]["used_product_GH"])
    rec["resource_match_ok"] = bool(
        wp["nbar_match"] and wp["old_nbar_match"] and wp["R_is_one"]
        and rec["product_at_0_pi"]
        and abs(rows["0"]["gh5_quad_cov_XP"]) < 1e-12
        and abs(rows["pi"]["gh5_quad_cov_XP"]) < 1e-12
        and abs(rec["envelope"]["omega0"] - OMEGA0_1600) < 1e-15
    )
    dump(OUT / "resources.json", rec)
    progress(
        f"resources match={rec['resource_match_ok']} alpha={ALPHA_STAR:.6f} "
        f"r={R_STAR:.6f} nbar={NBAR_HELD:.6f} R={field_R(ALPHA_STAR, R_STAR):.6f} "
        f"omega0={OMEGA0_1600} wrote_20261102=False"
    )
    return rec


def run_one_job(tag: str) -> dict:
    spec = job_spec(tag)
    neel, theta = spec["neel"], spec["theta"]
    hashes = source_hashes()
    bad = proto.hash_abort(hashes)
    if bad:
        rec = {"tag": tag, "status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(job_path(tag), rec)
        return rec
    if job_finished_ok(tag):
        rec = load_job(tag)
        if (rec.get("omega0") != OMEGA0_1600 or rec.get("n_steps") != N_FULL
                or abs(float(rec.get("alpha", 0.0)) - ALPHA_STAR) > 1e-12
                or abs(float(rec.get("r", 0.0)) - R_STAR) > 1e-12):
            rec = {"tag": tag, "status": "finished_contract_mismatch", "stopped": True,
                   "have": rec, "want_alpha": ALPHA_STAR, "want_r": R_STAR}
            dump(job_path(tag), rec)
            return rec
        progress(f"{tag} already complete n_run={rec.get('n_run')}")
        return rec
    OUT.mkdir(parents=True, exist_ok=True)
    if OUT.resolve() == OLD_OUT.resolve():
        raise RuntimeError("refusing to write 20261102 OUT")
    _, _, models, grounds, env = make_bundle(OMEGA0_1600)
    if abs(float(env.omega0) - OMEGA0_1600) > 1e-15:
        raise RuntimeError("env silently used Q6 omega0")
    gh = gh5_nodes(theta, ALPHA_STAR, R_STAR)
    nodes = gh["nodes"]
    ek0, emk0, mass = grounds[neel]
    model = models[neel]
    js = sample_js_for(N_FULL)
    ck = load_ckpt(tag)
    mismatch = ckpt_mismatch(ck, theta)
    if mismatch:
        rec = {"tag": tag, "status": mismatch, "stopped": True,
               "ckpt": {k: ck.get(k) for k in ("theta", "n_steps", "omega0", "alpha", "r", "n_done")},
               "job_theta": float(theta), "job_alpha": ALPHA_STAR, "job_r": R_STAR}
        dump(job_path(tag), rec)
        return rec
    i0, acc, wsum, max_n, t_arr = 0, None, 0.0, 0.0, None
    if ck is not None:
        i0, acc, wsum, max_n, t_arr = ck["n_done"], ck["acc"], ck["wsum"], ck["max_n"], ck["t"]
        progress(f"{tag} resume n_done={i0}/{len(nodes)}")
    clock = dyn.WallClock(JOB_WALL_S)
    progress(f"{tag} start neel={neel} theta={theta:.6f} alpha={ALPHA_STAR:.6f} "
             f"r={R_STAR:.6f} omega0={OMEGA0_1600} n_steps={N_FULL} "
             f"nodes={len(nodes)} product_GH={gh['used_product_GH']}")
    timed_out = False
    timeout_msg = None
    max_weyl_n = 0.0
    try:
        for i, node in enumerate(nodes):
            if i < i0:
                continue
            j_rows, t_rows, nrm, Xf, Pf = one_traj(
                model, ek0, emk0, node["X"], node["P"], env, N_FULL, js, clock)
            w = float(node["weight"])
            acc = w * j_rows if acc is None else acc + w * j_rows
            wsum += w
            max_n = max(max_n, nrm)
            max_weyl_n = max(max_weyl_n, 0.25 * (Xf * Xf + Pf * Pf) - 0.5)
            t_arr = t_rows
            save_ckpt(tag, neel, theta, i + 1, len(nodes), acc, wsum, max_n, t_arr)
            if (i + 1) == len(nodes) or (i + 1) % 5 == 0:
                progress(f"{tag} node {i + 1}/{len(nodes)} wall={clock.elapsed():.1f}s")
    except TimeoutError as exc:
        timed_out = True
        timeout_msg = str(exc)
        progress(f"{tag} timeout {timeout_msg}")
    n_done = 0 if acc is None else int(load_ckpt(tag)["n_done"] if ckpt_path(tag).is_file() else 0)
    if timed_out or n_done < len(nodes):
        rec = {
            "tag": tag, "neel": int(neel), "theta": float(theta),
            "status": "timeout_partial", "n_nodes": len(nodes), "n_done": n_done,
            "weight_sum": wsum, "max_abs_norm_minus_1": float(max_n),
            "max_weyl_n": float(max_weyl_n),
            "wall_s": clock.elapsed(), "timeout_msg": timeout_msg,
            "omega0": OMEGA0_1600, "n_steps": int(N_FULL),
            "alpha": ALPHA_STAR, "r": R_STAR, "g": G_ABS,
            "used_product_GH": True, "stopped": True,
            "did_not_call_proto_run_one_job": True,
        }
        dump(job_path(tag), rec)
        return rec
    j = acc / wsum
    save_series(tag, t_arr, j)
    rec = {
        "tag": tag, "neel": int(neel), "theta": float(theta), "status": "ok",
        "n_nodes": len(nodes), "n_run": n_done, "weight_sum": wsum,
        "max_abs_norm_minus_1": float(max_n), "max_weyl_n": float(max_weyl_n),
        "wall_s": clock.elapsed(), "omega0": OMEGA0_1600, "n_steps": int(N_FULL),
        "dt": float(env.t_total / N_FULL), "alpha": ALPHA_STAR, "r": R_STAR,
        "used_pf_OMEGA0": False, "g": G_ABS,
        "used_correlated_GH": False, "used_product_GH": True,
        "mass": mass, "envelope": envelope_meta(env),
        "gh5_is_not_cohen_mtef": True, "not_macroscopic_BSV": True,
        "did_not_call_proto_run_one_job": True,
        "did_not_write_20261102": True,
    }
    dump(job_path(tag), rec)
    progress(f"{tag} done wall={clock.elapsed():.1f}s alpha={ALPHA_STAR:.6f} r={R_STAR:.6f}")
    return rec


def load_pair(tag_p: str, tag_m: str):
    t_p, j_p = load_series(tag_p)
    t_m, j_m = load_series(tag_m)
    if not np.allclose(t_p, t_m, rtol=0, atol=1e-9):
        raise ValueError(f"time_grid_mismatch {tag_p} {tag_m}")
    return t_p, j_p, j_m


def window_proxy(t, odd0, oddpi, omega0, T0, t0, t1) -> dict:
    z0 = rect_jq(t, odd0, omega0, t0, t1, np.array([2.0]))[0, 0]
    zpi = rect_jq(t, oddpi, omega0, t0, t1, np.array([2.0]))[0, 0]
    pr = two_angle_proxy(z0, zpi)
    rho = float(pr["abs_AC_over_B"])
    return {
        "t0_over_T": float(t0 / T0), "t1_over_T": float(t1 / T0),
        "proxy": pr,
        "rho_proxy_AC_over_B": rho,
        "rho_over_Rstar": float(rho / max(R_RATIO_STAR, 1e-30)),
        "amp_Jx_pi_over_0": float(pr["amp_ratio_pi_over_0"]),
        "arg_AC_over_B_deg": pr["arg_AC_over_B_deg"],
        "arg_distance_to_pi_deg": pr["arg_distance_to_pi_deg"],
        "odd0_h2x": pack_c(z0), "oddpi_h2x": pack_c(zpi),
    }


def analyze() -> dict:
    hashes = source_hashes()
    bad = proto.hash_abort(hashes)
    if bad:
        rec = {"status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(OUT / "nbar_cancel.json", rec)
        return rec
    missing = [tag for tag in TAGS if not job_finished_ok(tag)]
    if missing:
        rec = {"status": "missing_jobs", "missing": missing, "stopped": True}
        dump(OUT / "nbar_cancel.json", rec)
        return rec
    env = proto.make_env()
    T0 = env.period
    omega0 = OMEGA0_1600
    t0, jp0, jm0 = load_pair("p_0", "m_0")
    tpi, jppi, jmpi = load_pair("p_pi", "m_pi")
    if not np.allclose(t0, tpi, rtol=0, atol=1e-9):
        rec = {"status": "time_grid_mismatch", "stopped": True}
        dump(OUT / "nbar_cancel.json", rec)
        return rec
    _, odd0 = odd.even_odd(jp0, jm0)
    _, oddpi = odd.even_odd(jppi, jmpi)
    main = window_proxy(t0, odd0, oddpi, omega0, T0, 2.0 * T0, 4.0 * T0)
    front = window_proxy(t0, odd0, oddpi, omega0, T0, 1.0 * T0, 3.0 * T0)
    back = window_proxy(t0, odd0, oddpi, omega0, T0, 3.0 * T0, 5.0 * T0)
    rho = main["rho_proxy_AC_over_B"]
    d_rho = abs(rho - 1.0)
    d_arg = main["arg_distance_to_pi_deg"]
    tracks = bool(d_rho <= RHO_TRACK_MARK and (d_arg is not None and d_arg <= ARG_TRACK_DEG))
    if tracks:
        verdict = "gh5_tracks_R_at_cancellation_STOP"
        answer = ("固定 nbar 的 GH5 相消点仍落在 R=1 上：θ=0 近相消，"
                  "|A+C|/|B|≈1。平均场共同核未被材料改写。停止扩 A/B/C，不开 QED。")
    else:
        verdict = "gh5_deviates_at_cancellation_still_no_QED"
        answer = ("固定 nbar 的 GH5 相消点偏离 R=1，但仍是平均场/有限窗问题，"
                  "不是精确量子修正。不开新 QED。")
    rec = {
        "status": "ok", "mode": "nbar_cancel", "verdict": verdict, "answer": answer,
        "tracks_R": tracks, "rho_track_mark": RHO_TRACK_MARK,
        "arg_track_deg": ARG_TRACK_DEG,
        "inherited_freeze_sha256": FROZEN_SHA,
        "alpha": ALPHA_STAR, "r": R_STAR, "R_star": R_RATIO_STAR,
        "nbar": NBAR_HELD, "omega0": omega0,
        "envelope": envelope_meta(env),
        "windows": {"main": main, "front": front, "back": back},
        "abs_rho_minus_1_main": d_rho,
        "cannot_separate_A_and_C": True,
        "gh5_is_not_qed": True, "did_not_start_new_QED": True,
        "did_not_expand_ABC": True, "did_not_write_20261102": True,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "nbar_cancel.json", rec)
    return rec


def launch_tags(tags) -> dict:
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
    return {
        "returncodes": {tag: (procs[tag].returncode if tag in procs else 0) for tag in tags},
        "driver_wall_s": time.perf_counter() - t0,
    }


def drive() -> int:
    hashes = source_hashes()
    bad = proto.hash_abort(hashes)
    if bad:
        dump(OUT / "nbar_cancel.json", {"status": "hash_abort", "hash_mismatch": bad, "stopped": True})
        progress("hash abort, no propagation")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    if OUT.resolve() == OLD_OUT.resolve():
        raise RuntimeError("refusing to write 20261102 OUT")
    res = collect_resources()
    if not res.get("resource_match_ok"):
        dump(OUT / "nbar_cancel.json", {"status": "resource_mismatch", "resources": res, "stopped": True})
        progress("resource mismatch, no propagation")
        return 3
    launch = launch_tags(TAGS)
    progress(f"jobs return {launch['returncodes']} driver_wall={launch['driver_wall_s']:.1f}s")
    if not all(job_finished_ok(tag) for tag in TAGS):
        rec = {"status": "timeout_partial", "mode": "nbar_cancel", "stopped": True, **launch,
               "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                              for tag in TAGS}}
        dump(OUT / "nbar_cancel.json", rec)
        return 1
    rec = analyze()
    progress(f"analyze done verdict={rec.get('verdict')} tracks_R={rec.get('tracks_R')}")
    return 0 if rec.get("status") == "ok" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("cancel",), default="")
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
            "alpha": rec.get("alpha"), "r": rec.get("r"),
            "nbar": rec.get("nbar"), "omega0": rec.get("omega0"),
            "hash_abort": rec.get("hash_abort"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") == "resources" and rec.get("resource_match_ok") else 3
    if args.job:
        rec = run_one_job(args.job)
        return 0 if rec.get("status") in ("ok", "timeout_partial") else 1
    if args.analyze_only:
        rec = analyze()
        print(json.dumps(jsonable({
            "status": rec.get("status"), "verdict": rec.get("verdict"),
            "tracks_R": rec.get("tracks_R"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") == "ok" else 1
    return drive()


if __name__ == "__main__":
    raise SystemExit(main())
