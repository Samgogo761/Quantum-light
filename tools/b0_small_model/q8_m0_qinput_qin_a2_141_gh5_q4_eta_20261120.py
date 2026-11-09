#!/usr/bin/env python3
"""Qin_a2_1_4_1 GH5/MTEF Q4: magnetic exchange-strength intervention.

Frozen light: DS(alpha=0.5, r=0.3), theta=pi/2 vs 3pi/2 only.
New jobs: eta=0.5 and 1.5, each with N=+/-1 (eight GH5 paths).
eta=1 is frozen Q2 GH5, read-only. No new QED, extra squeeze angles,
Cn, BSV, GH7, extra k, nmax, D2, Floquet, R2, or server.

Magnetic-odd term in h8 is only neel * p.delta. Implement
h_pmN(eta)=h_e +/- eta h_m as M0Params(delta=eta*R0.delta).
Do not edit q8_m0_model.py. Rebuild zero-field filling per eta.
Correlated 2x2 GH is required. Integer Fourier is not certified HHG.
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
import q8_m0_qinput_qin_a2_141_gh5_q2_covsign_20261120 as q2
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import M0Params, R0, build_geometry, h8, op_P, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_qin_a2_141_gh5_q4_eta_20261120"
FIG = OUT / "figures"
Q2_DIR = ROOT / "results/q_line/qinput_qin_a2_141_gh5_q2_covsign_20261120"
Q3_JSON = ROOT / "results/q_line/qinput_qin_a2_141_qed_q3_covsign_20261120/qed_q3_covsign.json"
DOCS = ROOT / "docs/04_BSV量子光"
TASK_TAG = "20261120"
RECORD_DATE = "2026-10-24"
PROTOCOL = "Qin_a2_1_4_1"
ORDER = q2.ORDER
N_NODES = q2.N_NODES
N_FULL = q2.N_FULL
SAMPLE_EVERY = q2.SAMPLE_EVERY
DRIVER_WALL_S = 900.0
JOB_WALL_S = 520.0
THETA_H = q2.THETA_H
THETA_T = q2.THETA_T
ETA_NEW = (0.5, 1.5)
ETA_FROZEN = 1.0
GAP_ABORT = 0.02
HERM_ABORT = 1e-10
INV_ABORT = 1e-8
R_ABS_MARK = 0.20
R_ARG_MARK = 20.0 * np.pi / 180.0
JOBS = (
    ("e05_p_h", 0.5, 1, THETA_H),
    ("e05_m_h", 0.5, -1, THETA_H),
    ("e05_p_t", 0.5, 1, THETA_T),
    ("e05_m_t", 0.5, -1, THETA_T),
    ("e15_p_h", 1.5, 1, THETA_H),
    ("e15_m_h", 1.5, -1, THETA_H),
    ("e15_p_t", 1.5, 1, THETA_T),
    ("e15_m_t", 1.5, -1, THETA_T),
)
JOB_MAP = {tag: (eta, neel, theta) for tag, eta, neel, theta in JOBS}
BATCHES = (
    (0.5, ("e05_p_h", "e05_m_h", "e05_p_t", "e05_m_t")),
    (1.5, ("e15_p_h", "e15_m_h", "e15_p_t", "e15_m_t")),
)
Q2_TAGS = {1: "ds_p_h", -1: "ds_m_h", "t1": "ds_p_t", "t-1": "ds_m_t"}
EXPECTED_HASHES = dict(q2.EXPECTED_HASHES)
FROZEN = tuple(HERE / name for name in EXPECTED_HASHES)
jsonable = q2.jsonable
cz = q2.cz
pack_c = q2.pack_c
wrap = q2.wrap
angdiff = q2.angdiff
sample_js_for = q2.sample_js_for
make_env = q2.make_env
analytic_xp_theta = q2.analytic_xp_theta
gh5_nodes = q2.gh5_nodes
one_traj = q2.one_traj
rect_jq = q2.rect_jq
slot_pack = q2.slot_pack
h2_jx = q2.h2_jx
fit_angle_model = q2.fit_angle_model


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


def params_eta(eta: float) -> M0Params:
    return M0Params(
        t_h=R0.t_h, lam_so=R0.lam_so, delta=float(eta) * R0.delta,
        u=R0.u, v=R0.v, label=f"R0_eta{eta:g}",
    )


def ground_pair_eta(geo, p, k, neel: int):
    ek, mass_k, leak_k = sh.slater36(dyn.electron_ground_state(geo, p, k, neel))
    emk, mass_mk, leak_mk = sh.slater36(dyn.electron_ground_state(geo, p, -k, neel))
    return ek, emk, {
        "neel": int(neel),
        "mass_k": mass_k, "mass_mk": mass_mk,
        "leak_k": leak_k, "leak_mk": leak_mk,
        "norm_k": float(np.linalg.norm(ek)),
        "norm_mk": float(np.linalg.norm(emk)),
        "filling": 4,
        "params_label": p.label, "delta": float(p.delta),
    }


def make_bundle(eta: float):
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    p = params_eta(eta)
    models, grounds = {}, {}
    for neel in (1, -1):
        models[neel] = sh.build_shared(
            geo, p, k, neel, n_max=0, omega0=pf.OMEGA0, g=pf.G_ABS)
        grounds[neel] = ground_pair_eta(geo, p, k, neel)
    return geo, k, p, models, grounds, make_env()


def eta_audit(eta: float) -> dict:
    geo = build_geometry()
    k = np.asarray(geo.k_cart(sh.K_FRAC), float)
    p = params_eta(eta)
    p8 = op_P()
    issues = []
    bands = []
    for neel in (1, -1):
        for kk, lab in ((k, "k"), (-k, "mk")):
            H = h8(geo, p, kk, neel)
            herm = float(np.max(np.abs(H - H.conj().T)))
            ev = np.linalg.eigvalsh(H)
            gap = float(ev[4] - ev[3])
            occ = {
                "neel": int(neel), "k": lab, "herm": herm, "gap": gap,
                "e4": float(ev[3]), "e5": float(ev[4]),
                "filling": 4, "n_bands": 8,
            }
            bands.append(occ)
            if herm > HERM_ABORT:
                issues.append(f"herm_{lab}_N{neel}")
            if gap <= GAP_ABORT:
                issues.append(f"gap_{lab}_N{neel}")
    hres_k = float(np.max(np.abs(p8 @ h8(geo, p, k, 1) @ p8.conj().T - h8(geo, p, -k, -1))))
    hres_mk = float(np.max(np.abs(p8 @ h8(geo, p, -k, 1) @ p8.conj().T - h8(geo, p, k, -1))))
    if hres_k > INV_ABORT or hres_mk > INV_ABORT:
        issues.append("domain_inversion")
    grounds = {}
    for neel in (1, -1):
        ek, emk, mass = ground_pair_eta(geo, p, k, neel)
        grounds[str(neel)] = mass
        if abs(mass["norm_k"] - 1.0) > 1e-10 or abs(mass["norm_mk"] - 1.0) > 1e-10:
            issues.append(f"norm_N{neel}")
        if mass["leak_k"] > 1e-12 or mass["leak_mk"] > 1e-12:
            issues.append(f"slater_leak_N{neel}")
    wp = sh.wedge_p36()
    ek_p, emk_p, _ = ground_pair_eta(geo, p, k, 1)
    ek_m, emk_m, _ = ground_pair_eta(geo, p, k, -1)
    ov_k = abs(complex(np.vdot(emk_m, wp @ ek_p)))
    ov_mk = abs(complex(np.vdot(ek_m, wp @ emk_p)))
    if min(ov_k, ov_mk) < 0.99:
        issues.append("P_map_overlap")
    ok = len(issues) == 0
    rec = {
        "eta": float(eta), "delta": float(p.delta), "delta_R0": float(R0.delta),
        "label": p.label, "ok": ok, "issues": issues,
        "herm_abort": HERM_ABORT, "inv_abort": INV_ABORT, "gap_abort": GAP_ABORT,
        "h8_P_k_plus_vs_mk_minus": hres_k,
        "h8_P_mk_plus_vs_k_minus": hres_mk,
        "abs_P_ek_plus_vs_emk_minus": ov_k,
        "abs_P_emk_plus_vs_ek_minus": ov_mk,
        "bands": bands, "grounds": grounds,
        "did_not_edit_q8_m0_model": True,
        "magnetic_term": "only neel * p.delta in _layer_block",
        "peierls_unchanged": True,
    }
    return rec


def collect_eta_audits(etas=ETA_NEW + (ETA_FROZEN,)) -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "eta_audit", "source_hashes": hashes, "hash_abort": bad,
        "etas": {}, "stopped": True,
    }
    if bad:
        rec["status"] = "hash_abort"
        dump(OUT / "eta_audit.json", rec)
        return rec
    all_ok = True
    for eta in etas:
        rec["etas"][f"{eta:g}"] = eta_audit(eta)
        all_ok = all_ok and rec["etas"][f"{eta:g}"]["ok"]
        progress(
            f"eta_audit eta={eta:g} ok={rec['etas'][f'{eta:g}']['ok']} "
            f"delta={rec['etas'][f'{eta:g}']['delta']:.4f} "
            f"issues={rec['etas'][f'{eta:g}']['issues']}"
        )
    rec["all_ok"] = all_ok
    dump(OUT / "eta_audit.json", rec)
    return rec


def collect_resources() -> dict:
    rec = q2.collect_resources()
    rec["task_tag"] = TASK_TAG
    rec["record_date"] = RECORD_DATE
    rec["note"] = "Q4 reuses Q2 light resources; magnetic eta is the only new control."
    dump(OUT / "resources.json", rec)
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


def load_series_q4(tag: str):
    z = np.load(series_path(tag))
    t = np.asarray(z["t"], float)
    j = np.stack([z["Jx"], z["Jy"]], axis=1)
    return t, j


def load_series_q2(neel: int, which: str):
    tag = "ds_p_h" if (which == "h" and neel == 1) else (
        "ds_m_h" if (which == "h" and neel == -1) else (
            "ds_p_t" if neel == 1 else "ds_m_t"))
    return q2.load_series(tag, Q2_DIR)


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
    audit = eta_audit(eta)
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
    _, _, p, models, grounds, env = make_bundle(eta)
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
             f"delta={p.delta:.4f} cov={gh['quad_cov_XP']:.6f} nodes={len(nodes)}")
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
        "used_correlated_GH": True,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
        "did_not_reuse_eta1_ground": True,
    }
    dump(job_path(tag), rec)
    progress(f"{tag} done wall={clock.elapsed():.1f}s")
    return rec


def pack_win(name, t, omega0, T0, w0, w1, arrs):
    return q2.pack_win(name, t, omega0, T0, w0, w1, arrs)


def q3_ab_recheck() -> dict:
    if not Q3_JSON.is_file():
        return {"status": "missing_q3_json"}
    raw = json.loads(Q3_JSON.read_text(encoding="utf-8"))
    thetas = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    keys = ["0", "h", "pi", "t"]
    odd_z = np.array([cz(raw["angles"][k]["h2_odd"]["jx"]) for k in keys], complex)
    even_z = np.array([cz(raw["angles"][k]["h2_even"]["jx"]) for k in keys], complex)
    fit_o = fit_angle_model(thetas, odd_z)
    fit_e = fit_angle_model(thetas, even_z)
    Ao, Bo = cz(fit_o["A"]), cz(fit_o["B"])
    Ae, Be = cz(fit_e["A"]), cz(fit_e["B"])
    rel_o = abs(Ao) / max(abs(Bo), 1e-30)
    rel_e = abs(Ae) / max(abs(Be), 1e-30)
    return {
        "status": "ok",
        "source": str(Q3_JSON.as_posix()),
        "odd": {"B": fit_o["B"], "A": fit_o["A"], "C": fit_o["C"],
                "abs_B": abs(Bo), "abs_A": abs(Ao), "abs_A_over_B": rel_o,
                "rel_resid": fit_o["rel_resid"]},
        "even": {"B": fit_e["B"], "A": fit_e["A"], "C": fit_e["C"],
                 "abs_B": abs(Be), "abs_A": abs(Ae), "abs_A_over_B": rel_e,
                 "rel_resid": fit_e["rel_resid"]},
        "selectivity_ratio": rel_o / max(rel_e, 1e-30),
        "abs_A_even_over_odd": abs(Ae) / max(abs(Ao), 1e-30),
        "note": "normalized angular selectivity, not absolute magnetic enhancement",
        "not_a_new_selection_rule": True,
        "expected_abs_A_over_B_odd": 1.351,
        "expected_abs_A_over_B_even": 0.00435,
        "expected_ratio": 310.0,
    }


def load_eta_currents(eta: float):
    if abs(eta - 1.0) < 1e-12:
        t_h_p, j_h_p = load_series_q2(1, "h")
        t_h_m, j_h_m = load_series_q2(-1, "h")
        t_t_p, j_t_p = load_series_q2(1, "t")
        t_t_m, j_t_m = load_series_q2(-1, "t")
        src = "frozen_q2"
    else:
        pref = "e05" if abs(eta - 0.5) < 1e-12 else "e15"
        t_h_p, j_h_p = load_series_q4(f"{pref}_p_h")
        t_h_m, j_h_m = load_series_q4(f"{pref}_m_h")
        t_t_p, j_t_p = load_series_q4(f"{pref}_p_t")
        t_t_m, j_t_m = load_series_q4(f"{pref}_m_t")
        src = "new_gh5"
    even_h, odd_h = odd.even_odd(j_h_p, j_h_m)
    even_t, odd_t = odd.even_odd(j_t_p, j_t_m)
    return {
        "eta": float(eta), "source": src, "t": t_h_p,
        "t_h_m": t_h_m, "t_t_p": t_t_p, "t_t_m": t_t_m,
        "even_h": even_h, "odd_h": odd_h, "even_t": even_t, "odd_t": odd_t,
        "d_odd": odd_t - odd_h, "d_even": even_t - even_h,
    }


def summarize_eta(pack, env) -> dict:
    t = pack["t"]
    T0 = env.period
    omega0 = pf.OMEGA0
    arrs = {
        "J_odd_h": pack["odd_h"], "J_odd_t": pack["odd_t"], "dJ_odd": pack["d_odd"],
        "J_even_h": pack["even_h"], "J_even_t": pack["even_t"], "dJ_even": pack["d_even"],
    }
    main = pack_win("plateau_center_2T", t, omega0, T0, 2.0 * T0, 4.0 * T0, arrs)
    front = pack_win("front_2T", t, omega0, T0, 1.0 * T0, 3.0 * T0, arrs)
    back = pack_win("back_2T", t, omega0, T0, 3.0 * T0, 5.0 * T0, arrs)
    zh = h2_jx(main["slots"], "J_odd_h")
    zt = h2_jx(main["slots"], "J_odd_t")
    zd = h2_jx(main["slots"], "dJ_odd")
    eh = h2_jx(main["slots"], "J_even_h")
    et = h2_jx(main["slots"], "J_even_t")
    ed = h2_jx(main["slots"], "dJ_even")
    zf = h2_jx(front["slots"], "dJ_odd")
    zb = h2_jx(back["slots"], "dJ_odd")
    mean_o = 0.5 * (zh[0] + zt[0])
    mean_e = 0.5 * (eh[0] + et[0])
    h1h6_odd = []
    for iq in range(6):
        dh = cz(main["slots"]["dJ_odd"]["Jx"][iq])
        h1h6_odd.append({"q": iq + 1, "jx": pack_c(dh), "abs": float(abs(dh))})
    return {
        "eta": pack["eta"], "source": pack["source"],
        "windows": {"main": main, "front": front, "back": back},
        "jx_h": pack_c(zh[0]), "jx_t": pack_c(zt[0]), "delta": pack_c(zd[0]),
        "abs_h": zh[2], "abs_t": zt[2], "abs_delta": zd[2],
        "phase_h": zh[3], "phase_t": zt[3], "phase_delta": zd[3],
        "dphi_rad": abs(angdiff(zt[3], zh[3])),
        "even_jx_h": pack_c(eh[0]), "even_jx_t": pack_c(et[0]),
        "even_delta": pack_c(ed[0]), "abs_even_h": eh[2], "abs_even_t": et[2],
        "abs_even_delta": ed[2],
        "mean_odd": pack_c(mean_o), "mean_even": pack_c(mean_e),
        "rel_odd": float(zd[2] / max(abs(mean_o), 1e-30)),
        "rel_even": float(ed[2] / max(abs(mean_e), 1e-30)),
        "abs_front": zf[2], "phase_front": zf[3],
        "abs_back": zb[2], "phase_back": zb[3],
        "h1h6_dodd_jx": h1h6_odd,
        "odd_over_even_abs_h": float(zh[2] / max(eh[2], 1e-30)),
        "odd_over_even_abs_t": float(zt[2] / max(et[2], 1e-30)),
    }


def figure_dperp_plane(rows, path, title, ylabel="Im δ⊥ Jx (H2 odd)"):
    lim = max(abs(z) for _, z, _ in rows) * 1.28 + 1e-30
    x0, y0, w, h = 90, 40, 740, 430
    body = q2.svg_axes(x0, y0, w, h, "Re δ⊥ Jx (H2 odd)", ylabel, title)
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    lx = 70
    for label, z, color in rows:
        body += q2.arrow(z, lim, x0, y0, w, h, color, label, lx, 36)
        lx += 140
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "GH5/MTEF; frozen eta=1 from Q2; not certified HHG; not new QED</text>"
    )
    return q2.write_svg(path, body)


def figure_vs_eta(etas, yodd, yeven, path, ylabel, title, logy=False):
    x0, y0, w, h = 80, 40, 760, 430
    body = q2.svg_axes(x0, y0, w, h, "η", ylabel, title)
    xs = np.asarray(etas, float)
    yo = np.asarray(yodd, float)
    ye = np.asarray(yeven, float)
    ymin = min(float(np.min(yo)), float(np.min(ye)))
    ymax = max(float(np.max(yo)), float(np.max(ye)))
    if logy:
        ymin = max(min(yo[yo > 0].min(), ye[ye > 0].min()) / 1.4, 1e-20)
        ymax = max(float(np.max(yo)), float(np.max(ye))) * 1.4

    def py(val):
        if logy:
            return y0 + (1.0 - (np.log10(val) - np.log10(ymin)) / max(np.log10(ymax / ymin), 1e-12)) * h
        return y0 + (1.0 - (val - ymin) / max(ymax - ymin, 1e-30)) * h

    def px(eta):
        return x0 + (float(eta) - 0.3) / 1.4 * w

    pts_o, pts_e = [], []
    for eta, a, b in zip(xs, yo, ye):
        pts_o.append(f"{px(eta):.1f},{py(a):.1f}")
        pts_e.append(f"{px(eta):.1f},{py(b):.1f}")
        body += f'<circle cx="{px(eta):.1f}" cy="{py(a):.1f}" r="5" fill="#7a4ea3"/>'
        body += f'<circle cx="{px(eta):.1f}" cy="{py(b):.1f}" r="5" fill="#c45c26"/>'
    body += f'<polyline fill="none" stroke="#7a4ea3" stroke-width="2" points="{" ".join(pts_o)}"/>'
    body += f'<polyline fill="none" stroke="#c45c26" stroke-width="2" points="{" ".join(pts_e)}"/>'
    body += '<text x="80" y="36" font-size="11" fill="#7a4ea3">odd</text>'
    body += '<text x="150" y="36" font-size="11" fill="#c45c26">even</text>'
    return q2.write_svg(path, body)


def figure_phase(etas, phis, path):
    x0, y0, w, h = 80, 40, 760, 430
    body = q2.svg_axes(x0, y0, w, h, "η", "arg δ⊥ (deg)", "H2 odd δ⊥ phase vs η")
    ph = np.asarray(phis, float) * 180.0 / np.pi
    ymin, ymax = float(np.min(ph)) - 15, float(np.max(ph)) + 15

    def px(eta):
        return x0 + (float(eta) - 0.3) / 1.4 * w

    def py(val):
        return y0 + (1.0 - (val - ymin) / max(ymax - ymin, 1e-12)) * h

    pts = []
    for eta, a in zip(etas, ph):
        pts.append(f"{px(eta):.1f},{py(a):.1f}")
        body += f'<circle cx="{px(eta):.1f}" cy="{py(a):.1f}" r="5" fill="#2f9e44"/>'
        body += f'<text x="{px(eta)+8:.1f}" y="{py(a)-8:.1f}" font-size="11" fill="#2f9e44">{a:.1f}°</text>'
    body += f'<polyline fill="none" stroke="#2f9e44" stroke-width="2" points="{" ".join(pts)}"/>'
    return q2.write_svg(path, body)


def write_reports(rec: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    v = rec["verdict"]
    ans = rec["answer"]
    q3 = rec["q3_recheck"]
    full = []
    full.append("# 量子光多周期磁序响应：Q4 磁性交换场强度干预（GH5/MTEF）\n")
    full.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。")
    full.append("冻结四角 QED/GH5 与 η=1 的 Q2 数据，只读。本轮仅新增 η=0.5、1.5 的 θ=π/2、3π/2，各 N=±1。")
    full.append("磁性项只缩放 `p.delta`。算符顺序 D(α)S(ξ)|0⟩。相关二维 GH5。`stopped = true`。\n")
    full.append(f"**结论：{ans}**\n")
    full.append("## 输入与磁性控制\n")
    full.append("光场冻结：α=0.5，r=0.3，⟨X⟩=1，⟨P⟩=0，⟨n⟩≈0.342733，")
    full.append("Var(X)=Var(P)≈1.185465，Cov(X,P)=∓sinh(0.6)。")
    full.append("η 只进入八带 Hamiltonian 的磁序奇交换 `neel * η * R0.delta`。Peierls、SOC、跃迁不变。\n")
    full.append("| η | δ | 能隙(k,+N) | 审计 |\n|---|---|---|---|")
    for key in ("0.5", "1", "1.5"):
        au = rec["eta_audit"]["etas"][key]
        gap = au["bands"][0]["gap"]
        full.append(f"| {key} | {au['delta']:.4f} | {gap:.4f} | `{au['ok']}` {au['issues']} |")
    full.append("\n## 主窗 H2 磁序奇 δ⊥\n")
    full.append("| η | δ⊥ Jx | |δ⊥| | arg | |δ⊥|/η | r=δ⊥/(η δ⊥(1)) |\n|---|---|---|---|---|---|")
    for eta in rec["eta_order"]:
        s = rec["by_eta"][f"{eta:g}"]
        if s.get("unavailable"):
            full.append(f"| {eta:g} | unavailable | — | — | — | — |")
            continue
        d = cz(s["delta"])
        rr = s.get("r_vs_eta1")
        rtxt = "—" if rr is None else f"{rr['abs']:.3f}∠{rr['arg_deg']:.1f}°"
        full.append(
            f"| {eta:g} | {d.real:.3e}{d.imag:+.3e}i | {s['abs_delta']:.3e} | "
            f"{s['phase_delta']:.3f} rad | {s['abs_delta']/eta:.3e} | {rtxt} |"
        )
    sc = rec["scale_test"]
    full.append(
        f"\nδ⊥(1.5)/δ⊥(0.5) = {sc['ratio_15_over_05_abs']:.3f}∠{sc['ratio_15_over_05_arg_deg']:.1f}°"
        f"（线性因子化期望 3）。工程标记 |r−1|<{R_ABS_MARK} 且 |arg r|<{R_ARG_MARK*180/np.pi:.0f}°，不是物理定律。\n"
    )
    full.append("## 偶/奇通道相对角向敏感性（主窗 H2，Jx）\n")
    full.append("| η | |δ⊥_odd| | |δ⊥_even| | |δ⊥_odd|/|mean_odd| | |δ⊥_even|/|mean_even| |\n|---|---|---|---|---|")
    for eta in rec["eta_order"]:
        s = rec["by_eta"][f"{eta:g}"]
        if s.get("unavailable"):
            continue
        full.append(
            f"| {eta:g} | {s['abs_delta']:.3e} | {s['abs_even_delta']:.3e} | "
            f"{s['rel_odd']:.3f} | {s['rel_even']:.3e} |"
        )
    full.append("\n磁序奇绝对值仍远弱于偶通道。相对敏感性不是绝对磁性输出增强。\n")
    if q3.get("status") == "ok":
        full.append("## 冻结 Q3 四角 A/B 复核\n")
        full.append(
            f"|A_o/B_o|={q3['odd']['abs_A_over_B']:.3f}，|A_e/B_e|={q3['even']['abs_A_over_B']:.5f}，"
            f"比值 {q3['selectivity_ratio']:.1f}。"
            "这是归一化角向选择性，不是新的磁性选择定则。\n"
        )
    full.append("## 解释边界\n")
    full.append("- 本轮是 GH5/MTEF，不是新的精确 QED。")
    full.append("- η=0 时 J^o 必为零，没有发现价值；比较的是 δ⊥(η)/η。")
    full.append("- 即使偏离线性，也先考虑能带/能隙/共振，不自动宣称新型量子磁性选择定则。")
    full.append("- 输入是位移压缩态 α=0.5，不是纯 BSV。两周期矩形 Fourier ≠ 已认证 HHG。")
    full.append("- 未启动精确 QED、额外压缩角、Cn、更多 k 或服务器。\n")
    full.append("完成后停机。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q4_GH5_eta_20261120.md").write_text(
        "\n".join(full) + "\n", encoding="utf-8")
    one = []
    one.append("# 量子光多周期磁序响应：Q4 交换场干预（一页）\n")
    one.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。冻结 η=1 Q2 与四角 QED。`stopped = true`。\n")
    one.append(f"**{ans}**\n")
    one.append("| 量 | 结果 |\n|---|---|")
    one.append(f"| 控制 | 光场冻结；η=0.5,1,1.5 只缩放磁交换 |")
    one.append(f"| 分类 | `{v}` |")
    one.append(f"| |r(0.5)−1| / |r(1.5)−1| | {sc['r05_absdev']:.3f} / {sc['r15_absdev']:.3f} |")
    one.append(f"| arg r (deg) | {sc['r05_arg_deg']:.1f} / {sc['r15_arg_deg']:.1f} |")
    one.append("| 新 QED / 额外 θ | 现在不启动 |\n")
    one.append("**问题：** 固定压缩光资源，只改变静态反铁磁交换强度，H2 磁序奇协方差符号差是否只被 η 整体缩放？\n")
    one.append(f"**{rec['answer_short']}**\n")
    one.append("停机。不自动开精确 QED。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q4_GH5_eta_一页_20261120.md").write_text(
        "\n".join(one) + "\n", encoding="utf-8")


def analyze() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(OUT / "gh5_q4_eta.json", rec)
        return rec
    for tag in ("ds_p_h", "ds_m_h", "ds_p_t", "ds_m_t"):
        if not (Q2_DIR / f"series_{tag}.npz").is_file():
            rec = {"status": "missing_frozen_q2", "tag": tag, "stopped": True}
            dump(OUT / "gh5_q4_eta.json", rec)
            return rec
    audits = collect_eta_audits()
    res = collect_resources()
    if not res.get("resource_match_ok"):
        rec = {"status": "resource_mismatch", "resources": rec_safe(res), "stopped": True}
        dump(OUT / "gh5_q4_eta.json", rec)
        return rec
    env = make_env()
    by_eta = {}
    packs = {}
    eta_order = [0.5, 1.0, 1.5]
    for eta in eta_order:
        if abs(eta - 1.0) < 1e-12 or eta_jobs_ok(eta):
            pack = load_eta_currents(eta)
            tref = None
            if packs:
                tref = next(iter(packs.values()))["t"]
                if not np.allclose(pack["t"], tref, rtol=0, atol=1e-9):
                    rec = {"status": "time_grid_mismatch", "eta": eta, "stopped": True}
                    dump(OUT / "gh5_q4_eta.json", rec)
                    return rec
            packs[eta] = pack
            by_eta[f"{eta:g}"] = summarize_eta(pack, env)
        else:
            by_eta[f"{eta:g}"] = {
                "eta": eta, "unavailable": True,
                "reason": "jobs incomplete or eta abort",
                "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                               for tag, e, _, _ in JOBS if abs(e - eta) < 1e-12},
            }
    if "1" not in by_eta or by_eta["1"].get("unavailable"):
        rec = {"status": "missing_eta1", "stopped": True}
        dump(OUT / "gh5_q4_eta.json", rec)
        return rec
    d1 = cz(by_eta["1"]["delta"])
    for eta in (0.5, 1.5):
        s = by_eta[f"{eta:g}"]
        if s.get("unavailable"):
            continue
        d = cz(s["delta"])
        r = d / (eta * d1)
        s["r_vs_eta1"] = {
            "re": float(r.real), "im": float(r.imag), "abs": float(abs(r)),
            "arg": float(np.angle(r)), "arg_deg": float(np.angle(r) * 180.0 / np.pi),
            "abs_minus_1": float(abs(r - 1.0)),
        }
        s["delta_over_eta"] = pack_c(d / eta)
    have05 = not by_eta["0.5"].get("unavailable")
    have15 = not by_eta["1.5"].get("unavailable")
    if have05 and have15:
        r15 = cz(by_eta["1.5"]["delta"]) / cz(by_eta["0.5"]["delta"])
        ratio_abs, ratio_arg = float(abs(r15)), float(np.angle(r15))
    else:
        r15, ratio_abs, ratio_arg = None, None, None
    r05 = by_eta["0.5"].get("r_vs_eta1")
    r15p = by_eta["1.5"].get("r_vs_eta1")
    r05_dev = None if r05 is None else r05["abs_minus_1"]
    r15_dev = None if r15p is None else r15p["abs_minus_1"]
    r05_arg = 0.0 if r05 is None else abs(r05["arg"])
    r15_arg = 0.0 if r15p is None else abs(r15p["arg"])
    factor_ok = False
    if have05 and have15:
        factor_ok = (r05_dev < R_ABS_MARK and r15_dev < R_ABS_MARK
                     and r05_arg < R_ARG_MARK and r15_arg < R_ARG_MARK)
    if not have05 and not have15:
        verdict = "no_new_eta_completed"
        answer = "新 η 点未完成，不能判断因子化。保存已完成结果，不自动续算，不启动 QED。"
        answer_short = "未完成新 η"
    elif factor_ok:
        verdict = "compatible_with_eta_factorization"
        answer = (
            "在当前 GH5/MTEF、冻结光场与 H2 观测量下，θ=π/2 与 3π/2 的磁序奇复差向量"
            "近似随 η 成比例缩放，相位结构没有超出工程标记的额外旋转。"
            "这与“通用压缩统计响应 × 磁性二阶系数”的因子化图像相容。"
            "不是精确 QED，也不是新的磁性选择定则。"
        )
        answer_short = "相容：δ⊥ 近似随 η 缩放"
    elif have05 or have15:
        verdict = "beyond_simple_eta_factorization_or_unresolved"
        unstable = False
        for eta in (0.5, 1.5):
            au = audits["etas"][f"{eta:g}"]
            if not au["ok"]:
                unstable = True
        if unstable:
            verdict = "eta_unstable_stop_scan"
            answer = (
                "某个 η 使占据或能隙结构越出受控范围。停止扩大参数扫描。"
                "不自动认定新选择定则，不启动 QED。"
            )
            answer_short = "η 不稳定，停止扫描"
        else:
            answer = (
                "δ⊥(η)/η 相对 η=1 的复比例偏离工程标记，或相位随 η 明显转动。"
                "这可能来自磁性交换改变能带、能隙或共振，而不是已证明的新型量子磁性选择定则。"
                "本轮不扩大 η 扫描，不启动 QED。"
            )
            answer_short = "超出简单 η 因子化（需能带解释，非新选择定则）"
    else:
        verdict = "unresolved"
        answer = "数值分辨不足或窗口不完整。停止扩大扫描。"
        answer_short = "分辨不足"
    FIG.mkdir(parents=True, exist_ok=True)
    plane_rows, scaled_rows = [], []
    colors = {0.5: "#7a4ea3", 1.0: "#2f9e44", 1.5: "#c45c26"}
    etas_ok, abs_o, abs_e, rel_o, rel_e, phis = [], [], [], [], [], []
    for eta in eta_order:
        s = by_eta[f"{eta:g}"]
        if s.get("unavailable"):
            continue
        d = cz(s["delta"])
        plane_rows.append((f"η={eta:g}", d, colors[eta]))
        scaled_rows.append((f"η={eta:g}", d / eta, colors[eta]))
        etas_ok.append(eta)
        abs_o.append(s["abs_delta"])
        abs_e.append(s["abs_even_delta"])
        rel_o.append(s["rel_odd"])
        rel_e.append(s["rel_even"])
        phis.append(s["phase_delta"])
    figs = {}
    if plane_rows:
        figs["dperp_plane"] = figure_dperp_plane(
            plane_rows, FIG / "dperp_odd_complex_plane.svg",
            "H2 odd δ⊥ vs magnetic strength η")
        figs["dperp_over_eta"] = figure_dperp_plane(
            scaled_rows, FIG / "dperp_odd_over_eta.svg",
            "H2 odd δ⊥/η (factorization test)")
        figs["abs_odd_even"] = figure_vs_eta(
            etas_ok, abs_o, abs_e, FIG / "abs_dperp_odd_even.svg",
            "|δ⊥ Jx|", "Absolute H2 δ⊥: odd vs even", logy=True)
        figs["rel_odd_even"] = figure_vs_eta(
            etas_ok, rel_o, rel_e, FIG / "rel_dperp_odd_even.svg",
            "|δ⊥|/|mean|", "Normalized angular response: odd vs even", logy=True)
        figs["phase_vs_eta"] = figure_phase(etas_ok, phis, FIG / "dperp_phase_vs_eta.svg")
    q3 = q3_ab_recheck()
    jobs = {tag: load_job(tag) for tag, _, _, _ in JOBS}
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "ok", "stopped": True, "verdict": verdict,
        "answer": answer, "answer_short": answer_short,
        "n_max": 0, "order": ORDER, "n_nodes": N_NODES, "n_steps": N_FULL,
        "dt": env.t_total / N_FULL,
        "q2_archive": str(Q2_DIR.as_posix()),
        "frozen_eta1_not_rerun": True, "qed_not_rerun": True,
        "operator_order": "D(alpha)S(xi)|0>",
        "used_correlated_GH": True,
        "magnetic_term": "neel * eta * R0.delta only",
        "source_hashes": hashes, "hash_abort": [],
        "resources": res, "eta_audit": audits,
        "eta_order": eta_order, "by_eta": by_eta,
        "scale_test": {
            "r05_absdev": r05_dev, "r15_absdev": r15_dev,
            "r05_arg_deg": None if r05 is None else r05["arg_deg"],
            "r15_arg_deg": None if r15p is None else r15p["arg_deg"],
            "ratio_15_over_05_abs": ratio_abs,
            "ratio_15_over_05_arg_deg": None if ratio_arg is None else ratio_arg * 180.0 / np.pi,
            "marks_not_laws": True, "abs_mark": R_ABS_MARK,
            "arg_mark_deg": R_ARG_MARK * 180.0 / np.pi,
            "factor_ok": factor_ok,
        },
        "q3_recheck": q3,
        "jobs": {tag: None if jobs[tag] is None else {k: jobs[tag].get(k) for k in
                 ("status", "n_run", "n_done", "wall_s", "max_abs_norm_minus_1",
                  "weight_sum", "eta", "neel", "theta", "delta", "quad_cov_XP")}
                 for tag, _, _, _ in JOBS},
        "figures": figs,
        "physics_boundaries": {
            "GH5_is_not_exact_QED": True,
            "eta0_odd_current_is_identically_zero": True,
            "displaced_squeezed_not_pure_BSV": True,
            "integer_Fourier_is_not_certified_HHG": True,
            "normalized_selectivity_is_not_absolute_enhancement": True,
            "did_not_claim_new_magnetic_selection_rule": True,
        },
        "did_not": [
            "new QED", "extra squeeze angles", "rerun eta=1 GH5",
            "Cn", "pure BSV", "GH7", "extra k", "nmax scan",
            "D2", "Floquet", "R2 positive-P optimization", "server",
        ],
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "gh5_q4_eta.json", rec)
    write_reports(rec)
    return rec


def rec_safe(res):
    return res


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
        dump(OUT / "gh5_q4_eta.json", {"status": "hash_abort", "hash_mismatch": bad, "stopped": True})
        progress("hash abort, no propagation")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    res = collect_resources()
    if not res.get("resource_match_ok"):
        dump(OUT / "gh5_q4_eta.json", {"status": "resource_mismatch", "resources": res, "stopped": True})
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
    if eta_jobs_ok(0.5) or eta_jobs_ok(1.5):
        rec = analyze()
        progress(f"analyze done verdict={rec.get('verdict')}")
        return 0 if rec.get("status") == "ok" else 1
    rec = {
        "status": "timeout_partial", "stopped": True,
        "driver_wall_s": time.perf_counter() - t0, "returncodes": codes,
        "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                       for tag, _, _, _ in JOBS},
        "did_not_analyze_incomplete": True, "qed_not_rerun": True,
    }
    dump(OUT / "gh5_q4_eta.json", rec)
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
            "cov_sign_flipped": rec.get("cov_sign_flipped"),
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
