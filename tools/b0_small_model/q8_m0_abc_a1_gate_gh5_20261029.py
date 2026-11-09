"""A1: Q6-protocol GH5 gates Delta=0 and SOC=0 at theta=0/pi.

Product GH5 only (Cov(X,P)=0). Do not call Q2 correlated gh5_nodes.
Do not edit q8_m0_model.py. No new QED, no nmax=10, no M52 grid.
Spectrum uses linear_exact endpoint-fixed Fourier. Two angles cannot
separate A from C; report B_proxy and (A+C)_proxy only.
Delta=0 odd vanishing is a consistency check, not a new mechanism.
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
import qinput_spectrum_reanalysis_20261028 as spec
from q8_m0_model import M0Params, R0, build_geometry, h8, op_P, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/abc_a1_gate_gh5_20261029"
FIG = OUT / "figures"
DOCS = ROOT / "docs/04_BSV量子光"
FROZEN_TH0 = ROOT / "results/q_line/qinput_qin_a2_141_gh5_20261120"
FROZEN_THPI = ROOT / "results/q_line/qinput_qin_a2_141_gh5_theta_pi_20261120"
TASK_TAG = "20261029"
RECORD_DATE = "2026-10-29"
PROTOCOL = "Qin_a2_1_4_1"
ORDER = 5
N_NODES = ORDER * ORDER
N_FULL = 11154
SAMPLE_EVERY = 32
DRIVER_WALL_S = 7200.0
JOB_WALL_S = 1800.0
ALPHA = odd.ALPHA
R_SQ = odd.R_SQ
THETA_0 = 0.0
THETA_PI = float(np.pi)
METAL_ABORT = 0.0
GAP_WARN = 0.02
HERM_ABORT = 1e-10
INV_ABORT = 1e-8
QS = (1, 2, 3, 4, 5)
MAIN_WIN = (2.0, 4.0)

CONTROLS = {
    "d0": {"delta": 0.0, "lam_so": float(R0.lam_so), "label": "Delta=0"},
    "s0": {"delta": float(R0.delta), "lam_so": 0.0, "label": "SOC=0"},
}
JOBS = (
    ("d0_p_0", "d0", 1, THETA_0),
    ("d0_m_0", "d0", -1, THETA_0),
    ("d0_p_pi", "d0", 1, THETA_PI),
    ("d0_m_pi", "d0", -1, THETA_PI),
    ("s0_p_0", "s0", 1, THETA_0),
    ("s0_m_0", "s0", -1, THETA_0),
    ("s0_p_pi", "s0", 1, THETA_PI),
    ("s0_m_pi", "s0", -1, THETA_PI),
)
JOB_MAP = {tag: (ctrl, neel, theta) for tag, ctrl, neel, theta in JOBS}
BATCHES = (
    ("d0", ("d0_p_0", "d0_m_0", "d0_p_pi", "d0_m_pi")),
    ("s0", ("s0_p_0", "s0_m_0", "s0_p_pi", "s0_m_pi")),
)


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


def cpack(z):
    z = complex(z)
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z)),
            "arg_deg": float(np.degrees(np.angle(z)))}


def cz(pack) -> complex:
    return complex(pack["re"], pack["im"])


def wrap(phi: float) -> float:
    return float((phi + np.pi) % (2.0 * np.pi) - np.pi)


def angdiff_deg(a: float, b: float) -> float:
    return float(np.degrees(wrap(np.radians(float(a) - float(b)))))


def sample_js_for(n_steps: int):
    js = list(range(0, n_steps, SAMPLE_EVERY))
    if not js or js[-1] != n_steps - 1:
        js.append(n_steps - 1)
    return js


def make_env():
    return dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=4.0)


def params_ctrl(ctrl: str) -> M0Params:
    c = CONTROLS[ctrl]
    return M0Params(
        t_h=R0.t_h, lam_so=float(c["lam_so"]), delta=float(c["delta"]),
        u=R0.u, v=R0.v, label=f"R0_{ctrl}",
    )


def analytic_xp_theta(theta: float) -> dict:
    a = complex(ALPHA)
    mx, mp = 2.0 * a.real, 2.0 * a.imag
    r = float(R_SQ)
    th = float(theta)
    if abs(th) < 1e-15:
        vx, vp = float(np.exp(-2.0 * r)), float(np.exp(2.0 * r))
        axis = "X_squeezed"
    elif abs(th - np.pi) < 1e-12:
        vx, vp = float(np.exp(2.0 * r)), float(np.exp(-2.0 * r))
        axis = "P_squeezed"
    else:
        raise ValueError("A1 ships theta=0 or pi only; product GH requires Cov=0")
    nbar = abs(a) ** 2 + np.sinh(r) ** 2
    return {
        "kind": "ds", "alpha": pf.cpack(a), "r": r, "theta": th,
        "mean_X": mx, "mean_P": mp, "var_X": vx, "var_P": vp,
        "cov_XP": 0.0, "mean_n_analytic": float(nbar),
        "mean_n_Weyl": 0.25 * (mx * mx + mp * mp + vx + vp) - 0.5,
        "product_GH_valid": True, "squeeze_axis": axis,
        "did_not_call_q2_gh5_nodes": True,
    }


def gh5_nodes(theta: float) -> dict:
    xi, w1 = hermgauss(ORDER)
    mom = analytic_xp_theta(theta)
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
        "nodes": nodes, "analytic": mom,
        "product_GH": True, "did_not_call_q2_gh5_nodes": True,
    }


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


def ground_pair_ctrl(geo, p, k, neel: int):
    ek, mass_k, leak_k = sh.slater36(dyn.electron_ground_state(geo, p, k, neel))
    emk, mass_mk, leak_mk = sh.slater36(dyn.electron_ground_state(geo, p, -k, neel))
    return ek, emk, {
        "neel": int(neel),
        "mass_k": mass_k, "mass_mk": mass_mk,
        "leak_k": leak_k, "leak_mk": leak_mk,
        "norm_k": float(np.linalg.norm(ek)),
        "norm_mk": float(np.linalg.norm(emk)),
        "filling": 4,
        "params_label": p.label, "delta": float(p.delta), "lam_so": float(p.lam_so),
    }


def make_bundle(ctrl: str):
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    p = params_ctrl(ctrl)
    models, grounds = {}, {}
    for neel in (1, -1):
        models[neel] = sh.build_shared(
            geo, p, k, neel, n_max=0, omega0=pf.OMEGA0, g=pf.G_ABS)
        grounds[neel] = ground_pair_ctrl(geo, p, k, neel)
    return geo, k, p, models, grounds, make_env()


def ctrl_audit(ctrl: str) -> dict:
    geo = build_geometry()
    k = np.asarray(geo.k_cart(sh.K_FRAC), float)
    p = params_ctrl(ctrl)
    p8 = op_P()
    issues = []
    warnings = []
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
                "evals": [float(x) for x in ev],
            }
            bands.append(occ)
            if herm > HERM_ABORT:
                issues.append(f"herm_{lab}_N{neel}")
            if gap <= METAL_ABORT:
                issues.append(f"metal_gap_{lab}_N{neel}")
            elif gap < GAP_WARN:
                warnings.append(f"small_gap_{lab}_N{neel}={gap:.4g}")
    h_plus = h8(geo, p, k, 1)
    h_minus = h8(geo, p, k, -1)
    ham_identical = float(np.max(np.abs(h_plus - h_minus)))
    hres_k = float(np.max(np.abs(p8 @ h8(geo, p, k, 1) @ p8.conj().T - h8(geo, p, -k, -1))))
    hres_mk = float(np.max(np.abs(p8 @ h8(geo, p, -k, 1) @ p8.conj().T - h8(geo, p, k, -1))))
    if hres_k > INV_ABORT or hres_mk > INV_ABORT:
        issues.append("domain_inversion")
    grounds = {}
    for neel in (1, -1):
        ek, emk, mass = ground_pair_ctrl(geo, p, k, neel)
        grounds[str(neel)] = mass
        if abs(mass["norm_k"] - 1.0) > 1e-10 or abs(mass["norm_mk"] - 1.0) > 1e-10:
            issues.append(f"norm_N{neel}")
        if mass["leak_k"] > 1e-12 or mass["leak_mk"] > 1e-12:
            issues.append(f"slater_leak_N{neel}")
    wp = sh.wedge_p36()
    ek_p, emk_p, _ = ground_pair_ctrl(geo, p, k, 1)
    ek_m, emk_m, _ = ground_pair_ctrl(geo, p, k, -1)
    ov_k = abs(complex(np.vdot(emk_m, wp @ ek_p)))
    ov_mk = abs(complex(np.vdot(ek_m, wp @ emk_p)))
    if min(ov_k, ov_mk) < 0.99:
        warnings.append("P_map_overlap_lt_0.99")
    min_gap = min(b["gap"] for b in bands)
    rec = {
        "ctrl": ctrl, "label": CONTROLS[ctrl]["label"],
        "delta": float(p.delta), "lam_so": float(p.lam_so),
        "delta_R0": float(R0.delta), "lam_so_R0": float(R0.lam_so),
        "ok": len(issues) == 0, "issues": issues, "warnings": warnings,
        "herm_abort": HERM_ABORT, "inv_abort": INV_ABORT,
        "metal_abort": METAL_ABORT, "gap_warn": GAP_WARN, "min_gap": min_gap,
        "plus_minus_H_identical": ham_identical,
        "h8_P_k_plus_vs_mk_minus": hres_k,
        "h8_P_mk_plus_vs_k_minus": hres_mk,
        "abs_P_ek_plus_vs_emk_minus": ov_k,
        "abs_P_emk_plus_vs_ek_minus": ov_mk,
        "bands": bands, "grounds": grounds,
        "did_not_edit_q8_m0_model": True,
        "did_not_reuse_eta1_ground": True,
        "note_delta0_odd_zero_is_definitional": ctrl == "d0",
    }
    return rec


def collect_audits() -> dict:
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "ctrl_audit", "controls": {}, "stopped": True,
        "source_file": Path(__file__).name,
        "source_sha256": sha256_file(Path(__file__)),
        "model_sha256": sha256_file(HERE / "q8_m0_model.py"),
    }
    all_ok = True
    for ctrl in CONTROLS:
        rec["controls"][ctrl] = ctrl_audit(ctrl)
        all_ok = all_ok and rec["controls"][ctrl]["ok"]
        au = rec["controls"][ctrl]
        progress(
            f"audit {ctrl} ok={au['ok']} gap={au['min_gap']:.6g} "
            f"dHpm={au['plus_minus_H_identical']:.3e} "
            f"issues={au['issues']} warn={au['warnings']}"
        )
    rec["all_ok"] = all_ok
    dump(OUT / "ctrl_audit.json", rec)
    return rec


def ckpt_path(tag: str) -> Path:
    return OUT / f"ckpt_{tag}.npz"


def job_path(tag: str) -> Path:
    return OUT / f"job_{tag}.json"


def series_path(tag: str) -> Path:
    return OUT / f"series_{tag}.npz"


def save_ckpt(tag, ctrl, neel, theta, n_done, n_nodes, acc, wsum, max_n, t_arr) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc = np.asarray(acc, complex)
    np.savez_compressed(
        ckpt_path(tag),
        acc_re=acc.real, acc_im=acc.imag, wsum=float(wsum),
        n_done=int(n_done), n_nodes=int(n_nodes), max_n=float(max_n),
        t=np.asarray(t_arr, float), neel=int(neel), theta=float(theta),
        ctrl=np.asarray(ctrl),
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
        "neel": int(z["neel"]) if "neel" in z.files else None,
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
    path = (directory or OUT) / (tag if tag.endswith(".npz") else f"series_{tag}.npz")
    if not path.name.startswith("series_") and directory is not None:
        path = directory / f"series_{tag}.npz"
    z = np.load(path)
    t = np.asarray(z["t"], float)
    j = np.stack([z["Jx"], z["Jy"]], axis=1)
    return t, j


def run_one_job(tag: str) -> dict:
    if tag not in JOB_MAP:
        raise ValueError(tag)
    ctrl, neel, theta = JOB_MAP[tag]
    audit = ctrl_audit(ctrl)
    if not audit["ok"]:
        rec = {"tag": tag, "status": "ctrl_abort", "ctrl_audit": audit, "stopped": True}
        dump(job_path(tag), rec)
        progress(f"{tag} ctrl abort issues={audit['issues']}")
        return rec
    if job_finished_ok(tag):
        rec = load_job(tag)
        progress(f"{tag} already complete n_run={rec.get('n_run')}")
        return rec
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, p, models, grounds, env = make_bundle(ctrl)
    gh = gh5_nodes(theta)
    if abs(gh["quad_cov_XP"]) > 1e-10:
        rec = {"tag": tag, "status": "cov_not_zero_abort",
               "quad_cov_XP": gh["quad_cov_XP"], "stopped": True}
        dump(job_path(tag), rec)
        return rec
    nodes = gh["nodes"]
    ek0, emk0, mass = grounds[neel]
    model = models[neel]
    js = sample_js_for(N_FULL)
    ck = load_ckpt(tag)
    i0, acc, wsum, max_n, t_arr = 0, None, 0.0, 0.0, None
    if ck is not None:
        if ck.get("theta") is not None and abs(float(ck["theta"]) - float(theta)) > 1e-12:
            rec = {"tag": tag, "status": "ckpt_theta_mismatch", "stopped": True}
            dump(job_path(tag), rec)
            return rec
        i0, acc, wsum, max_n, t_arr = ck["n_done"], ck["acc"], ck["wsum"], ck["max_n"], ck["t"]
        progress(f"{tag} resume n_done={i0}/{len(nodes)}")
    clock = dyn.WallClock(JOB_WALL_S)
    progress(f"{tag} start ctrl={ctrl} neel={neel} theta={theta:.6f} "
             f"delta={p.delta:.4f} lam_so={p.lam_so:.4f} nodes={len(nodes)}")
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
            save_ckpt(tag, ctrl, neel, theta, i + 1, len(nodes), acc, wsum, max_n, t_arr)
            if (i + 1) == len(nodes) or (i + 1) % 5 == 0:
                progress(f"{tag} node {i + 1}/{len(nodes)} wall={clock.elapsed():.1f}s")
    except TimeoutError as exc:
        timed_out = True
        timeout_msg = str(exc)
        progress(f"{tag} timeout {timeout_msg}")
    n_done = 0 if acc is None else int(load_ckpt(tag)["n_done"] if ckpt_path(tag).is_file() else 0)
    if timed_out or n_done < len(nodes):
        rec = {
            "tag": tag, "ctrl": ctrl, "neel": int(neel), "theta": float(theta),
            "status": "timeout_partial", "n_nodes": len(nodes), "n_done": n_done,
            "weight_sum": wsum, "max_abs_norm_minus_1": float(max_n),
            "wall_s": clock.elapsed(), "timeout_msg": timeout_msg,
            "n_max": 0, "protocol": PROTOCOL, "n_steps": N_FULL,
            "dt": env.t_total / N_FULL, "stopped": True, "electron": mass,
            "quad_cov_XP": gh["quad_cov_XP"], "delta": float(p.delta),
            "lam_so": float(p.lam_so),
        }
        dump(job_path(tag), rec)
        return rec
    avg = acc / (wsum + 1e-30)
    save_series(tag, t_arr, avg)
    rec = {
        "tag": tag, "ctrl": ctrl, "neel": int(neel), "order": ORDER, "theta": float(theta),
        "status": "ok", "n_nodes": len(nodes), "n_run": len(nodes),
        "n_done": len(nodes), "weight_sum": wsum,
        "max_abs_norm_minus_1": float(max_n),
        "wall_s": clock.elapsed(), "n_max": 0, "protocol": PROTOCOL,
        "n_steps": N_FULL, "dt": env.t_total / N_FULL,
        "n_samples": int(len(t_arr)), "t0": float(t_arr[0]), "t_end": float(t_arr[-1]),
        "electron": mass, "stopped": True, "delta": float(p.delta),
        "lam_so": float(p.lam_so),
        "wigner": analytic_xp_theta(theta),
        "quad_cov_XP": gh["quad_cov_XP"],
        "used_product_GH": True, "used_correlated_GH": False,
        "operator_order": "D(alpha)S(xi)|0>, xi=r exp(i theta)",
        "did_not_reuse_eta1_ground": True,
        "did_not_call_q2_gh5_nodes": True,
    }
    dump(job_path(tag), rec)
    progress(f"{tag} done wall={clock.elapsed():.1f}s")
    return rec


def h2_from_series(t, j):
    T0 = 2.0 * np.pi / pf.OMEGA0
    lo, hi = MAIN_WIN[0] * T0, MAIN_WIN[1] * T0
    out = {}
    for name, (a, b) in (("main", MAIN_WIN), ("front", (1.0, 3.0)), ("back", (3.0, 5.0))):
        jq = spec.spectrum(t, j, a * T0, b * T0, QS, method="linear_exact", omega=pf.OMEGA0)
        out[name] = {f"H{q}": [cpack(jq[i, 0]), cpack(jq[i, 1])] for i, q in enumerate(QS)}
    jq = spec.spectrum(t, j, lo, hi, QS, method="linear_exact", omega=pf.OMEGA0)
    return jq, out


def load_pm_h2(ctrl: str, theta: float):
    th = "0" if abs(theta) < 1e-15 else "pi"
    tp, jp = load_series(f"{ctrl}_p_{th}")
    tm, jm = load_series(f"{ctrl}_m_{th}")
    if np.max(np.abs(tp - tm)) > 1e-12:
        raise RuntimeError(f"time grid mismatch {ctrl} theta={th}")
    qp, pack_p = h2_from_series(tp, jp)
    qm, pack_m = h2_from_series(tm, jm)
    odd_j = 0.5 * (qp - qm)
    even_j = 0.5 * (qp + qm)
    return {
        "theta": float(theta),
        "plus_H2": [cpack(qp[1, 0]), cpack(qp[1, 1])],
        "minus_H2": [cpack(qm[1, 0]), cpack(qm[1, 1])],
        "odd_H2": [cpack(odd_j[1, 0]), cpack(odd_j[1, 1])],
        "even_H2": [cpack(even_j[1, 0]), cpack(even_j[1, 1])],
        "odd_norm2D": float(np.linalg.norm(odd_j[1])),
        "even_norm2D": float(np.linalg.norm(even_j[1])),
        "odd_over_even": float(np.linalg.norm(odd_j[1]) / max(np.linalg.norm(even_j[1]), 1e-30)),
        "windows": {"plus": pack_p, "minus": pack_m},
        "harmonics_odd_abs": {f"H{q}": float(np.linalg.norm(0.5 * (qp[i] - qm[i])))
                              for i, q in enumerate(QS)},
        "harmonics_even_abs": {f"H{q}": float(np.linalg.norm(0.5 * (qp[i] + qm[i])))
                               for i, q in enumerate(QS)},
    }


def two_angle_proxy(z0, zpi):
    z0, zpi = complex(z0), complex(zpi)
    B = 0.5 * (z0 + zpi)
    AC = 0.5 * (z0 - zpi)
    return {
        "B_proxy": cpack(B),
        "A_plus_C_proxy": cpack(AC),
        "abs_AC_over_B": float(abs(AC) / max(abs(B), 1e-30)),
        "J0": cpack(z0), "Jpi": cpack(zpi),
        "amp_ratio_pi_over_0": float(abs(zpi) / max(abs(z0), 1e-30)),
        "phase_diff_deg": float(np.degrees(wrap(np.angle(zpi) - np.angle(z0)))),
        "cannot_separate_A_and_C": True,
    }


def load_frozen_gh5():
    rows = {}
    for theta, folder in ((THETA_0, FROZEN_TH0), (THETA_PI, FROZEN_THPI)):
        tp, jp = load_series("ds_p", folder)
        tm, jm = load_series("ds_m", folder)
        qp, _ = h2_from_series(tp, jp)
        qm, _ = h2_from_series(tm, jm)
        odd_j = 0.5 * (qp - qm)
        even_j = 0.5 * (qp + qm)
        key = "theta0" if abs(theta) < 1e-15 else "theta_pi"
        rows[key] = {
            "odd_H2": [cpack(odd_j[1, 0]), cpack(odd_j[1, 1])],
            "even_H2": [cpack(even_j[1, 0]), cpack(even_j[1, 1])],
            "odd_norm2D": float(np.linalg.norm(odd_j[1])),
            "even_norm2D": float(np.linalg.norm(even_j[1])),
        }
    z0 = cz(rows["theta0"]["odd_H2"][0])
    zpi = cz(rows["theta_pi"]["odd_H2"][0])
    rows["odd_x_proxy"] = two_angle_proxy(z0, zpi)
    rows["odd_norm_ratio_pi_over_0"] = float(
        rows["theta_pi"]["odd_norm2D"] / max(rows["theta0"]["odd_norm2D"], 1e-30))
    return rows


def summarize_ctrl(ctrl: str, audit: dict) -> dict:
    a0 = load_pm_h2(ctrl, THETA_0)
    api = load_pm_h2(ctrl, THETA_PI)
    odd_x = two_angle_proxy(cz(a0["odd_H2"][0]), cz(api["odd_H2"][0]))
    even_x = two_angle_proxy(cz(a0["even_H2"][0]), cz(api["even_H2"][0]))
    odd_gate = max(a0["odd_norm2D"], api["odd_norm2D"])
    even_keep = max(a0["even_norm2D"], api["even_norm2D"])
    return {
        "ctrl": ctrl, "label": CONTROLS[ctrl]["label"],
        "delta": audit["delta"], "lam_so": audit["lam_so"],
        "min_gap": audit["min_gap"],
        "plus_minus_H_identical": audit["plus_minus_H_identical"],
        "P_map_overlap": [audit["abs_P_ek_plus_vs_emk_minus"],
                          audit["abs_P_emk_plus_vs_ek_minus"]],
        "theta0": a0, "theta_pi": api,
        "odd_x_proxy": odd_x, "even_x_proxy": even_x,
        "odd_max_norm2D": odd_gate, "even_max_norm2D": even_keep,
        "odd_over_even_max": float(odd_gate / max(even_keep, 1e-30)),
        "warnings": audit.get("warnings", []),
        "issues": audit.get("issues", []),
    }


def write_svg_bars(path: Path, groups, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 720, 360
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
            f'<rect width="{w}" height="{h}" fill="#fff"/>',
            f'<text x="20" y="28" font-size="16">{title}</text>']
    ymax = max(max(v for _, v in g[1]) for g in groups) * 1.15 + 1e-30
    x0, y0, bw, bh = 80, 50, 560, 260
    n = len(groups)
    gw = bw / n
    for i, (lab, vals) in enumerate(groups):
        gx = x0 + i * gw
        body.append(f'<text x="{gx + gw/2:.1f}" y="{y0+bh+24}" text-anchor="middle" font-size="12">{lab}</text>')
        for j, (name, val) in enumerate(vals):
            hh = bh * (val / ymax)
            xx = gx + 20 + j * 50
            yy = y0 + bh - hh
            color = "#1f77b4" if j == 0 else "#d62728"
            body.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="36" height="{hh:.1f}" fill="{color}"/>')
            body.append(f'<text x="{xx+18:.1f}" y="{yy-4:.1f}" text-anchor="middle" font-size="10">{val:.2e}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def write_reports(rec: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    groups = []
    frozen = rec.get("frozen_gh5", {})
    if frozen:
        groups.append(("frozen η=1", [
            ("odd0", frozen["theta0"]["odd_norm2D"]),
            ("oddπ", frozen["theta_pi"]["odd_norm2D"]),
        ]))
    for ctrl, row in rec.get("summaries", {}).items():
        groups.append((row["label"], [
            ("odd0", row["theta0"]["odd_norm2D"]),
            ("oddπ", row["theta_pi"]["odd_norm2D"]),
        ]))
    if groups:
        write_svg_bars(FIG / "odd_H2_norm.svg", groups, "H2 odd 2D |J|  main window")
    one = DOCS / "线Q_A1门控GH5_一页_20261029.md"
    longp = DOCS / "线Q_A1门控GH5_20261029.md"
    d0 = rec["summaries"].get("d0", {})
    s0 = rec["summaries"].get("s0", {})
    fr = frozen
    lines = [
        "# 线 Q：A1 门控 GH5 一页结论（2026-10-29）",
        "",
        "Q6 原协议乘积 GH5，θ=0/π。无新 QED，不覆盖旧归档。",
        "Δ=0 奇电流归零是程序一致性，不是新机制。SOC=0 须同时报告能隙。",
        "",
        "## 1. Δ=0 是否关掉奇通道？",
        "",
    ]
    if d0:
        lines += [
            f"- ±N Hamiltonian 差 {d0['plus_minus_H_identical']:.3e}（应为 0）。",
            f"- 最小能隙 {d0['min_gap']:.6g} eV。",
            f"- H2 奇 2D 范数：θ=0 为 {d0['theta0']['odd_norm2D']:.3e}，θ=π 为 {d0['theta_pi']['odd_norm2D']:.3e}。",
            f"- 奇/偶 最大比 {d0['odd_over_even_max']:.3e}。偶通道仍随 θ 变化：|A+C|/|B|={d0['even_x_proxy']['abs_AC_over_B']:.3g}。",
            "",
        ]
    lines += ["## 2. SOC=0 是否关掉奇通道？", ""]
    if s0:
        lines += [
            f"- 最小能隙 {s0['min_gap']:.6g} eV（R0 对照见审计）。P 映射重叠 {s0['P_map_overlap'][0]:.6f}, {s0['P_map_overlap'][1]:.6f}。",
            f"- H2 奇 2D 范数：θ=0 为 {s0['theta0']['odd_norm2D']:.3e}，θ=π 为 {s0['theta_pi']['odd_norm2D']:.3e}。",
            f"- 奇/偶 最大比 {s0['odd_over_even_max']:.3e}。",
            "- 不能仅凭信号消失宣称 SOC 独有机制；能隙与占据已记录。",
            "",
        ]
    lines += ["## 3. 与冻结 η=1 GH5 对照", ""]
    if fr:
        lines += [
            f"- 冻结奇 2D：θ=0 为 {fr['theta0']['odd_norm2D']:.3e}，θ=π 为 {fr['theta_pi']['odd_norm2D']:.3e}，比 {fr['odd_norm_ratio_pi_over_0']:.3g}。",
            f"- 冻结奇 Jx 两角代理 |A+C|/|B|={fr['odd_x_proxy']['abs_AC_over_B']:.3g}，相位差 {fr['odd_x_proxy']['phase_diff_deg']:.2f}°。",
            "- 权威 QED 数字仍是 4.98 倍、110.6°、ρ=1.456；本轮不改写它们。",
            "",
        ]
    lines += [
        "## 4. 下一轮",
        "",
        rec.get("next_step", "见长文。"),
        "",
        f"归档：`{OUT.as_posix()}`。长文：`线Q_A1门控GH5_20261029.md`。",
        "",
    ]
    one.write_text("\n".join(lines), encoding="utf-8")
    long_lines = [
        "# 八带 M0：A1 对称性门控 GH5（2026-10-29）",
        "",
        "本轮只做 Q6 协议下 Δ=0 与 SOC=0 的乘积 GH5。",
        "输入态仍是 D(0.5)S(0.3 e^{iθ})|0⟩，θ=0/π。",
        "两角不能分离 A 与 C。Δ=0 奇通道归零是定义性检验。",
        "",
        "## 审计",
        "",
        "```json",
        json.dumps(jsonable({k: {
            "ok": v["ok"], "min_gap": v["min_gap"], "issues": v["issues"],
            "warnings": v["warnings"], "delta": v["delta"], "lam_so": v["lam_so"],
            "plus_minus_H_identical": v["plus_minus_H_identical"],
            "P_map": [v["abs_P_ek_plus_vs_emk_minus"], v["abs_P_emk_plus_vs_ek_minus"]],
        } for k, v in rec.get("audits", {}).items()}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 摘要",
        "",
        "```json",
        json.dumps(jsonable(rec.get("summaries", {})), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 冻结对照",
        "",
        "```json",
        json.dumps(jsonable(fr), ensure_ascii=False, indent=2),
        "```",
        "",
        rec.get("verdict_text", ""),
        "",
        "## 未做",
        "",
        "- 新 QED / nmax=10 / 加密 M52 / 相关 GH（θ=π/2）",
        "- 把 Δ=0 归零写成新物理发现",
        "- 覆盖 Q6 旧目录",
        "",
    ]
    longp.write_text("\n".join(long_lines), encoding="utf-8")
    rec["reports"] = {"one_page": str(one), "long": str(longp)}


def analyze() -> dict:
    audits_raw = json.loads((OUT / "ctrl_audit.json").read_text(encoding="utf-8")) if (OUT / "ctrl_audit.json").is_file() else collect_audits()
    audits = audits_raw.get("controls", {})
    summaries = {}
    missing = []
    for ctrl, _tags in BATCHES:
        tags = [tag for tag, c, _, _ in JOBS if c == ctrl]
        if not all(job_finished_ok(tag) for tag in tags):
            missing.extend([tag for tag in tags if not job_finished_ok(tag)])
            continue
        summaries[ctrl] = summarize_ctrl(ctrl, audits[ctrl])
    frozen = None
    try:
        frozen = load_frozen_gh5()
    except Exception as exc:
        frozen = {"status": "frozen_load_failed", "error": str(exc)}
    d0 = summaries.get("d0")
    s0 = summaries.get("s0")
    verdict = "incomplete"
    notes = []
    if d0:
        if d0["odd_over_even_max"] < 1e-4:
            notes.append("Delta=0 odd H2 is consistent with numerical zero (definitional).")
            verdict = "delta0_odd_off"
        else:
            notes.append("Delta=0 odd H2 did not vanish at the expected noise floor.")
            verdict = "delta0_odd_residual"
    if s0:
        if s0["odd_over_even_max"] < 1e-4:
            notes.append("SOC=0 odd H2 is off at this k-pair; report gap before claiming a unique SOC mechanism.")
            if verdict.startswith("delta0"):
                verdict = verdict + "_soc0_odd_off"
            else:
                verdict = "soc0_odd_off"
        else:
            notes.append("SOC=0 odd H2 remains finite; not a hard gate at this protocol.")
            if verdict.startswith("delta0"):
                verdict = verdict + "_soc0_odd_on"
            else:
                verdict = "soc0_odd_on"
    next_step = "A2：同一 1+4+1 开关与主窗的有限时间二阶核，解释约 79° 相位差。"
    if "residual" in verdict or "odd_on" in verdict:
        next_step = "先解释门控失败（窗效应/高阶/能隙），不要调参强行恢复，也不要加密 M52。"
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "ok" if summaries and not missing else "partial",
        "verdict": verdict, "notes": notes, "next_step": next_step,
        "audits": audits, "summaries": summaries, "frozen_gh5": frozen,
        "missing_jobs": missing,
        "qed_not_rerun": True, "did_not_overwrite_q6": True,
        "used_product_GH": True, "used_linear_exact": True,
        "cannot_separate_A_and_C": True,
        "input_state": "D(0.5)S(0.3 exp(i theta))|0>",
        "utc": datetime.now(timezone.utc).isoformat(),
        "verdict_text": " ".join(notes),
    }
    write_reports(rec)
    dump(OUT / "a1_gate.json", rec)
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
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    audits = collect_audits()
    t0 = time.perf_counter()
    codes = {}
    for ctrl, tags in BATCHES:
        remaining = DRIVER_WALL_S - (time.perf_counter() - t0)
        au = audits["controls"][ctrl]
        if not au["ok"]:
            progress(f"skip {ctrl} audit issues={au['issues']}")
            continue
        if remaining < 60.0:
            progress(f"skip {ctrl}: remaining {remaining:.1f}s < 60")
            break
        progress(f"batch {ctrl} remaining={remaining:.1f}s")
        codes.update(launch_batch(tags, t0))
        if time.perf_counter() - t0 >= DRIVER_WALL_S:
            progress("driver wall reached after batch")
            break
    progress(f"jobs return {codes} driver_wall={time.perf_counter()-t0:.1f}s")
    done = [tag for tag, _, _, _ in JOBS if job_finished_ok(tag)]
    if done:
        rec = analyze()
        progress(f"analyze done verdict={rec.get('verdict')}")
        return 0 if rec.get("status") in ("ok", "partial") else 1
    rec = {
        "status": "timeout_partial", "stopped": True,
        "driver_wall_s": time.perf_counter() - t0, "returncodes": codes,
        "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                       for tag, _, _, _ in JOBS},
        "did_not_analyze_incomplete": True, "qed_not_rerun": True,
    }
    dump(OUT / "a1_gate.json", rec)
    progress("incomplete, no full analyze")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=str, default="")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        rec = collect_audits()
        print(json.dumps(jsonable({
            "status": rec.get("status"), "all_ok": rec.get("all_ok"),
            "controls": {k: {
                "ok": v["ok"], "min_gap": v["min_gap"], "issues": v["issues"],
                "warnings": v["warnings"], "delta": v["delta"], "lam_so": v["lam_so"],
                "plus_minus_H_identical": v["plus_minus_H_identical"],
            } for k, v in rec.get("controls", {}).items()},
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("all_ok") else 3
    if args.job:
        rec = run_one_job(args.job)
        return 0 if rec.get("status") in ("ok", "timeout_partial", "ctrl_abort") else 1
    if args.analyze_only:
        rec = analyze()
        print(json.dumps(jsonable({
            "status": rec.get("status"), "verdict": rec.get("verdict"),
            "notes": rec.get("notes"),
        }), ensure_ascii=False, indent=2))
        return 0 if rec.get("status") in ("ok", "partial") else 1
    return drive()


if __name__ == "__main__":
    raise SystemExit(main())
