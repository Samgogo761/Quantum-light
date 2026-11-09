#!/usr/bin/env python3
"""Qin_a2_1_4_1 GH5/MTEF displaced-thermal control vs frozen DS mix.

Only two new jobs: D(alpha) rho_th(sinh^2 r) D^dagger(alpha), N=+/-1.
Does not rerun QED, theta=0 GH5, or theta=pi GH5. Does not start Cn, SV,
GH7, extra k, nmax scan, G1/H2, D2, Floquet, or server.

nth = sinh^2(r) is thermal noise only. Total <n> = |alpha|^2 + nth.
Wigner is the isotropic positive Gaussian, M=< (Delta a)^2 > = 0.
Do not call mp.analytic_xp / gauss_hermite_nodes for this state.
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
OUT = ROOT / "results/q_line/qinput_qin_a2_141_gh5_thermal_20261120"
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
DRIVER_WALL_S = 600.0
JOB_WALL_S = 570.0
ALPHA = odd.ALPHA
R_SQ = odd.R_SQ
THETA_0 = 0.0
THETA_PI = float(np.pi)
JOBS = (
    ("th_p", "thermal", 1),
    ("th_m", "thermal", -1),
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
    "q8_m0_qinput_shared_kk_mtef_map_20261108.py":
        "cc69ae6aa963cb4ac8af9bcc308c24950c6db01dc9b94ca6a971253f4bf1b34c",
    "q8_m0_qinput_shared_kk_mtef_compare_20261109.py":
        "391a75b37470e2670d9e782bf580191fa9c67d786515fd17df5f2569b192de19",
    "q8_m0_qinput_shared_kk_mtef_gh5_20261110.py":
        "3af3c05b33c304238b252d601124a2f6016e162b234aaa1821221e7aac4a3c14",
}
FROZEN = tuple(HERE / name for name in EXPECTED_HASHES)
EPS_MARK = 0.20
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


def anomalous_M(var_x: float, var_p: float, cov_xp: float = 0.0) -> complex:
    """M = <(Delta a)^2> with X=a+a†, P=-i(a-a†)."""
    return 0.25 * ((var_x - var_p) + 2.0j * cov_xp)


def analytic_xp_ds(theta: float) -> dict:
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
        raise ValueError("frozen archives only ship theta=0 or pi")
    nth = float(np.sinh(r) ** 2)
    nbar = abs(a) ** 2 + nth
    mval = complex(-0.5 * np.exp(1j * th) * np.sinh(2.0 * r))
    return {
        "kind": "ds", "alpha": pf.cpack(a), "r": r, "theta": th,
        "mean_X": mx, "mean_P": mp, "var_X": vx, "var_P": vp, "cov_XP": 0.0,
        "nth_thermal_noise_only": nth,
        "mean_n_analytic": float(nbar),
        "mean_n_Weyl": 0.25 * (mx * mx + mp * mp + vx + vp) - 0.5,
        "M_anomalous": pf.cpack(mval),
        "squeeze_axis": axis,
        "not_a_coherent_state": True,
        "did_not_call_mp_analytic_xp_ds": True,
    }


def analytic_xp_thermal() -> dict:
    a = complex(ALPHA)
    r = float(R_SQ)
    mx, mp = 2.0 * a.real, 2.0 * a.imag
    nth = float(np.sinh(r) ** 2)
    vx = vp = float(np.cosh(2.0 * r))
    nbar = abs(a) ** 2 + nth
    nbar_weyl = 0.25 * (mx * mx + mp * mp + vx + vp) - 0.5
    if abs(nbar - nth) < 1e-6:
        raise RuntimeError("mean n collapsed onto nth; thermal noise is not total n")
    if abs(vx - 1.0) < 1e-12:
        raise RuntimeError("thermal Wigner collapsed onto a coherent state")
    return {
        "kind": "displaced_thermal",
        "alpha": pf.cpack(a), "r": r, "theta": None,
        "mean_X": mx, "mean_P": mp, "var_X": vx, "var_P": vp, "cov_XP": 0.0,
        "nth_thermal_noise_only": nth,
        "mean_n_analytic": float(nbar),
        "mean_n_Weyl": float(nbar_weyl),
        "M_anomalous": pf.cpack(0.0 + 0.0j),
        "positive_P": True,
        "isotropic_gaussian_wigner": True,
        "not_a_coherent_state": True,
        "nth_is_not_total_n": True,
        "did_not_call_mp_analytic_xp": True,
    }


def gh5_nodes_thermal() -> dict:
    """Same (xi, eta) CRN as frozen GH5; isotropic sx=sp=sqrt(cosh(2r))."""
    xi, w1 = hermgauss(ORDER)
    mom = analytic_xp_thermal()
    sx = sp = np.sqrt(mom["var_X"])
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
    mquad = anomalous_M(vx, vp, 0.0)
    return {
        "kind": "displaced_thermal", "order": ORDER, "n_nodes": len(nodes),
        "weight_sum": wsum, "quad_mean_X": float(mx), "quad_mean_P": float(mpv),
        "quad_var_X": float(vx), "quad_var_P": float(vp),
        "quad_M": pf.cpack(mquad),
        "target_mean_X": mom["mean_X"], "target_mean_P": mom["mean_P"],
        "target_var_X": mom["var_X"], "target_var_P": mom["var_P"],
        "nodes": nodes, "analytic": mom,
        "did_not_call_mp_gauss_hermite_nodes": True,
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


def ckpt_path(tag: str) -> Path:
    return OUT / f"ckpt_{tag}.npz"


def job_path(tag: str) -> Path:
    return OUT / f"job_{tag}.json"


def series_path(tag: str) -> Path:
    return OUT / f"series_{tag}.npz"


def save_ckpt(tag, kind, neel, n_done, n_nodes, acc, wsum, max_n, t_arr) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc = np.asarray(acc, complex)
    np.savez_compressed(
        ckpt_path(tag),
        acc_re=acc.real, acc_im=acc.imag, wsum=float(wsum),
        n_done=int(n_done), n_nodes=int(n_nodes), max_n=float(max_n),
        t=np.asarray(t_arr, float), kind=str(kind), neel=int(neel),
        state="displaced_thermal",
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


def load_series(tag: str, directory: Path):
    z = np.load(directory / f"series_{tag}.npz")
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


def run_one_job(tag: str) -> dict:
    if tag not in JOB_MAP:
        raise ValueError(tag)
    kind, neel = JOB_MAP[tag]
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
    nodes = gh5_nodes_thermal()["nodes"]
    ek0, emk0, mass = grounds[neel]
    model = models[neel]
    js = sample_js_for(N_FULL)
    ck = load_ckpt(tag)
    i0, acc, wsum, max_n, t_arr = 0, None, 0.0, 0.0, None
    if ck is not None:
        i0, acc, wsum, max_n, t_arr = ck["n_done"], ck["acc"], ck["wsum"], ck["max_n"], ck["t"]
        progress(f"{tag} resume n_done={i0}/{len(nodes)}")
    clock = dyn.WallClock(JOB_WALL_S)
    progress(f"{tag} start kind={kind} neel={neel} thermal nodes={len(nodes)} n_steps={N_FULL}")
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
            save_ckpt(tag, kind, neel, i + 1, len(nodes), acc, wsum, max_n, t_arr)
            if (i + 1) == len(nodes) or (i + 1) % 5 == 0:
                progress(f"{tag} node {i + 1}/{len(nodes)} wall={clock.elapsed():.1f}s")
    except TimeoutError as exc:
        timed_out = True
        timeout_msg = str(exc)
        progress(f"{tag} timeout {timeout_msg}")
    n_done = 0
    if ckpt_path(tag).is_file():
        n_done = int(load_ckpt(tag)["n_done"])
    if timed_out or n_done < len(nodes):
        rec = {
            "tag": tag, "kind": kind, "neel": int(neel), "order": ORDER,
            "state": "displaced_thermal",
            "status": "timeout_partial", "n_nodes": len(nodes), "n_done": n_done,
            "weight_sum": wsum, "max_abs_norm_minus_1": float(max_n),
            "wall_s": clock.elapsed(), "timeout_msg": timeout_msg,
            "n_max": 0, "protocol": PROTOCOL, "n_steps": N_FULL,
            "dt": env.t_total / N_FULL, "stopped": True, "electron": mass,
        }
        dump(job_path(tag), rec)
        return rec
    avg = acc / (wsum + 1e-30)
    save_series(tag, t_arr, avg)
    rec = {
        "tag": tag, "kind": kind, "neel": int(neel), "order": ORDER,
        "state": "displaced_thermal",
        "status": "ok", "n_nodes": len(nodes), "n_run": len(nodes),
        "n_done": len(nodes), "weight_sum": wsum,
        "max_abs_norm_minus_1": float(max_n),
        "wall_s": clock.elapsed(), "n_max": 0, "protocol": PROTOCOL,
        "n_steps": N_FULL, "dt": env.t_total / N_FULL,
        "n_samples": int(len(t_arr)), "t0": float(t_arr[0]), "t_end": float(t_arr[-1]),
        "electron": mass, "stopped": True,
        "wigner": analytic_xp_thermal(),
        "not_a_coherent_state": True,
        "nth_is_not_total_n": True,
    }
    dump(job_path(tag), rec)
    progress(f"{tag} done wall={clock.elapsed():.1f}s")
    return rec


def collect_resources() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "resources", "source_hashes": hashes, "hash_abort": bad,
        "state": "D(alpha) rho_th(sinh^2 r) D^dagger(alpha)",
        "alpha": pf.cpack(complex(ALPHA)), "r": float(R_SQ),
        "did_not_call_mp_analytic_xp": True,
        "stopped": True,
    }
    if bad:
        rec["status"] = "hash_abort"
        dump(OUT / "resources.json", rec)
        return rec
    th = analytic_xp_thermal()
    d0 = analytic_xp_ds(THETA_0)
    dp = analytic_xp_ds(THETA_PI)
    gh = gh5_nodes_thermal()
    v_avg = 0.5 * (d0["var_X"] + dp["var_X"])
    nbar = th["mean_n_analytic"]
    alpha_abs2 = abs(complex(ALPHA)) ** 2
    class_lo, class_hi = 1.0, 1.0 + 4.0 * (nbar - alpha_abs2)
    rec.update({
        "thermal": th,
        "theta0": d0,
        "theta_pi": dp,
        "gh5_quad": {
            "mean_X": gh["quad_mean_X"], "mean_P": gh["quad_mean_P"],
            "var_X": gh["quad_var_X"], "var_P": gh["quad_var_P"],
            "M": gh["quad_M"], "weight_sum": gh["weight_sum"],
            "n_nodes": gh["n_nodes"],
        },
        "same_mean_X": bool(abs(th["mean_X"] - d0["mean_X"]) < 1e-14
                            and abs(th["mean_X"] - dp["mean_X"]) < 1e-14),
        "same_mean_P": bool(abs(th["mean_P"] - d0["mean_P"]) < 1e-14
                            and abs(th["mean_P"] - dp["mean_P"]) < 1e-14),
        "same_nbar": bool(abs(th["mean_n_analytic"] - d0["mean_n_analytic"]) < 1e-12
                          and abs(th["mean_n_analytic"] - dp["mean_n_analytic"]) < 1e-12),
        "thermal_is_average_covariance": bool(abs(v_avg - th["var_X"]) < 1e-12),
        "M_thermal_zero": bool(abs(cz(th["M_anomalous"])) < 1e-15),
        "nth_is_not_total_n": bool(abs(th["mean_n_analytic"] - th["nth_thermal_noise_only"]) > 0.1),
        "not_a_coherent_state": bool(abs(th["var_X"] - 1.0) > 0.05),
        "classical_P_varX_interval": [class_lo, class_hi],
        "thermal_varX_inside_classical_interval": bool(class_lo - 1e-12 <= th["var_X"] <= class_hi + 1e-12),
        "ds_varX_outside_classical_interval": bool(
            d0["var_X"] < class_lo - 1e-12 and dp["var_X"] > class_hi + 1e-12),
    })
    rec["resource_match_ok"] = bool(
        rec["same_mean_X"] and rec["same_mean_P"] and rec["same_nbar"]
        and rec["thermal_is_average_covariance"] and rec["M_thermal_zero"]
        and rec["nth_is_not_total_n"] and rec["not_a_coherent_state"]
    )
    dump(OUT / "resources.json", rec)
    progress(
        f"resources match={rec['resource_match_ok']} "
        f"nbar={th['mean_n_analytic']:.6f} nth={th['nth_thermal_noise_only']:.6f} "
        f"varX_th={th['var_X']:.6f} M=0"
    )
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


def arrow(z, lim, x0, y0, w, h, color, label, lx, ly, mark="circle"):
    cx, cy = x0 + w / 2, y0 + h / 2
    px, py = map_xy(z, lim, x0, y0, w, h)
    head = (
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>'
        if mark == "circle" else
        f'<rect x="{px-4:.1f}" y="{py-4:.1f}" width="8" height="8" fill="{color}"/>'
    )
    return (
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
        f'stroke="{color}" stroke-width="2"/>'
        f"{head}"
        f'<text x="{lx}" y="{ly}" font-size="11" fill="{color}">{label}</text>'
    )


def figure_h2_plane(z0, zpi, zth, zmix, path):
    zs = [z0, zpi, zth, zmix]
    lim = max(abs(z) for z in zs) * 1.25 + 1e-30
    x0, y0, w, h = 90, 40, 740, 430
    body = svg_axes(x0, y0, w, h, "Re Jx (H2 odd)", "Im Jx (H2 odd)",
                    "H2 odd Jx: thermal vs squeeze mix (GH5)")
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    body += arrow(z0, lim, x0, y0, w, h, "#4c6b8a", "θ=0", 80, 36)
    body += arrow(zpi, lim, x0, y0, w, h, "#c45c26", "θ=π", 140, 36)
    body += arrow(zth, lim, x0, y0, w, h, "#6b3fa0", "thermal", 200, 36)
    body += arrow(zmix, lim, x0, y0, w, h, "#2f9e44", "mix", 280, 36, mark="square")
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "same α, same ⟨n⟩, same mean covariance; thermal is one positive-P control, not all classical inputs; "
        "GH5 not QED; integer Fourier not certified HHG</text>"
    )
    return write_svg(path, body)


def figure_windows(rows, path):
    lim = max(abs(z) for _, z, _ in rows) * 1.25 + 1e-30
    x0, y0, w, h = 90, 40, 740, 430
    body = svg_axes(x0, y0, w, h, "Re R2 Jx (H2 odd)", "Im R2 Jx (H2 odd)",
                    "H2 R2 = J_th − J_mix, three windows")
    cx, cy = x0 + w / 2, y0 + h / 2
    body += f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+w}" y2="{cy:.1f}" stroke="#8a9096"/>'
    body += f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#8a9096"/>'
    lx = 80
    for label, z, color in rows:
        body += arrow(z, lim, x0, y0, w, h, color, label, lx, 36)
        lx += 150
    body += (
        '<text x="460" y="548" font-size="10" fill="#5c5148">'
        "ε2 uses the full complex R2. Engineering marks are not physical laws.</text>"
    )
    return write_svg(path, body)


def write_reports(rec: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    h2 = rec["h2_main"]
    res = rec["resources"]
    th = res["thermal"]
    a0 = res["theta0"]
    ap = res["theta_pi"]
    v = rec["verdict"]
    ans = rec["answer"]
    full = []
    full.append("# 量子光多周期磁序响应：位移热态 GH5/MTEF 对照\n")
    full.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。")
    full.append("冻结 θ=0 / θ=π 精确 QED 与 GH5，全部只读，不重算。本轮仅两路新 GH5：位移热态，N=±1。")
    full.append("ρ = D(α) ρ_th(sinh² r) D†(α)，α=0.5，r=0.3。nth 只是热噪声光子数，不是总 ⟨n⟩。")
    full.append("`stopped = true`。位移热态是一个正 P 对照，不是全部经典输入的最优界。\n")
    full.append(f"**结论：{ans}**\n")
    full.append("## 输入资源（固定位移、固定平均光子数、平均协方差）\n")
    full.append("| 量 | θ=0 | θ=π | 位移热态 |\n|---|---|---|---|")
    full.append(f"| ⟨X⟩ | {a0['mean_X']:.6f} | {ap['mean_X']:.6f} | {th['mean_X']:.6f} |")
    full.append(f"| ⟨P⟩ | {a0['mean_P']:.6f} | {ap['mean_P']:.6f} | {th['mean_P']:.6f} |")
    full.append(f"| ⟨n⟩ | {a0['mean_n_analytic']:.6f} | {ap['mean_n_analytic']:.6f} | {th['mean_n_analytic']:.6f} |")
    full.append(f"| nth（仅热噪声） | {a0['nth_thermal_noise_only']:.6f} | {ap['nth_thermal_noise_only']:.6f} | {th['nth_thermal_noise_only']:.6f} |")
    full.append(f"| Var(X) | {a0['var_X']:.6f} | {ap['var_X']:.6f} | {th['var_X']:.6f} |")
    full.append(f"| Var(P) | {a0['var_P']:.6f} | {ap['var_P']:.6f} | {th['var_P']:.6f} |")
    full.append(f"| M=⟨(Δa)²⟩ | {cz(a0['M_anomalous']).real:+.5f} | {cz(ap['M_anomalous']).real:+.5f} | {cz(th['M_anomalous']).real:+.5f} |")
    lo, hi = res["classical_P_varX_interval"]
    full.append(
        f"\n资源匹配 `{res['resource_match_ok']}`。"
        f"(V₀+V_π)/2 = V_th 为 `{res['thermal_is_average_covariance']}`。"
        f"正 P 的 Var(X) 区间约 [{lo:.5f}, {hi:.5f}]；"
        f"热态在区间内 `{res['thermal_varX_inside_classical_interval']}`，"
        f"两压缩态在区间外 `{res['ds_varX_outside_classical_interval']}`。\n"
    )
    full.append("该差别证明**输入资源**的非经典性，但不直接证明**电流响应**的非经典性。\n")
    full.append("## 主窗 H2 磁序奇复源（Jx）\n")
    full.append("| 量 | θ=0 GH5 | θ=π GH5 | mix | 位移热态 |\n|---|---|---|---|---|")
    full.append(
        f"| \\(\\lvert J_{{2,x}}^o\\rvert\\) | {h2['abs_jx_0']:.3e} | {h2['abs_jx_pi']:.3e} "
        f"| {h2['abs_jx_mix']:.3e} | {h2['abs_jx_th']:.3e} |"
    )
    full.append(
        f"| arg(Jx) | {h2['phase_0']:.3f} | {h2['phase_pi']:.3f} "
        f"| {h2['phase_mix']:.3f} | {h2['phase_th']:.3f} |"
    )
    full.append(
        f"\n\\(J_{{\\mathrm{{mix}}}}=[J(\\theta=0)+J(\\theta=\\pi)]/2\\)，"
        f"\\(R_2=J_{{\\mathrm{{th}}}}-J_{{\\mathrm{{mix}}}}="
        f"{h2['R2_re']:.3e}{h2['R2_im']:+.3e}i\\)，"
        f"|R₂|={h2['abs_R2']:.3e}，"
        f"ε₂=|R₂|/|J(π)−J(0)|={h2['eps2']:.3f}。\n"
    )
    full.append("| 窗 | |R₂| | arg(R₂) | ε₂ |\n|---|---|---|---|")
    full.append(f"| 前 [1,3]T₀ | {h2['abs_R2_front']:.3e} | {h2['phase_R2_front']:.3f} | {h2['eps2_front']:.3f} |")
    full.append(f"| 主 [2,4]T₀ | {h2['abs_R2']:.3e} | {h2['phase_R2']:.3f} | {h2['eps2']:.3f} |")
    full.append(f"| 后 [3,5]T₀ | {h2['abs_R2_back']:.3e} | {h2['phase_R2_back']:.3f} | {h2['eps2_back']:.3f} |\n")
    full.append(
        "工程分类标记（不是物理定律）："
        f"`ε₂<{EPS_MARK}` 记为与二阶输入统计主导相容；"
        f"`ε₂≥{EPS_MARK}` 记为相同一、二阶统计不足以决定该响应。\n"
    )
    full.append(f"verdict `{v}`。\n")
    full.append("## 解释边界\n")
    full.append("- 本轮是 GH5/MTEF，不是精确 QED。")
    full.append("- R₂ 小 ≠ 证明量子优势；R₂ 大 ≠ 否定压缩输入的非经典性。")
    full.append("- 位移热态是一个正 P 对照，不是全部经典随机输入的最优界。")
    full.append("- 两压缩态 Var(X) 落在正 P 区间外，只证明输入资源非经典。")
    full.append("- 两周期矩形 Fourier ≠ 已认证 HHG。")
    full.append("- 未启动新 QED、Cn、SV、GH7、额外 k、nmax、G1/H2 或服务器。\n")
    full.append("完成后停机。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q1a_GH5_thermal_20261120.md").write_text(
        "\n".join(full) + "\n", encoding="utf-8")
    one = []
    one.append("# 量子光多周期磁序响应：位移热态 GH5（一页）\n")
    one.append(f"`{TASK_TAG}`；执行 {RECORD_DATE}。协议 `{PROTOCOL}`。冻结 QED 与两压缩角 GH5。`stopped = true`。\n")
    one.append(f"**{ans}**\n")
    one.append("| 量 | 结果 |\n|---|---|")
    one.append(
        f"| 资源 | ⟨X⟩=1, ⟨n⟩={th['mean_n_analytic']:.4f}, nth={th['nth_thermal_noise_only']:.4f}, "
        f"Var(X)_th={th['var_X']:.3f}, M=0 |"
    )
    one.append(f"| 主窗 |R₂| / ε₂ | {h2['abs_R2']:.3e} / {h2['eps2']:.3f} |")
    one.append(f"| 分类 | `{v}` |")
    one.append("| 正 P 区间 / 压缩态 | 输入非经典已成立；电流非经典未证明 |")
    one.append("| 新 QED / 服务器 | 现在不启动 |\n")
    one.append("**外审问题：** 具有相同平均光场、平均光子数及平均协方差，但不同高阶统计的位移热态，是否产生与两压缩态等权混合相同的磁序奇 H2 复响应？\n")
    one.append(f"**{rec['answer_short']}**\n")
    one.append("停机。不自动开 QED、Cn、SV、GH7 或服务器。\n")
    (DOCS / "线Q_量子光多周期磁序响应_Q1a_GH5_thermal_一页_20261120.md").write_text(
        "\n".join(one) + "\n", encoding="utf-8")


def analyze() -> dict:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        rec = {"status": "hash_abort", "hash_mismatch": bad, "stopped": True}
        dump(OUT / "gh5_thermal.json", rec)
        return rec
    missing = [tag for tag, _, _ in JOBS if not job_finished_ok(tag)]
    if missing:
        rec = {"status": "missing_jobs", "missing": missing, "stopped": True,
               "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                              for tag, _, _ in JOBS}}
        dump(OUT / "gh5_thermal.json", rec)
        return rec
    for directory, tag in (
        (THETA0_DIR, "ds_p"), (THETA0_DIR, "ds_m"),
        (THETAPI_DIR, "ds_p"), (THETAPI_DIR, "ds_m"),
    ):
        if not (directory / f"series_{tag}.npz").is_file():
            rec = {"status": "missing_frozen_archive", "dir": str(directory), "tag": tag, "stopped": True}
            dump(OUT / "gh5_thermal.json", rec)
            return rec
    res = collect_resources()
    if not res.get("resource_match_ok"):
        rec = {"status": "resource_mismatch", "resources": res, "stopped": True}
        dump(OUT / "gh5_thermal.json", rec)
        return rec
    env = make_env()
    T0 = env.period
    omega0 = pf.OMEGA0
    t_th_p, j_th_p = load_series("th_p", OUT)
    t_th_m, j_th_m = load_series("th_m", OUT)
    t0_p, j0_p = load_series("ds_p", THETA0_DIR)
    t0_m, j0_m = load_series("ds_m", THETA0_DIR)
    tpi_p, jpi_p = load_series("ds_p", THETAPI_DIR)
    tpi_m, jpi_m = load_series("ds_m", THETAPI_DIR)
    t = t_th_p
    for label, tt in (("th_m", t_th_m), ("0_p", t0_p), ("0_m", t0_m), ("pi_p", tpi_p), ("pi_m", tpi_m)):
        if not np.allclose(tt, t, rtol=0, atol=1e-9):
            rec = {"status": "time_grid_mismatch", "tag": label, "stopped": True}
            dump(OUT / "gh5_thermal.json", rec)
            return rec
    even_th, odd_th = odd.even_odd(j_th_p, j_th_m)
    even_0, odd_0 = odd.even_odd(j0_p, j0_m)
    even_pi, odd_pi = odd.even_odd(jpi_p, jpi_m)
    odd_mix = 0.5 * (odd_0 + odd_pi)
    even_mix = 0.5 * (even_0 + even_pi)
    r_odd = odd_th - odd_mix
    d_odd = odd_pi - odd_0
    arrs = {
        "J_odd_0": odd_0, "J_odd_pi": odd_pi, "J_odd_mix": odd_mix,
        "J_odd_th": odd_th, "R_odd": r_odd, "dJ_odd": d_odd,
        "J_even_0": even_0, "J_even_pi": even_pi, "J_even_th": even_th,
        "J_even_mix": even_mix,
    }
    win_main = (2.0 * T0, 4.0 * T0)
    win_front = (1.0 * T0, 3.0 * T0)
    win_back = (3.0 * T0, 5.0 * T0)
    main = pack_win("plateau_center_2T", t, omega0, T0, *win_main, arrs)
    front = pack_win("front_2T", t, omega0, T0, *win_front, arrs)
    back = pack_win("back_2T", t, omega0, T0, *win_back, arrs)

    def take(winobj, key):
        return h2_jx(winobj["slots"], key)

    z0 = take(main, "J_odd_0")
    zpi = take(main, "J_odd_pi")
    zmix = take(main, "J_odd_mix")
    zth = take(main, "J_odd_th")
    zr = take(main, "R_odd")
    zd = take(main, "dJ_odd")
    zrf = take(front, "R_odd")
    zrb = take(back, "R_odd")
    eps2 = float(zr[2] / max(zd[2], 1e-30))
    eps2_f = float(zrf[2] / max(zd[2], 1e-30))
    eps2_b = float(zrb[2] / max(zd[2], 1e-30))
    dcomplex = float(abs(zrb[0] - zrf[0]) / max(zr[2], 1e-30))
    # time-domain scale reminder
    rms_odd_th = float(np.sqrt(np.mean(np.abs(odd_th[:, 0]) ** 2)))
    rms_even_th = float(np.sqrt(np.mean(np.abs(even_th[:, 0]) ** 2)))
    if eps2 < EPS_MARK:
        verdict = "second_order_statistics_compatible_not_quantum_advantage"
        answer = (
            "主窗磁序奇 H2 上，位移热态与两压缩态等权混合的复差 R₂ 相对压缩角差很小。"
            "这与“当前 H2 响应主要受输入光一、二阶统计控制”相容，"
            "但不能证明量子优势，也不能把单个正 P 热态对照写成全部经典输入的不可模拟性。"
        )
        answer_short = "R₂ 小：与二阶统计主导相容，非量子优势证明"
    else:
        verdict = "same_first_two_moments_do_not_fix_h2_odd"
        answer = (
            "主窗磁序奇 H2 上，位移热态与两压缩态等权混合的复差 R₂ 相对压缩角差并不小。"
            "相同的一、二阶输入统计不足以决定该电子响应，需要更高阶统计、非线性动力学或方法误差。"
            "这既不直接证明、也不否定量子优势。"
        )
        answer_short = "R₂ 大：前两阶不足以决定响应，非量子优势判决"
    FIG.mkdir(parents=True, exist_ok=True)
    f_plane = figure_h2_plane(z0[0], zpi[0], zth[0], zmix[0], FIG / "h2_complex_plane_thermal_vs_mix.svg")
    f_win = figure_windows(
        [("front", zrf[0], "#4c6b8a"), ("main", zr[0], "#2f9e44"), ("back", zrb[0], "#c45c26")],
        FIG / "h2_R2_three_windows.svg",
    )
    jobs = {tag: load_job(tag) for tag, _, _ in JOBS}
    rec = {
        "task_tag": TASK_TAG, "record_date": RECORD_DATE, "protocol": PROTOCOL,
        "status": "ok", "stopped": True, "verdict": verdict,
        "answer": answer, "answer_short": answer_short,
        "n_max": 0, "order": ORDER, "n_nodes": N_NODES, "n_steps": N_FULL,
        "dt": env.t_total / N_FULL,
        "theta0_archive": str(THETA0_DIR.as_posix()),
        "thetapi_archive": str(THETAPI_DIR.as_posix()),
        "frozen_not_rerun": True, "qed_not_rerun": True,
        "state": "D(alpha) rho_th(sinh^2 r) D^dagger(alpha)",
        "source_hashes": hashes, "hash_abort": [],
        "resources": res, "windows": {"main": main, "front": front, "back": back},
        "h2_main": {
            "jx_0": pack_c(z0[0]), "jx_pi": pack_c(zpi[0]),
            "jx_mix": pack_c(zmix[0]), "jx_th": pack_c(zth[0]),
            "jx_R2": pack_c(zr[0]), "jx_dtheta": pack_c(zd[0]),
            "abs_jx_0": z0[2], "abs_jx_pi": zpi[2],
            "abs_jx_mix": zmix[2], "abs_jx_th": zth[2],
            "abs_R2": zr[2], "abs_dtheta": zd[2],
            "phase_0": z0[3], "phase_pi": zpi[3],
            "phase_mix": zmix[3], "phase_th": zth[3], "phase_R2": zr[3],
            "R2_re": float(zr[0].real), "R2_im": float(zr[0].imag),
            "eps2": eps2, "eps2_front": eps2_f, "eps2_back": eps2_b,
            "abs_R2_front": zrf[2], "phase_R2_front": zrf[3],
            "abs_R2_back": zrb[2], "phase_R2_back": zrb[3],
            "delta_complex_R2": dcomplex,
            "engineering_marks_not_physical_laws": True,
            "eps_mark": EPS_MARK, "dcomplex_mark": DCOMPLEX_MARK,
            "note": "eps2 uses full complex R2 = J_th - J_mix over |J(pi)-J(0)|, H2 Jx.",
        },
        "time_domain_scale": {
            "rms_Jx_odd_thermal": rms_odd_th,
            "rms_Jx_even_thermal": rms_even_th,
            "odd_over_even": rms_odd_th / max(rms_even_th, 1e-30),
        },
        "jobs": {tag: {k: jobs[tag].get(k) for k in
                       ("status", "n_run", "n_done", "wall_s", "max_abs_norm_minus_1",
                        "weight_sum", "kind", "neel", "state")}
                 for tag, _, _ in JOBS},
        "figures": {"complex_plane": f_plane, "three_windows": f_win},
        "physics_boundaries": {
            "GH5_thermal_is_not_exact_QED": True,
            "small_R2_is_not_quantum_advantage": True,
            "large_R2_does_not_negate_input_nonclassicality": True,
            "thermal_is_one_positive_P_control_not_all_classical_inputs": True,
            "squeezed_VarX_outside_classical_interval_is_input_not_current": True,
            "integer_Fourier_is_not_certified_HHG": True,
            "nth_is_not_total_n": True,
            "thermal_is_not_a_coherent_state": True,
        },
        "did_not": [
            "rerun QED", "rerun theta=0 GH5", "rerun theta=pi GH5",
            "Cn", "squeezed vacuum", "GH7", "G1", "H2 extra k",
            "nmax scan", "D2", "Floquet", "server",
        ],
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "gh5_thermal.json", rec)
    write_reports(rec)
    return rec


def drive() -> int:
    hashes = source_hashes()
    bad = hash_abort(hashes)
    if bad:
        dump(OUT / "gh5_thermal.json", {"status": "hash_abort", "hash_mismatch": bad, "stopped": True})
        progress("hash abort, no propagation")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    res = collect_resources()
    if not res.get("resource_match_ok"):
        dump(OUT / "gh5_thermal.json", {"status": "resource_mismatch", "resources": res, "stopped": True})
        progress("resource mismatch, no propagation")
        return 3
    t0 = time.perf_counter()
    procs, logs = {}, {}
    for tag, kind, neel in JOBS:
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
    codes = {tag: (procs[tag].returncode if tag in procs else 0) for tag, _, _ in JOBS}
    progress(f"jobs return {codes} driver_wall={time.perf_counter()-t0:.1f}s")
    if all(job_finished_ok(tag) for tag, _, _ in JOBS):
        rec = analyze()
        progress(f"analyze done verdict={rec.get('verdict')}")
        return 0
    rec = {
        "status": "timeout_partial", "stopped": True,
        "driver_wall_s": time.perf_counter() - t0, "returncodes": codes,
        "job_status": {tag: None if load_job(tag) is None else load_job(tag).get("status")
                       for tag, _, _ in JOBS},
        "jobs_n_done": {tag: None if load_job(tag) is None else load_job(tag).get("n_done")
                        for tag, _, _ in JOBS},
        "did_not_analyze_incomplete": True, "qed_not_rerun": True,
    }
    dump(OUT / "gh5_thermal.json", rec)
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
            "nth_is_not_total_n": rec.get("nth_is_not_total_n"),
            "not_a_coherent_state": rec.get("not_a_coherent_state"),
            "thermal_is_average_covariance": rec.get("thermal_is_average_covariance"),
            "mean_n": None if rec.get("thermal") is None else rec["thermal"]["mean_n_analytic"],
            "nth": None if rec.get("thermal") is None else rec["thermal"]["nth_thermal_noise_only"],
            "var_X": None if rec.get("thermal") is None else rec["thermal"]["var_X"],
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
