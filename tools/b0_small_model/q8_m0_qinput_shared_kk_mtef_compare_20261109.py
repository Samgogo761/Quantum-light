#!/usr/bin/env python3
"""First QED vs Wigner/MTEF compare on the frozen shared-mode object.

GH1 negative control, GH3 (36 Ehrenfest), then GH3 dt/2 and GH5 inside
a 180 s wall. Times locked to archived 20261102 samples. Not HHG, not
advantage, not a BSV pulse. Does not rewrite 20261026-20261108 archives.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_mtef_map_20261108 as mp
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/q_line/qinput_a_shared_kk_mtef_compare_20261109"
MODEL_ID = "M0_shared_mode_k_minus_k_mtef_qed_compare_v1"
RECORD_DATE = "2026-11-09"
WALL_CAP_S = 180.0
N_STEPS = mp.N_STEPS_LOCK
SAMPLE_EVERY = mp.SAMPLE_EVERY
N_STEPS_HALF = 2 * N_STEPS
ARCHIVE_QED = ROOT / "results/q_line/qinput_a_shared_kk_ds_odd_20261102/compare.json"
ARCHIVE_COH_DT = ROOT / "results/q_line/qinput_a_shared_kk_coh_dt_20261105/coh_dt.json"
OLD_ARCHIVES = mp.OLD_ARCHIVES + (
    ROOT / "results/q_line/qinput_a_shared_kk_mtef_map_20261108/map.json",
    ARCHIVE_QED,
    ARCHIVE_COH_DT,
)
KINDS = ("coherent", "ds")
NEELS = (1, -1)
JOBS = (("coh_p", "coherent", 1), ("coh_m", "coherent", -1),
        ("ds_p", "ds", 1), ("ds_m", "ds", -1))


def _progress(progress, msg):
    mp._progress(progress, msg)


def unpack_rows(rows):
    return np.array([[complex(z["re"], z["im"]) for z in row] for row in rows], complex)


def pack_j(j):
    return [[pf.cpack(z) for z in row] for row in np.asarray(j, complex)]


def sample_js(n_steps: int = N_STEPS, sample_every: int = SAMPLE_EVERY):
    js = list(range(0, n_steps, sample_every))
    if not js or js[-1] != n_steps - 1:
        js.append(n_steps - 1)
    return js


def sample_js_half(n_old: int = N_STEPS, sample_every: int = SAMPLE_EVERY):
    return [2 * j + 1 for j in sample_js(n_old, sample_every)]


def times_of(n_steps, js, t_ramp):
    dt = t_ramp / n_steps
    return np.array([(j + 1) * dt for j in js], float)


def load_qed():
    with ARCHIVE_QED.open(encoding="utf-8") as fh:
        o = json.load(fh)
    ser = o["compare"]["series"]
    out = {
        "t": np.asarray(ser["t"], float),
        "J_odd_coh": unpack_rows(ser["J_odd_coh"]),
        "J_odd_ds": unpack_rows(ser["J_odd_ds"]),
        "dJ_odd": unpack_rows(ser["dJ_odd"]),
        "J_even_coh": unpack_rows(ser["J_even_coh"]),
        "J_even_ds": unpack_rows(ser["J_even_ds"]),
        "max_abs": o["compare"]["max_abs"],
        "dt": o["cost"]["dt"],
        "n_steps": o["cost"]["n_steps"],
        "hash": mp.sha256_file(ARCHIVE_QED),
    }
    if ARCHIVE_COH_DT.is_file():
        with ARCHIVE_COH_DT.open(encoding="utf-8") as fh:
            h = json.load(fh)
        ser_h = h.get("compare", {}).get("series", {})
        if "dJ_odd_half" in ser_h:
            out["dJ_odd_qed_dt2"] = unpack_rows(ser_h["dJ_odd_half"])
            out["qed_dt2_max_abs"] = h["compare"]["max_abs"]
    return out


def build_bundle(geo, k):
    models, grounds = {}, {}
    for neel in NEELS:
        models[neel] = sh.build_shared(
            geo, R0, k, neel, n_max=0, omega0=pf.OMEGA0, g=pf.G_ABS)
        ek, emk, mass = ds.ground_pair_neel(geo, k, neel)
        grounds[neel] = (ek, emk, mass)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    return models, grounds, env


def one_traj(model, ek0, emk0, X0, P0, env, n_steps, js, clock):
    ek, emk = np.array(ek0, complex), np.array(emk0, complex)
    X, P = float(X0), float(P0)
    dt = env.t_ramp / n_steps
    want = set(int(j) for j in js)
    j_rows, meta = [], []
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
            meta.append({"t": t, "X": X, "P": P, "s": float(env.s(t))})
    return np.asarray(j_rows, complex), meta, max_n, X, P


def average_kind_neel(kind, neel, order, models, grounds, env, n_steps, js, clock, progress, tag):
    nodes = mp.gauss_hermite_nodes(kind, order)["nodes"]
    ek0, emk0, mass = grounds[neel]
    model = models[neel]
    acc, wsum, max_n, n_run = None, 0.0, 0.0, 0
    _progress(progress, f"{tag} start kind={kind} neel={neel} order={order} nodes={len(nodes)}")
    for i, node in enumerate(nodes):
        j_rows, meta, nrm, X, P = one_traj(
            model, ek0, emk0, node["X"], node["P"], env, n_steps, js, clock)
        w = float(node["weight"])
        acc = w * j_rows if acc is None else acc + w * j_rows
        wsum += w
        max_n = max(max_n, nrm)
        n_run += 1
        if (i + 1) == len(nodes) or (i + 1) % 5 == 0:
            _progress(progress, f"{tag} node {i + 1}/{len(nodes)} wall={clock.elapsed():.2f}s")
    avg = acc / (wsum + 1e-30)
    t = np.array([m["t"] for m in meta], float)
    return {
        "kind": kind, "neel": int(neel), "order": int(order),
        "n_nodes": len(nodes), "n_run": n_run, "weight_sum": wsum,
        "t": t, "J": avg, "max_abs_norm_minus_1": float(max_n),
        "electron": mass, "end_X": float(X), "end_P": float(P),
    }


def four_jobs(order, models, grounds, env, n_steps, js, clock, progress, prefix):
    runs = {}
    for tag, kind, neel in JOBS:
        runs[tag] = average_kind_neel(
            kind, neel, order, models, grounds, env, n_steps, js, clock,
            progress, f"{prefix}_{tag}")
    t0 = runs["coh_p"]["t"]
    for tag in runs:
        if not np.allclose(runs[tag]["t"], t0):
            raise RuntimeError(f"time grid mismatch {tag}")
    even_c, odd_c = ds.even_odd(runs["coh_p"]["J"], runs["coh_m"]["J"])
    even_d, odd_d = ds.even_odd(runs["ds_p"]["J"], runs["ds_m"]["J"])
    d_odd = odd_d - odd_c
    d_even = even_d - even_c
    max_abs = {
        "dJ_odd_x": float(np.max(np.abs(d_odd[:, 0]))),
        "dJ_odd_y": float(np.max(np.abs(d_odd[:, 1]))),
        "dJ_even_x": float(np.max(np.abs(d_even[:, 0]))),
        "dJ_even_y": float(np.max(np.abs(d_even[:, 1]))),
        "J_odd_coh_x": float(np.max(np.abs(odd_c[:, 0]))),
        "J_odd_coh_y": float(np.max(np.abs(odd_c[:, 1]))),
        "J_odd_ds_x": float(np.max(np.abs(odd_d[:, 0]))),
        "J_odd_ds_y": float(np.max(np.abs(odd_d[:, 1]))),
        "imag_J_floor": float(max(np.max(np.abs(runs[t]["J"].imag)) for t in runs)),
        "max_abs_norm_minus_1": float(max(runs[t]["max_abs_norm_minus_1"] for t in runs)),
    }
    return {
        "order": int(order), "n_steps": int(n_steps), "n_traj": 4 * (order * order),
        "t": t0, "even_c": even_c, "odd_c": odd_c, "even_d": even_d, "odd_d": odd_d,
        "d_odd": d_odd, "d_even": d_even, "max_abs": max_abs,
        "jobs": {tag: {k: runs[tag][k] for k in runs[tag] if k != "J"} for tag in runs},
    }


def vs_qed(block, qed):
    if not np.allclose(block["t"], qed["t"]):
        raise RuntimeError("MTEF times are not locked to 20261102")
    d_m, d_q = block["d_odd"], qed["dJ_odd"]
    diff = d_m - d_q
    scale_x = max(float(np.max(np.abs(d_q[:, 0]))), 1e-30)
    scale_y = max(float(np.max(np.abs(d_q[:, 1]))), 1e-30)
    def corr(a, b):
        ar, br = np.real(a), np.real(b)
        if np.std(ar) < 1e-30 or np.std(br) < 1e-30:
            return 0.0
        return float(np.corrcoef(ar, br)[0, 1])
    return {
        "n_times": int(d_q.shape[0]),
        "aligned_without_interpolation": True,
        "qed_archive": "20261102 compare.json",
        "max_abs_MTEF_minus_QED_x": float(np.max(np.abs(diff[:, 0]))),
        "max_abs_MTEF_minus_QED_y": float(np.max(np.abs(diff[:, 1]))),
        "rms_MTEF_minus_QED_x": float(np.sqrt(np.mean(np.abs(diff[:, 0]) ** 2))),
        "rms_MTEF_minus_QED_y": float(np.sqrt(np.mean(np.abs(diff[:, 1]) ** 2))),
        "ratio_to_qed_dJ_odd_x": float(np.max(np.abs(diff[:, 0])) / scale_x),
        "ratio_to_qed_dJ_odd_y": float(np.max(np.abs(diff[:, 1])) / scale_y),
        "qed_dJ_odd_max_x": float(np.max(np.abs(d_q[:, 0]))),
        "qed_dJ_odd_max_y": float(np.max(np.abs(d_q[:, 1]))),
        "mtef_dJ_odd_max_x": float(np.max(np.abs(d_m[:, 0]))),
        "mtef_dJ_odd_max_y": float(np.max(np.abs(d_m[:, 1]))),
        "corr_x": corr(d_m[:, 0], d_q[:, 0]),
        "corr_y": corr(d_m[:, 1], d_q[:, 1]),
        "qed_definition": "J_odd=(J_+N-J_-N)/2, Delta=J_odd_DS-J_odd_Calpha",
    }


def series_pack(block):
    return {
        "t": block["t"].tolist(),
        "J_odd_coh": pack_j(block["odd_c"]),
        "J_odd_ds": pack_j(block["odd_d"]),
        "J_even_coh": pack_j(block["even_c"]),
        "J_even_ds": pack_j(block["even_d"]),
        "dJ_odd": pack_j(block["d_odd"]),
        "dJ_even": pack_j(block["d_even"]),
    }


def write_plots(out_dir: Path, qed, gh1, gh3, gh5, dt2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = qed["t"]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6), sharex=False)
    ax = axes[0, 0]
    ax.plot(t, qed["dJ_odd"][:, 0].real, color="#1d4e89", lw=1.6, label=r"QED $\Delta J_{\mathrm{odd},x}$")
    ax.plot(t, gh3["d_odd"][:, 0].real, color="#c45c26", lw=1.4, ls="--", label=r"MTEF GH3")
    if gh5 is not None:
        ax.plot(t, gh5["d_odd"][:, 0].real, color="#2f9e44", lw=1.0, ls=":", label=r"MTEF GH5")
    ax.plot(t, gh1["d_odd"][:, 0].real, color="#888", lw=0.9, label=r"MTEF GH1")
    ax.set_title(r"$\Delta J_{\mathrm{odd},x}$")
    ax.axhline(0.0, color="#aaa", lw=0.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    ax = axes[0, 1]
    ax.plot(t, qed["dJ_odd"][:, 1].real, color="#1d4e89", lw=1.6, label=r"QED $\Delta J_{\mathrm{odd},y}$")
    ax.plot(t, gh3["d_odd"][:, 1].real, color="#c45c26", lw=1.4, ls="--", label=r"MTEF GH3")
    if gh5 is not None:
        ax.plot(t, gh5["d_odd"][:, 1].real, color="#2f9e44", lw=1.0, ls=":", label=r"MTEF GH5")
    ax.plot(t, gh1["d_odd"][:, 1].real, color="#888", lw=0.9, label=r"MTEF GH1")
    ax.set_title(r"$\Delta J_{\mathrm{odd},y}$")
    ax.axhline(0.0, color="#aaa", lw=0.5)
    ax.grid(True, alpha=0.25)
    ax = axes[1, 0]
    ax.plot(t, (gh3["d_odd"] - qed["dJ_odd"])[:, 0].real, color="#c45c26", lw=1.3, label="x")
    ax.plot(t, (gh3["d_odd"] - qed["dJ_odd"])[:, 1].real, color="#1d4e89", lw=1.3, label="y")
    ax.set_title(r"GH3 $-$ QED")
    ax.axhline(0.0, color="#aaa", lw=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("t")
    ax = axes[1, 1]
    ax.plot(t, gh3["odd_c"][:, 0].real, color="#4c6b8a", lw=1.1, label=r"MTEF $J_{\mathrm{odd},x}^{\mathrm{C}\alpha}$")
    ax.plot(t, gh3["odd_d"][:, 0].real, color="#c45c26", lw=1.1, label=r"MTEF $J_{\mathrm{odd},x}^{\mathrm{DS}}$")
    ax.plot(t, qed["J_odd_coh"][:, 0].real, color="#4c6b8a", lw=0.8, ls=":", label="QED Cα")
    ax.plot(t, qed["J_odd_ds"][:, 0].real, color="#c45c26", lw=0.8, ls=":", label="QED DS")
    ax.set_title(r"$J_{\mathrm{odd},x}$")
    ax.axhline(0.0, color="#aaa", lw=0.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("t")
    fig.suptitle(
        r"shared $k,-k$: QED vs Wigner/MTEF $\Delta J_{\mathrm{odd}}$"
        "\nnot HHG, not advantage, one-cycle rising edge",
        fontsize=10)
    fig.tight_layout()
    svg = out_dir / "qed_mtef_dJ_odd.svg"
    png = out_dir / "qed_mtef_dJ_odd.png"
    fig.savefig(png, dpi=140)
    fig.savefig(svg)
    plt.close(fig)
    return [str(png.as_posix()), str(svg.as_posix())]


def write_table(out_dir: Path, rows):
    path = out_dir / "error_table.tsv"
    keys = list(rows[0].keys())
    lines = ["\t".join(keys)]
    for r in rows:
        lines.append("\t".join(str(r[k]) for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.as_posix())


def remaining(clock):
    return clock.limit - clock.elapsed()


def can_afford(clock, est, reserve=10.0):
    return remaining(clock) > est + reserve


def parity_check(models, grounds, env, n_steps, js, clock, progress):
    """J_{-N}(t; -X,-P) + J_{+N}(t; X,P) should vanish (photon parity × Néel)."""
    mom = mp.analytic_xp("coherent")
    X0, P0 = mom["mean_X"], mom["mean_P"]
    ekp, emkp, _ = grounds[1]
    ekm, emkm, _ = grounds[-1]
    jp, meta, n1, _, _ = one_traj(models[1], ekp, emkp, X0, P0, env, n_steps, js, clock)
    jm, _, n2, _, _ = one_traj(models[-1], ekm, emkm, -X0, -P0, env, n_steps, js, clock)
    s = jp + jm
    _progress(progress, f"parity done max|Jp+Jm|={np.max(np.abs(s)):.3e}")
    return {
        "statement": "J(-N; -X,-P) = -J(+N; X,P) at the coherent mean",
        "max_abs_sum_x": float(np.max(np.abs(s[:, 0]))),
        "max_abs_sum_y": float(np.max(np.abs(s[:, 1]))),
        "max_abs_norm_minus_1": float(max(n1, n2)),
        "t": [m["t"] for m in meta],
    }


def slim_block(block):
    return {
        "order": block["order"], "n_steps": block["n_steps"], "n_traj": block["n_traj"],
        "max_abs": block["max_abs"], "jobs": block["jobs"],
        "series": series_pack(block),
    }


def run(progress=None):
    hashes_before = {str(p.as_posix()): mp.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    qed = load_qed()
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    models, grounds, env = build_bundle(geo, k)
    js = sample_js()
    t_lock = times_of(N_STEPS, js, env.t_ramp)
    if not np.allclose(t_lock, qed["t"]):
        raise RuntimeError("locked MTEF times do not match 20261102")
    clock = dyn.WallClock(WALL_CAP_S)
    report = {
        "record_date": RECORD_DATE,
        "model_id": MODEL_ID,
        "claim_level": "first_QED_vs_MTEF_DeltaJ_odd_not_HHG_not_advantage",
        "odd_current_precision_campaign_closed": True,
        "vacuum_squeeze_branch_closed": True,
        "did_not": [
            "rewrite any 20261026-20261108 json", "G1", "H2", "HHG spectra",
            "vacuum", "theta scan", "Cn", "nmax scan", "output photon statistics",
            "change g/omega0/alpha/r",
        ],
        "definition": {
            "H": "omega0 n + He(k, X; N) + He(-k, X; N)",
            "X": "a + a_dagger", "P": "-i (a - a_dagger)",
            "envelope": "1-cycle sin^2 rising edge, stop at s=1, no plateau",
            "g": pf.G_ABS, "omega0": pf.OMEGA0,
            "alpha": 0.5, "r": mp.R_SQ, "theta": mp.THETA,
            "n_steps": N_STEPS, "sample_every": SAMPLE_EVERY,
            "J_odd": "(J_+N - J_-N)/2",
            "Delta_J_odd": "J_odd_DS - J_odd_Calpha",
            "time_lock": "t=(j+1) dt, j mod 32 = 0 plus endpoint, 42 points",
        },
        "qed_ref": {
            "path": str(ARCHIVE_QED.as_posix()),
            "hash": qed["hash"],
            "max_abs": qed["max_abs"],
            "n_times": int(qed["t"].size),
        },
        "status": "running",
    }
    gh1 = gh3 = gh5 = dt2 = parity = None
    notes = []
    try:
        t_probe = time.perf_counter()
        _progress(progress, "GH1 start")
        gh1 = four_jobs(1, models, grounds, env, N_STEPS, js, clock, progress, "gh1")
        per_ramp = (time.perf_counter() - t_probe) / 4.0
        gh1_pass = (gh1["max_abs"]["dJ_odd_x"] < 1e-12 and gh1["max_abs"]["dJ_odd_y"] < 1e-12)
        notes.append({"gh1_negative_control_pass": gh1_pass, "per_ramp_s": per_ramp})
        _progress(progress, f"GH1 done dJodd_x={gh1['max_abs']['dJ_odd_x']:.3e} pass={gh1_pass}")
        _progress(progress, "GH3 start")
        gh3 = four_jobs(3, models, grounds, env, N_STEPS, js, clock, progress, "gh3")
        cmp3 = vs_qed(gh3, qed)
        _progress(progress, f"GH3 done dQED_x={cmp3['max_abs_MTEF_minus_QED_x']:.3e}")
        if can_afford(clock, 2.2 * per_ramp, reserve=12.0):
            parity = parity_check(models, grounds, env, N_STEPS, js, clock, progress)
        est_dt2 = 36.0 * 2.0 * per_ramp
        if can_afford(clock, est_dt2, reserve=12.0):
            _progress(progress, "GH3 dt/2 start")
            js2 = sample_js_half()
            dt2 = four_jobs(3, models, grounds, env, N_STEPS_HALF, js2, clock, progress, "dt2")
            if not np.allclose(dt2["t"], gh3["t"]):
                raise RuntimeError("dt/2 times not locked")
            ddt = dt2["d_odd"] - gh3["d_odd"]
            dt2["timestep"] = {
                "max_abs_x": float(np.max(np.abs(ddt[:, 0]))),
                "max_abs_y": float(np.max(np.abs(ddt[:, 1]))),
                "below_1e-8": bool(np.max(np.abs(ddt)) < 1e-8),
            }
            _progress(progress, f"GH3 dt/2 done dx={dt2['timestep']['max_abs_x']:.3e}")
        else:
            notes.append({"skipped": "gh3_dt2", "need_s": est_dt2, "left_s": remaining(clock)})
        est_gh5 = 100.0 * per_ramp
        if can_afford(clock, est_gh5, reserve=8.0):
            _progress(progress, "GH5 start")
            gh5 = four_jobs(5, models, grounds, env, N_STEPS, js, clock, progress, "gh5")
            dgh = gh5["d_odd"] - gh3["d_odd"]
            gh5["quad_vs_gh3"] = {
                "max_abs_x": float(np.max(np.abs(dgh[:, 0]))),
                "max_abs_y": float(np.max(np.abs(dgh[:, 1]))),
                "below_1e-8": bool(np.max(np.abs(dgh)) < 1e-8),
            }
            _progress(progress, f"GH5 done dx={gh5['quad_vs_gh3']['max_abs_x']:.3e}")
            est_gh7 = 196.0 * per_ramp
            need_gh7 = not gh5["quad_vs_gh3"]["below_1e-8"]
            if need_gh7 and can_afford(clock, est_gh7, reserve=5.0):
                notes.append({"gh7": "needed_but_will_run"})
            else:
                notes.append({"skipped": "gh7", "need": need_gh7, "left_s": remaining(clock)})
        else:
            notes.append({"skipped": "gh5", "need_s": est_gh5, "left_s": remaining(clock)})
        report["status"] = "ok"
    except TimeoutError as exc:
        report["status"] = "wall_timeout_partial"
        notes.append({"timeout": str(exc), "elapsed": clock.elapsed()})
        _progress(progress, f"TIMEOUT {exc}")

    hashes_after = {str(p.as_posix()): mp.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    report["wall_seconds"] = clock.elapsed()
    report["notes"] = notes
    report["gh1"] = None if gh1 is None else slim_block(gh1)
    report["gh3"] = None if gh3 is None else slim_block(gh3)
    report["gh5"] = None if gh5 is None else slim_block(gh5)
    if dt2 is not None:
        report["gh3_dt2"] = slim_block(dt2)
        report["gh3_dt2"]["timestep"] = dt2.get("timestep")
    report["parity"] = parity
    report["compare_gh3_vs_qed"] = None if gh3 is None else vs_qed(gh3, qed)
    if gh5 is not None:
        report["compare_gh5_vs_qed"] = vs_qed(gh5, qed)
        report["gh5"]["quad_vs_gh3"] = gh5.get("quad_vs_gh3")
    report["old_archive_hashes_before"] = hashes_before
    report["old_archive_hashes_unchanged"] = bool(hashes_before == hashes_after)
    report["stopped"] = True
    if gh1 is not None and gh3 is not None:
        rows = [
            {"qty": "QED_dJ_odd_max", "x": qed["max_abs"]["dJ_odd_x"], "y": qed["max_abs"]["dJ_odd_y"]},
            {"qty": "GH1_dJ_odd_max", "x": gh1["max_abs"]["dJ_odd_x"], "y": gh1["max_abs"]["dJ_odd_y"]},
            {"qty": "GH3_dJ_odd_max", "x": gh3["max_abs"]["dJ_odd_x"], "y": gh3["max_abs"]["dJ_odd_y"]},
        ]
        c3 = report["compare_gh3_vs_qed"]
        rows.append({"qty": "GH3_minus_QED_max", "x": c3["max_abs_MTEF_minus_QED_x"],
                     "y": c3["max_abs_MTEF_minus_QED_y"]})
        rows.append({"qty": "GH3_minus_QED_over_QED", "x": c3["ratio_to_qed_dJ_odd_x"],
                     "y": c3["ratio_to_qed_dJ_odd_y"]})
        if gh5 is not None:
            rows.append({"qty": "GH5_minus_GH3_max", "x": gh5["quad_vs_gh3"]["max_abs_x"],
                         "y": gh5["quad_vs_gh3"]["max_abs_y"]})
        if dt2 is not None and "timestep" in dt2:
            rows.append({"qty": "GH3_dt2_minus_GH3_max", "x": dt2["timestep"]["max_abs_x"],
                         "y": dt2["timestep"]["max_abs_y"]})
        if parity is not None:
            rows.append({"qty": "parity_Jp_plus_Jm", "x": parity["max_abs_sum_x"],
                         "y": parity["max_abs_sum_y"]})
        report["error_table"] = rows
        report["_plot"] = (qed, gh1, gh3, gh5, dt2)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress")
    ap.add_argument("--out", default=str(OUT_DIR / "compare.json"))
    a = ap.parse_args(argv)
    progress = Path(a.progress) if a.progress else OUT_DIR / "progress.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = run(progress)
    plots = []
    table = None
    if "_plot" in report:
        qed, gh1, gh3, gh5, dt2 = report.pop("_plot")
        plots = write_plots(OUT_DIR, qed, gh1, gh3, gh5, dt2)
        if "error_table" in report:
            table = write_table(OUT_DIR, report["error_table"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(mp.jsonable(report), fh, ensure_ascii=False, indent=1)
    _progress(progress, f"wrote {out} status={report['status']}")
    print(f"status={report['status']} wall={report['wall_seconds']:.3f}s out={out}")
    if plots:
        print("plots", plots)
    if table:
        print("table", table)
    c3 = report.get("compare_gh3_vs_qed") or {}
    print("GH3_vs_QED", {k: c3[k] for k in c3 if k.startswith("max") or k.startswith("ratio") or k.startswith("corr")})
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
