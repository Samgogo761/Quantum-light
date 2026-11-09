#!/usr/bin/env python3
"""GH5 quadrature check of frozen shared-mode MTEF vs archived GH3/QED.

Per-job checkpoints. Does not rewrite 20261026-20261109. Not HHG.
GH7 is not in this driver. dt/2 only if wall remains after GH5.
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
import q8_m0_qinput_shared_kk_mtef_compare_20261109 as cmp
import q8_m0_qinput_shared_kk_mtef_map_20261108 as mp
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import build_geometry

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/q_line/qinput_a_shared_kk_mtef_gh5_20261110"
MODEL_ID = "M0_shared_mode_k_minus_k_mtef_gh5_v1"
RECORD_DATE = "2026-11-10"
WALL_CAP_S = 240.0
ORDER = 5
N_NODES = ORDER * ORDER
N_TRAJ = 4 * N_NODES
ARCHIVE_GH3 = ROOT / "results/q_line/qinput_a_shared_kk_mtef_compare_20261109/compare.json"
OLD_ARCHIVES = cmp.OLD_ARCHIVES + (ARCHIVE_GH3,)
JOBS = ds.JOBS


def _progress(progress, msg):
    mp._progress(progress, msg)


def load_gh3():
    with ARCHIVE_GH3.open(encoding="utf-8") as fh:
        o = json.load(fh)
    ser = o["gh3"]["series"]
    return {
        "t": np.asarray(ser["t"], float),
        "odd_c": cmp.unpack_rows(ser["J_odd_coh"]),
        "odd_d": cmp.unpack_rows(ser["J_odd_ds"]),
        "d_odd": cmp.unpack_rows(ser["dJ_odd"]),
        "even_c": cmp.unpack_rows(ser["J_even_coh"]),
        "even_d": cmp.unpack_rows(ser["J_even_ds"]),
        "max_abs": o["gh3"]["max_abs"],
        "hash": mp.sha256_file(ARCHIVE_GH3),
        "n_traj": o["gh3"]["n_traj"],
        "order": o["gh3"]["order"],
    }


def xy_max(a):
    a = np.asarray(a, complex)
    return float(np.max(np.abs(a[:, 0]))), float(np.max(np.abs(a[:, 1])))


def obs_pack(odd_c, odd_d):
    d_odd = odd_d - odd_c
    return {
        "J_odd_coh_x": float(np.max(np.abs(odd_c[:, 0]))),
        "J_odd_coh_y": float(np.max(np.abs(odd_c[:, 1]))),
        "J_odd_ds_x": float(np.max(np.abs(odd_d[:, 0]))),
        "J_odd_ds_y": float(np.max(np.abs(odd_d[:, 1]))),
        "dJ_odd_x": float(np.max(np.abs(d_odd[:, 0]))),
        "dJ_odd_y": float(np.max(np.abs(d_odd[:, 1]))),
    }


def pair_err(a, b, name):
    d = np.asarray(a, complex) - np.asarray(b, complex)
    mx, my = xy_max(d)
    return {
        "qty": name,
        "max_abs_x": mx,
        "max_abs_y": my,
        "rms_x": float(np.sqrt(np.mean(np.abs(d[:, 0]) ** 2))),
        "rms_y": float(np.sqrt(np.mean(np.abs(d[:, 1]) ** 2))),
    }


def amp_ratio(mtef, qed):
    out = {}
    for k in ("J_odd_coh_x", "J_odd_coh_y", "J_odd_ds_x", "J_odd_ds_y", "dJ_odd_x", "dJ_odd_y"):
        den = max(abs(qed[k]), 1e-30)
        out[k] = float(mtef[k] / den)
    return out


def ckpt_path(tag):
    return OUT_DIR / f"ckpt_{tag}.json"


def job_path(tag):
    return OUT_DIR / f"job_{tag}.json"


def save_ckpt(tag, payload):
    p = ckpt_path(tag)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(mp.jsonable(payload), fh, ensure_ascii=False, indent=1)


def load_ckpt(tag):
    p = ckpt_path(tag)
    if not p.is_file():
        return None
    with p.open(encoding="utf-8") as fh:
        d = json.load(fh)
    acc = cmp.unpack_rows(d["acc"]) if d.get("acc") is not None else None
    return {
        "n_done": int(d["n_done"]),
        "acc": acc,
        "wsum": float(d["wsum"]),
        "max_n": float(d["max_n"]),
        "t": np.asarray(d["t"], float) if d.get("t") else None,
    }


def save_job(tag, payload):
    p = job_path(tag)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(mp.jsonable(payload), fh, ensure_ascii=False, indent=1)


def load_job(tag):
    p = job_path(tag)
    if not p.is_file():
        return None
    with p.open(encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("status") != "ok":
        return None
    return {
        "kind": d["kind"], "neel": int(d["neel"]),
        "n_run": int(d["n_run"]), "weight_sum": float(d["weight_sum"]),
        "t": np.asarray(d["t"], float),
        "J": cmp.unpack_rows(d["J"]),
        "max_abs_norm_minus_1": float(d["max_abs_norm_minus_1"]),
        "status": "ok",
    }


def run_one_job(tag, kind, neel, models, grounds, env, js, clock, progress):
    done = load_job(tag)
    if done is not None:
        _progress(progress, f"{tag} already complete n_run={done['n_run']}")
        return done
    nodes = mp.gauss_hermite_nodes(kind, ORDER)["nodes"]
    ek0, emk0, _ = grounds[neel]
    model = models[neel]
    ck = load_ckpt(tag)
    i0, acc, wsum, max_n, t_saved = 0, None, 0.0, 0.0, None
    if ck is not None:
        i0, acc, wsum, max_n, t_saved = ck["n_done"], ck["acc"], ck["wsum"], ck["max_n"], ck["t"]
        _progress(progress, f"{tag} resume n_done={i0}/{len(nodes)}")
    _progress(progress, f"{tag} start kind={kind} neel={neel} nodes={len(nodes)}")
    t_arr = t_saved
    for i, node in enumerate(nodes):
        if i < i0:
            continue
        j_rows, meta, nrm, _, _ = cmp.one_traj(
            model, ek0, emk0, node["X"], node["P"], env, cmp.N_STEPS, js, clock)
        w = float(node["weight"])
        acc = w * j_rows if acc is None else acc + w * j_rows
        wsum += w
        max_n = max(max_n, nrm)
        t_arr = np.array([m["t"] for m in meta], float)
        save_ckpt(tag, {
            "tag": tag, "kind": kind, "neel": neel, "n_done": i + 1,
            "n_nodes": len(nodes), "acc": cmp.pack_j(acc), "wsum": wsum,
            "max_n": max_n, "t": t_arr.tolist(),
        })
        if (i + 1) == len(nodes) or (i + 1) % 5 == 0:
            _progress(progress, f"{tag} node {i + 1}/{len(nodes)} wall={clock.elapsed():.2f}s")
    avg = acc / (wsum + 1e-30)
    out = {
        "tag": tag, "kind": kind, "neel": int(neel), "order": ORDER,
        "n_nodes": len(nodes), "n_run": len(nodes), "weight_sum": wsum,
        "t": t_arr, "J": avg, "max_abs_norm_minus_1": float(max_n),
        "status": "ok",
    }
    save_job(tag, {
        **{k: out[k] for k in out if k != "J"},
        "J": cmp.pack_j(avg), "t": t_arr.tolist(),
    })
    return out


def cost_preflight(models, grounds, env, js, progress):
    node = mp.gauss_hermite_nodes("ds", ORDER)["nodes"][0]
    ek, emk, _ = grounds[1]
    clock = dyn.WallClock(30.0)
    t0 = time.perf_counter()
    cmp.one_traj(models[1], ek, emk, node["X"], node["P"], env, cmp.N_STEPS, js, clock)
    per = time.perf_counter() - t0
    est = per * N_TRAJ
    out = {
        "per_traj_s": per, "n_traj": N_TRAJ, "n_nodes": N_NODES,
        "est_wall_s": est, "wall_cap_s": WALL_CAP_S,
        "under_cap": bool(est < WALL_CAP_S),
    }
    _progress(progress, f"cost per_traj={per:.4f}s est={est:.1f}s cap={WALL_CAP_S} under={out['under_cap']}")
    return out


def assemble(runs):
    t0 = runs["coh_p"]["t"]
    for tag in runs:
        if not np.allclose(runs[tag]["t"], t0):
            raise RuntimeError(f"time grid mismatch {tag}")
    even_c, odd_c = ds.even_odd(runs["coh_p"]["J"], runs["coh_m"]["J"])
    even_d, odd_d = ds.even_odd(runs["ds_p"]["J"], runs["ds_m"]["J"])
    d_odd = odd_d - odd_c
    return {
        "t": t0, "odd_c": odd_c, "odd_d": odd_d, "d_odd": d_odd,
        "even_c": even_c, "even_d": even_d,
        "max_abs": obs_pack(odd_c, odd_d),
        "n_traj": N_TRAJ, "order": ORDER,
        "max_abs_norm_minus_1": float(max(runs[t]["max_abs_norm_minus_1"] for t in runs)),
        "imag_J_floor": float(max(np.max(np.abs(runs[t]["J"].imag)) for t in runs)),
    }


def cancellation_split(odd_c, odd_d, q_c, q_d):
    err_c = odd_c - q_c
    err_d = odd_d - q_d
    common = 0.5 * (err_c + err_d)
    delta_err = err_d - err_c
    cx, cy = xy_max(common)
    dx, dy = xy_max(delta_err)
    ecx, ecy = xy_max(err_c)
    edx, edy = xy_max(err_d)
    return {
        "err_J_odd_coh": {"x": ecx, "y": ecy},
        "err_J_odd_ds": {"x": edx, "y": edy},
        "common_half_sum": {"x": cx, "y": cy},
        "differential_ds_minus_coh": {"x": dx, "y": dy},
        "common_over_coh_err_x": float(cx / max(ecx, 1e-30)),
        "common_over_coh_err_y": float(cy / max(ecy, 1e-30)),
        "diff_over_coh_err_x": float(dx / max(ecx, 1e-30)),
        "diff_over_coh_err_y": float(dy / max(ecy, 1e-30)),
        "note": "common=0.5(err_coh+err_ds); differential=err_ds-err_coh=Delta_MTEF-Delta_QED",
    }


def quad_flags(d_gh, d_qed, floor=1e-8, frac=0.2):
    out = {}
    for axis in ("x", "y"):
        g = abs(d_gh[f"max_abs_{axis}"])
        q = abs(d_qed[f"max_abs_{axis}"])
        out[axis] = {
            "gh5_minus_gh3": g,
            "gh3_minus_qed": q,
            "below_1e-8": bool(g < floor),
            "below_20pct_of_qed_residual": bool(g < frac * max(q, floor)),
            "not_converged": bool(g >= floor and g >= frac * max(q, floor)),
        }
    out["any_not_converged"] = bool(out["x"]["not_converged"] or out["y"]["not_converged"])
    return out


def write_plots(out_dir, qed, gh3, gh5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = qed["t"]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.8))
    specs = (
        (axes[0, 0], qed["J_odd_coh"], gh3["odd_c"], gh5["odd_c"] if gh5 else None, 0,
         r"$J_{\mathrm{odd},x}^{\mathrm{C}\alpha}$"),
        (axes[0, 1], qed["J_odd_ds"], gh3["odd_d"], gh5["odd_d"] if gh5 else None, 0,
         r"$J_{\mathrm{odd},x}^{\mathrm{DS}}$"),
        (axes[1, 0], qed["dJ_odd"], gh3["d_odd"], gh5["d_odd"] if gh5 else None, 0,
         r"$\Delta J_{\mathrm{odd},x}$"),
        (axes[1, 1], qed["dJ_odd"], gh3["d_odd"], gh5["d_odd"] if gh5 else None, 1,
         r"$\Delta J_{\mathrm{odd},y}$"),
    )
    for ax, q, g3, g5, col, title in specs:
        ax.plot(t, q[:, col].real, color="#1d4e89", lw=1.6, label="QED")
        ax.plot(t, g3[:, col].real, color="#c45c26", lw=1.3, ls="--", label="GH3")
        if g5 is not None:
            ax.plot(t, g5[:, col].real, color="#2f9e44", lw=1.1, ls=":", label="GH5")
        ax.set_title(title)
        ax.axhline(0.0, color="#aaa", lw=0.5)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("GH5 vs GH3 vs QED  (not HHG, frozen shared-mode)", fontsize=10)
    fig.tight_layout()
    png = out_dir / "gh5_j_odd.png"
    svg = out_dir / "gh5_j_odd.svg"
    fig.savefig(png, dpi=140)
    fig.savefig(svg)
    plt.close(fig)
    return [str(png.as_posix()), str(svg.as_posix())]


def write_table(out_dir, rows):
    path = out_dir / "error_table.tsv"
    keys = list(rows[0].keys())
    lines = ["\t".join(keys)]
    for r in rows:
        lines.append("\t".join(str(r[k]) for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.as_posix())


def run(progress=None):
    hashes_before = {str(p.as_posix()): mp.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    qed = cmp.load_qed()
    gh3 = load_gh3()
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    models, grounds, env = cmp.build_bundle(geo, k)
    js = cmp.sample_js()
    t_lock = cmp.times_of(cmp.N_STEPS, js, env.t_ramp)
    if not np.allclose(t_lock, qed["t"]) or not np.allclose(t_lock, gh3["t"]):
        raise RuntimeError("GH5 times not locked to 20261102/20261109")
    report = {
        "record_date": RECORD_DATE,
        "model_id": MODEL_ID,
        "claim_level": "gh5_quadrature_check_not_HHG_not_advantage",
        "odd_current_precision_campaign_closed": True,
        "vacuum_squeeze_branch_closed": True,
        "did_not": [
            "rewrite any 20261026-20261109 json", "G1", "H2", "HHG spectra",
            "vacuum", "theta scan", "Cn", "nmax scan", "output photon statistics",
            "change g/omega0/alpha/r", "GH7", "new QED propagation",
        ],
        "definition": {
            "H": "omega0 n + He(k, X; N) + He(-k, X; N)",
            "X": "a + a_dagger", "P": "-i (a - a_dagger)",
            "g": pf.G_ABS, "omega0": pf.OMEGA0, "alpha": 0.5, "r": mp.R_SQ, "theta": mp.THETA,
            "n_steps": cmp.N_STEPS, "sample_every": cmp.SAMPLE_EVERY, "gh_order": ORDER,
            "J_odd": "(J_+N - J_-N)/2", "Delta_J_odd": "J_odd_DS - J_odd_Calpha",
            "time_lock": "t=(j+1) dt, j mod 32 = 0 plus endpoint, 42 points",
            "common_nodes": "same (xi,eta) for coherent and DS, affine map by Var(X),Var(P)",
        },
        "qed_ref": {"path": str(cmp.ARCHIVE_QED.as_posix()), "hash": qed["hash"],
                    "max_abs": obs_pack(qed["J_odd_coh"], qed["J_odd_ds"])},
        "gh3_ref": {"path": str(ARCHIVE_GH3.as_posix()), "hash": gh3["hash"],
                    "max_abs": obs_pack(gh3["odd_c"], gh3["odd_d"])},
        "status": "running",
    }
    notes = []
    gh5 = None
    dt2 = None
    _progress(progress, "cost preflight start")
    cost = cost_preflight(models, grounds, env, js, progress)
    report["cost_preflight"] = cost
    if not cost["under_cap"]:
        report["status"] = "cost_gate_stop"
        notes.append({"stopped": "est_wall exceeds cap", "est_s": cost["est_wall_s"]})
        report["notes"] = notes
        report["wall_seconds"] = cost["per_traj_s"]
        report["gh5"] = None
        report["stopped"] = True
        hashes_after = {str(p.as_posix()): mp.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
        report["old_archive_hashes_before"] = hashes_before
        report["old_archive_hashes_unchanged"] = bool(hashes_before == hashes_after)
        return report

    clock = dyn.WallClock(WALL_CAP_S)
    runs = {}
    try:
        for tag, kind, neel in JOBS:
            runs[tag] = run_one_job(tag, kind, neel, models, grounds, env, js, clock, progress)
        gh5 = assemble(runs)
        report["status"] = "ok"
        _progress(progress, f"GH5 assembled dJodd_x={gh5['max_abs']['dJ_odd_x']:.3e}")
        left = clock.limit - clock.elapsed()
        est_dt2 = 36.0 * 2.0 * cost["per_traj_s"]
        if left > est_dt2 + 15.0:
            _progress(progress, f"GH3 dt/2 start left={left:.1f}s est={est_dt2:.1f}s")
            try:
                js2 = cmp.sample_js_half()
                dt2 = cmp.four_jobs(3, models, grounds, env, cmp.N_STEPS_HALF, js2, clock, progress, "dt2")
                ddt = dt2["d_odd"] - gh3["d_odd"]
                djc = dt2["odd_c"] - gh3["odd_c"]
                djd = dt2["odd_d"] - gh3["odd_d"]
                dt2["timestep"] = {
                    "dJ_odd": {"x": float(np.max(np.abs(ddt[:, 0]))), "y": float(np.max(np.abs(ddt[:, 1])))},
                    "J_odd_coh": {"x": float(np.max(np.abs(djc[:, 0]))), "y": float(np.max(np.abs(djc[:, 1])))},
                    "J_odd_ds": {"x": float(np.max(np.abs(djd[:, 0]))), "y": float(np.max(np.abs(djd[:, 1])))},
                }
                _progress(progress, f"dt2 dJodd_x={dt2['timestep']['dJ_odd']['x']:.3e}")
            except TimeoutError as exc:
                notes.append({"skipped": "gh3_dt2_timeout", "elapsed": clock.elapsed(), "exc": str(exc)})
                dt2 = None
                _progress(progress, "GH3 dt/2 timeout; GH5 kept")
        else:
            notes.append({"skipped": "gh3_dt2", "need_s": est_dt2, "left_s": left})
    except TimeoutError as exc:
        report["status"] = "wall_timeout_partial"
        notes.append({"timeout": str(exc), "elapsed": clock.elapsed(),
                      "jobs_ok": [t for t in runs if runs[t].get("status") == "ok"]})
        _progress(progress, f"TIMEOUT {exc}")
        if all(t in runs for t in ("coh_p", "coh_m", "ds_p", "ds_m")):
            gh5 = assemble(runs)

    hashes_after = {str(p.as_posix()): mp.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    report["wall_seconds"] = clock.elapsed()
    report["notes"] = notes
    report["jobs_present"] = {t: bool(load_job(t) is not None or t in runs) for t, _, _ in JOBS}

    q_abs = obs_pack(qed["J_odd_coh"], qed["J_odd_ds"])
    g3_abs = obs_pack(gh3["odd_c"], gh3["odd_d"])
    report["max_abs_qed"] = q_abs
    report["max_abs_gh3"] = g3_abs
    report["ratio_gh3_over_qed"] = amp_ratio(g3_abs, q_abs)

    if gh5 is not None:
        g5_abs = gh5["max_abs"]
        report["max_abs_gh5"] = g5_abs
        report["ratio_gh5_over_qed"] = amp_ratio(g5_abs, q_abs)
        e53_c = pair_err(gh5["odd_c"], gh3["odd_c"], "GH5_minus_GH3_J_odd_coh")
        e53_d = pair_err(gh5["odd_d"], gh3["odd_d"], "GH5_minus_GH3_J_odd_ds")
        e53_dj = pair_err(gh5["d_odd"], gh3["d_odd"], "GH5_minus_GH3_dJ_odd")
        e3q_c = pair_err(gh3["odd_c"], qed["J_odd_coh"], "GH3_minus_QED_J_odd_coh")
        e3q_d = pair_err(gh3["odd_d"], qed["J_odd_ds"], "GH3_minus_QED_J_odd_ds")
        e3q_dj = pair_err(gh3["d_odd"], qed["dJ_odd"], "GH3_minus_QED_dJ_odd")
        e5q_c = pair_err(gh5["odd_c"], qed["J_odd_coh"], "GH5_minus_QED_J_odd_coh")
        e5q_d = pair_err(gh5["odd_d"], qed["J_odd_ds"], "GH5_minus_QED_J_odd_ds")
        e5q_dj = pair_err(gh5["d_odd"], qed["dJ_odd"], "GH5_minus_QED_dJ_odd")
        report["gh5_minus_gh3"] = {"J_odd_coh": e53_c, "J_odd_ds": e53_d, "dJ_odd": e53_dj}
        report["gh3_minus_qed"] = {"J_odd_coh": e3q_c, "J_odd_ds": e3q_d, "dJ_odd": e3q_dj}
        report["gh5_minus_qed"] = {"J_odd_coh": e5q_c, "J_odd_ds": e5q_d, "dJ_odd": e5q_dj}
        report["quad_flags"] = {
            "J_odd_coh": quad_flags(e53_c, e3q_c),
            "J_odd_ds": quad_flags(e53_d, e3q_d),
            "dJ_odd": quad_flags(e53_dj, e3q_dj),
        }
        report["cancellation_gh3"] = cancellation_split(
            gh3["odd_c"], gh3["odd_d"], qed["J_odd_coh"], qed["J_odd_ds"])
        report["cancellation_gh5"] = cancellation_split(
            gh5["odd_c"], gh5["odd_d"], qed["J_odd_coh"], qed["J_odd_ds"])
        report["gh5_meta"] = {
            "order": ORDER, "n_traj": N_TRAJ,
            "max_abs_norm_minus_1": gh5["max_abs_norm_minus_1"],
            "imag_J_floor": gh5["imag_J_floor"],
            "series": {
                "t": gh5["t"].tolist(),
                "J_odd_coh": cmp.pack_j(gh5["odd_c"]),
                "J_odd_ds": cmp.pack_j(gh5["odd_d"]),
                "dJ_odd": cmp.pack_j(gh5["d_odd"]),
            },
        }
        any_nc = any(report["quad_flags"][k]["any_not_converged"] for k in report["quad_flags"])
        report["gh3_quadrature_converged"] = (not any_nc)
        report["verdict_gate"] = (
            "quadrature_not_converged_do_not_interpret_QED_MTEF"
            if any_nc else
            "quadrature_stable_GH3_to_GH5_residual_is_method_not_node_count"
        )
        report["error_table"] = [
            {"qty": "QED_J_odd_coh", "x": q_abs["J_odd_coh_x"], "y": q_abs["J_odd_coh_y"]},
            {"qty": "QED_J_odd_ds", "x": q_abs["J_odd_ds_x"], "y": q_abs["J_odd_ds_y"]},
            {"qty": "QED_dJ_odd", "x": q_abs["dJ_odd_x"], "y": q_abs["dJ_odd_y"]},
            {"qty": "GH3_J_odd_coh", "x": g3_abs["J_odd_coh_x"], "y": g3_abs["J_odd_coh_y"]},
            {"qty": "GH3_J_odd_ds", "x": g3_abs["J_odd_ds_x"], "y": g3_abs["J_odd_ds_y"]},
            {"qty": "GH3_dJ_odd", "x": g3_abs["dJ_odd_x"], "y": g3_abs["dJ_odd_y"]},
            {"qty": "GH5_J_odd_coh", "x": g5_abs["J_odd_coh_x"], "y": g5_abs["J_odd_coh_y"]},
            {"qty": "GH5_J_odd_ds", "x": g5_abs["J_odd_ds_x"], "y": g5_abs["J_odd_ds_y"]},
            {"qty": "GH5_dJ_odd", "x": g5_abs["dJ_odd_x"], "y": g5_abs["dJ_odd_y"]},
            {"qty": "GH5_minus_GH3_J_odd_coh", "x": e53_c["max_abs_x"], "y": e53_c["max_abs_y"]},
            {"qty": "GH5_minus_GH3_J_odd_ds", "x": e53_d["max_abs_x"], "y": e53_d["max_abs_y"]},
            {"qty": "GH5_minus_GH3_dJ_odd", "x": e53_dj["max_abs_x"], "y": e53_dj["max_abs_y"]},
            {"qty": "GH3_minus_QED_J_odd_coh", "x": e3q_c["max_abs_x"], "y": e3q_c["max_abs_y"]},
            {"qty": "GH3_minus_QED_J_odd_ds", "x": e3q_d["max_abs_x"], "y": e3q_d["max_abs_y"]},
            {"qty": "GH3_minus_QED_dJ_odd", "x": e3q_dj["max_abs_x"], "y": e3q_dj["max_abs_y"]},
            {"qty": "GH5_minus_QED_J_odd_coh", "x": e5q_c["max_abs_x"], "y": e5q_c["max_abs_y"]},
            {"qty": "GH5_minus_QED_J_odd_ds", "x": e5q_d["max_abs_x"], "y": e5q_d["max_abs_y"]},
            {"qty": "GH5_minus_QED_dJ_odd", "x": e5q_dj["max_abs_x"], "y": e5q_dj["max_abs_y"]},
            {"qty": "ratio_GH5_QED_J_odd_coh", "x": report["ratio_gh5_over_qed"]["J_odd_coh_x"],
             "y": report["ratio_gh5_over_qed"]["J_odd_coh_y"]},
            {"qty": "ratio_GH5_QED_J_odd_ds", "x": report["ratio_gh5_over_qed"]["J_odd_ds_x"],
             "y": report["ratio_gh5_over_qed"]["J_odd_ds_y"]},
            {"qty": "ratio_GH5_QED_dJ_odd", "x": report["ratio_gh5_over_qed"]["dJ_odd_x"],
             "y": report["ratio_gh5_over_qed"]["dJ_odd_y"]},
            {"qty": "common_err_GH5", "x": report["cancellation_gh5"]["common_half_sum"]["x"],
             "y": report["cancellation_gh5"]["common_half_sum"]["y"]},
            {"qty": "differential_err_GH5", "x": report["cancellation_gh5"]["differential_ds_minus_coh"]["x"],
             "y": report["cancellation_gh5"]["differential_ds_minus_coh"]["y"]},
        ]
        report["_plot"] = (qed, gh3, gh5)

    if dt2 is not None and "timestep" in dt2:
        report["gh3_dt2"] = dt2["timestep"]
        report["error_table"].append({
            "qty": "GH3_dt2_minus_GH3_dJ_odd",
            "x": dt2["timestep"]["dJ_odd"]["x"], "y": dt2["timestep"]["dJ_odd"]["y"],
        })
        report["error_table"].append({
            "qty": "GH3_dt2_minus_GH3_J_odd_coh",
            "x": dt2["timestep"]["J_odd_coh"]["x"], "y": dt2["timestep"]["J_odd_coh"]["y"],
        })

    report["old_archive_hashes_before"] = hashes_before
    report["old_archive_hashes_unchanged"] = bool(hashes_before == hashes_after)
    report["stopped"] = True
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress")
    ap.add_argument("--out", default=str(OUT_DIR / "gh5.json"))
    a = ap.parse_args(argv)
    progress = Path(a.progress) if a.progress else OUT_DIR / "progress.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = run(progress)
    plots = []
    table = None
    if "_plot" in report:
        qed, gh3, gh5 = report.pop("_plot")
        plots = write_plots(OUT_DIR, qed, gh3, gh5)
        if "error_table" in report:
            table = write_table(OUT_DIR, report["error_table"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(mp.jsonable(report), fh, ensure_ascii=False, indent=1)
    _progress(progress, f"wrote {out} status={report['status']}")
    print(f"status={report['status']} wall={report.get('wall_seconds', 0):.3f}s out={out}")
    if plots:
        print("plots", plots)
    if table:
        print("table", table)
    print("verdict", report.get("verdict_gate"), "gh3_quad", report.get("gh3_quadrature_converged"))
    return 0 if report["status"] in ("ok", "cost_gate_stop") else 2


if __name__ == "__main__":
    raise SystemExit(main())
