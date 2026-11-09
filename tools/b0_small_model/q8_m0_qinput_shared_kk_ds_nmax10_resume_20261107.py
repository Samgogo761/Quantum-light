#!/usr/bin/env python3
"""Resume 20261106 nmax=10 DS±N from checkpoints. Prefix J_odd first, then tail.

Does not restart from t=0. Does not rewrite trunc10.json or older archives.
Extra wall 240 s. Coherent stays nmax=8. Not HHG / nmax=12 / θ / G1 / H2.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_coh_dt_20261105 as coh
import q8_m0_qinput_shared_kk_ds_dt_20261104 as dth
import q8_m0_qinput_shared_kk_ds_nmax10_20261106 as n10
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "M0_shared_mode_k_minus_k_ds_nmax10_resume_v1"
RECORD_DATE = "2026-11-07"
WALL_CAP_S = 240.0
TARGET_ABS = 1e-8
CLEARLY_OVER = 1e-7
DIM = 36 * 36 * (n10.NMAX + 1)
OUT_DIR = ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax10_20261106"
TRUNC10 = OUT_DIR / "trunc10.json"
OLD_ARCHIVES = n10.OLD_ARCHIVES + (TRUNC10,)
JOBS = n10.JOBS


def inspect_one(ckpt_dir: Path, tag: str, expected_next_j: int | None, sample_js, t_all):
    loaded = dth.load_ckpt(ckpt_dir, tag)
    if loaded is None:
        return {"tag": tag, "ok": False, "reason": "missing_ckpt"}
    psi, next_j, samples = loaded
    psi = np.asarray(psi, complex)
    nrm = float(np.linalg.norm(psi))
    n_samp = len(samples)
    t_ck = np.array([s["t"] for s in samples], float) if n_samp else np.zeros(0)
    nmax_s = [int(s.get("quadrature", {}).get("n_max", -1)) for s in samples]
    n_e = [float(s.get("n_e", -1)) for s in samples]
    last_j = int(sample_js[n_samp - 1]) if n_samp else None
    expected_n_prefix = sum(1 for j in sample_js if j < next_j)
    ok_shape = psi.shape == (DIM,)
    ok_finite = bool(np.isfinite(psi).all())
    ok_norm = abs(nrm - 1.0) < 1e-10
    ok_next = expected_next_j is None or next_j == expected_next_j
    ok_count = n_samp == expected_n_prefix and n_samp > 0
    ok_times = bool(n_samp) and np.allclose(t_ck, t_all[:n_samp], atol=1e-12, rtol=0.0)
    ok_nmax = all(v == n10.NMAX for v in nmax_s)
    ok_ne = all(abs(v - 8.0) < 1e-12 for v in n_e)
    ok_order = last_j is None or last_j < next_j
    return {
        "tag": tag,
        "ok": bool(ok_shape and ok_finite and ok_norm and ok_next and ok_count
                   and ok_times and ok_nmax and ok_ne and ok_order),
        "psi_shape": list(psi.shape),
        "psi_dtype": str(psi.dtype),
        "dim_expected": DIM,
        "next_j": int(next_j),
        "expected_next_j": expected_next_j,
        "n_steps": n10.N_STEPS,
        "remaining_steps": n10.N_STEPS - int(next_j),
        "norm": nrm,
        "abs_norm_minus_1": abs(nrm - 1.0),
        "finite": ok_finite,
        "n_samples": n_samp,
        "expected_n_prefix": expected_n_prefix,
        "last_sample_j": last_j,
        "t_first": float(t_ck[0]) if n_samp else None,
        "t_last": float(t_ck[-1]) if n_samp else None,
        "nmax_from_samples": nmax_s[:1],
        "n_e_from_samples": n_e[:1],
        "npz_stores_only_psi_and_next_j": True,
        "model_params_from_frozen_driver": {
            "n_max": n10.NMAX, "n_steps": n10.N_STEPS,
            "theta": ds.THETA, "order": pf.ORDER, "g": pf.G_ABS, "omega0": pf.OMEGA0,
        },
        "flags": {
            "shape": ok_shape, "finite": ok_finite, "norm": ok_norm,
            "next_j": ok_next, "count": ok_count, "times": ok_times,
            "nmax": ok_nmax, "n_e": ok_ne, "order": ok_order,
        },
        "_samples": samples,
    }


def snapshot_ckpt(tag: str, neel: int, next_j: int, psi):
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    model = sh.build_shared(geo, R0, k, neel, n_max=n10.NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    if psi.shape != (model.dim,):
        return {"tag": tag, "ok": False, "reason": "dim_mismatch", "model_dim": model.dim}
    dt = env.t_ramp / n10.N_STEPS
    t = next_j * dt
    snap = pf.snapshot(model, psi, env.s(t))
    return {
        "tag": tag, "ok": True, "model_dim": model.dim, "t_at_next_j": t, "s": env.s(t),
        "norm": snap["norm"], "n_e": snap["n_e"], "n_ph": snap["n_ph"],
        "energy": snap["energy"], "top_fock_mass": snap["top_fock_mass"],
        "n_max_model": model.n_max,
    }


def prefix_compare(ds8, samples_p, samples_m):
    t_p, jp = ds.j_series(samples_p)
    t_m, jm = ds.j_series(samples_m)
    n = min(len(t_p), len(t_m), len(ds8["t"]))
    if not np.allclose(t_p[:n], t_m[:n], atol=1e-12, rtol=0.0):
        raise RuntimeError("prefix ±N times differ")
    if not np.allclose(t_p[:n], ds8["t"][:n], atol=1e-12, rtol=0.0):
        raise RuntimeError("prefix times do not match 20261104")
    even10, odd10 = ds.even_odd(jp[:n], jm[:n])
    odd8 = ds8["J_odd_ds_half"][:n]
    even8 = ds8["J_even_ds_half"][:n]
    d_odd = odd10 - odd8
    abs_x = np.abs(d_odd[:, 0].real)
    abs_y = np.abs(d_odd[:, 1].real)
    ix = int(np.argmax(abs_x))
    iy = int(np.argmax(abs_y))
    max_x = float(abs_x[ix])
    max_y = float(abs_y[iy])
    max_both = max(max_x, max_y)
    return {
        "n_prefix": n,
        "aligned_without_interpolation": True,
        "max_abs_dJ_odd_ds_10_minus_8_x": max_x,
        "max_abs_dJ_odd_ds_10_minus_8_y": max_y,
        "t_at_max_x": float(t_p[ix]),
        "t_at_max_y": float(t_p[iy]),
        "index_at_max_x": ix,
        "index_at_max_y": iy,
        "J_odd_ds10_max_abs_x": float(np.max(np.abs(odd10[:, 0].real))),
        "J_odd_ds10_max_abs_y": float(np.max(np.abs(odd10[:, 1].real))),
        "J_odd_ds8_prefix_max_abs_x": float(np.max(np.abs(odd8[:, 0].real))),
        "J_odd_ds8_prefix_max_abs_y": float(np.max(np.abs(odd8[:, 1].real))),
        "below_1e-8_x": max_x < TARGET_ABS,
        "below_1e-8_y": max_y < TARGET_ABS,
        "clearly_over_1e-7": max_both > CLEARLY_OVER,
        "resume_allowed": max_both <= CLEARLY_OVER,
        "target_abs": TARGET_ABS,
        "clearly_over": CLEARLY_OVER,
        "note": "明显超过取 1e-7（与 ~1e-7 奇通道同量级）；1e-8~1e-7 仍补完尾段",
        "dJ_even_max_abs_x": float(np.max(np.abs((even10 - even8)[:, 0].real))),
        "dJ_even_max_abs_y": float(np.max(np.abs((even10 - even8)[:, 1].real))),
        "imag_J_floor": float(max(np.max(np.abs(jp.imag)), np.max(np.abs(jm.imag)))),
        "abs_dJ_odd_x": [float(x) for x in abs_x],
        "abs_dJ_odd_y": [float(x) for x in abs_y],
    }


def inspect_prefix(ckpt_dir: Path, progress=None):
    n_new, sample_js = dth.sample_steps_half_dt(n10.N_STEPS_OLD, n10.SAMPLE_EVERY_OLD)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    t_all = dth.times_from_steps(n_new, env.t_ramp, sample_js)
    ds8 = coh.load_dt_ds(n10.DT_DS)
    expected = {}
    if TRUNC10.is_file():
        with TRUNC10.open(encoding="utf-8") as fh:
            old = json.load(fh)
        jobs = old.get("jobs", {})
        expected = {tag: jobs.get(tag, {}).get("next_j") for tag, _ in JOBS}
    recs = {}
    for tag, _neel in JOBS:
        recs[tag] = inspect_one(ckpt_dir, tag, expected.get(tag), sample_js, t_all)
        ds._progress(progress, f"inspect {tag} next_j={recs[tag].get('next_j')} "
                     f"nsamp={recs[tag].get('n_samples')} ok={recs[tag].get('ok')}")
    ckpt_ok = all(recs[tag]["ok"] for tag, _ in JOBS)
    snaps = {}
    if ckpt_ok:
        for tag, neel in JOBS:
            psi, next_j, _s = dth.load_ckpt(ckpt_dir, tag)
            snaps[tag] = snapshot_ckpt(tag, neel, next_j, psi)
            ds._progress(progress, f"snapshot {tag} norm={snaps[tag].get('norm')} "
                         f"n_e={snaps[tag].get('n_e')}")
    snap_ok = bool(snaps) and all(s.get("ok") for s in snaps.values())
    cmp = None
    if ckpt_ok:
        npref = min(recs["ds_p"]["n_samples"], recs["ds_m"]["n_samples"])
        cmp = prefix_compare(ds8, recs["ds_p"]["_samples"][:npref], recs["ds_m"]["_samples"][:npref])
        ds._progress(progress, f"prefix max|dJ_odd| x={cmp['max_abs_dJ_odd_ds_10_minus_8_x']:.3e} "
                     f"y={cmp['max_abs_dJ_odd_ds_10_minus_8_y']:.3e} "
                     f"resume_allowed={cmp['resume_allowed']}")
    slim = {tag: {k: v for k, v in recs[tag].items() if k != "_samples"} for tag, _ in JOBS}
    resume_allowed = bool(ckpt_ok and snap_ok and cmp and cmp["resume_allowed"])
    return {
        "record_date": RECORD_DATE,
        "model_id": MODEL_ID,
        "claim_level": "prefix_J_odd_fock_not_HHG_not_discovery",
        "ckpt": slim,
        "snapshot_at_next_j": snaps,
        "prefix_compare": cmp,
        "ckpt_ok": ckpt_ok,
        "snapshot_ok": snap_ok,
        "resume_allowed": resume_allowed,
        "did_not_propagate": True,
        "sample_js": sample_js,
        "n_aligned_full": len(t_all),
        "dt": env.t_ramp / n10.N_STEPS,
        "trunc10_untouched": True,
    }, resume_allowed, recs


def _jxy(sample):
    return [float(sample["mean_J"][a]["re"]) for a in range(2)]


def splice_check(runs, prefix_n, sample_js, t_all):
    out = {}
    for tag, _ in JOBS:
        job = runs[tag]
        samples = job.get("samples") or []
        t = np.array([s["t"] for s in samples], float) if samples else np.zeros(0)
        nrm = np.array([s["norm"] for s in samples], float) if samples else np.zeros(0)
        has_new = len(samples) > prefix_n
        out[tag] = {
            "n_samples": len(samples),
            "expected_n_samples": len(sample_js),
            "times_match_full_grid": bool(len(samples) == len(sample_js)
                                          and np.allclose(t, t_all, atol=1e-12, rtol=0.0)),
            "prefix_n": prefix_n,
            "first_new_index": prefix_n if has_new else None,
            "t_last_prefix": float(t[prefix_n - 1]) if len(samples) >= prefix_n else None,
            "t_first_new": float(t[prefix_n]) if has_new else None,
            "dt_across_splice": (
                float(t[prefix_n] - t[prefix_n - 1]) if has_new and len(samples) >= prefix_n else None),
            "norm_last_prefix": float(nrm[prefix_n - 1]) if len(nrm) >= prefix_n else None,
            "norm_first_new": float(nrm[prefix_n]) if has_new else None,
            "J_last_prefix": _jxy(samples[prefix_n - 1]) if len(samples) >= prefix_n else None,
            "J_first_new": _jxy(samples[prefix_n]) if has_new else None,
            "no_duplicate_times": bool(len(t) == len(np.unique(np.round(t, 12)))),
            "max_abs_norm_minus_1": float(np.max(np.abs(nrm - 1.0))) if len(nrm) else None,
            "status": job.get("status"),
            "next_j": job.get("ledger", {}).get("next_j") or job.get("next_j"),
        }
    return out


def _parallel_job(tag, neel, n_steps, sample_js, progress_s, ckpt_s):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    progress = Path(progress_s) if progress_s else None
    ckpt_dir = Path(ckpt_s)
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    clock = dyn.WallClock(WALL_CAP_S)
    return n10.run_job(geo, k, env, neel, n_steps, sample_js, clock, progress, tag, ckpt_dir)


def run(progress=None, ckpt_dir: Path | None = None, resume: bool = True):
    hashes_before = {str(p.as_posix()): ds.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    if ckpt_dir is None:
        ckpt_dir = OUT_DIR
    prefix, allowed, recs = inspect_prefix(ckpt_dir, progress)
    prefix["old_archive_hashes_before"] = hashes_before
    if not allowed:
        prefix["status"] = "prefix_block_no_resume" if prefix.get("ckpt_ok") else "ckpt_incomplete_no_rerun"
        prefix["stopped"] = True
        prefix["resumable"] = bool(prefix.get("ckpt_ok"))
        return prefix
    if not resume:
        prefix["status"] = "prefix_ok_no_resume_flag"
        prefix["stopped"] = True
        return prefix

    n_new, sample_js = dth.sample_steps_half_dt(n10.N_STEPS_OLD, n10.SAMPLE_EVERY_OLD)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    t_all = dth.times_from_steps(n_new, env.t_ramp, sample_js)
    ds8 = coh.load_dt_ds(n10.DT_DS)
    coh8 = n10.load_coh_dt(n10.COH_DT)
    ds._progress(progress, f"resume wall={WALL_CAP_S}s remaining "
                 f"p={recs['ds_p']['remaining_steps']} m={recs['ds_m']['remaining_steps']}")
    t_par0 = time.perf_counter()
    progress_s = str(progress) if progress is not None else ""
    runs = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {
            ex.submit(_parallel_job, tag, neel, n10.N_STEPS, sample_js, progress_s, str(ckpt_dir)): tag
            for tag, neel in JOBS
        }
        for fut in futs:
            tag = futs[fut]
            runs[tag] = fut.result()
    wall = time.perf_counter() - t_par0
    prefix_n = prefix["prefix_compare"]["n_prefix"]
    report = {
        "record_date": RECORD_DATE,
        "model_id": MODEL_ID,
        "claim_level": "ds_nmax10_fock_resume_not_HHG_not_discovery",
        "vacuum_squeeze_branch_closed": True,
        "did_not": n10.base_report(hashes_before, env, sample_js, n10.photon_static_compare())["did_not"]
        + ["restart from t=0", "overwrite trunc10.json", "nmax=12", "coherent nmax=10"],
        "definition": {
            "n_max": n10.NMAX, "n_steps": n10.N_STEPS, "dt": env.t_ramp / n10.N_STEPS,
            "wall_cap_s": WALL_CAP_S, "resume_from_checkpoint": True,
            "coherent_baseline_nmax8": True,
        },
        "prefix": {k: prefix[k] for k in prefix if k != "old_archive_hashes_before"},
        "wall_seconds_total": wall,
        "old_archive_hashes_before": hashes_before,
        "splice": splice_check(runs, prefix_n, sample_js, t_all),
        "p_top_is_not_convergence": True,
        "not_semiclassical_failure": True,
        "coherent_still_nmax8": True,
        "not_full_delta_J_odd_fock_certification": True,
    }
    if any(job.get("status") != "ok" for job in runs.values()):
        report["jobs"] = {tag: {k: v for k, v in job.items() if k != "samples"} for tag, job in runs.items()}
        report["status"] = "wall_timeout_checkpointed"
        report["resumable"] = True
        report["stopped"] = True
        return report
    slim = {tag: {k: job[k] for k in job if k != "samples"} for tag, job in runs.items()}
    cmp_block, t, odd8, odd10, even8, even10, d_odd, dJ8, dJ10 = n10.compare_nmax(ds8, coh8, runs)
    cmp_block["cursor_stop"] = (
        "DS nmax=8 vs 10 的 J_odd 截断差进入 1e-8；ΔJ_odd 仍混用 coherent nmax=8，"
        "不得写成完整 ΔJ_odd Fock 收敛"
        if (cmp_block["max_abs"]["dJ_odd_ds_10_minus_8_x"] < TARGET_ABS
            and cmp_block["max_abs"]["dJ_odd_ds_10_minus_8_y"] < TARGET_ABS) else
        "DS J_odd 截断差未进入 1e-8；本轮停机，不升 nmax"
    )
    report["jobs"] = slim
    report["compare"] = {k: cmp_block[k] for k in cmp_block if k != "series"}
    report["compare"]["series"] = cmp_block["series"]
    report["numerical_floor"] = {
        "max_last_taylor": {tag: runs[tag]["ledger"]["max_last_taylor_term"] for tag in runs},
        "max_abs_norm_minus_1": {tag: runs[tag]["max_abs_norm_minus_1"] for tag in runs},
        "max_top_fock": {tag: runs[tag]["max_top_fock"] for tag in runs},
        "imag_J_floor": cmp_block["max_abs"]["imag_J_floor_10"],
        "target_abs_delta_J_odd": TARGET_ABS,
    }
    report["status"] = "ok"
    report["stopped"] = True
    report["_plot_arrays"] = (t, odd8, odd10, even8, even10, d_odd, dJ8, dJ10)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=OUT_DIR / "trunc10_resume.json")
    ap.add_argument("--prefix-report", type=Path, default=OUT_DIR / "prefix.json")
    ap.add_argument("--progress", type=Path, default=None)
    ap.add_argument("--ckpt-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--prefix-only", action="store_true")
    args = ap.parse_args(argv)
    if args.report.resolve() == TRUNC10.resolve():
        print("refusing to overwrite trunc10.json")
        return 3
    if (not args.prefix_only) and args.report.exists():
        print(f"report already exists: {args.report}")
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    report = run(progress=args.progress, ckpt_dir=args.ckpt_dir, resume=not args.prefix_only)
    plot_arrays = report.pop("_plot_arrays", None)
    report["wall_seconds_process"] = time.perf_counter() - t0
    hashes_after = {str(p.as_posix()): ds.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    report["old_archive_hashes_unchanged"] = hashes_after == report.get("old_archive_hashes_before", hashes_after)
    prefix_dump = {k: report[k] for k in (
        "ckpt", "snapshot_at_next_j", "prefix_compare", "ckpt_ok", "snapshot_ok",
        "resume_allowed", "status", "did_not_propagate") if k in report}
    if "prefix" in report:
        prefix_dump = report["prefix"]
        prefix_dump["status_after_resume"] = report.get("status")
    args.prefix_report.write_text(json.dumps(ds.jsonable(prefix_dump), ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    blocked = args.prefix_only or report.get("status") in (
        "prefix_block_no_resume", "ckpt_incomplete_no_rerun", "prefix_ok_no_resume_flag")
    if blocked:
        print(json.dumps(ds.jsonable({
            "status": report.get("status"),
            "resume_allowed": report.get("resume_allowed"),
            "prefix_compare": report.get("prefix_compare") or report.get("prefix", {}).get("prefix_compare"),
            "ckpt_ok": report.get("ckpt_ok"),
            "snapshot_ok": report.get("snapshot_ok"),
            "old_untouched": report.get("old_archive_hashes_unchanged"),
            "wrote_prefix": str(args.prefix_report),
            "did_not_write_resume_report": True,
        }), ensure_ascii=False, indent=2))
        return 0 if report.get("resume_allowed") or report.get("status") == "prefix_ok_no_resume_flag" else 6
    if plot_arrays is not None:
        report["figures"] = n10.write_plots(args.report.parent, *plot_arrays)
    with args.report.open("x", encoding="utf-8") as fh:
        json.dump(ds.jsonable(report), fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    summary = {
        "status": report.get("status"),
        "wall": report.get("wall_seconds_total"),
        "max_abs": None if "compare" not in report else report["compare"].get("max_abs"),
        "below_1e-8": None if "compare" not in report else report["compare"].get("below_1e-8_target"),
        "gate": None if "compare" not in report else report["compare"].get("preliminary_truncation_gate"),
        "cursor_stop": None if "compare" not in report else report["compare"].get("cursor_stop"),
        "splice": report.get("splice"),
        "old_untouched": report.get("old_archive_hashes_unchanged"),
        "prefix_max": None if "prefix" not in report else (
            None if not report["prefix"].get("prefix_compare") else {
                "x": report["prefix"]["prefix_compare"]["max_abs_dJ_odd_ds_10_minus_8_x"],
                "y": report["prefix"]["prefix_compare"]["max_abs_dJ_odd_ds_10_minus_8_y"],
            }),
        "coherent_still_nmax8": True,
        "not_full_delta_J_odd_fock_certification": True,
    }
    print(json.dumps(ds.jsonable(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
