#!/usr/bin/env python3
"""Low-cost nmax=12 performance probe. No full propagation. Cap 120 s."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax12_20261103"
TRUNC = OUT_DIR / "trunc.json"
NMAX = 12
N_MINI = 32
DT_ARCH = 0.046013806716804
N_STEPS_ARCH = 2731
SAMPLE_EVERY = 32
WALL_DIAG_S = 120.0


def _mean(xs):
    xs = list(xs)
    return float(sum(xs) / max(len(xs), 1))


def _timed(n, fn):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return _mean(ts), float(min(ts)), float(max(ts))


def _build():
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    model = sh.build_shared(geo, R0, k, 1, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    rng = np.random.default_rng(20261103)
    v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    v = v / np.linalg.norm(v)
    return model, v


def _mini_loop(model, v, dt, n_steps, sample_every):
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    psi = v.copy()
    n_snap = 0
    t_taylor = 0.0
    t_sample = 0.0
    for j in range(n_steps):
        tm = (j + 0.5) * dt
        s = env.s(tm)
        Ds = dyn.peierls_at(model.q_units, s, model.n_max)
        t0 = time.perf_counter()
        psi, _ = pf.taylor_step(model, psi, s, dt, Ds=Ds)
        t_taylor += time.perf_counter() - t0
        take = (j % sample_every == 0) or (j == n_steps - 1)
        if take:
            t1 = time.perf_counter()
            pf.snapshot(model, psi, env.s((j + 1) * dt), Ds=None)
            pf.dHds_expect(model, psi, env.s((j + 1) * dt))
            t_sample += time.perf_counter() - t1
            n_snap += 1
    return {
        "n_steps": n_steps,
        "n_snap": n_snap,
        "taylor_seconds": t_taylor,
        "sample_seconds": t_sample,
        "taylor_per_step": t_taylor / n_steps,
        "sample_per_snap": t_sample / max(n_snap, 1),
        "loop_seconds": t_taylor + t_sample,
    }


def _worker_mini(n_steps, dt):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    model, v = _build()
    t0 = time.perf_counter()
    rec = _mini_loop(model, v, dt, n_steps, SAMPLE_EVERY)
    rec["wall_including_build"] = time.perf_counter() - t0
    rec["pid"] = os.getpid()
    return rec


def checkpoint_audit():
    names = sorted(p.name for p in OUT_DIR.iterdir()) if OUT_DIR.is_dir() else []
    npy = [n for n in names if n.endswith((".npy", ".npz"))]
    progress = (OUT_DIR / "progress.txt").read_text(encoding="utf-8") if (OUT_DIR / "progress.txt").is_file() else ""
    trunc = {}
    if TRUNC.is_file():
        trunc = json.loads(TRUNC.read_text(encoding="utf-8"))
    return {
        "folder_files": names,
        "npy_npz": npy,
        "progress_txt": progress.strip().splitlines(),
        "progress_has_step_index": any("step=" in ln for ln in progress.splitlines()),
        "trunc_has_jobs": "jobs" in trunc,
        "trunc_status": trunc.get("status"),
        "driver_saves_psi": False,
        "resumable": False,
        "reason": "no psi/npy checkpoint; progress only has start lines; timeout raised before jobs were packed",
    }


def blas_probe(model, v, Ds):
    info = {"env_OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
        info["threadpool"] = [
            {"prefix": x.get("prefix"), "num_threads": x.get("num_threads"),
             "user_api": x.get("user_api")}
            for x in threadpool_info()
        ]
        times = {}
        for nthr in (1, 4):
            with threadpool_limits(nthr):
                mean, lo, hi = _timed(6, lambda: sh.apply_h(model, v, 0.4, Ds=Ds))
            times[str(nthr)] = {"mean": mean, "min": lo, "max": hi}
        info["apply_h_by_threads"] = times
        info["threads_help"] = bool(times["4"]["mean"] < 0.9 * times["1"]["mean"])
    except Exception as exc:
        info["threadpoolctl"] = str(exc)
    return info


def run():
    t_all = time.perf_counter()
    report = {
        "record_date": "2026-11-03",
        "model_id": "M0_shared_mode_k_minus_k_ds_nmax12_perf_v1",
        "claim_level": "performance_probe_not_truncation_not_HHG",
        "did_not": [
            "full DS ±N nmax=12", "extend wall", "rewrite trunc.json",
            "rewrite 20261102 compare.json", "G1", "H2", "vacuum", "theta scan",
            "change Hamiltonian", "lower Taylor order",
        ],
        "checkpoint": checkpoint_audit(),
        "budget_s": WALL_DIAG_S,
    }
    t_build0 = time.perf_counter()
    model, v = _build()
    t_build = time.perf_counter() - t_build0
    Ds = dyn.peierls_at(model.q_units, 0.4, model.n_max)
    sh.apply_h(model, v, 0.4, Ds=Ds)
    pf.taylor_step(model, v, 0.4, DT_ARCH, Ds=Ds)

    t_apply, a_lo, a_hi = _timed(8, lambda: sh.apply_h(model, v, 0.4, Ds=Ds))
    t_tay, y_lo, y_hi = _timed(6, lambda: pf.taylor_step(model, v, 0.4, DT_ARCH, Ds=Ds))
    t_peierls, _, _ = _timed(8, lambda: dyn.peierls_at(model.q_units, 0.4, model.n_max))
    t_snap, _, _ = _timed(2, lambda: pf.snapshot(model, v, 0.4, Ds=None))
    t_dhds, _, _ = _timed(2, lambda: pf.dHds_expect(model, v, 0.4))

    serial = _mini_loop(model, v, DT_ARCH, N_MINI, SAMPLE_EVERY)
    t_par0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(_worker_mini, N_MINI, DT_ARCH) for _ in range(2)]
        par_jobs = [f.result() for f in futs]
    t_par = time.perf_counter() - t_par0

    n_samp = 1 + (N_STEPS_ARCH - 1) // SAMPLE_EVERY
    step_s = serial["taylor_per_step"]
    step_p = max(j["taylor_per_step"] for j in par_jobs)
    snap_s = t_snap + t_dhds
    est = {
        "serial_one_job_s": step_s * N_STEPS_ARCH + snap_s * n_samp,
        "parallel_two_jobs_wall_s": step_p * N_STEPS_ARCH + snap_s * n_samp,
        "sequential_two_jobs_s": 2.0 * (step_s * N_STEPS_ARCH + snap_s * n_samp),
        "n_samples_full": n_samp,
        "from_failed_run_implied_step_s": 600.0 / N_STEPS_ARCH,
        "original_estimate_s": 356.97804968826824,
        "original_timed_taylor_s": 0.10054870002204552,
    }
    report["probe"] = {
        "dim": model.dim,
        "n_group": int(model.q_units.size),
        "taylor_order": pf.ORDER,
        "build_seconds": t_build,
        "apply_h_s": {"mean": t_apply, "min": a_lo, "max": a_hi},
        "peierls_s": t_peierls,
        "taylor_step_s": {"mean": t_tay, "min": y_lo, "max": y_hi,
                          "apply_h_times_order": t_apply * pf.ORDER},
        "snapshot_s": t_snap,
        "dHds_expect_s": t_dhds,
        "sample_bundle_s": snap_s,
        "serial_mini32": serial,
        "parallel_mini32_wall_s": t_par,
        "parallel_mini32_jobs": par_jobs,
        "contention_step_ratio": step_p / max(step_s, 1e-12),
        "blas": blas_probe(model, v, Ds),
    }
    report["why_357_vs_600"] = {
        "parent_timed_one_coldish_taylor_only": True,
        "did_not_time_snapshot_or_dHds": True,
        "did_not_time_two_process_contention": True,
        "fudge_1p3_from_nmax8_fast_wave": True,
        "nmax12_n_steps_2p12x_nmax8": True,
        "measured_serial_taylor_s": step_s,
        "measured_parallel_taylor_s": step_p,
        "sampling_not_the_bottleneck": bool(snap_s * n_samp < 0.15 * est["serial_one_job_s"]),
    }
    report["reestimate"] = est
    report["dt_error_design"] = {
        "problem": "nmax=8 used dt≈0.0978; nmax=12 used dt≈0.0460. A finished 12-vs-8 J_odd mixs Fock and time-discretization.",
        "do_not": "attribute all 12-minus-8 current difference to Fock cutoff",
        "independent_dt_check": {
            "object": "DS θ=0 ±N, freeze nmax",
            "cheap_first": "nmax=8, n_steps=2570 (half the archived dt), compare J_odd to 20261102 archive on the same t grid",
            "not_run": True,
            "why_nmax8": "signal to certify is already at nmax=8; dim=11664; avoids mixing Fock into the dt test",
            "nmax12_same_dt_as_8": "only if Taylor last-term at dt=0.0978 stays acceptable; not a silent precision drop; not run",
        },
    }
    report["economical_finish"] = {
        "cannot_resume": True,
        "do_not_start_now": True,
        "options_not_executed": [
            {
                "id": "serial_two_jobs",
                "what": "one DS job at a time, nmax=12, same dt=0.046",
                "est_s": est["sequential_two_jobs_s"],
                "note": "avoids 2-process contention if ratio>1",
            },
            {
                "id": "parallel_if_contention_small",
                "what": "two DS jobs together only if contention_step_ratio≲1.15",
                "est_s": est["parallel_two_jobs_wall_s"],
            },
            {
                "id": "checkpoint_slices",
                "what": "save psi every ~400 steps; 10 min slices; same Hamiltonian",
                "est_s_per_slice": 400 * step_s,
            },
            {
                "id": "dt_first_on_nmax8",
                "what": "before any nmax=12 rerun, half-dt DS ±N at nmax=8",
                "est_s_parallel": "≈ nmax=8 job wall × 2 steps, two jobs parallel",
            },
        ],
        "must_not": "raise wall and rerun the same 2-job nmax=12 setup unchanged",
    }
    report["wall_seconds"] = time.perf_counter() - t_all
    report["stopped"] = True
    report["status"] = "ok" if report["wall_seconds"] <= WALL_DIAG_S else "diag_over_2min_but_no_full_run"
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=OUT_DIR / "perf.json")
    args = ap.parse_args(argv)
    if args.report.exists():
        print(f"report already exists: {args.report}")
        return 2
    report = run()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as fh:
        json.dump(ds.jsonable(report), fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    summary = {
        "status": report["status"],
        "resumable": report["checkpoint"]["resumable"],
        "wall": report["wall_seconds"],
        "reestimate": report["reestimate"],
        "contention_step_ratio": report["probe"]["contention_step_ratio"],
        "why": report["why_357_vs_600"],
    }
    print(json.dumps(ds.jsonable(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
