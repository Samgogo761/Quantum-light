#!/usr/bin/env python3
"""nmax=8 coherent α=0.5 ±N, dt halved vs 20261102.

Completes ΔJ_odd = J_odd^DS − J_odd^Cα time-step check by pairing this
run with the already-accepted 20261104 DS dt/2 series.

Not HHG, not Fock-truncation certification, not nmax=12, not squeeze-angle.
Does not rewrite old archives. Wall cap 360 s; over-estimate stops first.
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
import q8_m0_qinput_shared_kk_ds_dt_20261104 as dt
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "M0_shared_mode_k_minus_k_coh_dt_half_v1"
RECORD_DATE = "2026-11-05"
WALL_CAP_S = 360.0
NMAX = 8
N_STEPS_OLD = 1285
N_STEPS = 2570
SAMPLE_EVERY_OLD = 32
CHECKPOINT_EVERY = 400
TARGET_ABS = 1e-8
JOBS = (("coh_p", 1), ("coh_m", -1))
ARCHIVE_8 = ROOT / "results/q_line/qinput_a_shared_kk_ds_odd_20261102/compare.json"
TRUNC_12 = ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax12_20261103/trunc.json"
PERF_12 = ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax12_20261103/perf.json"
DT_DS = ROOT / "results/q_line/qinput_a_shared_kk_ds_dt_20261104/dt.json"
OLD_ARCHIVES = ds.OLD_ARCHIVES + (ARCHIVE_8, TRUNC_12, PERF_12, DT_DS)


def load_compare8(path: Path):
    with path.open(encoding="utf-8") as fh:
        o = json.load(fh)
    ser = o["compare"]["series"]
    t = np.asarray(ser["t"], float)
    return {
        "t": t,
        "J_odd_ds": dt.j_from_rows(ser["J_odd_ds"]),
        "J_odd_coh": dt.j_from_rows(ser["J_odd_coh"]),
        "J_even_ds": dt.j_from_rows(ser["J_even_ds"]),
        "J_even_coh": dt.j_from_rows(ser["J_even_coh"]),
        "dJ_odd": dt.j_from_rows(ser["dJ_odd"]),
        "dt": o["cost"]["dt"],
        "n_steps": o["cost"]["n_steps"],
        "sample_every": o["definition"]["sample_every"],
        "max_abs": o["compare"]["max_abs"],
    }


def load_dt_ds(path: Path):
    with path.open(encoding="utf-8") as fh:
        o = json.load(fh)
    ser = o["compare"]["series"]
    return {
        "t": np.asarray(ser["t"], float),
        "J_odd_ds_half": dt.j_from_rows(ser["J_odd_ds_half"]),
        "J_even_ds_half": dt.j_from_rows(ser["J_even_ds_half"]),
        "status": o["status"],
        "n_steps": o["definition"]["n_steps"],
        "dt": o["definition"]["dt"],
        "max_abs": o["compare"]["max_abs"],
    }


def rel_amp(num, den):
    return float(np.max(np.abs(num)) / max(float(np.max(np.abs(den))), 1e-300))


def run_job(geo, k, env, neel, n_steps, sample_js, clock, progress, tag, ckpt_dir: Path):
    ek, emk, masses = ds.ground_pair_neel(geo, k, neel)
    raw = ds.photon_raw("coherent", NMAX)
    raw_diag = ds.photon_moments(raw)
    c = raw / np.sqrt(raw_diag["unnormalized_mass"])
    model = sh.build_shared(geo, R0, k, neel, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    psi, prod_n = ds.product_normalized(ek, emk, c)
    start_j = 0
    samples = []
    loaded = dt.load_ckpt(ckpt_dir, tag)
    if loaded is not None:
        psi, start_j, samples = loaded
        ds._progress(progress, f"{tag} resume next_j={start_j}")
    s0 = pf.snapshot(model, psi, env.s(start_j * (env.t_ramp / n_steps)))
    ds._progress(progress, f"{tag} start neel={neel} dim={model.dim} steps={n_steps} j0={start_j}")
    psi, led = dt.propagate_ckpt(
        model, psi, 0.0, env.t_ramp, env, n_steps, sample_js, clock,
        ckpt_dir, tag, start_j=start_j, samples=samples)
    if led["timed_out"]:
        ds._progress(progress, f"{tag} timeout next_j={led['next_j']} wall={clock.elapsed():.1f}s")
        return {
            "tag": tag, "kind": "coherent", "neel": int(neel),
            "status": "wall_timeout_checkpointed",
            "next_j": led["next_j"], "n_steps": n_steps,
            "ckpt": str(dt.ckpt_paths(ckpt_dir, tag)[0].as_posix()),
            "n_samples_so_far": len(led["samples"]),
            "wall_seconds": clock.elapsed(),
            "resumable": True,
        }
    send = pf.snapshot(model, psi, 1.0)
    packed = pf.pack_samples(led["samples"])
    top = [s["top_fock_mass"] for s in led["samples"]]
    norms = [s["norm"] for s in led["samples"]]
    dt.save_ckpt(ckpt_dir, tag, psi, n_steps, led["samples"])
    out = {
        "tag": tag, "kind": "coherent", "neel": int(neel), "status": "ok", "n_max": NMAX,
        "electron": masses,
        "photon_unnormalized": raw_diag,
        "photon_used_normalized": True,
        "product_norm_before_renorm": prod_n,
        "initial": s0, "after_ramp": send,
        "delta_energy": send["energy"] - s0["energy"],
        "work_trap_dHds_dsdt": led["work_trap_dHds_dsdt"],
        "energy_minus_work": (send["energy"] - s0["energy"]) - led["work_trap_dHds_dsdt"],
        "ledger": {k: led[k] for k in ("steps", "dt", "order", "max_last_taylor_term", "next_j")},
        "max_top_fock": float(np.max(top) if top else send["top_fock_mass"]),
        "max_abs_norm_minus_1": float(np.max(np.abs(np.asarray(norms) - 1.0))) if norms else abs(send["norm"] - 1.0),
        "time_series": packed,
        "samples": led["samples"],
        "wall_seconds": clock.elapsed(),
        "resumable": False,
        "top_two_alarm_0p05_is_not_convergence": True,
    }
    ds._progress(progress, f"{tag} done wall={clock.elapsed():.1f}s dE={out['delta_energy']:.3e}")
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
    return run_job(geo, k, env, neel, n_steps, sample_js, clock, progress, tag, ckpt_dir)


def compare_delta(arch, ds_half, runs):
    t_p, jp = ds.j_series(runs["coh_p"]["samples"])
    t_m, jm = ds.j_series(runs["coh_m"]["samples"])
    if not np.allclose(t_p, t_m, atol=1e-12, rtol=0.0):
        raise RuntimeError("half-dt coherent ±N time grids differ")
    if not np.allclose(t_p, arch["t"], atol=1e-12, rtol=0.0):
        raise RuntimeError("half-dt times do not match 20261102 compare.json")
    if not np.allclose(t_p, ds_half["t"], atol=1e-12, rtol=0.0):
        raise RuntimeError("half-dt times do not match 20261104 dt.json")
    even_h, odd_h = ds.even_odd(jp, jm)
    d_odd_old = arch["J_odd_ds"] - arch["J_odd_coh"]
    if not np.allclose(d_odd_old, arch["dJ_odd"], atol=1e-18, rtol=0.0):
        raise RuntimeError("archived dJ_odd is not J_odd_ds - J_odd_coh")
    d_odd_half = ds_half["J_odd_ds_half"] - odd_h
    d_coh = odd_h - arch["J_odd_coh"]
    d_even = even_h - arch["J_even_coh"]
    d_delta = d_odd_half - d_odd_old
    max_abs = {
        "J_odd_coh_half_x": float(np.max(np.abs(odd_h[:, 0]))),
        "J_odd_coh_half_y": float(np.max(np.abs(odd_h[:, 1]))),
        "J_odd_coh_old_x": float(np.max(np.abs(arch["J_odd_coh"][:, 0]))),
        "J_odd_coh_old_y": float(np.max(np.abs(arch["J_odd_coh"][:, 1]))),
        "dJ_odd_coh_half_minus_old_x": float(np.max(np.abs(d_coh[:, 0]))),
        "dJ_odd_coh_half_minus_old_y": float(np.max(np.abs(d_coh[:, 1]))),
        "dJ_even_coh_half_minus_old_x": float(np.max(np.abs(d_even[:, 0]))),
        "dJ_even_coh_half_minus_old_y": float(np.max(np.abs(d_even[:, 1]))),
        "dJ_odd_old_x": float(np.max(np.abs(d_odd_old[:, 0]))),
        "dJ_odd_old_y": float(np.max(np.abs(d_odd_old[:, 1]))),
        "dJ_odd_half_x": float(np.max(np.abs(d_odd_half[:, 0]))),
        "dJ_odd_half_y": float(np.max(np.abs(d_odd_half[:, 1]))),
        "d_delta_J_odd_half_minus_old_x": float(np.max(np.abs(d_delta[:, 0]))),
        "d_delta_J_odd_half_minus_old_y": float(np.max(np.abs(d_delta[:, 1]))),
        "imag_J_floor": float(max(np.max(np.abs(jp.imag)), np.max(np.abs(jm.imag)))),
    }
    scale = {
        "dJ_odd_half_over_J_odd_ds_half_x": rel_amp(d_odd_half[:, 0], ds_half["J_odd_ds_half"][:, 0]),
        "dJ_odd_half_over_J_odd_ds_half_y": rel_amp(d_odd_half[:, 1], ds_half["J_odd_ds_half"][:, 1]),
        "dJ_odd_half_over_J_odd_coh_half_x": rel_amp(d_odd_half[:, 0], odd_h[:, 0]),
        "dJ_odd_half_over_J_odd_coh_half_y": rel_amp(d_odd_half[:, 1], odd_h[:, 1]),
        "timestep_dJ_odd_over_dJ_odd_old_x": rel_amp(d_delta[:, 0], d_odd_old[:, 0]),
        "timestep_dJ_odd_over_dJ_odd_old_y": rel_amp(d_delta[:, 1], d_odd_old[:, 1]),
        "timestep_J_odd_coh_over_J_odd_coh_old_x": rel_amp(d_coh[:, 0], arch["J_odd_coh"][:, 0]),
        "timestep_J_odd_coh_over_J_odd_coh_old_y": rel_amp(d_coh[:, 1], arch["J_odd_coh"][:, 1]),
    }
    below = {
        "dJ_odd_coh_x_lt_1e-8": max_abs["dJ_odd_coh_half_minus_old_x"] < TARGET_ABS,
        "dJ_odd_coh_y_lt_1e-8": max_abs["dJ_odd_coh_half_minus_old_y"] < TARGET_ABS,
        "d_delta_J_odd_x_lt_1e-8": max_abs["d_delta_J_odd_half_minus_old_x"] < TARGET_ABS,
        "d_delta_J_odd_y_lt_1e-8": max_abs["d_delta_J_odd_half_minus_old_y"] < TARGET_ABS,
        "target_abs": TARGET_ABS,
        "time_discretization_of_current_fock_model_only": True,
        "fock_truncation_unverified": True,
        "not_a_physics_pass": True,
        "no_quantum_advantage_claim": True,
    }
    return {
        "aligned_without_interpolation": True,
        "delta_definition": "J_odd_DS - J_odd_coh; half uses 20261104 DS series, not a new DS run",
        "max_abs": max_abs,
        "relative_amplitude": scale,
        "below_1e-8_target": below,
        "series": {
            "t": t_p.tolist(),
            "J_odd_coh_half": [[pf.cpack(z) for z in row] for row in odd_h],
            "J_even_coh_half": [[pf.cpack(z) for z in row] for row in even_h],
            "dJ_odd_coh_half_minus_old": [[pf.cpack(z) for z in row] for row in d_coh],
            "dJ_odd_half": [[pf.cpack(z) for z in row] for row in d_odd_half],
            "dJ_odd_old": [[pf.cpack(z) for z in row] for row in d_odd_old],
            "d_delta_J_odd_half_minus_old": [[pf.cpack(z) for z in row] for row in d_delta],
        },
        "no_quantum_advantage_claim": True,
        "nmax12_not_run": True,
        "hhg_not_computed": True,
    }, t_p, arch["J_odd_coh"], odd_h, d_odd_old, d_odd_half, d_coh, d_delta


def write_plots(out_dir, t, odd_old, odd_h, d_odd_old, d_odd_half, d_coh, d_delta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)
    rows = (
        (axes[0, 0], odd_old[:, 0].real, odd_h[:, 0].real, d_coh[:, 0].real,
         r"$J_{\mathrm{odd},x}^{\mathrm{C}\alpha}$"),
        (axes[0, 1], odd_old[:, 1].real, odd_h[:, 1].real, d_coh[:, 1].real,
         r"$J_{\mathrm{odd},y}^{\mathrm{C}\alpha}$"),
        (axes[1, 0], d_odd_old[:, 0].real, d_odd_half[:, 0].real, d_delta[:, 0].real,
         r"$\Delta J_{\mathrm{odd},x}$"),
        (axes[1, 1], d_odd_old[:, 1].real, d_odd_half[:, 1].real, d_delta[:, 1].real,
         r"$\Delta J_{\mathrm{odd},y}$"),
    )
    for ax, a, b, dlt, title in rows:
        ax.plot(t, a, color="#4c6b8a", lw=1.4, label=r"dt")
        ax.plot(t, b, color="#c45c26", lw=1.4, label=r"dt/2")
        ax.plot(t, dlt, color="#2f9e44", lw=1.1, ls="--", label=r"half-old")
        ax.set_title(title)
        ax.axhline(0.0, color="#888", lw=0.6)
        ax.grid(True, alpha=0.25)
    axes[0, 1].legend(loc="best", fontsize=8)
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")
    fig.suptitle(
        r"C$\alpha$ $\alpha$=0.5 $\pm$N time-step and $\Delta J_{\mathrm{odd}}$"
        "\nnot HHG, not Fock certification, not quantum advantage",
        fontsize=10)
    fig.tight_layout()
    png = out_dir / "j_coh_dt.svg"
    fig.savefig(png)
    fig.savefig(out_dir / "j_coh_dt.png", dpi=140)
    plt.close(fig)
    return [str(png.as_posix())]


def base_report(hashes_before, arch, env, sample_js):
    dt_h = env.t_ramp / N_STEPS
    return {
        "record_date": RECORD_DATE,
        "model_id": MODEL_ID,
        "claim_level": "coh_dt_half_delta_J_odd_time_step_not_fock_not_HHG",
        "vacuum_squeeze_branch_closed": True,
        "did_not": [
            "rewrite trial.json", "rewrite 20261027 preflight.json",
            "rewrite 20261028 diag.json", "rewrite static.json",
            "rewrite 20261030 preflight.json", "rewrite 20261031 diag.json",
            "rewrite 20261101 switch.json", "rewrite 20261102 compare.json",
            "rewrite 20261103 trunc.json", "rewrite 20261103 perf.json",
            "rewrite 20261104 dt.json",
            "G1", "H2", "HHG spectra", "theta=pi/2", "theta=pi", "Cn",
            "nmax=12 rerun", "DS dt rerun", "Wigner trajectories",
            "full BZ", "output photon statistics", "change Hamiltonian",
        ],
        "definition": {
            "H": "omega0 n + He(k, X; N) + He(-k, X; N)",
            "envelope": "1-cycle sin^2 rising edge, stop at s=1, no plateau",
            "n_max": NMAX, "g": pf.G_ABS, "omega0": pf.OMEGA0,
            "alpha": 0.5, "kind": "coherent",
            "jobs": [list(j) for j in JOBS],
            "k_fractional": list(sh.K_FRAC),
            "n_steps_old": N_STEPS_OLD,
            "n_steps": N_STEPS,
            "dt_old": arch["dt"],
            "dt": dt_h,
            "t_end": env.t_ramp,
            "sample_js": sample_js,
            "sample_rule": "j_new=2*j_old+1 so t matches archive; not naive every-64",
            "parallel_jobs": 2,
            "checkpoint_every": CHECKPOINT_EVERY,
            "archive8": str(ARCHIVE_8.as_posix()),
            "dt_ds_archive": str(DT_DS.as_posix()),
            "delta_uses_20261104_DS_half": True,
        },
        "old_archive_hashes_before": hashes_before,
        "fock_truncation_unverified": True,
        "time_discretization_of_current_fock_model_only": True,
        "decision_point_after_this_run": True,
    }


def run(progress=None, ckpt_dir: Path | None = None):
    hashes_before = {str(p.as_posix()): ds.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    if not ARCHIVE_8.is_file():
        return {"status": "missing_nmax8_archive", "stopped": True,
                "old_archive_hashes_before": hashes_before}
    if not DT_DS.is_file():
        return {"status": "missing_ds_dt_archive", "stopped": True,
                "old_archive_hashes_before": hashes_before}
    arch = load_compare8(ARCHIVE_8)
    ds_half = load_dt_ds(DT_DS)
    if ds_half["status"] != "ok":
        return {"status": "ds_dt_archive_not_ok", "stopped": True,
                "ds_dt_status": ds_half["status"]}
    if arch["n_steps"] != N_STEPS_OLD or arch["sample_every"] != SAMPLE_EVERY_OLD:
        return {"status": "archive_step_mismatch", "stopped": True,
                "n_steps": arch["n_steps"], "sample_every": arch["sample_every"]}
    if ds_half["n_steps"] != N_STEPS:
        return {"status": "ds_half_step_mismatch", "stopped": True}
    n_new, sample_js = dt.sample_steps_half_dt(arch["n_steps"], arch["sample_every"])
    if n_new != N_STEPS:
        return {"status": "n_steps_lock_failed", "stopped": True}
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    dt_h = env.t_ramp / N_STEPS
    t_align = dt.times_from_steps(N_STEPS, env.t_ramp, sample_js)
    if not np.allclose(t_align, arch["t"], atol=1e-12, rtol=0.0):
        return {"status": "time_alignment_failed_no_propagation", "stopped": True,
                "t_align0": float(t_align[0]), "t_arch0": float(arch["t"][0])}
    if not np.allclose(t_align, ds_half["t"], atol=1e-12, rtol=0.0):
        return {"status": "ds_half_time_alignment_failed_no_propagation", "stopped": True}
    if ckpt_dir is None:
        ckpt_dir = ROOT / "results/q_line/qinput_a_shared_kk_coh_dt_20261105"
    report = base_report(hashes_before, arch, env, sample_js)
    cost = dt.measure_cost(dt_h)
    cost.update({"n_steps": N_STEPS, "dt": dt_h, "dim": 36 * 36 * (NMAX + 1),
                 "kind": "coherent", "ds_dt_half_wall_s": 183.58})
    report["cost"] = cost
    ds._progress(progress, f"cost est_wall={cost['estimated_wall_parallel_seconds']:.1f}s "
                 f"par_step={cost['parallel_taylor_per_step']:.4f}s steps={N_STEPS} dt={dt_h}")
    if cost["exceeds_6min"]:
        report["status"] = "cost_gate_failed_no_propagation"
        report["stopped"] = True
        return report

    t_par0 = time.perf_counter()
    progress_s = str(progress) if progress is not None else ""
    runs = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {
            ex.submit(_parallel_job, tag, neel, N_STEPS, sample_js, progress_s, str(ckpt_dir)): tag
            for tag, neel in JOBS
        }
        for fut in futs:
            tag = futs[fut]
            runs[tag] = fut.result()
    clock_elapsed = time.perf_counter() - t_par0
    report["wall_seconds_total"] = clock_elapsed
    if any(job.get("status") != "ok" for job in runs.values()):
        report["jobs"] = {tag: {k: v for k, v in job.items() if k != "samples"} for tag, job in runs.items()}
        report["status"] = "wall_timeout_checkpointed"
        report["resumable"] = True
        report["stopped"] = True
        return report
    slim = {tag: {k: job[k] for k in job if k != "samples"} for tag, job in runs.items()}
    cmp_block, t, odd_old, odd_h, d_odd_old, d_odd_half, d_coh, d_delta = compare_delta(arch, ds_half, runs)
    report["jobs"] = slim
    report["compare"] = {k: cmp_block[k] for k in cmp_block if k != "series"}
    report["compare"]["series"] = cmp_block["series"]
    report["numerical_floor"] = {
        "max_last_taylor": {tag: runs[tag]["ledger"]["max_last_taylor_term"] for tag in runs},
        "max_abs_norm_minus_1": {tag: runs[tag]["max_abs_norm_minus_1"] for tag in runs},
        "max_top_fock": {tag: runs[tag]["max_top_fock"] for tag in runs},
        "imag_J_floor": cmp_block["max_abs"]["imag_J_floor"],
        "target_abs_J_odd": TARGET_ABS,
        "fock_truncation_unverified": True,
    }
    report["status"] = "ok"
    report["stopped"] = True
    report["_plot_arrays"] = (t, odd_old, odd_h, d_odd_old, d_odd_half, d_coh, d_delta)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path,
                    default=ROOT / "results/q_line/qinput_a_shared_kk_coh_dt_20261105/coh_dt.json")
    ap.add_argument("--progress", type=Path, default=None)
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--cost-only", action="store_true")
    args = ap.parse_args(argv)
    ckpt_dir = args.ckpt_dir or args.report.parent
    if args.report.exists():
        print(f"report already exists: {args.report}")
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.cost_only:
        arch = load_compare8(ARCHIVE_8)
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        dt_h = env.t_ramp / N_STEPS
        cost = dt.measure_cost(dt_h)
        cost.update({"n_steps": N_STEPS, "dt": dt_h, "n_steps_old": arch["n_steps"]})
        payload = {"status": "cost_only", "cost": cost, "stopped": True,
                   "exceeds_6min": cost["exceeds_6min"]}
        with args.report.open("x", encoding="utf-8") as fh:
            json.dump(ds.jsonable(payload), fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(json.dumps(ds.jsonable(payload), ensure_ascii=False, indent=2))
        return 1 if cost["exceeds_6min"] else 0
    t0 = time.perf_counter()
    report = run(progress=args.progress, ckpt_dir=ckpt_dir)
    plot_arrays = report.pop("_plot_arrays", None)
    report["wall_seconds_process"] = time.perf_counter() - t0
    if plot_arrays is not None:
        report["figures"] = write_plots(args.report.parent, *plot_arrays)
    hashes_after = {str(p.as_posix()): ds.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    report["old_archive_hashes_unchanged"] = hashes_after == report.get("old_archive_hashes_before", hashes_after)
    with args.report.open("x", encoding="utf-8") as fh:
        json.dump(ds.jsonable(report), fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    summary = {
        "status": report.get("status"),
        "wall": report.get("wall_seconds_total"),
        "max_abs": None if "compare" not in report else report["compare"].get("max_abs"),
        "relative_amplitude": None if "compare" not in report else report["compare"].get("relative_amplitude"),
        "below_1e-8": None if "compare" not in report else report["compare"].get("below_1e-8_target"),
        "old_untouched": report.get("old_archive_hashes_unchanged"),
        "cost": report.get("cost"),
        "resumable": report.get("resumable", False),
    }
    print(json.dumps(ds.jsonable(summary), ensure_ascii=False, indent=2))
    if report.get("status") == "cost_gate_failed_no_propagation":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
