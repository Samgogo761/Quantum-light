#!/usr/bin/env python3
"""DS(θ=0) ±N nmax=10 Fock truncation vs archived nmax=8 dt/2.

Same locked dt as 20261104/20261105. Coherent baseline stays nmax=8.
Not HHG, not θ/Cn/Wigner, not nmax=12 restart. Wall cap 360 s.
Does not rewrite old archives. Vacuum / G1 / H2 stay frozen.
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
import q8_m0_qinput_shared_kk_ds_nmax12_20261103 as tr12
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "M0_shared_mode_k_minus_k_ds_nmax10_trunc_v1"
RECORD_DATE = "2026-11-06"
WALL_CAP_S = 360.0
NMAX = 10
N_STEPS_OLD = 1285
N_STEPS = 2570
SAMPLE_EVERY_OLD = 32
CHECKPOINT_EVERY = 400
N_MINI = 24
SUSTAINED = 1.15
TARGET_ABS = 1e-8
JOBS = (("ds_p", 1), ("ds_m", -1))
ARCHIVE_8 = ROOT / "results/q_line/qinput_a_shared_kk_ds_odd_20261102/compare.json"
TRUNC_12 = ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax12_20261103/trunc.json"
PERF_12 = ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax12_20261103/perf.json"
DT_DS = ROOT / "results/q_line/qinput_a_shared_kk_ds_dt_20261104/dt.json"
COH_DT = ROOT / "results/q_line/qinput_a_shared_kk_coh_dt_20261105/coh_dt.json"
OLD_ARCHIVES = ds.OLD_ARCHIVES + (ARCHIVE_8, TRUNC_12, PERF_12, DT_DS, COH_DT)


def photon_static_compare():
    recs = {n: tr12.photon_input_record(n) for n in (8, 10, 12)}

    def row(r):
        u = r["unnormalized"]
        return {
            "n_max": r["n_max"],
            "mean_n": u["mean_n"],
            "mean_a_re": u["mean_a"]["re"],
            "var_n": u["var_n"],
            "mean_aa_re": u["mean_aa"]["re"],
            "omitted_unnormalized": u["omitted_unnormalized"],
            "p_boundary": u["p_n"][-1],
            "top_two_mass": u["top_two_mass"],
            "large80_P_n_gt_nmax": r["large80_P_n_gt_nmax"],
            "var_X": r["X_peierls"]["var_X"],
        }

    return {
        "nmax8": recs[8],
        "nmax10": recs[10],
        "nmax12": recs[12],
        "table": [row(recs[k]) for k in (8, 10, 12)],
        "p_boundary_is_not_convergence": True,
        "not_a_D_S_convention_recheck": True,
    }


def load_coh_dt(path: Path):
    with path.open(encoding="utf-8") as fh:
        o = json.load(fh)
    ser = o["compare"]["series"]
    return {
        "t": np.asarray(ser["t"], float),
        "J_odd_coh_half": dth.j_from_rows(ser["J_odd_coh_half"]),
        "dJ_odd_half": dth.j_from_rows(ser["dJ_odd_half"]),
        "status": o["status"],
        "n_steps": o["definition"]["n_steps"],
        "dt": o["definition"]["dt"],
        "max_abs": o["compare"]["max_abs"],
    }


def _mini_loop(model, v, dt, n_steps):
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    psi = v.copy()
    worst = 0.0
    t0 = time.perf_counter()
    for j in range(n_steps):
        s = env.s((j + 0.5) * dt)
        Ds = dyn.peierls_at(model.q_units, s, model.n_max)
        psi, last = pf.taylor_step(model, psi, s, dt, Ds=Ds)
        worst = max(worst, last)
    return time.perf_counter() - t0, worst, float(np.linalg.norm(psi))


def _worker_mini(n_steps, dt):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    model = sh.build_shared(geo, R0, k, 1, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    rng = np.random.default_rng(20261106)
    v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    v = v / np.linalg.norm(v)
    loop_s, last, nrm = _mini_loop(model, v, dt, n_steps)
    return {
        "loop_s": loop_s,
        "taylor_per_step": loop_s / n_steps,
        "max_last_taylor": last,
        "norm_end": nrm,
        "pid": os.getpid(),
    }


def measure_cost(dt: float):
    t_serial0 = time.perf_counter()
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    model = sh.build_shared(geo, R0, k, 1, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    rng = np.random.default_rng(20261106)
    v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    v = v / np.linalg.norm(v)
    rho, _ = pf.spectral_radius_power(model, 1.0, rng)
    n_spec = dyn.n_steps_for(env.t_ramp, rho, requested=32, hdt_max=pf.TAYLOR_HDT_MAX)
    _, last_probe = pf.taylor_step(model, v, 0.4, dt)
    serial_loop, serial_last, serial_nrm = _mini_loop(model, v, dt, N_MINI)
    serial_step = serial_loop / N_MINI
    serial_probe_s = time.perf_counter() - t_serial0
    t_par0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(_worker_mini, N_MINI, dt) for _ in range(2)]
        recs = [f.result() for f in futs]
    par_wall = time.perf_counter() - t_par0
    par_step = max(r["taylor_per_step"] for r in recs)
    one = par_step * N_STEPS * SUSTAINED
    finite = bool(np.isfinite(last_probe) and np.isfinite(serial_last) and np.isfinite(serial_nrm))
    return {
        "n_mini": N_MINI,
        "dim": model.dim,
        "order": pf.ORDER,
        "locked_n_steps": N_STEPS,
        "locked_dt": dt,
        "spectral_radius": rho,
        "rho_dt": rho * dt,
        "taylor_hdt_max": pf.TAYLOR_HDT_MAX,
        "n_steps_if_spectral": n_spec,
        "dt_kept_equal_to_nmax8_half": True,
        "timed_taylor_last_term": last_probe,
        "serial_mini_max_last_taylor": serial_last,
        "serial_mini_norm_end": serial_nrm,
        "taylor_finite": finite,
        "serial_taylor_per_step": serial_step,
        "parallel_taylor_per_step": par_step,
        "parallel_mini_wall_s": par_wall,
        "serial_probe_s": serial_probe_s,
        "contention_step_ratio": par_step / max(serial_step, 1e-30),
        "sustained_factor": SUSTAINED,
        "estimated_one_job_seconds": one,
        "estimated_wall_parallel_seconds": one,
        "estimated_serial_two_jobs_seconds": 2.0 * serial_step * N_STEPS * SUSTAINED,
        "nmax8_ds_dt_half_wall_s": 183.58,
        "wall_cap_s": WALL_CAP_S,
        "exceeds_6min": bool(one > WALL_CAP_S),
        "parallel_jobs": recs,
        "note": "locked dt; loop timing not isolated Taylor; x1.15 sustained",
    }


def run_job(geo, k, env, neel, n_steps, sample_js, clock, progress, tag, ckpt_dir: Path):
    ek, emk, masses = ds.ground_pair_neel(geo, k, neel)
    raw = ds.photon_raw("ds", NMAX, ds.THETA)
    raw_diag = ds.photon_moments(raw)
    c = raw / np.sqrt(raw_diag["unnormalized_mass"])
    model = sh.build_shared(geo, R0, k, neel, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    psi, prod_n = ds.product_normalized(ek, emk, c)
    start_j = 0
    samples = []
    loaded = dth.load_ckpt(ckpt_dir, tag)
    if loaded is not None:
        psi, start_j, samples = loaded
        ds._progress(progress, f"{tag} resume next_j={start_j}")
    s0 = pf.snapshot(model, psi, env.s(start_j * (env.t_ramp / n_steps)))
    ds._progress(progress, f"{tag} start neel={neel} dim={model.dim} steps={n_steps} j0={start_j}")
    psi, led = dth.propagate_ckpt(
        model, psi, 0.0, env.t_ramp, env, n_steps, sample_js, clock,
        ckpt_dir, tag, start_j=start_j, samples=samples)
    if led["timed_out"]:
        ds._progress(progress, f"{tag} timeout next_j={led['next_j']} wall={clock.elapsed():.1f}s")
        return {
            "tag": tag, "kind": "ds", "neel": int(neel),
            "status": "wall_timeout_checkpointed",
            "next_j": led["next_j"], "n_steps": n_steps,
            "ckpt": str(dth.ckpt_paths(ckpt_dir, tag)[0].as_posix()),
            "n_samples_so_far": len(led["samples"]),
            "wall_seconds": clock.elapsed(),
            "resumable": True,
        }
    send = pf.snapshot(model, psi, 1.0)
    packed = pf.pack_samples(led["samples"])
    top = [s["top_fock_mass"] for s in led["samples"]]
    norms = [s["norm"] for s in led["samples"]]
    dth.save_ckpt(ckpt_dir, tag, psi, n_steps, led["samples"])
    out = {
        "tag": tag, "kind": "ds", "neel": int(neel), "status": "ok", "n_max": NMAX,
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


def compare_nmax(ds8, coh8, runs):
    t_p, jp = ds.j_series(runs["ds_p"]["samples"])
    t_m, jm = ds.j_series(runs["ds_m"]["samples"])
    if not np.allclose(t_p, t_m, atol=1e-12, rtol=0.0):
        raise RuntimeError("nmax=10 ±N time grids differ")
    if not np.allclose(t_p, ds8["t"], atol=1e-12, rtol=0.0):
        raise RuntimeError("nmax=10 times do not match 20261104 dt.json")
    if not np.allclose(t_p, coh8["t"], atol=1e-12, rtol=0.0):
        raise RuntimeError("nmax=10 times do not match 20261105 coh_dt.json")
    even10, odd10 = ds.even_odd(jp, jm)
    odd8 = ds8["J_odd_ds_half"]
    even8 = ds8["J_even_ds_half"]
    d_odd_ds = odd10 - odd8
    d_even_ds = even10 - even8
    dJ_odd_10 = odd10 - coh8["J_odd_coh_half"]
    dJ_odd_8 = coh8["dJ_odd_half"]
    d_delta = dJ_odd_10 - dJ_odd_8
    max_abs = {
        "J_odd_ds10_x": float(np.max(np.abs(odd10[:, 0]))),
        "J_odd_ds10_y": float(np.max(np.abs(odd10[:, 1]))),
        "J_odd_ds8_x": float(np.max(np.abs(odd8[:, 0]))),
        "J_odd_ds8_y": float(np.max(np.abs(odd8[:, 1]))),
        "dJ_odd_10_vs_coh8_x": float(np.max(np.abs(dJ_odd_10[:, 0]))),
        "dJ_odd_10_vs_coh8_y": float(np.max(np.abs(dJ_odd_10[:, 1]))),
        "dJ_odd_8_vs_coh8_x": float(np.max(np.abs(dJ_odd_8[:, 0]))),
        "dJ_odd_8_vs_coh8_y": float(np.max(np.abs(dJ_odd_8[:, 1]))),
        "dJ_odd_ds_10_minus_8_x": float(np.max(np.abs(d_odd_ds[:, 0]))),
        "dJ_odd_ds_10_minus_8_y": float(np.max(np.abs(d_odd_ds[:, 1]))),
        "delta_of_dJ_odd_x": float(np.max(np.abs(d_delta[:, 0]))),
        "delta_of_dJ_odd_y": float(np.max(np.abs(d_delta[:, 1]))),
        "dJ_even_ds_10_minus_8_x": float(np.max(np.abs(d_even_ds[:, 0]))),
        "dJ_even_ds_10_minus_8_y": float(np.max(np.abs(d_even_ds[:, 1]))),
        "imag_J_floor_10": float(max(np.max(np.abs(jp.imag)), np.max(np.abs(jm.imag)))),
    }
    below = {
        "delta_of_dJ_odd_x_lt_1e-8": max_abs["delta_of_dJ_odd_x"] < TARGET_ABS,
        "delta_of_dJ_odd_y_lt_1e-8": max_abs["delta_of_dJ_odd_y"] < TARGET_ABS,
        "dJ_odd_ds_10_minus_8_x_lt_1e-8": max_abs["dJ_odd_ds_10_minus_8_x"] < TARGET_ABS,
        "dJ_odd_ds_10_minus_8_y_lt_1e-8": max_abs["dJ_odd_ds_10_minus_8_y"] < TARGET_ABS,
        "target_abs": TARGET_ABS,
        "p_top_is_not_this_gate": True,
        "coherent_still_nmax8": True,
        "not_a_physics_pass": True,
        "not_semiclassical_failure": True,
    }
    gate_ok = bool(below["delta_of_dJ_odd_x_lt_1e-8"] and below["delta_of_dJ_odd_y_lt_1e-8"])
    return {
        "aligned_without_interpolation": True,
        "delta_definition": "J_odd_DS(nmax) - J_odd_Cα(nmax=8 dt/2)",
        "max_abs": max_abs,
        "below_1e-8_target": below,
        "preliminary_truncation_gate": gate_ok,
        "nmax12_not_started": True,
        "cursor_stop": (
            "初步截断门通过；是否还要 nmax=12 最终确认留给外审，本轮停机"
            if gate_ok else
            "ΔJ_odd 绝对截断变化未进入 1e-8；不追加 nmax=12，先停机"
        ),
        "series": {
            "t": t_p.tolist(),
            "J_odd_ds10": [[pf.cpack(z) for z in row] for row in odd10],
            "J_even_ds10": [[pf.cpack(z) for z in row] for row in even10],
            "dJ_odd_10_vs_coh8": [[pf.cpack(z) for z in row] for row in dJ_odd_10],
            "dJ_odd_ds_10_minus_8": [[pf.cpack(z) for z in row] for row in d_odd_ds],
            "delta_of_dJ_odd": [[pf.cpack(z) for z in row] for row in d_delta],
        },
        "no_quantum_advantage_claim": True,
        "hhg_not_computed": True,
        "coherent_still_nmax8": True,
    }, t_p, odd8, odd10, even8, even10, d_odd_ds, dJ_odd_8, dJ_odd_10


def write_plots(out_dir, t, odd8, odd10, even8, even10, d_odd, dJ8, dJ10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)
    rows = (
        (axes[0, 0], odd8[:, 0].real, odd10[:, 0].real, d_odd[:, 0].real,
         r"$J_{\mathrm{odd},x}^{\mathrm{DS}}$"),
        (axes[0, 1], odd8[:, 1].real, odd10[:, 1].real, d_odd[:, 1].real,
         r"$J_{\mathrm{odd},y}^{\mathrm{DS}}$"),
        (axes[1, 0], dJ8[:, 0].real, dJ10[:, 0].real, (dJ10 - dJ8)[:, 0].real,
         r"$\Delta J_{\mathrm{odd},x}$"),
        (axes[1, 1], dJ8[:, 1].real, dJ10[:, 1].real, (dJ10 - dJ8)[:, 1].real,
         r"$\Delta J_{\mathrm{odd},y}$"),
    )
    for ax, a, b, dlt, title in rows:
        ax.plot(t, a, color="#4c6b8a", lw=1.4, label=r"$n_{\max}=8$")
        ax.plot(t, b, color="#c45c26", lw=1.4, label=r"$n_{\max}=10$")
        ax.plot(t, dlt, color="#2f9e44", lw=1.1, ls="--", label=r"$10-8$")
        ax.set_title(title)
        ax.axhline(0.0, color="#888", lw=0.6)
        ax.grid(True, alpha=0.25)
    axes[0, 1].legend(loc="best", fontsize=8)
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")
    fig.suptitle(
        r"DS $\theta$=0 $\pm$N Fock: $n_{\max}$ 8 vs 10, same dt"
        "\nnot HHG, not semiclassical failure, not output squeezing",
        fontsize=10)
    fig.tight_layout()
    svg = out_dir / "j_nmax10.svg"
    fig.savefig(svg)
    fig.savefig(out_dir / "j_nmax10.png", dpi=140)
    plt.close(fig)
    return [str(svg.as_posix())]


def base_report(hashes_before, env, sample_js, photon):
    dt_h = env.t_ramp / N_STEPS
    return {
        "record_date": RECORD_DATE,
        "model_id": MODEL_ID,
        "claim_level": "ds_nmax10_fock_trunc_not_HHG_not_discovery",
        "vacuum_squeeze_branch_closed": True,
        "did_not": [
            "rewrite trial.json", "rewrite 20261027 preflight.json",
            "rewrite 20261028 diag.json", "rewrite static.json",
            "rewrite 20261030 preflight.json", "rewrite 20261031 diag.json",
            "rewrite 20261101 switch.json", "rewrite 20261102 compare.json",
            "rewrite 20261103 trunc.json", "rewrite 20261103 perf.json",
            "rewrite 20261104 dt.json", "rewrite 20261105 coh_dt.json",
            "G1", "H2", "HHG spectra", "theta=pi/2", "theta=pi", "Cn",
            "nmax=12 rerun", "Wigner trajectories", "full BZ",
            "output photon statistics", "change Hamiltonian",
            "coherent nmax=10", "dt rescan",
        ],
        "definition": {
            "H": "omega0 n + He(k, X; N) + He(-k, X; N)",
            "envelope": "1-cycle sin^2 rising edge, stop at s=1, no plateau",
            "n_max": NMAX, "g": pf.G_ABS, "omega0": pf.OMEGA0,
            "alpha": 0.5, "r": ds.R_SQ, "theta": ds.THETA,
            "jobs": [list(j) for j in JOBS],
            "k_fractional": list(sh.K_FRAC),
            "n_steps_old": N_STEPS_OLD,
            "n_steps": N_STEPS,
            "dt": dt_h,
            "t_end": env.t_ramp,
            "sample_js": sample_js,
            "sample_rule": "j_new=2*j_old+1 so t matches 20261104/05; same locked dt",
            "parallel_jobs": 2,
            "checkpoint_every": CHECKPOINT_EVERY,
            "archive8": str(ARCHIVE_8.as_posix()),
            "dt_ds_archive": str(DT_DS.as_posix()),
            "coh_dt_archive": str(COH_DT.as_posix()),
            "coherent_baseline_nmax8": True,
        },
        "photon_input": photon,
        "old_archive_hashes_before": hashes_before,
        "p_top_is_not_convergence": True,
        "not_semiclassical_failure": True,
    }


def run(progress=None, ckpt_dir: Path | None = None):
    hashes_before = {str(p.as_posix()): ds.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    photon = photon_static_compare()
    if not DT_DS.is_file() or not COH_DT.is_file():
        return {
            "status": "missing_dt_or_coh_archive", "stopped": True,
            "old_archive_hashes_before": hashes_before,
            "photon_input": photon,
        }
    ds8 = coh.load_dt_ds(DT_DS)
    coh8 = load_coh_dt(COH_DT)
    if ds8["status"] != "ok" or coh8["status"] != "ok":
        return {
            "status": "baseline_archive_not_ok", "stopped": True,
            "ds_status": ds8["status"], "coh_status": coh8["status"],
            "photon_input": photon,
        }
    if ds8["n_steps"] != N_STEPS or coh8["n_steps"] != N_STEPS:
        return {"status": "baseline_step_mismatch", "stopped": True, "photon_input": photon}
    n_new, sample_js = dth.sample_steps_half_dt(N_STEPS_OLD, SAMPLE_EVERY_OLD)
    if n_new != N_STEPS:
        return {"status": "n_steps_lock_failed", "stopped": True, "photon_input": photon}
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    dt_h = env.t_ramp / N_STEPS
    t_align = dth.times_from_steps(N_STEPS, env.t_ramp, sample_js)
    if not np.allclose(t_align, ds8["t"], atol=1e-12, rtol=0.0):
        return {"status": "time_alignment_failed_no_propagation", "stopped": True,
                "photon_input": photon}
    if ckpt_dir is None:
        ckpt_dir = ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax10_20261106"
    report = base_report(hashes_before, env, sample_js, photon)
    cost = measure_cost(dt_h)
    cost.update({"n_steps": N_STEPS, "dt": dt_h, "dim": 36 * 36 * (NMAX + 1)})
    report["cost"] = cost
    ds._progress(progress, f"cost est_wall={cost['estimated_wall_parallel_seconds']:.1f}s "
                 f"par_step={cost['parallel_taylor_per_step']:.4f}s steps={N_STEPS} dt={dt_h} "
                 f"rho_dt={cost['rho_dt']:.4g} last={cost['timed_taylor_last_term']:.3e}")
    if not cost["taylor_finite"]:
        report["status"] = "taylor_unstable_no_propagation"
        report["stopped"] = True
        return report
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
    cmp_block, t, odd8, odd10, even8, even10, d_odd, dJ8, dJ10 = compare_nmax(ds8, coh8, runs)
    report["jobs"] = slim
    report["compare"] = {k: cmp_block[k] for k in cmp_block if k != "series"}
    report["compare"]["series"] = cmp_block["series"]
    report["numerical_floor"] = {
        "max_last_taylor": {tag: runs[tag]["ledger"]["max_last_taylor_term"] for tag in runs},
        "max_abs_norm_minus_1": {tag: runs[tag]["max_abs_norm_minus_1"] for tag in runs},
        "max_top_fock": {tag: runs[tag]["max_top_fock"] for tag in runs},
        "imag_J_floor": cmp_block["max_abs"]["imag_J_floor_10"],
        "target_abs_delta_J_odd": TARGET_ABS,
        "p_top_is_not_convergence": True,
    }
    report["status"] = "ok"
    report["stopped"] = True
    report["_plot_arrays"] = (t, odd8, odd10, even8, even10, d_odd, dJ8, dJ10)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path,
                    default=ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax10_20261106/trunc10.json")
    ap.add_argument("--progress", type=Path, default=None)
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    ckpt_dir = args.ckpt_dir or args.report.parent
    if args.report.exists():
        print(f"report already exists: {args.report}")
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
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
        "photon_table": None if "photon_input" not in report else report["photon_input"].get("table"),
        "max_abs": None if "compare" not in report else report["compare"].get("max_abs"),
        "below_1e-8": None if "compare" not in report else report["compare"].get("below_1e-8_target"),
        "gate": None if "compare" not in report else report["compare"].get("preliminary_truncation_gate"),
        "cursor_stop": None if "compare" not in report else report["compare"].get("cursor_stop"),
        "old_untouched": report.get("old_archive_hashes_unchanged"),
        "cost": report.get("cost"),
        "resumable": report.get("resumable", False),
    }
    print(json.dumps(ds.jsonable(summary), ensure_ascii=False, indent=2))
    if report.get("status") == "cost_gate_failed_no_propagation":
        return 4
    if report.get("status") == "taylor_unstable_no_propagation":
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
