#!/usr/bin/env python3
"""nmax=8 DS(θ=0) ±N, dt halved vs 20261102. Time-step check of J_odd^DS only.

Not HHG, not ΔJ_odd certification (coherent dt not repeated), not nmax=12.
Does not rewrite old archives. Wall cap 360 s; over-estimate stops before run.
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
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "M0_shared_mode_k_minus_k_ds_dt_half_v1"
RECORD_DATE = "2026-11-04"
WALL_CAP_S = 360.0
NMAX = 8
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
OLD_ARCHIVES = ds.OLD_ARCHIVES + (ARCHIVE_8, TRUNC_12)


def c_from_pack(z):
    return complex(float(z["re"]), float(z["im"]))


def j_from_rows(rows):
    return np.array([[c_from_pack(z) for z in row] for row in rows], complex)


def load_archive8(path: Path):
    with path.open(encoding="utf-8") as fh:
        o = json.load(fh)
    ser = o["compare"]["series"]
    t = np.asarray(ser["t"], float)
    return {
        "t": t,
        "J_odd_ds": j_from_rows(ser["J_odd_ds"]),
        "J_even_ds": j_from_rows(ser["J_even_ds"]),
        "dt": o["cost"]["dt"],
        "n_steps": o["cost"]["n_steps"],
        "sample_every": o["definition"]["sample_every"],
        "max_abs": o["compare"]["max_abs"],
    }


def sample_steps_half_dt(n_old: int, sample_every_old: int):
    """Old propagate samples after step j when j%32==0 or last.
    t=(j+1)*dt_old. Half dt uses j_new=2*j_old+1 so t matches exactly.
    Naive sample_every=64 on the new grid would miss the +1 offset.
    """
    n_new = 2 * int(n_old)
    old = list(range(0, n_old, sample_every_old))
    if not old or old[-1] != n_old - 1:
        old.append(n_old - 1)
    return n_new, [2 * j + 1 for j in old]


def times_from_steps(n_steps: int, t1: float, js):
    dt = t1 / n_steps
    return np.array([(j + 1) * dt for j in js], float)


def ckpt_paths(ckpt_dir: Path, tag: str):
    return ckpt_dir / f"ckpt_{tag}.npz", ckpt_dir / f"ckpt_{tag}_samples.json"


def save_ckpt(ckpt_dir: Path, tag: str, psi, next_j: int, samples):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    p_npz, p_js = ckpt_paths(ckpt_dir, tag)
    np.savez_compressed(p_npz, psi=np.asarray(psi, complex), next_j=np.int64(next_j))
    p_js.write_text(json.dumps(ds.jsonable(samples), ensure_ascii=False), encoding="utf-8")


def load_ckpt(ckpt_dir: Path, tag: str):
    p_npz, p_js = ckpt_paths(ckpt_dir, tag)
    if not p_npz.is_file() or not p_js.is_file():
        return None
    z = np.load(p_npz)
    samples = json.loads(p_js.read_text(encoding="utf-8"))
    return np.asarray(z["psi"], complex), int(z["next_j"]), samples


def propagate_ckpt(model, psi, t0, t1, env, n_steps, sample_js, clock,
                   ckpt_dir: Path, tag: str, start_j: int = 0, samples=None,
                   ckpt_every: int = CHECKPOINT_EVERY):
    dt = (t1 - t0) / n_steps
    sample_set = set(int(j) for j in sample_js)
    samples = list(samples or [])
    worst_last = 0.0
    j = int(start_j)
    timed_out = False
    try:
        while j < n_steps:
            clock.check()
            tm = t0 + (j + 0.5) * dt
            s = env.s(tm)
            Ds = dyn.peierls_at(model.q_units, s, model.n_max)
            psi, last = pf.taylor_step(model, psi, s, dt, Ds=Ds)
            worst_last = max(worst_last, last)
            if j in sample_set:
                obs = pf.snapshot(model, psi, env.s(t0 + (j + 1) * dt), Ds=None)
                obs["t"] = t0 + (j + 1) * dt
                dhds = pf.dHds_expect(model, psi, env.s(obs["t"]))
                dsdt = env.ds(obs["t"])
                obs["dHds_expect"] = pf.cpack(dhds)
                obs["dsdt"] = float(dsdt)
                obs["power_dHds_dsdt"] = float((dhds * dsdt).real)
                samples.append(obs)
            if (j + 1) % 32 == 0 or j == n_steps - 1:
                nrm = float(np.linalg.norm(psi))
                if (not np.isfinite(psi).all()) or nrm > 2.0 or nrm < 0.5:
                    raise RuntimeError(f"norm {nrm} at step {j}")
            j += 1
            if ckpt_every and j % ckpt_every == 0:
                save_ckpt(ckpt_dir, tag, psi, j, samples)
    except TimeoutError:
        timed_out = True
        save_ckpt(ckpt_dir, tag, psi, j, samples)
    work = 0.0
    if len(samples) >= 2:
        for a, b in zip(samples[:-1], samples[1:]):
            work += 0.5 * (a["power_dHds_dsdt"] + b["power_dHds_dsdt"]) * (b["t"] - a["t"])
    return psi, {
        "steps": n_steps, "dt": dt, "order": pf.ORDER,
        "max_last_taylor_term": worst_last,
        "samples": samples,
        "work_trap_dHds_dsdt": float(work),
        "timed_out": timed_out,
        "next_j": j,
        "resumable": bool(timed_out),
    }


def _mini_loop(model, v, dt, n_steps):
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    psi = v.copy()
    t0 = time.perf_counter()
    for j in range(n_steps):
        s = env.s((j + 0.5) * dt)
        Ds = dyn.peierls_at(model.q_units, s, model.n_max)
        psi, _ = pf.taylor_step(model, psi, s, dt, Ds=Ds)
    return time.perf_counter() - t0


def _worker_mini(n_steps, dt):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    model = sh.build_shared(geo, R0, k, 1, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    rng = np.random.default_rng(20261104)
    v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    v = v / np.linalg.norm(v)
    loop_s = _mini_loop(model, v, dt, n_steps)
    return {"loop_s": loop_s, "taylor_per_step": loop_s / n_steps, "pid": os.getpid()}


def measure_cost(dt: float):
    t_serial0 = time.perf_counter()
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    model = sh.build_shared(geo, R0, k, 1, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    rng = np.random.default_rng(20261104)
    v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    v = v / np.linalg.norm(v)
    serial_loop = _mini_loop(model, v, dt, N_MINI)
    serial_step = serial_loop / N_MINI
    serial_probe_s = time.perf_counter() - t_serial0
    t_par0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(_worker_mini, N_MINI, dt) for _ in range(2)]
        recs = [f.result() for f in futs]
    par_wall = time.perf_counter() - t_par0
    par_step = max(r["taylor_per_step"] for r in recs)
    one = par_step * N_STEPS * SUSTAINED
    return {
        "n_mini": N_MINI,
        "dim": model.dim,
        "serial_taylor_per_step": serial_step,
        "parallel_taylor_per_step": par_step,
        "parallel_mini_wall_s": par_wall,
        "serial_probe_s": serial_probe_s,
        "contention_step_ratio": par_step / max(serial_step, 1e-30),
        "sustained_factor": SUSTAINED,
        "estimated_one_job_seconds": one,
        "estimated_wall_parallel_seconds": one,
        "estimated_serial_two_jobs_seconds": 2.0 * serial_step * N_STEPS * SUSTAINED,
        "archive8_job_wall_s": 108.58,
        "archive8_scaled_two_times_steps_s": 108.58 * (N_STEPS / N_STEPS_OLD),
        "wall_cap_s": WALL_CAP_S,
        "exceeds_6min": bool(one > WALL_CAP_S),
        "parallel_jobs": recs,
        "note": "loop timing not isolated Taylor; x1.15 for sustained drift seen at nmax=12",
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
    loaded = load_ckpt(ckpt_dir, tag)
    if loaded is not None:
        psi, start_j, samples = loaded
        ds._progress(progress, f"{tag} resume next_j={start_j}")
    s0 = pf.snapshot(model, psi, env.s(start_j * (env.t_ramp / n_steps)))
    ds._progress(progress, f"{tag} start neel={neel} dim={model.dim} steps={n_steps} j0={start_j}")
    psi, led = propagate_ckpt(
        model, psi, 0.0, env.t_ramp, env, n_steps, sample_js, clock,
        ckpt_dir, tag, start_j=start_j, samples=samples)
    if led["timed_out"]:
        ds._progress(progress, f"{tag} timeout next_j={led['next_j']} wall={clock.elapsed():.1f}s")
        return {
            "tag": tag, "kind": "ds", "neel": int(neel), "status": "wall_timeout_checkpointed",
            "next_j": led["next_j"], "n_steps": n_steps,
            "ckpt": str(ckpt_paths(ckpt_dir, tag)[0].as_posix()),
            "n_samples_so_far": len(led["samples"]),
            "wall_seconds": clock.elapsed(),
            "resumable": True,
        }
    send = pf.snapshot(model, psi, 1.0)
    packed = pf.pack_samples(led["samples"])
    top = [s["top_fock_mass"] for s in led["samples"]]
    norms = [s["norm"] for s in led["samples"]]
    save_ckpt(ckpt_dir, tag, psi, n_steps, led["samples"])
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


def compare_dt(arch, runs):
    t_p, jp = ds.j_series(runs["ds_p"]["samples"])
    t_m, jm = ds.j_series(runs["ds_m"]["samples"])
    if not np.allclose(t_p, t_m, atol=1e-12, rtol=0.0):
        raise RuntimeError("half-dt ±N time grids differ")
    if not np.allclose(t_p, arch["t"], atol=1e-12, rtol=0.0):
        raise RuntimeError("half-dt times do not match 20261102 compare.json")
    even_h, odd_h = ds.even_odd(jp, jm)
    d_odd = odd_h - arch["J_odd_ds"]
    d_even = even_h - arch["J_even_ds"]
    max_abs = {
        "J_odd_ds_half_x": float(np.max(np.abs(odd_h[:, 0]))),
        "J_odd_ds_half_y": float(np.max(np.abs(odd_h[:, 1]))),
        "J_odd_ds_old_x": float(np.max(np.abs(arch["J_odd_ds"][:, 0]))),
        "J_odd_ds_old_y": float(np.max(np.abs(arch["J_odd_ds"][:, 1]))),
        "dJ_odd_ds_half_minus_old_x": float(np.max(np.abs(d_odd[:, 0]))),
        "dJ_odd_ds_half_minus_old_y": float(np.max(np.abs(d_odd[:, 1]))),
        "dJ_even_ds_half_minus_old_x": float(np.max(np.abs(d_even[:, 0]))),
        "dJ_even_ds_half_minus_old_y": float(np.max(np.abs(d_even[:, 1]))),
        "imag_J_floor": float(max(np.max(np.abs(jp.imag)), np.max(np.abs(jm.imag)))),
    }
    below = {
        "dJ_odd_ds_x_lt_1e-8": max_abs["dJ_odd_ds_half_minus_old_x"] < TARGET_ABS,
        "dJ_odd_ds_y_lt_1e-8": max_abs["dJ_odd_ds_half_minus_old_y"] < TARGET_ABS,
        "target_abs": TARGET_ABS,
        "does_not_certify_delta_J_odd": True,
        "coherent_dt_not_repeated": True,
        "not_a_physics_pass": True,
    }
    return {
        "aligned_without_interpolation": True,
        "max_abs": max_abs,
        "below_1e-8_target": below,
        "series": {
            "t": t_p.tolist(),
            "J_odd_ds_half": [[pf.cpack(z) for z in row] for row in odd_h],
            "J_even_ds_half": [[pf.cpack(z) for z in row] for row in even_h],
            "dJ_odd_ds_half_minus_old": [[pf.cpack(z) for z in row] for row in d_odd],
            "dJ_even_ds_half_minus_old": [[pf.cpack(z) for z in row] for row in d_even],
        },
        "no_quantum_advantage_claim": True,
        "nmax12_not_run": True,
    }, t_p, arch["J_odd_ds"], odd_h, arch["J_even_ds"], even_h, d_odd


def write_plots(out_dir: Path, t, odd_old, odd_h, even_old, even_h, d_odd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)
    rows = (
        (axes[0, 0], odd_old[:, 0].real, odd_h[:, 0].real, d_odd[:, 0].real, r"$J_{\mathrm{odd},x}^{\mathrm{DS}}$"),
        (axes[0, 1], odd_old[:, 1].real, odd_h[:, 1].real, d_odd[:, 1].real, r"$J_{\mathrm{odd},y}^{\mathrm{DS}}$"),
        (axes[1, 0], even_old[:, 0].real, even_h[:, 0].real, (even_h - even_old)[:, 0].real, r"$J_{\mathrm{even},x}^{\mathrm{DS}}$"),
        (axes[1, 1], even_old[:, 1].real, even_h[:, 1].real, (even_h - even_old)[:, 1].real, r"$J_{\mathrm{even},y}^{\mathrm{DS}}$"),
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
    fig.suptitle("DS θ=0 ±N time-step: nmax=8 dt vs dt/2\nnot HHG, not ΔJ_odd certification", fontsize=10)
    fig.tight_layout()
    png = out_dir / "j_dt.svg"
    fig.savefig(png)
    fig.savefig(out_dir / "j_dt.png", dpi=140)
    plt.close(fig)
    return [str(png.as_posix())]


def base_report(hashes_before, arch, env, sample_js):
    dt = env.t_ramp / N_STEPS
    return {
        "record_date": RECORD_DATE,
        "model_id": MODEL_ID,
        "claim_level": "ds_dt_half_J_odd_not_HHG_not_delta_certified",
        "vacuum_squeeze_branch_closed": True,
        "did_not": [
            "rewrite trial.json", "rewrite 20261027 preflight.json",
            "rewrite 20261028 diag.json", "rewrite static.json",
            "rewrite 20261030 preflight.json", "rewrite 20261031 diag.json",
            "rewrite 20261101 switch.json", "rewrite 20261102 compare.json",
            "rewrite 20261103 trunc.json", "rewrite 20261103 perf.json",
            "G1", "H2", "HHG spectra", "theta=pi/2", "theta=pi", "Cn",
            "nmax=12 rerun", "coherent dt rerun", "Wigner trajectories",
            "full BZ", "output photon statistics", "change Hamiltonian",
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
            "dt_old": arch["dt"],
            "dt": dt,
            "t_end": env.t_ramp,
            "sample_js": sample_js,
            "sample_rule": "j_new=2*j_old+1 so t matches archive; not naive every-64",
            "parallel_jobs": 2,
            "checkpoint_every": CHECKPOINT_EVERY,
            "archive8": str(ARCHIVE_8.as_posix()),
        },
        "old_archive_hashes_before": hashes_before,
        "delta_J_odd_not_certified": True,
    }


def run(progress=None, ckpt_dir: Path | None = None):
    hashes_before = {str(p.as_posix()): ds.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    if not ARCHIVE_8.is_file():
        return {"status": "missing_nmax8_archive", "stopped": True,
                "old_archive_hashes_before": hashes_before}
    arch = load_archive8(ARCHIVE_8)
    if arch["n_steps"] != N_STEPS_OLD or arch["sample_every"] != SAMPLE_EVERY_OLD:
        return {"status": "archive_step_mismatch", "stopped": True,
                "n_steps": arch["n_steps"], "sample_every": arch["sample_every"]}
    n_new, sample_js = sample_steps_half_dt(arch["n_steps"], arch["sample_every"])
    if n_new != N_STEPS:
        return {"status": "n_steps_lock_failed", "stopped": True}
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    dt = env.t_ramp / N_STEPS
    t_align = times_from_steps(N_STEPS, env.t_ramp, sample_js)
    if not np.allclose(t_align, arch["t"], atol=1e-12, rtol=0.0):
        return {"status": "time_alignment_failed_no_propagation", "stopped": True,
                "t_align0": float(t_align[0]), "t_arch0": float(arch["t"][0])}
    if ckpt_dir is None:
        ckpt_dir = ROOT / "results/q_line/qinput_a_shared_kk_ds_dt_20261104"
    report = base_report(hashes_before, arch, env, sample_js)
    cost = measure_cost(dt)
    cost.update({"n_steps": N_STEPS, "dt": dt, "dim": 36 * 36 * (NMAX + 1)})
    report["cost"] = cost
    ds._progress(progress, f"cost est_wall={cost['estimated_wall_parallel_seconds']:.1f}s "
                 f"par_step={cost['parallel_taylor_per_step']:.4f}s steps={N_STEPS} dt={dt}")
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
    cmp_block, t, odd_old, odd_h, even_old, even_h, d_odd = compare_dt(arch, runs)
    report["jobs"] = slim
    report["compare"] = {k: cmp_block[k] for k in cmp_block if k != "series"}
    report["compare"]["series"] = cmp_block["series"]
    report["numerical_floor"] = {
        "max_last_taylor": {tag: runs[tag]["ledger"]["max_last_taylor_term"] for tag in runs},
        "max_abs_norm_minus_1": {tag: runs[tag]["max_abs_norm_minus_1"] for tag in runs},
        "max_top_fock": {tag: runs[tag]["max_top_fock"] for tag in runs},
        "imag_J_floor": cmp_block["max_abs"]["imag_J_floor"],
        "target_abs_J_odd": TARGET_ABS,
        "coherent_dt_not_repeated": True,
    }
    report["status"] = "ok"
    report["stopped"] = True
    report["_plot_arrays"] = (t, odd_old, odd_h, even_old, even_h, d_odd)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path,
                    default=ROOT / "results/q_line/qinput_a_shared_kk_ds_dt_20261104/dt.json")
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
        arch = load_archive8(ARCHIVE_8)
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        dt = env.t_ramp / N_STEPS
        cost = measure_cost(dt)
        cost.update({"n_steps": N_STEPS, "dt": dt, "n_steps_old": arch["n_steps"]})
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
