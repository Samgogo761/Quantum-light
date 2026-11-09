#!/usr/bin/env python3
"""DS(θ=0) ±N nmax=12 truncation check vs archived nmax=8.

Not HHG, not advantage, not a dt scan, not θ/Cn/Wigner. Does not rewrite
old archives. Vacuum / G1 / H2 stay frozen. Wall cap 600 s; over budget
report and stop.
"""
from __future__ import annotations

import argparse
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
MODEL_ID = "M0_shared_mode_k_minus_k_ds_nmax12_trunc_v1"
RECORD_DATE = "2026-11-03"
WALL_CAP_S = 600.0
NMAX = 12
SAMPLE_EVERY = 32
JOBS = (("ds_p", 1), ("ds_m", -1))
ARCHIVE_8 = ROOT / "results/q_line/qinput_a_shared_kk_ds_odd_20261102/compare.json"
OLD_ARCHIVES = ds.OLD_ARCHIVES + (ARCHIVE_8,)
TARGET_ABS = 1e-8


def c_from_pack(z):
    return complex(float(z["re"]), float(z["im"]))


def j_from_rows(rows):
    return np.array([[c_from_pack(z) for z in row] for row in rows], complex)


def interp_j(t_src, j_src, t_dst):
    t_src = np.asarray(t_src, float)
    t_dst = np.asarray(t_dst, float)
    j_src = np.asarray(j_src, complex)
    out = np.empty((t_dst.size, j_src.shape[1]), complex)
    for a in range(j_src.shape[1]):
        out[:, a] = (
            np.interp(t_dst, t_src, j_src[:, a].real)
            + 1j * np.interp(t_dst, t_src, j_src[:, a].imag)
        )
    return out


def x_peierls_moments(mean_a, mean_n, mean_aa):
    """X = a+a† as in Peierls. Assumes ⟨[a,a†]⟩=1."""
    a = complex(mean_a)
    aa = complex(mean_aa)
    n = float(mean_n)
    mean_x = 2.0 * a.real
    mean_x2 = 2.0 * aa.real + 2.0 * n + 1.0
    return {
        "convention": "X=a+adag, commutator 1",
        "mean_X": mean_x,
        "mean_X2": mean_x2,
        "var_X": mean_x2 - mean_x * mean_x,
    }


def load_archive8(path: Path):
    import json
    with path.open(encoding="utf-8") as fh:
        o = json.load(fh)
    ser = o["compare"]["series"]
    t = np.asarray(ser["t"], float)
    return {
        "t": t,
        "J_odd_ds": j_from_rows(ser["J_odd_ds"]),
        "J_even_ds": j_from_rows(ser["J_even_ds"]),
        "J_odd_coh": j_from_rows(ser["J_odd_coh"]),
        "J_even_coh": j_from_rows(ser["J_even_coh"]),
        "dJ_odd": j_from_rows(ser["dJ_odd"]),
        "dJ_even": j_from_rows(ser["dJ_even"]),
        "max_abs": o["compare"]["max_abs"],
        "definition": o.get("definition"),
        "sample_every": o["definition"]["sample_every"],
        "n_max": o["definition"]["n_max"],
        "dt": o["cost"]["dt"],
        "n_steps": o["cost"]["n_steps"],
    }


def photon_input_record(n_max: int):
    raw = ds.photon_raw("ds", n_max, ds.THETA)
    mom = ds.photon_moments(raw)
    a = c_from_pack(mom["mean_a"])
    aa = c_from_pack(mom["mean_aa"])
    ref = ds.photon_raw("ds", ds.NMAX_REF, ds.THETA)
    p_gt = ds.p_n_gt(ref, n_max)
    return {
        "n_max": n_max,
        "unnormalized": mom,
        "X_peierls": x_peierls_moments(a, mom["mean_n"], aa),
        "large80_P_n_gt_nmax": p_gt,
        "not_a_D_S_convention_recheck": True,
    }


def run_job(geo, k, env, neel, n_steps, clock, progress, tag):
    ek, emk, masses = ds.ground_pair_neel(geo, k, neel)
    raw = ds.photon_raw("ds", NMAX, ds.THETA)
    raw_diag = ds.photon_moments(raw)
    c = raw / np.sqrt(raw_diag["unnormalized_mass"])
    model = sh.build_shared(geo, R0, k, neel, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    psi, prod_n = ds.product_normalized(ek, emk, c)
    s0 = pf.snapshot(model, psi, 0.0)
    ds._progress(progress, f"{tag} start neel={neel} dim={model.dim} steps={n_steps}")
    psi, led = pf.propagate(model, psi, 0.0, env.t_ramp, env, n_steps, clock, SAMPLE_EVERY)
    send = pf.snapshot(model, psi, 1.0)
    packed = pf.pack_samples(led["samples"])
    top = [s["top_fock_mass"] for s in led["samples"]]
    norms = [s["norm"] for s in led["samples"]]
    out = {
        "tag": tag, "kind": "ds", "neel": int(neel), "n_max": NMAX,
        "electron": masses,
        "photon_unnormalized": raw_diag,
        "photon_used_normalized": True,
        "product_norm_before_renorm": prod_n,
        "initial": s0, "after_ramp": send,
        "delta_energy": send["energy"] - s0["energy"],
        "work_trap_dHds_dsdt": led["work_trap_dHds_dsdt"],
        "energy_minus_work": (send["energy"] - s0["energy"]) - led["work_trap_dHds_dsdt"],
        "ledger": {k: led[k] for k in ("steps", "dt", "order", "max_last_taylor_term")},
        "max_top_fock": float(np.max(top) if top else send["top_fock_mass"]),
        "max_abs_norm_minus_1": float(np.max(np.abs(np.asarray(norms) - 1.0))) if norms else abs(send["norm"] - 1.0),
        "time_series": packed,
        "samples": led["samples"],
        "wall_seconds": clock.elapsed(),
        "top_two_alarm_0p05_is_not_convergence": True,
    }
    ds._progress(progress, f"{tag} done wall={clock.elapsed():.1f}s dE={out['delta_energy']:.3e}")
    return out


def _parallel_job(tag, neel, n_steps, progress_s):
    progress = Path(progress_s) if progress_s else None
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    clock = dyn.WallClock(WALL_CAP_S)
    return run_job(geo, k, env, neel, n_steps, clock, progress, tag)


def compare_trunc(arch, runs):
    t12, jp = ds.j_series(runs["ds_p"]["samples"])
    t_m, jm = ds.j_series(runs["ds_m"]["samples"])
    if not np.allclose(t12, t_m):
        raise RuntimeError("nmax=12 ±N time grids differ")
    even12, odd12 = ds.even_odd(jp, jm)
    t8 = arch["t"]
    if float(t12[0]) > float(t8[0]) + 1e-12 or float(t12[-1]) < float(t8[-1]) - 1e-12:
        raise RuntimeError("nmax=12 time grid does not cover archived nmax=8 samples")
    odd12_on8 = interp_j(t12, odd12, t8)
    even12_on8 = interp_j(t12, even12, t8)
    d_odd_ds = odd12_on8 - arch["J_odd_ds"]
    d_even_ds = even12_on8 - arch["J_even_ds"]
    dJ_odd_12_vs_coh8 = odd12_on8 - arch["J_odd_coh"]
    d_delta_odd = dJ_odd_12_vs_coh8 - arch["dJ_odd"]
    max_abs = {
        "J_odd_ds12_x": float(np.max(np.abs(odd12[:, 0]))),
        "J_odd_ds12_y": float(np.max(np.abs(odd12[:, 1]))),
        "J_even_ds12_x": float(np.max(np.abs(even12[:, 0]))),
        "J_even_ds12_y": float(np.max(np.abs(even12[:, 1]))),
        "dJ_odd_ds_12_minus_8_x": float(np.max(np.abs(d_odd_ds[:, 0]))),
        "dJ_odd_ds_12_minus_8_y": float(np.max(np.abs(d_odd_ds[:, 1]))),
        "dJ_even_ds_12_minus_8_x": float(np.max(np.abs(d_even_ds[:, 0]))),
        "dJ_even_ds_12_minus_8_y": float(np.max(np.abs(d_even_ds[:, 1]))),
        "dJ_odd_12vs_coh8_x": float(np.max(np.abs(dJ_odd_12_vs_coh8[:, 0]))),
        "dJ_odd_12vs_coh8_y": float(np.max(np.abs(dJ_odd_12_vs_coh8[:, 1]))),
        "delta_of_dJ_odd_x": float(np.max(np.abs(d_delta_odd[:, 0]))),
        "delta_of_dJ_odd_y": float(np.max(np.abs(d_delta_odd[:, 1]))),
        "archived_dJ_odd_x": float(np.max(np.abs(arch["dJ_odd"][:, 0]))),
        "archived_dJ_odd_y": float(np.max(np.abs(arch["dJ_odd"][:, 1]))),
        "imag_J_floor_12": float(max(np.max(np.abs(jp.imag)), np.max(np.abs(jm.imag)))),
    }
    below = {
        "dJ_odd_ds_trunc_x_lt_1e-8": max_abs["dJ_odd_ds_12_minus_8_x"] < TARGET_ABS,
        "dJ_odd_ds_trunc_y_lt_1e-8": max_abs["dJ_odd_ds_12_minus_8_y"] < TARGET_ABS,
        "target_abs": TARGET_ABS,
        "not_a_physics_pass": True,
    }
    return {
        "t8": t8.tolist(),
        "t12": t12.tolist(),
        "interpolated_12_onto_8": True,
        "max_abs": max_abs,
        "below_1e-8_target": below,
        "series": {
            "J_odd_ds12_on8": [[pf.cpack(z) for z in row] for row in odd12_on8],
            "J_even_ds12_on8": [[pf.cpack(z) for z in row] for row in even12_on8],
            "dJ_odd_ds_12_minus_8": [[pf.cpack(z) for z in row] for row in d_odd_ds],
            "dJ_even_ds_12_minus_8": [[pf.cpack(z) for z in row] for row in d_even_ds],
            "dJ_odd_12vs_coh8": [[pf.cpack(z) for z in row] for row in dJ_odd_12_vs_coh8],
        },
        "not_certified_harmonic_spectrum": True,
        "no_quantum_advantage_claim": True,
        "coherent_still_nmax8": True,
        "dt_refinement_not_run": True,
        "minus_alpha_not_propagated": True,
    }, t8, arch["J_odd_ds"], odd12_on8, arch["J_even_ds"], even12_on8, d_odd_ds


def write_plots(out_dir: Path, t, odd8, odd12, even8, even12, d_odd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)
    rows = (
        (axes[0, 0], odd8[:, 0].real, odd12[:, 0].real, d_odd[:, 0].real, r"$J_{\mathrm{odd},x}^{\mathrm{DS}}$"),
        (axes[0, 1], odd8[:, 1].real, odd12[:, 1].real, d_odd[:, 1].real, r"$J_{\mathrm{odd},y}^{\mathrm{DS}}$"),
        (axes[1, 0], even8[:, 0].real, even12[:, 0].real, (even12 - even8)[:, 0].real, r"$J_{\mathrm{even},x}^{\mathrm{DS}}$"),
        (axes[1, 1], even8[:, 1].real, even12[:, 1].real, (even12 - even8)[:, 1].real, r"$J_{\mathrm{even},y}^{\mathrm{DS}}$"),
    )
    for ax, a, b, dlt, title in rows:
        ax.plot(t, a, color="#4c6b8a", lw=1.4, label=r"$n_{\max}=8$")
        ax.plot(t, b, color="#c45c26", lw=1.4, label=r"$n_{\max}=12$")
        ax.plot(t, dlt, color="#2f9e44", lw=1.1, ls="--", label=r"$12-8$")
        ax.set_title(title)
        ax.axhline(0.0, color="#888", lw=0.6)
        ax.grid(True, alpha=0.25)
    axes[0, 1].legend(loc="best", fontsize=8)
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")
    fig.suptitle("DS θ=0 ±N truncation: nmax 8 vs 12\nnot HHG, not advantage, not dt scan", fontsize=10)
    fig.tight_layout()
    png = out_dir / "j_trunc.png"
    svg = out_dir / "j_trunc.svg"
    fig.savefig(png, dpi=140)
    fig.savefig(svg)
    plt.close(fig)
    return [str(png.as_posix()), str(svg.as_posix())]


def run(progress=None):
    hashes_before = {str(p.as_posix()): ds.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    if not ARCHIVE_8.is_file():
        return {
            "status": "missing_nmax8_archive", "stopped": True,
            "old_archive_hashes_before": hashes_before,
        }
    arch = load_archive8(ARCHIVE_8)
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    report = {
        "record_date": RECORD_DATE,
        "model_id": MODEL_ID,
        "claim_level": "ds_nmax12_truncation_check_not_HHG_not_odd_discovery",
        "vacuum_squeeze_branch_closed": True,
        "did_not": [
            "rewrite trial.json", "rewrite 20261027 preflight.json",
            "rewrite 20261028 diag.json", "rewrite static.json",
            "rewrite 20261030 preflight.json", "rewrite 20261031 diag.json",
            "rewrite 20261101 switch.json", "rewrite 20261102 compare.json",
            "G1", "H2", "HHG spectra", "theta=pi/2", "theta=pi", "Cn",
            "Wigner trajectories", "full BZ", "output photon statistics",
            "dt refinement", "minus-alpha joint-inversion jobs",
            "recheck D(alpha)S(xi) convention",
        ],
        "definition": {
            "H": "omega0 n + He(k, X; N) + He(-k, X; N)",
            "envelope": "1-cycle sin^2 rising edge, stop at s=1, no plateau",
            "n_max": NMAX, "g": pf.G_ABS, "omega0": pf.OMEGA0,
            "alpha": 0.5, "r": ds.R_SQ, "theta": ds.THETA,
            "jobs": [list(j) for j in JOBS],
            "k_fractional": list(sh.K_FRAC),
            "sample_every": SAMPLE_EVERY,
            "parallel_jobs": 2,
            "archive8": str(ARCHIVE_8.as_posix()),
        },
        "photon_input": {
            "nmax8": photon_input_record(8),
            "nmax12": photon_input_record(12),
        },
        "joint_inversion_current_interface": {
            "formula": "J_{-N}(t; alpha, xi) = -J_{+N}(t; -alpha, xi)",
            "minus_alpha_not_propagated": True,
            "cost_if_added_two_jobs": "same as this round; not executed",
        },
        "old_archive_hashes_before": hashes_before,
        "archive8_sample_dt": float(arch["dt"] * arch["sample_every"]),
        "sample_every_unchanged_32": True,
    }
    rng = np.random.default_rng(20261103)
    model_probe = sh.build_shared(geo, R0, k, 1, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    rho, _ = pf.spectral_radius_power(model_probe, 1.0, rng)
    n_steps = dyn.n_steps_for(env.t_ramp, rho, requested=32, hdt_max=pf.TAYLOR_HDT_MAX)
    dt = env.t_ramp / n_steps
    v = rng.normal(size=model_probe.dim) + 1j * rng.normal(size=model_probe.dim)
    v = v / np.linalg.norm(v)
    t0 = time.perf_counter()
    pf.taylor_step(model_probe, v, 0.4, dt)
    t_step = time.perf_counter() - t0
    est_one_raw = t_step * n_steps
    est_one = 1.3 * est_one_raw
    est_serial = 2.0 * est_one
    est_wall = est_one
    report["cost"] = {
        "dim": model_probe.dim, "n_steps": n_steps, "dt": dt,
        "timed_taylor_step_seconds": t_step,
        "sample_every": SAMPLE_EVERY,
        "estimated_one_job_raw_seconds": est_one_raw,
        "estimated_one_job_seconds": est_one,
        "estimated_serial_seconds": est_serial,
        "estimated_wall_parallel_seconds": est_wall,
        "wall_cap_s": WALL_CAP_S,
        "exceeds_10min": bool(est_wall > WALL_CAP_S),
        "serial_two_jobs_exceed_10min": bool(est_serial > WALL_CAP_S),
        "note": "x1.3 from nmax=8 sample_every=32 actual ~109s vs ~90s raw. Two DS jobs in parallel.",
        "minus_alpha_extra_wall_if_run_parallel_s": est_wall,
    }
    ds._progress(progress, f"cost est_wall={est_wall:.1f}s serial={est_serial:.1f}s steps={n_steps} dt={dt} dim={model_probe.dim}")
    if est_wall > WALL_CAP_S:
        report["status"] = "cost_gate_failed_no_propagation"
        report["stopped"] = True
        return report

    t_par0 = time.perf_counter()
    progress_s = str(progress) if progress is not None else ""
    runs = {}
    try:
        with ProcessPoolExecutor(max_workers=2) as ex:
            futs = {ex.submit(_parallel_job, tag, neel, n_steps, progress_s): tag for tag, neel in JOBS}
            for fut in futs:
                tag = futs[fut]
                runs[tag] = fut.result()
    except TimeoutError as exc:
        report["status"] = "wall_timeout_no_full_propagation"
        report["stopped"] = True
        report["error"] = str(exc)
        report["wall_seconds_total"] = time.perf_counter() - t_par0
        return report
    clock_elapsed = time.perf_counter() - t_par0
    slim = {tag: {k: job[k] for k in job if k != "samples"} for tag, job in runs.items()}
    cmp_block, t, odd8, odd12, even8, even12, d_odd = compare_trunc(arch, runs)
    report["jobs"] = slim
    report["compare"] = {k: cmp_block[k] for k in cmp_block if k != "series"}
    report["compare"]["series"] = cmp_block["series"]
    report["numerical_floor"] = {
        "max_last_taylor": {tag: runs[tag]["ledger"]["max_last_taylor_term"] for tag in runs},
        "max_abs_norm_minus_1": {tag: runs[tag]["max_abs_norm_minus_1"] for tag in runs},
        "max_top_fock": {tag: runs[tag]["max_top_fock"] for tag in runs},
        "imag_J_floor": cmp_block["max_abs"]["imag_J_floor_12"],
        "no_dt_refinement_this_round": True,
        "vacuum_dt_Wsq_rel_was_2e-7_not_a_J_errorbar": True,
        "target_abs_J_odd": TARGET_ABS,
    }
    report["wall_seconds_total"] = clock_elapsed
    report["status"] = "ok"
    report["stopped"] = True
    report["_plot_arrays"] = (t, odd8, odd12, even8, even12, d_odd)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path,
                    default=ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax12_20261103/trunc.json")
    ap.add_argument("--progress", type=Path, default=None)
    ap.add_argument("--cost-only", action="store_true")
    args = ap.parse_args(argv)
    if args.report.exists():
        print(f"report already exists: {args.report}")
        return 2
    if args.cost_only:
        geo = build_geometry()
        k = geo.k_cart(sh.K_FRAC)
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        rng = np.random.default_rng(20261103)
        model = sh.build_shared(geo, R0, k, 1, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
        rho, _ = pf.spectral_radius_power(model, 1.0, rng)
        n_steps = dyn.n_steps_for(env.t_ramp, rho, requested=32, hdt_max=pf.TAYLOR_HDT_MAX)
        dt = env.t_ramp / n_steps
        v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
        v = v / np.linalg.norm(v)
        t0 = time.perf_counter()
        pf.taylor_step(model, v, 0.4, dt)
        t_step = time.perf_counter() - t0
        est = 1.3 * t_step * n_steps
        payload = {
            "dim": model.dim, "n_steps": n_steps, "dt": dt,
            "timed_taylor_step_seconds": t_step,
            "estimated_wall_parallel_seconds": est,
            "wall_cap_s": WALL_CAP_S,
            "exceeds_10min": bool(est > WALL_CAP_S),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        import json
        with args.report.open("x", encoding="utf-8") as fh:
            json.dump(ds.jsonable(payload), fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(ds.jsonable(payload))
        return 1 if est > WALL_CAP_S else 0
    t0 = time.perf_counter()
    report = run(progress=args.progress)
    plot_arrays = report.pop("_plot_arrays", None)
    report["wall_seconds_process"] = time.perf_counter() - t0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if plot_arrays is not None:
        report["figures"] = write_plots(args.report.parent, *plot_arrays)
    hashes_after = {str(p.as_posix()): ds.sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    report["old_archive_hashes_unchanged"] = hashes_after == report.get("old_archive_hashes_before", hashes_after)
    import json
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
    }
    print(json.dumps(ds.jsonable(summary), ensure_ascii=False, indent=2))
    if report.get("status") == "cost_gate_failed_no_propagation":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
