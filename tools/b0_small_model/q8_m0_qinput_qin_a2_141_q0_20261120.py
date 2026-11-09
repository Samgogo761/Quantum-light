#!/usr/bin/env python3
"""Q0: 400-step prefix of Qin_a2_1_4_1 shared-mode QED.

Single job: coherent α=0.5, N=+1, k/-k shared photon. Cost and initial
Fock only. Not a spectrum, not HHG, not a plateau Fock certificate.
Does not start Q1a/b/c, MTEF, G1/H2, vacuum, or classical Floquet.
Does not rewrite frozen archives or Hamiltonian physics.
"""
from __future__ import annotations

import json
import os
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

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_ds_odd_20261102 as odd
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_qin_a2_141_q0_20261120"
OLD_COMPARE = ROOT / "results/q_line/qinput_a_shared_kk_ds_odd_20261102/compare.json"
TASK_TAG = "20261120"
RECORD_DATE = "2026-10-22"
PROTOCOL = "Qin_a2_1_4_1"
MU = -0.009010620372198276
NMAX = odd.NMAX
ALPHA = odd.ALPHA
NEEL = 1
Q0_STEPS = 400
SAMPLE_EVERY = 32
WALL_CAP_S = 120.0
FOCK_ALARM = 0.05
HIST_RAMP_STEPS = 1285
HIST_DT = 0.09779276742691963
HIST_FOUR_WALL = 109.83249960001558
HIST_COH_P_WALL = 108.89288060000399
HIST_COH_P_TOP = 7.352547965451707e-10
HIST_DS_P_TOP = 1.737852253360549e-05
FROZEN = (
    HERE / "q8_m0_qinput_dynamics_20261026.py",
    HERE / "q8_m0_qinput_minimal_20261020.py",
    HERE / "q8_m0_qinput_shared_kk_static_20261029.py",
    HERE / "q8_m0_qinput_shared_kk_dyn_preflight_20261030.py",
    HERE / "q8_m0_qinput_shared_kk_ds_odd_20261102.py",
    HERE / "q8_m0_model.py",
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


def source_hashes() -> dict:
    out = {p.name: sha256_file(p) for p in FROZEN}
    out[Path(__file__).name] = sha256_file(Path(__file__))
    return out


def a_star_mean(geo) -> float:
    return 2.0 * float(geo.a_nn_mean_ang) * float(pf.G_ABS) * abs(complex(ALPHA))


def propagate_prefix(model, psi, t0, t1, env, n_steps, clock, sample_every):
    """Thin wrapper around pf.taylor_step / pf.snapshot.

    Unlike pf.propagate, a WallClock timeout keeps already sampled points.
    Does not edit pf.propagate.
    """
    dt = (t1 - t0) / n_steps
    samples = []
    work = 0.0
    worst_last = 0.0
    completed = 0
    timed_out = False
    timeout_msg = None
    t_loop0 = time.perf_counter()
    for j in range(n_steps):
        try:
            clock.check()
        except TimeoutError as exc:
            timed_out = True
            timeout_msg = str(exc)
            break
        tm = t0 + (j + 0.5) * dt
        s = env.s(tm)
        Ds = dyn.peierls_at(model.q_units, s, model.n_max)
        psi, last = pf.taylor_step(model, psi, s, dt, Ds=Ds)
        worst_last = max(worst_last, last)
        completed = j + 1
        take = bool(sample_every) and (j % sample_every == 0 or j == n_steps - 1)
        if take:
            t_now = t0 + (j + 1) * dt
            obs = pf.snapshot(model, psi, env.s(t_now), Ds=None)
            obs["t"] = t_now
            dhds = pf.dHds_expect(model, psi, env.s(t_now))
            dsdt = env.ds(t_now)
            obs["dHds_expect"] = pf.cpack(dhds)
            obs["dsdt"] = float(dsdt)
            obs["power_dHds_dsdt"] = float((dhds * dsdt).real)
            obs["step"] = completed
            samples.append(obs)
        if (j + 1) % 32 == 0 or j == n_steps - 1:
            nrm = float(np.linalg.norm(psi))
            if (not np.isfinite(psi).all()) or nrm > 2.0 or nrm < 0.5:
                raise RuntimeError(f"norm {nrm} at step {j}")
    if len(samples) >= 2:
        for a, b in zip(samples[:-1], samples[1:]):
            work += 0.5 * (a["power_dHds_dsdt"] + b["power_dHds_dsdt"]) * (b["t"] - a["t"])
    return psi, {
        "steps_requested": int(n_steps),
        "steps_completed": int(completed),
        "dt": float(dt),
        "t0": float(t0),
        "t1_requested": float(t1),
        "t1_reached": float(t0 + completed * dt),
        "order": pf.ORDER,
        "max_last_taylor_term": float(worst_last),
        "samples": samples,
        "work_trap_dHds_dsdt": float(work),
        "timed_out": bool(timed_out),
        "timeout_msg": timeout_msg,
        "loop_seconds": float(time.perf_counter() - t_loop0),
    }


def series_max(samples, key, default=None):
    if not samples:
        return default
    return float(np.max([s[key] for s in samples]))


def write_figure(samples, env, path: Path) -> list:
    if not samples:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    t = np.array([s["t"] for s in samples], float)
    s = np.array([s["s"] for s in samples], float)
    top = np.array([s["top_fock_mass"] for s in samples], float)
    nrm = np.array([s["norm"] for s in samples], float)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    axes[0].plot(t / env.period, s, color="#145860", lw=1.6)
    axes[0].set_ylabel(r"$s(t)$")
    axes[0].set_title("Q0 prefix only: coupling switch on the rising edge\nnot a spectrum, not plateau Fock")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(t / env.period, top, color="#c45c26", lw=1.6, label=r"$p_{n_{\max}}$")
    axes[1].axhline(FOCK_ALARM, color="#a33", ls="--", lw=0.9, label="alarm 0.05")
    axes[1].set_xlabel(r"$t/T_0$")
    axes[1].set_ylabel(r"top Fock mass")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    axn = axes[1].twinx()
    axn.plot(t / env.period, np.abs(nrm - 1.0), color="#4c6b8a", lw=1.0, alpha=0.7)
    axn.set_ylabel(r"$|\lVert\psi\rVert-1|$")
    fig.tight_layout()
    png = path.with_suffix(".png")
    svg = path.with_suffix(".svg")
    fig.savefig(png, dpi=140)
    fig.savefig(svg)
    plt.close(fig)
    return [str(png.as_posix()), str(svg.as_posix())]


def extrapolate(q0_wall, n_completed, n_full):
    n_completed = max(int(n_completed), 1)
    n_full = int(n_full)
    single_from_q0 = q0_wall * (n_full / n_completed)
    par_factor = HIST_FOUR_WALL / HIST_COH_P_WALL
    four_from_q0 = single_from_q0 * par_factor
    single_from_hist = HIST_COH_P_WALL * (n_full / HIST_RAMP_STEPS)
    four_from_hist = single_from_hist * par_factor
    return {
        "do_not_use_taylor_step_quote": True,
        "q0_is_one_job_not_four_way_wall": True,
        "n_q0_completed": n_completed,
        "n_full_1_4_1": n_full,
        "historical_ramp_steps": HIST_RAMP_STEPS,
        "historical_four_way_wall_s": HIST_FOUR_WALL,
        "historical_slowest_job_coh_p_s": HIST_COH_P_WALL,
        "historical_parallel_factor_four_over_slowest": par_factor,
        "single_job_1_4_1_from_q0_s": single_from_q0,
        "four_way_1_4_1_from_q0_times_hist_parallel_s": four_from_q0,
        "single_job_1_4_1_from_hist_scale_s": single_from_hist,
        "four_way_1_4_1_from_hist_scale_s": four_from_hist,
        "scheme_prior_four_way_min": 10.9,
        "local_10min_s": 600.0,
        "suggested_q1a_budget_s": 900.0,
        "q1a_from_q0_under_10min": bool(four_from_q0 <= 600.0),
        "q1a_from_q0_under_900s": bool(four_from_q0 <= 900.0),
        "keep_1_4_1_if_runnable": (
            "External review: do not shorten the plateau to save a little time. "
            "Q1a remains 1+4+1 if the machine can run it."
        ),
    }


def recommend(ext, q0_ok, fock_alarm, timed_out) -> dict:
    if timed_out or not q0_ok:
        verdict = "do_not_start_q1_q0_failed"
        why = "Q0 timed out or failed the numerical gate. Do not enter Q1."
    elif fock_alarm:
        verdict = "do_not_start_q1_fock_alarm"
        why = "Q0 top Fock already exceeded 0.05 on the rising-edge prefix."
    elif ext["q1a_from_q0_under_10min"]:
        verdict = "ask_to_approve_q1a_1_4_1"
        why = "Q0 extrapolation of four-way 1+4+1 is ≤10 min. Keep the four-cycle plateau."
    elif ext["q1a_from_q0_under_900s"]:
        verdict = "ask_to_approve_q1a_1_4_1_accept_about_11min"
        why = (
            "Q0 extrapolation of four-way 1+4+1 is between 10 min and 900 s. "
            "External review prefers keeping 1+4+1 rather than shrinking to 1+2+1."
        )
    else:
        verdict = "ask_user_q1a_too_slow_choose_q1b_or_q1c"
        why = (
            "Q0 extrapolation exceeds the 900 s Q1a budget. "
            "This script does not choose Q1b or Q1c."
        )
    return {
        "verdict": verdict,
        "why": why,
        "this_script_does_not_approve_q1": True,
        "start_q1a": False,
        "start_q1b": False,
        "start_q1c": False,
        "server_needed_now": False,
    }


def run() -> dict:
    t_all0 = time.perf_counter()
    hashes = source_hashes()
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=4.0)
    assert abs(env.t_total / env.period - 6.0) < 1e-12
    assert abs(env.t_ramp / env.period - 1.0) < 1e-12
    assert abs(env.t_flat / env.period - 4.0) < 1e-12
    plat = env.plateau
    s_mid = env.s(0.5 * (plat[0] + plat[1]))
    s_half = env.s(plat[0] + 0.5 * env.period)
    drive_half_res = abs(s_mid - 1.0) + abs(s_half - 1.0)

    rng = np.random.default_rng(20261120)
    model = sh.build_shared(geo, R0, k, NEEL, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    rho, rho_nrm = pf.spectral_radius_power(model, 1.0, rng)
    n_full = dyn.n_steps_for(env.t_total, rho, requested=32, hdt_max=pf.TAYLOR_HDT_MAX)
    dt = env.t_total / n_full
    t1 = Q0_STEPS * dt
    v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    v = v / np.linalg.norm(v)
    t_step0 = time.perf_counter()
    pf.taylor_step(model, v, 0.4, dt)
    t_step = time.perf_counter() - t_step0

    ek, emk, masses = odd.ground_pair_neel(geo, k, NEEL)
    raw = odd.photon_raw("coherent", NMAX)
    raw_diag = odd.photon_moments(raw)
    c = raw / np.sqrt(raw_diag["unnormalized_mass"])
    psi, prod_n = odd.product_normalized(ek, emk, c)
    s0 = pf.snapshot(model, psi, 0.0)

    progress(
        f"Q0 start dim={model.dim} n_full={n_full} dt={dt:.16f} "
        f"t1={t1:.6f} steps={Q0_STEPS} cap={WALL_CAP_S}s"
    )
    clock = dyn.WallClock(WALL_CAP_S)
    finite_ok = True
    err = None
    try:
        psi, led = propagate_prefix(model, psi, 0.0, t1, env, Q0_STEPS, clock, SAMPLE_EVERY)
    except Exception as exc:
        finite_ok = False
        err = f"{type(exc).__name__}: {exc}"
        led = {
            "steps_requested": Q0_STEPS,
            "steps_completed": 0,
            "dt": float(dt),
            "samples": [],
            "timed_out": False,
            "timeout_msg": None,
            "max_last_taylor_term": None,
            "work_trap_dHds_dsdt": None,
            "loop_seconds": clock.elapsed(),
            "t1_reached": 0.0,
        }
        progress(f"Q0 exception {err}")

    wall = clock.elapsed()
    samples = led.get("samples") or []
    send = None
    if finite_ok and samples:
        send = samples[-1]
    elif finite_ok:
        send = pf.snapshot(model, psi, env.s(led.get("t1_reached", t1)))

    tops = [s["top_fock_mass"] for s in samples]
    if send is not None:
        tops.append(send["top_fock_mass"])
    max_top = float(np.max(tops)) if tops else float(s0["top_fock_mass"])
    norms = [s["norm"] for s in samples]
    if send is not None:
        norms.append(send["norm"])
    max_dn = float(np.max(np.abs(np.asarray(norms) - 1.0))) if norms else abs(s0["norm"] - 1.0)
    fock_alarm = bool(max_top > FOCK_ALARM)
    timed_out = bool(led.get("timed_out"))
    completed = int(led.get("steps_completed") or 0)
    q0_ok = bool(finite_ok and completed > 0 and np.isfinite(max_top) and max_dn < 1e-6)
    ext = extrapolate(wall, completed if completed else 1, n_full)
    rec = recommend(ext, q0_ok and not timed_out, fock_alarm, timed_out)

    if q0_ok:
        status = "ok_prefix_only" if not timed_out and completed == Q0_STEPS else (
            "timeout_partial" if timed_out else "incomplete"
        )
    else:
        status = "failed"
    if fock_alarm:
        status = "fock_alarm"

    figures = write_figure(samples, env, OUT / "figure_q0_prefix")

    report = {
        "task_tag": TASK_TAG,
        "record_date": RECORD_DATE,
        "utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "status": status,
        "stopped": True,
        "claim_level": "q0_cost_and_initial_fock_not_spectrum_not_hhg",
        "did_not": [
            "Q1a", "Q1b", "Q1c", "MTEF", "GH5", "Cn", "theta=pi", "squeezed vacuum",
            "G1", "H2", "harmonic analysis of 400 steps", "magnetic odd current spectrum",
            "plateau Fock certificate", "classical Floquet", "D2", "Gamma192", "T2",
            "vacuum switch control job", "rewrite frozen archives",
        ],
        "physics_boundaries_not_tested_here": {
            "integer_Fourier_is_not_certified_HHG": True,
            "plateau_H_time_independent_but_mode_still_oscillates": True,
            "coupling_switch_can_prepare_transients": True,
            "same_alpha_is_not_same_nbar": True,
            "A_star_mean_not_classical_A_star_0p8": True,
            "q0_covers_about_0p31_cycle_only": True,
        },
        "definition": {
            "H": "omega0 n + He(k, g s(t) (a+a^dagger); N) + He(-k, same; N)",
            "s_t": "Envelope(ramp_cycles=1, flat_cycles=4): 1+4+1 sin^2 switch",
            "s_is_coupling_switch_not_free_space_pulse": True,
            "s_is_not_classical_Peierls_A": True,
            "n_max": NMAX,
            "g": pf.G_ABS,
            "omega0": pf.OMEGA0,
            "alpha": {"re": float(ALPHA.real), "im": float(ALPHA.imag)},
            "kind": "coherent",
            "neel": NEEL,
            "k_fractional": list(sh.K_FRAC),
            "mu_half_filling_documented": MU,
            "filling": "lowest 4 of 8 bands at each k, same as frozen QED",
            "sample_every": SAMPLE_EVERY,
            "taylor_order": pf.ORDER,
            "q0_steps": Q0_STEPS,
            "wall_cap_s": WALL_CAP_S,
            "fock_alarm": FOCK_ALARM,
            "period": env.period,
            "t_total": env.t_total,
            "t_ramp": env.t_ramp,
            "t_flat": env.t_flat,
            "plateau": list(plat),
            "a_nn_mean_ang": float(geo.a_nn_mean_ang),
            "A_star_mean": a_star_mean(geo),
            "classical_A_star": 0.8,
            "A_star_ratio_classical_over_mean": 0.8 / a_star_mean(geo),
        },
        "envelope_self_check": {
            "s_plateau_is_1": float(s_mid),
            "drive_half_period_residual_on_s": float(drive_half_res),
            "note": "s=1 on the plateau does not make the joint current periodic.",
        },
        "hilbert": {
            "dim": int(model.dim),
            "el36": int(sh.EL36),
            "d1": int(model.n_max + 1),
            "expected_11664": bool(model.dim == 11664),
        },
        "time_grid": {
            "rho_power_s1": float(rho),
            "rho_power_norm": float(rho_nrm),
            "n_full_1_4_1": int(n_full),
            "dt": float(dt),
            "historical_dt": HIST_DT,
            "dt_minus_historical": float(dt - HIST_DT),
            "q0_t1": float(t1),
            "q0_t1_over_T0": float(t1 / env.period),
            "timed_taylor_step_seconds": float(t_step),
            "do_not_quote_taylor_for_q1a": True,
        },
        "initial": {
            "electron": masses,
            "photon_unnormalized": raw_diag,
            "product_norm_before_renorm": prod_n,
            "snapshot_s0": s0,
        },
        "job": {
            "tag": "coh_p_q0",
            "kind": "coherent",
            "neel": NEEL,
            "ledger": {k: led[k] for k in led if k != "samples"},
            "endpoint": send,
            "max_top_fock": max_top,
            "max_abs_norm_minus_1": max_dn,
            "fock_alarm_0p05": fock_alarm,
            "finite_state": finite_ok,
            "error": err,
            "wall_seconds": wall,
            "seconds_per_step": (wall / completed) if completed else None,
            "historical_coh_p_max_top_fock": HIST_COH_P_TOP,
            "historical_ds_p_max_top_fock_one_cycle": HIST_DS_P_TOP,
            "q0_fock_does_not_certify_plateau": True,
        },
        "time_series": pf.pack_samples(samples) if samples else {},
        "extrapolation": ext,
        "recommendation": rec,
        "source_hashes": hashes,
        "old_compare_untouched": True,
        "old_compare_path": str(OLD_COMPARE.as_posix()),
        "figures": figures,
        "wall_seconds_process": None,
    }
    report["wall_seconds_process"] = time.perf_counter() - t_all0
    return report


def main(argv=None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = run()
    dump(OUT / "q0.json", report)
    summary = {
        "status": report["status"],
        "wall": report["job"]["wall_seconds"],
        "steps": report["job"]["ledger"].get("steps_completed"),
        "dt": report["time_grid"]["dt"],
        "max_top_fock": report["job"]["max_top_fock"],
        "four_way_from_q0_s": report["extrapolation"]["four_way_1_4_1_from_q0_times_hist_parallel_s"],
        "four_way_from_hist_s": report["extrapolation"]["four_way_1_4_1_from_hist_scale_s"],
        "verdict": report["recommendation"]["verdict"],
        "start_q1a": False,
    }
    dump(OUT / "q0_summary.json", summary)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
    return 0 if report["status"] in ("ok_prefix_only", "timeout_partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
