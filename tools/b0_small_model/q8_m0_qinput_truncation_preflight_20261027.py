#!/usr/bin/env python3
"""Short Fock-truncation preflight at weak g. Not HHG, not a BSV pulse.

Independent archive. Does not rewrite
results/q_line/qinput_a_dynamics_trial_20261026/trial.json.
Does not run G1/H2, k,-k, semiclassical sampling, or displaced-squeezed scans.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn

G_DIAG = 0.005
NMAX_PAIR = (8, 12)
KIND = "coherent"
T_FRACTION_OF_RAMP = 1.0
WALL_LIMIT = 300.0
TOP_TWO_ALARM = 0.05
MASS_GT8_ALARM = 0.02
JJ_TIMES = 3

STEP1 = {
    "old_trial_json": "results/q_line/qinput_a_dynamics_trial_20261026/trial.json",
    "old_trial_untouched": True,
    "nopump_interface_now": (
        "source_spectra(..., s_flat=) selects the eigenbasis of H(s_flat). "
        "Driven plateau must use s_flat=1 with J(s=1). No-pump reference must "
        "use a state evolved under H(s=0) and s_flat=0 with J(s=0). "
        "Archived nopump_flat_spectra still used H(s=1); do not cite it."
    ),
    "source_spectrum_convention": (
        "Jhat_q = int_0^T w(t) e^{+i q omega0 t} J(t) dt, Hann w, q=0..8. "
        "This is the +q Fourier slot of the source current, not the optical "
        "e^{-i omega t} detector convention and not a radiated far-field. "
        "S_q = <Jhat^dag Jhat> in that same slot; connected = S - mean^* mean. "
        "Positive and negative q are different slots; this driver only stores q>=0. "
        "Equal-time <J^dag J> is not S_q."
    ),
    "extract_from_archive": [
        "zero-coupling conservation",
        "ramp time series of n_ph / top_two_mass / norm / <J> at g=0.02",
        "wall times and dimensions",
        "driven-flat S_q only as a truncated-model number, not physics",
    ],
    "must_repropagate": [
        "any no-pump background spectrum",
        "Fock convergence of observables",
        "electron / photon / hop energy split",
        "equal-time current correlators at g!=0.02",
        "anything at g=0.005 or n_max!=8",
    ],
    "peierls": (
        "Propagator keeps Galerkin P exp(i q s X_inf) P. It is not exp(i q P X P). "
        "Photon surge at g=0.02 is not explained by a silent operator swap. "
        "q = g e·delta; s(t) scales the Peierls argument, not a classical A(t) pulse."
    ),
}


def observe(model: dyn.JointModel, psi: np.ndarray, s: float, Ds=None) -> dict:
    snap = dyn.snapshot(model, psi, s, Ds=Ds)
    d1 = model.d1
    psi = np.asarray(psi, complex)
    e_ph = model.omega0 * snap["n_ph"]
    e_diag = float(np.vdot(psi, model.diag * psi).real)
    e_tot = snap["energy"]
    p_n = np.asarray(snap["p_n"], float)
    Jv = dyn.apply_current(model, psi, s, Ds=Ds)
    jj = np.array([[np.vdot(Jv[a], Jv[b]) for b in range(2)] for a in range(2)])
    mean = np.array([snap["mean_J_re"][a] + 1j * snap["mean_J_im"][a] for a in range(2)])
    conn = jj - np.outer(mean.conj(), mean)
    return {
        "s": snap["s"],
        "norm": snap["norm"],
        "n_ph": snap["n_ph"],
        "top_two_mass": snap["top_two_mass"],
        "mass_n_gt_8": float(p_n[9:].sum()) if d1 > 9 else 0.0,
        "p_n": p_n.tolist(),
        "e_total": e_tot,
        "e_photon_number_term": e_ph,
        "e_el_onsite": e_diag - e_ph,
        "e_hop_mixed": e_tot - e_diag,
        "mean_J_re": snap["mean_J_re"],
        "mean_J_im": snap["mean_J_im"],
        "JJ_re": np.real(jj).tolist(),
        "connected_trace": float(np.trace(conn).real),
        "photon_even_mass": snap["photon_even_mass"],
    }


def propagate_window(model, psi, env, t1, n_steps, clock, n_record=17):
    dt = t1 / n_steps
    rec = sorted(set([0] + [min(n_steps, round(i * n_steps / (n_record - 1)))
                            for i in range(n_record)]))
    rec_set = set(rec)
    samples = [dict(observe(model, psi, env.s(0.0)), t=0.0)]
    worst = 0.0
    e0 = samples[0]["e_total"]
    for j in range(n_steps):
        clock.check()
        tm = (j + 0.5) * dt
        s = env.s(tm)
        Ds = dyn.peierls_at(model.q_units, s, model.n_max)
        psi, hdt, last = dyn.taylor_step(model, psi, s, dt, order=16, Ds=Ds)
        worst = max(worst, hdt)
        if not np.isfinite(psi).all():
            raise RuntimeError(f"non-finite at step {j} last_term={last}")
        if (j + 1) in rec_set:
            t = (j + 1) * dt
            obs = observe(model, psi, env.s(t))
            obs["t"] = t
            obs["external_work"] = obs["e_total"] - e0
            samples.append(obs)
    return psi, samples, {"steps": n_steps, "dt": dt, "max_abs_Hshift_dt": worst}


def pack_series(samples):
    keys = ("t", "s", "norm", "n_ph", "top_two_mass", "mass_n_gt_8",
            "e_total", "e_photon_number_term", "e_el_onsite", "e_hop_mixed",
            "connected_trace", "photon_even_mass")
    tab = {k: [s[k] for s in samples] for k in keys}
    e0 = samples[0]["e_total"]
    tab["external_work"] = [s["e_total"] - e0 for s in samples]
    tab["mean_J_re"] = [s["mean_J_re"] for s in samples]
    return tab


def jj_at(samples, n=JJ_TIMES):
    idx = np.unique(np.linspace(0, len(samples) - 1, n, dtype=int))
    out = []
    for i in idx:
        s = samples[i]
        out.append({"t": s["t"], "s": s["s"], "mean_J_re": s["mean_J_re"],
                    "JJ_re": s["JJ_re"], "connected_trace": s["connected_trace"]})
    return out


def compare(a, b):
    ta, tb = np.array(a["t"]), np.array(b["t"])
    if ta.shape != tb.shape or np.max(np.abs(ta - tb)) > 1e-9:
        return {"times_match": False}
    dn = np.abs(np.array(a["n_ph"]) - np.array(b["n_ph"]))
    dj = np.max(np.abs(np.array(a["mean_J_re"]) - np.array(b["mean_J_re"])), axis=1)
    de = np.abs(np.array(a["e_total"]) - np.array(b["e_total"]))
    return {
        "times_match": True,
        "max_abs_delta_n_ph": float(dn.max()),
        "max_abs_delta_mean_J": float(dj.max()),
        "max_abs_delta_e_total": float(de.max()),
        "delta_n_ph": dn.tolist(),
        "delta_mean_J_max_over_xy": dj.tolist(),
    }


def run_one(n_max, g, env, t1, n_steps, clock):
    _, _, model, occupied = dyn.build_trial_model(n_max, g=g)
    coeff, meta = dyn.photon_input(KIND, n_max, dyn.TRIAL_NBAR)
    psi0 = dyn.product_state(model, occupied, coeff)
    rho = dyn.spectral_radius(model, 1.0)
    psi, samples, led = propagate_window(model, psi0, env, t1, n_steps, clock)
    series = pack_series(samples)
    return {
        "n_max": n_max,
        "dim": model.dim,
        "g": g,
        "rho": rho,
        "q_units": model.q_units.tolist(),
        "q_abs_max": float(np.max(np.abs(model.q_units))),
        "input": meta,
        "input_norm2": float(np.vdot(coeff, coeff).real),
        "ledger": led,
        "max_top_two_mass": float(max(series["top_two_mass"])),
        "max_n_ph": float(max(series["n_ph"])),
        "max_mass_n_gt_8": float(max(series["mass_n_gt_8"])),
        "final": {k: series[k][-1] for k in ("t", "s", "norm", "n_ph", "top_two_mass",
                                             "mass_n_gt_8", "e_total", "e_photon_number_term",
                                             "e_el_onsite", "e_hop_mixed", "external_work")},
        "time_series": series,
        "equal_time_JJ": jj_at(samples),
        "peierls": "P exp(i q s X_inf) P, not exp(i q P X P)",
    }, model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path,
                    default=Path("results/q_line/qinput_a_truncation_preflight_20261027/preflight.json"))
    ap.add_argument("--wall-seconds", type=float, default=WALL_LIMIT)
    args = ap.parse_args(argv)
    if args.report.exists():
        print(f"report already exists: {args.report}", file=sys.stderr)
        return 2

    env = dyn.Envelope(dyn.TRIAL_OMEGA, ramp_cycles=1.0, flat_cycles=2.0)
    t1 = T_FRACTION_OF_RAMP * env.t_ramp
    clock = dyn.WallClock(args.wall_seconds)
    t_wall0 = time.perf_counter()
    jobs = {}
    stopped_reason = "completed_pair"
    try:
        n_steps = 32
        for n_max in NMAX_PAIR:
            _, _, probe, _ = dyn.build_trial_model(n_max, g=G_DIAG)
            rho = dyn.spectral_radius(probe, 1.0)
            n_steps = max(n_steps, dyn.n_steps_for(t1, rho, requested=32))
        for n_max in NMAX_PAIR:
            jobs[str(n_max)], _ = run_one(n_max, G_DIAG, env, t1, n_steps, clock)
    except TimeoutError as exc:
        stopped_reason = f"wall_timeout: {exc}"
    except Exception as exc:
        stopped_reason = f"{type(exc).__name__}: {exc}"
        raise

    j8, j12 = jobs.get("8"), jobs.get("12")
    alarm = False
    comparison = None
    if j8 and j12:
        comparison = compare(j8["time_series"], j12["time_series"])
        alarm = (j8["max_top_two_mass"] >= TOP_TWO_ALARM
                 or j12["max_top_two_mass"] >= TOP_TWO_ALARM
                 or j12["max_mass_n_gt_8"] >= MASS_GT8_ALARM)
    found_region = bool(j8 and j12 and not alarm)
    report = {
        "record_date": "2026-10-27",
        "model_id": "M0_single_k_truncation_preflight_g0005_v1",
        "claim_level": "truncation_preflight_not_magnetic_HHG",
        "not_free_space_BSV_pulse": True,
        "g_is_diagnostic_not_physical_optimum": True,
        "g": G_DIAG,
        "n_max_pair": list(NMAX_PAIR),
        "kind": KIND,
        "t_window": t1,
        "t_window_note": "one switch-on ramp only; no plateau, no fall, no S_q",
        "k_fractional": dyn.K_FRAC,
        "omega0": dyn.TRIAL_OMEGA,
        "nbar_target": dyn.TRIAL_NBAR,
        "step1_readonly": STEP1,
        "jobs": jobs,
        "comparison_8_vs_12": comparison,
        "alarms": {
            "top_two_threshold": TOP_TWO_ALARM,
            "mass_n_gt_8_threshold": MASS_GT8_ALARM,
            "triggered": alarm,
        },
        "found_convergent_region": found_region,
        "stopped_reason": stopped_reason,
        "did_not": ["rewrite trial.json", "G1", "H2", "k,-k", "semiclassical",
                    "displaced squeezed scan", "HHG interpretation"],
        "files": {
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (dyn.__file__, __file__)
        },
        "wall_seconds_total": time.perf_counter() - t_wall0,
        "stopped": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    print(json.dumps({
        "found_convergent_region": found_region,
        "alarm": alarm,
        "wall": report["wall_seconds_total"],
        "max_top_two": {k: jobs[k]["max_top_two_mass"] for k in jobs},
        "max_n_ph": {k: jobs[k]["max_n_ph"] for k in jobs},
        "mass_gt8_12": None if not j12 else j12["max_mass_n_gt_8"],
        "comparison": comparison,
        "stopped_reason": stopped_reason,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
