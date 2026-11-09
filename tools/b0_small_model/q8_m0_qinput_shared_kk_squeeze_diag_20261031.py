#!/usr/bin/env python3
"""Shared-mode vacuum-ramp quadrature postprocess + one nmax=12 cutoff test.

Reads 20261030 preflight.json (never rewritten). Not HHG, not output squeezing,
not a BSV pulse. No G1/H2, SV, BZ, or coherent scan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry

MODEL_ID = "M0_shared_mode_k_minus_k_squeeze_diag_v1"
RECORD_DATE = "2026-10-31"
WALL_CAP_S = 420.0
OLD_PREFLIGHT = Path(
    "results/q_line/qinput_a_shared_kk_dyn_preflight_20261030/preflight.json")
OLD_ARCHIVES = (
    "results/q_line/qinput_a_dynamics_trial_20261026/trial.json",
    "results/q_line/qinput_a_truncation_preflight_20261027/preflight.json",
    "results/q_line/qinput_a_j0_drive_diag_20261028/diag.json",
    "results/q_line/qinput_a_shared_kk_static_20261029/static.json",
    "results/q_line/qinput_a_shared_kk_dyn_preflight_20261030/preflight.json",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments_from_snap(snap: dict) -> dict:
    n_ph = float(snap["n_ph"])
    mean_a = pf.unpack_c(snap["mean_a"])
    mean_aa = pf.unpack_c(snap["mean_aa"])
    n_c = float(snap["n_incoherent"])
    m_c = pf.unpack_c(snap["m_incoherent"])
    p_n = np.asarray(snap["p_n"], float)
    n_from_pn = float(np.dot(np.arange(p_n.size), p_n))
    n_inc_from_a = n_ph - abs(mean_a) ** 2
    m_from_aa = mean_aa - mean_a ** 2
    n_max = int(p_n.size - 1)
    quad = pf.quadrature_from_moments(n_c, m_c, n_max=n_max, p_top=float(p_n[-1]))
    nu2 = (n_c + 0.5) ** 2 - abs(m_c) ** 2
    gauss_nu = float(np.sqrt(max(nu2, 0.0)))
    gauss_purity = float(1.0 / (2.0 * gauss_nu)) if gauss_nu > 0.0 else None
    return {
        "n_ph": n_ph,
        "n_from_p_n": n_from_pn,
        "n_c_archive": n_c,
        "n_c_from_n_minus_abs_a2": float(n_inc_from_a),
        "m_c_archive": pf.cpack(m_c),
        "m_c_from_aa_minus_a2": pf.cpack(m_from_aa),
        "p_n_sum": float(p_n.sum()),
        "p_n": [float(x) for x in p_n],
        "purity_archive": float(snap["purity"]),
        "trace_photon": float(snap.get("trace_photon", snap.get("trace", 1.0))),
        "norm": float(snap["norm"]),
        "energy": float(snap["energy"]),
        "s": float(snap["s"]),
        "max_abs_n_ph_minus_sum_n_pn": abs(n_ph - n_from_pn),
        "max_abs_n_c_minus_rebuild": abs(n_c - n_inc_from_a),
        "max_abs_m_c_minus_rebuild": abs(m_c - m_from_aa),
        "gaussian_symplectic_nu_from_moments": gauss_nu,
        "gaussian_purity_from_moments": gauss_purity,
        "purity_minus_gaussian": (
            None if gauss_purity is None else float(snap["purity"]) - gauss_purity),
        "quadrature": quad,
        "rho_ph_not_in_archive": True,
    }


def trap_work(t, power):
    t = np.asarray(t, float)
    p = np.asarray(power, float)
    if t.size < 2:
        return 0.0
    return float(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(t)))


def work_audit(driven: dict) -> dict:
    ts = driven["time_series"]
    t = np.asarray(ts["t"], float)
    power = np.asarray(ts["power_dHds_dsdt"], float)
    energy = np.asarray(ts["energy"], float)
    work_ts = trap_work(t, power)
    work_arch = float(driven["work_trap_dHds_dsdt"])
    dE = float(driven["delta_energy"])
    n_samp = int(t.size)
    n_steps = (int(driven["prefix_ledger"]["steps"])
               + int(driven["ramp_rest_ledger"]["steps"]))
    gap = dE - work_arch
    rel = abs(gap) / max(abs(dE), 1e-30)
    return {
        "n_samples": n_samp,
        "n_steps": n_steps,
        "sample_stride_nominal": n_steps // max(n_samp - 1, 1),
        "delta_energy": dE,
        "work_archive": work_arch,
        "work_recomputed_from_time_series": work_ts,
        "energy_minus_work": float(driven["energy_minus_work"]),
        "gap_recomputed": dE - work_ts,
        "relative_gap": rel,
        "first_t": float(t[0]),
        "last_t": float(t[-1]),
        "t_ramp": float(driven["t_ramp"]),
        "power_peak": float(np.max(np.abs(power))),
        "note": (
            "∫⟨∂H/∂s⟩ṡ dt uses ~n_steps/8 samples, not every Taylor step. "
            "The ~1% ΔE−work gap is consistent with trapezoid undersampling "
            "of a smooth power hump, not booked dissipation. "
            "Denser sampling needs a new propagation; cost is proposed, not run."
        ),
        "dense_sample_cost_not_run": {
            "nmax8_sample_every_1": (
                "re-propagate 1210 steps + snapshot/dHds each step, "
                "~1.2–1.4× the 107 s nmax=8 wall, ≈130–150 s"
            ),
            "nmax12_sample_every_1_same_dt": (
                "same 1210 steps at dim 16848, ≈180–220 s"
            ),
        },
    }


def joint_inversion_map() -> dict:
    return {
        "intertwiner_U": "(P⊗P) Π_γ",
        "hamiltonian": "U H(N,g) U^{-1} = Swap H(-N, same g)",
        "do_not_flip_g_again": True,
        "state": (
            "If i∂t|ψ_N⟩=H(N,g)|ψ_N⟩ then |ψ_{-N}⟩ := Swap U |ψ_N⟩ "
            "solves i∂t|ψ_{-N}⟩=H(-N,g)|ψ_{-N}⟩ in unswapped (k,-k) order."
        ),
        "operators": {
            "Pi a Pi": "-a",
            "Pi n Pi": "n",
            "Pi X Pi": "-X",
            "Pi aa Pi": "aa  (m_c even under photon parity)",
            "P_k on each electronic factor": "orbital inversion",
            "Swap": "k ↔ -k electronic factors",
        },
        "photon_moments_under_U": {
            "n_c": "invariant under Π_γ; ±N comparison still needs the mapped electrons",
            "m_c": "invariant under a→-a; dressing can still change with N",
            "W_sq": "not automatically N-even until a -N run exists",
            "mean_a": "Π_γ flips sign; product-vacuum at t=0 is even",
        },
        "current_data": "+N vacuum ramp only; no magnetic-order selectivity claimed",
    }


def turnoff_design(nmax8_wall_s: float, nmax12_est_s: float) -> dict:
    return {
        "purpose": (
            "After the coupled s=1 measurement, smoothly take s(t)→0 and "
            "read the free photon mode. Distinguishes interaction dressing "
            "from a leftover field that survives decoupling. Still not a "
            "free-space radiation experiment."
        ),
        "not_executed_this_round": True,
        "protocol": {
            "envelope": "reuse 20261026 Envelope sin^2",
            "option_A_down_only": "1-cycle up (already have) + 1-cycle down, skip plateau",
            "option_B_full": "1-cycle up + 2-cycle plateau + 1-cycle down = 4× ramp",
            "readout": "s=0 snapshot: n_c, m_c, W_sq, purity, p_n, commutator deficit",
            "optional_free_photon": "exact_constant_h at s=0, gX off",
        },
        "cost": {
            "nmax8_down_only_s": nmax8_wall_s,
            "nmax8_full_envelope_s": 4.0 * nmax8_wall_s,
            "nmax12_down_only_same_dt_s": nmax12_est_s,
            "nmax12_full_envelope_s": 4.0 * nmax12_est_s,
            "recommendation": (
                "One nmax=8 down-ramp, same dt as 20261030, wall ~nmax8 ramp. "
                "Do not scan parameters. Stop the vacuum-squeezing branch after that."
            ),
        },
    }


def postprocess_archive(path: Path) -> dict:
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    driven = data["dynamics"]["driven"]
    snaps = {
        "initial": moments_from_snap(driven["initial"]),
        "after_prefix": moments_from_snap(driven["after_prefix"]),
        "after_ramp": moments_from_snap(driven["after_ramp"]),
    }
    ts = driven["time_series"]
    w_series = []
    for n_c, m in zip(ts["n_incoherent"], ts["m_incoherent"]):
        q = pf.quadrature_from_moments(n_c, pf.unpack_c(m), n_max=8, p_top=0.0)
        w_series.append(q["W_sq"])
    q_end = snaps["after_ramp"]["quadrature"]
    return {
        "source": str(path).replace("\\", "/"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "model_id": data["model_id"],
        "record_date": data["record_date"],
        "n_max": driven["n_max"],
        "g": driven["g"],
        "reached_ramp_end": driven["reached_ramp_end"],
        "snaps": snaps,
        "W_sq_time_series": w_series,
        "headline_end": {
            "n_c": q_end["n_c"],
            "m_c": q_end["m_c"],
            "abs_m": q_end["abs_m"],
            "W_sq": q_end["W_sq"],
            "var_min": q_end["var_min"],
            "squeeze_fraction_vs_vacuum_I": q_end["squeeze_fraction_vs_vacuum_I"],
            "squeeze_dB_vs_vacuum_I": q_end["squeeze_dB_vs_vacuum_I"],
            "classical_P_bound_violated": q_end["classical_P_bound_violated"],
            "purity": snaps["after_ramp"]["purity_archive"],
            "commutator_deficit": q_end["commutator_deficit"],
        },
        "work": work_audit(driven),
        "interpretation": (
            "Bare photon mode at s=1, still coupled to both electronic "
            "subsystems. Candidate dressing/envelope excitation, not a "
            "propagating output squeezed field."
        ),
    }


def compare_cutoffs(a: dict, b: dict) -> dict:
    keys_p = min(len(a["p_n"]), len(b["p_n"]))
    qa, qb = a["quadrature"], b["quadrature"]
    def rel(x, y):
        return float(abs(x - y) / max(abs(x), abs(y), 1e-30))
    return {
        "W_sq": {"nmax8": qa["W_sq"], "nmax12": qb["W_sq"],
                 "abs_diff": abs(qa["W_sq"] - qb["W_sq"]),
                 "rel_diff": rel(qa["W_sq"], qb["W_sq"])},
        "n_c": {"nmax8": qa["n_c"], "nmax12": qb["n_c"],
                "abs_diff": abs(qa["n_c"] - qb["n_c"]),
                "rel_diff": rel(qa["n_c"], qb["n_c"])},
        "abs_m": {"nmax8": qa["abs_m"], "nmax12": qb["abs_m"],
                  "abs_diff": abs(qa["abs_m"] - qb["abs_m"]),
                  "rel_diff": rel(qa["abs_m"], qb["abs_m"])},
        "purity": {"nmax8": a["purity_archive"], "nmax12": b["purity_archive"],
                   "abs_diff": abs(a["purity_archive"] - b["purity_archive"])},
        "energy": {"nmax8": a["energy"], "nmax12": b["energy"],
                   "abs_diff": abs(a["energy"] - b["energy"])},
        "p_n_low": {
            "nmax8": a["p_n"][:keys_p],
            "nmax12": b["p_n"][:keys_p],
            "max_abs_diff": float(np.max(np.abs(
                np.asarray(a["p_n"][:keys_p]) - np.asarray(b["p_n"][:keys_p])))),
        },
        "both_violate_classical_P": bool(
            qa["classical_P_bound_violated"] and qb["classical_P_bound_violated"]),
        "both_W_sq_negative": bool(qa["W_sq"] < 0.0 and qb["W_sq"] < 0.0),
    }


def run_nmax12(n_steps: int, clock: dyn.WallClock, progress=None) -> dict:
    geo = build_geometry()
    k = geo.k_cart(pf.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=2.0)
    ek, emk, masses = pf.ground_pair(geo, k)
    _progress(progress, "building nmax=12 shared model")
    model = sh.build_shared(
        geo, R0, k, pf.NEEL, n_max=12, omega0=pf.OMEGA0, g=pf.G_ABS)
    t0 = time.perf_counter()
    v = np.zeros(model.dim, complex)
    v[0] = 1.0
    _out, probe_last = pf.taylor_step(model, v, 0.4, env.t_ramp / n_steps)
    t_step = time.perf_counter() - t0
    est = t_step * n_steps
    out = {
        "n_max": 12,
        "dim": model.dim,
        "n_steps": n_steps,
        "dt": env.t_ramp / n_steps,
        "timed_taylor_step_seconds": t_step,
        "probe_max_last_taylor_term": float(probe_last),
        "estimated_ramp_seconds": est,
        "electron_slater": masses,
        "attempted": False,
        "reason": None,
    }
    if probe_last > 1e-8:
        out["reason"] = (
            f"Taylor remainder {probe_last:.2e} at dt={env.t_ramp / n_steps:.4f} "
            "too large; same-dt cutoff comparison is not trustworthy"
        )
        return out
    if est > WALL_CAP_S:
        out["reason"] = f"estimate {est:.1f}s exceeds {WALL_CAP_S:.0f}s cap"
        _progress(progress, out["reason"])
        return out
    psi, raw_n = pf.vacuum_product(model, ek, emk)
    s0 = pf.snapshot(model, psi, 0.0)
    out["attempted"] = True
    _progress(progress, f"propagate nmax=12 n_steps={n_steps} est={est:.1f}s")
    try:
        psi, led = pf.propagate(
            model, psi, 0.0, env.t_ramp, env, n_steps, clock,
            sample_every=max(1, n_steps // 8))
        snap_f = pf.snapshot(model, psi, 1.0)
    except TimeoutError as exc:
        out["reason"] = str(exc)
        out["wall_seconds"] = clock.elapsed()
        _progress(progress, f"timeout {exc}")
        return out
    mom0 = moments_from_snap(s0)
    momf = moments_from_snap(snap_f)
    work = float(led["work_trap_dHds_dsdt"])
    dE = momf["energy"] - mom0["energy"]
    _progress(progress, "nmax=12 ramp finished")
    out.update({
        "input_raw_product_norm": raw_n,
        "initial": mom0,
        "after_ramp": momf,
        "after_ramp_raw_snapshot_keys": sorted(snap_f),
        "delta_energy": dE,
        "work_trap_dHds_dsdt": work,
        "energy_minus_work": dE - work,
        "time_series": pf.pack_samples(led["samples"]),
        "ledger": {k: led[k] for k in ("steps", "dt", "order", "max_last_taylor_term")},
        "wall_seconds": clock.elapsed(),
        "reason": None,
    })
    return out


def _progress(progress, msg):
    if progress is None:
        return
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def run() -> dict:
    old = Path(OLD_PREFLIGHT)
    hashes_before = {p: sha256_file(Path(p)) for p in OLD_ARCHIVES}
    post = postprocess_archive(old)
    nmax8_wall = 107.3052408999647
    n_steps8 = 1210
    progress = Path("results/q_line/qinput_a_shared_kk_squeeze_diag_20261031/progress.txt")
    _progress(progress, "archive postprocess done")
    clock = dyn.WallClock(WALL_CAP_S)
    n12 = run_nmax12(n_steps8, clock, progress=progress)
    cmp = None
    if n12.get("after_ramp") is not None:
        cmp = compare_cutoffs(post["snaps"]["after_ramp"], n12["after_ramp"])
    hashes_after = {p: sha256_file(Path(p)) for p in OLD_ARCHIVES}
    untouched = {p: hashes_before[p] == hashes_after[p] for p in OLD_ARCHIVES}
    n12_est = n12.get("estimated_ramp_seconds") or (nmax8_wall * 16848 / 11664)
    _progress(progress, "compare done")
    return {
        "record_date": RECORD_DATE,
        "model_id": MODEL_ID,
        "claim_level": (
            "bare_photon_quadrature_cutoff_check_not_output_squeezing_not_HHG"
        ),
        "not_free_space_output_squeezing": True,
        "measured_while_s_equals_1_still_coupled": True,
        "did_not": [
            "rewrite trial.json", "rewrite 20261027 preflight.json",
            "rewrite diag.json", "rewrite static.json",
            "rewrite 20261030 preflight.json",
            "G1", "H2", "HHG spectra", "SV scan", "coherent scan",
            "semiclassical trajectories", "full BZ", "AI inverse design",
            "s(t)→0 turn-off", "-N run",
        ],
        "definition": {
            "X_theta": "(a exp(-iθ)+a† exp(iθ))/2",
            "W_sq": "n_c - |m_c|",
            "var_min": "1/4 + W_sq/2 with truncated commutator recorded separately",
            "n_c": "<Δa† Δa>",
            "m_c": "<Δa Δa>",
            "dt_choice": (
                "nmax=12 uses the same 1210 steps / dt≈0.104 as nmax=8, "
                "to isolate Fock cutoff. n_steps_for(ρ12) would be 3229 "
                "because power-iteration ρ jumped 3.85→10.28; Taylor remainder "
                "at nmax=8 was ~5e-13, so dt is already small. Time-step "
                "refinement is a later independent run."
            ),
        },
        "archive_postprocess": post,
        "nmax12": n12,
        "cutoff_compare": cmp,
        "joint_inversion_map": joint_inversion_map(),
        "turnoff_design": turnoff_design(
            nmax8_wall, float(n12_est)),
        "old_archives_untouched": untouched,
        "old_archive_sha256": hashes_after,
        "files": {
            Path(p).name: sha256_file(Path(p))
            for p in (__file__, pf.__file__, sh.__file__, dyn.__file__)
        },
        "stopped": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report", type=Path,
        default=Path("results/q_line/qinput_a_shared_kk_squeeze_diag_20261031/diag.json"))
    args = ap.parse_args(argv)
    if args.report.exists():
        print(f"report already exists: {args.report}", file=__import__("sys").stderr)
        return 2
    t0 = time.perf_counter()
    report = run()
    report["wall_seconds_total"] = time.perf_counter() - t0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    cmp = report.get("cutoff_compare")
    n12 = report["nmax12"]
    head = report["archive_postprocess"]["headline_end"]
    print(json.dumps({
        "model_id": report["model_id"],
        "nmax8_W_sq": head["W_sq"],
        "nmax8_var_min": head["var_min"],
        "nmax8_purity": head["purity"],
        "nmax12_attempted": n12["attempted"],
        "nmax12_reason": n12.get("reason"),
        "nmax12_W_sq": None if not n12.get("after_ramp") else n12["after_ramp"]["quadrature"]["W_sq"],
        "nmax12_purity": None if not n12.get("after_ramp") else n12["after_ramp"]["purity_archive"],
        "W_sq_rel_diff": None if cmp is None else cmp["W_sq"]["rel_diff"],
        "purity_abs_diff": None if cmp is None else cmp["purity"]["abs_diff"],
        "both_W_sq_negative": None if cmp is None else cmp["both_W_sq_negative"],
        "old_untouched": all(report["old_archives_untouched"].values()),
        "wall": report["wall_seconds_total"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
