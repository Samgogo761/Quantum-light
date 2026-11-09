#!/usr/bin/env python3
"""Shared-mode vacuum rise+fall close-out. Time-step check + s→0 readout.

One cycle up, one cycle down, nmax=8, no plateau. Not HHG, not output light,
not a BSV pulse. Does not rewrite old archives. No G1/H2, SV, BZ, -N, or
coherent/displaced-squeezed scan. Closes the vacuum squeeze branch.
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

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "M0_shared_mode_k_minus_k_vacuum_switch_v1"
RECORD_DATE = "2026-11-01"
WALL_CAP_S = 600.0
NMAX = 8
ARCHIVE_STEPS = 1210
ARCHIVE_DT = 0.10385430259800969
ARCHIVE_WALL_S = 107.3052408999647
DT_REFINE = 2
SAMPLE_EVERY = 8
OLD_ARCHIVES = tuple(
    ROOT / p for p in (
        "results/q_line/qinput_a_dynamics_trial_20261026/trial.json",
        "results/q_line/qinput_a_truncation_preflight_20261027/preflight.json",
        "results/q_line/qinput_a_j0_drive_diag_20261028/diag.json",
        "results/q_line/qinput_a_shared_kk_static_20261029/static.json",
        "results/q_line/qinput_a_shared_kk_dyn_preflight_20261030/preflight.json",
        "results/q_line/qinput_a_shared_kk_squeeze_diag_20261031/diag.json",
    )
)
OLD_PREFLIGHT = OLD_ARCHIVES[4]
CKPT_DIRS = (
    ROOT / "results/q_line/qinput_a_shared_kk_dyn_preflight_20261030",
    ROOT / "results/q_line/qinput_a_shared_kk_squeeze_diag_20261031",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _progress(progress, msg):
    if progress is None:
        return
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def find_joint_state_files(roots=CKPT_DIRS):
    hits = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.iterdir():
            if p.suffix.lower() in {".npy", ".npz"}:
                hits.append(str(p.as_posix()))
    return hits


def switch_envelope():
    return dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)


def qpack(snap):
    q = snap["quadrature"]
    return {
        "n_ph": float(snap["n_ph"]),
        "n_coherent": float(snap["n_coherent"]),
        "n_c": float(q["n_c"]),
        "m_c": q["m_c"],
        "abs_m": float(q["abs_m"]),
        "W_sq": float(q["W_sq"]),
        "var_min": float(q["var_min_if_commutator_1"]),
        "var_min_truncated_comm": float(q["var_min"]),
        "purity": float(snap["purity"]),
        "p_n": [float(x) for x in snap["p_n"]],
        "top_fock_mass": float(snap["top_fock_mass"]),
        "s": float(snap["s"]),
        "energy": float(snap["energy"]),
        "norm": float(snap["norm"]),
        "mean_a": snap["mean_a"],
        "trace_photon": float(snap["trace_photon"]),
        "classical_P_bound_violated": bool(q["classical_P_bound_violated"]),
        "squeeze_dB_vs_vacuum_I": q["squeeze_dB_vs_vacuum_I"],
        "not_free_space_output_squeezing": True,
    }


def archive_cycle1():
    driven = json.loads(OLD_PREFLIGHT.read_text(encoding="utf-8"))["dynamics"]["driven"]
    snap = driven["after_ramp"]
    n_c = float(snap["n_incoherent"])
    m_c = pf.unpack_c(snap["m_incoherent"])
    p_n = np.asarray(snap["p_n"], float)
    quad = pf.quadrature_from_moments(n_c, m_c, n_max=NMAX, p_top=float(p_n[-1]))
    return {
        "source": str(OLD_PREFLIGHT.as_posix()),
        "n_max": NMAX,
        "steps": int(driven["prefix_ledger"]["steps"]) + int(driven["ramp_rest_ledger"]["steps"]),
        "dt_prefix": float(driven["prefix_ledger"]["dt"]),
        "dt_rest": float(driven["ramp_rest_ledger"]["dt"]),
        "packed": qpack({**snap, "quadrature": quad}),
        "energy": float(snap["energy"]),
        "work_trap": float(driven["work_trap_dHds_dsdt"]),
        "delta_energy": float(driven["delta_energy"]),
        "energy_minus_work": float(driven["energy_minus_work"]),
    }


def rel_gap(a, b):
    return float(abs(a - b) / max(abs(a), abs(b), 1e-30))


def compare_cycle1(new_pack, arch):
    a = arch["packed"]
    keys = ("n_c", "abs_m", "W_sq", "purity", "var_min", "n_ph")
    out = {}
    for k in keys:
        out[k] = {
            "new": float(new_pack[k]),
            "archive": float(a[k]),
            "abs_diff": float(abs(new_pack[k] - a[k])),
            "rel_diff": rel_gap(new_pack[k], a[k]),
        }
    w = out["W_sq"]["rel_diff"]
    n = out["n_c"]["rel_diff"]
    p = out["purity"]["abs_diff"]
    out["time_step_pass"] = bool(w < 1e-4 and n < 1e-4 and p < 1e-6)
    out["note"] = (
        "Same envelope rise, nmax=8, dt halved vs 20261030. "
        "Pass = relative W_sq and n_c gaps < 1e-4 and purity abs < 1e-6. "
        "Not a Fock-cutoff test."
    )
    return out


def decouple_check(model, psi, geo, k):
    h = sh.apply_h(model, psi, 0.0)
    m0 = sh.build_shared(geo, R0, k, pf.NEEL, n_max=model.n_max, omega0=pf.OMEGA0, g=0.0)
    h0 = sh.apply_h(m0, psi, 0.0)
    d1 = model.d1
    n_g = int(model.q_units.size)
    DsI = np.repeat(np.eye(d1, dtype=complex)[None, :, :], n_g, axis=0) if n_g else None
    hI = sh.apply_h(model, psi, 0.0, Ds=DsI)
    Ds0 = dyn.peierls_at(model.q_units, 0.0, model.n_max) if n_g else np.zeros((0, d1, d1), complex)
    eye = np.eye(d1, dtype=complex)
    peierls_err = 0.0 if n_g == 0 else float(np.max(np.abs(Ds0 - eye)))
    g_err = float(np.max(np.abs(h - h0)))
    i_err = float(np.max(np.abs(h - hI)))
    return {
        "s": 0.0,
        "max_abs_H_s0_g_vs_g0": g_err,
        "max_abs_H_s0_vs_identity_D": i_err,
        "max_abs_peierls_s0_minus_I": peierls_err,
        "strictly_decoupled": bool(g_err < 1e-12 and i_err < 1e-12 and peierls_err < 1e-15),
        "note": "s=0 forces D(qsX)=I; H = He(k)+He(-k)+ω0 n, independent of g.",
    }


def estimate_cost(model, env, n_total, rng):
    dt = env.t_ramp / (ARCHIVE_STEPS * DT_REFINE)
    t0 = time.perf_counter()
    v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    v = v / np.linalg.norm(v)
    _, last = pf.taylor_step(model, v, 0.4, dt)
    t_step = time.perf_counter() - t0
    scaled = ARCHIVE_WALL_S * (n_total / ARCHIVE_STEPS)
    timed = t_step * n_total
    return {
        "timed_taylor_step_seconds": t_step,
        "probe_max_last_taylor_term": float(last),
        "n_total": n_total,
        "dt": dt,
        "estimated_from_probe_seconds": timed,
        "estimated_from_20261030_scale_seconds": scaled,
        "estimated_seconds": min(timed, scaled) if timed > 0.0 else scaled,
        "wall_cap_s": WALL_CAP_S,
        "exceeds_10min": bool(min(timed, scaled) > WALL_CAP_S),
    }


def run(progress=None):
    hashes_before = {str(p.as_posix()): sha256_file(p) for p in OLD_ARCHIVES}
    ckpt = find_joint_state_files()
    _progress(progress, f"checkpoints={ckpt}")
    env = switch_envelope()
    n_cycle = ARCHIVE_STEPS * DT_REFINE
    n_total = 2 * n_cycle
    geo = build_geometry()
    k = geo.k_cart(pf.K_FRAC)
    rng = np.random.default_rng(20261101)
    ek, emk, masses = pf.ground_pair(geo, k)
    model = sh.build_shared(
        geo, R0, k, pf.NEEL, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    cost = estimate_cost(model, env, n_total, rng)
    _progress(progress, f"est={cost['estimated_seconds']:.1f}s n_total={n_total} dt={cost['dt']}")
    arch = archive_cycle1()
    report = {
        "record_date": RECORD_DATE,
        "model_id": MODEL_ID,
        "claim_level": "vacuum_switch_timestep_check_not_output_squeezing_not_HHG",
        "not_free_space_output_squeezing": True,
        "vacuum_squeeze_branch_closed": True,
        "did_not": [
            "rewrite trial.json", "rewrite 20261027 preflight.json",
            "rewrite 20261028 diag.json", "rewrite static.json",
            "rewrite 20261030 preflight.json", "rewrite 20261031 diag.json",
            "G1", "H2", "HHG spectra", "SV scan", "coherent scan",
            "displaced squeezed scan", "semiclassical trajectories",
            "full BZ", "AI inverse design", "-N dynamics", "nmax=12",
        ],
        "definition": {
            "H": "omega0 n + He(k, X; N) + He(-k, X; N)",
            "envelope": "1-cycle sin^2 up, 0 plateau, 1-cycle sin^2 down, s(T)=0",
            "n_max": NMAX,
            "g": pf.G_ABS,
            "omega0": pf.OMEGA0,
            "neel": pf.NEEL,
            "k_fractional": list(pf.K_FRAC),
            "dt_refine": DT_REFINE,
            "archive_steps_one_cycle": ARCHIVE_STEPS,
            "archive_dt": ARCHIVE_DT,
            "sample_every": SAMPLE_EVERY,
        },
        "joint_inversion_domain_even": {
            "statement": (
                "For electronic initial states related by the certified joint map "
                "U=(P⊗P)Π_γ and a parity-even vacuum photon input, "
                "n_c(-N)=n_c(+N), m_c(-N)=m_c(+N), W_sq(-N)=W_sq(+N). "
                "Photon parity sends a→-a, n→n, aa→aa. No -N run is required "
                "to establish this evenness, and this vacuum squeeze cannot "
                "read the Néel sign."
            ),
            "ran_minus_N": False,
        },
        "checkpoint_search": {
            "roots": [str(p.as_posix()) for p in CKPT_DIRS],
            "npy_npz": ckpt,
            "restart_from_t0": True,
            "reason": "no recoverable joint state; photon moments cannot rebuild psi",
        },
        "cost_estimate": cost,
        "electron_slater": masses,
        "envelope": {
            "t_ramp": env.t_ramp,
            "t_flat": env.t_flat,
            "t_total": env.t_total,
            "s_0": env.s(0.0),
            "s_tramp": env.s(env.t_ramp),
            "s_total": env.s(env.t_total),
        },
        "archive_cycle1": arch,
        "attempted": False,
        "reason": None,
    }
    if cost["probe_max_last_taylor_term"] > 1e-8:
        report["reason"] = "Taylor remainder too large at refined dt"
        report["stopped"] = True
        return report
    if cost["exceeds_10min"]:
        report["reason"] = (
            f"estimate {cost['estimated_seconds']:.1f}s exceeds {WALL_CAP_S:.0f}s; paused"
        )
        report["stopped"] = True
        _progress(progress, report["reason"])
        return report

    psi, raw_n = pf.vacuum_product(model, ek, emk)
    s0 = pf.snapshot(model, psi, 0.0)
    clock = dyn.WallClock(WALL_CAP_S)
    report["attempted"] = True
    pack1 = None
    cmp1 = None
    try:
        _progress(progress, "cycle1 rise")
        psi, led1 = pf.propagate(
            model, psi, 0.0, env.t_ramp, env, n_cycle, clock, SAMPLE_EVERY)
        snap1 = pf.snapshot(model, psi, env.s(env.t_ramp))
        pack1 = qpack(snap1)
        cmp1 = compare_cycle1(pack1, arch)
        report["after_rise"] = pack1
        report["cycle1_vs_20261030"] = cmp1
        _progress(progress, f"cycle1 done W_sq={pack1['W_sq']:.12g} pass={cmp1['time_step_pass']}")
        _progress(progress, "cycle2 fall")
        psi, led2 = pf.propagate(
            model, psi, env.t_ramp, env.t_total, env, n_cycle, clock, SAMPLE_EVERY)
        snap2 = pf.snapshot(model, psi, env.s(env.t_total))
        pack2 = qpack(snap2)
        dec = decouple_check(model, psi, geo, k)
        _progress(progress, f"cycle2 done W_sq={pack2['W_sq']:.12g} decoupled={dec['strictly_decoupled']}")
    except TimeoutError as exc:
        report["reason"] = str(exc)
        report["wall_seconds"] = clock.elapsed()
        report["stopped"] = True
        _progress(progress, f"timeout {exc}")
        hashes_after = {str(p.as_posix()): sha256_file(p) for p in OLD_ARCHIVES}
        report["old_archives_untouched"] = {
            k: hashes_before[k] == hashes_after[k] for k in hashes_before
        }
        if pack1 is not None:
            report["answers"] = {
                "cycle1_timestep_pass": None if cmp1 is None else cmp1["time_step_pass"],
                "squeeze_after_decouple": None,
                "purity_after_rise": pack1["purity"],
                "W_sq_after_rise": pack1["W_sq"],
            }
        return report

    samples = list(led1["samples"]) + list(led2["samples"])
    work = float(led1["work_trap_dHds_dsdt"] + led2["work_trap_dHds_dsdt"])
    dE = float(snap2["energy"] - s0["energy"])
    dE1 = float(snap1["energy"] - s0["energy"])
    work1 = float(led1["work_trap_dHds_dsdt"])
    hashes_after = {str(p.as_posix()): sha256_file(p) for p in OLD_ARCHIVES}
    report.update({
        "input_raw_product_norm": raw_n,
        "initial": qpack(s0),
        "after_rise": pack1,
        "after_fall": pack2,
        "cycle1_vs_20261030": cmp1,
        "decouple_at_s0": dec,
        "delta_energy_rise": dE1,
        "work_trap_rise": work1,
        "energy_minus_work_rise": dE1 - work1,
        "delta_energy_full": dE,
        "work_trap_full": work,
        "energy_minus_work_full": dE - work,
        "work_note": (
            "ΔE−∫(ds/dt)⟨∂H/∂s⟩dt uses sample_every=8. "
            "Do not book a residual as dissipation."
        ),
        "ledgers": {
            "rise": {k: led1[k] for k in ("steps", "dt", "order", "max_last_taylor_term")},
            "fall": {k: led2[k] for k in ("steps", "dt", "order", "max_last_taylor_term")},
            "n_work_samples": len(samples),
        },
        "time_series": pf.pack_samples(samples),
        "wall_seconds": clock.elapsed(),
        "old_archives_untouched": {k: hashes_before[k] == hashes_after[k] for k in hashes_before},
        "old_archive_sha256": hashes_after,
        "files": {
            Path(p).name: sha256_file(Path(p))
            for p in (__file__, pf.__file__, sh.__file__, dyn.__file__)
        },
        "answers": {
            "cycle1_timestep_pass": cmp1["time_step_pass"],
            "squeeze_after_decouple": bool(pack2["W_sq"] < 0.0),
            "purity_initial": float(s0["purity"]),
            "purity_after_rise": pack1["purity"],
            "purity_after_fall": pack2["purity"],
            "W_sq_after_rise": pack1["W_sq"],
            "W_sq_after_fall": pack2["W_sq"],
        },
        "stopped": True,
        "reason": None,
    })
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report", type=Path,
        default=ROOT / "results/q_line/qinput_a_shared_kk_vacuum_switch_20261101/switch.json")
    ap.add_argument(
        "--progress", type=Path,
        default=ROOT / "results/q_line/qinput_a_shared_kk_vacuum_switch_20261101/progress.txt")
    args = ap.parse_args(argv)
    if args.report.exists():
        print(f"report already exists: {args.report}", file=__import__("sys").stderr)
        return 2
    t0 = time.perf_counter()
    report = run(progress=args.progress)
    report["wall_seconds_total"] = time.perf_counter() - t0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    ans = report.get("answers") or {}
    print(json.dumps({
        "model_id": report["model_id"],
        "attempted": report["attempted"],
        "reason": report.get("reason"),
        "est_s": report["cost_estimate"]["estimated_seconds"],
        "cycle1_timestep_pass": ans.get("cycle1_timestep_pass"),
        "W_sq_rise": ans.get("W_sq_after_rise"),
        "W_sq_fall": ans.get("W_sq_after_fall"),
        "squeeze_after_decouple": ans.get("squeeze_after_decouple"),
        "purity_rise": ans.get("purity_after_rise"),
        "purity_fall": ans.get("purity_after_fall"),
        "decoupled": None if "decouple_at_s0" not in report else report["decouple_at_s0"]["strictly_decoupled"],
        "old_untouched": all(report.get("old_archives_untouched", {}).values()),
        "wall": report["wall_seconds_total"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
