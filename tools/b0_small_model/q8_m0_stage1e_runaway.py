#!/usr/bin/env python3
"""Stage-1E: runaway source diagnostics. Same physics as frozen G_self.

Energy flow, nonlinear seed+self-field, low-frequency account, small k sets.
No T2, no DC removal, no P1 change, no 48/96/192, no Fock expansion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump, propagate_cjj  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_gsheet import gamma_self  # noqa: E402
from q8_m0_loop import ResummationNotChosenError, apply_resummation_to_output  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_runaway import (  # noqa: E402
    RUNAWAY_ID,
    k_sets,
    propagate_seed_self,
    seed_slow_sin2,
    window_report,
)
from q8_m0_stage1 import stage1_spec  # noqa: E402
from q8_m0_volterra import h2_kernels, volterra_forward  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1e_runaway_20260921"
MU = -0.009010620372198276
NTPC = 400
EPS = 3.0e-5
GAMMA_SCALES = (0.0, 0.01, 0.03, 0.1, 1.0)


def _slim_window(rec: dict) -> dict:
    keep = dict(rec)
    keep.pop("W_el_cumulative", None)
    for key in ("I_bands", "A_self_bands", "dc_ramp", "growth"):
        if key in keep and isinstance(keep[key], dict):
            keep[key] = {k: v for k, v in keep[key].items() if k != "mean"}
            if "mean" in rec[key]:
                keep[key]["mean_abs"] = rec[key].get("mean_abs")
                keep[key]["mean"] = rec[key].get("mean")
    return keep


def linear_on_set(geo, prot, kf, gamma, scales, e_perp, F, times, dt, omega0):
    print(f"  CJJ X+D nk={len(kf)}", flush=True)
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=NTPC, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=kf, save_full_x=True, save_mean_D=True,
    )
    x = r0["X_full"]
    w = r0["w_k"]
    mean_d = r0["mean_D_t"]
    i_p1 = np.asarray(r0["mu_per_cell"])
    h_re = np.real(F)[:, None] * e_perp[None, :]
    peak = float(np.max(np.abs(h_re)))
    a_in = (EPS / max(peak, 1e-30)) * h_re
    ref_vol0 = volterra_forward(x, w, mean_d, dt, 0.0, a_L_in=a_in)
    ref_peak = float(np.max(np.linalg.norm(ref_vol0["I"], axis=1)))
    rows = []
    for sc in scales:
        vol = volterra_forward(x, w, mean_d, dt, sc * gamma, a_L_in=a_in)
        wr = window_report(vol["I"], a_in, times, dt, sc * gamma, omega0, ref_peak)
        wr["gamma_scale"] = sc
        wr["finite"] = bool(vol["finite"] and not wr["halted_before_end"])
        wr["maxabs_I_full"] = vol["maxabs_I"]
        rows.append(_slim_window(wr))
        print(
            f"    lin sc={sc} W_el={wr['W_el']:.3e} {wr['W_el_sign']} "
            f"|I|={wr['maxabs_I']:.3e} dc={wr['I_bands']['frac_dc_below_0.25_omega0']:.3f} "
            f"halt={wr['t_halt']:.1f}",
            flush=True,
        )
    return {
        "I_p1": i_p1,
        "ref_peak": ref_peak,
        "a_in": a_in,
        "linear": rows,
        "r0_wall": float(r0["wall_seconds"]),
        "n_k": int(r0["n_k"]),
    }


def nonlinear_on_set(geo, prot, kf, gamma, i_p1, a_in_h2, times, dt, omega0, ref_peak):
    rows = []
    seeds = {
        "H2_Re": a_in_h2,
        "slow_sin2": seed_slow_sin2(times, prot.t_pulse, geo, EPS),
    }
    for sname, seed in seeds.items():
        for sc in (0.0, 0.03, 1.0):
            print(f"    nl {sname} sc={sc}", flush=True)
            run = propagate_seed_self(
                geo, R0, prot, kf=kf, mu=MU, neel=+1, nt_per_cycle=NTPC,
                seed=seed, i_ref=i_p1, gamma=sc * gamma, verbose=False,
            )
            wr = window_report(run["dI"], seed, times, dt, sc * gamma, omega0, ref_peak)
            wr["gamma_scale"] = sc
            wr["seed"] = sname
            wr["nl_halted"] = run["halted"]
            wr["nl_maxabs_I"] = run["maxabs_I"]
            wr["nl_maxabs_dI"] = run["maxabs_dI"]
            wr["nl_maxabs_A_self"] = run["maxabs_A_self"]
            wr["wall"] = run["wall_seconds"]
            rows.append(_slim_window(wr))
            print(
                f"      W_el={wr['W_el']:.3e} {wr['W_el_sign']} "
                f"|dI|={run['maxabs_dI']:.3e} Gamma={wr['growth']['Gamma']:.4e}",
                flush=True,
            )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma = gamma_self(geo)
    blocked = False
    try:
        apply_resummation_to_output()
    except ResummationNotChosenError:
        blocked = True
    sets = k_sets(geo, R0, MU)
    # time grid from a dummy protocol length
    nt = prot.n_cycles * NTPC
    dt = prot.period / NTPC
    times = np.arange(nt + 1) * dt
    hk = h2_kernels(times, prot.omega0, prot.t_pulse, geo)
    F = hk["F"]
    e_perp = hk["e_perp"]
    omega0 = prot.omega0

    out_sets = {}
    series = {}
    for name, spec in sets.items():
        print(f"[stage1e] k-set {name} nk={spec['n']}", flush=True)
        lin = linear_on_set(
            geo, prot, spec["kf"], gamma, GAMMA_SCALES, e_perp, F, times, dt, omega0,
        )
        i_p1 = lin.pop("I_p1")
        a_in = lin.pop("a_in")
        ref_peak = lin["ref_peak"]
        nl = nonlinear_on_set(
            geo, prot, spec["kf"], gamma, i_p1, a_in, times, dt, omega0, ref_peak,
        )
        out_sets[name] = {
            "n_k": spec["n"],
            "labels": list(spec["labels"]),
            "kf": np.asarray(spec["kf"]).tolist(),
            "linear": lin["linear"],
            "nonlinear": nl,
            "cjj_wall": lin["r0_wall"],
        }
        if name == "probe6":
            series["times"] = times
            series["I_p1"] = i_p1
            series["a_in_H2Re"] = a_in

    np.savez(OUT / "probe6_series.npz", **series)
    payload = {
        "runaway_id": RUNAWAY_ID,
        "spec": stage1_spec(),
        "language": (
            "finite_time_linear_self_consistent_runaway_amplification_"
            "not_a_statement_that_the_Volterra_equation_has_no_solution"
        ),
        "frozen": {
            "no_T2": True,
            "no_DC_subtraction": True,
            "P1_local_classical": True,
            "no_full_BZ": True,
            "no_fock_expansion": True,
            "production_V_still_born": True,
        },
        "eps_seed_kappa": EPS,
        "gamma_self": gamma,
        "nt_per_cycle": NTPC,
        "resummation_production_blocked": blocked,
        "k_sets": out_sets,
        "nk96": "not_run",
        "g2": "not_run",
        "squeezing_claim": "not_evaluated",
        "gain_medium_claim": "not_evaluated_diagnostics_only",
    }
    dump(OUT / "stage1e_runaway.json", payload)
    probe_lin = out_sets["probe6"]["linear"]
    probe_nl = out_sets["probe6"]["nonlinear"]
    summary = {
        "wrote": str(OUT / "stage1e_runaway.json"),
        "probe6_linear": [
            {
                "sc": r["gamma_scale"],
                "W_el": r["W_el"],
                "sign": r["W_el_sign"],
                "dc": r["I_bands"]["frac_dc_below_0.25_omega0"],
                "ramp": r["dc_ramp"]["ramp_over_A_self_end"],
                "Gamma": r["growth"]["Gamma"],
                "halt": r["t_halt"],
                "maxabs_I": r["maxabs_I"],
            }
            for r in probe_lin
        ],
        "probe6_nl_H2": [
            {
                "sc": r["gamma_scale"],
                "W_el": r["W_el"],
                "sign": r["W_el_sign"],
                "Gamma": r["growth"]["Gamma"],
                "maxabs_dI": r["nl_maxabs_dI"],
                "halted": r["nl_halted"],
            }
            for r in probe_nl if r["seed"] == "H2_Re"
        ],
        "kset_W_el_sc1_linear": {
            k: next((r["W_el"] for r in v["linear"] if r["gamma_scale"] == 1.0), None)
            for k, v in out_sets.items()
        },
        "kset_nl_max_dI_sc1": {
            k: max(r["nl_maxabs_dI"] for r in v["nonlinear"] if r["gamma_scale"] == 1.0)
            for k, v in out_sets.items()
        },
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
