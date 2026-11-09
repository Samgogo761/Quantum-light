#!/usr/bin/env python3
"""P1-a2 drive + S1-v0 reduced collection. Local CPU.

S1 is a source-filter coordinate, not a photon number. Does not freeze g_μ,
N_cell, bosonic [b,b†]=1, quantum pump, Δ, FM, or nk192.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import (  # noqa: E402
    CJJ_DEFINITION,
    ORDERS,
    choose_probe_k,
    domain_combine,
    dump,
    gram_psd_residual,
    undriven_trapz_check,
    undriven_y,
)
from q8_m0_cjj_validate import (  # noqa: E402
    CJJ_ABS_REPORT_RESOLUTION,
    CJJ_ABS_UNIT,
    check_sixk,
    mu_from_static,
    pack_z,
    run_cached,
    signed_kernel_diff,
)
from q8_m0_classical import p1_a2_protocol, project_harmonic  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_s1_kernel import (  # noqa: E402
    S1_ID,
    S1_STATUS,
    parallel_perp_from_cartesian,
    s1_report_from_y,
    spec_dict,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/cjj_m0_p1_a2_s1_20260917"
FORBIDDEN_RATIO_STOP = 0.5
SHIFT_REL_FOR_V96 = 0.25
MEAN_ABS_FLOOR = 1e-16
NOTE = (
    "S1-v0 adopted as source-filter collection, not photon mode. "
    "P1 envelope unchanged; polarization a2. No quantum pump."
)


def z_of(pack: dict) -> complex:
    return complex(pack["re"], pack["im"])


def combine_pack(p: dict, m: dict) -> dict:
    zp, zm = z_of(p), z_of(m)
    return {"even": pack_z(0.5 * (zp + zm)), "odd": pack_z(0.5 * (zp - zm))}


def mean_par_perp_table(times, je, jo, prot, geo) -> dict:
    je_par, je_perp = parallel_perp_from_cartesian(je, geo)
    jo_par, jo_perp = parallel_perp_from_cartesian(jo, geo)
    stacked_e = np.stack([je_par, je_perp], axis=-1)
    stacked_o = np.stack([jo_par, jo_perp], axis=-1)
    out = {}
    for n in ORDERS:
        ze = project_harmonic(times, stacked_e, n * prot.omega0, "sin2")
        zo = project_harmonic(times, stacked_o, n * prot.omega0, "sin2")
        out[str(n)] = {
            "Je_par": pack_z(ze[0]),
            "Je_perp": pack_z(ze[1]),
            "Jo_par": pack_z(zo[0]),
            "Jo_perp": pack_z(zo[1]),
        }
    return out


def s1_from_run(run: dict, geo) -> dict:
    y = np.asarray(run["Y"])
    w = np.asarray(run["w_k"], dtype=float)
    rec = s1_report_from_y(y, w, geo)
    rec["psd"] = gram_psd_residual(np.asarray(run["d_namb"]))
    rec["mean_vs_rho_maxabs"] = float(np.asarray(run["mean_vs_rho_maxabs"]))
    rec["wall_seconds"] = float(np.asarray(run["wall_seconds"]))
    return rec


def combine_s1(plus: dict, minus: dict) -> dict:
    keys = plus["d_s1"].keys()
    d_s1 = {k: combine_pack(plus["d_s1"][k], minus["d_s1"][k]) for k in keys}
    extra = {}
    for k in (
        "d_parallel_H1H1",
        "d_perp_H2H2",
        "d_parperp_all_H1H2",
        "d_parperp_H2H2",
        "d_parpar_H2H2",
        "d_perpperp_H1H1",
    ):
        extra[k] = combine_pack(plus[k], minus[k])
    return {"d_s1": d_s1, **extra}


def health_from_mean(mean_h: dict) -> dict:
    h1 = mean_h["1"]
    h2 = mean_h["2"]
    je_par = z_of(h1["Je_par"])
    jo_par = z_of(h1["Jo_par"])
    jo_perp = z_of(h2["Jo_perp"])
    je_perp = z_of(h2["Je_perp"])
    r_h1 = abs(jo_par) / max(abs(je_par), MEAN_ABS_FLOOR)
    r_h2 = abs(je_perp) / max(abs(jo_perp), MEAN_ABS_FLOOR)
    h2_ok = abs(je_perp) < FORBIDDEN_RATIO_STOP * max(abs(jo_perp), 1e-9)
    ok = (r_h1 < FORBIDDEN_RATIO_STOP) and h2_ok
    return {
        "forbidden_over_allowed_H1_Jo_par_over_Je_par": r_h1,
        "forbidden_over_allowed_H2_Je_perp_over_Jo_perp": r_h2,
        "stop_threshold": FORBIDDEN_RATIO_STOP,
        "usable_for_finer_grid": bool(ok),
        "note": "engineering leakage test against approximate C2h, not a mechanism claim",
    }


def rel_change(a: complex, b: complex) -> float:
    return abs(a - b) / max(abs(a), abs(b), MEAN_ABS_FLOOR)


def sign_flip(a: complex, b: complex, floor: float = 1e-12) -> bool:
    if abs(a) < floor or abs(b) < floor:
        return False
    return (a.real * b.real < 0.0) and (abs(a.real) > 0.1 * abs(a)) and (abs(b.real) > 0.1 * abs(b))


def stage_sixk(geo, prot, mu) -> dict:
    probe = choose_probe_k(geo, R0, mu)
    dump(OUT / "probe_k.json", {
        "labels": probe["labels"],
        "kf": probe["kf"],
        "note": probe["note"],
        "protocol_id": prot.protocol_id,
        "s1_id": S1_ID,
    })
    runs = {}
    t0 = time.perf_counter()
    for neel in (+1, -1):
        runs[neel] = run_cached(
            geo, prot, mu, name="B0_sixk", nk=6, ntpc=400, shift=0.0,
            neel=neel, kf_frac=probe["kf"], save_full_x=True, verbose=True, outdir=OUT,
        )
    wall = time.perf_counter() - t0
    checks = {str(neel): check_sixk(runs[neel], geo, prot) for neel in (+1, -1)}
    ud_chk = {
        str(neel): undriven_trapz_check(geo, R0, prot, probe["kf"], mu, neel, 400) for neel in (+1, -1)
    }
    rec = {
        "protocol_id": prot.protocol_id,
        "cjj_definition": CJJ_DEFINITION,
        "s1": spec_dict(geo),
        "wall_seconds_pair": wall,
        "algebra": checks,
        "undriven_trapz_vs_analytic": ud_chk,
        "pass": all(v["pass"] for v in checks.values()),
        "quantum_input": "not_run",
        "photon_output": "not_run",
    }
    dump(OUT / "B0_sixk_report.json", rec)
    print(json.dumps({"stage": "B0_a2", "pass": rec["pass"], "wall": wall}, indent=2), flush=True)
    return rec


def stage_config(geo, prot, mu, name, nk, ntpc, shift) -> dict:
    t0 = time.perf_counter()
    runs = {}
    for neel in (+1, -1):
        runs[neel] = run_cached(
            geo, prot, mu, name=name, nk=nk, ntpc=ntpc, shift=shift,
            neel=neel, save_full_x=False, verbose=True, outdir=OUT,
        )
    wall = time.perf_counter() - t0
    s1p = s1_from_run(runs[+1], geo)
    s1m = s1_from_run(runs[-1], geo)
    comb_cart = domain_combine(np.asarray(runs[+1]["d_namb"]), np.asarray(runs[-1]["d_namb"]))
    times = np.asarray(runs[+1]["times"])
    a_cell = float(np.asarray(runs[+1]["a_cell_ang2"]))
    mu_p = np.asarray(runs[+1]["mu_per_cell"])
    mu_m = np.asarray(runs[-1]["mu_per_cell"])
    je = 0.5 * (mu_p + mu_m) / a_cell
    jo = 0.5 * (mu_p - mu_m) / a_cell
    mean_h = mean_par_perp_table(times, je, jo, prot, geo)
    und = {
        "+1": undriven_y(geo, R0, prot, np.asarray(runs[+1]["kf"]), mu, +1),
        "-1": undriven_y(geo, R0, prot, np.asarray(runs[-1]["kf"]), mu, -1),
    }
    und_s1 = {
        "+1": s1_report_from_y(und["+1"]["Y"], np.asarray(runs[+1]["w_k"], dtype=float), geo),
        "-1": s1_report_from_y(und["-1"]["Y"], np.asarray(runs[-1]["w_k"], dtype=float), geo),
    }
    rec = {
        "name": name,
        "nk": nk,
        "ntpc": ntpc,
        "grid_shift": shift,
        "protocol_id": prot.protocol_id,
        "s1_id": S1_ID,
        "s1_status": S1_STATUS,
        "wall_seconds_pair": wall,
        "mean_vs_rho_maxabs": {
            "+1": float(np.asarray(runs[+1]["mean_vs_rho_maxabs"])),
            "-1": float(np.asarray(runs[-1]["mean_vs_rho_maxabs"])),
        },
        "psd_plus": comb_cart["psd_plus"],
        "psd_minus": comb_cart["psd_minus"],
        "mean_harmonics_sheet_sin2_parperp": mean_h,
        "s1_plus": s1p,
        "s1_minus": s1m,
        "s1_domain": combine_s1(s1p, s1m),
        "undriven_s1_plus": und_s1["+1"],
        "undriven_s1_minus": und_s1["-1"],
        "pumped_minus_undriven_d_parperp": {
            "+1": signed_kernel_diff(
                z_of(s1p["d_s1"]["d_parperp_H1H2"]),
                z_of(und_s1["+1"]["d_s1"]["d_parperp_H1H2"]),
            ),
            "-1": signed_kernel_diff(
                z_of(s1m["d_s1"]["d_parperp_H1H2"]),
                z_of(und_s1["-1"]["d_s1"]["d_parperp_H1H2"]),
            ),
        },
        "health": health_from_mean(mean_h),
        "cjj_definition": CJJ_DEFINITION,
        "cjj_abs_unit": CJJ_ABS_UNIT,
        "cjj_abs_floor": CJJ_ABS_REPORT_RESOLUTION,
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "note": NOTE,
    }
    dump(OUT / name / "summary.json", rec)
    spotlight = {
        "stage": name,
        "wall": wall,
        "Je_par_H1": mean_h["1"]["Je_par"],
        "Jo_perp_H2": mean_h["2"]["Jo_perp"],
        "Jo_par_H1": mean_h["1"]["Jo_par"],
        "Je_perp_H2": mean_h["2"]["Je_perp"],
        "d_o_parperp": rec["s1_domain"]["d_s1"]["d_parperp_H1H2"]["odd"],
        "d_e_parpar": rec["s1_domain"]["d_s1"]["d_parpar_H1H1"]["even"],
        "health": rec["health"],
    }
    print(json.dumps(spotlight, indent=2), flush=True)
    return rec


def decide_v96(v48: dict, v48s: dict | None) -> dict:
    if not v48["health"]["usable_for_finer_grid"]:
        return {"run": False, "reason": "V48 forbidden/allowed above engineering stop"}
    if v48s is None:
        return {"run": False, "reason": "V48S not available"}
    if not v48s["health"]["usable_for_finer_grid"]:
        return {"run": False, "reason": "V48S forbidden/allowed above engineering stop"}
    jo_g = z_of(v48["mean_harmonics_sheet_sin2_parperp"]["2"]["Jo_perp"])
    jo_s = z_of(v48s["mean_harmonics_sheet_sin2_parperp"]["2"]["Jo_perp"])
    d_g = z_of(v48["s1_domain"]["d_s1"]["d_parperp_H1H2"]["odd"])
    d_s = z_of(v48s["s1_domain"]["d_s1"]["d_parperp_H1H2"]["odd"])
    reasons = []
    if sign_flip(jo_g, jo_s):
        reasons.append("Jo_perp H2 Γ/shift Re sign flip")
    if sign_flip(d_g, d_s):
        reasons.append("d_o parperp Γ/shift Re sign flip")
    rel_j = rel_change(jo_g, jo_s)
    rel_d = rel_change(d_g, d_s)
    if rel_j > SHIFT_REL_FOR_V96:
        reasons.append(f"Jo_perp Γ/shift rel {rel_j:.3f} > {SHIFT_REL_FOR_V96}")
    if rel_d > SHIFT_REL_FOR_V96:
        reasons.append(f"d_o parperp Γ/shift rel {rel_d:.3f} > {SHIFT_REL_FOR_V96}")
    return {
        "run": len(reasons) == 0,
        "reason": "ok" if not reasons else "; ".join(reasons),
        "rel_Jo_perp": rel_j,
        "rel_d_o_parperp": rel_d,
        "note": "no V96S, no nk192; V96 is only a trend look",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-after", choices=("sixk", "v48", "v48s", "v96"), default="v96")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    mu = mu_from_static()
    dump(OUT / "s1_v0_adopted.json", spec_dict(geo))
    ident = {
        "note": NOTE,
        "protocol_id": prot.protocol_id,
        "s1_id": S1_ID,
        "s1_status": S1_STATUS,
        "mu": mu,
        "forbidden_ratio_stop": FORBIDDEN_RATIO_STOP,
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "delta_scan": "not_run",
        "nk192": "not_run",
    }
    dump(OUT / "campaign.json", ident)
    t_all = time.perf_counter()
    sixk = stage_sixk(geo, prot, mu)
    jobs = {"B0_sixk": sixk}
    if args.stop_after == "sixk" or not sixk["pass"]:
        dump(OUT / "summary.json", {"ident": ident, "jobs": jobs, "stop": "sixk" if sixk["pass"] else "sixk_failed"})
        return
    v48 = stage_config(geo, prot, mu, "V48", 48, 400, 0.0)
    jobs["V48"] = v48
    if args.stop_after == "v48":
        dump(OUT / "summary.json", {"ident": ident, "jobs": jobs, "stop": "v48"})
        return
    v48s = None
    if v48["health"]["usable_for_finer_grid"]:
        v48s = stage_config(geo, prot, mu, "V48S", 48, 400, 0.5)
        jobs["V48S"] = v48s
    else:
        print(json.dumps({"skip": "V48S", "reason": "V48 leakage"}, indent=2), flush=True)
    if args.stop_after == "v48s":
        dump(OUT / "summary.json", {"ident": ident, "jobs": jobs, "stop": "v48s"})
        return
    decision = decide_v96(v48, v48s)
    dump(OUT / "v96_decision.json", decision)
    if decision["run"]:
        jobs["V96"] = stage_config(geo, prot, mu, "V96", 96, 400, 0.0)
    else:
        print(json.dumps({"skip": "V96", **decision}, indent=2), flush=True)
    out = {
        "ident": ident,
        "jobs": {k: {kk: vv for kk, vv in rec.items() if kk not in ("s1_plus", "s1_minus", "undriven_s1_plus", "undriven_s1_minus")} if k != "B0_sixk" else rec for k, rec in jobs.items()},
        "v96_decision": decision,
        "wall_seconds_all": time.perf_counter() - t_all,
        "mechanism_claim": "not_made",
        "quantum_input": "not_run",
        "photon_output": "not_run",
    }
    dump(OUT / "summary.json", out)
    print(json.dumps({
        "done": True,
        "wall": out["wall_seconds_all"],
        "v96": decision,
        "V48_health": v48["health"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
