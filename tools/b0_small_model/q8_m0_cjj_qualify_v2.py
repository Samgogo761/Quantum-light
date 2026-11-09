#!/usr/bin/env python3
"""Round-2 CJJ finite quadrature: V96 and V96S only, ±N, then stop.

Does not run nk192, dt at 96, quantum pump, photon output, or a new drive.
Round-1 NPZ is read-only.
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
    DEFINITION_REVIEW,
    ORDERS,
    d_from_y,
    domain_combine,
    dump,
    gram_psd_residual,
    undriven_y,
)
from q8_m0_cjj_validate import (  # noqa: E402
    CJJ_ABS_FLOOR_RATIONALE,
    CJJ_ABS_REPORT_RESOLUTION,
    CJJ_ABS_UNIT,
    d_catalog,
    explicit_rel_columns,
    mu_from_static,
    pack_z,
    run_cached,
    signed_kernel_diff,
)
from q8_m0_classical import p1_protocol, project_harmonic  # noqa: E402
from q8_m0_h2_qualify import PHASE_UNSTABLE_DEG, QUAL_REL, TREND_REL_CAP, wrapped_phase_diff_deg  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/cjj_m0_p1_qualification_v2_20260917"
ROUND1 = REPO / "results/q_line/cjj_m0_p1_validation_20260916"
STOCK96 = REPO / "results/q_line/m0_p0p1_20260914/p1_A0.8_nk96_nt400/analysis.json"
PRIMARY = ("d_H2x,H2x", "d_H6y,H6y", "d_H2x,H3x")
CONFIGS = (("V96", 96, 400, 0.0), ("V96S", 96, 400, 0.5))
MAX_TRAJECTORIES = 4
V2_NOTE = "finite_quadrature_upgrade_v2_20260917"
DT_ROUND1_NOTE = "V48 vs V48dt only; not re-verified at nk=96"


def load_round1_d(name: str) -> dict:
    plus = np.load(ROUND1 / name / f"{name}_N+1_A+1.npz", allow_pickle=True)
    minus = np.load(ROUND1 / name / f"{name}_N-1_A+1.npz", allow_pickle=True)
    comb = domain_combine(np.asarray(plus["d_namb"]), np.asarray(minus["d_namb"]))
    return {
        "d_plus": np.asarray(plus["d_namb"]),
        "d_minus": np.asarray(minus["d_namb"]),
        "d_dom_e": comb["d_dom_e"],
        "d_dom_o": comb["d_dom_o"],
        "times": np.asarray(plus["times"]),
        "mu_plus": np.asarray(plus["mu_per_cell"]),
        "a_cell": float(np.asarray(plus["a_cell_ang2"])),
    }


def z_of_d(d: np.ndarray, key: str) -> complex:
    cat = d_catalog(d)
    pack = cat[key]
    return complex(pack["re"], pack["im"])


def classify_v2(z96, z48, z96s, z48s, *, even96=None) -> dict:
    """Γ48→Γ96, S48→S96, Γ96↔S96. No dt at 96."""
    z96, z48, z96s, z48s = map(complex, (z96, z48, z96s, z48s))
    mag = abs(z96)
    d_g = abs(z96 - z48)
    d_s_chain = abs(z96s - z48s)
    d_shift = abs(z96 - z96s)
    rel_g = d_g / mag if mag else None
    rel_s_chain = d_s_chain / abs(z96s) if abs(z96s) else None
    rel_shift96 = d_shift / mag if mag else None
    phases = {
        "G48_vs_G96": wrapped_phase_diff_deg(z48, z96),
        "S48_vs_S96": wrapped_phase_diff_deg(z48s, z96s),
        "G96_vs_S96": wrapped_phase_diff_deg(z96s, z96),
    }
    floor = CJJ_ABS_REPORT_RESOLUTION
    reasons: list[str] = []
    if mag < floor and max(d_g, d_s_chain, d_shift) < floor:
        status = "below_reporting_resolution"
        reasons.append("reference_and_all_absolute_changes_below_reporting_resolution")
    else:
        phase_unstable = False
        for key, ang, zo, zr in (
            ("G48_vs_G96", phases["G48_vs_G96"], z48, z96),
            ("S48_vs_S96", phases["S48_vs_S96"], z48s, z96s),
            ("G96_vs_S96", phases["G96_vs_S96"], z96s, z96),
        ):
            if ang is None:
                continue
            if abs(zo) >= floor and abs(zr) >= floor and abs(ang) > PHASE_UNSTABLE_DEG:
                phase_unstable = True
                reasons.append(f"phase_{key}_abs_{abs(ang):.2f}_deg_exceeds_{PHASE_UNSTABLE_DEG}")
        rels = [r for r in (rel_g, rel_s_chain, rel_shift96) if r is not None]
        max_d = max(d_g, d_s_chain, d_shift)
        if mag < floor and max_d >= floor:
            status = "unresolved"
            reasons.append("reference_below_reporting_resolution_but_absolute_changes_are_not")
        elif rels and all(r <= QUAL_REL for r in rels) and not phase_unstable:
            status = "qualified_preliminary"
            reasons.append("G48_G96_and_S48_S96_and_G96_S96_at_or_below_5pct")
        elif (not phase_unstable) and mag > 0 and max_d < TREND_REL_CAP * mag:
            status = "trend_not_qualified"
            reasons.append("repeatable_nonzero_but_misses_5pct_and_change_smaller_than_|z_G96|")
        else:
            status = "unresolved"
            if phase_unstable:
                reasons.append("phase_or_direction_unstable")
            if mag > 0 and max_d >= TREND_REL_CAP * mag:
                reasons.append("absolute_change_comparable_to_or_larger_than_|z_G96|")
            if not reasons:
                reasons.append("failed_repeatability_boundary")
    even = None if even96 is None else complex(even96)
    scale = None if even is None or abs(even) == 0 else mag / abs(even)
    abs_vs_even = None if even is None or abs(even) == 0 else max(d_g, d_s_chain, d_shift) / abs(even)
    return {
        "z_G96": pack_z(z96),
        "z_G48": pack_z(z48),
        "z_S96": pack_z(z96s),
        "z_S48": pack_z(z48s),
        "abs_G48_to_G96": d_g,
        "abs_S48_to_S96": d_s_chain,
        "abs_G96_to_S96": d_shift,
        "rel_G48_to_G96": rel_g,
        "rel_S48_to_S96": rel_s_chain,
        "rel_G96_to_S96": rel_shift96,
        "phase_diff_deg": phases,
        "status": status,
        "reasons": reasons,
        "abs_floor": floor,
        "abs_unit": CJJ_ABS_UNIT,
        "note": CJJ_ABS_FLOOR_RATIONALE,
        "odd_over_even_G96": scale,
        "max_abs_change_over_even_G96": abs_vs_even,
        "dt_round1": DT_ROUND1_NOTE,
    }


def mean_harmonics(times, mu_p, mu_m, a_cell, prot) -> dict:
    je = 0.5 * (mu_p + mu_m) / a_cell
    jo = 0.5 * (mu_p - mu_m) / a_cell
    out = {}
    for n in ORDERS:
        ze = project_harmonic(times, je, n * prot.omega0, "sin2")
        zo = project_harmonic(times, jo, n * prot.omega0, "sin2")
        out[str(n)] = {
            "Je_x": [float(ze[0].real), float(ze[0].imag)],
            "Je_y": [float(ze[1].real), float(ze[1].imag)],
            "Jo_x": [float(zo[0].real), float(zo[0].imag)],
            "Jo_y": [float(zo[1].real), float(zo[1].imag)],
        }
    return out


def stock96_check(mean_h: dict) -> dict:
    if not STOCK96.exists():
        return {"available": False}
    tab = json.loads(STOCK96.read_text(encoding="utf-8"))["analysis"]["windows"]["sin2"]
    out = {"available": True, "pairs": {}, "note": "implementation consistency only; does not qualify mean Jo(H2)"}
    for n, who, comp, key in (
        (1, "J_e", "x", "Je_x"),
        (2, "J_o", "x", "Jo_x"),
        (2, "J_o", "y", "Jo_y"),
        (3, "J_e", "x", "Je_x"),
    ):
        z_stock = complex(*tab[who][str(n)][comp])
        z_new = complex(*mean_h[str(n)][key])
        out["pairs"][f"{who}_{comp}_H{n}"] = {
            "stock": [z_stock.real, z_stock.imag],
            "v96": [z_new.real, z_new.imag],
            "abs_diff": abs(z_new - z_stock),
        }
    return out


def estimate_cost() -> dict:
    v48 = json.loads((ROUND1 / "V48" / "summary.json").read_text(encoding="utf-8"))
    v48s = json.loads((ROUND1 / "V48S" / "summary.json").read_text(encoding="utf-8"))
    wall48 = float(v48["wall_seconds_pair"])
    wall48s = float(v48s["wall_seconds_pair"])
    k_ratio = (96 * 96) / (48 * 48)
    y_bytes = 96 * 96 * 9 * 2 * 4 * 4 * 16
    snap_bytes = 25 * 96 * 96 * 2 * 4 * 4 * 16
    return {
        "basis": "actual V48/V48S pair walls, linear in Nk, nt=400 unchanged",
        "V48_pair_s": wall48,
        "V48S_pair_s": wall48s,
        "k_ratio_96_over_48": k_ratio,
        "V96_pair_s_est": wall48 * k_ratio,
        "V96S_pair_s_est": wall48s * k_ratio,
        "four_trajectories_s_est": wall48 * k_ratio + wall48s * k_ratio,
        "Y_complex128_bytes_est": y_bytes,
        "snapshot_X_bytes_est": snap_bytes,
        "note": "local CPU; not a SLURM job. Stop after these four. No nk192.",
        "max_trajectories": MAX_TRAJECTORIES,
        "quantum_input": "not_run",
    }


def stage_pair(geo, prot, mu, name, nk, ntpc, shift) -> dict:
    t0 = time.perf_counter()
    runs = {}
    for neel in (+1, -1):
        runs[neel] = run_cached(
            geo, prot, mu, name=name, nk=nk, ntpc=ntpc, shift=shift,
            neel=neel, save_full_x=False, verbose=True, outdir=OUT,
        )
    wall = time.perf_counter() - t0
    d_plus = np.asarray(runs[+1]["d_namb"])
    d_minus = np.asarray(runs[-1]["d_namb"])
    comb = domain_combine(d_plus, d_minus)
    health = {
        "mean_vs_rho_maxabs": {
            "+1": float(np.asarray(runs[+1]["mean_vs_rho_maxabs"])),
            "-1": float(np.asarray(runs[-1]["mean_vs_rho_maxabs"])),
        },
        "psd_plus": comb["psd_plus"],
        "psd_minus": comb["psd_minus"],
        "y_rebuild_maxabs": {
            "+1": float(np.max(np.abs(d_from_y(np.asarray(runs[+1]["Y"]), np.asarray(runs[+1]["w_k"])) - d_plus))),
            "-1": float(np.max(np.abs(d_from_y(np.asarray(runs[-1]["Y"]), np.asarray(runs[-1]["w_k"])) - d_minus))),
        },
        "n_k": int(np.asarray(runs[+1]["n_k"])),
        "times": int(np.asarray(runs[+1]["times"]).size),
    }
    und = {
        "+1": undriven_y(geo, R0, prot, np.asarray(runs[+1]["kf"]), mu, +1),
        "-1": undriven_y(geo, R0, prot, np.asarray(runs[-1]["kf"]), mu, -1),
    }
    mu_p = np.asarray(runs[+1]["mu_per_cell"])
    mu_m = np.asarray(runs[-1]["mu_per_cell"])
    times = np.asarray(runs[+1]["times"])
    a_cell = float(np.asarray(runs[+1]["a_cell_ang2"]))
    rec = {
        "name": name,
        "nk": nk,
        "ntpc": ntpc,
        "grid_shift": shift,
        "wall_seconds_pair": wall,
        "health": health,
        "d_plus": d_catalog(d_plus),
        "d_minus": d_catalog(d_minus),
        "d_dom_e": d_catalog(comb["d_dom_e"]),
        "d_dom_o": d_catalog(comb["d_dom_o"]),
        "mean_harmonics_sheet_sin2": mean_harmonics(times, mu_p, mu_m, a_cell, prot),
        "pumped_minus_undriven_H2xx": {
            "+1": signed_kernel_diff(d_plus[1, 0, 1, 0], und["+1"]["d_namb"][1, 0, 1, 0]),
            "-1": signed_kernel_diff(d_minus[1, 0, 1, 0], und["-1"]["d_namb"][1, 0, 1, 0]),
        },
        "pumped_minus_undriven_H6yy": {
            "+1": signed_kernel_diff(d_plus[5, 1, 5, 1], und["+1"]["d_namb"][5, 1, 5, 1]),
            "-1": signed_kernel_diff(d_minus[5, 1, 5, 1], und["-1"]["d_namb"][5, 1, 5, 1]),
        },
        "cjj_definition": CJJ_DEFINITION,
        "v2_campaign": V2_NOTE,
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "arrays": {"d_plus": d_plus, "d_minus": d_minus, "d_dom_e": comb["d_dom_e"], "d_dom_o": comb["d_dom_o"]},
    }
    slim = {k: rec[k] for k in rec if k != "arrays"}
    dump(OUT / name / "summary.json", slim)
    print(json.dumps({
        "stage": name,
        "wall": wall,
        "n_k": health["n_k"],
        "psd_ok": health["psd_plus"]["psd_ok"] and health["psd_minus"]["psd_ok"],
        "d_H2xx_plus": rec["d_plus"]["d_H2x,H2x"],
        "d_H6yy_plus": rec["d_plus"]["d_H6y,H6y"],
    }, indent=2), flush=True)
    return rec


def round1_explicit_table() -> dict:
    inv = json.loads((ROUND1 / "cjj_wrapup_inventory.json").read_text(encoding="utf-8"))
    conv = inv["source_observable_convergence"]
    keys = PRIMARY + ("d_H5y,H5y", "d_H7x,H7x", "d_H1x,H1x", "d_H3x,H3x")
    out = {}
    for key in keys:
        if key not in conv["d_plus"]:
            continue
        out[key] = {
            "plus": conv["d_plus"][key]["z_V48"],
            "minus": conv["d_minus"][key]["z_V48"],
            "dom_o": conv["d_dom_o"][key]["z_V48"],
            "rel": explicit_rel_columns(conv["d_plus"][key], conv["d_minus"][key]),
            "status_plus": conv["d_plus"][key]["status"],
            "status_minus": conv["d_minus"][key]["status"],
            "status_dom_o": conv["d_dom_o"][key]["status"],
        }
    return out


def summarize(jobs: dict, r1: dict) -> dict:
    sides = ("d_plus", "d_minus", "d_dom_e", "d_dom_o")
    conv = {}
    for side in sides:
        conv[side] = {}
        for key, pack in jobs["V96"][side].items():
            even = None
            if side == "d_dom_o":
                even = complex(jobs["V96"]["d_dom_e"][key]["re"], jobs["V96"]["d_dom_e"][key]["im"])
            conv[side][key] = classify_v2(
                complex(pack["re"], pack["im"]),
                z_of_d(r1["V48"][side], key),
                complex(jobs["V96S"][side][key]["re"], jobs["V96S"][side][key]["im"]),
                z_of_d(r1["V48S"][side], key),
                even96=even,
            )
    primary = {key: {side: conv[side][key] for side in sides} for key in PRIMARY}
    health_ok = all(
        jobs[name]["health"]["psd_plus"]["psd_ok"]
        and jobs[name]["health"]["psd_minus"]["psd_ok"]
        and max(jobs[name]["health"]["mean_vs_rho_maxabs"].values()) < 1e-8
        for name in ("V96", "V96S")
    )
    return {
        "cjj_definition": CJJ_DEFINITION,
        "definition_review": DEFINITION_REVIEW,
        "v2_campaign": V2_NOTE,
        "m0_cjj_status": "finite_quadrature_v2_complete_stop",
        "trajectories_run": MAX_TRAJECTORIES,
        "did_not_start_nk192": True,
        "did_not_rerun_dt_at_96": True,
        "health_ok": health_ok,
        "primary": primary,
        "source_observable_convergence": conv,
        "v96_vs_stock_nk96_mean": stock96_check(jobs["V96"]["mean_harmonics_sheet_sin2"]),
        "round1_explicit_rel_columns": round1_explicit_table(),
        "jobs_health": {k: jobs[k]["health"] for k in jobs},
        "jobs_wall": {k: jobs[k]["wall_seconds_pair"] for k in jobs},
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "new_drive_protocol": "not_run",
        "stop_rule": "four trajectories done; do not auto-start a third round",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("estimate", "v96", "v96s", "summarize", "auto"), default="auto")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_protocol(0.8)
    mu = mu_from_static()
    header = {
        "role": "launch_record",
        "cjj_definition": CJJ_DEFINITION,
        "definition_review": DEFINITION_REVIEW,
        "v2_campaign": V2_NOTE,
        "max_trajectories": MAX_TRAJECTORIES,
        "configs": [c[0] for c in CONFIGS],
        "round1_readonly": str(ROUND1),
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "new_drive_protocol": "not_run",
        "did_not_start_nk192": True,
    }
    if not (OUT / "campaign_header.json").exists():
        dump(OUT / "campaign_header.json", header)
    est = estimate_cost()
    dump(OUT / "cost_estimate.json", est)
    dump(OUT / "round1_explicit_rel_columns.json", round1_explicit_table())
    print(json.dumps({"cost_estimate": est}, indent=2), flush=True)
    if args.stage == "estimate":
        return 0
    jobs = {}
    run_which = list(CONFIGS) if args.stage == "auto" else (
        [CONFIGS[0]] if args.stage == "v96" else [CONFIGS[1]] if args.stage == "v96s" else []
    )
    n_done = 0
    for name, nk, ntpc, shift in run_which:
        rec = stage_pair(geo, prot, mu, name, nk, ntpc, shift)
        jobs[name] = rec
        n_done += 2
        if n_done >= MAX_TRAJECTORIES and args.stage == "auto":
            break
        health = rec["health"]
        if not (health["psd_plus"]["psd_ok"] and health["psd_minus"]["psd_ok"]):
            dump(OUT / "STOP.json", {"reason": "PSD health failed", "name": name, "health": health})
            return 1
        if max(health["mean_vs_rho_maxabs"].values()) >= 1e-8:
            dump(OUT / "STOP.json", {"reason": "mean_vs_rho health failed", "name": name, "health": health})
            return 1
    if args.stage in ("summarize", "auto") or (OUT / "V96S" / "summary.json").exists():
        for name, *_ in CONFIGS:
            p = OUT / name / "summary.json"
            if p.exists() and name not in jobs:
                jobs[name] = json.loads(p.read_text(encoding="utf-8"))
            npz_plus = OUT / name / f"{name}_N+1_A+1.npz"
            npz_minus = OUT / name / f"{name}_N-1_A+1.npz"
            if npz_plus.exists() and npz_minus.exists():
                d_plus = np.asarray(np.load(npz_plus)["d_namb"])
                d_minus = np.asarray(np.load(npz_minus)["d_namb"])
                comb = domain_combine(d_plus, d_minus)
                jobs.setdefault(name, {})
                jobs[name]["d_plus"] = d_catalog(d_plus)
                jobs[name]["d_minus"] = d_catalog(d_minus)
                jobs[name]["d_dom_e"] = d_catalog(comb["d_dom_e"])
                jobs[name]["d_dom_o"] = d_catalog(comb["d_dom_o"])
        if set(("V96", "V96S")) <= set(jobs) and "d_plus" in jobs["V96"] and "d_plus" in jobs["V96S"]:
            r1 = {"V48": load_round1_d("V48"), "V48S": load_round1_d("V48S")}
            summary = summarize(jobs, r1)
            dump(OUT / "cjj_v2_inventory.json", summary)
            dump(OUT / "cjj_v2_primary.json", {
                "primary": summary["primary"],
                "health_ok": summary["health_ok"],
                "v96_vs_stock_nk96_mean": summary["v96_vs_stock_nk96_mean"],
                "jobs_wall": summary["jobs_wall"],
                "m0_cjj_status": summary["m0_cjj_status"],
                "quantum_input": "not_run",
            })
            dump(OUT / "campaign_status.json", {
                "role": "canonical_status",
                "m0_cjj_status": summary["m0_cjj_status"],
                "trajectories_run": MAX_TRAJECTORIES,
                "did_not_start_nk192": True,
                "quantum_input": "not_run",
            })
            print(json.dumps({
                "m0_cjj_status": summary["m0_cjj_status"],
                "health_ok": summary["health_ok"],
                "H2x_plus": summary["primary"]["d_H2x,H2x"]["d_plus"]["status"],
                "H2x_dom_o": summary["primary"]["d_H2x,H2x"]["d_dom_o"]["status"],
                "H6y_plus": summary["primary"]["d_H6y,H6y"]["d_plus"]["status"],
                "H6y_dom_o": summary["primary"]["d_H6y,H6y"]["d_dom_o"]["status"],
                "H2xH3x_plus": summary["primary"]["d_H2x,H3x"]["d_plus"]["status"],
                "H2xH3x_dom_o": summary["primary"]["d_H2x,H3x"]["d_dom_o"]["status"],
            }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
