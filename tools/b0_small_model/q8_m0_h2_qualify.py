#!/usr/bin/env python3
"""Fixed-P1 H2 numerical qualification (2026-09-15 task).

M0/R0/P1 A*=0.8 unchanged. At most four new ±N paired configs.
Does not run M0 CJJ or quantum light.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from q8_m0_classical import (
    POSTPROCESS_VERSION,
    P0Protocol,
    analyse_set,
    compare_complex_amplitudes,
    field_resources,
    k_grid,
    p1_protocol,
    run_set,
    target_quantities,
)
from q8_m0_model import MODEL_ID, R0, build_geometry

REPO = Path(__file__).resolve().parents[2]
OLD_CAMPAIGN = REPO / "results/q_line/m0_p0p1_20260914"
OLD_STATIC = REPO / "results/q_line/m0_v1_20260912/static/gap_report.json"
OLD_CLASSICAL = REPO / "results/q_line/m0_v1_20260912/classical/classical_report.json"
OUT_ROOT = REPO / "results/q_line/p1_h2_qualification_20260915"
ORDERS = list(range(1, 10))
H2_NAMES = ("J_plus", "J_minus", "J_e", "J_o")
ABS_REPORT_RESOLUTION = 1e-9
ABS_REPORT_UNIT = "sheet current harmonic z [e t_h/(ħ Å)]"
QUAL_REL = 0.05
TREND_REL_CAP = 1.0
PHASE_UNSTABLE_DEG = 90.0
CLASSIFICATION_RULE = "20260915_unresolved_v2"


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default, ensure_ascii=False), encoding="utf-8")


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def z_of(tab: dict, name: str, n: int, comp: str) -> complex:
    return complex(*tab[name][str(n)][comp])


def h2_octuple(tab: dict) -> dict:
    out = {}
    for name in H2_NAMES:
        for comp in ("x", "y"):
            z = z_of(tab, name, 2, comp)
            out[f"{name}_{comp}"] = {"re": z.real, "im": z.imag, "abs": abs(z), "phase_deg": float(np.angle(z, deg=True)) if abs(z) else None}
    return out


def inversion_unpaired(kf) -> int:
    keys = {tuple(np.round(np.mod(row, 1.0), 12)) for row in np.asarray(kf)}
    n = 0
    for row in kf:
        partner = tuple(np.round(np.mod(-np.asarray(row), 1.0), 12))
        if partner not in keys:
            n += 1
    return n


def wrapped_phase_diff_deg(za: complex, zb: complex) -> float | None:
    if abs(za) == 0.0 or abs(zb) == 0.0:
        return None
    return float(np.angle(complex(za) * np.conj(complex(zb)), deg=True))


def classify_channel(
    z_ref: complex,
    *,
    z_k: complex | None = None,
    z_ind: complex | None = None,
    z_dt: complex | None = None,
    d_k: float | None = None,
    d_ind: float | None = None,
    d_dt: float | None = None,
    abs_floor: float | None = None,
) -> dict:
    """Map finite-check complex amplitudes onto the original task labels.

    `trend_not_qualified` means a repeatable nonzero value whose checks miss 5%,
    but the change is still smaller than |z_ref| and the phase does not flip.
    Changes comparable to or larger than the signal, or a >90° phase jump,
    are `unresolved`. A tiny reference modulus is not `below_reporting_resolution`
    if the absolute disagreements are themselves above the reporting floor.

    `abs_floor` defaults to the mean-current sheet-harmonic floor 1e-9.
    Source-kernel (CJJ) callers must pass their own same-unit floor.
    """
    floor = ABS_REPORT_RESOLUTION if abs_floor is None else float(abs_floor)
    mag = abs(z_ref)
    if d_k is None and z_k is not None:
        d_k = abs(z_k - z_ref)
    if d_ind is None and z_ind is not None:
        d_ind = abs(z_ind - z_ref)
    if d_dt is None and z_dt is not None:
        d_dt = abs(z_dt - z_ref)
    diffs = (d_k, d_ind, d_dt)
    reasons: list[str] = []
    phases = {
        "k": wrapped_phase_diff_deg(z_k, z_ref) if z_k is not None else None,
        "ind": wrapped_phase_diff_deg(z_ind, z_ref) if z_ind is not None else None,
        "dt": wrapped_phase_diff_deg(z_dt, z_ref) if z_dt is not None else None,
    }
    if any(d is None for d in diffs):
        reasons.append("missing_one_or_more_of_dk_dind_ddt")
        return {"status": "unresolved", "reasons": reasons, "phase_diff_deg": phases}

    present = [d for d in diffs if d is not None]
    max_d = max(present)
    if mag < floor and max_d < floor:
        reasons.append("reference_and_all_absolute_changes_below_reporting_resolution")
        return {"status": "below_reporting_resolution", "reasons": reasons, "phase_diff_deg": phases}
    if mag < floor and max_d >= floor:
        reasons.append("reference_below_reporting_resolution_but_absolute_changes_are_not")
        return {"status": "unresolved", "reasons": reasons, "phase_diff_deg": phases}

    phase_unstable = False
    for key, ang in phases.items():
        if ang is None:
            continue
        z_other = {"k": z_k, "ind": z_ind, "dt": z_dt}[key]
        if z_other is None:
            continue
        if abs(z_other) >= floor and abs(ang) > PHASE_UNSTABLE_DEG:
            phase_unstable = True
            reasons.append(f"phase_{key}_abs_{abs(ang):.2f}_deg_exceeds_{PHASE_UNSTABLE_DEG}")

    if all(d <= QUAL_REL * mag for d in present) and not phase_unstable:
        reasons.append("all_three_relative_changes_at_or_below_5pct")
        return {"status": "qualified_preliminary", "reasons": reasons, "phase_diff_deg": phases}

    if (not phase_unstable) and max_d < TREND_REL_CAP * mag:
        reasons.append("repeatable_nonzero_but_misses_5pct_and_change_smaller_than_|z_ref|")
        return {"status": "trend_not_qualified", "reasons": reasons, "phase_diff_deg": phases}

    if phase_unstable:
        reasons.append("phase_or_direction_unstable")
    if max_d >= TREND_REL_CAP * mag:
        reasons.append("absolute_change_comparable_to_or_larger_than_|z_ref|")
    if not reasons:
        reasons.append("failed_original_task_repeatability_boundary")
    return {"status": "unresolved", "reasons": reasons, "phase_diff_deg": phases}


def load_analysis(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def n_exc_local_from_npz(npz_path: Path) -> dict | None:
    if not npz_path.exists():
        return None
    d = np.load(npz_path)
    if "n_exc_k_peak" not in d.files:
        return None
    n = np.asarray(d["n_exc_k_peak"], dtype=float)
    w = np.asarray(d["w_k"], dtype=float) if "w_k" in d.files else np.full(n.size, 1.0 / n.size)
    frac = n / 4.0
    return {
        "source": str(npz_path),
        "bz_mean_at_global_peak_diag_n_exc_over_4": float(w @ frac),
        "local_max_n_exc_over_4_at_global_peak_snapshot": float(frac.max()),
        "fraction_k_above_0.25_at_that_snapshot": float(np.mean(frac > 0.25)),
        "note": (
            "n_exc_k_peak is the k-map at the diagnostic-sampled global mean peak, "
            "not each k's own historical maximum"
        ),
    }


def work_from_npz(npz_path: Path, a_cell: float) -> dict | None:
    if not npz_path.exists():
        return None
    d = np.load(npz_path)
    need = ("times", "J_sheet", "E_over")
    if any(k not in d.files for k in need):
        return None
    times = np.asarray(d["times"], dtype=float)
    jt = np.asarray(d["J_sheet"], dtype=float) * a_cell
    e = np.asarray(d["E_over"], dtype=float)
    work = float(np.trapezoid(np.sum(jt * e, axis=-1), times))
    out = {"source": str(npz_path), "work_from_current_and_E": work}
    if "delta_U_e" in d.files:
        out["delta_U_e_stored"] = float(d["delta_U_e"])
    if "work_integral" in d.files:
        out["work_integral_stored"] = float(d["work_integral"])
    return out


def archival_p0_complex() -> dict:
    rep = json.loads(OLD_CLASSICAL.read_text(encoding="utf-8"))
    sets = {"nk144": "k_fine2", "nk192": "k_fine3", "nk48": "base", "nk96": "k_fine"}
    tabs = {}
    for label, key in sets.items():
        tabs[label] = rep["sets"][key]["analysis"]["windows"]["sin2"]
    keys = [("J_e", 1, "x"), ("J_o", 2, "y"), ("J_o", 2, "x")]
    cmp144_192 = {}
    for name, n, comp in keys:
        za = z_of(tabs["nk144"], name, n, comp)
        zb = z_of(tabs["nk192"], name, n, comp)
        cmp144_192[f"{name}_{comp}_H{n}"] = compare_complex_amplitudes(za, zb)
    dc144 = float(rep["low_frequency_Jo"]["k_fine2"]["J_o"]["dc_sin2_y"])
    dc192 = float(rep["low_frequency_Jo"]["k_fine3"]["J_o"]["dc_sin2_y"])
    cmp144_192["J_o_y_H0"] = compare_complex_amplitudes(dc144, dc192)
    return {"compare_144_vs_192_sin2": cmp144_192, "nk192_octuple_H2": h2_octuple(tabs["nk192"])}


def stock_corrections(geo, gap) -> dict:
    eg = float(gap["E_g_direct"])
    a_cell = float(geo.a_cell_ang2)
    resources = {
        "P0": field_resources(geo, P0Protocol(), eg),
        "P1_A0.8": field_resources(geo, p1_protocol(0.8), eg),
    }
    p48 = load_analysis(OLD_CAMPAIGN / "p1_A0.8_nk48_nt400" / "analysis.json")
    p96 = load_analysis(OLD_CAMPAIGN / "p1_A0.8_nk96_nt400" / "analysis.json")
    p48dt = load_analysis(OLD_CAMPAIGN / "p1_A0.8_nk48_nt800" / "analysis.json")
    p0_96 = load_analysis(OLD_CAMPAIGN / "p0_nk96_nt400" / "analysis.json")
    tab48 = p48["analysis"]["windows"]["sin2"]
    tab96 = p96["analysis"]["windows"]["sin2"]
    tab48dt = p48dt["analysis"]["windows"]["sin2"]
    complex_48_96 = {}
    for name in H2_NAMES:
        for n in (0, 1, 2):
            for comp in ("x", "y"):
                complex_48_96[f"{name}_{comp}_H{n}"] = compare_complex_amplitudes(
                    z_of(tab48, name, n, comp), z_of(tab96, name, n, comp)
                )
    dt_48 = {}
    for name, n, comp in (("J_e", 1, "x"), ("J_o", 2, "y"), ("J_o", 2, "x")):
        dt_48[f"{name}_{comp}_H{n}"] = compare_complex_amplitudes(
            z_of(tab48, name, n, comp), z_of(tab48dt, name, n, comp)
        )
    due48 = p48["analysis"]["diagnostics"]["N+1_A+1"]["delta_U_e"]
    due96 = p96["analysis"]["diagnostics"]["N+1_A+1"]["delta_U_e"]
    due_p0_96 = p0_96["analysis"]["diagnostics"]["N+1_A+1"]["delta_U_e"]
    old_dir = REPO / "results/q_line/m0_v1_20260912/classical"
    old_npzs = sorted(old_dir.glob("*.npz")) if old_dir.exists() else []
    old_work = {}
    for fn in old_npzs:
        rec = work_from_npz(fn, a_cell)
        if rec:
            old_work[fn.name] = rec
    p1_96_npz = OLD_CAMPAIGN / "p1_A0.8_nk96_nt400" / "run_p1_A0.8_nk96_nt400_N+1_A+1.npz"
    local = n_exc_local_from_npz(p1_96_npz)
    kf0, _, w0 = k_grid(geo, 192, shift=0.0)
    kf1, _, w1 = k_grid(geo, 192, shift=0.5)
    plan = {
        "final_verification_level": {
            "nk": 192,
            "nt_per_cycle_main": 400,
            "reason": (
                "Existing 48→96 already fails the 5% complex-H2 test by a large margin. "
                "nk=192 is a 4× density step matching archival P0; independent quadrature "
                "and dt-halving are reserved for this level instead of spending the budget "
                "on a second unresolved k-refinement."
            ),
        },
        "budget": {
            "max_new_configs": 4,
            "planned": [
                "p1_A0.8_nk192_nt400",
                "p1_A0.8_nk192_nt400_shift",
                "p1_A0.8_nk192_nt800",
            ],
            "reserved_fourth": "only if 96→192 complex H2 change is within ~15% and a second k-level could still meet 5%",
        },
        "absolute_reporting_resolution": {
            "value": ABS_REPORT_RESOLUTION,
            "unit": ABS_REPORT_UNIT,
            "basis": (
                "Existing field-reversal residuals ~2e-15 relative to Jmax~3e-2 give ~1e-16 "
                "sheet-current noise. Unitarity/idempotence diagnostics are ~1e-13. "
                "1e-9 is a conservative reporting floor, still ~1e3 below the current |Jo(H2)|~1e-6 scale. "
                "It is a resolvability cut, not a physical-zero definition."
            ),
        },
        "quadrature": {
            "nk192_shift0_unpaired": inversion_unpaired(kf0),
            "nk192_shift0.5_unpaired": inversion_unpaired(kf1),
            "weight_sum_shift0": float(np.sum(w0)),
            "weight_sum_shift0.5": float(np.sum(w1)),
            "shift0.5_excludes_gamma": bool(np.min(np.linalg.norm(geo.k_cart(kf1), axis=1)) > 1e-8),
        },
    }
    out = {
        "postprocess_version": POSTPROCESS_VERSION,
        "model_id": MODEL_ID,
        "parameter_set": "R0",
        "classical_protocol": "P1_Astar0.8_8cycles_original_pulse",
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "m0_cjj_physics": "not_run",
        "cjj_definition": "pending_human_review",
        "field_resources_corrected": resources,
        "p1_A0.8_complex_48_vs_96": complex_48_96,
        "p1_A0.8_dt_nk48_400_vs_800": dt_48,
        "p1_A0.8_h2_octuple_nk48": h2_octuple(tab48),
        "p1_A0.8_h2_octuple_nk96": h2_octuple(tab96),
        "delta_U_e_Nplus": {
            "p1_nk48": due48,
            "p1_nk96": due96,
            "p0_nk96": due_p0_96,
            "ratio_p1_nk96_over_p0_nk96": due96 / due_p0_96,
            "p1_48_to_96_rel_vs_96": abs(due48 - due96) / abs(due96),
            "note": "2.25 is the nk96/nk96 N=+1 ΔUe ratio, not a converged constant",
        },
        "n_exc_local_nk96_Nplus": local,
        "old_p0_npz_work": {
            "found_npz": [p.name for p in old_npzs],
            "work": old_work,
            "note": "If found_npz is empty, archival work recovery is unavailable in this workspace; JSON harmonics still exist.",
        },
        "archival_p0": archival_p0_complex(),
        "execution_plan": plan,
    }
    return out


def run_one(geo, mu, name: str, nk: int, ntpc: int, shift: float) -> dict:
    prot = p1_protocol(0.8)
    outdir = OUT_ROOT / name
    t0 = time.perf_counter()
    runs = run_set(
        geo, R0, prot, mu, nk, ntpc,
        signs=(+1,), neels=(+1, -1), tag=name, outdir=outdir, verbose=True,
        grid_shift=shift, subspace=False, k_orders=(),
    )
    rec = {
        "name": name,
        "nk": nk,
        "ntpc": ntpc,
        "grid_shift": shift,
        "protocol_id": prot.protocol_id,
        "wall_seconds_pair": time.perf_counter() - t0,
        "postprocess_version": POSTPROCESS_VERSION,
    }
    ana, _traces = analyse_set(runs, prot, ORDERS)
    rec["analysis"] = ana
    rec["targets_sin2"] = target_quantities(ana["windows"]["sin2"])
    rec["h2_octuple_sin2"] = h2_octuple(ana["windows"]["sin2"])
    rec["Je_x_H1"] = rec["targets_sin2"]["Je_x_H1"]
    rec["physics_identity"] = {
        "N+1": json.loads((outdir / f"run_{name}_N+1_A+1.meta.json").read_text(encoding="utf-8")).get("physics_identity_sha256"),
        "N-1": json.loads((outdir / f"run_{name}_N-1_A+1.meta.json").read_text(encoding="utf-8")).get("physics_identity_sha256"),
    }
    dump(outdir / "analysis.json", rec)
    return rec


def summarize(stock: dict, jobs: dict) -> dict:
    tab96 = load_analysis(OLD_CAMPAIGN / "p1_A0.8_nk96_nt400" / "analysis.json")["analysis"]["windows"]["sin2"]
    ref_name = "p1_A0.8_nk192_nt400" if "p1_A0.8_nk192_nt400" in jobs else None
    channels = {}
    if ref_name:
        tab_ref = jobs[ref_name]["analysis"]["windows"]["sin2"]
        tab_shift = jobs.get("p1_A0.8_nk192_nt400_shift", {}).get("analysis", {}).get("windows", {}).get("sin2")
        tab_dt = jobs.get("p1_A0.8_nk192_nt800", {}).get("analysis", {}).get("windows", {}).get("sin2")
        for name, n, comp, label in (
            ("J_o", 2, "x", "Jo_x_H2"),
            ("J_o", 2, "y", "Jo_y_H2"),
            ("J_e", 1, "x", "Je_x_H1"),
        ):
            z_ref = z_of(tab_ref, name, n, comp)
            z96 = z_of(tab96, name, n, comp)
            z_shift = z_of(tab_shift, name, n, comp) if tab_shift else None
            z_fine_dt = z_of(tab_dt, name, n, comp) if tab_dt else None
            d_k = abs(z96 - z_ref)
            d_ind = abs(z_shift - z_ref) if z_shift is not None else None
            d_dt = abs(z_fine_dt - z_ref) if z_fine_dt is not None else None
            verdict = classify_channel(
                z_ref, z_k=z96, z_ind=z_shift, z_dt=z_fine_dt, d_k=d_k, d_ind=d_ind, d_dt=d_dt,
            )
            channels[label] = {
                "z_ref": {"re": z_ref.real, "im": z_ref.imag, "abs": abs(z_ref)},
                "d_k": d_k,
                "d_ind": d_ind,
                "d_dt": d_dt,
                "rel_k": d_k / abs(z_ref) if abs(z_ref) else None,
                "rel_ind": (d_ind / abs(z_ref)) if d_ind is not None and abs(z_ref) else None,
                "rel_dt": (d_dt / abs(z_ref)) if d_dt is not None and abs(z_ref) else None,
                "threshold_5pct_of_|z_ref|": QUAL_REL * abs(z_ref),
                "status": verdict["status"],
                "status_legacy_10x_rule": "trend_not_qualified" if label.startswith("Jo_") else verdict["status"],
                "status_correction_reason": verdict["reasons"],
                "phase_diff_deg": verdict["phase_diff_deg"],
                "empirical_change_scale": max(d for d in (d_k, d_ind, d_dt) if d is not None) if any(d is not None for d in (d_k, d_ind, d_dt)) else None,
                "empirical_change_scale_is_not": "not a statistical CI or a strict continuum error bound",
            }
    return {
        "model": "M0",
        "parameter_set": "R0",
        "classical_protocol": "P1_Astar0.8_8cycles_original_pulse",
        "h2_qualification_x": channels.get("Jo_x_H2", {}).get("status", "unresolved"),
        "h2_qualification_y": channels.get("Jo_y_H2", {}).get("status", "unresolved"),
        "cjj_definition": "pending_human_review",
        "cjj_algebra_tests": "PASS; results/q_line/cjj_algebra_fixture_20260915/",
        "m0_cjj_physics": "not_run",
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "classification_rule": CLASSIFICATION_RULE,
        "classification_rule_note": (
            "trend_not_qualified no longer accepts |Δz| up to 10|z_ref|; "
            "phase flips and changes >= |z_ref| are unresolved. "
            "Numerical z values are unchanged."
        ),
        "channels": channels,
        "jobs_present": list(jobs),
        "absolute_reporting_resolution": ABS_REPORT_RESOLUTION,
        "stock_key_spectrum_P1": stock["field_resources_corrected"]["P1_A0.8"]["E_spectrum_weight_above_direct_gap"],
        "stock_key_spectrum_P0": stock["field_resources_corrected"]["P0"]["E_spectrum_weight_above_direct_gap"],
    }


def complex_plane_figure(stock: dict, jobs: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.6))
    series = [("nk48", stock["p1_A0.8_h2_octuple_nk48"]), ("nk96", stock["p1_A0.8_h2_octuple_nk96"])]
    if "p1_A0.8_nk192_nt400" in jobs:
        series.append(("nk192", jobs["p1_A0.8_nk192_nt400"]["h2_octuple_sin2"]))
    if "p1_A0.8_nk192_nt400_shift" in jobs:
        series.append(("nk192_shift", jobs["p1_A0.8_nk192_nt400_shift"]["h2_octuple_sin2"]))
    if "p1_A0.8_nk192_nt800" in jobs:
        series.append(("nk192_dt800", jobs["p1_A0.8_nk192_nt800"]["h2_octuple_sin2"]))
    for ax, key, title in (
        (axs[0], "J_o_x", r"$J_{o,x}(H_2)$"),
        (axs[1], "J_o_y", r"$J_{o,y}(H_2)$"),
    ):
        for label, octu in series:
            z = octu[key]
            ax.scatter([z["re"]], [z["im"]], label=label)
            ax.annotate(label, (z["re"], z["im"]), fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Re z")
        ax.set_ylabel("Im z")
        ax.axhline(0.0, color="k", lw=0.4)
        ax.axvline(0.0, color="k", lw=0.4)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("stock", "nk192", "shift", "dt", "auto", "summarize"), default="stock")
    args = ap.parse_args(argv)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    gap = json.loads(OLD_STATIC.read_text(encoding="utf-8"))
    mu = float(gap["mu_half_filling"])
    stock_path = OUT_ROOT / "stock_corrections.json"
    if args.stage == "stock" or not stock_path.exists():
        stock = stock_corrections(geo, gap)
        dump(stock_path, stock)
        dump(OLD_CAMPAIGN / "field_resources_20260915.json", stock["field_resources_corrected"])
        dump(OUT_ROOT / "field_resources.json", stock["field_resources_corrected"])
    else:
        stock = json.loads(stock_path.read_text(encoding="utf-8"))
    jobs = {}
    for name in ("p1_A0.8_nk192_nt400", "p1_A0.8_nk192_nt400_shift", "p1_A0.8_nk192_nt800"):
        ana = OUT_ROOT / name / "analysis.json"
        if ana.exists():
            jobs[name] = load_analysis(ana)
    if args.stage in ("nk192", "auto"):
        jobs["p1_A0.8_nk192_nt400"] = run_one(geo, mu, "p1_A0.8_nk192_nt400", 192, 400, 0.0)
    if args.stage in ("shift", "auto"):
        jobs["p1_A0.8_nk192_nt400_shift"] = run_one(geo, mu, "p1_A0.8_nk192_nt400_shift", 192, 400, 0.5)
    if args.stage in ("dt", "auto"):
        jobs["p1_A0.8_nk192_nt800"] = run_one(geo, mu, "p1_A0.8_nk192_nt800", 192, 800, 0.0)
    if args.stage == "auto" and "p1_A0.8_nk192_nt400" in jobs:
        y = jobs["p1_A0.8_nk192_nt400"]["h2_octuple_sin2"]["J_o_y"]
        x = jobs["p1_A0.8_nk192_nt400"]["h2_octuple_sin2"]["J_o_x"]
        z96y = complex(stock["p1_A0.8_h2_octuple_nk96"]["J_o_y"]["re"], stock["p1_A0.8_h2_octuple_nk96"]["J_o_y"]["im"])
        z96x = complex(stock["p1_A0.8_h2_octuple_nk96"]["J_o_x"]["re"], stock["p1_A0.8_h2_octuple_nk96"]["J_o_x"]["im"])
        zy = complex(y["re"], y["im"])
        zx = complex(x["re"], x["im"])
        rel = max(abs(z96y - zy) / max(abs(zy), 1e-300), abs(z96x - zx) / max(abs(zx), 1e-300))
        fourth = {"used": False, "rel_96_to_192": rel}
        if rel <= 0.15:
            fourth = {"used": True, "name": "p1_A0.8_nk256_nt400", "rel_96_to_192": rel}
            jobs["p1_A0.8_nk256_nt400"] = run_one(geo, mu, "p1_A0.8_nk256_nt400", 256, 400, 0.0)
        dump(OUT_ROOT / "fourth_config.json", fourth)
    summary = summarize(stock, jobs)
    fourth_path = OUT_ROOT / "fourth_config.json"
    if fourth_path.exists():
        fourth = json.loads(fourth_path.read_text(encoding="utf-8"))
        summary["fourth_config"] = fourth
        summary["fourth_config_used"] = bool(fourth.get("used"))
    dump(OUT_ROOT / "h2_qualification_summary.json", summary)
    dump(OUT_ROOT / "campaign_status.json", {"jobs": list(jobs), "summary_fields": {k: summary[k] for k in summary if k != "channels"}, "channels": summary.get("channels")})
    try:
        complex_plane_figure(stock, jobs, OUT_ROOT / "fig_h2_complex_plane.png")
    except Exception as exc:  # plotting must not hide numerical results
        dump(OUT_ROOT / "figure_error.json", {"error": str(exc)})
    print(json.dumps({k: summary[k] for k in ("h2_qualification_x", "h2_qualification_y", "jobs_present")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
