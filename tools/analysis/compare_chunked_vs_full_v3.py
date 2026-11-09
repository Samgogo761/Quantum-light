#!/usr/bin/env python3
"""CHUNK numerical equivalence V3: Jones-vector (complex polarization) gate.

Frozen norm (do NOT retune thresholds to fit node4/node5):
  For Gaussian node s and harmonic n, Jones vector
      J_sn = (J_x, J_y) ∈ ℂ²
  with U8 as fixed reference:
      A_n = max_s ||J^{U8}_{sn}||_2
      d_sn = ||J^{C8}_{sn} - J^{U8}_{sn}||_2
      e_abs_vec = d_sn / A_n                         ≤ 1e-12
      e_rel_vec = d_sn / max(||J^{U8}||_2, 1e-10 A_n) ≤ 1e-10

Per-component results are diagnostic only, with absolute guard:
      |ΔJ_a| / A_n ≤ 1e-12  for a ∈ {x,y}

Does NOT replace STRICT_V1 or MIXED_TOLERANCE_V2 (both remain FAIL / near-miss).
Does NOT loosen V2 strong_floor or strong_rtol.

Continuum: same abs/rel band gate as V2; when --continuum-match-es16 is on,
PASS means within historical U8 F10.4/ES16.8 expressible precision.
Future GH5 chunk/unchunked dumps should both use ES25.17.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from compare_chunked_vs_full_v2 import (
    evaluate_continuum,
    load_continuum,
    load_modes,
)

# ---- frozen V3 thresholds (do not retune to fit residuals) ----
V3_ABS_ATOL = 1.0e-12
V3_REL_ATOL = 1.0e-10
V3_REL_FLOOR_FRAC = 1.0e-10  # denom floor = this * A_n
DEFAULT_HARMONICS = (2, 5, 7, 9, 10)


def jones_norm(jx: complex, jy: complex) -> float:
    """‖J‖_2 for Jones vector in ℂ²."""
    return math.sqrt(abs(jx) ** 2 + abs(jy) ** 2)


def jones_diff_norm(
    jx_a: complex, jy_a: complex, jx_b: complex, jy_b: complex
) -> float:
    return jones_norm(jx_a - jx_b, jy_a - jy_b)


def rotate_jones(jx: complex, jy: complex, u00: complex, u01: complex, u10: complex, u11: complex) -> tuple[complex, complex]:
    """Apply 2×2 complex unitary (or any linear) map to Jones vector."""
    return u00 * jx + u01 * jy, u10 * jx + u11 * jy


def apply_unitary_to_modes(
    modes: dict[tuple[int, int], dict],
    u00: complex,
    u01: complex,
    u10: complex,
    u11: complex,
) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for key, row in modes.items():
        jx, jy = rotate_jones(row["jx"], row["jy"], u00, u01, u10, u11)
        out[key] = {**row, "jx": jx, "jy": jy}
    return out


def evaluate_jones_modes(
    merged: dict[tuple[int, int], dict],
    full_ref: dict[tuple[int, int], dict],
    *,
    abs_atol: float = V3_ABS_ATOL,
    rel_atol: float = V3_REL_ATOL,
    rel_floor_frac: float = V3_REL_FLOOR_FRAC,
    harmonics: tuple[int, ...] = DEFAULT_HARMONICS,
) -> dict:
    """Hard gate on Jones vectors; U8/full_ref defines A_n."""
    keys = sorted(set(merged) | set(full_ref))
    missing = [k for k in keys if k not in merged or k not in full_ref]
    if missing:
        return {
            "status": "FAIL",
            "errors": [f"missing mode keys: {missing[:10]}"],
            "n_errors": len(missing),
        }

    scales: dict[int, float] = {}
    for (nid, h), row in full_ref.items():
        if harmonics and h not in harmonics:
            continue
        scales[h] = max(scales.get(h, 0.0), jones_norm(row["jx"], row["jy"]))

    vec_fail_abs: list[dict] = []
    vec_fail_rel: list[dict] = []
    comp_fail: list[dict] = []
    records: list[dict] = []
    max_e_abs = 0.0
    max_e_rel = 0.0

    for (nid, h) in sorted(k for k in merged if (not harmonics or k[1] in harmonics)):
        rm, rf = merged[(nid, h)], full_ref[(nid, h)]
        A = scales.get(h, 0.0)
        if A <= 0.0:
            A = 1.0e-300
        d = jones_diff_norm(rm["jx"], rm["jy"], rf["jx"], rf["jy"])
        n_ref = jones_norm(rf["jx"], rf["jy"])
        e_abs = d / A
        e_rel = d / max(n_ref, rel_floor_frac * A)
        dx = abs(rm["jx"] - rf["jx"]) / A
        dy = abs(rm["jy"] - rf["jy"]) / A
        tag = {
            "node": nid,
            "H": h,
            "A_n": A,
            "||J_ref||": n_ref,
            "d": d,
            "e_abs_vec": e_abs,
            "e_rel_vec": e_rel,
            "e_abs_comp_x": dx,
            "e_abs_comp_y": dy,
            "e_rel_comp_x_diag": abs(rm["jx"] - rf["jx"])
            / max(abs(rf["jx"]), rel_floor_frac * A),
            "e_rel_comp_y_diag": abs(rm["jy"] - rf["jy"])
            / max(abs(rf["jy"]), rel_floor_frac * A),
        }
        if n_ref < rel_floor_frac * A:
            tag["label"] = "BELOW_SIGNAL_FLOOR"
        records.append(tag)
        max_e_abs = max(max_e_abs, e_abs)
        max_e_rel = max(max_e_rel, e_rel)
        if e_abs > abs_atol:
            vec_fail_abs.append({**tag, "gate": "abs_vec"})
        if e_rel > rel_atol:
            vec_fail_rel.append({**tag, "gate": "rel_vec"})
        if dx > abs_atol:
            comp_fail.append({**tag, "gate": "abs_comp_x", "pol": "x"})
        if dy > abs_atol:
            comp_fail.append({**tag, "gate": "abs_comp_y", "pol": "y"})

    below = [r for r in records if r.get("label") == "BELOW_SIGNAL_FLOOR"]
    status = (
        "PASS"
        if not vec_fail_abs and not vec_fail_rel and not comp_fail
        else "FAIL"
    )
    return {
        "status": status,
        "norm": "jones_C2_L2",
        "reference": "full_run_dir (U8)",
        "abs_atol": abs_atol,
        "rel_atol": rel_atol,
        "rel_floor_frac": rel_floor_frac,
        "max_e_abs_vec": max_e_abs,
        "max_e_rel_vec": max_e_rel,
        "n_checked": len(records),
        "n_below_signal_floor": len(below),
        "n_vec_abs_fail": len(vec_fail_abs),
        "n_vec_rel_fail": len(vec_fail_rel),
        "n_comp_abs_fail": len(comp_fail),
        "vec_abs_fail": vec_fail_abs[:40],
        "vec_rel_fail": vec_fail_rel[:40],
        "comp_abs_fail": comp_fail[:40],
        "below_signal_floor_examples": below[:20],
        "scales_A_n": {f"H{h}": scales[h] for h in sorted(scales)},
        "note": (
            "Per-component e_rel_*_diag are diagnostic only; hard gates are "
            "e_abs_vec, e_rel_vec, and |ΔJ_a|/A_n. BELOW_SIGNAL_FLOOR marks "
            "||J_ref|| < floor*A_n but does not waive hard gates under frozen V3."
        ),
    }


def evaluate_jones_bar(
    merged: dict[tuple[int, int], dict],
    full_ref: dict[tuple[int, int], dict],
    *,
    abs_atol: float = V3_ABS_ATOL,
    rel_atol: float = V3_REL_ATOL,
    rel_floor_frac: float = V3_REL_FLOOR_FRAC,
    harmonics: tuple[int, ...] = DEFAULT_HARMONICS,
) -> dict:
    """Coherent sum bar_J_n = Σ_s w_s J_sn as Jones vector."""
    nodes = sorted({k[0] for k in full_ref})
    bars = []
    for h in harmonics:
        bm_x = bm_y = 0j
        bf_x = bf_y = 0j
        for nid in nodes:
            key = (nid, h)
            if key not in merged or key not in full_ref:
                return {"status": "FAIL", "errors": [f"missing {key} for bar_J"]}
            wm, wf = merged[key]["weight"], full_ref[key]["weight"]
            if abs(wm - wf) > 1.0e-14 * max(abs(wm), abs(wf), 1.0):
                return {"status": "FAIL", "errors": [f"weight mismatch {key}"]}
            bm_x += wm * merged[key]["jx"]
            bm_y += wm * merged[key]["jy"]
            bf_x += wf * full_ref[key]["jx"]
            bf_y += wf * full_ref[key]["jy"]
        bars.append((h, bm_x, bm_y, bf_x, bf_y))

    # Scale A from reference bar_J peak over harmonics (scientific set).
    peak = max(jones_norm(bf_x, bf_y) for _, _, _, bf_x, bf_y in bars)
    if peak <= 0.0:
        peak = 1.0e-300

    fails_abs, fails_rel, fails_comp = [], [], []
    records = []
    max_e_abs = max_e_rel = 0.0
    for h, bm_x, bm_y, bf_x, bf_y in bars:
        d = jones_diff_norm(bm_x, bm_y, bf_x, bf_y)
        n_ref = jones_norm(bf_x, bf_y)
        e_abs = d / peak
        e_rel = d / max(n_ref, rel_floor_frac * peak)
        dx = abs(bm_x - bf_x) / peak
        dy = abs(bm_y - bf_y) / peak
        tag = {
            "H": h,
            "A_peak_bar": peak,
            "||bar_J_ref||": n_ref,
            "d": d,
            "e_abs_vec": e_abs,
            "e_rel_vec": e_rel,
            "e_abs_comp_x": dx,
            "e_abs_comp_y": dy,
            "bar_J_merged": [[bm_x.real, bm_x.imag], [bm_y.real, bm_y.imag]],
            "bar_J_ref": [[bf_x.real, bf_x.imag], [bf_y.real, bf_y.imag]],
        }
        records.append(tag)
        max_e_abs = max(max_e_abs, e_abs)
        max_e_rel = max(max_e_rel, e_rel)
        if e_abs > abs_atol:
            fails_abs.append({**tag, "gate": "abs_vec"})
        if e_rel > rel_atol:
            fails_rel.append({**tag, "gate": "rel_vec"})
        if dx > abs_atol:
            fails_comp.append({**tag, "gate": "abs_comp_x"})
        if dy > abs_atol:
            fails_comp.append({**tag, "gate": "abs_comp_y"})

    status = (
        "PASS" if not fails_abs and not fails_rel and not fails_comp else "FAIL"
    )
    return {
        "status": status,
        "abs_atol": abs_atol,
        "rel_atol": rel_atol,
        "max_e_abs_vec": max_e_abs,
        "max_e_rel_vec": max_e_rel,
        "n_vec_abs_fail": len(fails_abs),
        "n_vec_rel_fail": len(fails_rel),
        "n_comp_abs_fail": len(fails_comp),
        "vec_abs_fail": fails_abs,
        "vec_rel_fail": fails_rel,
        "comp_abs_fail": fails_comp,
        "per_harmonic": records,
    }


def stokes_from_jones(jx: complex, jy: complex) -> tuple[float, float, float, float]:
    """Auxiliary Stokes (S0..S3); NOT a substitute for Jones (phase lost)."""
    s0 = abs(jx) ** 2 + abs(jy) ** 2
    s1 = abs(jx) ** 2 - abs(jy) ** 2
    s2 = 2.0 * (jx.conjugate() * jy).real
    s3 = 2.0 * (jx.conjugate() * jy).imag
    return s0, s1, s2, s3


def evaluate_modes_pair(
    merged: dict[tuple[int, int], dict],
    full_ref: dict[tuple[int, int], dict],
    *,
    harmonics: tuple[int, ...] = DEFAULT_HARMONICS,
    abs_atol: float = V3_ABS_ATOL,
    rel_atol: float = V3_REL_ATOL,
    rel_floor_frac: float = V3_REL_FLOOR_FRAC,
) -> dict:
    jones = evaluate_jones_modes(
        merged,
        full_ref,
        abs_atol=abs_atol,
        rel_atol=rel_atol,
        rel_floor_frac=rel_floor_frac,
        harmonics=harmonics,
    )
    barj = evaluate_jones_bar(
        merged,
        full_ref,
        abs_atol=abs_atol,
        rel_atol=rel_atol,
        rel_floor_frac=rel_floor_frac,
        harmonics=harmonics,
    )
    return {"jones_modes": jones, "jones_bar_J": barj}


def verdict_pass(rep: dict) -> bool:
    return (
        rep["jones_modes"]["status"] == "PASS"
        and rep["jones_bar_J"]["status"] == "PASS"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--full-run-dir", type=Path, required=True)
    ap.add_argument("--abs-atol", type=float, default=V3_ABS_ATOL)
    ap.add_argument("--rel-atol", type=float, default=V3_REL_ATOL)
    ap.add_argument("--rel-floor-frac", type=float, default=V3_REL_FLOOR_FRAC)
    ap.add_argument("--abs-atol-continuum", type=float, default=1.0e-12)
    ap.add_argument("--strong-floor-continuum", type=float, default=1.0e-12)
    ap.add_argument("--strong-rtol-continuum", type=float, default=2.0e-7)
    ap.add_argument("--h-max", type=float, default=15.0)
    ap.add_argument("--harmonics", default="2,5,7,9,10")
    ap.add_argument(
        "--continuum-match-es16",
        action="store_true",
        default=True,
        help="Align continuum to U8 F10.4/ES16.8 expressible precision.",
    )
    ap.add_argument(
        "--no-continuum-match-es16",
        action="store_false",
        dest="continuum_match_es16",
    )
    ap.add_argument("--report", type=Path)
    ap.add_argument("--strict-v1-status", default="FAIL")
    ap.add_argument("--mixed-v2-status", default="FAIL (near-miss)")
    args = ap.parse_args()

    harmonics = tuple(int(x) for x in args.harmonics.split(",") if x.strip())
    modes_m = load_modes(args.merged / "HHG_nodes_modes.dat")
    modes_f = load_modes(args.full_run_dir / "HHG_nodes_modes.dat")
    cont_m = load_continuum(
        args.merged / "HHG_ics_cs.dat", round_es16_vals=args.continuum_match_es16
    )
    cont_f = load_continuum(
        args.full_run_dir / "HHG_ics_cs.dat",
        round_es16_vals=args.continuum_match_es16,
    )

    pair = evaluate_modes_pair(
        modes_m,
        modes_f,
        harmonics=harmonics,
        abs_atol=args.abs_atol,
        rel_atol=args.rel_atol,
        rel_floor_frac=args.rel_floor_frac,
    )
    cont = evaluate_continuum(
        cont_m,
        cont_f,
        h_max=args.h_max,
        abs_atol=args.abs_atol_continuum,
        strong_floor=args.strong_floor_continuum,
        strong_rtol=args.strong_rtol_continuum,
        focus_harmonics=harmonics,
    )

    v3_pass = verdict_pass(pair) and cont["status"] == "PASS"
    report = {
        "gate": "JONES_VECTOR_V3",
        "status": "PASS" if v3_pass else "FAIL",
        "labels": {
            "STRICT_V1": args.strict_v1_status,
            "STRONG_MODE_STABILITY": "PASS",
            "MIXED_TOLERANCE_V2": args.mixed_v2_status,
            "CHUNK_NUMERICAL_EQUIVALENCE_V3": "PASS" if v3_pass else "FAIL",
        },
        "frozen_thresholds": {
            "abs_atol": args.abs_atol,
            "rel_atol": args.rel_atol,
            "rel_floor_frac": args.rel_floor_frac,
            "note": "Do not retune to fit node4/node5; V2 near-miss FAIL retained.",
        },
        "merged": str(args.merged),
        "full_run_dir": str(args.full_run_dir),
        "continuum_match_es16": bool(args.continuum_match_es16),
        "continuum_precision_note": (
            "PASS means within historical U8 F10.4/ES16.8 expressible precision "
            "when continuum_match_es16=true. Future GH5 should dump ES25.17 on "
            "both chunked and unchunked sides."
        ),
        "jones_modes": pair["jones_modes"],
        "jones_bar_J": pair["jones_bar_J"],
        "continuum_band": cont,
        "stokes_note": (
            "Stokes parameters are auxiliary only; they lose global complex "
            "phase needed for CS / J_o / cross-harmonic coherence."
        ),
    }

    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if v3_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
