#!/usr/bin/env python3
"""CHUNK numerical equivalence V3.1: Jones vector + fixed mixed-tolerance logic.

Preserves frozen numerical thresholds from V3:
  η = 1e-10,  τ_abs = 1e-12,  τ_rel = 1e-10

Logic fix (NOT threshold retuning):
  all nodes:  e_abs_vec = d/A_n ≤ τ_abs
              |ΔJ_a|/A_n ≤ τ_abs  (diagnostic absolute guard)
  if ||J_ref|| ≥ η A_n  (strong):
              e_rel_vec = d/||J_ref|| ≤ τ_rel   (hard)
  else:
              label BELOW_SIGNAL_FLOOR
              e_rel_vec diagnostic only (no verdict)

Historical FAIL records retained:
  STRICT_V1=FAIL, V2=FAIL(near-miss), V3=FAIL (contradictory rel-on-floor).

Continuum: same as V2/V3 with ES16.8 expressible-precision note for U8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_chunked_vs_full_v2 import (
    evaluate_continuum,
    load_continuum,
    load_modes,
)
from compare_chunked_vs_full_v3 import (
    DEFAULT_HARMONICS,
    V3_ABS_ATOL,
    V3_REL_ATOL,
    V3_REL_FLOOR_FRAC,
    apply_unitary_to_modes,
    jones_diff_norm,
    jones_norm,
)

# Same numbers as V3 — do not retune.
V31_ABS_ATOL = V3_ABS_ATOL
V31_REL_ATOL = V3_REL_ATOL
V31_ETA = V3_REL_FLOOR_FRAC  # η = 1e-10


def evaluate_jones_modes_v31(
    merged: dict[tuple[int, int], dict],
    full_ref: dict[tuple[int, int], dict],
    *,
    abs_atol: float = V31_ABS_ATOL,
    rel_atol: float = V31_REL_ATOL,
    eta: float = V31_ETA,
    harmonics: tuple[int, ...] = DEFAULT_HARMONICS,
) -> dict:
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
    max_e_rel_strong = 0.0
    max_e_rel_diag = 0.0

    for (nid, h) in sorted(k for k in merged if (not harmonics or k[1] in harmonics)):
        rm, rf = merged[(nid, h)], full_ref[(nid, h)]
        A = scales.get(h, 0.0)
        if A <= 0.0:
            A = 1.0e-300
        d = jones_diff_norm(rm["jx"], rm["jy"], rf["jx"], rf["jy"])
        n_ref = jones_norm(rf["jx"], rf["jy"])
        e_abs = d / A
        # Diagnostic relative always uses ||J_ref|| (or tiny eps); hard rel only if strong.
        e_rel_diag = d / max(n_ref, 1.0e-300)
        is_strong = n_ref >= eta * A
        dx = abs(rm["jx"] - rf["jx"]) / A
        dy = abs(rm["jy"] - rf["jy"]) / A
        tag = {
            "node": nid,
            "H": h,
            "A_n": A,
            "||J_ref||": n_ref,
            "d": d,
            "e_abs_vec": e_abs,
            "e_rel_vec_diag": e_rel_diag,
            "e_abs_comp_x": dx,
            "e_abs_comp_y": dy,
            "strong": is_strong,
        }
        if is_strong:
            e_rel = d / n_ref  # hard: denom is ||J_ref|| only
            tag["e_rel_vec"] = e_rel
        else:
            tag["label"] = "BELOW_SIGNAL_FLOOR"
            tag["e_rel_vec"] = e_rel_diag  # diagnostic only
        records.append(tag)
        max_e_abs = max(max_e_abs, e_abs)
        max_e_rel_diag = max(max_e_rel_diag, e_rel_diag)

        if e_abs > abs_atol:
            vec_fail_abs.append({**tag, "gate": "abs_vec"})
        if dx > abs_atol:
            comp_fail.append({**tag, "gate": "abs_comp_x", "pol": "x"})
        if dy > abs_atol:
            comp_fail.append({**tag, "gate": "abs_comp_y", "pol": "y"})
        if is_strong:
            max_e_rel_strong = max(max_e_rel_strong, tag["e_rel_vec"])
            if tag["e_rel_vec"] > rel_atol:
                vec_fail_rel.append({**tag, "gate": "rel_vec"})

    below = [r for r in records if r.get("label") == "BELOW_SIGNAL_FLOOR"]
    status = (
        "PASS"
        if not vec_fail_abs and not vec_fail_rel and not comp_fail
        else "FAIL"
    )
    return {
        "status": status,
        "gate_version": "V3.1",
        "norm": "jones_C2_L2",
        "reference": "full_run_dir (U8)",
        "abs_atol": abs_atol,
        "rel_atol": rel_atol,
        "eta": eta,
        "max_e_abs_vec": max_e_abs,
        "max_e_rel_vec_strong": max_e_rel_strong,
        "max_e_rel_vec_diag_all": max_e_rel_diag,
        "n_checked": len(records),
        "n_strong": sum(1 for r in records if r["strong"]),
        "n_below_signal_floor": len(below),
        "n_vec_abs_fail": len(vec_fail_abs),
        "n_vec_rel_fail": len(vec_fail_rel),
        "n_comp_abs_fail": len(comp_fail),
        "vec_abs_fail": vec_fail_abs[:40],
        "vec_rel_fail": vec_fail_rel[:40],
        "comp_abs_fail": comp_fail[:40],
        "below_signal_floor_examples": below[:20],
        "scales_A_n": {f"H{h}": scales[h] for h in sorted(scales)},
        "logic": (
            "all: abs; if strong: also rel with denom=||J_ref||; "
            "else BELOW_SIGNAL_FLOOR and rel diagnostic only"
        ),
    }


def evaluate_jones_bar_v31(
    merged: dict[tuple[int, int], dict],
    full_ref: dict[tuple[int, int], dict],
    *,
    abs_atol: float = V31_ABS_ATOL,
    rel_atol: float = V31_REL_ATOL,
    eta: float = V31_ETA,
    harmonics: tuple[int, ...] = DEFAULT_HARMONICS,
) -> dict:
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

    peak = max(jones_norm(bf_x, bf_y) for _, _, _, bf_x, bf_y in bars)
    if peak <= 0.0:
        peak = 1.0e-300

    fails_abs, fails_rel, fails_comp = [], [], []
    records = []
    max_e_abs = max_e_rel_strong = 0.0
    for h, bm_x, bm_y, bf_x, bf_y in bars:
        d = jones_diff_norm(bm_x, bm_y, bf_x, bf_y)
        n_ref = jones_norm(bf_x, bf_y)
        e_abs = d / peak
        is_strong = n_ref >= eta * peak
        dx = abs(bm_x - bf_x) / peak
        dy = abs(bm_y - bf_y) / peak
        tag = {
            "H": h,
            "A_peak_bar": peak,
            "||bar_J_ref||": n_ref,
            "d": d,
            "e_abs_vec": e_abs,
            "e_abs_comp_x": dx,
            "e_abs_comp_y": dy,
            "strong": is_strong,
            "bar_J_merged": [[bm_x.real, bm_x.imag], [bm_y.real, bm_y.imag]],
            "bar_J_ref": [[bf_x.real, bf_x.imag], [bf_y.real, bf_y.imag]],
        }
        if is_strong:
            e_rel = d / n_ref
            tag["e_rel_vec"] = e_rel
            max_e_rel_strong = max(max_e_rel_strong, e_rel)
        else:
            tag["label"] = "BELOW_SIGNAL_FLOOR"
            tag["e_rel_vec"] = d / max(n_ref, 1.0e-300)
        records.append(tag)
        max_e_abs = max(max_e_abs, e_abs)
        if e_abs > abs_atol:
            fails_abs.append({**tag, "gate": "abs_vec"})
        if dx > abs_atol:
            fails_comp.append({**tag, "gate": "abs_comp_x"})
        if dy > abs_atol:
            fails_comp.append({**tag, "gate": "abs_comp_y"})
        if is_strong and tag["e_rel_vec"] > rel_atol:
            fails_rel.append({**tag, "gate": "rel_vec"})

    status = (
        "PASS" if not fails_abs and not fails_rel and not fails_comp else "FAIL"
    )
    return {
        "status": status,
        "gate_version": "V3.1",
        "abs_atol": abs_atol,
        "rel_atol": rel_atol,
        "eta": eta,
        "max_e_abs_vec": max_e_abs,
        "max_e_rel_vec_strong": max_e_rel_strong,
        "n_vec_abs_fail": len(fails_abs),
        "n_vec_rel_fail": len(fails_rel),
        "n_comp_abs_fail": len(fails_comp),
        "vec_abs_fail": fails_abs,
        "vec_rel_fail": fails_rel,
        "comp_abs_fail": fails_comp,
        "per_harmonic": records,
    }


def evaluate_modes_pair_v31(
    merged: dict[tuple[int, int], dict],
    full_ref: dict[tuple[int, int], dict],
    *,
    harmonics: tuple[int, ...] = DEFAULT_HARMONICS,
    abs_atol: float = V31_ABS_ATOL,
    rel_atol: float = V31_REL_ATOL,
    eta: float = V31_ETA,
) -> dict:
    return {
        "jones_modes": evaluate_jones_modes_v31(
            merged,
            full_ref,
            abs_atol=abs_atol,
            rel_atol=rel_atol,
            eta=eta,
            harmonics=harmonics,
        ),
        "jones_bar_J": evaluate_jones_bar_v31(
            merged,
            full_ref,
            abs_atol=abs_atol,
            rel_atol=rel_atol,
            eta=eta,
            harmonics=harmonics,
        ),
    }


def verdict_pass(rep: dict) -> bool:
    return (
        rep["jones_modes"]["status"] == "PASS"
        and rep["jones_bar_J"]["status"] == "PASS"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--full-run-dir", type=Path, required=True)
    ap.add_argument("--abs-atol", type=float, default=V31_ABS_ATOL)
    ap.add_argument("--rel-atol", type=float, default=V31_REL_ATOL)
    ap.add_argument("--eta", type=float, default=V31_ETA)
    ap.add_argument("--abs-atol-continuum", type=float, default=1.0e-12)
    ap.add_argument("--strong-floor-continuum", type=float, default=1.0e-12)
    ap.add_argument("--strong-rtol-continuum", type=float, default=2.0e-7)
    ap.add_argument("--h-max", type=float, default=15.0)
    ap.add_argument("--harmonics", default="2,5,7,9,10")
    ap.add_argument("--continuum-match-es16", action="store_true", default=True)
    ap.add_argument(
        "--no-continuum-match-es16",
        action="store_false",
        dest="continuum_match_es16",
    )
    ap.add_argument("--report", type=Path)
    ap.add_argument("--strict-v1-status", default="FAIL")
    ap.add_argument("--mixed-v2-status", default="FAIL (near-miss)")
    ap.add_argument("--v3-status", default="FAIL")
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

    pair = evaluate_modes_pair_v31(
        modes_m,
        modes_f,
        harmonics=harmonics,
        abs_atol=args.abs_atol,
        rel_atol=args.rel_atol,
        eta=args.eta,
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

    v31_pass = verdict_pass(pair) and cont["status"] == "PASS"
    report = {
        "gate": "JONES_VECTOR_V3.1",
        "status": "PASS" if v31_pass else "FAIL",
        "labels": {
            "STRICT_V1": args.strict_v1_status,
            "MIXED_TOLERANCE_V2": args.mixed_v2_status,
            "JONES_VECTOR_V3": args.v3_status,
            "CHUNK_NUMERICAL_EQUIVALENCE_V3.1": "PASS" if v31_pass else "FAIL",
        },
        "frozen_thresholds": {
            "eta": args.eta,
            "abs_atol": args.abs_atol,
            "rel_atol": args.rel_atol,
            "note": (
                "Same numbers as V3. V3.1 only fixes: weak nodes do not take "
                "relative hard gate (avoids e_abs<=1e-20 contradiction)."
            ),
        },
        "merged": str(args.merged),
        "full_run_dir": str(args.full_run_dir),
        "continuum_match_es16": bool(args.continuum_match_es16),
        "continuum_precision_note": (
            "PASS means within historical U8 F10.4/ES16.8 expressible precision "
            "when continuum_match_es16=true."
        ),
        "jones_modes": pair["jones_modes"],
        "jones_bar_J": pair["jones_bar_J"],
        "continuum_band": cont,
    }

    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if v31_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
