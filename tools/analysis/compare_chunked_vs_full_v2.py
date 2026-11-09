#!/usr/bin/env python3
"""MIXED_TOLERANCE_V2 gate: chunked merge vs unchunked (zero-cost reanalysis).

Does NOT replace STRICT_V1. Preserves historical FAIL; adds abs-normalized
and strong-signal relative gates suited to near-zero residuals.

Modes (per harmonic n, pol a in {jx,jy}):
  A_na = max_s max(|J^C|, |J^U|)
  e_abs = |dJ| / A_na                 require <= abs_atol (default 1e-12)
  if |J_ref| >= strong_floor * A_na:  also e_rel <= strong_rtol (1e-10)
  else: mark BELOW_SIGNAL_FLOOR

Continuum (scientific band 0 <= h <= h_max, default 15):
  S_peak = max S_ref in band
  all bins: |dS|/S_peak <= cont_abs_atol (1e-12)
  strong bins S_ref >= cont_strong_floor * S_peak: also e_rel <= cont_rtol (2e-7)

Complex CS at target harmonics: reconstruct bar_J = sum_s w_s J_s from modes,
apply same abs/rel pattern to bar_Jx/bar_Jy (U8 has no continuum complex dump).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_HARMONICS = (2, 5, 7, 9, 10)


def load_modes(path: Path) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 10:
            raise ValueError(f"{path}: expected 10 columns, got {len(f)}")
        key = (int(f[0]), int(f[4]))
        out[key] = {
            "weight": float(f[1]),
            "jx": complex(float(f[5]), float(f[6])),
            "jy": complex(float(f[7]), float(f[8])),
            "power": float(f[9]),
        }
    if not out:
        raise ValueError(f"{path}: empty modes")
    return out


def round_es16(x: float) -> float:
    """Match Fortran ES16.8 print/parse round-trip (~9 significant digits)."""
    if x == 0.0 or not math.isfinite(x):
        return x
    return float(f"{x:.8E}")


def load_continuum(path: Path, *, round_es16_vals: bool = False) -> list[dict]:
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 4:
            raise ValueError(f"{path}: expected >=4 continuum columns")
        h, omega, ics, cs = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
        if round_es16_vals:
            # U8 write_hhg_ics_cs uses F10.4 + ES16.8; merge uses ES25.17.
            h = float(f"{h:.4f}")
            omega = round_es16(omega)
            ics = round_es16(ics)
            cs = round_es16(cs)
        rows.append(
            {
                "iw": len(rows) + 1,
                "h_order": h,
                "omega": omega,
                "ICS": ics,
                "CS": cs,
            }
        )
    if not rows:
        raise ValueError(f"{path}: empty continuum")
    return rows


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def cplx_rel(a: complex, b: complex) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def evaluate_modes(
    merged: dict[tuple[int, int], dict],
    full: dict[tuple[int, int], dict],
    *,
    abs_atol: float,
    strong_floor: float,
    strong_rtol: float,
    harmonics: tuple[int, ...],
) -> dict:
    keys = sorted(set(merged) | set(full))
    missing = [k for k in keys if k not in merged or k not in full]
    if missing:
        return {
            "status": "FAIL",
            "errors": [f"missing mode keys: {missing[:10]}"],
            "n_errors": len(missing),
        }

    # Per (harmonic, pol): global amplitude scale A_na
    scales: dict[tuple[int, str], float] = {}
    for (nid, h), row_m in merged.items():
        if harmonics and h not in harmonics:
            continue
        row_f = full[(nid, h)]
        for pol in ("jx", "jy"):
            a = max(abs(row_m[pol]), abs(row_f[pol]))
            key = (h, pol)
            scales[key] = max(scales.get(key, 0.0), a)

    records = []
    hard_fail = []
    strong_fail = []
    below_floor = []
    max_e_abs = 0.0
    max_e_rel_strong = 0.0

    for (nid, h) in sorted(merged):
        if harmonics and h not in harmonics:
            continue
        rm, rf = merged[(nid, h)], full[(nid, h)]
        for pol in ("jx", "jy"):
            jm, jf = rm[pol], rf[pol]
            A = scales[(h, pol)]
            if A <= 0.0:
                A = 1.0e-300
            e_abs = abs(jm - jf) / A
            e_rel = cplx_rel(jm, jf)
            jref = max(abs(jm), abs(jf))
            is_strong = jref >= strong_floor * A
            tag = {
                "node": nid,
                "H": h,
                "pol": pol,
                "e_abs": e_abs,
                "e_rel": e_rel,
                "A_na": A,
                "|J|_max": jref,
                "strong": is_strong,
            }
            records.append(tag)
            max_e_abs = max(max_e_abs, e_abs)
            # Abs-normalized gate applies to ALL nodes (including near-zero).
            if e_abs > abs_atol:
                hard_fail.append({**tag, "gate": "abs"})
            if not is_strong:
                below_floor.append({**tag, "label": "BELOW_SIGNAL_FLOOR"})
                continue
            max_e_rel_strong = max(max_e_rel_strong, e_rel)
            if e_rel > strong_rtol:
                strong_fail.append({**tag, "gate": "rel"})

    status = "PASS" if not hard_fail and not strong_fail else "FAIL"
    return {
        "status": status,
        "abs_atol": abs_atol,
        "strong_floor": strong_floor,
        "strong_rtol": strong_rtol,
        "max_e_abs": max_e_abs,
        "max_e_rel_strong": max_e_rel_strong,
        "n_checked": len(records),
        "n_below_signal_floor": len(below_floor),
        "n_hard_fail": len(hard_fail),
        "n_strong_rel_fail": len(strong_fail),
        "hard_fail": hard_fail[:40],
        "strong_rel_fail": strong_fail[:40],
        "below_signal_floor_examples": below_floor[:20],
        "scales_A_na": {
            f"H{h}_{pol}": scales[(h, pol)] for (h, pol) in sorted(scales)
        },
    }


def evaluate_bar_j_from_modes(
    merged: dict[tuple[int, int], dict],
    full: dict[tuple[int, int], dict],
    *,
    abs_atol: float,
    strong_floor: float,
    strong_rtol: float,
    harmonics: tuple[int, ...],
) -> dict:
    """Coherent sum bar_J_n = sum_s w_s J_s at target harmonics (complex CS proxy)."""
    nodes = sorted({k[0] for k in merged})
    results = []
    hard_fail = []
    strong_fail = []
    below = []
    max_e_abs = 0.0
    max_e_rel_strong = 0.0

    for h in harmonics:
        for pol in ("jx", "jy"):
            bm = 0j
            bf = 0j
            for nid in nodes:
                key = (nid, h)
                if key not in merged or key not in full:
                    return {
                        "status": "FAIL",
                        "errors": [f"missing {(nid, h)} for bar_J"],
                    }
                wm, wf = merged[key]["weight"], full[key]["weight"]
                if abs(wm - wf) > 1.0e-14 * max(abs(wm), abs(wf), 1.0):
                    return {
                        "status": "FAIL",
                        "errors": [f"weight mismatch node{nid} H{h}"],
                    }
                bm += wm * merged[key][pol]
                bf += wf * full[key][pol]
            A = max(abs(bm), abs(bf), 1.0e-300)
            e_abs = abs(bm - bf) / A
            e_rel = cplx_rel(bm, bf)
            jref = max(abs(bm), abs(bf))
            is_strong = jref >= strong_floor * A
            # For a single coherent sum, A == jref by construction when one side
            # dominates; still apply floor vs A for API symmetry (always strong
            # unless both ~0). Use absolute peak across harmonics for floor.
            tag = {
                "H": h,
                "pol": pol,
                "e_abs": e_abs,
                "e_rel": e_rel,
                "|bar_J|_max": jref,
                "bar_J_merged": [bm.real, bm.imag],
                "bar_J_full": [bf.real, bf.imag],
                "strong": is_strong,
            }
            results.append(tag)
            max_e_abs = max(max_e_abs, e_abs)
            if not is_strong:
                below.append(tag)
                continue
            max_e_rel_strong = max(max_e_rel_strong, e_rel)
            if e_abs > abs_atol:
                hard_fail.append({**tag, "gate": "abs"})
            if e_rel > strong_rtol:
                strong_fail.append({**tag, "gate": "rel"})

    # Re-evaluate strong floor using global peak |bar_J| across reported H/pols
    peak = max((t["|bar_J|_max"] for t in results), default=0.0)
    hard_fail2, strong_fail2, below2 = [], [], []
    max_e_abs2 = 0.0
    max_e_rel2 = 0.0
    for tag in results:
        jref = tag["|bar_J|_max"]
        A = peak if peak > 0 else 1.0e-300
        e_abs = abs(
            complex(*tag["bar_J_merged"]) - complex(*tag["bar_J_full"])
        ) / A
        e_rel = tag["e_rel"]
        is_strong = jref >= strong_floor * A
        tag2 = {**tag, "e_abs_vs_peak": e_abs, "A_peak": A, "strong": is_strong}
        max_e_abs2 = max(max_e_abs2, e_abs)
        if e_abs > abs_atol:
            hard_fail2.append({**tag2, "gate": "abs"})
        if not is_strong:
            below2.append({**tag2, "label": "BELOW_SIGNAL_FLOOR"})
            continue
        max_e_rel2 = max(max_e_rel2, e_rel)
        if e_rel > strong_rtol:
            strong_fail2.append({**tag2, "gate": "rel"})

    status = "PASS" if not hard_fail2 and not strong_fail2 else "FAIL"
    return {
        "status": status,
        "note": "bar_J reconstructed from HHG_nodes_modes; continuum complex dump absent on U8",
        "abs_atol": abs_atol,
        "strong_floor": strong_floor,
        "strong_rtol": strong_rtol,
        "peak_|bar_J|": peak,
        "max_e_abs_vs_peak": max_e_abs2,
        "max_e_rel_strong": max_e_rel2,
        "n_hard_fail": len(hard_fail2),
        "n_strong_rel_fail": len(strong_fail2),
        "n_below_signal_floor": len(below2),
        "hard_fail": hard_fail2,
        "strong_rel_fail": strong_fail2,
        "below_signal_floor": below2,
        "per_harmonic": results,
    }


def evaluate_continuum(
    merged: list[dict],
    full: list[dict],
    *,
    h_max: float,
    abs_atol: float,
    strong_floor: float,
    strong_rtol: float,
    focus_harmonics: tuple[int, ...],
) -> dict:
    if len(merged) != len(full):
        return {
            "status": "FAIL",
            "errors": [f"continuum length {len(merged)} vs {len(full)}"],
        }

    band_m = [r for r in merged if 0.0 <= r["h_order"] <= h_max + 1.0e-9]
    band_f = [r for r in full if 0.0 <= r["h_order"] <= h_max + 1.0e-9]
    if len(band_m) != len(band_f):
        # Align by iw within band using merged filter indices
        pass

    # Align by iw for rows in scientific band
    full_by_iw = {r["iw"]: r for r in full}
    pairs = []
    for rm in merged:
        if not (0.0 <= rm["h_order"] <= h_max + 1.0e-9):
            continue
        rf = full_by_iw.get(rm["iw"])
        if rf is None:
            return {"status": "FAIL", "errors": [f"missing iw={rm['iw']} on full"]}
        pairs.append((rm, rf))

    out = {}
    for kind in ("ICS", "CS"):
        peak = max(max(abs(rm[kind]), abs(rf[kind])) for rm, rf in pairs)
        if peak <= 0.0:
            peak = 1.0e-300
        hard_fail = []
        strong_fail = []
        below = []
        max_e_abs = 0.0
        max_e_rel_strong = 0.0
        focus = []
        for rm, rf in pairs:
            s_m, s_f = rm[kind], rf[kind]
            s_ref = max(abs(s_m), abs(s_f))
            e_abs = abs(s_m - s_f) / peak
            e_rel = rel_err(s_m, s_f)
            is_strong = s_ref >= strong_floor * peak
            tag = {
                "iw": rm["iw"],
                "h_order": rm["h_order"],
                "e_abs": e_abs,
                "e_rel": e_rel,
                "S_ref": s_ref,
                "strong": is_strong,
            }
            max_e_abs = max(max_e_abs, e_abs)
            # Focus harmonics: nearest bins to integer H
            for H in focus_harmonics:
                if abs(rm["h_order"] - H) < 0.15:
                    focus.append({**tag, "focus_H": H})
            if e_abs > abs_atol:
                hard_fail.append({**tag, "gate": "abs"})
            if not is_strong:
                below.append({**tag, "label": "BELOW_SIGNAL_FLOOR"})
                continue
            max_e_rel_strong = max(max_e_rel_strong, e_rel)
            if e_rel > strong_rtol:
                strong_fail.append({**tag, "gate": "rel"})

        status = "PASS" if not hard_fail and not strong_fail else "FAIL"
        out[kind] = {
            "status": status,
            "h_max": h_max,
            "S_peak": peak,
            "n_bins_in_band": len(pairs),
            "max_e_abs": max_e_abs,
            "max_e_rel_strong": max_e_rel_strong,
            "n_below_signal_floor": len(below),
            "n_hard_fail": len(hard_fail),
            "n_strong_rel_fail": len(strong_fail),
            "hard_fail": hard_fail[:40],
            "strong_rel_fail": strong_fail[:40],
            "focus_harmonics": focus,
        }

    status = (
        "PASS"
        if out["ICS"]["status"] == "PASS" and out["CS"]["status"] == "PASS"
        else "FAIL"
    )
    return {
        "status": status,
        "abs_atol": abs_atol,
        "strong_floor": strong_floor,
        "strong_rtol": strong_rtol,
        "ICS": out["ICS"],
        "CS": out["CS"],
        "note": "Scalar ICS/CS only; complex continuum dump not present on unchunked U8",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--full-run-dir", type=Path, required=True)
    ap.add_argument("--abs-atol-modes", type=float, default=1.0e-12)
    ap.add_argument("--strong-floor-modes", type=float, default=1.0e-10)
    ap.add_argument("--strong-rtol-modes", type=float, default=1.0e-10)
    ap.add_argument("--abs-atol-continuum", type=float, default=1.0e-12)
    ap.add_argument("--strong-floor-continuum", type=float, default=1.0e-12)
    ap.add_argument("--strong-rtol-continuum", type=float, default=2.0e-7)
    ap.add_argument("--h-max", type=float, default=15.0)
    ap.add_argument("--harmonics", default="2,5,7,9,10")
    ap.add_argument(
        "--continuum-match-es16",
        action="store_true",
        default=True,
        help="Round continuum to F10.4/ES16.8 before compare (U8 write format).",
    )
    ap.add_argument(
        "--no-continuum-match-es16",
        action="store_false",
        dest="continuum_match_es16",
    )
    ap.add_argument("--report", type=Path)
    ap.add_argument(
        "--strict-v1-status",
        default="FAIL",
        help="Historical STRICT_V1 label to record (do not overwrite).",
    )
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

    modes_rep = evaluate_modes(
        modes_m,
        modes_f,
        abs_atol=args.abs_atol_modes,
        strong_floor=args.strong_floor_modes,
        strong_rtol=args.strong_rtol_modes,
        harmonics=harmonics,
    )
    barj_rep = evaluate_bar_j_from_modes(
        modes_m,
        modes_f,
        abs_atol=args.abs_atol_modes,
        strong_floor=args.strong_floor_modes,
        strong_rtol=args.strong_rtol_modes,
        harmonics=harmonics,
    )
    cont_rep = evaluate_continuum(
        cont_m,
        cont_f,
        h_max=args.h_max,
        abs_atol=args.abs_atol_continuum,
        strong_floor=args.strong_floor_continuum,
        strong_rtol=args.strong_rtol_continuum,
        focus_harmonics=harmonics,
    )

    # Strong-mode stability (documentation label): components with |J| >= 1e-3 A_na.
    # Distinct from pre-registered strong_floor=1e-10 used in V2 hard gates.
    strong_vis_fail = []
    for item in modes_rep.get("hard_fail", []) + modes_rep.get("strong_rel_fail", []):
        if item.get("|J|_max", 0.0) >= 1.0e-3 * item.get("A_na", 1.0):
            strong_vis_fail.append(item)
    # Also scan would-be strong-visible from below_floor? no.
    if not strong_vis_fail and modes_rep.get("max_e_abs", 1.0) <= args.abs_atol_modes:
        strong_mode_label = "PASS"
    else:
        strong_mode_label = "FAIL"

    v2_pass = (
        modes_rep["status"] == "PASS"
        and barj_rep["status"] == "PASS"
        and cont_rep["status"] == "PASS"
    )
    report = {
        "gate": "MIXED_TOLERANCE_V2",
        "status": "PASS" if v2_pass else "FAIL",
        "labels": {
            "STRICT_V1": args.strict_v1_status,
            "STRONG_MODE_STABILITY": strong_mode_label,
            "MIXED_TOLERANCE_V2": "PASS" if v2_pass else "FAIL",
            "CHUNK_NUMERICAL_EQUIVALENCE_V2": (
                "PASS" if v2_pass else "FAIL"
            ),
        },
        "merged": str(args.merged),
        "full_run_dir": str(args.full_run_dir),
        "continuum_match_es16": bool(args.continuum_match_es16),
        "modes": modes_rep,
        "coherent_bar_J_target_harmonics": barj_rep,
        "continuum_band": cont_rep,
        "note": (
            "STRICT_V1 historical FAIL retained. V2 uses abs-normalized + "
            "strong-signal relative gates; BELOW_SIGNAL_FLOOR is not PASS. "
            "U8 continuum written as F10.4/ES16.8; default compare matches that format."
        ),
    }

    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if v2_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
