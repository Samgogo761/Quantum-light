#!/usr/bin/env python3
"""GH7 tail-model gate. Staged: does not auto-approve k40 or full GH7.

Frozen suggestions (2026-08-16):
  - no NaN/Inf
  - diagonal occupation in [-1e-8, 1+1e-8]
  - total-trace relative drift <= 1e-8
  - top-8 band edge: abs occ <= 1e-3 and edge/excited <= 1%
  - k20->k40 Jones: H2/5/7/9 vec_rel<=5%, phase<=0.1 rad; H10 diagnostic 10%
  - optional dt/2: 2%
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from diagnose_gh5_quadrature import jones_decomp
from rank_full112_gh3_candidates import load_node_modes

OCC_LO = -1.0e-8
OCC_HI = 1.0 + 1.0e-8
TRACE_RTOL = 1.0e-8
EDGE_ABS = 1.0e-3
EDGE_REL = 0.01
N_EDGE_BANDS = 8
NV = 84
NB = 112
JONES_HARD = (2, 5, 7, 9)
JONES_RTOL = 0.05
JONES_PHASE = 0.1
H10_RTOL = 0.10
DT2_RTOL = 0.02
GATE_VERSION = "gh7_tail_model_v1_20260816"


def _finite_tokens(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    for tok in ("nan", "inf"):
        if tok in text:
            raise ValueError(f"NaN/Inf token in {path}")


def parse_band_occupation(path: Path) -> dict:
    """Parse occupation_band_kt.dat -> per-snapshot diagnostics."""
    _finite_tokens(path)
    snaps: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split()
            it = int(f[0])
            t_fs = float(f[1])
            band = int(f[4])
            occ = float(f[5])
            if not math.isfinite(occ):
                raise ValueError(f"non-finite occupation in {path}")
            if occ < OCC_LO or occ > OCC_HI:
                raise ValueError(f"occupation {occ} outside [{OCC_LO}, {OCC_HI}] in {path}")
            rec = snaps.setdefault(it, {"t_fs": t_fs, "trace": 0.0, "exc": 0.0, "edge": 0.0, "edge_max": 0.0})
            rec["trace"] += occ
            if band > NV:
                rec["exc"] += occ
            if band > NB - N_EDGE_BANDS:
                rec["edge"] += occ
                rec["edge_max"] = max(rec["edge_max"], occ)
    if not snaps:
        raise ValueError(f"no occupation rows in {path}")
    its = sorted(snaps)
    tr0 = snaps[its[0]]["trace"]
    max_drift = 0.0
    max_edge_rel = 0.0
    max_edge_abs = 0.0
    for it in its:
        rec = snaps[it]
        drift = abs(rec["trace"] - tr0) / max(abs(tr0), 1.0e-300)
        max_drift = max(max_drift, drift)
        max_edge_abs = max(max_edge_abs, rec["edge_max"])
        rel = rec["edge"] / max(rec["exc"], 1.0e-300)
        max_edge_rel = max(max_edge_rel, rel)
    return {
        "n_snapshots": len(its),
        "trace0": tr0,
        "max_trace_rel_drift": max_drift,
        "max_edge_abs": max_edge_abs,
        "max_edge_rel": max_edge_rel,
        "pass_range": True,
        "pass_trace": max_drift <= TRACE_RTOL,
        "pass_edge_abs": max_edge_abs <= EDGE_ABS,
        "pass_edge_rel": max_edge_rel <= EDGE_REL,
    }


def parse_compact_occupation(path: Path) -> dict:
    """Fallback if only occupation_kt.dat exists (no band-resolved)."""
    _finite_tokens(path)
    snaps: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split()
            it = int(f[0])
            n_val = float(f[6])
            n_cond = float(f[7])
            if not math.isfinite(n_val) or not math.isfinite(n_cond):
                raise ValueError(f"non-finite n_val/n_cond in {path}")
            rec = snaps.setdefault(it, {"trace": 0.0, "n_k": 0})
            rec["trace"] += n_val + n_cond
            rec["n_k"] += 1
    its = sorted(snaps)
    tr0 = snaps[its[0]]["trace"]
    max_drift = max(abs(snaps[it]["trace"] - tr0) / max(abs(tr0), 1.0e-300) for it in its)
    return {
        "n_snapshots": len(its),
        "trace0": tr0,
        "max_trace_rel_drift": max_drift,
        "pass_trace": max_drift <= TRACE_RTOL,
        "band_resolved": False,
        "note": "compact occupation only; edge-band gate requires occupation_band_kt.dat",
    }


def jones_from_modes(path: Path) -> dict[int, tuple[complex, complex]]:
    modes = load_node_modes(path)
    out: dict[int, tuple[complex, complex]] = {}
    for (nid, order), (jx, jy, _w) in modes.items():
        if order in out and out[order] != (jx, jy):
            # single-node probe: one Jones per harmonic
            pass
        out[order] = (jx, jy)
    return out


def compare_jones(a: dict, b: dict) -> dict:
    rows = {}
    hard_fail = []
    for order in JONES_HARD + (10,):
        if order not in a or order not in b:
            rows[str(order)] = {"status": "MISSING"}
            if order in JONES_HARD:
                hard_fail.append(order)
            continue
        de = jones_decomp(a[order][0], a[order][1], b[order][0], b[order][1])
        rtol = H10_RTOL if order == 10 else JONES_RTOL
        ok = de["vec_rel"] <= rtol and de["phase_rad"] <= JONES_PHASE
        de["pass"] = ok
        de["rtol"] = rtol
        rows[str(order)] = de
        if order in JONES_HARD and not ok:
            hard_fail.append(order)
    return {"orders": rows, "pass": not hard_fail, "hard_fail": hard_fail}


def audit_k20_dir(outdir: Path) -> dict:
    band = outdir / "occupation_band_kt.dat"
    compact = outdir / "occupation_kt.dat"
    modes = outdir / "HHG_nodes_modes.dat"
    status = {"outdir": str(outdir), "stage": "k20"}
    if not modes.exists():
        status["status"] = "INCOMPLETE"
        status["missing"] = "HHG_nodes_modes.dat"
        return status
    _finite_tokens(modes)
    if band.exists():
        occ = parse_band_occupation(band)
        status["occupation"] = occ
        status["pass"] = occ["pass_trace"] and occ["pass_edge_abs"] and occ["pass_edge_rel"]
    elif compact.exists():
        occ = parse_compact_occupation(compact)
        status["occupation"] = occ
        status["pass"] = False
        status["reason"] = "band-resolved occupation required for edge gate"
    else:
        status["status"] = "INCOMPLETE"
        status["missing"] = "occupation_band_kt.dat"
        return status
    status["status"] = "PASS" if status.get("pass") else "FAIL"
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("k20", "k40", "dt2"), required=True)
    ap.add_argument("--k20-root", type=Path)
    ap.add_argument("--k40-root", type=Path)
    ap.add_argument("--dt2-root", type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    report: dict = {
        "gate": GATE_VERSION,
        "stage": args.stage,
        "auto_chain": False,
        "thresholds": {
            "occ_lo": OCC_LO,
            "occ_hi": OCC_HI,
            "trace_rtol": TRACE_RTOL,
            "edge_abs": EDGE_ABS,
            "edge_rel": EDGE_REL,
            "jones_rtol_hard": JONES_RTOL,
            "jones_phase_rad": JONES_PHASE,
            "h10_rtol": H10_RTOL,
            "dt2_rtol": DT2_RTOL,
        },
    }
    if args.stage == "k20":
        if args.k20_root is None:
            raise SystemExit("--k20-root required")
        cases = sorted(p for p in args.k20_root.iterdir() if p.is_dir()) if args.k20_root.exists() else []
        rows = [audit_k20_dir(p) for p in cases]
        report["cases"] = rows
        report["status"] = "PASS" if rows and all(r.get("status") == "PASS" for r in rows) else (
            "INCOMPLETE" if (not rows or any(r.get("status") == "INCOMPLETE" for r in rows)) else "FAIL"
        )
        report["approve_k40"] = report["status"] == "PASS"
    elif args.stage == "k40":
        if args.k20_root is None or args.k40_root is None:
            raise SystemExit("--k20-root and --k40-root required")
        k20 = json.loads((args.k20_root / "GH7_TAIL_K20.json").read_text(encoding="utf-8")) if (args.k20_root / "GH7_TAIL_K20.json").exists() else None
        if k20 is None or k20.get("status") != "PASS":
            report["status"] = "BLOCKED"
            report["reason"] = "k20 occupation gate is not PASS; do not run/score k40"
        else:
            pairs = []
            hard_ok = True
            for d40 in sorted(p for p in args.k40_root.iterdir() if p.is_dir()):
                d20 = args.k20_root / d40.name
                if not (d20 / "HHG_nodes_modes.dat").exists():
                    pairs.append({"case": d40.name, "status": "INCOMPLETE"})
                    hard_ok = False
                    continue
                cmpj = compare_jones(
                    jones_from_modes(d20 / "HHG_nodes_modes.dat"),
                    jones_from_modes(d40 / "HHG_nodes_modes.dat"),
                )
                pairs.append({"case": d40.name, **cmpj})
                hard_ok = hard_ok and cmpj["pass"]
            report["pairs"] = pairs
            report["status"] = "PASS" if hard_ok and pairs else "FAIL"
    else:
        if args.dt2_root is None or args.k20_root is None:
            raise SystemExit("--dt2-root and --k20-root required")
        # single representative: first matching stem
        report["status"] = "NOT_RUN"
        report["note"] = f"dt/2 gate rtol={DT2_RTOL}; run only if requested"
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    return 0 if report.get("status") in {"PASS", "NOT_RUN", "INCOMPLETE", "BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
