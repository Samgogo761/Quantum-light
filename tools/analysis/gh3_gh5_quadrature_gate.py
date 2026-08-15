#!/usr/bin/env python3
"""Independent GH3-v2 ↔ GH5-v2 quadrature gate (not the CEP 1e-3 covariance gate).

Frozen 2026-08-15:
  hard harmonics H2/H5/H7/H9; H10 diagnostic only
  ICS/CS and resolved Je: 5% relative
  Jo: mixed abs/rel, 10%
  strong-signal Jones phase: < 0.1 rad
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hhg_fft_utils import parse_input_nml, pick_ics_cs_at_order
from rank_full112_gh3_candidates import amp, load_node_modes

GATE_VERSION = "gh3_gh5_quadrature_v1_20260815"
HARD_ORDERS = (2, 5, 7, 9)
DIAG_ORDERS = (10,)
ICS_CS_RTOL = 0.05
JE_RTOL = 0.05
JO_RTOL = 0.10
JO_JE_ETA = 1.0e-4
PHASE_ATOL_RAD = 0.1
STATES = ("sv_r2p5_th000", "sv_r2p5_th180")


def jones_phase_diff(jx_a: complex, jy_a: complex, jx_b: complex, jy_b: complex) -> float:
    """Principal-value phase between two Jones vectors via Hermitian inner product."""
    inner = (jx_a.conjugate() * jx_b) + (jy_a.conjugate() * jy_b)
    if abs(inner) == 0.0:
        return 0.0
    dphi = abs(math.atan2(inner.imag, inner.real))
    return min(dphi, 2.0 * math.pi - dphi)


def ensemble_current(
    plus: dict[tuple[int, int], tuple[complex, complex, float]],
    minus: dict[tuple[int, int], tuple[complex, complex, float]],
    order: int,
    *,
    odd: bool,
) -> tuple[complex, complex, float]:
    jx = jy = 0j
    wsum = 0.0
    sign = -1.0 if odd else 1.0
    for (nid, o), (jxp, jyp, w) in plus.items():
        if o != order:
            continue
        jxm, jym, _wm = minus[(nid, o)]
        jx += w * 0.5 * (jxp + sign * jxm)
        jy += w * 0.5 * (jyp + sign * jym)
        wsum += w
    return jx, jy, wsum


def mixed_rel(delta: float, ref: float, scale: float, rtol: float, eta: float) -> dict:
    strong = ref >= eta * max(scale, 1.0e-300)
    if strong:
        e_rel = delta / max(ref, 1.0e-300)
        return {
            "strong": True,
            "e_rel": e_rel,
            "e_abs_over_scale": delta / max(scale, 1.0e-300),
            "pass": e_rel <= rtol,
        }
    e_abs = delta / max(scale, 1.0e-300)
    return {
        "strong": False,
        "e_rel": delta / max(ref, 1.0e-300),
        "e_abs_over_scale": e_abs,
        "pass": e_abs <= rtol,
        "label": "BELOW_SIGNAL_FLOOR",
    }


def compare_state(stem: str, gh3_root: Path, gh5_root: Path) -> dict:
    g3p, g3m = gh3_root / f"{stem}_plusN", gh3_root / f"{stem}_minusN"
    g5p, g5m = gh5_root / f"{stem}_plusN", gh5_root / f"{stem}_minusN"
    grid = parse_input_nml(g3p / "input.nml")
    dt = float(grid["dt_au"])
    nt = int(grid["nt"])
    omega0 = float(grid["omega0"])

    m3p = load_node_modes(g3p / "HHG_nodes_modes.dat")
    m3m = load_node_modes(g3m / "HHG_nodes_modes.dat")
    m5p = load_node_modes(g5p / "HHG_nodes_modes.dat")
    m5m = load_node_modes(g5m / "HHG_nodes_modes.dat")

    jo_scale = 0.0
    je_scale = 0.0
    per: dict[str, dict] = {}
    for order in HARD_ORDERS + DIAG_ORDERS:
        jox3, joy3, _ = ensemble_current(m3p, m3m, order, odd=True)
        jox5, joy5, _ = ensemble_current(m5p, m5m, order, odd=True)
        jex3, jey3, _ = ensemble_current(m3p, m3m, order, odd=False)
        jex5, jey5, _ = ensemble_current(m5p, m5m, order, odd=False)
        jo3 = amp(jox3, joy3)
        jo5 = amp(jox5, joy5)
        je3 = amp(jex3, jey3)
        je5 = amp(jex5, jey5)
        jo_scale = max(jo_scale, jo3, jo5)
        je_scale = max(je_scale, je3, je5)
        ics3 = pick_ics_cs_at_order(g3p / "HHG_ics_cs.dat", order, nt=nt, dt=dt, omega0=omega0)
        ics5 = pick_ics_cs_at_order(g5p / "HHG_ics_cs.dat", order, nt=nt, dt=dt, omega0=omega0)
        per[str(order)] = {
            "Jo_gh3": jo3,
            "Jo_gh5": jo5,
            "d_Jo": abs(jo5 - jo3),
            "Je_gh3": je3,
            "Je_gh5": je5,
            "d_Je": abs(je5 - je3),
            "ICS_gh3": ics3["ICS"],
            "ICS_gh5": ics5["ICS"],
            "CS_gh3": ics3["CS"],
            "CS_gh5": ics5["CS"],
            "dphi_Jo": jones_phase_diff(jox3, joy3, jox5, joy5),
            "jx3": [jox3.real, jox3.imag],
            "jy3": [joy3.real, joy3.imag],
            "jx5": [jox5.real, jox5.imag],
            "jy5": [joy5.real, joy5.imag],
        }

    hard_fail = []
    for order in HARD_ORDERS:
        row = per[str(order)]
        ics_rel = abs(row["ICS_gh5"] - row["ICS_gh3"]) / max(abs(row["ICS_gh3"]), 1.0e-300)
        cs_rel = abs(row["CS_gh5"] - row["CS_gh3"]) / max(abs(row["CS_gh3"]), 1.0e-300)
        jo = mixed_rel(row["d_Jo"], row["Jo_gh3"], jo_scale, JO_RTOL, JO_JE_ETA)
        je = mixed_rel(row["d_Je"], row["Je_gh3"], je_scale, JE_RTOL, JO_JE_ETA)
        phase_ok = (not jo["strong"]) or (row["dphi_Jo"] <= PHASE_ATOL_RAD)
        row["ICS_rel"] = ics_rel
        row["CS_rel"] = cs_rel
        row["Jo_gate"] = jo
        row["Je_gate"] = je
        row["phase_pass"] = phase_ok
        row["pass"] = (
            ics_rel <= ICS_CS_RTOL
            and cs_rel <= ICS_CS_RTOL
            and jo["pass"]
            and je["pass"]
            and phase_ok
        )
        if not row["pass"] and order in HARD_ORDERS:
            hard_fail.append(order)

    for order in DIAG_ORDERS:
        row = per[str(order)]
        row["note"] = "H10 diagnostic only; not in verdict"

    return {
        "state": stem,
        "pass": not hard_fail,
        "hard_fail_orders": hard_fail,
        "orders": per,
    }


def frozen_thresholds() -> dict:
    return {
        "schema": GATE_VERSION,
        "hard_orders": list(HARD_ORDERS),
        "diag_orders": list(DIAG_ORDERS),
        "ics_cs_rtol": ICS_CS_RTOL,
        "je_resolved_rtol": JE_RTOL,
        "jo_mixed_rtol": JO_RTOL,
        "jo_je_eta": JO_JE_ETA,
        "phase_strong_atol_rad": PHASE_ATOL_RAD,
        "not_the_cep_rel_1e-3_gate": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gh3-root", type=Path, required=True)
    ap.add_argument("--gh5-root", type=Path, required=True)
    ap.add_argument("--states", nargs="+", default=list(STATES))
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    states = [compare_state(s, args.gh3_root, args.gh5_root) for s in args.states]
    report = {
        "gate": GATE_VERSION,
        "status": "PASS" if all(s["pass"] for s in states) else "FAIL",
        "thresholds": frozen_thresholds(),
        "gh3_root": str(args.gh3_root),
        "gh5_root": str(args.gh5_root),
        "states": states,
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
