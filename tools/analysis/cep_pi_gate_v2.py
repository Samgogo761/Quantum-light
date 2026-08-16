#!/usr/bin/env python3
"""CEP-π V2 gate: Jones-vector residual with mixed-tolerance logic (zero-cost).

Preserves historical CEP V1 FAIL records. Does not retune τ_rel=1e-3.

For anti-node pair (s, s̄) with α_s̄ = −α_s (α=0 self-paired):
  d_sn = || J_{+N,s,n} + J_{−N,s̄,n} ||_2
  A_n  = max_s max(||J_{+N,s,n}||, ||J_{−N,s̄,n}||)

All nodes:
  e_abs = d_sn / A_n ≤ 1e-12

Strong (max(||J+||,||J−||) ≥ 1e-10 A_n):
  e_rel = d_sn / max(||J+||,||J−||) ≤ 1e-3   (hard)

Weak:
  BELOW_SIGNAL_FLOOR; relative diagnostic only.

Both directions checked for non-self pairs:
  A: J_{+N}(α) + J_{−N}(−α)
  B: J_{+N}(−α) + J_{−N}(α)

Hard harmonics: H2,H5,H7,H9. H10 diagnostic only.
ICS/CS equality is auxiliary, not the verdict.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from audit_full112_gh3_candidates import (
    ALPHA_TOL,
    HARD_ORDERS,
    INTENSITY_RTOL,
    META_KEYS,
    PHASE_TOL,
    TWOPI,
    WEIGHT_TOL,
    amp_xy,
    compare_pm_metadata,
    load_manifest,
    load_node_modes,
    pair_nodes_by_neg_alpha,
    phase_diff,
    read_metadata,
    validate_modes_vs_manifest,
)
from compare_chunked_vs_full_v3 import apply_unitary_to_modes, jones_norm

# Frozen CEP-π V2 thresholds:
#   τ_abs=1e-12 is a *preset acceptance* threshold (not the measured chunk floor;
#   chunk V3.1 max_e_abs_vec ≈ 4.8e-15). τ_rel=1e-3 keeps the old CEP relative gate.
CEP_V2_ABS_ATOL = 1.0e-12
CEP_V2_REL_ATOL = 1.0e-3
CEP_V2_ETA = 1.0e-10
DIAG_ORDERS = (10,)
DEFAULT_STATES = (
    "sv_r2p5_th000",
    "sv_r2p5_th180",
    "sv_r0p5_th180",
)


def wrap_phase(phi: float) -> float:
    return float(phi) % TWOPI


def phase_equals_plus_pi(phi_a: float, phi_b: float, tol: float = PHASE_TOL) -> bool:
    """φ_b ≡ φ_a + π (mod 2π), treating 0 and 2π as equal."""
    target = wrap_phase(wrap_phase(phi_a) + math.pi)
    return phase_diff(wrap_phase(phi_b), target) <= tol


def validate_antipode_phases(manifest: list[dict], pairs: list[dict]) -> list[str]:
    """Structural checks: phase_bar ≡ phase + π (mod 2π) for nonzero α."""
    by_id = {int(r["id"]): r for r in manifest}
    errs: list[str] = []
    for pr in pairs:
        ia, ib = int(pr["id_alpha"]), int(pr["id_neg"])
        if pr["kind"] == "self_alpha0":
            if ia != ib:
                errs.append(f"self_alpha0 pair ids differ: {ia}/{ib}")
            continue
        ra, rb = by_id[ia], by_id[ib]
        if not phase_equals_plus_pi(float(ra["phase"]), float(rb["phase"])):
            errs.append(
                f"phase antipode fail nodes {ia}/{ib}: "
                f"phi={float(ra['phase'])}, phi_bar={float(rb['phase'])}"
            )
    return errs


def jones_residual(
    jx_a: complex, jy_a: complex, jx_b: complex, jy_b: complex
) -> float:
    return jones_norm(jx_a + jx_b, jy_a + jy_b)


def evaluate_pair_harmonic(
    *,
    jx_p: complex,
    jy_p: complex,
    jx_m: complex,
    jy_m: complex,
    A_n: float,
    abs_atol: float,
    rel_atol: float,
    eta: float,
) -> dict:
    d = jones_residual(jx_p, jy_p, jx_m, jy_m)
    amp_p = amp_xy(jx_p, jy_p)
    amp_m = amp_xy(jx_m, jy_m)
    den_local = max(amp_p, amp_m)
    e_abs = d / max(A_n, 1.0e-300)
    e_rel_diag = d / max(den_local, 1.0e-300)
    is_strong = den_local >= eta * A_n
    tag = {
        "d": d,
        "A_n": A_n,
        "amp_plus": amp_p,
        "amp_minus": amp_m,
        "e_abs": e_abs,
        "e_rel_diag": e_rel_diag,
        "strong": is_strong,
    }
    abs_fail = e_abs > abs_atol
    rel_fail = False
    if is_strong:
        e_rel = d / den_local
        tag["e_rel"] = e_rel
        rel_fail = e_rel > rel_atol
    else:
        tag["label"] = "BELOW_SIGNAL_FLOOR"
        tag["e_rel"] = e_rel_diag
    tag["abs_fail"] = abs_fail
    tag["rel_fail"] = rel_fail
    tag["pass"] = (not abs_fail) and (not rel_fail)
    return tag


def scale_A_n(
    modes_p: dict,
    modes_m: dict,
    pairs: list[dict],
    order: int,
) -> float:
    A = 0.0
    for pr in pairs:
        ia, ib = int(pr["id_alpha"]), int(pr["id_neg"])
        # Cover both nodes appearing on +N and −N sides
        for nid in {ia, ib}:
            A = max(A, amp_xy(modes_p[(nid, order)]["jx"], modes_p[(nid, order)]["jy"]))
            A = max(A, amp_xy(modes_m[(nid, order)]["jx"], modes_m[(nid, order)]["jy"]))
    return A if A > 0.0 else 1.0e-300


def audit_state_cep_v2(
    stem: str,
    outroot: Path,
    *,
    hard_orders: tuple[int, ...] = HARD_ORDERS,
    diag_orders: tuple[int, ...] = DIAG_ORDERS,
    abs_atol: float = CEP_V2_ABS_ATOL,
    rel_atol: float = CEP_V2_REL_ATOL,
    eta: float = CEP_V2_ETA,
) -> dict:
    plus = outroot / f"{stem}_plusN"
    minus = outroot / f"{stem}_minusN"
    if not (plus / "SUCCESS").exists() or not (minus / "SUCCESS").exists():
        return {
            "state": stem,
            "status": "INCOMPLETE",
            "cep_pi_v2_pass": False,
        }

    try:
        meta_p = read_metadata(plus / "run_metadata.txt")
        meta_m = read_metadata(minus / "run_metadata.txt")
        compare_pm_metadata(meta_p, meta_m, stem)
        # TB / nodes hashes already in META_KEYS via compare_pm_metadata
        man_p = load_manifest(plus / "nodes_manifest.input.dat")
        man_m = load_manifest(minus / "nodes_manifest.input.dat")
        if len(man_p) != len(man_m):
            raise ValueError(f"{stem}: ±N manifest length mismatch")
        for ra, rb in zip(man_p, man_m):
            if int(ra["id"]) != int(rb["id"]):
                raise ValueError(f"{stem}: manifest id mismatch")
            if abs(float(ra["weight"]) - float(rb["weight"])) > WEIGHT_TOL:
                raise ValueError(f"{stem}: manifest weight mismatch id={ra['id']}")
            if abs(float(ra["re"]) - float(rb["re"])) > ALPHA_TOL:
                raise ValueError(f"{stem}: manifest re(alpha) mismatch id={ra['id']}")
            if abs(float(ra["im"]) - float(rb["im"])) > ALPHA_TOL:
                raise ValueError(f"{stem}: manifest im(alpha) mismatch id={ra['id']}")
            ia, ib = float(ra["intensity"]), float(rb["intensity"])
            if abs(ia - ib) > INTENSITY_RTOL * max(abs(ia), abs(ib), 1.0):
                raise ValueError(f"{stem}: manifest intensity mismatch id={ra['id']}")
            if phase_diff(float(ra["phase"]), float(rb["phase"])) > PHASE_TOL:
                raise ValueError(f"{stem}: manifest phase mismatch id={ra['id']}")

        modes_p = load_node_modes(plus / "HHG_nodes_modes.dat")
        modes_m = load_node_modes(minus / "HHG_nodes_modes.dat")
        validate_modes_vs_manifest(modes_p, man_p, f"{stem}+N")
        validate_modes_vs_manifest(modes_m, man_m, f"{stem}-N")
        pairs = pair_nodes_by_neg_alpha(man_p)
        phase_errs = validate_antipode_phases(man_p, pairs)
        if phase_errs:
            raise ValueError("; ".join(phase_errs[:5]))

        ids = sorted(int(r["id"]) for r in man_p)
        if len(ids) != len(set(ids)):
            raise ValueError(f"{stem}: duplicate node ids")

    except (OSError, ValueError) as exc:
        return {
            "state": stem,
            "status": "FAIL",
            "error": str(exc),
            "cep_pi_v2_pass": False,
        }

    all_orders = tuple(hard_orders) + tuple(diag_orders)
    by_order: dict[str, dict] = {}
    hard_pass = True
    abs_fails: list[dict] = []
    rel_fails: list[dict] = []

    for order in all_orders:
        A_n = scale_A_n(modes_p, modes_m, pairs, order)
        checks: list[dict] = []
        n_floor = 0
        max_e_abs = 0.0
        max_e_rel_strong = 0.0
        order_abs_fail = False
        order_rel_fail = False

        for pr in pairs:
            ia, ib = int(pr["id_alpha"]), int(pr["id_neg"])
            directions = [("A", ia, ib)]
            if pr["kind"] != "self_alpha0":
                directions.append(("B", ib, ia))
            for dname, id_p, id_m in directions:
                mp = modes_p[(id_p, order)]
                mm = modes_m[(id_m, order)]
                res = evaluate_pair_harmonic(
                    jx_p=mp["jx"],
                    jy_p=mp["jy"],
                    jx_m=mm["jx"],
                    jy_m=mm["jy"],
                    A_n=A_n,
                    abs_atol=abs_atol,
                    rel_atol=rel_atol,
                    eta=eta,
                )
                row = {
                    "dir": dname,
                    "id_plusN": id_p,
                    "id_minusN": id_m,
                    "kind": pr["kind"],
                    **res,
                }
                checks.append(row)
                max_e_abs = max(max_e_abs, float(res["e_abs"]))
                if not res["strong"]:
                    n_floor += 1
                else:
                    max_e_rel_strong = max(max_e_rel_strong, float(res["e_rel"]))
                if res["abs_fail"]:
                    order_abs_fail = True
                    if order in hard_orders:
                        abs_fails.append({"H": order, **row})
                if res["rel_fail"]:
                    order_rel_fail = True
                    if order in hard_orders:
                        rel_fails.append({"H": order, **row})

        order_pass = (not order_abs_fail) and (not order_rel_fail)
        if order in hard_orders and not order_pass:
            hard_pass = False

        # Auxiliary ICS/CS skipped here (needs nt/dt/omega0); node Jones is the hard gate.
        ics_aux = {"note": "ICS/CS auxiliary not evaluated in V2 core; Jones is primary"}

        by_order[str(order)] = {
            "A_n": A_n,
            "max_e_abs": max_e_abs,
            "max_e_rel_strong": max_e_rel_strong,
            "n_checks": len(checks),
            "n_below_signal_floor": n_floor,
            "pass": order_pass,
            "hard": order in hard_orders,
            "ics_cs_aux": ics_aux,
            "worst": max(checks, key=lambda r: float(r["e_abs"])),
            "n_abs_fail": sum(1 for r in checks if r["abs_fail"]),
            "n_rel_fail": sum(1 for r in checks if r["rel_fail"]),
        }

    return {
        "state": stem,
        "status": "OK" if hard_pass else "FAIL",
        "cep_pi_v2_pass": hard_pass,
        "gate_version": "CEP_PI_V2",
        "abs_atol": abs_atol,
        "rel_atol": rel_atol,
        "eta": eta,
        "hard_orders": list(hard_orders),
        "diag_orders": list(diag_orders),
        "n_pairs": len(pairs),
        "n_self_alpha0": sum(1 for p in pairs if p["kind"] == "self_alpha0"),
        "by_order": by_order,
        "abs_fails": abs_fails[:40],
        "rel_fails": rel_fails[:40],
        "logic": (
            "all: e_abs=d/A_n; strong: also e_rel=d/max(|J+|,|J-|); "
            "weak: BELOW_SIGNAL_FLOOR, rel diagnostic only; both ±α directions"
        ),
    }


def apply_unitary_to_state_modes(
    modes: dict[tuple[int, int], dict],
    u00: complex,
    u01: complex,
    u10: complex,
    u11: complex,
) -> dict[tuple[int, int], dict]:
    return apply_unitary_to_modes(modes, u00, u01, u10, u11)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outroot", type=Path, required=True)
    ap.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    ap.add_argument("--abs-atol", type=float, default=CEP_V2_ABS_ATOL)
    ap.add_argument("--rel-atol", type=float, default=CEP_V2_REL_ATOL)
    ap.add_argument("--eta", type=float, default=CEP_V2_ETA)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--historical-v1-status", default="FAIL")
    args = ap.parse_args()

    states = [
        audit_state_cep_v2(
            stem,
            args.outroot,
            abs_atol=args.abs_atol,
            rel_atol=args.rel_atol,
            eta=args.eta,
        )
        for stem in args.states
    ]
    all_pass = all(s.get("status") == "OK" and s.get("cep_pi_v2_pass") for s in states)
    report = {
        "gate": "CEP_PI_V2",
        "status": "PASS" if all_pass else "FAIL",
        "labels": {
            "CHUNK_NUMERICAL_EQUIVALENCE_V3.1": "PASS",
            "CEP_PI_V1": args.historical_v1_status,
            "CEP_PI_V2": "PASS" if all_pass else "FAIL",
            "GH5": "HOLD",
        },
        "outroot": str(args.outroot),
        "frozen_thresholds": {
            "abs_atol": args.abs_atol,
            "rel_atol": args.rel_atol,
            "eta": args.eta,
            "hard_orders": list(HARD_ORDERS),
            "note": "abs from chunk V3.1 floor; rel keeps historical CEP 1e-3",
        },
        "states": states,
        "note": (
            "chunk V3.1 closed. CEP_PI_V1 FAIL retained. "
            "GH5 still requires CEP_PI_V2 PASS + GH5 manifests/provenance freeze."
        ),
    }
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
