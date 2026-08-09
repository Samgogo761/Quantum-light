#!/usr/bin/env python3
"""Rank full112 GH3 ±N candidate states by ICS / CS / ensemble Jo.

ICS/CS are read at the Fortran ``harmonic_fft_index`` bin (not a ±0.25 window).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hhg_fft_utils import parse_input_nml, pick_ics_cs_at_order


def load_node_modes(path: Path) -> dict[tuple[int, int], tuple[complex, complex, float]]:
    rows: dict[tuple[int, int], tuple[complex, complex, float]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 10:
            raise ValueError(f"{path}: expected 10 columns, got {len(f)} in: {line[:80]}")
        nid = int(f[0])
        w = float(f[1])
        order = int(f[4])
        jx = complex(float(f[5]), float(f[6]))
        jy = complex(float(f[7]), float(f[8]))
        rows[(nid, order)] = (jx, jy, w)
    return rows


def amp(jx: complex, jy: complex) -> float:
    return math.sqrt(abs(jx) ** 2 + abs(jy) ** 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", type=Path, required=True)
    ap.add_argument(
        "--states",
        nargs="+",
        default=["sv_r2p5_th000", "sv_r2p5_th180", "sv_r0p5_th180"],
    )
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 5, 7, 9, 10])
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    grid = parse_input_nml(args.outroot / f"{args.states[0]}_plusN" / "input.nml")
    dt = float(grid["dt_au"])
    nt = int(grid["nt"])
    omega0 = float(grid["omega0"])

    summary = []
    for stem in args.states:
        plus = args.outroot / f"{stem}_plusN"
        minus = args.outroot / f"{stem}_minusN"
        if not (plus / "SUCCESS").exists() or not (minus / "SUCCESS").exists():
            summary.append({"state": stem, "status": "INCOMPLETE", "plus": str(plus), "minus": str(minus)})
            continue
        modes_p = load_node_modes(plus / "HHG_nodes_modes.dat")
        modes_m = load_node_modes(minus / "HHG_nodes_modes.dat")

        per_order = {}
        for order in args.orders:
            jox = 0.0 + 0.0j
            joy = 0.0 + 0.0j
            w_used = 0.0
            n_nodes = 0
            for (nid, o), (jxp, jyp, w) in modes_p.items():
                if o != order:
                    continue
                key = (nid, o)
                if key not in modes_m:
                    raise SystemExit(f"{stem}: missing -N node {nid} H{order}")
                jxm, jym, wm = modes_m[key]
                if abs(w - wm) > 1.0e-12:
                    raise SystemExit(f"{stem}: weight mismatch node {nid}: {w} vs {wm}")
                jox += w * 0.5 * (jxp - jxm)
                joy += w * 0.5 * (jyp - jym)
                w_used += w
                n_nodes += 1
            ics_p = pick_ics_cs_at_order(plus / "HHG_ics_cs.dat", order, nt=nt, dt=dt, omega0=omega0)
            ics_m = pick_ics_cs_at_order(minus / "HHG_ics_cs.dat", order, nt=nt, dt=dt, omega0=omega0)
            per_order[str(order)] = {
                "ICS_plus": ics_p["ICS"],
                "CS_plus": ics_p["CS"],
                "ICS_minus": ics_m["ICS"],
                "CS_minus": ics_m["CS"],
                "ics_fft_index": ics_p["fft_index"],
                "ics_h_bin": ics_p["harmonic_order_bin"],
                "Jo_amp": amp(jox, joy),
                "Jo_jx_re": jox.real,
                "Jo_jx_im": jox.imag,
                "Jo_jy_re": joy.real,
                "Jo_jy_im": joy.imag,
                "weight_sum": w_used,
                "n_nodes": n_nodes,
                "note": "weighted sum first; do not zero low-SNR nodes",
            }
        summary.append({"state": stem, "status": "OK", "orders": per_order})

    for item in summary:
        if item["status"] != "OK":
            continue
        o2 = item["orders"].get("2", {})
        item["rank_keys"] = {
            "Jo2_amp": o2.get("Jo_amp"),
            "ICS2_plus": o2.get("ICS_plus"),
            "CS2_plus": o2.get("CS_plus"),
        }

    text = json.dumps(summary, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
