#!/usr/bin/env python3
"""Compare trial-shell ranking across radial families. Does not choose a parent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SHELLS = ["Cr_s", "Cr_p", "Cr_d", "I_s", "I_p", "I_d"]
SCHEMA = "shell_radial_robust_v1"


def centers_of(report: dict) -> list[dict]:
    if report.get("families"):
        raise RuntimeError("pass one family, not the whole multi-family report")
    return report["centers"]


def family_blocks(report: dict) -> list[dict]:
    if report.get("families"):
        return report["families"]
    return [
        {
            "tag": "r1_z1p0",
            "rvalue": 1,
            "zona": 1.0,
            "centers": report["centers"],
        }
    ]


def summarize_center(c: dict) -> dict:
    wt = c["Q32_rx_weighted"]["W_S_mean_trial"]
    uw = c["Q32_unweighted"]["W_S_mean_trial"]
    cover_wt = float(sum(wt[s] for s in SHELLS))
    cover_uw = float(sum(uw[s] for s in SHELLS))
    leader = max(SHELLS, key=lambda s: wt[s])
    others = [s for s in SHELLS if s != "I_d"]
    max_other = max(wt[s] for s in others)
    bands = sorted(c["Q32_bands"], key=lambda b: b["rweight_Pi_rx_col"], reverse=True)
    top8 = bands[:8]
    n_id = sum(1 for b in top8 if max(SHELLS, key=lambda s: b["trial"][s]) == "I_d")
    return {
        "label": c["label"],
        "coverage_rx": cover_wt,
        "coverage_unweighted": cover_uw,
        "leader_rx": leader,
        "I_d_rx": float(wt["I_d"]),
        "I_d_over_next": float(wt["I_d"] / max_other) if max_other > 0 else None,
        "Cr_d_plus_I_p_rx": float(wt["Cr_d"] + wt["I_p"]),
        "top8_I_d_winner": n_id,
        "W_S_rx": {s: float(wt[s]) for s in SHELLS},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--radial", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    rad = json.loads(args.radial.read_text(encoding="utf-8"))
    blocks = family_blocks(base) + family_blocks(rad)
    rows = []
    for fam in blocks:
        for c in fam["centers"]:
            rec = summarize_center(c)
            rec["family"] = fam.get("tag")
            rec["rvalue"] = fam.get("rvalue")
            rec["zona"] = fam.get("zona")
            rows.append(rec)
    by_k: dict[str, list] = {}
    for rec in rows:
        by_k.setdefault(rec["label"], []).append(rec)
    id_always = all(r["leader_rx"] == "I_d" for r in rows if r["label"] != "k_ctrl")
    coverages = [r["coverage_rx"] for r in rows]
    report = {
        "schema": SCHEMA,
        "NOT": [
            "232-band parent",
            "chemical I-d 144",
            "HHG mechanism",
            "SCDM started",
        ],
        "working_hypothesis": "I-d leading candidate on optically weighted Q; Q mixed/possibly diffuse",
        "hot_k_I_d_always_leader": id_always,
        "coverage_rx_min": min(coverages),
        "coverage_rx_max": max(coverages),
        "by_k": by_k,
        "rows": rows,
        "note": (
            "If I-d ranking flips with radial params, or coverage stays ~0.25, "
            "do not start a 232-band parent. User decides 232 vs general subspace."
        ),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} hot_k_I_d_always_leader={id_always}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
