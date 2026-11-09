#!/usr/bin/env python3
"""Summarize the one-shot UPF projector check. Does not choose 232 vs SCDM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SHELLS = ["Cr_s", "Cr_p", "Cr_d", "I_s", "I_p", "I_d"]
HOT = ("k_eta", "k_H5", "k_H7")
COMPETITORS = ("Cr_d", "Cr_p", "I_p")
SCHEMA = "upf_projector_c_v1"


def summarize_center(c: dict) -> dict:
    wt = c["Q32_rx_weighted"]["W_S_mean_trial"]
    uw = c["Q32_unweighted"]["W_S_mean_trial"]
    cover_wt = float(sum(wt[s] for s in SHELLS))
    leader = max(SHELLS, key=lambda s: wt[s])
    max_comp = max(wt[s] for s in COMPETITORS)
    bands = sorted(c["Q32_bands"], key=lambda b: b["rweight_Pi_rx_col"], reverse=True)
    top8 = bands[:8]
    n_id = sum(1 for b in top8 if max(SHELLS, key=lambda s: b["trial"][s]) == "I_d")
    wsum8 = sum(b["rweight_Pi_rx_col"] for b in top8) or 1.0
    w_id8 = sum(
        b["rweight_Pi_rx_col"]
        for b in top8
        if max(SHELLS, key=lambda s: b["trial"][s]) == "I_d"
    )
    return {
        "label": c["label"],
        "coverage_rx": cover_wt,
        "coverage_unweighted": float(sum(uw[s] for s in SHELLS)),
        "leader_rx": leader,
        "I_d_rx": float(wt["I_d"]),
        "Cr_d_rx": float(wt["Cr_d"]),
        "Cr_p_rx": float(wt["Cr_p"]),
        "I_p_rx": float(wt["I_p"]),
        "max_competitor": max(COMPETITORS, key=lambda s: wt[s]),
        "I_d_over_max_competitor": float(wt["I_d"] / max_comp) if max_comp > 0 else None,
        "top8_I_d_winner": n_id,
        "top8_I_d_weight_fraction": float(w_id8 / wsum8),
        "W_S_rx": {s: float(wt[s]) for s in SHELLS},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=Path, required=True)
    ap.add_argument("--radials", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    rad = json.loads(args.radials.read_text(encoding="utf-8"))
    centers = screen["centers"] if screen.get("centers") else screen["families"][0]["centers"]
    rows = [summarize_center(c) for c in centers]
    by_k = {r["label"]: r for r in rows}
    hot = [by_k[k] for k in HOT if k in by_k]
    id_leader = all(r["leader_rx"] == "I_d" for r in hot)
    over = [r["I_d_over_max_competitor"] for r in hot]
    report = {
        "schema": SCHEMA,
        "NOT": [
            "232-band parent started",
            "SCDM started",
            "chemical I-d 144",
            "HHG mechanism",
            "zona scan",
        ],
        "working_hypothesis": (
            "optically active Q is relatively diffuse with an I-d signature; "
            "chemical identity is not closed"
        ),
        "projector_definition": rad.get("definition"),
        "mean_r_bohr": rad.get("mean_r_bohr"),
        "hot_k": HOT,
        "hot_k_I_d_leader": id_leader,
        "hot_k_min_I_d_over_competitor": min(over) if over and all(x is not None for x in over) else None,
        "by_k": by_k,
        "questions": {
            "I_d_still_leads_on_eta_H5_H7": id_leader,
            "I_d_above_Cr_d_Cr_p_I_p": {
                k: {
                    "I_d": by_k[k]["I_d_rx"],
                    "Cr_d": by_k[k]["Cr_d_rx"],
                    "Cr_p": by_k[k]["Cr_p_rx"],
                    "I_p": by_k[k]["I_p_rx"],
                    "ratio_vs_max_competitor": by_k[k]["I_d_over_max_competitor"],
                }
                for k in HOT
                if k in by_k
            },
            "lead_on_optically_active_Q": {
                k: {
                    "top8_I_d_winner": by_k[k]["top8_I_d_winner"],
                    "top8_I_d_weight_fraction": by_k[k]["top8_I_d_weight_fraction"],
                }
                for k in HOT
                if k in by_k
            },
        },
        "human_gate": [
            "232-band reference parent at k10/k20 classical only",
            "drop the I-d label and construct a projectability/SCDM parent",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"hot_k_I_d_leader={id_leader}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
