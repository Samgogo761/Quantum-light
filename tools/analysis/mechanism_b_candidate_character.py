#!/usr/bin/env python3
"""Certified-label character of Track-B candidate pairs. Not attribution."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from construct_p_conjugate_tb import read_tb_blocks

N_OCC = 84
CANDIDATES = [
    {"tag": "H5_rank1", "k": [0.50, 0.55, 0.0], "pairs": [[73, 92], [73, 90]]},
    {"tag": "H5_offmesh_a1h", "k": [0.45, 0.55, 0.0], "pairs": [[68, 86], [73, 92]]},
    {"tag": "H7_onmesh_not_a1h", "k": [0.50, 0.00, 0.0], "pairs": [[56, 90], [53, 90]]},
        {"tag": "H7_offmesh_a1h", "k": [0.45, 0.05, 0.0], "pairs": [[56, 90], [53, 90]]},
]


def interpolate(irvec, ndegen, h_r, r_r, kfrac: np.ndarray):
    phase = np.exp(2j * np.pi * (irvec @ kfrac)) / ndegen
    hk = np.tensordot(phase, h_r, axes=(0, 0))
    hk = 0.5 * (hk + hk.conj().T)
    rk = np.tensordot(phase, r_r, axes=(0, 0))
    evals, u = np.linalg.eigh(hk)
    rx = u.conj().T @ rk[:, :, 0] @ u
    return evals, rx, u


def weights(u_col: np.ndarray, labels: list[dict]) -> dict:
    w = np.abs(u_col) ** 2
    out = {
        "Cr": 0.0,
        "I": 0.0,
        "layer1": 0.0,
        "layer2": 0.0,
        "I_interface": 0.0,
        "I_vacuum": 0.0,
        "px": 0.0,
        "py": 0.0,
        "pz": 0.0,
    }
    for i, lab in enumerate(labels):
        wi = float(w[i])
        out[lab["species"]] += wi
        out[f"layer{lab['layer']}"] += wi
        if lab["species"] == "I":
            out[f"I_{lab['site']}"] += wi
            orb = lab["orbital_nnkp"]
            if orb in out:
                out[orb] += wi
    return {k: float(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tb", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["wf"]
    print(f"reading {args.tb} ...", flush=True)
    irvec, ndegen, h_r, r_r = read_tb_blocks(args.tb, read_position=True)
    recs = []
    for cand in CANDIDATES:
        k = np.array(cand["k"], dtype=float)
        evals, rx, u = interpolate(irvec, ndegen, h_r, r_r, k)
        item = {
            "tag": cand["tag"],
            "frac_k": cand["k"],
            "gap_eV": float(evals[N_OCC] - evals[N_OCC - 1]),
            "pairs": [],
        }
        for v, c in cand["pairs"]:
            iv, ic = v - 1, c - 1
            item["pairs"].append(
                {
                    "pair_1based": [v, c],
                    "dE_eV": float(evals[ic] - evals[iv]),
                    "|rx|^2": float(np.abs(rx[iv, ic]) ** 2),
                    "v": weights(u[:, iv], labels),
                    "c": weights(u[:, ic], labels),
                }
            )
        recs.append(item)
        print(f"  done {cand['tag']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": "CANDIDATE_CHARACTER_NOT_ATTRIBUTION",
        "orbital_pxpypz": "NNKP_INITIAL_PROJECTION",
        "species_layer_site": "CERTIFIED_FINAL_STATE",
        "n_occ": N_OCC,
        "points": recs,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
