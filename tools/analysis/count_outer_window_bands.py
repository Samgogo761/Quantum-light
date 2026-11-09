#!/usr/bin/env python3
"""Count QE bands in [-6, E] vs k and pick an outer-window E_max for 232.

Constraint: min_k N_outer(k) > num_wann (disentangle surplus), not N_outer == 232.
Target surplus matches production 112: 144/112, so ceil(232 * 144/112) = 299.
Do not shift by the NSCF Fermi; use printed KS eigenvalues as A1-H did.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

WIN_MIN_EV = -6.0
N_WANN = 232
PROD_WIN_COUNT = 144
PROD_WANN = 112
TARGET_RATIO = PROD_WIN_COUNT / PROD_WANN
TARGET_N = int(math.ceil(N_WANN * TARGET_RATIO))
MIN_N_STRICT = N_WANN + 1
HOT_LABELS = ("gamma", "eta", "H5", "H7", "mesh_X", "mesh_Y", "mesh_XY", "mesh_K")

BAND_BLOCK = re.compile(
    r"k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)[^\n]*bands \(ev\):\s*\n\s*\n((?:[ \t-.\d]+\n)+)",
    re.I,
)


def parse_qe_nscf_kblocks(text: str) -> list[dict]:
    blocks = []
    for m in BAND_BLOCK.finditer(text):
        vals = np.fromstring(m.group(4), sep=" ", dtype=float)
        if vals.size == 0:
            raise RuntimeError("empty band block")
        blocks.append(
            {
                "k_cart": [float(m.group(1)), float(m.group(2)), float(m.group(3))],
                "e_eV": vals,
            }
        )
    if not blocks:
        raise RuntimeError("no QE k-blocks with bands (ev)")
    return blocks


def select_kblocks(blocks: list[dict], n_k: int) -> list[dict]:
    if len(blocks) == n_k:
        return blocks
    if len(blocks) > n_k and len(blocks) % n_k == 0:
        return blocks[-n_k:]
    raise RuntimeError(f"expected {n_k} k-blocks (or a reprint multiple), got {len(blocks)}")


def n_in_window(e: np.ndarray, emin: float, emax: float) -> int:
    return int(np.sum((e >= emin) & (e <= emax)))


def min_n_at(blocks: list[dict], emin: float, emax: float) -> int:
    return min(n_in_window(b["e_eV"], emin, emax) for b in blocks)


def smallest_emax_for_min_n(blocks: list[dict], emin: float, need: int) -> float | None:
    candidates = np.unique(
        np.concatenate([b["e_eV"][b["e_eV"] >= emin] for b in blocks])
    )
    for emax in candidates:
        if min_n_at(blocks, emin, float(emax)) >= need:
            return float(emax)
    return None


def summarize_k(blocks: list[dict], emin: float, emax: float) -> list[dict]:
    rows = []
    for i, b in enumerate(blocks):
        e = b["e_eV"]
        n = n_in_window(e, emin, emax)
        label = HOT_LABELS[i] if i < len(HOT_LABELS) else f"k{i:02d}"
        rows.append(
            {
                "ik": i,
                "label": label,
                "k_cart": b["k_cart"],
                "nbnd": int(e.size),
                "n_in_window": n,
                "e_min_eV": float(e[0]),
                "e_max_eV": float(e[-1]),
                "e_last_in_window_eV": float(e[e <= emax][-1]) if n else None,
            }
        )
    return rows


def build_report(blocks: list[dict], win_min: float = WIN_MIN_EV) -> dict:
    nbnd = int(blocks[0]["e_eV"].size)
    for b in blocks:
        if int(b["e_eV"].size) != nbnd:
            raise RuntimeError("inconsistent nbnd across k")
    n_at_8 = [n_in_window(b["e_eV"], win_min, 8.0) for b in blocks]
    e_gt232 = smallest_emax_for_min_n(blocks, win_min, MIN_N_STRICT)
    e_target = smallest_emax_for_min_n(blocks, win_min, TARGET_N)
    ceiling = min(float(b["e_eV"][-1]) for b in blocks)
    if e_target is not None:
        e_max = e_target
        pick = "target_surplus"
    elif e_gt232 is not None:
        e_max = e_gt232
        pick = "strict_gt_232_only"
    else:
        e_max = None
        pick = "nbnd_insufficient"
    n_at_pick = (
        [n_in_window(b["e_eV"], win_min, e_max) for b in blocks] if e_max is not None else []
    )
    sufficient = e_target is not None
    return {
        "schema": "ref232_outer_window_count_v1",
        "n_k": len(blocks),
        "nbnd": nbnd,
        "win_min_eV": win_min,
        "n_wann": N_WANN,
        "n_outer_must_exceed": N_WANN,
        "min_n_strict": MIN_N_STRICT,
        "target_ratio_144_over_112": TARGET_RATIO,
        "target_n_surplus": TARGET_N,
        "n_at_production_8eV": {
            "min": int(min(n_at_8)),
            "max": int(max(n_at_8)),
            "per_k": n_at_8,
        },
        "e_max_min_n_gt_232_eV": e_gt232,
        "e_max_target_surplus_eV": e_target,
        "e_max_chosen_eV": e_max,
        "e_max_choice": pick,
        "min_n_at_chosen": int(min(n_at_pick)) if n_at_pick else None,
        "max_n_at_chosen": int(max(n_at_pick)) if n_at_pick else None,
        "highest_band_min_over_k_eV": ceiling,
        "nbnd_sufficient_for_target": sufficient,
        "per_k_at_chosen": summarize_k(blocks, win_min, e_max) if e_max is not None else [],
            "hot_k_n_at_8eV": {
            HOT_LABELS[i]: n_at_8[i] for i in range(min(len(HOT_LABELS), len(n_at_8)))
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nscf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-k", type=int, default=8)
    args = ap.parse_args()
    text = args.nscf.read_text(encoding="utf-8", errors="replace")
    blocks = select_kblocks(parse_qe_nscf_kblocks(text), args.n_k)
    report = build_report(blocks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"n_k={report['n_k']} nbnd={report['nbnd']} "
        f"N[-6,8]=[{report['n_at_production_8eV']['min']},{report['n_at_production_8eV']['max']}] "
        f"E_max={report['e_max_chosen_eV']} choice={report['e_max_choice']} "
        f"min_n={report['min_n_at_chosen']}",
        flush=True,
    )
    if not report["nbnd_sufficient_for_target"]:
        print("NBND_INSUFFICIENT_FOR_TARGET_SURPLUS", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
