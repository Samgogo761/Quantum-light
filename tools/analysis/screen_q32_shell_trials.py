#!/usr/bin/env python3
"""Screen Q32 / 3.5-8 eV trial-shell weights and |Pi r Q|-weighted G_S.

Trial A overlaps are screening numbers, not orthogonal percentages.
Does not choose a new Wannier parent or announce an HHG mechanism.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1o_berry_math import berry_from_pm_crystal, energy_pi_q, phase_align_columns, window_indices
from audit_pi_rq_coupling import load_eig, parse_nnkp, read_mmn_for_k

WIN_MIN = -6.0
WIN_MAX = 8.0
SCREEN_LO = 3.5
SCREEN_HI = 8.0
N_WINDOW = 144
N_Q = 32
N_PI = 112
DELTA = 0.002
SCHEMA = "q32_shell_trial_screen_v2"


def families_from_map(shell_map: dict) -> list[dict]:
    fams = shell_map.get("families")
    if fams:
        return fams
    return [
        {
            "tag": "r1_z1p0",
            "rvalue": shell_map.get("rvalue", 1),
            "zona": shell_map.get("zona_bohr_inv", 1.0),
            "radial": shell_map.get("radial_rvalue_1"),
            "shells": shell_map["shells"],
        }
    ]


def load_amn_k(path: Path, ik_1based: int, n_wann: int, nbnd: int) -> np.ndarray:
    a = np.zeros((nbnd, n_wann), dtype=complex)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.readline()
        _nb, _nw, _nk = (int(x) for x in handle.readline().split()[:3])
        for line in handle:
            parts = line.split()
            if int(parts[2]) != ik_1based:
                continue
            n = int(parts[0]) - 1
            m = int(parts[1]) - 1
            a[n, m] = complex(float(parts[3]), float(parts[4]))
    return a


def shell_weights(a_band: np.ndarray, shells: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for sh in shells:
        sl = slice(int(sh["index0"]), int(sh["index0"]) + int(sh["n_spinor"]))
        out[sh["shell"]] = float(np.sum(np.abs(a_band[:, sl]) ** 2))
    return out


def berry_rx_center(
    mats: list[np.ndarray],
    headers: list[tuple],
    neighbors: list[dict],
    eig_all: np.ndarray,
    ik_1based: int,
    recip: np.ndarray,
) -> np.ndarray:
    sel = [nb for nb in neighbors if abs(nb["delta"] - DELTA) < 1e-12]
    by_dir = {nb["dir"]: nb for nb in sel}
    idx_k = window_indices(eig_all[ik_1based - 1], WIN_MIN, WIN_MAX)
    aligned = {}
    for name in ("+x", "-x", "+y", "-y"):
        nb = by_dir[name]
        slot = int(nb["nn_slot"])
        mat = mats[slot]
        _ik, jk, _g1, _g2, _g3 = headers[slot]
        idx_b = window_indices(eig_all[jk - 1], WIN_MIN, WIN_MAX)
        block = mat[np.ix_(idx_k, idx_b)]
        aligned[name] = phase_align_columns(block)
    return berry_from_pm_crystal(
        aligned["+x"], aligned["-x"], aligned["+y"], aligned["-y"], DELTA, recip
    )


def screen_one(
    label: str,
    ik: int,
    eig_k: np.ndarray,
    a: np.ndarray,
    r_win: np.ndarray,
    shells: list[dict],
    trial_note: str | None = None,
) -> dict:
    idx_win = window_indices(eig_k, WIN_MIN, WIN_MAX)
    pi, q = energy_pi_q(N_WINDOW, N_Q)
    q_abs = idx_win[q]
    pi_abs = idx_win[pi]
    w_q = np.sum(np.abs(r_win[0][np.ix_(pi, q)]) ** 2, axis=0)
    denom = float(np.sum(w_q))
    idx_hi = np.flatnonzero((eig_k >= SCREEN_LO) & (eig_k <= SCREEN_HI))

    def accumulate(band_abs: np.ndarray, weights: np.ndarray | None) -> dict:
        w_s = {sh["shell"]: 0.0 for sh in shells}
        wsum = 0.0
        for j, n in enumerate(band_abs):
            wt = 1.0 if weights is None else float(weights[j])
            wsum += wt
            sw = shell_weights(a[n : n + 1], shells)
            for name, val in sw.items():
                w_s[name] += wt * val
        if wsum > 0:
            w_s = {k: v / wsum for k, v in w_s.items()}
        return {"W_S_mean_trial": w_s, "weight_sum": wsum}

    unweighted = accumulate(q_abs, None)
    weighted = accumulate(q_abs, w_q)
    weighted["G_S_rx_weighted"] = dict(weighted["W_S_mean_trial"])
    hi = accumulate(idx_hi, None)
    bands = []
    for j, n in enumerate(q_abs):
        rec = {
            "qe_band_1based": int(n + 1),
            "window_index": int(q[j]),
            "energy_eV": float(eig_k[n]),
            "rweight_Pi_rx_col": float(w_q[j]),
            "trial": shell_weights(a[n : n + 1], shells),
        }
        bands.append(rec)
    return {
        "label": label,
        "k_index_1based": ik,
        "n_window": int(idx_win.size),
        "n_Q32": int(q_abs.size),
        "n_3p5_8eV": int(idx_hi.size),
        "PiQ_rx_colweight_sum": denom,
        "Q32_unweighted": unweighted,
        "Q32_rx_weighted": weighted,
        "screen_3p5_8eV_unweighted": hi,
        "Q32_bands": bands,
        "note": trial_note
        or "trial |A|^2 sums are screening metrics, not an orthogonal partition of unity.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amn", type=Path, required=True)
    ap.add_argument("--mmn", type=Path, required=True)
    ap.add_argument("--eig", type=Path, required=True)
    ap.add_argument("--nnkp-a1o", type=Path, required=True)
    ap.add_argument("--star-map", type=Path, required=True)
    ap.add_argument("--shell-map", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    shell_map = json.loads(args.shell_map.read_text(encoding="utf-8"))
    star = json.loads(args.star_map.read_text(encoding="utf-8"))
    eig = load_eig(args.eig)
    nnkp = parse_nnkp(args.nnkp_a1o)
    n_wann = int(shell_map["n_spinor"])
    fams = families_from_map(shell_map)
    packed = []
    for c in star["centers"]:
        ik = int(c["k_index_1based"])
        _nb, _nk, _nntot, headers, mats = read_mmn_for_k(args.mmn, ik)
        r_win = berry_rx_center(
            mats, headers, c["neighbors"], eig, ik, nnkp["recip_lattice"]
        )
        a = load_amn_k(args.amn, ik, n_wann, eig.shape[1])
        packed.append((c, ik, r_win, a))
    family_reports = []
    for fam in fams:
        centers = []
        for c, ik, r_win, a in packed:
            rec = screen_one(
                c["label"],
                ik,
                eig[ik - 1],
                a,
                r_win,
                fam["shells"],
                trial_note=shell_map.get("trial_note"),
            )
            rec["family"] = fam["tag"]
            rec["rvalue"] = fam.get("rvalue")
            rec["zona"] = fam.get("zona")
            centers.append(rec)
        family_reports.append(
            {
                "tag": fam["tag"],
                "rvalue": fam.get("rvalue"),
                "zona": fam.get("zona"),
                "radial": fam.get("radial"),
                "centers": centers,
            }
        )
    report = {
        "schema": SCHEMA,
        "NOT": [
            "new Wannier parent",
            "chemical I-d 144",
            "232-band parent",
            "orthogonal orbital percent",
            "HHG mechanism attribution",
            "GH7",
            "k40",
        ],
        "radial": shell_map.get("radial_rvalue_1"),
        "win_eV": [WIN_MIN, WIN_MAX],
        "screen_eV": [SCREEN_LO, SCREEN_HI],
        "families": family_reports,
        "centers": family_reports[0]["centers"] if len(family_reports) == 1 else [],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
