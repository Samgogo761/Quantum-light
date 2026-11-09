#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orbital-resolved analysis interface on a *fixed* full112 Wannier basis.

This round: scaffold only — load full112 tb, project observables onto
orbital groups (Cr-d / I-px,py,pz by WF index convention), and dump
group-resolved dipole / hopping weight tables. No quantum-light / QRT.

WF layout (full112, CrI3 SOC):
  1..40   : 4 Cr × 10 (d, spinor)
  41..112 : 12 I × 6 (px,py,pz × 2 spinor)  — order within each I: typically
            2×px, 2×py, 2×pz (same as historical construct_p_conjugate).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


I_WF_PER_ATOM = 6
N_CR = 4
CR_WF = 10


def group_of_wf(i0: int) -> str:
    """0-based WF index -> group name."""
    if i0 < N_CR * CR_WF:
        return "Cr_d"
    b = i0 - N_CR * CR_WF
    intra = b % I_WF_PER_ATOM
    # 0,1 px; 2,3 py; 4,5 pz
    if intra < 2:
        return "I_px"
    if intra < 4:
        return "I_py"
    return "I_pz"


def read_tb_hop_weights(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        line = fh.readline()
        if "." in line:
            fh.readline()
            fh.readline()
            nw = int(fh.readline().split()[0])
        else:
            nw = int(line.split()[0])
        nr = int(fh.readline().split()[0])
        ndegen = []
        while len(ndegen) < nr:
            ndegen.extend(int(x) for x in fh.readline().split())
        # skip H for weight of |H| transport (off-diagonal R!=0 optional)
        # We accumulate onsite |H_ij| and all |H|
        W = np.zeros((nw, nw))
        for ir in range(nr):
            fh.readline()
            rvec = [int(x) for x in fh.readline().split()[:3]]
            for _ in range(nw * nw):
                m, n, rr, ri = fh.readline().split()[:4]
                m, n = int(m) - 1, int(n) - 1
                amp = abs(complex(float(rr), float(ri)))
                if rvec != [0, 0, 0] or m != n:
                    W[m, n] += amp
        # skip r block
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", type=Path, required=True, help="full112 CrI3_tb.dat")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    W = read_tb_hop_weights(args.tb)
    nw = W.shape[0]
    groups = [group_of_wf(i) for i in range(nw)]
    uniq = ["Cr_d", "I_px", "I_py", "I_pz"]
    gidx = {g: i for i, g in enumerate(uniq)}
    G = np.zeros((len(uniq), len(uniq)))
    for i in range(nw):
        for j in range(nw):
            G[gidx[groups[i]], gidx[groups[j]]] += W[i, j]

    with (args.outdir / "orbital_group_hopping_weights.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from\\to"] + uniq)
        for i, gi in enumerate(uniq):
            w.writerow([gi] + [f"{G[i, j]:.6e}" for j in range(len(uniq))])

    # API stub manifest
    (args.outdir / "INTERFACE.md").write_text(
        """# Orbital-resolved interface (full112 fixed basis)

## Groups
- `Cr_d`: WF 1–40
- `I_px` / `I_py` / `I_pz`: within each I atom, spinor pairs (0–1 / 2–3 / 4–5)

## Current outputs
- `orbital_group_hopping_weights.csv`: transport |H| weights between groups

## Planned hooks (not this round)
- Project SBE currents onto group-resolved velocity / dipole channels
- Band-pair tagging for H10+ anomalies using full112 eigenvectors
- No quantum-light / QRT in this interface yet
""",
        encoding="utf-8",
    )
    print("Wrote", args.outdir)


if __name__ == "__main__":
    main()
