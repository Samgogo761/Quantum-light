#!/usr/bin/env python3
"""Candidate-mechanism static map on an existing Wannier TB (Track B).

This is NOT a transition attribution. It ranks k-points and band pairs by
|r_vc|^2 near H5/H7 energy windows. H7 = 7 hbar omega_0 does not mean the
static pair with that gap produced H7.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from construct_p_conjugate_tb import read_tb_blocks

N_OCC = 84
HBAR_OMEGA_EV = 1239.84193 / 3200.0
N_CR_WF = 40
CR_PER_ATOM = 10
# win atoms: Cr1,Cr1 (layer 1, z~0.24), Cr2,Cr2 (layer 2, z~0.50), then 12 I.
LAYER1_CR = {0, 1}
LAYER2_CR = {2, 3}


def k20_grid() -> list[tuple[int, int, float, float]]:
    out = []
    for ikx in range(20):
        for iky in range(20):
            out.append((ikx, iky, ikx / 20.0, iky / 20.0))
    return out


def interpolate(irvec, ndegen, h_r, r_r, kfrac: np.ndarray):
    phase = np.exp(2j * np.pi * (irvec @ kfrac)) / ndegen
    hk = np.tensordot(phase, h_r, axes=(0, 0))
    hk = 0.5 * (hk + hk.conj().T)
    rk = np.tensordot(phase, r_r, axes=(0, 0))
    evals, u = np.linalg.eigh(hk)
    r_b = np.empty((evals.size, evals.size, 3), dtype=np.complex128)
    for a in range(3):
        r_b[:, :, a] = u.conj().T @ rk[:, :, a] @ u
    return evals, r_b, u


def wf_layer_species(nw: int) -> tuple[np.ndarray, np.ndarray]:
    species = np.array(["I"] * nw, dtype="<U2")
    layer = np.zeros(nw, dtype=int)
    for i in range(min(N_CR_WF, nw)):
        species[i] = "Cr"
        atom = i // CR_PER_ATOM
        layer[i] = 1 if atom in LAYER1_CR else 2
    # I atoms 0-5 first block after Cr ~ layer 1 (z~0.17-0.30), 6-11 layer 2.
    for j, i in enumerate(range(N_CR_WF, nw)):
        atom = j // 6
        layer[i] = 1 if atom < 6 else 2
    return species, layer


def pair_character(u: np.ndarray, v: int, c: int, species: np.ndarray, layer: np.ndarray) -> dict:
    wv = np.abs(u[:, v]) ** 2
    wc = np.abs(u[:, c]) ** 2
    return {
        "v_Cr": float(wv[species == "Cr"].sum()),
        "c_Cr": float(wc[species == "Cr"].sum()),
        "v_layer1": float(wv[layer == 1].sum()),
        "c_layer1": float(wc[layer == 1].sum()),
    }


def window_weight(de: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((de - center) / sigma) ** 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-occ", type=int, default=N_OCC)
    ap.add_argument("--sigma-eV", type=float, default=0.15)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    print(f"reading {args.tb} ...", flush=True)
    irvec, ndegen, h_r, r_r = read_tb_blocks(args.tb, read_position=True)
    nw = h_r.shape[1]
    species, layer = wf_layer_species(nw)
    print(f"nwann={nw} nrpts={irvec.shape[0]}", flush=True)

    targets = {
        "H5": 5.0 * HBAR_OMEGA_EV,
        "H7": 7.0 * HBAR_OMEGA_EV,
    }
    per_k = []
    for ikx, iky, kx, ky in k20_grid():
        evals, r_b, u = interpolate(irvec, ndegen, h_r, r_r, np.array([kx, ky, 0.0]))
        de = evals[None, args.n_occ :] - evals[: args.n_occ, None]
        rx2 = np.abs(r_b[: args.n_occ, args.n_occ :, 0]) ** 2
        rec = {
            "ikx": ikx,
            "iky": iky,
            "frac_k": [kx, ky, 0.0],
            "gap_eV": float(evals[args.n_occ] - evals[args.n_occ - 1]),
        }
        for name, center in targets.items():
            w = window_weight(de, center, args.sigma_eV)
            score = rx2 * w
            flat = int(np.argmax(score))
            iv, ic = np.unravel_index(flat, score.shape)
            rec[name] = {
                "score": float(score.sum()),
                "best_pair": [int(iv + 1), int(args.n_occ + ic + 1)],
                "best_dE_eV": float(de[iv, ic]),
                "best_|rx|^2": float(rx2[iv, ic]),
                "best_window_w": float(w[iv, ic]),
                "character": pair_character(u, int(iv), int(args.n_occ + ic), species, layer),
            }
        per_k.append(rec)
        if (ikx * 20 + iky + 1) % 40 == 0:
            print(f"  {ikx * 20 + iky + 1}/400", flush=True)

    out = {
        "label": "CANDIDATE_MECHANISM_NOT_ATTRIBUTION",
        "tb": str(args.tb),
        "n_occ": args.n_occ,
        "hbar_omega_eV": HBAR_OMEGA_EV,
        "sigma_eV": args.sigma_eV,
        "score": "sum_{v<=nocc,c>nocc} |r^x_vc|^2 * Gauss(dE - n hbar omega)",
        "targets_eV": targets,
        "top": {},
    }
    for name in targets:
        ranked = sorted(per_k, key=lambda r: r[name]["score"], reverse=True)
        out["top"][name] = [
            {
                "rank": i + 1,
                "ikx": r["ikx"],
                "iky": r["iky"],
                "frac_k": r["frac_k"],
                "gap_eV": r["gap_eV"],
                **r[name],
            }
            for i, r in enumerate(ranked[: args.top_k])
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
