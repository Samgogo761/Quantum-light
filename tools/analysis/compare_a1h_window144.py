#!/usr/bin/env python3
"""A1-H: independent 6-k QE eigenvalues vs interpolated WINDOW144 H(k).

Unique global energy shift from Gamma only. Window is Wannier dis_win
[-6, 8] eV on the .eig / QE eigenvalue scale. Do not use the 6-k NSCF Fermi.

A1-H PASS does not mean WINDOW144 HHG is trusted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_tb_eigenvalues import eigvals_at, read_hamiltonian_blocks

KPOINTS = [
    {"label": "k_ctrl", "crystal": (0.00, 0.00, 0.00), "role": "gamma_on_mesh"},
    {"label": "k_eta", "crystal": (0.10, 0.55, 0.00), "role": "eta_winner_off_mesh"},
    {"label": "k_occ", "crystal": (0.80, 0.05, 0.00), "role": "set1_edge_occ_off_mesh"},
    {"label": "k_bg", "crystal": (0.35, 0.75, 0.00), "role": "baseline_eta_off_mesh"},
    {"label": "k_H5", "crystal": (0.45, 0.55, 0.00), "role": "H5_candidate_off_mesh"},
    {"label": "k_H7", "crystal": (0.45, 0.05, 0.00), "role": "H7_candidate_off_mesh"},
]
WIN_MIN = -6.0
WIN_MAX = 8.0
N_WINDOW = 144
N_Q = 32


def parse_qe_nscf_bands(path: Path) -> tuple[list[np.ndarray], float | None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    fermi = None
    m_ef = re.search(r"the Fermi energy is\s+([-\d.]+)\s+ev", text, re.I)
    if m_ef:
        fermi = float(m_ef.group(1))
    blocks = list(
        re.finditer(
            r"k\s*=\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+[^\n]*bands \(ev\):\s*\n\s*\n((?:[ \t-.\d]+\n)+)",
            text,
        )
    )
    if len(blocks) < 6:
        raise RuntimeError(f"expected 6 QE k-blocks, got {len(blocks)}")
    bands = []
    for blk in blocks[:6]:
        vals = np.fromstring(blk.group(1), sep=" ", dtype=float)
        if vals.size != 200:
            raise RuntimeError(f"expected 200 bands, got {vals.size}")
        bands.append(vals)
    return bands, fermi


def window_mask(e: np.ndarray) -> np.ndarray:
    return (e >= WIN_MIN) & (e <= WIN_MAX)


def stats(diff: np.ndarray) -> dict:
    if diff.size == 0:
        return {"n": 0, "rms_eV": None, "maxabs_eV": None, "mean_eV": None}
    return {
        "n": int(diff.size),
        "rms_eV": float(np.sqrt(np.mean(diff * diff))),
        "maxabs_eV": float(np.max(np.abs(diff))),
        "mean_eV": float(np.mean(diff)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nscf", type=Path, required=True)
    ap.add_argument("--tb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    qe_bands, fermi_6k = parse_qe_nscf_bands(args.nscf)
    print(f"QE 6k Fermi (do not use for window) = {fermi_6k}", flush=True)

    t0 = time.time()
    print(f"Reading WINDOW144 H(R) from {args.tb} ...", flush=True)
    irvec, ndegen, h_r = read_hamiltonian_blocks(args.tb)
    print(
        f"  nwann={h_r.shape[1]} nrpts={h_r.shape[0]} in {time.time() - t0:.1f}s",
        flush=True,
    )
    if h_r.shape[1] != N_WINDOW:
        raise SystemExit(f"expected 144 WFs, got {h_r.shape[1]}")

    tb_e = []
    for spec in KPOINTS:
        tb_e.append(np.sort(eigvals_at(irvec, ndegen, h_r, spec["crystal"]).real))

    qe_win = []
    for e in qe_bands:
        sel = np.sort(e[window_mask(e)])
        qe_win.append(sel)

    n_gamma = int(qe_win[0].size)
    if n_gamma != N_WINDOW:
        print(f"WARNING Gamma window count {n_gamma} != 144", flush=True)
        n_match = min(n_gamma, N_WINDOW)
        shift = float(np.mean(tb_e[0][:n_match] - qe_win[0][:n_match]))
    else:
        shift = float(np.mean(tb_e[0] - qe_win[0]))

    rows = []
    off_rms = []
    off_q_rms = []
    for i, spec in enumerate(KPOINTS):
        qe = qe_win[i]
        tb = tb_e[i]
        n_w = int(qe.size)
        n = min(n_w, tb.size)
        qe_s = qe[:n] + shift
        tb_s = tb[:n]
        d = tb_s - qe_s
        q_slice = slice(max(0, n - N_Q), n)
        occ_slice = slice(0, min(84, n))
        row = {
            **spec,
            "crystal": list(spec["crystal"]),
            "n_qe_window": n_w,
            "window_is_144": n_w == N_WINDOW,
            "qe_window_min_eV": float(qe[0]) if n_w else None,
            "qe_window_max_eV": float(qe[-1]) if n_w else None,
            "tb_min_eV": float(tb[0]),
            "tb_max_eV": float(tb[-1]),
            "all": stats(d),
            "occ84": stats(d[occ_slice]),
            "q32": stats(d[q_slice]),
        }
        rows.append(row)
        if spec["label"] != "k_ctrl":
            off_rms.append(row["all"]["rms_eV"])
            off_q_rms.append(row["q32"]["rms_eV"])
        print(
            f"{spec['label']:6s} nwin={n_w:3d}  rms={row['all']['rms_eV']:.6e}  "
            f"max={row['all']['maxabs_eV']:.6e}  q32_rms={row['q32']['rms_eV']:.6e}",
            flush=True,
        )

    payload = {
        "schema": "a1h_window144_vs_qe_v1",
        "job": 32293,
        "A1H_PASS_NE_HHG_TRUSTED": True,
        "window_eV": [WIN_MIN, WIN_MAX],
        "nscf_6k_fermi_eV_not_used": fermi_6k,
        "global_shift_from_gamma_eV": shift,
        "note_shift": "Added to QE window eigenvalues. Determined once at Gamma. Not per-k.",
        "gamma": rows[0],
        "kpoints": rows,
        "off_mesh_rms_max_eV": float(max(off_rms)) if off_rms else None,
        "off_mesh_q32_rms_max_eV": float(max(off_q_rms)) if off_q_rms else None,
        "read_tb_seconds": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
