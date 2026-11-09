#!/usr/bin/env python3
"""A1-H spectrum-matching diagnostic: sorted L∞ vs Hausdorff vs cluster assignment.

Sorted pairing of two equal-length 1D spectra is optimal for Wasserstein/L^p.
It can still look worse than Hausdorff when two clusters are filled differently.
This does NOT use wavefunction overlap; QE–Wannier projectors at off-mesh k
need extra PW overlaps and are out of scope here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_a1h_window144 import (
    KPOINTS,
    N_Q,
    N_WINDOW,
    WIN_MAX,
    WIN_MIN,
    parse_qe_nscf_bands,
    window_mask,
)
from compare_tb_eigenvalues import eigvals_at, read_hamiltonian_blocks

HBAR_OMEGA_EV = 1239.84193 / 3200.0


def hausdorff(a: np.ndarray, b: np.ndarray) -> dict:
    da = np.min(np.abs(a[:, None] - b[None, :]), axis=1)
    db = np.min(np.abs(b[:, None] - a[None, :]), axis=1)
    return {
        "tb_to_qe_max_eV": float(da.max()),
        "qe_to_tb_max_eV": float(db.max()),
        "symmetric_eV": float(max(da.max(), db.max())),
        "tb_to_qe_p95_eV": float(np.quantile(da, 0.95)),
        "qe_to_tb_p95_eV": float(np.quantile(db, 0.95)),
        "n_tb_farther_than_20meV": int(np.sum(da > 0.02)),
        "n_qe_farther_than_20meV": int(np.sum(db > 0.02)),
        "argmax_tb_to_qe": {
            "tb_index0": int(np.argmax(da)),
            "tb_eV": float(a[np.argmax(da)]),
            "nearest_qe_eV": float(b[np.argmin(np.abs(b - a[np.argmax(da)]))]),
            "gap_eV": float(da.max()),
        },
    }


def consecutive_gaps(e: np.ndarray) -> np.ndarray:
    return np.diff(e)


def cluster_ids(e: np.ndarray, gap_eV: float) -> np.ndarray:
    ids = np.zeros(e.size, dtype=int)
    c = 0
    for i in range(1, e.size):
        if e[i] - e[i - 1] > gap_eV:
            c += 1
        ids[i] = c
    return ids


def cluster_l2(tb: np.ndarray, qe: np.ndarray, gap_eV: float) -> dict:
    """Match connected components split by gaps > gap_eV, then sort inside."""
    id_t = cluster_ids(tb, gap_eV)
    id_q = cluster_ids(qe, gap_eV)
    nct, ncq = int(id_t.max()) + 1, int(id_q.max()) + 1
    # Walk both cluster sequences in energy order (already sorted).
    resid = []
    unmatched_tb = 0
    unmatched_qe = 0
    it = iq = 0
    while it < nct and iq < ncq:
        tb_c = tb[id_t == it]
        qe_c = qe[id_q == iq]
        if tb_c.size == qe_c.size:
            resid.extend((tb_c - qe_c).tolist())
            it += 1
            iq += 1
            continue
        # If counts differ, this is a real filling mismatch, not a label swap.
        if tb_c.mean() < qe_c.mean():
            unmatched_tb += int(tb_c.size)
            it += 1
        else:
            unmatched_qe += int(qe_c.size)
            iq += 1
    unmatched_tb += int(np.sum(id_t >= it))
    unmatched_qe += int(np.sum(id_q >= iq))
    r = np.asarray(resid, dtype=float)
    return {
        "gap_eV": gap_eV,
        "n_tb_clusters": nct,
        "n_qe_clusters": ncq,
        "n_paired_levels": int(r.size),
        "unmatched_tb": unmatched_tb,
        "unmatched_qe": unmatched_qe,
        "paired_rms_eV": float(np.sqrt(np.mean(r * r))) if r.size else None,
        "paired_maxabs_eV": float(np.max(np.abs(r))) if r.size else None,
    }


def sorted_stats(tb: np.ndarray, qe: np.ndarray) -> dict:
    d = tb - qe
    return {
        "rms_eV": float(np.sqrt(np.mean(d * d))),
        "maxabs_eV": float(np.max(np.abs(d))),
        "argmax_index0": int(np.argmax(np.abs(d))),
        "tb_at_argmax_eV": float(tb[np.argmax(np.abs(d))]),
        "qe_at_argmax_eV": float(qe[np.argmax(np.abs(d))]),
        "tb_gap_below_eV": float(tb[np.argmax(np.abs(d))] - tb[np.argmax(np.abs(d)) - 1])
        if np.argmax(np.abs(d)) > 0
        else None,
        "qe_gap_below_eV": float(qe[np.argmax(np.abs(d))] - qe[np.argmax(np.abs(d)) - 1])
        if np.argmax(np.abs(d)) > 0
        else None,
        "tb_gap_above_eV": float(tb[np.argmax(np.abs(d)) + 1] - tb[np.argmax(np.abs(d))])
        if np.argmax(np.abs(d)) + 1 < tb.size
        else None,
        "qe_gap_above_eV": float(qe[np.argmax(np.abs(d)) + 1] - qe[np.argmax(np.abs(d))])
        if np.argmax(np.abs(d)) + 1 < qe.size
        else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nscf", type=Path, required=True)
    ap.add_argument("--tb", type=Path, required=True)
    ap.add_argument("--eta-scan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    qe_bands, _fermi = parse_qe_nscf_bands(args.nscf)
    irvec, ndegen, h_r = read_hamiltonian_blocks(args.tb)
    eta = json.loads(args.eta_scan.read_text(encoding="utf-8"))
    per = eta["k20_eta_scan"]["per_k20"]

    # Unique Gamma shift, same as A1-H compare.
    qe0 = np.sort(qe_bands[0][window_mask(qe_bands[0])])
    tb0 = np.sort(eigvals_at(irvec, ndegen, h_r, KPOINTS[0]["crystal"]).real)
    shift = float(np.mean(tb0 - qe0))

    rows = []
    for i, spec in enumerate(KPOINTS):
        qe = np.sort(qe_bands[i][window_mask(qe_bands[i])]) + shift
        tb = np.sort(eigvals_at(irvec, ndegen, h_r, spec["crystal"]).real)
        assert qe.size == N_WINDOW and tb.size == N_WINDOW
        sl = sorted_stats(tb, qe)
        sl_q = sorted_stats(tb[-N_Q:], qe[-N_Q:])
        hh = hausdorff(tb, qe)
        hh_q = hausdorff(tb[-N_Q:], qe[-N_Q:])
        # eta scan is on k20 grid; match by crystal coords to 1e-9.
        hit = None
        for row in per:
            fk = row["frac_k"]
            if abs(fk[0] - spec["crystal"][0]) < 1e-9 and abs(fk[1] - spec["crystal"][1]) < 1e-9:
                hit = row
                break
        eta_ctx = None
        if hit is not None:
            dE = float(hit["gap_pi_q_eV"])
            eta_ctx = {
                "gap_pi_q_eV": dE,
                "eta_r_max_baseline": hit["eta"]["baseline_2e11"]["eta_r_max"],
                "wannier_k_index_1based": hit["wannier_k_index_1based"],
                "sorted_maxabs_over_pi_q_gap": (sl["maxabs_eV"] / dE) if dE else None,
                "hausdorff_over_pi_q_gap": (hh["symmetric_eV"] / dE) if dE else None,
                "sorted_maxabs_over_hbar_omega": sl["maxabs_eV"] / HBAR_OMEGA_EV,
                "hausdorff_over_hbar_omega": hh["symmetric_eV"] / HBAR_OMEGA_EV,
            }
        rows.append(
            {
                **spec,
                "crystal": list(spec["crystal"]),
                "sorted": sl,
                "sorted_q32": sl_q,
                "hausdorff": hh,
                "hausdorff_q32": hh_q,
                "cluster_20meV": cluster_l2(tb, qe, 0.02),
                "cluster_50meV": cluster_l2(tb, qe, 0.05),
                "eta_scan_at_this_k20": eta_ctx,
                "n_qe_spacings_lt_20meV": int(np.sum(consecutive_gaps(qe) < 0.02)),
                "n_tb_spacings_lt_20meV": int(np.sum(consecutive_gaps(tb) < 0.02)),
            }
        )

    payload = {
        "schema": "a1h_spectrum_matching_v1",
        "job": 32293,
        "hbar_omega0_eV": HBAR_OMEGA_EV,
        "global_shift_from_gamma_eV": shift,
        "note": (
            "Hausdorff = max_i min_j |E_i-E_j|. If Hausdorff << sorted L∞, "
            "the large sorted residual is a cluster-filling / order-statistics "
            "effect, not a missing isolated level. True QE–Wannier subspace "
            "overlap at off-mesh k is NOT computed (needs PW overlaps)."
        ),
        "kpoints": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['label']:6s} sorted_max={row['sorted']['maxabs_eV']:.4f}  "
            f"Haus={row['hausdorff']['symmetric_eV']:.4f}  "
            f"q32_sorted={row['sorted_q32']['maxabs_eV']:.4f}  "
            f"q32_Haus={row['hausdorff_q32']['symmetric_eV']:.4f}  "
            f"cl20_unmatched={row['cluster_20meV']['unmatched_tb']}+{row['cluster_20meV']['unmatched_qe']}  "
            f"PiQgap={None if row['eta_scan_at_this_k20'] is None else round(row['eta_scan_at_this_k20']['gap_pi_q_eV'], 3)}",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
