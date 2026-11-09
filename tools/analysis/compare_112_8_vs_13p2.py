#!/usr/bin/env python3
"""Compare 112_[-6,8] with the finished 112_[-6,13.2] window control.

232_I-d did not finish (job 33307 TIMEOUT during disentanglement) and is not
a parent in this comparison. H5/H7 are not a PASS/FAIL gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_112old_large_qe as C
from a1o_berry_math import frobenius_block, isolated_pair_abs
from audit_wannier_chk_subspace import index_checkpoint
from construct_p_conjugate_tb import read_tb_blocks
from count_outer_window_bands import parse_qe_nscf_kblocks, select_kblocks

SCHEMA = "compare_112_8_vs_13p2_v1"
EXTRA = ("above8", 8.0, 13.2)
C.WINDOWS = tuple(C.WINDOWS) + (EXTRA,)

NOTES = (
    "112_13.2 reuses the 112_large AMN/MMN/EIG/nnkp. Only dis_win_max changes 8.0 -> 13.2.",
    "232_I-d job 33307 timed out in disentanglement and has no chk or _tb.dat.",
    "I-d necessity is not decided here.",
    "QE reference is the nbnd=400 8-k NSCF (job 32911), not A1-H nbnd=200.",
    "H5/H7 HHG was not run.",
)


def r_report_with_above8(evals: np.ndarray, r_cart: np.ndarray, energy_shift: float) -> dict:
    report = C.r_report(evals, r_cart, energy_shift)
    e_win = evals + energy_shift
    left = C.occ_indices(evals)["nv84"]
    name, lo, hi = EXTRA
    right = np.flatnonzero(C.in_window(e_win, lo, hi))
    if right.size == 0 or left.size == 0:
        report["blocks"][name] = {"n_right": int(right.size), "empty": True}
        return report
    fb = frobenius_block(r_cart, left, right, zero_diag=False)
    report["blocks"][name] = {
        "empty": False,
        "n_right": int(right.size),
        "lo_eV": lo,
        "hi_eV": hi,
        "frobenius": {k: float(v) for k, v in fb.items()},
        "top_isolated_|rx|": isolated_pair_abs(r_cart[0], e_win, left, right, C.ISO_GAP_EV, max_pairs=8),
        "top_isolated_|ry|": isolated_pair_abs(r_cart[1], e_win, left, right, C.ISO_GAP_EV, max_pairs=8),
    }
    return report


def p_windows_one(index, eig: np.ndarray) -> dict:
    if eig.shape[0] != index.num_kpts:
        raise RuntimeError(f"eig nk={eig.shape[0]} != chk nk={index.num_kpts}")
    if eig.shape[1] < index.num_bands:
        raise RuntimeError(f"eig nb={eig.shape[1]} < chk nb={index.num_bands}")
    eig = eig[:, : index.num_bands]
    udis = C.open_udis(index)
    acc = {name: [] for name, _lo, _hi in C.WINDOWS}
    try:
        for ik in range(index.num_kpts):
            pdiag = C.pdiag_at_k(index, udis, ik)
            e = eig[ik]
            for name, lo, hi in C.WINDOWS:
                mask = C.in_window(e, lo, hi)
                if np.any(mask):
                    acc[name].append(pdiag[mask])
    finally:
        del udis
    return {name: C.p_stats(np.concatenate(chunks) if chunks else np.array([])) for name, chunks in acc.items()}


def build(args: argparse.Namespace) -> dict:
    t0 = time.time()
    nscf_text = Path(args.nscf).read_text(encoding="utf-8", errors="replace")
    blocks = select_kblocks(parse_qe_nscf_kblocks(nscf_text), C.N_K_COUNT)
    if any(int(b["e_eV"].size) != 400 for b in blocks):
        raise RuntimeError("QE 8-k reference must be nbnd=400")

    print(f"Reading w8 {args.tb_w8}", flush=True)
    ir8, nd8, h8, r8 = read_tb_blocks(Path(args.tb_w8), read_position=True)
    print(f"Reading w132 {args.tb_w132}", flush=True)
    ir13, nd13, h13, r13 = read_tb_blocks(Path(args.tb_w132), read_position=True)
    if h8.shape[1] != 112 or h13.shape[1] != 112:
        raise RuntimeError(f"expected 112, got {h8.shape[1]} and {h13.shape[1]}")

    per_k = []
    evals8 = []
    evals13 = []
    r8k = []
    r13k = []
    for spec in C.COUNT_K:
        e8, rr8 = C.tb_at_k(ir8, nd8, h8, r8, spec["crystal"])
        e13, rr13 = C.tb_at_k(ir13, nd13, h13, r13, spec["crystal"])
        evals8.append(e8)
        evals13.append(e13)
        r8k.append(rr8)
        r13k.append(rr13)
        print(f"  interpolated {spec['label']}", flush=True)

    qe_gamma = blocks[0]["e_eV"]
    shift8 = C.unique_gamma_shift(evals8[0], qe_gamma, -1.5, 0.8)
    shift13 = C.unique_gamma_shift(evals13[0], qe_gamma, -1.5, 0.8)
    s8 = shift8["shift_eV"] if shift8["applied"] else 0.0
    s13 = shift13["shift_eV"] if shift13["applied"] else 0.0

    for i, spec in enumerate(C.COUNT_K):
        row = {
            "label": spec["label"],
            "crystal": list(spec["crystal"]),
            "w8": {"spectrum": C.window_spectrum(evals8[i], blocks[i]["e_eV"], s8)},
            "w132": {"spectrum": C.window_spectrum(evals13[i], blocks[i]["e_eV"], s13)},
        }
        if spec["optics"] and r8k[i] is not None and r13k[i] is not None:
            row["w8"]["r_mn"] = r_report_with_above8(evals8[i], r8k[i], s8)
            row["w132"]["r_mn"] = r_report_with_above8(evals13[i], r13k[i], s13)
        tb8 = np.sort(evals8[i] + s8)
        tb13 = np.sort(evals13[i] + s13)
        row["tb_coverage_eV"] = {
            "w8": [float(tb8.min()), float(tb8.max())],
            "w132": [float(tb13.min()), float(tb13.max())],
        }
        per_k.append(row)

    del h8, r8, h13, r13

    print("Indexing checkpoints", flush=True)
    idx8 = index_checkpoint(Path(args.chk_w8))
    idx13 = index_checkpoint(Path(args.chk_w132))
    for idx, tag in ((idx8, "w8"), (idx13, "w132")):
        if idx.num_wann != 112 or idx.num_bands != 400:
            raise RuntimeError(f"{tag} chk num_wann={idx.num_wann} num_bands={idx.num_bands}")
    eig = C.load_eig(Path(args.eig))
    print("Mesh projectability w8", flush=True)
    p8 = p_windows_one(idx8, eig)
    print("Mesh projectability w132", flush=True)
    p13 = p_windows_one(idx13, eig)

    return {
        "schema": SCHEMA,
        "status": "DESCRIPTIVE_COMPARISON_NO_PASS_FAIL",
        "notes": list(NOTES),
        "question": "Does opening the same 112 d+p parent from [-6,8] to [-6,13.2] pull in the high conduction bands?",
        "not_id_test": True,
        "gamma_shift": {"w8": shift8, "w132": shift13},
        "tb": {
            "w8": {"nwann": 112, "nrpts": int(nd8.size), "path": str(args.tb_w8)},
            "w132": {"nwann": 112, "nrpts": int(nd13.size), "path": str(args.tb_w132)},
        },
        "checkpoint": {
            "w8_omega_invariant": idx8.omega_invariant,
            "w132_omega_invariant": idx13.omega_invariant,
        },
        "qe_8k": {"nscf": str(args.nscf), "nbnd": 400, "source": "job_32911"},
        "mesh_mean_P": {"w8": p8, "w132": p13},
        "per_k": per_k,
        "timing_s": float(time.time() - t0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nscf", type=Path, required=True)
    ap.add_argument("--tb-w8", type=Path, required=True)
    ap.add_argument("--tb-w132", type=Path, required=True)
    ap.add_argument("--chk-w8", type=Path, required=True)
    ap.add_argument("--chk-w132", type=Path, required=True)
    ap.add_argument("--eig", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    payload = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
