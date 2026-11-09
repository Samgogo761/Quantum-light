#!/usr/bin/env python3
"""A1-O: WINDOW144 r(k) vs independent QE MMN/Berry r at eta, H5, H7.

Isolated bands: |r_mn|. Near-degenerate groups: Pi-Q block Frobenius / SVD.
Pi-Pi occupied block is a method control. Do not announce an HHG mechanism.
A1-O numbers do not promote WINDOW144 to a production parent.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1o_berry_math import (
    berry_from_pm_crystal,
    energy_pi_q,
    frobenius_block,
    interpolate_tb_hr,
    isolated_pair_abs,
    phase_align_columns,
    tb_weight_pi_q,
    window_indices,
)
from audit_pi_rq_coupling import load_eig, parse_nnkp, read_mmn_for_k
from construct_p_conjugate_tb import read_tb_blocks

WIN_MIN = -6.0
WIN_MAX = 8.0
N_WINDOW = 144
N_Q = 32
N_PI = 112
N_OCC = 84
ISO_GAP_EV = 0.050
SCHEMA = "a1o_window144_vs_qe_berry_v1"


def rel_err(a: float, b: float) -> float | None:
    den = max(abs(a), abs(b), 1.0e-16)
    return float(abs(a - b) / den)


def slice_mmn(mat: np.ndarray, idx_k: np.ndarray, idx_kb: np.ndarray) -> np.ndarray:
    return mat[np.ix_(idx_k, idx_kb)]


def berry_at_delta(
    mats: list[np.ndarray],
    headers: list[tuple],
    neighbors: list[dict],
    delta: float,
    eig_all: np.ndarray,
    ik_1based: int,
    kpoints: np.ndarray,
    recip: np.ndarray,
    kfrac: np.ndarray,
) -> dict:
    sel = [nb for nb in neighbors if abs(nb["delta"] - delta) < 1e-12]
    if len(sel) != 4:
        raise RuntimeError(f"expected 4 neighbors at delta={delta}, got {len(sel)}")
    idx_k = window_indices(eig_all[ik_1based - 1], WIN_MIN, WIN_MAX)
    aligned: dict[str, np.ndarray] = {}
    nwin_sat = []
    for nb in sel:
        slot = int(nb["nn_slot"])
        mat = mats[slot]
        _ik, jk, _g1, _g2, _g3 = headers[slot]
        if jk != nb["k_index_1based"]:
            raise RuntimeError(f"mmn neighbor {jk} != map {nb['k_index_1based']}")
        idx_b = window_indices(eig_all[jk - 1], WIN_MIN, WIN_MAX)
        nwin_sat.append(int(idx_b.size))
        if idx_k.size != N_WINDOW or idx_b.size != N_WINDOW:
            return {
                "ok": False,
                "delta": delta,
                "n_window_k": int(idx_k.size),
                "n_window_satellites": nwin_sat,
                "reason": "window_not_144",
            }
        aligned[nb["dir"]] = phase_align_columns(slice_mmn(mat, idx_k, idx_b))
    r_ang = berry_from_pm_crystal(
        aligned["+x"],
        aligned["-x"],
        aligned["+y"],
        aligned["-y"],
        delta,
        recip,
    )
    eig_w = eig_all[ik_1based - 1][idx_k]
    return {
        "ok": True,
        "delta": delta,
        "n_window_k": int(idx_k.size),
        "n_window_satellites": nwin_sat,
        "r_ang": r_ang,
        "eig_window": eig_w,
        "idx_k": idx_k,
    }


def pack_metrics(r_ang: np.ndarray, eig_w: np.ndarray, u_tb: np.ndarray | None) -> dict:
    pi_e, q_e = energy_pi_q(eig_w.size, N_Q)
    occ = np.arange(min(N_OCC, eig_w.size))
    out = {
        "energy_piq": frobenius_block(r_ang, pi_e, q_e),
        "energy_pipi_occ": frobenius_block(r_ang, occ, occ),
        "energy_pipi_occ_offdiag": frobenius_block(r_ang, occ, occ, zero_diag=True),
        "isolated_piq_abs_rx_top": isolated_pair_abs(r_ang[0], eig_w, pi_e, q_e, ISO_GAP_EV),
    }
    if u_tb is not None:
        pi_w, q_w = tb_weight_pi_q(u_tb, N_PI, N_Q)
        out["tb_weight_piq_index_counts"] = {"n_pi": int(pi_w.size), "n_q": int(q_w.size)}
        if pi_w.size and q_w.size:
            out["tb_weight_piq"] = frobenius_block(r_ang, pi_w, q_w)
    return out


def compare_side(qe: dict, tb: dict, key: str) -> dict:
    if key not in qe or key not in tb:
        return {}
    a, b = qe[key], tb[key]
    out = {}
    for ax in ("x", "y"):
        qf, tf = a[f"F_{ax}"], b[f"F_{ax}"]
        out[ax] = {
            "qe_F": qf,
            "tb_F": tf,
            "rel_err": rel_err(qf, tf),
            "ratio_tb_over_qe": float(tf / qf) if abs(qf) > 1e-16 else None,
            "qe_svd1": a[f"svd1_{ax}"],
            "tb_svd1": b[f"svd1_{ax}"],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mmn", type=Path, required=True)
    ap.add_argument("--eig", type=Path, required=True)
    ap.add_argument("--nnkp", type=Path, required=True)
    ap.add_argument("--star-map", type=Path, required=True)
    ap.add_argument("--tb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    star = json.loads(args.star_map.read_text(encoding="utf-8"))
    nnkp = parse_nnkp(args.nnkp)
    eig = load_eig(args.eig)
    print(f"eig shape={eig.shape} mmn={args.mmn} reading WINDOW144 TB...", flush=True)
    irvec, ndegen, h_r, r_r = read_tb_blocks(args.tb, read_position=True)
    if r_r is None:
        raise RuntimeError("WINDOW144 tb has no position block")
    print(f"TB nw={h_r.shape[1]} nrpts={h_r.shape[0]} in {time.time() - t0:.1f}s", flush=True)

    recip = nnkp["recip_lattice"]
    kpoints = nnkp["kpoints"]
    centers_json = []
    headline_rel = []

    for rec in star["centers"]:
        ik = int(rec["k_index_1based"])
        kfrac = np.array(rec["crystal"], dtype=float)
        nb, nk, nntot, headers, mats = read_mmn_for_k(args.mmn, ik)
        if nntot != star["nnb"]:
            raise RuntimeError(f"nntot {nntot} != map {star['nnb']}")
        evals_tb, r_tb, u_tb = interpolate_tb_hr(irvec, ndegen, h_r, r_r, kfrac)
        if evals_tb.size != N_WINDOW:
            raise RuntimeError(f"TB dim {evals_tb.size} != {N_WINDOW}")
        tb_metrics = pack_metrics(r_tb, evals_tb, u_tb)
        per_delta = []
        for delta in star["deltas_crystal"]:
            got = berry_at_delta(
                mats,
                headers,
                rec["neighbors"],
                float(delta),
                eig,
                ik,
                kpoints,
                recip,
                kfrac,
            )
            if not got["ok"]:
                per_delta.append({k: v for k, v in got.items() if k != "r_ang"})
                continue
            qe_metrics = pack_metrics(got["r_ang"], got["eig_window"], None)
            row = {
                "delta": float(delta),
                "n_window_k": got["n_window_k"],
                "n_window_satellites": got["n_window_satellites"],
                "qe": {k: v for k, v in qe_metrics.items() if k != "isolated_piq_abs_rx_top"},
                "tb": {k: v for k, v in tb_metrics.items() if k != "isolated_piq_abs_rx_top"},
                "compare_energy_piq": compare_side(qe_metrics, tb_metrics, "energy_piq"),
                "compare_pipi_occ": compare_side(qe_metrics, tb_metrics, "energy_pipi_occ"),
                "compare_pipi_occ_offdiag": compare_side(
                    qe_metrics, tb_metrics, "energy_pipi_occ_offdiag"
                ),
                "qe_isolated_top": qe_metrics["isolated_piq_abs_rx_top"],
                "tb_isolated_top": tb_metrics["isolated_piq_abs_rx_top"],
            }
            fx = row["compare_energy_piq"].get("x", {})
            if fx.get("rel_err") is not None:
                headline_rel.append(float(fx["rel_err"]))
            per_delta.append(row)
        conv = {}
        fs = [
            d["compare_energy_piq"]["x"]["qe_F"]
            for d in per_delta
            if d.get("compare_energy_piq", {}).get("x", {}).get("qe_F") is not None
        ]
        if len(fs) >= 2:
            conv = {
                "qe_piq_Fx_by_delta": fs,
                "qe_piq_Fx_rel_spread": rel_err(max(fs), min(fs)),
            }
        centers_json.append(
            {
                "label": rec["label"],
                "role": rec["role"],
                "crystal": rec["crystal"],
                "k_index_1based": ik,
                "n_window_center": int(window_indices(eig[ik - 1], WIN_MIN, WIN_MAX).size),
                "per_delta": per_delta,
                "fd_convergence": conv,
            }
        )

    max_rel = max(headline_rel) if headline_rel else None
    payload = {
        "schema": SCHEMA,
        "A1O_CLOSES": "HHG_OPTICAL_INTERP_ONLY",
        "A1H_PASS_NE_HHG_TRUSTED": True,
        "NOT": [
            "WINDOW144 production parent",
            "HHG mechanism attribution",
            "TB dH/dk as QE truth",
            "GH7",
            "k40",
            "chemical I-d 144",
        ],
        "mmn": str(args.mmn),
        "eig": str(args.eig),
        "tb": str(args.tb),
        "win_eV": [WIN_MIN, WIN_MAX],
        "iso_gap_eV": ISO_GAP_EV,
        "primary_metric": "energy_ordered_PiQ_Frobenius_rx",
        "centers": centers_json,
        "max_energy_piq_Fx_rel_err": max_rel,
        "human_gate": (
            "A1-O asks whether WINDOW144 r interpolation error is already large "
            "enough to strip 31678 of quantitative meaning. Static Pi-Q Frobenius "
            "does not by itself assign a fraction of the 4.76 H5 ratio. HHG is "
            "nonlinear, time-dependent, and coherent. Gamma Pi-Pi control must be "
            "finite; a ~0 QE r with finite TB r is a reconstruction bug, not physics. "
            "Do not announce a mechanism."
        ),
        "elapsed_s": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "max_energy_piq_Fx_rel_err": max_rel,
                "elapsed_s": payload["elapsed_s"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
