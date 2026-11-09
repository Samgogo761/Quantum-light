#!/usr/bin/env python3
"""Near-gap Cr-d / I-p weights from the existing full112 tb.dat.

Not new DFT, not a downfold, not an 8-band model change.
Wannier90 convention: H(k)=Σ_R e^{i k_cart·R_cart} H(R)/n_R (no τ in the phase).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from q8_tperp_static_audit import SCF, TB, parse_scf

OUT = Path(__file__).resolve().parents[2] / "results/q_line/ip_band_character_20260910"
N_OCC = 84  # 4 Cr3+ d3 + 12 I- p6, spinor bands
NEAR = 8  # bands on each side of the gap to tabulate


def wf_tag(i: int) -> str:
    if i < 40:
        return "Cr_d"
    return ("I_pz", "I_px", "I_py")[((i - 40) % 6) // 2]


def stream_hk(path: Path, k_carts: list[np.ndarray], lat: np.ndarray) -> list[np.ndarray]:
    mats = [None, None]
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for _ in range(3):
            fh.readline()
        nw = int(fh.readline().split()[0])
        nr = int(fh.readline().split()[0])
        ndegen: list[int] = []
        while len(ndegen) < nr:
            ndegen.extend(int(x) for x in fh.readline().split())
        mats = [np.zeros((nw, nw), dtype=complex) for _ in k_carts]
        for ir in range(nr):
            line = fh.readline()
            while line and not line.strip():
                line = fh.readline()
            rfrac = np.array([int(x) for x in line.split()[:3]], dtype=float)
            rcart = rfrac @ lat
            deg = float(ndegen[ir])
            phases = [np.exp(1j * float(np.dot(kc, rcart))) / deg for kc in k_carts]
            for _ in range(nw * nw):
                p = fh.readline().split()
                i, j = int(p[0]) - 1, int(p[1]) - 1
                amp = (float(p[2]) + 1j * float(p[3]))
                for ik, ph in enumerate(phases):
                    mats[ik][i, j] += amp * ph
    return mats


def character(vec: np.ndarray) -> dict:
    w = {"Cr_d": 0.0, "I_pz": 0.0, "I_px": 0.0, "I_py": 0.0}
    for i, c in enumerate(vec):
        w[wf_tag(i)] += float(abs(c) ** 2)
    w["I_p"] = w["I_pz"] + w["I_px"] + w["I_py"]
    return w


def summarize_k(name: str, hk: np.ndarray) -> dict:
    evals, evecs = np.linalg.eigh(hk)
    ev = evals.real
    vbm = ev[N_OCC - 1]
    cbm = ev[N_OCC]
    rows = []
    lo = max(0, N_OCC - NEAR)
    hi = min(hk.shape[0], N_OCC + NEAR)
    for n in range(lo, hi):
        w = character(evecs[:, n])
        rows.append(
            {
                "band_1based": n + 1,
                "role": "VBM_side" if n < N_OCC else "CBM_side",
                "E_minus_midgap_eV": float(ev[n] - 0.5 * (vbm + cbm)),
                **w,
            }
        )
    def _avg(mask):
        sel = [r for r in rows if mask(r)]
        if not sel:
            return None
        keys = ("Cr_d", "I_p", "I_pz", "I_px", "I_py")
        return {k: float(np.mean([r[k] for r in sel])) for k in keys}

    return {
        "k": name,
        "gap_eV": float(cbm - vbm),
        "VBM_eV": float(vbm),
        "CBM_eV": float(cbm),
        "bands": rows,
        "mean_top8_valence": _avg(lambda r: r["role"] == "VBM_side"),
        "mean_bottom8_conduction": _avg(lambda r: r["role"] == "CBM_side"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    documented_qe = {
        "source": "docs/08_QE与Wannier/CrI3_40WF审计第一轮_20260714.md §4; March projwfc 160 bands × 666 k, Cr-d only",
        "csv_in_repo": False,
        "Cr_d_projectability_k_mean": {
            "VBM_133_140": [0.035, 0.056],
            "mid_valence_121_132": [0.06, 0.075],
            "deep_valence_103_120": [0.10, 0.18],
            "first_CB_141_148": [0.40, 0.47],
            "second_CB_149_160": [0.81, 0.86],
        },
        "note": "QE table is Cr-d projectability, so I-p dominance on VBM is 1-pd, not a resolved px/py/pz QE fat-band.",
    }
    documented_models = {
        "d_p_hopping_share": {"source": "CrI3_PhaseB_hopping排名与M52草案_20260715.md", "d-d": 0.0046, "d-p": 0.671, "p-p": 0.324},
        "intra_dp_representability": {"M88_pxpy": 0.81, "M64_pz": 0.30, "M52_interface_pz": 0.16, "d_only": 0.06},
        "HHG_vs_full112": {
            "source": "docs/09_少带模型/给场外AI_S2截断与E0诊断全量报告_20260720.md",
            "M88_full_H1": 0.994,
            "M88_full_H5": 1.012,
            "M52_full_H1": 0.630,
            "M52_c6_H5_over_parent": 16.27,
            "d_only40_H1_over_nb104": 0.344,
        },
    }
    payload = {
        "not_a_downfold": True,
        "not_an_8band_decision": True,
        "N_occ_spinor": N_OCC,
        "documented_qe_projwfc": documented_qe,
        "documented_reduced_models": documented_models,
        "wannier_interpolation": None,
    }
    if TB.is_file():
        lat, _atoms = parse_scf(SCF)
        recip = 2.0 * np.pi * np.linalg.inv(lat).T
        fracs = {"Gamma": [0.0, 0.0, 0.0], "M": [0.5, 0.0, 0.0], "K": [1.0 / 3.0, 1.0 / 3.0, 0.0]}
        names = list(fracs)
        kcarts = [np.array(fracs[n], dtype=float) @ recip for n in names]
        hks = stream_hk(TB, kcarts, lat)
        payload["wannier_interpolation"] = {
            "tb": str(TB),
            "convention": "Wannier90 exp(ik·R) H(R)/n_R; weights are WF |c|^2, Ω_D~0.005 so WF≈initial projections",
            "k": [summarize_k(n, hk) for n, hk in zip(names, hks)],
        }
    (OUT / "near_gap_character.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("OUT", OUT)
    if payload["wannier_interpolation"]:
        for rec in payload["wannier_interpolation"]["k"]:
            print(rec["k"], "gap", rec["gap_eV"], "val", rec["mean_top8_valence"], "cond", rec["mean_bottom8_conduction"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
