#!/usr/bin/env python3
"""Stage-1R: G_self emergence from two-port photon continuum.

Prescribed I_cell. No hand-made G_self. No BZ. No production V.
No continuum Pauli yet. Diagnostic N_cell only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_gself_emerge import (  # noqa: E402
    GSELF_EMERGE_ID,
    emerge_on_basis,
    long_window_bins,
)
from q8_m0_photon_op import h2_phi  # noqa: E402
from q8_m0_model import build_geometry  # noqa: E402
from q8_m0_phgauss import PAULI_C_C, PAULI_C_P, pauli_area_curve  # noqa: E402
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1r_gself_20260923"
NTPC = 32
SCALES = (1, 2, 4, 8)
NCELL_DIAG = (1.0, 4.0, 16.0, 64.0)
G1R_ID = "gself_emergence_continuum_20260923"


def log(msg: str) -> None:
    print(f"[stage1r] {msg}", flush=True)


def slim_cmp(rec: dict) -> dict:
    c = rec["compare"]
    return {
        "n_packets": rec["n_packets"],
        "n_ph": rec["n_ph"],
        "N_cell_diag": rec["split"]["N_cell_diag"],
        "eta_vac": rec["split"]["eta_vac"],
        "eta_drive": rec["split"]["eta_drive"],
        "gamma_target": rec["gamma_target"],
        "rel_maxabs_A": c["rel_maxabs_A"],
        "l2_rel_A": c["l2_rel_A"],
        "A0": c["A0"],
        "maxabs_A_minus_Maxwell": c["maxabs_A_minus_Maxwell"],
        "maxabs_dA_minus_gamma_I_past": c["maxabs_dA_minus_gamma_I_past"],
    }


def _plot(scales, rels, ncell, rel_n, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax[0].plot(scales, rels, "o-")
    ax[0].plot(scales, [1.0 / s for s in scales], "k--", label="Tp/T_obs")
    ax[0].set_xlabel("T_obs / Tp")
    ax[0].set_ylabel("rel max |A-A_Maxwell| on [0,Tp]")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].semilogx(ncell, rel_n, "s-")
    ax[1].set_xlabel("diagnostic N_cell")
    ax[1].set_ylabel("rel max |A-A_Maxwell|")
    ax[1].set_ylim(0.0, max(0.2, 1.2 * max(rel_n)))
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    e_par, e_perp = parallel_perp_basis(geo)
    t0 = time.perf_counter()

    ladder = []
    rels = []
    for scale in SCALES:
        pack = long_window_bins(prot, e_perp, t_obs_over_tp=scale, ntpc=NTPC, harm_max=8.0)
        rec = emerge_on_basis(
            pack["phis"], pack["omega"], pack["times"], pack["dw"],
            pack["i_cell"], pack["dt"], geo, 1.0, n_pulse=pack["n_pulse"],
        )
        row = slim_cmp(rec)
        row["t_obs_over_tp"] = scale
        row["n_omega"] = pack["n_omega"]
        ladder.append(row)
        rels.append(row["rel_maxabs_A"])
        log(f"scale={scale} n_om={pack['n_omega']} relA={row['rel_maxabs_A']:.4f} A0={row['A0']:.2e}")

    pack8 = long_window_bins(prot, e_perp, t_obs_over_tp=8.0, ntpc=NTPC, harm_max=8.0)
    h2 = [h2_phi(pack8["omega"], prot.omega0, prot.t_pulse, e_perp)]
    rec_h2 = emerge_on_basis(
        h2, pack8["omega"], pack8["times"], pack8["dw"],
        pack8["i_cell"], pack8["dt"], geo, 1.0, n_pulse=pack8["n_pulse"],
    )
    h2s = slim_cmp(rec_h2)
    log(f"H2-only n_ph={h2s['n_ph']} relA={h2s['rel_maxabs_A']:.4f}")

    ncell_rows = []
    rel_n = []
    a_refs = []
    for n in NCELL_DIAG:
        rec = emerge_on_basis(
            pack8["phis"], pack8["omega"], pack8["times"], pack8["dw"],
            pack8["i_cell"], pack8["dt"], geo, n, n_pulse=pack8["n_pulse"],
        )
        row = slim_cmp(rec)
        ncell_rows.append(row)
        rel_n.append(row["rel_maxabs_A"])
        a_refs.append(float(np.max(np.abs(rec["A_ph"][: pack8["n_pulse"]]))))
        log(
            f"Ndiag={n:g} eta_vac={row['eta_vac']:.5g} eta_drive={row['eta_drive']:.5g} "
            f"relA={row['rel_maxabs_A']:.6f}"
        )
    ncell_spread = float(np.max(rel_n) - np.min(rel_n))
    a_spread = float(np.max(a_refs) - np.min(a_refs))

    _plot(list(SCALES), rels, list(NCELL_DIAG), rel_n, OUT / "stage1r_gself.png")
    curve = pauli_area_curve(float(geo.a_cell_ang2))
    best = ladder[-1]
    kernel_indep = bool(ncell_spread < 1e-10 and a_spread < 1e-10)
    ir_improves = bool(rels[-1] < rels[0] * 0.3)
    payload = {
        "g1r_id": G1R_ID,
        "emerge_id": GSELF_EMERGE_ID,
        "spec": stage1_spec(),
        "frozen": {
            "no_hand_G_self": True,
            "A_self_t0": 0,
            "not_bare_1_over_omega": True,
            "S_not_fixed": True,
            "N_cell_diag_only": True,
            "no_BZ": True,
            "no_production_V": True,
            "no_continuum_Pauli_yet": True,
        },
        "pauli_area_curve_prototype": curve,
        "pauli_C_P": PAULI_C_P,
        "pauli_C_C": PAULI_C_C,
        "ladder": ladder,
        "h2_only": h2s,
        "ncell_diag": ncell_rows,
        "ncell_rel_spread": ncell_spread,
        "ncell_A_spread": a_spread,
        "kernel_independent_of_N_cell_diag": kernel_indep,
        "IR_error_tracks_Tp_over_Tobs": ir_improves,
        "best_on_pulse_window": best,
        "verdict": {
            "two_H2_is_not_continuum": bool(h2s["rel_maxabs_A"] > 0.5),
            "finite_positive_freq_basis_approaches_Maxwell": ir_improves,
            "Maxwell_kernel_independent_of_diagnostic_N_cell": kernel_indep,
            "hand_G_self_used": False,
            "continuum_Pauli": "not_this_round",
        },
        "human_gate": False,
        "next": "continuum_Pauli_area_curve_then_BZ",
        "wall_s": time.perf_counter() - t0,
    }
    dump(OUT / "stage1r_gself.json", payload)
    print(json.dumps({
        "wrote": str(OUT / "stage1r_gself.json"),
        "ladder_relA": rels,
        "h2_relA": h2s["rel_maxabs_A"],
        "ncell_relA": rel_n,
        "kernel_indep": kernel_indep,
        "wall_s": payload["wall_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
