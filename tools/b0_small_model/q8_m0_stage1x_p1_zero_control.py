#!/usr/bin/env python3
"""P1=0 dressing control for the approved p-h/photon Gaussian model.

The joint Hamiltonian is switched on at t=0 with the electron in the bare
ground state and photons in the bare vacuum. If the coupling itself already
generates large p-h occupation without any classical drive, the large numbers
seen in the P1 runs are partly/wholly a dressing quench rather than P1-driven
quantum fluctuations.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import propagate_cjj  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_mode_io import eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_phgauss import pauli_from_phi, propagate_phi  # noqa: E402
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1s_continuum import open_grid  # noqa: E402
from q8_m0_stage1t_continuum_pauli import continuum_waveforms  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1x_p1_zero_control_20260923"
MU = -0.009010620372198276
K_FRAC = np.array([[0.0, 0.0], [0.17, 0.41]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntpc", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--nquad", type=int, default=96)
    args = parser.parse_args()

    geo = build_geometry()
    prot0 = p1_a2_protocol(0.0)  # zero classical drive
    e_par, e_perp = parallel_perp_basis(geo)
    eta = float(np.sqrt(eta2_ncell(geo)))
    omega, weights = open_grid(8.0 * prot0.omega0, args.nquad)

    rows = []
    target = OUT / "stage1x_p1_zero_control.json"
    if target.exists():
        rows = json.loads(target.read_text(encoding="utf-8"))["rows"]

    for ntpc in args.ntpc:
        t0 = time.perf_counter()
        rec = propagate_cjj(
            geo, R0, prot0, nt_per_cycle=ntpc, neel=+1, sign=+1, mu=MU,
            verbose=False, kf_frac=K_FRAC, save_full_x=True, save_mean_D=True,
        )
        waves = continuum_waveforms(omega, weights, rec["times"], e_par, e_perp)
        ev = propagate_phi(
            rec["X_full"], rec["w_k"], rec["mean_D_t"], waves,
            float(rec["dt"]), eta, save_pauli=True,
        )
        p = pauli_from_phi(ev["Phi"], ev["n_ph"], ev["n_B"], rec["w_k"])
        row = {
            "ntpc": ntpc,
            "n_t": int(len(rec["times"])),
            "nquad": args.nquad,
            "omega_max_over_omega0": float(omega[-1] / prot0.omega0),
            "C_P_final": p["Ncell_times_eps_Pauli"],
            "C_C_final": p["Ncell_times_C_minus_I"],
            "N_B_max": float(np.max(ev["N_B_max_t"])),
            "Phi_maxabs": float(np.max(np.abs(ev["Phi"]))),
            "symplectic_abs": ev["symplectic_maxabs"],
            "symplectic_scaled_by_maxPhi_squared": float(
                ev["symplectic_maxabs"] / max(1.0, float(np.max(np.abs(ev["Phi"])))) ** 2
            ),
            "wall_seconds": time.perf_counter() - t0,
        }
        rows = [old for old in rows if not (old["ntpc"] == ntpc and old["nquad"] == args.nquad)] + [row]
        rows.sort(key=lambda r: (r["nquad"], r["ntpc"]))
        print(f"P1=0 ntpc={ntpc} CP={row['C_P_final']:.6g} "
              f"CC={row['C_C_final']:.6g} NBmax={row['N_B_max']:.6g} "
              f"wall={row['wall_seconds']:.1f}s", flush=True)

    payload = {
        "status": "p1_zero_dressing_control",
        "model": "same_P1_dressed_photon_Gaussian_two_k",
        "classical_drive": "P1_A0=0",
        "photon_frequency": f"positive_open_Gauss_{args.nquad}_0_to_8omega0",
        "photon_polarization": "both_inplane",
        "both_ports": True,
        "N_cell": "unset",
        "S": "unset",
        "rows": rows,
        "interpretation": "If CP/CC are already large here, part of the P1-run "
                          "occupation is a dressing quench, not P1-driven gain.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
