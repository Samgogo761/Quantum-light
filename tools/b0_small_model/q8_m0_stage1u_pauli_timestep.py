#!/usr/bin/env python3
"""Finite time-step stress check of one polarization of the photon continuum.

The 96-node positive quadrature already passes the prescribed-current Maxwell
kernel check.  This script checks whether its coupled p-h/photon occupation is
time-step stable.  It does not select a physical mode area or qualify output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Many small dense solves are faster and more reproducible without excessive
# BLAS threading.  Set this before NumPy loads its numerical libraries.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
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
OUT = REPO / "results/q_line/stage1u_pauli_timestep_20260923"
K_FRAC = np.array([[0.0, 0.0], [0.17, 0.41]])
MU = -0.009010620372198276


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntpc", type=int, nargs="+", default=[32, 64, 128])
    args = parser.parse_args()
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    _, e_perp = parallel_perp_basis(geo)
    eta = float(np.sqrt(eta2_ncell(geo)))
    omega, weights = open_grid(8.0 * prot.omega0, 96)
    target = OUT / "stage1u_pauli_timestep.json"
    rows = json.loads(target.read_text(encoding="utf-8"))["rows"] if target.exists() else []
    for ntpc in args.ntpc:
        rec = propagate_cjj(
            geo, R0, prot, nt_per_cycle=ntpc, neel=+1, sign=+1, mu=MU,
            verbose=False, kf_frac=K_FRAC, save_full_x=True, save_mean_D=True,
        )
        # A one-polarization stress test is cheaper than the full two-
        # polarization area calculation.  Both independent ports remain.
        waves = continuum_waveforms(omega, weights, rec["times"], e_perp)
        t0 = time.perf_counter()
        ev = propagate_phi(
            rec["X_full"], rec["w_k"], rec["mean_D_t"], waves,
            float(rec["dt"]), eta, save_pauli=False,
        )
        p = pauli_from_phi(ev["Phi"], ev["n_ph"], ev["n_B"], rec["w_k"])
        max_phi = float(np.max(np.abs(ev["Phi"])))
        row = {
            "ntpc": ntpc,
            "n_t": int(len(rec["times"])),
            "n_photon_oscillator": ev["n_ph"],
            "one_polarization_only": True,
            "two_ports": True,
            "C_P_final_unqualified": p["Ncell_times_eps_Pauli"],
            "C_C_final_unqualified": p["Ncell_times_C_minus_I"],
            "Phi_maxabs": max_phi,
            "symplectic_abs": ev["symplectic_maxabs"],
            "symplectic_scaled_by_maxPhi_squared": float(
                ev["symplectic_maxabs"] / max(1.0, max_phi**2)
            ),
            "wall_seconds": time.perf_counter() - t0,
        }
        rows = [old for old in rows if old["ntpc"] != ntpc] + [row]
        rows.sort(key=lambda old: old["ntpc"])
        print(f"ntpc={ntpc} CP_final={row['C_P_final_unqualified']:.6g} "
              f"scaled_symp={row['symplectic_scaled_by_maxPhi_squared']:.3g}",
              flush=True)

    payload = {
        "status": "time_step_unqualified_no_area_inference",
        "model": "same_P1_dressed_photon_Gaussian_two_k",
        "photon_frequency": "positive_open_Gauss_96_0_to_8omega0",
        "photon_polarization": "one_inplane_perp_stress_only",
        "N_cell": "unset",
        "S": "unset",
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
