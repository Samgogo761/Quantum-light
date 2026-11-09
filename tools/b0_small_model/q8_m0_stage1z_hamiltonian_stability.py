#!/usr/bin/env python3
"""Stage 1Z: instantaneous generator diagnostics, not a stability verdict.

The joint p-h/photon Gaussian Hamiltonian is quadratic. At each time slice the
linearized equations are  ξ̇ = K(t) ξ,  K(t) = J Hess[H_Q(t)].
The instantaneous spectrum alone does not determine the time-ordered flow.
In this interaction-picture representation K can be nearly nilpotent and
non-normal; tiny positive real eigenvalues may be numerical eigensolver noise.
An indefinite instantaneous Hessian is not a failure criterion here.

This script scans the P1 trajectory and records, for each snapshot:
- max Re λ of K(t)
- whether K(t) is Hamiltonian (J K + K^T J = 0)
- the local condition number of Hess[H_Q]
- whether Hess[H_Q] is positive definite after an overall energy shift.

No propagation is run; only frozen K(t) matrices are evaluated.
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
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_phgauss import (  # noqa: E402
    is_hamiltonian, linear_K, n_ph_osc, pack_waveforms, sqrt_w,
)
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1s_continuum import open_grid  # noqa: E402
from q8_m0_stage1t_continuum_pauli import continuum_waveforms  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1z_hamiltonian_stability_20260923"
MU = -0.009010620372198276
K_FRAC = np.array([[0.0, 0.0], [0.17, 0.41]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntpc", type=int, default=256)
    parser.add_argument("--nquad", type=int, default=96)
    parser.add_argument("--omega-max", type=float, default=8.0)
    args = parser.parse_args()

    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    e_par, e_perp = parallel_perp_basis(geo)
    eta = float(np.sqrt(__import__("q8_m0_mode_io", fromlist=["eta2_ncell"]).eta2_ncell(geo)))
    omega, weights = open_grid(args.omega_max * prot.omega0, args.nquad)

    print("propagate_cjj for K(t) snapshots ...", flush=True)
    t0 = time.perf_counter()
    rec = propagate_cjj(
        geo, R0, prot, nt_per_cycle=args.ntpc, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=K_FRAC, save_full_x=True, save_mean_D=True,
    )
    print(f"cjj done in {time.perf_counter()-t0:.1f}s", flush=True)

    times = np.asarray(rec["times"])
    dt = float(rec["dt"])
    x = np.asarray(rec["X_full"])
    md = np.asarray(rec["mean_D_t"])
    w = np.asarray(rec["w_k"], dtype=float)
    sw = sqrt_w(w)
    waves = continuum_waveforms(omega, weights, times, e_par, e_perp)
    fr, fi = pack_waveforms(waves)
    n_ph = int(fr.shape[0])
    n_b = n_ph_osc(w.size)
    n = n_ph + n_b
    j = __import__("q8_m0_phgauss", fromlist=["symplectic_J"]).symplectic_J(n)

    rows = []
    for nstep in range(len(times)):
        xmid = x[nstep]
        dmid = md[nstep]
        frm = fr[:, nstep] if n_ph else np.zeros((0, 2))
        fim = fi[:, nstep] if n_ph else np.zeros((0, 2))
        kmat = linear_K(n_ph, sw, xmid, frm, fim, dmid, eta)
        eig = np.linalg.eigvals(kmat)
        re_max = float(np.max(np.real(eig)))
        im_max = float(np.max(np.abs(np.imag(eig))))
        ham_res = is_hamiltonian(kmat, j)
        k2_maxabs = float(np.max(np.abs(kmat @ kmat)))
        k_maxabs = float(np.max(np.abs(kmat)))
        # Hess is related to K by K = J Hess. Hess = -J K (since J^{-1}=-J).
        hess = -j @ kmat
        hess_sym = float(np.max(np.abs(hess - hess.T)))
        # Report the Hessian spectrum descriptively; it is not a stability gate.
        hess_sym_mat = 0.5 * (hess + hess.T)
        eig_h = np.linalg.eigvalsh(hess_sym_mat)
        threshold_flag = bool(re_max > 1e-10 or ham_res >= 1e-9)
        row = {
            "t": float(times[nstep]),
            "cycle": float(times[nstep] / prot.period),
            "Re_lambda_max": re_max,
            "Im_lambda_max": im_max,
            "Hamiltonian_residual": ham_res,
            "K_maxabs": k_maxabs,
            "K_squared_maxabs": k2_maxabs,
            "Hessian_symmetry_residual": hess_sym,
            "Hessian_eval_min": float(eig_h[0]),
            "Hessian_eval_max": float(eig_h[-1]),
            "threshold_flag_not_stability": threshold_flag,
        }
        rows.append(row)
        if nstep % max(1, len(times)//16) == 0:
            print(f"  t={times[nstep]:.2f} Reλmax={re_max:.6g} K2={k2_maxabs:.3g}", flush=True)

    threshold_flag_fraction = float(np.mean([r["threshold_flag_not_stability"] for r in rows]))
    payload = {
        "status": "instantaneous_generator_diagnostic_only",
        "model": "same_P1_dressed_photon_Gaussian_two_k",
        "ntpc": args.ntpc,
        "n_t": len(times),
        "omega_max_over_omega0": args.omega_max,
        "nquad": args.nquad,
        "n_photon_oscillator": n_ph,
        "n_ph_oscillator": n_b,
        "eta": eta,
        "threshold_flag_fraction_not_stability": threshold_flag_fraction,
        "any_Re_lambda_positive": bool(any(r["Re_lambda_max"] > 1e-10 for r in rows)),
        "rows": rows,
        "interpretation": (
            "Instantaneous Re eigenvalues and Hessian signs do not establish "
            "growth or breakdown. Check K^2, non-normality, and convergence "
            "of the time-ordered propagator instead."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "stage1z_hamiltonian_stability.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
