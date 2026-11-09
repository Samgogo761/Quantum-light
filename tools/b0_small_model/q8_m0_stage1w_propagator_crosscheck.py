#!/usr/bin/env python3
"""Stage 1W: propagator cross-check using midpoint exponential.

The quadratic time-dependent Hamiltonian flow is ξ̇ = K(t) ξ with K(t)=J Hess[H_Q(t)].
We compare:
1. Cayley midpoint (current production)
2. 2nd-order Magnus with midpoint rule

Both use the same K(t) midpoint sample and are second-order for smooth K(t).
This compares the matrix step, not independent time sampling or fourth-order accuracy.
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
    cayley_step, is_hamiltonian, linear_K, n_ph_osc, pack_waveforms,
    pauli_from_phi, sqrt_w, symplectic_J,
)
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1s_continuum import open_grid  # noqa: E402
from q8_m0_stage1t_continuum_pauli import continuum_waveforms  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1w_propagator_crosscheck_20260923"
MU = -0.009010620372198276
K_FRAC = np.array([[0.0, 0.0], [0.17, 0.41]])


def expm_nopy(a: np.ndarray) -> np.ndarray:
    """Matrix exponential using scaling-and-squaring with Pade(13,13)."""
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    norm1 = np.linalg.norm(a, 1)
    if norm1 == 0.0:
        return np.eye(n)
    theta13 = 5.371920351148152
    s = max(0, int(np.ceil(np.log2(norm1 / theta13))))
    a_s = a / (2.0 ** s)
    b = [64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
         1187353796428800.0, 129060195264000.0, 10559470518600.0,
         670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
         960960.0, 16380.0, 182.0, 1.0]
    i = np.eye(n, dtype=float)
    a2 = a_s @ a_s
    a4 = a2 @ a2
    a6 = a4 @ a2
    u = a_s @ (a6 @ (b[13]*a6 + b[11]*a4 + b[9]*a2)
               + b[7]*a6 + b[5]*a4 + b[3]*a2 + b[1]*i)
    v = a6 @ (b[12]*a6 + b[10]*a4 + b[8]*a2) + b[6]*a6 + b[4]*a4 + b[2]*a2 + b[0]*i
    r = np.linalg.solve(v - u, v + u)
    for _ in range(s):
        r = r @ r
    return r


def cayley(phi, kmat, dt, j):
    return cayley_step(phi, kmat, dt)


def magnus2(phi, kmat, dt, j):
    """Midpoint-rule Magnus of order 2."""
    return expm_nopy(dt * kmat) @ phi


def propagate_with_stepper(x, w, mean_d, f_list, dt, eta, stepper, *, save_pauli=False):
    x = np.asarray(x)
    w = np.asarray(w, dtype=float)
    md = np.asarray(mean_d, dtype=float)
    nt, nk = x.shape[0], x.shape[1]
    sw = sqrt_w(w)
    fr, fi = pack_waveforms(f_list)
    n_ph = int(fr.shape[0])
    n_b = n_ph_osc(nk)
    n = n_ph + n_b
    j = symplectic_J(n)
    phi = np.eye(2 * n, dtype=float)
    ham_res = 0.0
    n_b_t = np.zeros(nt) if save_pauli else None
    eps_t = np.zeros(nt) if save_pauli else None
    cmi_t = np.zeros(nt) if save_pauli else None
    if save_pauli:
        p0 = pauli_from_phi(phi, n_ph, n_b, w)
        n_b_t[0] = p0["N_B_max"]
        eps_t[0] = p0["Ncell_times_eps_Pauli"]
        cmi_t[0] = p0["Ncell_times_C_minus_I"]

    for nstep in range(nt - 1):
        xmid = 0.5 * (x[nstep] + x[nstep + 1])
        dmid = 0.5 * (md[nstep] + md[nstep + 1])
        if n_ph:
            frm = 0.5 * (fr[:, nstep] + fr[:, nstep + 1])
            fim = 0.5 * (fi[:, nstep] + fi[:, nstep + 1])
        else:
            frm = np.zeros((0, 2))
            fim = np.zeros((0, 2))
        kmat = linear_K(n_ph, sw, xmid, frm, fim, dmid, eta)
        ham_res = max(ham_res, is_hamiltonian(kmat, j))
        phi = stepper(phi, kmat, dt, j)
        if save_pauli:
            pt = pauli_from_phi(phi, n_ph, n_b, w)
            n_b_t[nstep + 1] = pt["N_B_max"]
            eps_t[nstep + 1] = pt["Ncell_times_eps_Pauli"]
            cmi_t[nstep + 1] = pt["Ncell_times_C_minus_I"]

    resid = phi.T @ j @ phi - j
    out = {
        "Phi": phi,
        "n_ph": n_ph,
        "n_B": n_b,
        "n": n,
        "J": j,
        "symplectic_maxabs": float(np.max(np.abs(resid))),
        "hamiltonian_K_maxabs": ham_res,
        "eta": float(eta),
    }
    if save_pauli:
        out["N_B_max_t"] = n_b_t
        out["Ncell_times_eps_Pauli_t"] = eps_t
        out["Ncell_times_C_minus_I_t"] = cmi_t
        out["pauli_final"] = pauli_from_phi(phi, n_ph, n_b, w)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntpc", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--nquad", type=int, default=96)
    parser.add_argument("--omega-max", type=float, default=8.0)
    parser.add_argument("--a-star", type=float, default=0.8)
    args = parser.parse_args()

    geo = build_geometry()
    prot = p1_a2_protocol(args.a_star)
    e_par, e_perp = parallel_perp_basis(geo)
    eta = float(np.sqrt(__import__("q8_m0_mode_io", fromlist=["eta2_ncell"]).eta2_ncell(geo)))
    omega, weights = open_grid(args.omega_max * prot.omega0, args.nquad)

    rows = []
    for ntpc in args.ntpc:
        t0 = time.perf_counter()
        rec = propagate_cjj(
            geo, R0, prot, nt_per_cycle=ntpc, neel=+1, sign=+1, mu=MU,
            verbose=False, kf_frac=K_FRAC, save_full_x=True, save_mean_D=True,
        )
        waves = continuum_waveforms(omega, weights, rec["times"], e_par, e_perp)
        dt = float(rec["dt"])

        base = {
            "ntpc": ntpc,
            "n_t": len(rec["times"]),
            "dt": dt,
            "omega_max_over_omega0": args.omega_max,
            "nquad": args.nquad,
            "a_star": args.a_star,
        }

        phis = {}
        for name, stepper in (
            ("cayley_midpoint", cayley),
            ("magnus2_midpoint", magnus2),
        ):
            t1 = time.perf_counter()
            ev = propagate_with_stepper(
                rec["X_full"], rec["w_k"], rec["mean_D_t"], waves, dt, eta,
                stepper, save_pauli=False,
            )
            p = pauli_from_phi(ev["Phi"], ev["n_ph"], ev["n_B"], rec["w_k"])
            row = {
                **base,
                "propagator": name,
                "C_P_final": p["Ncell_times_eps_Pauli"],
                "C_C_final": p["Ncell_times_C_minus_I"],
                "N_B_max_final": float(p["N_B_max"]),
                "symplectic_maxabs": ev["symplectic_maxabs"],
                "symplectic_scaled_by_maxPhi_squared": float(
                    ev["symplectic_maxabs"] /
                    max(1.0, float(np.max(np.abs(ev["Phi"]))) ** 2)
                ),
                "hamiltonian_K_maxabs": ev["hamiltonian_K_maxabs"],
                "wall_seconds": time.perf_counter() - t1,
            }
            rows.append(row)
            print(f"ntpc={ntpc} {name} CP={row['C_P_final']:.6g} "
                  f"CC={row['C_C_final']:.6g} sym={row['symplectic_maxabs']:.6g} "
                  f"wall={row['wall_seconds']:.2f}s", flush=True)
            phis[name] = ev["Phi"]

        diff = float(np.linalg.norm(phis["cayley_midpoint"] - phis["magnus2_midpoint"]) /
                     max(np.linalg.norm(phis["cayley_midpoint"]), 1e-30))
        for r in rows[-2:]:
            r["Phi_fro_rel_diff_cayley_vs_magnus2"] = diff
        print(f"  Phi rel diff cayley vs magnus2: {diff:.6g}", flush=True)
        print(f"  ntpc={ntpc} total wall {time.perf_counter()-t0:.1f}s", flush=True)

    payload = {
        "status": "propagator_cross_check",
        "model": "same_P1_dressed_photon_Gaussian_two_k",
        "a_star": args.a_star,
        "propagators": ["cayley_midpoint", "magnus2_midpoint"],
        "rows": rows,
        "interpretation": "Both integrators use the same endpoint-averaged midpoint K. "
                          "Agreement tests the Cayley matrix step only; it does "
                          "not establish time-sampling, continuum, or model convergence.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / (f"stage1w_a{args.a_star:g}_n{args.nquad}_"
                    f"uv{args.omega_max:g}_ntpc{'-'.join(map(str, args.ntpc))}.json")
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
