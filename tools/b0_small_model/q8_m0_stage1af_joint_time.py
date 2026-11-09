#!/usr/bin/env python3
"""Stage 1AF: joint p-h/photon propagation on TRUE stage nodes.

The earlier production propagator approximated K(t_{n+1/2}) by the endpoint
average (K(t_n)+K(t_{n+1}))/2.  Stage 1AB showed that averaging the fast
electronic phase away changed C_P from 0.280 to 1.9e8 on the same coarse
grid, so endpoint averaging is banned.

Here the electronic background is propagated on its own fine sub-grid that
lands exactly on every required stage time (Gauss points c1, c2 and the
midpoint of each coarse step).  The joint generator K(t) is therefore
evaluated at true stage nodes for both integrators:

1. cayley_true_midpoint  - second-order baseline;
2. cf4                   - commutator-free Magnus 4 (independent reference).

Gates: stage-1AC manufactured-solution test must have reproduced orders
2 and 4; stage-1AD X_ref and stage-1AE source qualification define the
electron grid and photon basis.  P1=0 is run before P1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_classical import initial_rho, kappa_shift, p1_a2_protocol  # noqa: E402
from q8_m0_cjj import (  # noqa: E402
    occup_empty_bases, propagate_cjj, x_from_bases,
)
from q8_m0_mode_io import eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry, current_ops, h8  # noqa: E402
from q8_m0_phgauss import (  # noqa: E402
    cayley_step, is_hamiltonian, linear_K, n_ph_osc, pauli_from_phi,
    sqrt_w, symplectic_J,
)
from q8_m0_response import d_ops, mean_d_from_bases  # noqa: E402
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1s_continuum import open_grid  # noqa: E402
from q8_m0_stage1v_real_source_kernel import gauss_two_segment  # noqa: E402
from q8_m0_stage1w_propagator_crosscheck import expm_nopy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1af_joint_time_20260923"
CACHE = REPO / "results/q_line/stage1af_electron_stage_cache"
MU = -0.009010620372198276
K_FRAC = np.array([[0.0, 0.0], [0.17, 0.41]])

C1 = 0.5 - np.sqrt(3.0) / 6.0
C2 = 0.5 + np.sqrt(3.0) / 6.0
ALPHA = 0.25 + np.sqrt(3.0) / 6.0
BETA = 0.25 - np.sqrt(3.0) / 6.0
STAGE_FRACS = (C1, 0.5, C2, 1.0)


def electron_stage_data(geo, prot, ntpc, sub, a_star):
    """X and mean_D at t_n + f*h for f in STAGE_FRACS, fine electron grid.

    The electron is advanced with the same midpoint-unitary rule as
    propagate_cjj, but on a nonuniform node list that contains every stage
    time exactly.  Results are cached as npz.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"estage_a{a_star:g}_ntpc{ntpc}_sub{sub}.npz"
    path = CACHE / tag
    if path.exists():
        z = np.load(path)
        return {"times_stage": z["times_stage"], "X_stage": z["X_stage"],
                "D_stage": z["D_stage"], "w_k": z["w_k"], "h": float(z["h"]),
                "dt_e_max": float(z["dt_e_max"])}
    n_coarse = prot.n_cycles * ntpc
    h = prot.period / ntpc
    kf = np.mod(K_FRAC, 1.0)
    kc = geo.k_cart(kf)
    n_k = kc.shape[0]
    w = np.full(n_k, 1.0 / n_k)
    rho, _ = initial_rho(geo, R0, kc, +1, mu=MU)
    h0 = h8(geo, R0, kc, +1)
    vv, vc, _ = occup_empty_bases(h0, MU)

    fracs = np.array(STAGE_FRACS)
    n_st = len(fracs)
    x_stage = np.zeros((n_coarse, n_st, n_k, 2, 4, 4), dtype=complex)
    d_stage = np.zeros((n_coarse, n_st, 2, 2), dtype=float)
    times_stage = (np.arange(n_coarse)[:, None] + fracs[None, :]) * h
    dt_e_max = 0.0
    t_node = 0.0
    for n in range(n_coarse):
        f_prev = 0.0
        for i_st, f in enumerate(fracs):
            seg = (f - f_prev) * h
            m = max(1, sub)
            dt_e = seg / m
            dt_e_max = max(dt_e_max, dt_e)
            for _ in range(m):
                kmid = kc + kappa_shift(t_node + 0.5 * dt_e, geo, prot, +1)
                hm = h8(geo, R0, kmid, +1)
                em, wm = np.linalg.eigh(hm)
                ph = np.exp(-1j * em * dt_e)
                u = (wm * ph[:, None, :]) @ np.conj(np.swapaxes(wm, 1, 2))
                vv = u @ vv
                vc = u @ vc
                rho = u @ rho @ np.conj(np.swapaxes(u, 1, 2))
                t_node += dt_e
            kap = kc + kappa_shift(t_node, geo, prot, +1)
            j_now = current_ops(geo, R0, kap, +1)
            x_stage[n, i_st] = x_from_bases(j_now, vv, vc)
            d_k = mean_d_from_bases(d_ops(geo, R0, kap, +1), vv)
            d_stage[n, i_st] = np.einsum("k,kab->ab", w, d_k)
            f_prev = f
    np.savez_compressed(path, times_stage=times_stage, X_stage=x_stage,
                        D_stage=d_stage, w_k=w, h=h, dt_e_max=dt_e_max)
    return {"times_stage": times_stage, "X_stage": x_stage,
            "D_stage": d_stage, "w_k": w, "h": h, "dt_e_max": dt_e_max}


def electron_grid_check(ed, prot, ntpc):
    """Compare f=1.0 stage values against propagate_cjj on the coarse grid."""
    rec = propagate_cjj(
        geo_global, R0, prot, nt_per_cycle=ntpc, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=K_FRAC, save_full_x=True, save_mean_D=True,
    )
    x_ref = rec["X_full"][1:]  # coarse nodes t_{n+1}
    d_ref = rec["mean_D_t"][1:]
    dx = float(np.max(np.abs(ed["X_stage"][:, -1] - x_ref)) /
               max(np.max(np.abs(x_ref)), 1e-30))
    dd = float(np.max(np.abs(ed["D_stage"][:, -1] - d_ref)) /
               max(np.max(np.abs(d_ref)), 1e-30))
    return {"stage_f1_vs_propagate_cjj_X_maxrel": dx,
            "stage_f1_vs_propagate_cjj_D_maxrel": dd}


def photon_stage_waveforms(omega, weights, n_coarse, h, pols):
    """F at stage times for every quadrature node, polarization, port.

    Returns fr[i_st], fi[i_st] with shape (n_ph, n_coarse, 2); n_ph counts
    node x polarization x port oscillators.
    """
    om = np.asarray(omega, dtype=float)
    wt = np.asarray(weights, dtype=float)
    amp = np.sqrt(wt / om)
    fr_list, fi_list = [], []
    for frac in STAGE_FRACS:
        ts = (np.arange(n_coarse) + frac) * h
        phase = np.exp(1j * ts[:, None] * om[None, :]) * amp[None, :]
        fr_s, fi_s = [], []
        for pol in pols:
            pol = np.asarray(pol, dtype=float)
            for jnode in range(len(om)):
                fw = phase[:, jnode, None] * pol[None, :]
                fr_s.extend((np.real(fw), np.real(fw)))
                fi_s.extend((np.imag(fw), np.imag(fw)))
        fr_list.append(np.stack(fr_s, axis=0))
        fi_list.append(np.stack(fi_s, axis=0))
    # (i_st, n_ph, n_coarse, 2); n_ph = nodes x polarizations x 2 ports
    return np.stack(fr_list, axis=0), np.stack(fi_list, axis=0)


def propagate_joint(ed, fr, fi, eta, method, traj_every=8):
    x_st, d_st, w = ed["X_stage"], ed["D_stage"], ed["w_k"]
    n_coarse = x_st.shape[0]
    h = ed["h"]
    sw = sqrt_w(w)
    n_ph = int(fr.shape[1])
    n_b = n_ph_osc(len(w))
    n = n_ph + n_b
    j = symplectic_J(n)
    phi = np.eye(2 * n)
    i_mid = STAGE_FRACS.index(0.5)
    i_c1 = STAGE_FRACS.index(C1)
    i_c2 = STAGE_FRACS.index(C2)
    ham_res = 0.0
    n_tr = n_coarse // traj_every + 1
    cp_t = np.zeros(n_tr)
    cc_t = np.zeros(n_tr)
    nb_t = np.zeros(n_tr)
    p0 = pauli_from_phi(phi, n_ph, n_b, w)
    cp_t[0] = p0["Ncell_times_eps_Pauli"]
    cc_t[0] = p0["Ncell_times_C_minus_I"]
    nb_t[0] = p0["N_B_max"]
    for m in range(n_coarse):
        if method == "cayley_true_midpoint":
            kmat = linear_K(n_ph, sw, x_st[m, i_mid], fr[i_mid][:, m],
                            fi[i_mid][:, m], d_st[m, i_mid], eta)
            ham_res = max(ham_res, is_hamiltonian(kmat, j))
            phi = cayley_step(phi, kmat, h)
        elif method == "cf4":
            k1 = linear_K(n_ph, sw, x_st[m, i_c1], fr[i_c1][:, m],
                          fi[i_c1][:, m], d_st[m, i_c1], eta)
            k2 = linear_K(n_ph, sw, x_st[m, i_c2], fr[i_c2][:, m],
                          fi[i_c2][:, m], d_st[m, i_c2], eta)
            ham_res = max(ham_res, is_hamiltonian(k1, j),
                          is_hamiltonian(k2, j))
            phi = expm_nopy(h * (BETA * k1 + ALPHA * k2)) @ \
                expm_nopy(h * (ALPHA * k1 + BETA * k2)) @ phi
        else:
            raise ValueError(method)
        if (m + 1) % traj_every == 0 or m == n_coarse - 1:
            pt = pauli_from_phi(phi, n_ph, n_b, w)
            i_tr = (m + 1) // traj_every
            cp_t[i_tr] = pt["Ncell_times_eps_Pauli"]
            cc_t[i_tr] = pt["Ncell_times_C_minus_I"]
            nb_t[i_tr] = pt["N_B_max"]
    resid = phi.T @ j @ phi - j
    max_phi = float(np.max(np.abs(phi)))
    pf = pauli_from_phi(phi, n_ph, n_b, w)
    return {
        "Phi": phi, "n_ph": n_ph, "n_B": n_b,
        "C_P_final": pf["Ncell_times_eps_Pauli"],
        "C_C_final": pf["Ncell_times_C_minus_I"],
        "N_B_max_final": pf["N_B_max"],
        "C_P_t": cp_t, "C_C_t": cc_t, "N_B_t": nb_t,
        "Phi_maxabs": max_phi,
        "symplectic_abs": float(np.max(np.abs(resid))),
        "symplectic_normalized": float(np.max(np.abs(resid)) /
                                       max(1.0, max_phi ** 2)),
        "hamiltonian_K_maxres": ham_res,
    }


geo_global = None


def main():
    global geo_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-star", type=float, default=0.0)
    ap.add_argument("--ntpc", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--sub", type=int, default=8,
                    help="electron sub-steps per stage segment")
    ap.add_argument("--nquad", type=int, default=96)
    ap.add_argument("--omega-max", type=float, default=8.0)
    ap.add_argument("--uv-per-octave", type=int, default=0,
                    help="if >0 and omega-max>8: 96 low nodes below 8 w0 "
                         "plus this many Gauss nodes per octave above")
    ap.add_argument("--pol", choices=["perp", "par", "both"], default="perp")
    ap.add_argument("--methods", nargs="+",
                    default=["cayley_true_midpoint", "cf4"])
    ap.add_argument("--traj-every", type=int, default=8)
    ap.add_argument("--check-electron-grid", action="store_true")
    args = ap.parse_args()

    geo_global = build_geometry()
    geo = geo_global
    prot = p1_a2_protocol(args.a_star)
    e_par, e_perp = parallel_perp_basis(geo)
    eta = float(np.sqrt(eta2_ncell(geo)))
    if args.uv_per_octave > 0 and args.omega_max > 8.0:
        octaves = max(1, int(round(np.log2(args.omega_max / 8.0))))
        omega, weights = gauss_two_segment(
            0.0, 8.0 * prot.omega0, args.omega_max * prot.omega0,
            96, args.uv_per_octave * octaves)
    else:
        omega, weights = open_grid(args.omega_max * prot.omega0, args.nquad)
    pols = {"perp": (e_perp,), "par": (e_par,),
            "both": (e_par, e_perp)}[args.pol]

    runs = []
    for ntpc in args.ntpc:
        t0 = time.perf_counter()
        ed = electron_stage_data(geo, prot, ntpc, args.sub, args.a_star)
        t_el = time.perf_counter() - t0
        row_base = {"ntpc": ntpc, "sub": args.sub, "a_star": args.a_star,
                    "h": ed["h"], "dt_e_max": ed["dt_e_max"],
                    "electron_wall_seconds": t_el,
                    "nquad": args.nquad,
                    "n_frequency": int(len(omega)),
                    "uv_per_octave": args.uv_per_octave,
                    "omega_max_over_omega0": args.omega_max,
                    "polarizations": args.pol}
        if args.check_electron_grid:
            row_base["electron_grid_check"] = electron_grid_check(
                ed, prot, ntpc)
            print("electron grid check:",
                  row_base["electron_grid_check"], flush=True)
        fr, fi = photon_stage_waveforms(omega, weights,
                                       prot.n_cycles * ntpc, ed["h"], pols)
        phis = {}
        for method in args.methods:
            t1 = time.perf_counter()
            ev = propagate_joint(ed, fr, fi, eta, method, args.traj_every)
            phis[method] = ev
            row = {**row_base, "method": method,
                   "C_P_final": ev["C_P_final"], "C_C_final": ev["C_C_final"],
                   "N_B_max_final": ev["N_B_max_final"],
                   "Phi_maxabs": ev["Phi_maxabs"],
                   "symplectic_abs": ev["symplectic_abs"],
                   "symplectic_normalized": ev["symplectic_normalized"],
                   "hamiltonian_K_maxres": ev["hamiltonian_K_maxres"],
                   "wall_seconds": time.perf_counter() - t1,
                   "C_P_t": ev["C_P_t"].tolist(),
                   "C_C_t": ev["C_C_t"].tolist(),
                   "N_B_t": ev["N_B_t"].tolist(),
                   "traj_every": args.traj_every}
            sv = np.linalg.svd(ev["Phi"], compute_uv=False)
            row["Phi_singular_values_top"] = sv[:8].tolist()
            row["Phi_singular_values_bottom"] = sv[-4:].tolist()
            runs.append(row)
            print(f"ntpc={ntpc} {method:22s} CP={row['C_P_final']:.6g} "
                  f"CC={row['C_C_final']:.6g} "
                  f"symN={row['symplectic_normalized']:.3g} "
                  f"wall={row['wall_seconds']:.0f}s", flush=True)
        if len(phis) == 2:
            keys = list(phis)
            a, b = phis[keys[0]]["Phi"], phis[keys[1]]["Phi"]
            diff = float(np.linalg.norm(a - b) /
                         max(np.linalg.norm(a), 1e-30))
            for r in runs[-2:]:
                r["Phi_rel_fro_diff_between_methods"] = diff
            print(f"  Phi rel diff {keys[0]} vs {keys[1]}: {diff:.3g}",
                  flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / (f"stage1af_a{args.a_star:g}_n{args.nquad}_"
                    f"uv{args.omega_max:g}_{args.pol}_"
                    f"ntpc{'-'.join(map(str, args.ntpc))}_sub{args.sub}.json")
    payload = {
        "status": "joint_time_ordering_qualification_true_stage_nodes",
        "theory": "P1_dressed_particle_hole_photon_canonical_Gaussian",
        "stage_fracs": list(STAGE_FRACS),
        "cf4_coefficients": {"alpha": ALPHA, "beta": BETA},
        "endpoint_averaging": "banned_not_used",
        "photon_grid": f"open_Gauss_{args.nquad}_0_to_{args.omega_max}omega0",
        "ports": 2, "N_cell": "unset", "S": "unset",
        "runs": runs,
        "qualification_note": "all C_P/C_C remain unqualified until electron "
                              "grid, photon continuum and time ordering each "
                              "show a stable plateau",
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
