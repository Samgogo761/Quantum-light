#!/usr/bin/env python3
"""Read-only same-drive selection-rule check. No model change, no dynamics.

Candidate map: vertical mirror M containing a2
  (N, k, A_∥, j_∥, j_⊥)  ->  (-N, M k, A_∥, j_∥, -j_⊥)

Inversion P is a different map: (N, A) -> (-N, -A). It cannot prove same-drive ±N.

Hard boundaries
---------------
- Do not change Hamiltonian / P1 / initial state / phi / gamma / eta / D / k-grid.
- Do not run K', new adjoints/vertices, new U,V, classical re-propagate, quantum -N.
- Do not overwrite archived JSON (h2_class_pmN.json and all G1 archives).
- production_G1_pass=false. G1 frozen. Tranche 2 not authorized by this slice.
- Do not infer a theorem from classical beta sign-flip.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from q8_m0_classical import k_grid
from q8_m0_model import (
    R0,
    build_geometry,
    current_ops,
    h8,
    mirror_matrix_inplane,
    op_P,
    op_mirror_a2,
    op_sewing,
)
from q8_m0_s1_kernel import parallel_perp_basis

OUTDIR = ROOT / "results/q_line/h2_sel_rule_20261025"
OUTJSON = OUTDIR / "h2_sel_rule.json"
PROGRESS = OUTDIR / "progress.txt"
CLASSJSON = ROOT / "results/q_line/h2_class_pmN_run_20261025/h2_class_pmN.json"
SYMJSON = ROOT / "results/q_line/m0_symmetry_readonly_20260926/symmetry_residuals.json"
MU = -0.009010620372198276
NK = 12
KS = (
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0 / 3.0, 1.0 / 3.0),
    (0.17, -0.11),
    (0.31, 0.22),
    (-0.08, 0.41),
    (0.61, 0.13),
)
T0 = time.time()


def elapsed():
    return time.time() - T0


def prog(msg):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    line = f"{elapsed():.1f}s {msg}\n"
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def fro(a):
    return float(np.linalg.norm(np.asarray(a)))


def loadj(path):
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def z_of(d):
    return complex(d["re"], d["im"])


def cpack(z):
    z = np.asarray(z, complex).reshape(())
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z)), "arg": float(np.angle(z))}


def occup_rho(h, mu):
    e, w = np.linalg.eigh(h)
    occ = (e < mu).astype(float)
    n_occ = occ.sum(axis=-1)
    rho = np.einsum("...in,...n,...jn->...ij", w, occ, w.conj())
    return rho, n_occ, e


def stokes_from_beta(par, perp):
    s0 = abs(par) ** 2 + abs(perp) ** 2
    s2 = 2.0 * np.real(np.conj(par) * perp)
    s3 = 2.0 * np.imag(np.conj(par) * perp)
    i45 = 0.5 * abs(par + perp) ** 2
    i135 = 0.5 * abs(par - perp) ** 2
    return {
        "S0": float(s0),
        "S2": float(s2),
        "S3": float(s3),
        "S2_over_S0": float(s2 / s0),
        "S3_over_S0": float(s3 / s0),
        "I45": float(i45),
        "I135": float(i135),
    }


def stokes_block(raw):
    plus = raw["beta_src_plus"]
    minus = raw["beta_src_minus"]
    bp = z_of(plus["parallel"]), z_of(plus["perp"])
    bm = z_of(minus["parallel"]), z_of(minus["perp"])
    sp, sm = stokes_from_beta(*bp), stokes_from_beta(*bm)
    mean_i45 = 0.5 * (sp["I45"] + sm["I45"])
    return {
        "source": str(CLASSJSON),
        "not_calibrated_lab_intensity": True,
        "not_photon_number": True,
        "plus": sp,
        "minus": sm,
        "S0_plus_minus_abs": abs(sp["S0"] - sm["S0"]),
        "S2_over_S0_plus": sp["S2_over_S0"],
        "S2_over_S0_minus": sm["S2_over_S0"],
        "S3_over_S0_plus": sp["S3_over_S0"],
        "S3_over_S0_minus": sm["S3_over_S0"],
        "I45_plus": sp["I45"],
        "I45_minus": sm["I45"],
        "I45_contrast_vs_mean": float((sp["I45"] - sm["I45"]) / mean_i45) if mean_i45 else None,
        "outside_AI_quoted": {
            "S0_plus": 0.009424276,
            "S2_over_S0_plus": 0.047316,
            "S3_over_S0_plus": -0.010144,
            "I45_plus": 0.00493510,
            "I45_minus": 0.00448918,
            "I45_contrast_vs_mean": 0.0946,
        },
    }


def geometry_block(geo, Mm, e_par, e_perp):
    a1 = np.asarray(geo.a1_cart[:2], float)
    a2 = np.asarray(geo.a2_cart[:2], float)
    ang = float(np.degrees(np.arctan2(a1[0] * a2[1] - a1[1] * a2[0], np.dot(a1, a2))))
    kappa_max = 0.8 / float(geo.a_nn_mean_ang)
    A = 0.2 * e_par
    k = np.array([0.17, -0.11], float)
    peierls = fro(Mm @ (k + A) - (Mm @ k + A))
    return {
        "idealization": geo.idealization,
        "a_nn_mean_ang": float(geo.a_nn_mean_ang),
        "angle_a1_a2_deg": ang,
        "angle_minus_120_deg": ang - 120.0,
        "kappa_max_Astar08": float(kappa_max),
        "M_e_par_minus_e_par": fro(Mm @ e_par - e_par),
        "M_e_perp_plus_e_perp": fro(Mm @ e_perp + e_perp),
        "peierls_M_k_plus_Apar_identity": peierls,
        "drive_along_a2_is_M_even": True,
    }


def sample_mirror(geo, p, UM, Mm, e_par, e_perp, k, A_par):
    k = np.asarray(k, float)
    kap = k + A_par * e_par
    Mk = Mm @ kap
    h = h8(geo, p, kap, +1)
    ht = h8(geo, p, Mk, -1)
    j = current_ops(geo, p, kap, +1)
    jt = current_ops(geo, p, Mk, -1)
    h_map = UM @ h @ UM.conj().T
    j_map = np.einsum("ab,bij->aij", Mm, np.einsum("ij,bjl,lm->bim", UM, j, UM.conj().T))
    rho, n_occ, _e = occup_rho(h[None], MU)
    rho_t, n_occ_t, _e2 = occup_rho(ht[None], MU)
    rho_map = UM.conj().T @ rho_t[0] @ UM
    res_j = j_map - jt
    return {
        "A_par": float(A_par),
        "n_occ_plus": float(n_occ[0]),
        "n_occ_minus": float(n_occ_t[0]),
        "mirror_H": fro(h_map - ht),
        "mirror_H_rel": fro(h_map - ht) / max(fro(h), 1e-30),
        "mirror_j": fro(res_j),
        "mirror_j_par": fro(np.tensordot(e_par, res_j, axes=(0, 0))),
        "mirror_j_perp": fro(np.tensordot(e_perp, res_j, axes=(0, 0))),
        "occup_map": fro(rho[0] - rho_map),
        "P_same_A_is_not_this_map": True,
    }


def torus_frac_dist(a, b):
    d = np.mod(np.asarray(a, float) - np.asarray(b, float) + 0.5, 1.0) - 0.5
    return np.linalg.norm(d, axis=-1)


def grid_pairing(geo, Mm, nk):
    kf, kc, _w = k_grid(geo, nk, shift=0.0)
    Mk = kc @ Mm.T
    frac_raw = geo.k_frac(Mk)
    frac = np.mod(frac_raw, 1.0)
    idx = np.rint(frac * nk) % nk
    partner = (idx[:, 0] * nk + idx[:, 1]).astype(int)
    dist = torus_frac_dist(frac, kf[partner])
    gvec = (frac_raw - kf[partner]) @ geo.recip()
    cart = np.linalg.norm(Mk - kc[partner] - gvec, axis=1)
    return {
        "nk": nk,
        "n_k": int(kc.shape[0]),
        "max_torus_frac_dist_to_rounded_image": float(dist.max()),
        "max_cart_dist_after_G": float(cart.max()),
        "n_unpaired_frac_gt_1e8": int(np.sum(dist > 1e-8)),
        "closed_under_M_on_MP_grid": bool(dist.max() < 1e-8),
        "note": "Torus distance on crystal fractions. Cartesian M need not hit the MP grid exactly if the lattice is not 120°.",
    }


def occup_grid(geo, p, UM, Mm, nk, mu):
    _kf, kc, _w = k_grid(geo, nk, shift=0.0)
    Mk = kc @ Mm.T
    rho, n_occ, _e = occup_rho(h8(geo, p, kc, +1), mu)
    rho_t, n_occ_t, _e2 = occup_rho(h8(geo, p, Mk, -1), mu)
    rho_map = np.einsum("ij,kjl,lm->kim", UM.conj().T, rho_t, UM)
    return {
        "nk": nk,
        "n_occ_plus_unique": [float(x) for x in np.unique(n_occ)],
        "n_occ_minus_unique": [float(x) for x in np.unique(n_occ_t)],
        "max_occup_map": fro(rho - rho_map),
        "mean_occup_map": float(np.mean(np.linalg.norm(rho - rho_map, axis=(1, 2)))),
    }


def quantum_constraints_if():
    # Conditional only. Not a theorem of the production Hamiltonian.
    return {
        "operator_hypothesis": "b_par(-N) ~ b_par(+N), b_perp(-N) ~ -b_perp(+N)",
        "would_imply": {
            "n00_even": True,
            "n11_even": True,
            "m00_even": True,
            "m11_even": True,
            "n01_odd": True,
            "m01_odd": True,
            "C_XX_odd_like_classical_S2_S3": True,
        },
        "cannot_infer_from_beta_flip": True,
        "U0_without_transforming_incoming": {
            "U0_is_N_even_direct_transmission": True,
            "both_H2_pols_generically_nonzero": True,
            "code": "u0_from_bundle: u0[:,j] = phi(nu) * (pol @ ((sqrt(nu)/eta) * a_r))",
            "then_b_perp(-N)_equals_minus_b_perp(+N)": False,
            "m01_picks_up_even_U0_perp_V_par": True,
        },
        "complete_M_on_photon_field": {
            "E_par_even_E_perp_odd": True,
            "incoming_a_perp_to_minus_a_perp": True,
            "vacuum_invariant": True,
            "would_restore_operator_map_IF_electronic_M_exact": True,
            "electronic_M_exact": False,
        },
        "same_parity_as_classical_Stokes": True,
        "sign_flip_alone_is_not_new_magnetic_information": True,
    }


def decision(geo_b, samples, grid, occ, stokes, archived_worst):
    h_max = max(s["mirror_H"] for s in samples)
    j_max = max(s["mirror_j"] for s in samples)
    occ_max = max(s["occup_map"] for s in samples)
    proved = [
        "P: P H_N(k) P† = H_{-N}(-k) at machine precision (T1 / 20260926); maps (N,A)->(-N,-A), not same-drive ±N",
        "M e_par = e_par and M e_perp = -e_perp by construction of n_hat ⊥ a2",
        "therefore M(k + A e_par) = Mk + A e_par geometrically (Peierls along a2 is M-even)",
        "dh8 independent of N: zero-field current is N-blind",
        "classical beta_src ∥ even / ⊥ odd is an observed approximation in h2_class_pmN.json, not a theorem",
        "classical Stokes S2,S3 reverse with N on the same source-projection object",
    ]
    missing = [
        "U_M H_N(k) U_M† = H_{-N}(Mk) is NOT exact on the production lattice (no 120° forcing)",
        "finite-A Peierls generator therefore does not map strictly",
        "initial occup projector does not map strictly",
        "current ê_∥ even / ê_⊥ odd is not an operator identity",
        "nk=12 MP grid need not be closed under Cartesian M",
        "full I/O map also needs photon polar-vector action; U0 blocks b_perp(-N)=-b_perp(+N) if incoming is not transformed",
    ]
    return {
        "strict_map_proved": False,
        "hypothesis_status": "retained_not_theorem",
        "do_not_modify_model": True,
        "tranche2_not_authorized": True,
        "production_G1_pass": False,
        "proved": proved,
        "missing_or_counterexample": missing,
        "headline_residuals": {
            "max_mirror_H_this_run": h_max,
            "max_mirror_j_this_run": j_max,
            "max_occup_map_samples": occ_max,
            "max_occup_map_nk12": occ["max_occup_map"],
            "archived_20260926_mirror_H": archived_worst.get("mirror_H"),
            "archived_20260926_mirror_j": archived_worst.get("mirror_j"),
            "grid_closed_under_M": grid["closed_under_M_on_MP_grid"],
            "angle_minus_120_deg": geo_b["angle_minus_120_deg"],
        },
        "classical_stokes_is_the_baseline": True,
        "do_not_claim_quantum_sign_flip_is_new_info": True,
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if PROGRESS.exists():
        PROGRESS.unlink()
    prog("start sel-rule; no dynamics")
    raw = loadj(CLASSJSON)
    sym = loadj(SYMJSON)
    prog("loaded class JSON + 20260926 residuals")
    stokes = stokes_block(raw)
    geo = build_geometry()
    p = R0
    UM = op_mirror_a2(geo)
    Mm = mirror_matrix_inplane(geo)
    e_par, e_perp = parallel_perp_basis(geo)
    geo_b = geometry_block(geo, Mm, e_par, e_perp)
    prog(f"geometry angle-120={geo_b['angle_minus_120_deg']:.6e}")
    A_list = [0.0, 0.05, float(geo_b["kappa_max_Astar08"])]
    samples = []
    for kf in KS:
        k = geo.k_cart(kf)
        for A in A_list:
            rec = sample_mirror(geo, p, UM, Mm, e_par, e_perp, k, A)
            rec["k_frac"] = [float(kf[0]), float(kf[1])]
            samples.append(rec)
    prog(f"sampled {len(samples)} (k,A) mirror residuals")
    grid = grid_pairing(geo, Mm, NK)
    occ = occup_grid(geo, p, UM, Mm, NK, MU)
    prog("nk=12 pairing + occup map")
    # P at finite A along e_par is the WRONG map for same-drive ±N; record one number.
    k0 = geo.k_cart(KS[3])
    A = geo_b["kappa_max_Astar08"]
    P = op_P()
    hA = h8(geo, p, k0 + A * e_par, +1)
    h_wrong = h8(geo, p, -(k0 + A * e_par), -1)
    p_same_A = fro(P @ hA @ P.conj().T - h8(geo, p, -k0 + A * e_par, -1))
    p_reversed_A = fro(P @ hA @ P.conj().T - h_wrong)
    um_unit = fro(UM.conj().T @ UM - np.eye(8))
    payload = {
        "task": "same-drive H2 selection-rule check",
        "date": "2026-10-25",
        "production_G1_pass": False,
        "quasi_boson": "pending",
        "g1_frozen": True,
        "no_Kprime": True,
        "no_tranche2": True,
        "no_quantum_minusN": True,
        "no_classical_rerun": True,
        "do_not_modify_model": True,
        "candidate": {
            "name": "vertical_mirror_containing_a2",
            "code": "op_mirror_a2 / mirror_matrix_inplane",
            "expected": "U_M H_N(k) U_M† = H_{-N}(M k)",
            "UM_unitary_residual": um_unit,
            "not_inversion_P": True,
        },
        "stokes_baseline": stokes,
        "geometry": geo_b,
        "P_vs_M": {
            "P_maps_N_A_to_minusN_minusA": True,
            "P_H_at_finite_Apar_vs_same_A_minusN": p_same_A,
            "P_H_at_finite_Apar_vs_reversed_kappa_minusN": p_reversed_A,
            "same_drive_pmN_is_not_a_P_identity": True,
        },
        "samples": samples,
        "grid_pairing_nk12": grid,
        "occup_nk12": occ,
        "archived_20260926_worst": {
            "mirror_H": sym["worst"]["mirror_H"],
            "mirror_j": sym["worst"]["mirror_j"],
            "P_H": sym["worst"]["P_H"],
            "status": sym["status"],
        },
        "quantum_constraints_if_map_held": quantum_constraints_if(),
        "decision": decision(geo_b, samples, grid, occ, stokes, sym["worst"]),
        "stop": "hypothesis retained, not a theorem; no model change; tranche 2 not authorized",
    }
    OUTJSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    prog(f"wrote {OUTJSON}")
    d = payload["decision"]
    prog(f"strict_map_proved={d['strict_map_proved']} Hmax={d['headline_residuals']['max_mirror_H_this_run']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
