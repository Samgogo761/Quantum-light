"""Even/odd optical channels + inversion dictionary. Read-only.

Hard boundaries
---------------
- Do not change Hamiltonian / P1 / initial state / phi / gamma / eta / D / k-grid.
- Do not run K', new adjoints/vertices, new production U,V, long propagation,
  +/-N, H3, or two-color.
- Do not overwrite archived JSON (nm_tables, AB12, gram, A6, io_identity,
  kernel correction, port_parity, completeness, precheck).
- production_G1_pass=false. Quasi-boson pending.
- Do not treat m01 as N-odd a priori.
- Do not interpret 2-k n_exactly_dark 0->2 as a production dark-column error.
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

from q8_m0_cjj import occup_empty_bases, x_from_bases
from q8_m0_incoming_spectral_completeness_20261004 import (
    axis_near_zero_mask,
    dark_subspace,
)
from q8_m0_model import N_OCC, R0, build_geometry, current_ops, h8, op_P
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import (
    even_odd_from_lr,
    photon_port_indices,
)
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import even_bd, oneside_bd

OUTDIR = ROOT / "results/q_line/eo_inv_dict_20261023"
OUTJSON = OUTDIR / "eo_inv_dict.json"
PROGRESS = OUTDIR / "progress.txt"
PORT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/port_parity_macro2048.json"
PORTG = ROOT / "results/q_line/g1_gamma12_time_order_20261015/port_parity_macro2048_column_gram.json"
NMJSON = ROOT / "results/q_line/g1_nm_tables_20261017/nm_tables.json"
COMPL = ROOT / "results/q_line/incoming_spectral_completeness_20261004/completeness_checks.json"
REV = ROOT / "results/q_line/incoming_completeness_revision_20261004/revision_checks.json"
PRE = ROOT / "results/q_line/precheck_sym_cred_20261022/precheck_sym_cred.json"
MU = -0.009010620372198276
FROZEN_M01 = 3511.5609928247572
T0 = time.time()


def elapsed():
    return time.time() - T0


def prog(msg):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    line = f"{elapsed():.1f}s {msg}\n"
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def loadj(path):
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def even_bd_is_classical_not_operator():
    """even_bd L=R is a classical even-channel construction, not a_L=a_R."""
    rng = np.random.default_rng(23)
    n_el = 6
    gaps = np.array([0.31, 0.44, 0.62, 0.81, 1.05, 1.40], float)
    u = rng.normal(size=(2, n_el)) + 1j * rng.normal(size=(2, n_el))
    d = rng.normal(size=(2, 2))
    d = 0.5 * (d + d.T)
    gamma, eta, om = 0.0010807892763646022, 0.013, 0.11
    b1, bd1, a1 = oneside_bd(gaps, u, d, gamma, eta, om)
    b2, bd2, a2 = even_bd(gaps, u, d, gamma, eta, om)
    s = np.sqrt(2.0)
    labels = [
        (0.11, "L", 0),
        (0.11, "R", 0),
        (0.11, "L", 1),
        (0.11, "R", 1),
        (0.25, "L", 0),
        (0.25, "R", 0),
    ]
    left, right = photon_port_indices(labels)
    dummy = np.arange(6, dtype=float)[None, :]
    even_a, odd_a = even_odd_from_lr(np.vstack([dummy, dummy]), left, right)
    return {
        "even_bd_doc": "Even two-pol columns. L=R so even=√2×oneside; do not add L and R.",
        "even_bd_meaning": "classical_or_stationary_even_channel_construction",
        "not_operator_identity_aL_equals_aR": True,
        "independent_LR_vacuum_ports_kept": True,
        "quantum_CCR_ports": "photon_port_indices splits labels by side; even_odd_from_lr is (L±R)/sqrt(2) on columns",
        "even_is_sqrt2_oneside_maxabs": float(
            max(
                np.max(np.abs(b2 - s * b1)),
                np.max(np.abs(bd2 - s * bd1)),
                np.max(np.abs(a2 - s * a1)),
            )
        ),
        "n_L": int(left.size),
        "n_R": int(right.size),
        "even_odd_from_lr_shapes": [list(even_a.shape), list(odd_a.shape)],
        "oneside_readout": "a_r_out_from_column: L uses sheet a, R uses a-A_f; detector is one-sided",
        "code_anchors": {
            "even_bd": "tools/b0_small_model/q8_m0_p1_h2_finite_kcv_input_close_20261005.py",
            "photon_port_indices": "tools/b0_small_model/q8_m0_p1_h2_finite_kcv_caller_fix_20261005.py",
            "a_r_out_from_column": "tools/b0_small_model/q8_m0_p1_h2_finite_kcv_input_close_20261005.py",
        },
    }


def even_odd_from_archive():
    p = loadj(PORT)
    g = loadj(PORTG)
    du = float(p["max_du_L_minus_R"])
    dv = float(p["max_v_L_minus_R"])
    odd_dyn = float(p["dynamic_minus"]["normal_without_I"]["maxabs"])
    odd_full = float(p["full_minus"]["normal_without_I"]["maxabs"])
    u0m = float(p["u0_minus"]["normal_without_I"]["maxabs"])
    cross_n = float(p["plus_minus_cross_normal"]["maxabs"])
    cross_a = float(p["plus_minus_cross_anomalous"]["maxabs"])
    odd_free = bool(du < 1e-12 and dv < 1e-12 and odd_dyn < 1e-20)
    # Production n,m sum same-column U_j V_j. Odd V~0 so odd input drops out of n,m.
    return {
        "source": str(PORT.relative_to(ROOT)).replace("\\", "/"),
        "cache_sha256": p.get("cache_sha256"),
        "max_du_L_minus_R": du,
        "max_v_L_minus_R": dv,
        "max_u0_L_minus_R": float(p["max_u0_L_minus_R"]),
        "odd_dynamic_normal_maxabs": odd_dyn,
        "odd_full_normal_maxabs": odd_full,
        "u0_minus_normal_maxabs": u0m,
        "odd_minus_U0_maxabs": float(g["odd_minus_U0_maxabs"]),
        "plus_minus_cross_normal_maxabs": cross_n,
        "plus_minus_cross_anomalous_maxabs": cross_a,
        "cross_is_not_in_n_or_m": True,
        "cross_note": "U_+ U_-^dagger is coherence between different input columns; n=V*V^T and m=UV^T sum same-column products only.",
        "odd_channel_free_propagation": odd_free,
        "odd_V_vanishes": odd_free,
        "even_odd_dynamically_decoupled": odd_free,
        "U0_one_sided_asymmetric": bool(float(p["max_u0_L_minus_R"]) > 1e-3),
        "production_n_m_from_even_plus_dark_only": True,
        "hypothesis_A_oneside_extra_Neel_oddness": "excluded_for_centered_n_m",
        "oneside_vs_even": {
            "production_object": "one-sided 2-pol detector, not (b_R, b_L) pair",
            "if_two_sided_and_odd_vacuum": "n_R=n_L=n_s/2, m_R=m_L=m_s/2, same N-parity as even",
            "actual_production": "n=n_even_input+n_dark, m=m_even_input+m_dark; odd input does not enter",
            "oneside_does_not_create_new_N_parity": True,
        },
        "do_not_recompute_2048": True,
    }


def inversion_dictionary():
    """P-map vs same-A ±N. Occup projector is A=0 only."""
    geo = build_geometry()
    P = op_P()
    samples = [
        ("Gamma", geo.k_cart([0.0, 0.0])),
        ("K", geo.k_cart([1.0 / 3.0, 1.0 / 3.0])),
        ("Kp", geo.k_cart([-1.0 / 3.0, -1.0 / 3.0])),
        ("M", geo.k_cart([0.5, 0.0])),
    ]
    rng = np.random.default_rng(23)
    for i in range(4):
        samples.append((f"rng{i}", geo.k_cart(rng.uniform(-0.7, 0.7, 2))))
    max_rho = 0.0
    max_h = 0.0
    rows = []
    for name, k in samples:
        hp = h8(geo, R0, k, +1)
        hm = h8(geo, R0, -k, -1)
        h_res = float(np.linalg.norm(P @ hp @ P.conj().T - hm))
        # occup: eigh of 8x8
        ep, wp = np.linalg.eigh(hp)
        em, wm = np.linalg.eigh(hm)
        n_occ_p = int(np.sum(ep < MU))
        n_occ_m = int(np.sum(em < MU))
        rho_p = wp[:, :N_OCC] @ wp[:, :N_OCC].conj().T
        rho_m = wm[:, :N_OCC] @ wm[:, :N_OCC].conj().T
        rho_res = float(np.linalg.norm(P @ rho_p @ P.conj().T - rho_m))
        max_rho = max(max_rho, rho_res)
        max_h = max(max_h, h_res)
        rows.append(
            {
                "name": name,
                "n_occ_plus": n_occ_p,
                "n_occ_minus": n_occ_m,
                "n_occ_equals_N_OCC": bool(n_occ_p == N_OCC and n_occ_m == N_OCC),
                "P_rho_P_minus_rho_flipN_negk": rho_res,
                "P_H_P_minus_H_flipN_negk": h_res,
            }
        )
    pre = loadj(PRE)
    nm = loadj(NMJSON)
    m01 = float(nm.get("cross_pol", {}).get("m_a_01_abs", FROZEN_M01))
    return {
        "P_electronic_A0": "P H_N(k) P^dagger = H_{-N}(-k)",
        "max_H_residual_this_round": max_h,
        "occup_A0": {
            "identity": "P rho_N(k) P^dagger = rho_{-N}(-k) at A=0, mu frozen",
            "max_residual": max_rho,
            "holds_at_machine_prec": bool(max_rho < 1e-12),
            "closes_T1_initial_state_at_A0_only": True,
            "not_finite_pulse_occup": True,
            "samples": rows,
        },
        "precheck_T1_cited": {
            "max_residual_P_H_flipN": pre["t1"]["max_residual_P_H_flipN"],
            "j_insensitive_to_N_at_A0": pre["t1"]["current"]["j_insensitive_to_N_at_A0"],
        },
        "P_full_candidate_map": "(N, A, k, L/R) -> (-N, -A, -k, R/L)",
        "same_A_plus_minus_N": {
            "is_P_symmetry": False,
            "needs_extra_dynamics": True,
            "classical_run_set_identity_is": "(N,+A) vs (-N,-A), not same-A ±N",
        },
        "A_to_minus_A_fixed_N": {
            "is_P_symmetry": False,
            "needs_extra_dynamics": True,
        },
        "if_scattering_N_odd_at_fixed_A": {
            "status": "dynamical_hypothesis_not_a_theorem",
            "assume": "U0 N-even (linear sheet); DeltaU and V N-odd like SHG current",
            "n": "N-even (V* V^T)",
            "m_U0_V": "N-odd (U0 V^T); this is m_reference_source in input_close",
            "m_dU_V": "N-even (DeltaU V^T)",
            "abs_m_Q_n": "not clean N-odd diagnostics",
            "m01_modulus": "not N-odd; frozen |m01| is same-N two-pol correlator",
            "frozen_m01_abs": m01,
        },
        "port_swap_on_centered_nm": {
            "if_odd_vacuum_and_decoupled": "n_R=n_L, m_R=m_L so port swap does not change centered n,m",
            "then_P_constrains_same_side": "(N,A) vs (-N,-A) on the same detector for n,m",
            "still_not_same_A_pmN": True,
        },
        "strictly_constrained_now": [
            "A=0 electronic P H_N(k) P^dagger = H_{-N}(-k)",
            "A=0 occup projector tracks N with H",
            "A=0 current N-insensitive; P j P^dagger = -j(-k)",
            "even/odd dynamical decoupling on archived adjoint rows",
            "production n,m independent of odd-input V (V_odd~0)",
            "one-sided detection does not add a new N-parity to n,m",
        ],
        "still_needs_dynamics": [
            "same-A +N vs -N for V, DeltaU, signed m",
            "whether U0 is strictly N-even at the packet frequencies",
            "finite-pulse Peierls / CEP beyond A=0 occup",
            "H3 / two-color maps",
        ],
    }


def dark_two_k():
    geo = build_geometry()
    kc = geo.k_cart(np.array([[0.0, 0.0], [0.23, 0.12]]))
    weights = np.array([0.5, 0.5])
    rels = (1e-14, 1e-12, 1e-10, 1e-8, 1e-6)
    domains = []
    for neel in (1, -1):
        vv, vc, en = occup_empty_bases(h8(geo, R0, kc, neel), MU)
        x = x_from_bases(current_ops(geo, R0, kc, neel), vv, vc)
        gap = (en[:, 4:, None] - en[:, None, :4]).reshape(-1)
        u = np.einsum("k,kacv->akcv", np.sqrt(weights), x).reshape(2, -1)
        strength = np.sqrt(np.sum(np.abs(u) ** 2, axis=0))
        basis, info, near = dark_subspace(gap, u)
        axis_counts = {
            f"{rel:.0e}": int(np.count_nonzero(axis_near_zero_mask(u, relative=rel)))
            for rel in rels
        }
        svals = []
        for item in info:
            svals.append(
                {
                    "gap": item["gap"],
                    "size": item["size"],
                    "nullity": item["nullity"],
                    "kernel_floor": item["kernel_floor"],
                    "coupling_singular_values": item["coupling_singular_values"],
                    "unresolved_count": item["unresolved_count"],
                }
            )
        domains.append(
            {
                "neel": int(neel),
                "n_el": int(u.shape[1]),
                "min_vertex_strength": float(np.min(strength)),
                "max_vertex_strength": float(np.max(strength)),
                "axis_near_zero_vs_relative": axis_counts,
                "gap_subspace_nullity": int(basis.shape[1]),
                "n_exact_gap_clusters": len(info),
                "near_degenerate_pairs": len(near),
                "clusters": svals,
                "not_a_BZ_dark_dimension": True,
            }
        )
    compl = loadj(COMPL)
    rev = loadj(REV)
    archived = [
        {
            "neel": d["neel"],
            "n_exactly_dark": d["n_exactly_dark"],
            "min_vertex_strength": d["min_vertex_strength"],
        }
        for d in compl["M0_interface"]["domains"]
    ]
    return {
        "sample": "2-k fixture [[0,0],[0.23,0.12]], weights 1/2, same as completeness_checks",
        "threshold_note": "n_exactly_dark uses a 1e-10 relative threshold on this 2-k sample; it is not a BZ dark-space theorem.",
        "do_not_read_0_to_2_as_production_dark_error": True,
        "production_G1_dark_columns": 1976,
        "archived_completeness": archived,
        "archived_revision_nullity": rev["M0_two_k_inventory"]["domains"],
        "this_round": domains,
        "interpretation": (
            "N=+1 vs N=-1 axis count at 1e-10 jumps 0 vs 2 because one 2-k "
            "vertex strength sits near 3e-14, below 1e-10*scale. dark_subspace "
            "SVD kernel is a different object (gap-cluster ker(u*)). Neither "
            "is the G1 1976 dark columns."
        ),
    }


def decision(eo, inv, dark):
    a_excluded = eo["hypothesis_A_oneside_extra_Neel_oddness"] == "excluded_for_centered_n_m"
    signed_m_open = True
    return {
        "question": "Is existing H2 two-pol still worth being the main object for Néel-sensitive quantum correlations?",
        "answer": "keep_as_candidate",
        "previous_answer_retracted": "not_as_mainline",
        "reason_retract": (
            "not_as_mainline was not implied by the even/odd + inversion numbers. "
            "Optical left/right parity is not Néel N→−N parity. "
            "U0 even / ΔU,V odd in N at fixed A remains an unproved dynamical hypothesis."
        ),
        "hypothesis_A_oneside_makes_m01_N_odd": "excluded",
        "hypothesis_B_H2_stats_N_even": "expected_for_n_and_abs_m_under_unproved_scattering_oddness; not a P theorem",
        "what_is_proved": [
            "One-sided detection does not create extra Néel-oddness in centered n,m once even/odd are dynamically decoupled.",
            "Frozen |m01|~3511.56 is a same-N two-pol anomalous correlator, not a measured domain contrast.",
        ],
        "what_is_not_proved": [
            "H2 two-pol correlations are Néel-blind at fixed drive A.",
            "H2 two-pol must leave the paper mainline.",
            "U0 V^T by itself is a measurable domain quantum signal.",
        ],
        "modulus_contrast_identity": (
            "If m(+N)=m_e+m_o and m(-N)=m_e-m_o, then "
            "|m(+N)|^2-|m(-N)|^2 = 4 Re(m_e m_o*). "
            "Modulus not being a pure N-odd function does not imply no domain contrast."
        ),
        "observable_is_full_m": "m=U V^T; U0 V^T is a source tracer only",
        "not_forbidden_observables": [
            {
                "name": "full signed m_mu nu = U V^T, including signed m01 not |m01|",
                "N_parity_if_scattering_odd": "open; even+odd mix can still contrast via 4 Re(m_e m_o*)",
                "proved": False,
                "keep_as": "candidate",
            },
            {
                "name": "U0 V^T block (m_reference_source)",
                "N_parity_if_scattering_odd": "odd_piece_from_U0_V if that hypothesis holds",
                "proved": False,
                "keep_as": "source_tracer_not_standalone_observable",
            },
            {
                "name": "classical coherent beta_H2 under (N,A) vs (-N,-A) or same-A ±N",
                "N_parity": "classically_odd_in_N_at_fixed_A",
                "is_quantum_correlation": False,
            },
        ],
        "H2_H3_or_two_color": {
            "may_be_studied": True,
            "not_required_to_replace_H2_two_pol_yet": True,
            "not_authorized_this_round": True,
        },
        "do_not_start": ["+/-N propagation", "H3", "two-color", "K'", "new U,V"],
        "next_if_user_reopens": (
            "H2 two-pol remains a candidate. Do not treat |m| as the only diagnostic. "
            "A later same-A ±N read would still need user approval. "
            "H2-H3 / two-color are optional, not mandatory replacements."
        ),
        "even_odd_decoupled": bool(eo["even_odd_dynamically_decoupled"]),
        "hypothesis_A_excluded": a_excluded,
        "signed_m_still_open": signed_m_open,
        "dark_0_to_2_not_production_error": bool(dark["do_not_read_0_to_2_as_production_dark_error"]),
        "production_G1_pass": False,
        "quasi_boson": "pending",
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if PROGRESS.exists():
        PROGRESS.unlink()
    prog("eo-inv start")
    eo_code = even_bd_is_classical_not_operator()
    prog("even_bd vs quantum L/R")
    eo_arch = even_odd_from_archive()
    prog("port_parity archive")
    inv = inversion_dictionary()
    prog("inversion dictionary + A0 occup")
    dark = dark_two_k()
    prog("2-k dark SVD / threshold")
    dec = decision(eo_arch, inv, dark)
    prog("decision")
    out = {
        "task": "even/odd decoupling + inversion dictionary + 2-k dark projector 20261023",
        "production_G1_pass": False,
        "no_Kprime": True,
        "no_new_UV_assembly": True,
        "no_long_propagation": True,
        "no_pmN_H3_two_color": True,
        "hard_boundaries_honored": True,
        "even_bd_vs_quantum_ports": eo_code,
        "even_odd_channels": eo_arch,
        "inversion_dictionary": inv,
        "dark_two_k": dark,
        "decision": dec,
        "elapsed_s": elapsed(),
    }
    with OUTJSON.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    prog(f"wrote {OUTJSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
