#!/usr/bin/env python3
"""Stage-1C loop-gain diagnostics. Six k, +N, local CPU.

Freezes O(λ²) as Born/one-pass. Does not replace production V(θ).
Does not run 48/96/192, g2, or a Fock benchmark.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import choose_probe_k, dump, propagate_cjj  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_loop import (  # noqa: E402
    F_from_phi,
    LOOP_ID,
    ResummationNotChosenError,
    algebraic_resummation_diagnostics,
    apply_resummation_to_output,
    causal_one_mode_cnumber,
    feedback_2x2,
    gram_schmidt_packet,
    mode_matrix_elements,
    port_O4_piece,
    shape_leftover,
)
from q8_m0_mode_io import (  # noqa: E402
    close_output_moments,
    eta2_ncell,
    mode_overlap,
    mode_waveforms,
    response_to_complex_scalar,
)
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_s1_kernel import pack_complex, parallel_perp_basis  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1c_loopgain_20260921"
MU = -0.009010620372198276
NTPC = 400


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    kf = choose_probe_k(geo, R0, MU)["kf"]
    blocked = False
    try:
        apply_resummation_to_output()
    except ResummationNotChosenError:
        blocked = True

    print("[stage1c] unperturbed +N 400", flush=True)
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=NTPC, neel=+1, sign=+1, mu=MU,
        verbose=True, kf_frac=kf, save_full_x=True, save_mean_D=True,
    )
    times = np.asarray(r0["times"])
    dt = float(r0["dt"])
    x = r0["X_full"]
    w = r0["w_k"]
    mean_d = r0["mean_D_t"]
    closed = close_output_moments(x, w, r0["mu_per_cell"], mean_d, times, dt, geo, prot)
    A2 = complex(closed["A2"]["re"], closed["A2"]["im"])
    B2 = complex(closed["B2"]["re"], closed["B2"]["im"])
    n_src = complex(closed["term_ledger"]["n_source_O_lambda2"]["re"],
                    closed["term_ledger"]["n_source_O_lambda2"]["im"])
    m_src = complex(closed["term_ledger"]["m_source_O_lambda2"]["re"],
                    closed["term_ledger"]["m_source_O_lambda2"]["im"])
    comm_s = complex(closed["commutator_check"]["s1_commutator"]["re"],
                     closed["commutator_check"]["s1_commutator"]["im"])
    loop = feedback_2x2(A2, B2)
    alg = algebraic_resummation_diagnostics(A2, B2, n_src, m_src, comm_s)

    mw = mode_waveforms(times, prot.omega0, prot.t_pulse, geo)
    F2 = np.asarray(mw["F"])
    e_par, e_perp = parallel_perp_basis(geo)
    k_a = np.asarray(mw["K_a_shape"])
    # overwrite e_perp from mw with the same basis
    e_perp = np.asarray(mw["e_perp"])
    pref = eta2_ncell(geo)
    mat22 = mode_matrix_elements(x, w, mean_d, F2, F2, e_perp, dt, pref)
    leftover = shape_leftover(k_a, mat22["dmu_j"], dt, ehat=e_perp)

    print("[stage1c] orthogonal H3 packet", flush=True)
    gs = gram_schmidt_packet(prot.omega0, prot.t_pulse, n_main=2, n_other=3)
    F3 = F_from_phi(times, gs["omega"], gs["phi_orth"])
    mat23 = mode_matrix_elements(x, w, mean_d, F2, F3, e_perp, dt, pref)
    mat32 = mode_matrix_elements(x, w, mean_d, F3, F2, e_perp, dt, pref)
    mat33 = mode_matrix_elements(x, w, mean_d, F3, F3, e_perp, dt, pref)
    o4_same = float(abs(A2) ** 2 - abs(B2) ** 2)
    o4_cross = port_O4_piece(mat23["A"], mat23["B"]) + port_O4_piece(mat32["A"], mat32["B"])
    o4_33 = port_O4_piece(mat33["A"], mat33["B"])

    print("[stage1c] H1-orth and parallel-pol mixing", flush=True)
    gs1 = gram_schmidt_packet(prot.omega0, prot.t_pulse, n_main=2, n_other=1)
    F1 = F_from_phi(times, gs1["omega"], gs1["phi_orth"])
    mat21 = mode_matrix_elements(x, w, mean_d, F2, F1, e_perp, dt, pref)
    mat12 = mode_matrix_elements(x, w, mean_d, F1, F2, e_perp, dt, pref)
    o4_h1 = port_O4_piece(mat21["A"], mat21["B"]) + port_O4_piece(mat12["A"], mat12["B"])
    resp_par = response_to_complex_scalar(x, w, F2, e_par, dt, mean_d)
    resp_par_s = response_to_complex_scalar(x, w, np.conj(F2), e_par, dt, mean_d)
    k_perp = F2[:, None] * e_perp[None, :]
    A_p = 1j * pref * mode_overlap(k_perp, resp_par_s["total"], dt)
    B_p = 1j * pref * mode_overlap(k_perp, resp_par["total"], dt)
    o4_pol = port_O4_piece(A_p, B_p)

    print("[stage1c] causal c-number forward", flush=True)
    causal = causal_one_mode_cnumber(x, w, mean_d, F2, e_perp, dt, pref, alpha=1.0)
    born_cnum = A2 * 1.0 + B2 * 1.0
    causal_vs_born = abs(complex(causal["beta_final"]["re"], causal["beta_final"]["im"]) - born_cnum)

    out = {
        "loop_id": LOOP_ID,
        "spec": stage1_spec(),
        "resummation_production_blocked": blocked,
        "n_k": 6,
        "neel": "+1_method_only",
        "born_frozen": {
            "A2": closed["A2"],
            "B2": closed["B2"],
            "n_source": closed["term_ledger"]["n_source_O_lambda2"],
            "m_source": closed["term_ledger"]["m_source_O_lambda2"],
            "V_min_born": closed["V_theta"]["min"],
            "O2_identity_residual": closed["commutator_check"]["residual_abs"],
            "partial_O4_1_plus_|A2|^2_minus_|B2|^2": loop["partial_O4_1_plus_|A2|^2_minus_|B2|^2"],
            "label": "Born_one_pass_not_full_linear_IO",
        },
        "loop_2x2": loop,
        "shape_leftover": leftover,
        "algebraic_resummation": alg,
        "causal_cnumber": {
            **causal,
            "born_A2_plus_B2": pack_complex(born_cnum),
            "abs_causal_minus_born": float(causal_vs_born),
            "rel_causal_minus_born": float(
                causal_vs_born / max(abs(born_cnum), 1e-30)
            ),
        },
        "orthogonal_H3": {
            "overlap_before": gs["overlap_before"],
            "overlap_after": gs["overlap_after"],
            "A23": pack_complex(mat23["A"]),
            "B23": pack_complex(mat23["B"]),
            "A32": pack_complex(mat32["A"]),
            "B32": pack_complex(mat32["B"]),
            "A33": pack_complex(mat33["A"]),
            "B33": pack_complex(mat33["B"]),
            "O4_same_mode_|A22|^2_|B22|^2": o4_same,
            "O4_cross_23_plus_32": o4_cross,
            "O4_port3_|A33|^2_|B33|^2": o4_33,
            "cross_over_same_O4": abs(o4_cross) / max(abs(o4_same), 1e-30),
            "note": (
                "Other ports enter [b2,b2†] at O(λ⁴) via |A2j|²-|B2j|². "
                "They cannot repair the O(λ²) ledger; they may shift the partial O(λ⁴) piece."
            ),
        },
        "orthogonal_H1": {
            "overlap_after": gs1["overlap_after"],
            "A21": pack_complex(mat21["A"]),
            "B21": pack_complex(mat21["B"]),
            "O4_cross_21_plus_12": o4_h1,
            "cross_over_same_O4": abs(o4_h1) / max(abs(o4_same), 1e-30),
        },
        "parallel_pol_H2": {
            "A_perp_from_par_drive": pack_complex(A_p),
            "B_perp_from_par_drive": pack_complex(B_p),
            "O4_pol": o4_pol,
            "cross_over_same_O4": abs(o4_pol) / max(abs(o4_same), 1e-30),
        },
        "fock_benchmark": "not_run",
        "nk96": "not_run",
        "g2": "not_run",
        "squeezing_claim": "not_evaluated",
        "unresolved_magnetic_promoted": False,
        "production_V_still_born": True,
    }
    dump(OUT / "stage1c_loopgain.json", out)
    print(json.dumps({
        "wrote": str(OUT / "stage1c_loopgain.json"),
        "abs_A2": loop["A2"]["abs"],
        "spectral_radius": loop["spectral_radius"],
        "leftover_time_Re": leftover["time_shape_leftover_Re_drive"],
        "leftover_cart_Re": leftover["cartesian_along_ehat_Re_drive"],
        "causal_rel": out["causal_cnumber"]["rel_causal_minus_born"],
        "O4_same": o4_same,
        "O4_cross_H3": o4_cross,
        "O4_cross_H1": o4_h1,
        "O4_pol": o4_pol,
        "comm_A_minus_1": alg["convention_A_sample_equals_in_plus_rad"]["commutator_abs_minus_1"],
        "comm_B_minus_1": alg["convention_B_sample_equals_out"]["commutator_abs_minus_1"],
        "resummation_blocked": blocked,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
