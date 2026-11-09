#!/usr/bin/env python3
"""P1=0 two-port canonical I/O audit (commutators only).

Frozen ports (q8_m0_gsheet):
    A_R_out = A_L_in + A_self,  A_L_out = A_R_in + A_self
    A_self(z) = (i γ / z) I(z)
Electronics: I = ξ - K A_loc,  A_loc = A_L_in + A_R_in + A_self
    F = I_2 + (i γ / z) K,  I = F^{-1} (ξ - K (A_L+A_R)_in)
Odd:  A_-_out = - A_-_in, completely decoupled.
Even: A_+_out = S A_+_in + G ξ
    S = 2 F^{-1} - I_2 = (I - i (γ/z) K) F^{-1}
    G = √2 (i γ / z) F^{-1}
ξ commutator from the already-audited jump:
    C_I = i (K^R - K^A) = S^> - S^<
Scalar Cayley + that jump fix the even-A prefactor:
    S S† + (ω/γ) G C_I G† = I_2   (ω = Re z > 0)
No n, m, V, collection packet, Pauli, or initial state.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_classical import k_grid
from q8_m0_gsheet import gamma_self
from q8_m0_stage1ag_acceptance_audit import (
    REPO, MU, R0, build_geometry, h8, current_ops, occup_empty_bases,
    x_from_bases, d_ops, mean_d_from_bases,
)
from q8_m0_stage1ai_lowfreq_kernel import pack_matrix
from q8_m0_stage1ai_lowfreq_audit import spectral_data
from q8_m0_stage1al_pole_audit import (
    closed_loop_F, increment_over_z, kernel_off_cuts,
)

OUT = REPO / "results/q_line/stage1an_canonical_io_20260924"
GRIDS = (
    ("Gamma_96x96", 96, 0.0),
    ("Gamma_192x192", 192, 0.0),
    ("halfshift_192x192", 192, 0.5),
)
ETA = 1e-4


def even_maps(k, omega, gamma):
    """S and G for even A_+. omega>0."""
    omega = float(omega)
    f = closed_loop_F(k, omega, gamma, from_increment=False)[0]
    finv = np.linalg.inv(f)
    s = 2.0 * finv - np.eye(2, dtype=complex)
    g = np.sqrt(2.0) * (1j * gamma / omega) * finv
    return s, g, f


def current_jump(d, gap, c, omega, eta):
    kr = kernel_off_cuts(d, gap, c, omega + 1j * eta)[0]
    ka = kernel_off_cuts(d, gap, c, omega - 1j * eta)[0]
    return 1j * (kr - ka), kr, ka


def identity_matrices(s, g, c_i, omega, gamma):
    cel = (omega / gamma) * (g @ c_i @ g.conj().T)
    ss = s @ s.conj().T
    sts = s.conj().T @ s
    return {
        "S": pack_matrix(s),
        "C_el": pack_matrix(cel),
        "SS_dagger_plus_Cel_minus_I": pack_matrix(ss + cel - np.eye(2)),
        "S_dagger_S_plus_Cel_minus_I": pack_matrix(sts + cel - np.eye(2)),
        "SS_dagger_minus_I": pack_matrix(ss - np.eye(2)),
        "S_dagger_S_minus_I": pack_matrix(sts - np.eye(2)),
        "unitary_residual": float(np.max(np.abs(sts - np.eye(2)))),
        "canonical_ss_residual": float(np.max(np.abs(ss + cel - np.eye(2)))),
        "canonical_sts_residual": float(np.max(np.abs(sts + cel - np.eye(2)))),
        "odd_channel": {
            "A_minus_out_plus_A_minus_in": 0.0,
            "note": "A_-_out = -A_-_in from frozen A_R_out=A_L_in+A_self",
        },
    }


def assemble_grid(geo, name, nk, shift, gamma):
    _, kc, weights = k_grid(geo, nk, shift=shift)
    vv, vc, en = occup_empty_bases(h8(geo, R0, kc, 1), MU)
    x = x_from_bases(current_ops(geo, R0, kc, 1), vv, vc)
    gaps = en[:, 4:, None] - en[:, None, :4]
    gap, c = spectral_data(gaps, x, weights)
    d = np.einsum("k,kab->ab", weights, mean_d_from_bases(d_ops(geo, R0, kc, 1), vv))
    k0 = kernel_off_cuts(d, gap, c, 0.0)[0]
    min_gap = float(np.min(gap))
    max_gap = float(np.max(gap))
    omegas = [
        0.2 * min_gap, 0.5 * min_gap, 0.9 * min_gap,
        1.1 * min_gap, 2.0 * min_gap, min(1.0, 0.8 * max_gap),
    ]
    rows = []
    for w in omegas:
        below = bool(w < min_gap)
        k_inc = k0 + increment_over_z(gap, c, w)[0] * w
        k_raw = kernel_off_cuts(d, gap, c, w)[0]
        k_ret = kernel_off_cuts(d, gap, c, w + 1j * ETA)[0]
        jump, _, _ = current_jump(d, gap, c, w, ETA)
        rec = {
            "omega": w,
            "omega_over_min_gap": w / min_gap,
            "below_gap": below,
            "K_increment_minus_K_raw_maxabs": float(np.max(np.abs(k_inc - k_raw))),
        }
        # Subgap: unbroadened Hermitian K, C_I = 0. Continuum: K^R and the jump.
        if below:
            ledgers = (("increment", k_inc, np.zeros((2, 2), complex)),
                       ("raw", k_raw, np.zeros((2, 2), complex)))
        else:
            ledgers = (("increment", k_inc, jump),
                       ("retarded", k_ret, jump))
        for ledger, k, ci in ledgers:
            s, g, f = even_maps(k, w, gamma)
            block = identity_matrices(s, g, ci, w, gamma)
            detf = complex(np.linalg.det(f))
            block["F_det"] = {"real": detf.real, "imag": detf.imag}
            rec[ledger] = block
        rows.append(rec)
        key = "increment" if below else "retarded"
        print(name, f"w/gap={w/min_gap:.3f}",
              "U", f"{rec[key]['unitary_residual']:.3e}",
              "can", f"{rec[key]['canonical_ss_residual']:.3e}",
              flush=True)
    sub = [r for r in rows if r["below_gap"]]
    above = [r for r in rows if not r["below_gap"]]
    return {
        "grid": name, "n_k": int(len(kc)), "shift": shift,
        "min_direct_gap": min_gap, "max_direct_gap": max_gap,
        "eta_for_jump": ETA,
        "samples": rows,
        "subgap_increment_max_unitary": max(r["increment"]["unitary_residual"] for r in sub),
        "subgap_increment_max_canonical": max(r["increment"]["canonical_ss_residual"] for r in sub),
        "continuum_retarded_max_canonical": max(r["retarded"]["canonical_ss_residual"] for r in above),
        "continuum_retarded_max_unitary": max(r["retarded"]["unitary_residual"] for r in above),
        "subgap_raw_max_unitary": max(r["raw"]["unitary_residual"] for r in sub),
    }


def main():
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    rows = []
    for name, nk, shift in GRIDS:
        rec = assemble_grid(geo, name, nk, shift, gamma)
        rows.append(rec)
        print(name, "subU", rec["subgap_increment_max_unitary"],
              "subC", rec["subgap_increment_max_canonical"],
              "contC", rec["continuum_retarded_max_canonical"], flush=True)
    payload = {
        "status": "p1_zero_two_port_canonical_io_commutator_audit_only",
        "definition": {
            "ports": "A_R_out=A_L_in+A_self, A_L_out=A_R_in+A_self",
            "odd": "A_-_out=-A_-_in",
            "even_S": "2 F^{-1}-I",
            "even_G": "sqrt(2) (i gamma/z) F^{-1}",
            "C_I": "i (K^R-K^A)",
            "identity": "S S^dagger + (omega/gamma) G C_I G^dagger = I_2",
            "not_photon_number": True,
        },
        "gamma_self": gamma,
        "grids": rows,
        "blocked": [
            "n,m,V", "vacuum covariance", "collection packet",
            "Pauli", "initial state", "P1 drive",
        ],
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_gsheet.py",
                REPO / "tools/b0_small_model/q8_m0_stage1al_pole_audit.py",
                REPO / "tools/b0_small_model/q8_m0_stage1ak_intrinsic_noise.py",
            )
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "canonical_io.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
