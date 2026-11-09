#!/usr/bin/env python3
"""P1=0 intrinsic current-noise spectral audit.

Unsymmetrized C^> / C^< of the frozen insulating M0 model, plus the
spectral commutator identity implied by K = -R^kappa.

Fourier convention matches the existing retarded test:
  S(ω) = ∫ dt e^{+i ω t} C(t)
so emission C^>(t) ~ e^{-i Δ t} has support at ω = +Δ.

No photons, Maxwell feedback, poles, Pauli, or initial-state change.
The prescribed-input cross term is recorded as zero by construction of
an independent product reference, not as a self-consistent result.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np

from q8_m0_classical import k_grid
from q8_m0_stage1ag_acceptance_audit import (
    REPO, MU, R0, build_geometry, h8, current_ops, occup_empty_bases,
    x_from_bases, d_ops, mean_d_from_bases,
)
from q8_m0_stage1ai_lowfreq_kernel import OMEGA0_OVER_TH, pack_matrix
from q8_m0_stage1ai_lowfreq_audit import spectral_data

OUT = REPO / "results/q_line/stage1ak_intrinsic_noise_20260924"
GRIDS = (
    ("Gamma_96x96", 96, 0.0),
    ("Gamma_128x128", 128, 0.0),
    ("Gamma_192x192", 192, 0.0),
    ("halfshift_192x192", 192, 0.5),
)
ETA = 1e-4
SUBGAP_OVER_OMEGA0 = (1e-6, 1e-4, 1e-3, 1e-2, 3e-2)


def kernel_any(d, gap, c, z):
    """Same spectral sum as K(z), allowed in either half-plane off the cuts."""
    z = complex(z)
    return (np.asarray(d, complex)
            + np.einsum("abp,p->ab", c, 1 / (z - gap))
            - np.einsum("abp,p->ab", c.conj(), 1 / (z + gap)))


def lorentzian_S(gap, c, omega, eta, *, greater):
    """2π δ_η broadened S^> (ω=+Δ) or S^< (ω=-Δ)."""
    omega = float(omega)
    eta = float(eta)
    if greater:
        return np.einsum("abp,p->ab", c, 2.0 * eta / ((omega - gap)**2 + eta**2))
    return np.einsum("abp,p->ab", c.conj(), 2.0 * eta / ((omega + gap)**2 + eta**2))


def assemble_grid(geo, name, nk, shift):
    _, kc, weights = k_grid(geo, nk, shift=shift)
    vv, vc, en = occup_empty_bases(h8(geo, R0, kc, 1), MU)
    x = x_from_bases(current_ops(geo, R0, kc, 1), vv, vc)
    gaps = en[:, 4:, None] - en[:, None, :4]
    gap, c = spectral_data(gaps, x, weights)
    d = np.einsum("k,kab->ab", weights, mean_d_from_bases(d_ops(geo, R0, kc, 1), vv))
    min_gap = float(np.min(gap))
    max_gap = float(np.max(gap))
    n_trans = int(gap.size)
    below = int(np.count_nonzero(gap < min_gap - 1e-18))
    equal_time = np.einsum("abp->ab", c)
    # Unbroadened support: no poles inside (0, min_gap).
    subgap = []
    for r in SUBGAP_OVER_OMEGA0:
        w = r * OMEGA0_OVER_TH
        if w >= min_gap:
            continue
        subgap.append({
            "omega": w,
            "omega_over_omega0": r,
            "S_greater_unbroadened_maxabs": 0.0,
            "S_lesser_unbroadened_maxabs": 0.0,
            "n_poles_at_this_frequency": 0,
        })
    identity = []
    probes = [0.0, 0.5 * min_gap, 0.9 * min_gap, 1.1 * min_gap, 2.0 * min_gap, 1.0]
    for w in probes:
        sg = lorentzian_S(gap, c, w, ETA, greater=True)
        sl = lorentzian_S(gap, c, w, ETA, greater=False)
        kr = kernel_any(d, gap, c, w + 1j * ETA)
        ka = kernel_any(d, gap, c, w - 1j * ETA)
        lhs = kr - ka
        rhs = -1j * (sg - sl)
        identity.append({
            "omega": float(w),
            "omega_over_min_gap": float(w / min_gap),
            "below_gap": bool(abs(w) < min_gap),
            "S_greater_broadened": pack_matrix(sg),
            "S_lesser_broadened": pack_matrix(sl),
            "K_retarded_minus_K_advanced": pack_matrix(lhs),
            "minus_i_Sgreater_minus_Slesser": pack_matrix(rhs),
            "identity_maxabs": float(np.max(np.abs(lhs - rhs))),
            "S_greater_trace": float(np.trace(sg).real),
            "S_lesser_trace": float(np.trace(sl).real),
        })
    return {
        "grid": name,
        "n_k": int(len(kc)),
        "shift": shift,
        "n_transitions": n_trans,
        "min_direct_gap": min_gap,
        "max_direct_gap": max_gap,
        "n_transitions_strictly_below_min_gap": below,
        "equal_time_C_greater": pack_matrix(equal_time),
        "integrated_S_greater_over_2pi": pack_matrix(equal_time),
        "subgap_unbroadened": subgap,
        "spectral_commutator_eta": ETA,
        "spectral_identity": identity,
        "identity_maxabs_all_probes": max(row["identity_maxabs"] for row in identity),
        "max_S_greater_trace_below_gap_broadened": max(
            row["S_greater_trace"] for row in identity if row["below_gap"]
        ),
    }


def main():
    geo = build_geometry()
    rows = []
    for name, nk, shift in GRIDS:
        rec = assemble_grid(geo, name, nk, shift)
        rows.append(rec)
        print(name,
              "min_gap", rec["min_direct_gap"],
              "below", rec["n_transitions_strictly_below_min_gap"],
              "identity", rec["identity_maxabs_all_probes"],
              "S>_subgap_broad", rec["max_S_greater_trace_below_gap_broadened"],
              flush=True)
    payload = {
        "status": "p1_zero_intrinsic_current_noise_spectral_audit_only",
        "definition": {
            "C_greater": "⟨δI_a(t) δI_b(0)⟩ per cell, no extra -i",
            "C_lesser": "⟨δI_b(0) δI_a(t)⟩ = C_greater_ba(-t)",
            "S": "∫ dt e^{+i omega t} C(t), matching the retarded test",
            "K": "-R^kappa",
            "identity": "K(ω+iη)-K(ω-iη) = -i (S^>_η - S^<_η)",
        },
        "units": {
            "frequency": "hbar omega / t_h",
            "C": "current^2 per cell",
        },
        "omega0": OMEGA0_OVER_TH,
        "physical_linewidth": 0,
        "broadening_eta_for_identity_only": ETA,
        "grids": rows,
        "prescribed_independent_input_cross_term": {
            "connected_value": 0.0,
            "status": "zero_by_construction_of_product_reference",
            "not_a_self_consistent_Maxwell_result": True,
            "note": (
                "Electrons and the prescribed input field occupy independent "
                "factors. A connected ⟨δI_int δκ_in⟩ contraction does not exist "
                "in this control. Feedback A_self ↔ I reopens the term."
            ),
        },
        "blocked": [
            "self-consistent Maxwell",
            "coupled-mode poles",
            "full E_out",
            "collection packet",
            "n,m,V,S_min",
            "Pauli redefinition",
            "bare vs dressed vacuum",
        ],
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_stage1ai_lowfreq_audit.py",
                REPO / "tools/b0_small_model/q8_m0_response.py",
            )
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "intrinsic_noise.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
