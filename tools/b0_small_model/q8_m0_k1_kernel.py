#!/usr/bin/env python3
"""Finite-time radiation kernel for the H2⊥ reference packet.

K_shape(t) = ∫_0^∞ dω φ(ω)/√ω exp(+iω t).
K_shape is F(t). The same-mode SI layer is q8_m0_mode_io.py:
K = i η F with η² N_cell = α_fs L0²/A_uc for connected moments.
These helpers stay in filter units. Not far-field photon numbers.
"""
from __future__ import annotations

import numpy as np

from q8_m0_k1_packet import packet_on_grid
from q8_m0_s1_kernel import pack_complex, parallel_perp_basis

TIME_WINDOW = "finite_pulse_[0,Tp]_validation_not_farfield"
SI_PREFACTOR = "unset_i_sqrt_Z0_over_4pi_hbar_S"


def k_shape_on_times(times, omega0: float, tp: float, n_harm: int = 2) -> dict:
    rec = packet_on_grid(omega0, tp, n_harm=n_harm)
    omega = np.asarray(rec["omega"], dtype=float)
    phi = np.asarray(rec["phi"])
    times = np.asarray(times, dtype=float)
    wo = np.sqrt(np.maximum(omega, 0.0))
    dens = np.zeros_like(phi)
    mask = wo > 0.0
    dens[mask] = phi[mask] / wo[mask]
    phase = np.exp(1j * times[:, None] * omega[None, :])
    k_shape = np.trapezoid(dens[None, :] * phase, omega, axis=1)
    psi = np.trapezoid(phi[None, :] * phase, omega, axis=1)
    return {
        "times": times,
        "K_shape": k_shape,
        "psi": psi,
        "omega": omega,
        "phi": rec["phi"],
        "omega_c": rec["omega_c"],
        "commutator_integral": rec["commutator_integral"],
        "si_prefactor": SI_PREFACTOR,
        "time_window": TIME_WINDOW,
        "not_q9": True,
    }


def k_vector_perp(times, omega0, tp, geo, n_harm: int = 2) -> dict:
    rec = k_shape_on_times(times, omega0, tp, n_harm=n_harm)
    _e_par, e_perp = parallel_perp_basis(geo)
    k_a = rec["K_shape"][:, None] * e_perp[None, :]
    rec["e_perp"] = e_perp
    rec["K_a"] = k_a
    rec["polarization"] = "e_perp_H2_reference"
    return rec


def trapz_weight(nt: int, dt: float) -> np.ndarray:
    w = np.full(nt, dt, dtype=float)
    if nt >= 2:
        w[0] *= 0.5
        w[-1] *= 0.5
    return w


def project_current_kernel(x, k_a, dt: float):
    """Y_K[k,cv] = ∫ K_a(t) X_a(k,t) dt. x: (nt,nk,2,4,4), k_a: (nt,2)."""
    wt = trapz_weight(x.shape[0], dt)
    return np.einsum("t,ta,tkacv->kcv", wt, k_a, x)


def source_beta_nm(x, w, mu_t, k_a, dt: float) -> dict:
    """Per-cell finite-time source projection. Not ⟨n⟩ in photons.

    β_shape = ∫ K_a ⟨I_a⟩ dt
    n_shape = Σ_k w Tr[Y_K† Y_K] = ⟨δs† δs⟩_cell
    m_shape = Σ_k w ⟨Y_{K*}, Y_K⟩_F for ⟨δs δs⟩
    """
    yk = project_current_kernel(x, k_a, dt)
    yk_star = project_current_kernel(x, np.conj(k_a), dt)
    wt = trapz_weight(x.shape[0], dt)
    beta = complex(np.einsum("t,ta,ta->", wt, k_a, mu_t))
    n_src = complex(np.einsum("k,kcv,kcv->", w, np.conj(yk), yk))
    m_src = complex(np.einsum("k,kcv,kcv->", w, np.conj(yk_star), yk))
    return {
        "beta_shape": pack_complex(beta),
        "n_source_shape": pack_complex(n_src),
        "m_source_shape": pack_complex(m_src),
        "kind": "finite_time_per_cell_kernel_units",
        "si_prefactor": SI_PREFACTOR,
        "N_cell": "unset_not_equal_to_nk",
        "photon_output": "not_run",
        "time_window": TIME_WINDOW,
    }


def response_mode_coefficients(pred_u, pred_uconj, k_a, dt: float) -> dict:
    """α_± from δ⟨I⟩ predicted for A_in shapes u and u*.

    pred_* are real δμ(t) for those classical test waveforms (already include
    contact). α are still kernel units.
    """
    wt = trapz_weight(k_a.shape[0], dt)
    alpha_plus = complex(np.einsum("t,ta,ta->", wt, k_a, pred_u))
    alpha_minus = complex(np.einsum("t,ta,ta->", wt, k_a, pred_uconj))
    return {
        "alpha_plus_shape": pack_complex(alpha_plus),
        "alpha_minus_shape": pack_complex(alpha_minus),
        "note": (
            "Linear-response map from in-mode A-shape to radiated kernel. "
            "Not a dimensionless bosonic α until SI prefactors of K and A_in share S."
        ),
    }
