#!/usr/bin/env python3
"""Stage-1C diagnostics: one-pass loop gain vs candidate linear resummation.

Not a production quantum-output closure. Does not replace Born n,m,V.
Does not invent G_self (sheet radiation reaction) as Maxwell χ.
R^A is real-linear, so a scalar (1-χ(ω))^{-1} is the wrong language.
"""
from __future__ import annotations

import numpy as np

from q8_m0_k1_kernel import trapz_weight
from q8_m0_k1_packet import packet_on_grid
from q8_m0_mode_io import (
    eta2_ncell,
    mode_overlap,
    mode_waveforms,
    response_to_complex_scalar,
)
from q8_m0_s1_kernel import pack_complex, parallel_perp_basis

LOOP_ID = "stage1c_loopgain_diagnostic_20260921"
RESUMMATION_STATUS = "expectation_feedback_diagnostic_not_quantum_main"


class ResummationNotChosenError(RuntimeError):
    """Raised if code tries to write production V(θ) from a resummed map."""


def apply_resummation_to_output(*_a, **_k):
    raise ResummationNotChosenError(
        "R^A+ξ resolvent is an expectation-value diagnostic, not the quantum "
        "main theory. Production V(θ) stays blocked. Do not write a "
        "production ledger from the resolvent."
    )


def F_from_phi(times, omega, phi) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    omega = np.asarray(omega, dtype=float)
    phi = np.asarray(phi)
    wo = np.sqrt(np.maximum(omega, 0.0))
    dens = np.zeros_like(phi)
    mask = wo > 0.0
    dens[mask] = phi[mask] / wo[mask]
    phase = np.exp(1j * times[:, None] * omega[None, :])
    return np.trapezoid(dens[None, :] * phase, omega, axis=1)


def gram_schmidt_packet(omega0: float, tp: float, n_main: int = 2, n_other: int = 3) -> dict:
    """Frequency-orthogonal partner of the H2 packet, same ω grid."""
    main = packet_on_grid(omega0, tp, n_harm=n_main)
    other = packet_on_grid(omega0, tp, n_harm=n_other)
    omega = np.asarray(main["omega"], dtype=float)
    p2 = np.asarray(main["phi"])
    p3 = np.asarray(other["phi"])
    ov = complex(np.trapezoid(np.conj(p2) * p3, omega))
    raw = p3 - ov * p2
    mass = float(np.trapezoid(np.abs(raw) ** 2, omega))
    if mass <= 1e-18:
        raise RuntimeError("orthogonal packet has no L2 mass")
    phi_o = raw / np.sqrt(mass)
    mass_o = float(np.trapezoid(np.abs(phi_o) ** 2, omega))
    ov_after = complex(np.trapezoid(np.conj(p2) * phi_o, omega))
    return {
        "omega": omega,
        "phi_main": p2,
        "phi_orth": phi_o,
        "overlap_before": pack_complex(ov),
        "overlap_after": pack_complex(ov_after),
        "orth_l2": mass_o,
        "n_main": int(n_main),
        "n_other": int(n_other),
    }


def shape_leftover(k_a, dmu, dt: float, ehat=None) -> dict:
    """Split cartesian leakage vs time-shape mismatch on ê_mode.

    dμ is complex = R[ê Re F] + i R[ê Im F], each piece a real current.
    """
    k_a = np.asarray(k_a)
    dmu = np.asarray(dmu)
    wt = trapz_weight(k_a.shape[0], dt)
    if ehat is None:
        mag = np.linalg.norm(np.real(k_a), axis=1)
        i = int(np.argmax(mag))
        ehat = np.real(k_a[i])
        nrm_e = float(np.linalg.norm(ehat))
        ehat = ehat / nrm_e if nrm_e > 1e-18 else np.array([1.0, 0.0])
    ehat = np.asarray(ehat, dtype=float)
    ehat = ehat / float(np.linalg.norm(ehat))

    def nrm2(sig):
        sig = np.asarray(sig, dtype=float)
        if sig.ndim == 1:
            return float(np.einsum("t,t->", wt, sig ** 2))
        return float(np.einsum("t,t->", wt, np.sum(sig ** 2, axis=1)))

    d_re = np.real(dmu)
    d_im = np.imag(dmu)
    par_re = d_re @ ehat
    par_im = d_im @ ehat
    cart_re = nrm2(par_re) / max(nrm2(d_re), 1e-30)
    cart_im = nrm2(par_im) / max(nrm2(d_im), 1e-30)
    fr, fi = np.real(k_a) @ ehat, np.imag(k_a) @ ehat

    def time_leftover(par, f0, f1):
        g00 = float(np.einsum("t,t->", wt, f0 * f0))
        g01 = float(np.einsum("t,t->", wt, f0 * f1))
        g11 = float(np.einsum("t,t->", wt, f1 * f1))
        rhs = np.array([
            float(np.einsum("t,t->", wt, f0 * par)),
            float(np.einsum("t,t->", wt, f1 * par)),
        ])
        gram = np.array([[g00, g01], [g01, g11]])
        try:
            c = np.linalg.solve(gram + 1e-18 * np.eye(2), rhs)
        except np.linalg.LinAlgError:
            c = np.zeros(2)
        proj = c[0] * f0 + c[1] * f1
        return float(np.sqrt(nrm2(par - proj) / max(nrm2(par), 1e-30)))

    return {
        "cartesian_along_ehat_Re_drive": float(np.sqrt(max(cart_re, 0.0))),
        "cartesian_along_ehat_Im_drive": float(np.sqrt(max(cart_im, 0.0))),
        "time_shape_leftover_Re_drive": time_leftover(par_re, fr, fi),
        "time_shape_leftover_Im_drive": time_leftover(par_im, fr, fi),
        "note": (
            "cartesian_along_ehat is ||ê·δμ||/||δμ||. time_shape_leftover is "
            "the ê component outside span{Re F, Im F}. High leftover means "
            "one-mode 2x2 is not equivalent to full-time Volterra."
        ),
    }


def feedback_2x2(A2, B2) -> dict:
    """Real-linear loop on (b, b†). Not complex-linear (1-χ)^{-1}."""
    A2 = complex(A2)
    B2 = complex(B2)
    # Map v = (α, α*) -> (A2 α + B2 α*, A2* α* + B2* α)
    M = np.array([[A2, B2], [np.conj(B2), np.conj(A2)]], dtype=complex)
    evals = np.linalg.eigvals(M)
    I = np.eye(2, dtype=complex)
    det_I_M = complex(np.linalg.det(I - M))
    rho = float(np.max(np.abs(evals)))
    p = 1.0 - A2
    delta = abs(p) ** 2 - abs(B2) ** 2
    return {
        "A2": pack_complex(A2),
        "B2": pack_complex(B2),
        "spectral_radius": rho,
        "eigenvalues": [pack_complex(z) for z in evals],
        "det_1_minus_M": pack_complex(det_I_M),
        "invertible_2x2": bool(abs(det_I_M) > 1e-12),
        "scalar_1_minus_A2_abs": float(abs(p)),
        "Delta_|1-A2|^2_|B2|^2": float(delta),
        "partial_O4_1_plus_|A2|^2_minus_|B2|^2": float(1.0 + abs(A2) ** 2 - abs(B2) ** 2),
        "not_a_predicted_commutator": True,
        "complex_linear_1_minus_chi_wrong": True,
        "series_radius_lt_1": bool(rho < 1.0),
    }


def _coeffs_sample_equals_in_plus_rad(A2, B2):
    """Convention A: A_sample ~ A_in + A_rad, x = b_rad, α_tot = b_in + x."""
    A2 = complex(A2)
    B2 = complex(B2)
    p = 1.0 - A2
    delta = abs(p) ** 2 - abs(B2) ** 2
    if abs(delta) < 1e-18:
        raise RuntimeError("2x2 convention A is singular")
    c_s = np.conj(p) / delta
    c_sd = B2 / delta
    c_b = (np.conj(p) * A2 + abs(B2) ** 2) / delta
    c_bd = (np.conj(p) * B2 + B2 * np.conj(A2)) / delta
    return c_s, c_sd, 1.0 + c_b, c_bd, delta


def _coeffs_sample_equals_out(A2, B2):
    """Convention B: A_sample ~ A(b_out), (1-A2) b_out - B2 b_out† = b_in + s1."""
    A2 = complex(A2)
    B2 = complex(B2)
    p = 1.0 - A2
    delta = abs(p) ** 2 - abs(B2) ** 2
    if abs(delta) < 1e-18:
        raise RuntimeError("2x2 convention B is singular")
    # b_out = (1/Δ)[ p* (b_in+s1) + B2 (b_in+s1)† ]
    c_s = np.conj(p) / delta
    c_sd = B2 / delta
    s_coeff = np.conj(p) / delta          # of b_in
    t_coeff = B2 / delta                  # of b_in†
    return c_s, c_sd, s_coeff, t_coeff, delta


def vacuum_moments_from_coeffs(c_s, c_sd, S, T, n_src, m_src, comm_s) -> dict:
    n_src = complex(n_src)
    m_src = complex(m_src)
    comm_s = complex(comm_s)
    ss_dag = n_src + comm_s
    n = (
        abs(c_s) ** 2 * n_src
        + abs(c_sd) ** 2 * ss_dag
        + 2.0 * np.real(np.conj(c_s) * c_sd * np.conj(m_src))
        + abs(T) ** 2
    )
    # ⟨b_out b_out⟩
    # b_out = c_s s + c_sd s† + S b + T b†
    m = (
        c_s ** 2 * m_src
        + c_sd ** 2 * np.conj(m_src)
        + 2.0 * c_s * c_sd * ss_dag  # ⟨s s†⟩ mixed; actually ⟨s s†⟩ and ⟨s† s⟩
        + 2.0 * S * T * 1.0  # ⟨b b†⟩=1 term from S b * T b†? ⟨(S b)(T b†)⟩=0, ⟨(T b†)(S b)⟩=0, ⟨T b† T b†⟩=0
    )
    # redo m carefully:
    # ⟨(c_s s + c_sd s† + S b + T b†)^2⟩
    # ss: c_s^2 ⟨s s⟩ + c_sd^2 ⟨s† s†⟩ + c_s c_sd ⟨s s†⟩ + c_sd c_s ⟨s† s⟩
    # bb: S^2 ⟨b b⟩=0 + T^2 ⟨b† b†⟩=0 + S T ⟨b b†⟩ + T S ⟨b† b⟩ = S T
    m = (
        (c_s ** 2) * m_src
        + (c_sd ** 2) * np.conj(m_src)
        + c_s * c_sd * ss_dag
        + c_sd * c_s * n_src
        + S * T
    )
    comm_out = (abs(S) ** 2 - abs(T) ** 2) + (abs(c_s) ** 2 - abs(c_sd) ** 2) * comm_s
    theta = np.linspace(0.0, np.pi, 181, endpoint=True)
    v = 0.5 + float(np.real(n)) + np.real(np.exp(-2j * theta) * m)
    return {
        "n": pack_complex(n),
        "m": pack_complex(m),
        "commutator_cnumber": pack_complex(comm_out),
        "commutator_abs_minus_1": float(abs(comm_out - 1.0)),
        "S": pack_complex(S),
        "T": pack_complex(T),
        "c_s": pack_complex(c_s),
        "c_sd": pack_complex(c_sd),
        "V_min": float(np.min(v)),
        "V_max": float(np.max(v)),
        "kind": "algebraic_2x2_diagnostic_not_production",
    }


def algebraic_resummation_diagnostics(A2, B2, n_src, m_src, comm_s) -> dict:
    cA = _coeffs_sample_equals_in_plus_rad(A2, B2)
    cB = _coeffs_sample_equals_out(A2, B2)
    return {
        "status": RESUMMATION_STATUS,
        "convention_A_sample_equals_in_plus_rad": vacuum_moments_from_coeffs(
            cA[0], cA[1], cA[2], cA[3], n_src, m_src, comm_s
        ),
        "convention_B_sample_equals_out": vacuum_moments_from_coeffs(
            cB[0], cB[1], cB[2], cB[3], n_src, m_src, comm_s
        ),
        "loop_2x2": feedback_2x2(A2, B2),
        "note": (
            "Two sample-field conventions are not equivalent. Neither is approved. "
            "G_self vs collection K is unset. Use only as Stage-1C evidence."
        ),
    }


def mode_matrix_elements(x, w, mean_d, F_i, F_j, ehat, dt, pref) -> dict:
    """A_ij = i pref ∫ F_i ê · R[ê F_j*], B_ij = i pref ∫ F_i ê · R[ê F_j]."""
    k_i = np.asarray(F_i)[:, None] * np.asarray(ehat)[None, :]
    resp_j = response_to_complex_scalar(x, w, F_j, ehat, dt, mean_d)
    resp_js = response_to_complex_scalar(x, w, np.conj(F_j), ehat, dt, mean_d)
    beta_j = mode_overlap(k_i, resp_j["total"], dt)
    beta_js = mode_overlap(k_i, resp_js["total"], dt)
    return {
        "A": 1j * pref * beta_js,
        "B": 1j * pref * beta_j,
        "dmu_j": resp_j["total"],
    }


def causal_one_mode_cnumber(x, w, mean_d, F, ehat, dt, pref, alpha: complex = 1.0) -> dict:
    """Left-rect one-mode self-consistency for a c-number α. Convention A.

    h(t) = [F*(t) α_tot + F(t) α_tot*] ê,  α_tot = α + β_past
    dβ = i pref (F ê · I[h]) dt
    Compare β_final to A2 α + B2 α*.
    """
    from q8_m0_response import contact_dmu

    F = np.asarray(F)
    ehat = np.asarray(ehat, dtype=float)
    x = np.asarray(x)
    w = np.asarray(w, dtype=float)
    nt, nk = x.shape[0], x.shape[1]
    z = np.zeros((nk, 4, 4), dtype=complex)
    beta = 0.0 + 0.0j
    h_hist = np.zeros((nt, 2), dtype=float)
    for n in range(nt):
        at = alpha + beta
        scalar = np.conj(F[n]) * at + F[n] * np.conj(at)
        h = float(np.real(scalar)) * ehat
        h_hist[n] = h
        tr = np.einsum("kacv,kcv->ka", np.conj(x[n]), z)
        dmu_state = -2.0 * np.imag(w @ tr)
        dmu = dmu_state + contact_dmu(mean_d[n] if np.asarray(mean_d).ndim == 3 else mean_d, h[None, :])[0]
        dbeta = 1j * pref * complex(np.dot(F[n] * ehat, dmu)) * dt
        beta += dbeta
        z += dt * np.einsum("a,kacv->kcv", h, x[n])
    return {
        "alpha": pack_complex(alpha),
        "beta_final": pack_complex(beta),
        "kind": "causal_one_mode_cnumber_convention_A_not_production",
    }


def port_O4_piece(A_cross, B_cross) -> float:
    return float(abs(A_cross) ** 2 - abs(B_cross) ** 2)
