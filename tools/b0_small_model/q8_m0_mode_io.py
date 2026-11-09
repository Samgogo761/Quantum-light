#!/usr/bin/env python3
"""Same-mode A_in ↔ K_out for the H2⊥ packet. Finite [0,Tp] validation.

Ideal uniform film with mode area matching the sample area. Connected
second moments use η² N_cell = α_fs L0² / A_uc; S is not a free number.
Coherent ⟨β⟩ still scales as √N_cell. nk is not N_cell.

Working formula is the documented vacuum-sheet candidate, used here as a
finite-time I/O convention. Not far-field photon numbers, not Maxwell χ.
"""
from __future__ import annotations

import numpy as np

from q8_m0_k1_kernel import k_shape_on_times, k_vector_perp, source_beta_nm, trapz_weight
from q8_m0_response import predict_delta_mu
from q8_m0_s1_kernel import pack_complex, parallel_perp_basis

# CODATA 2018
ALPHA_FS = 7.2973525693e-3
L0_ANG = 1.0
TIME_WINDOW = "finite_pulse_[0,Tp]_validation_not_farfield"
MODE_IO_ID = "same_mode_H2perp_A_in_K_out_20260921"

# Accounting parameter only: δb_out = b_in + λ s1 + λ² (A2 b_in + B2 b† + …)
LAMBDA_ORDER_NOTE = (
    "λ bookkeeping: s1 = O(λ) from K~η and I^0; A2,B2 = O(λ²) from one-pass K R A_in. "
    "Connected n,m kept at O(λ²) Born. |A2|² and |B2|² are O(λ⁴) and are not a "
    "prediction of the full commutator. Resummation of G^R R^A is a candidate, not production."
)


def eta2_ncell(geo) -> float:
    """η² N_cell = α_fs L0² / A_uc. S = N_cell A_uc cancels. nk is not N_cell."""
    return float(ALPHA_FS * (L0_ANG ** 2) / float(geo.a_cell_ang2))


def area_scaling_table(geo) -> dict:
    pref = eta2_ncell(geo)
    return {
        "geometry": "ideal_uniform_film_mode_area_matches_sample",
        "S": "N_cell * A_uc_not_a_free_parameter",
        "N_cell": "unset_not_equal_to_nk",
        "A_uc_ang2": float(geo.a_cell_ang2),
        "L0_ang": L0_ANG,
        "alpha_fs": ALPHA_FS,
        "eta2_Ncell": pref,
        "eta2_Ncell_formula": "alpha_fs * L0^2 / A_uc",
        "connected_n_m": {
            "I_total_mean": "N_cell * I_cell",
            "connected_source": "N_cell * d_cell",
            "field_norm": "S^{-1/2}",
            "combination": "N_cell / S = 1 / A_uc",
            "prefactor": "eta2_Ncell = alpha_fs * L0^2 / A_uc",
            "S_cancels": True,
        },
        "coherent_beta": {
            "scales_as": "sqrt(N_cell) = sqrt(S / A_uc)",
            "S_cancels": False,
            "intensive_report": "beta / sqrt(N_cell) = i * sqrt(eta2_Ncell) * beta_F",
        },
        "nk_is_not_N_cell": True,
        "finite_spot_or_aperture": "not_this_geometry",
    }


def term_ledger_spec() -> dict:
    return {
        "lambda_bookkeeping": LAMBDA_ORDER_NOTE,
        "delta_b_out": "b_in + λ s1 + λ² (A2 b_in + B2 b_in_dagger + other_ports) + …",
        "initial_state": "electrons independent of vacuum input ports",
        "n_out_O_lambda2": "λ² ⟨s1† s1⟩ ; vacuum n_in = 0 ; |B2|² is O(λ⁴)",
        "m_out_O_lambda2": "λ² (⟨s1 s1⟩ + B2) ; B2 is vacuum-input × emission-response interference",
        "not_same_order": [
            "alpha_plus * alpha_minus",
            "|alpha_minus|^2",
            "using K_shape as A_in without eta",
        ],
        "contact": "inside R^A used for A2 and B2",
        "other_vacuum_ports": "allowed to contract at this order; not a closed single oscillator",
        "do_not_unit_convert_old_alpha_products_into_V": True,
        "bosonic_commutator_at_kept_order": (
            "2 Re A2 + ⟨[s1, s1†]⟩ = 0 is the O(λ²) identity. "
            "1+|A2|²-|B2|² is a partial O(λ⁴) piece, not a predicted [b,b†]."
        ),
        "approximation_name": "Born_one_pass_feedback",
    }


def f_and_psi(times, omega0: float, tp: float, n_harm: int = 2) -> dict:
    rec = k_shape_on_times(times, omega0, tp, n_harm=n_harm)
    rec["F"] = rec["K_shape"]
    rec["narrowband_1_over_sqrt_omega_c"] = 1.0 / np.sqrt(float(rec["omega_c"]))
    return rec


def f_vs_psi_shape_metrics(times, F, psi, omega_c: float, omega=None, phi=None) -> dict:
    """Compare F to psi/√ω_c. Not a photon-number error bar."""
    times = np.asarray(times, dtype=float)
    F = np.asarray(F)
    psi = np.asarray(psi)
    scale = 1.0 / np.sqrt(float(omega_c))
    psi_s = psi * scale
    mid = int(len(times) // 2)
    den_mid = max(abs(F[mid]), 1e-30)
    l2 = float(np.sqrt(np.trapezoid(np.abs(F - psi_s) ** 2, times)))
    l2F = float(np.sqrt(np.trapezoid(np.abs(F) ** 2, times)))
    inner = complex(np.trapezoid(np.conj(psi_s) * F, times))
    nrm = complex(np.trapezoid(np.abs(psi_s) ** 2, times))
    c_fit = inner / nrm if abs(nrm) > 1e-30 else 0.0
    l2_fit = float(np.sqrt(np.trapezoid(np.abs(F - c_fit * psi_s) ** 2, times)))
    out = {
        "omega_c": float(omega_c),
        "narrowband_scale_1_over_sqrt_omega_c": float(scale),
        "raw_midpoint_|F-psi|/|psi|": float(
            abs(F[mid] - psi[mid]) / max(abs(psi[mid]), 1e-30)
        ),
        "raw_midpoint_note": (
            "Dominated by the known scale ~√(2ω1)-1 when ω_c=0.1; "
            "not evidence of an O(1) shape change."
        ),
        "scaled_midpoint_|F-psi_s|/|F|": float(abs(F[mid] - psi_s[mid]) / den_mid),
        "scaled_full_window_L2_|F-psi_s|/||F||": l2 / max(l2F, 1e-30),
        "best_complex_constant_L2_residual_/||F||": l2_fit / max(l2F, 1e-30),
        "best_complex_constant": pack_complex(c_fit),
        "interpretation": "finite_bandwidth_correction_keep_true_1_over_sqrt_omega",
        "not_an_error_bar_on_n_m": True,
    }
    if omega is not None and phi is not None:
        omega = np.asarray(omega, dtype=float)
        phi = np.asarray(phi)
        wo = np.sqrt(np.maximum(omega, 0.0))
        dens = np.zeros_like(phi)
        mask = wo > 0.0
        dens[mask] = phi[mask] / wo[mask]
        dens_nb = np.zeros_like(phi)
        dens_nb[mask] = phi[mask] * scale
        l2w = float(np.sqrt(np.trapezoid(np.abs(dens - dens_nb) ** 2, omega)))
        l2d = float(np.sqrt(np.trapezoid(np.abs(dens) ** 2, omega)))
        out["frequency_line_L2_|phi_over_sqrt_omega-narrowband|/||F_omega||"] = (
            l2w / max(l2d, 1e-30)
        )
    return out


def mode_waveforms(times, omega0, tp, geo, n_harm: int = 2) -> dict:
    """F(t), real/imag test shapes along ê_⊥, and the i / conjugate ledger."""
    rec = f_and_psi(times, omega0, tp, n_harm=n_harm)
    _e_par, e_perp = parallel_perp_basis(geo)
    F = np.asarray(rec["F"])
    k_a = F[:, None] * e_perp[None, :]
    h_re = np.real(F)[:, None] * e_perp[None, :]
    h_im = np.imag(F)[:, None] * e_perp[None, :]
    rec.update({
        "e_perp": e_perp,
        "K_a_shape": k_a,
        "h_re_F": h_re,
        "h_im_F": h_im,
        "F_at_t0": pack_complex(F[0]),
        "abs_F_t0_over_max": float(abs(F[0]) / max(float(np.max(np.abs(F))), 1e-30)),
        "units": {
            "code": "hbar=e=t_h=1, length in Angstrom, kappa in Angstrom^{-1}",
            "F": "dimensionless in code frequency measure",
            "delta_kappa": "eta * (F* b + F b†) * e_perp",
            "K_full": "i * eta * F * e_perp ;  I is total current in S",
            "K_equals_i_over_hbar_u_star": True,
            "u_plus": "sqrt(hbar Z0 / (4 pi S)) * F*(t)  [A_in^(+) coefficient of b]",
        },
        "initial_state_if_F_t0_nonzero": (
            "Keep the field-free ground state of H0. Apply the test kappa from t=0. "
            "Do not add a silent envelope or change the initial electronic state."
        ),
    })
    rec["shape_metrics"] = f_vs_psi_shape_metrics(
        times, F, rec["psi"], rec["omega_c"], rec["omega"], rec["phi"]
    )
    return rec


def response_to_complex_scalar(x, w, f_complex, ehat, dt, mean_d) -> dict:
    """Real-linear R^A acting on (Re f) ê and (Im f) ê."""
    ehat = np.asarray(ehat, dtype=float)
    f = np.asarray(f_complex)
    hr = np.real(f)[:, None] * ehat[None, :]
    hi = np.imag(f)[:, None] * ehat[None, :]
    pr = predict_delta_mu(x, w, hr, dt, mean_d)
    pi = predict_delta_mu(x, w, hi, dt, mean_d)
    return {
        "total": pr["total"] + 1j * pi["total"],
        "state": pr["state"] + 1j * pi["state"],
        "contact": pr["contact"] + 1j * pi["contact"],
        "h_re": hr,
        "h_im": hi,
    }


def mode_overlap(k_a, dmu, dt: float) -> complex:
    wt = trapz_weight(k_a.shape[0], dt)
    return complex(np.einsum("t,ta,ta->", wt, k_a, dmu))


def sigma_commutator_per_cell(x, w, k_a, dt: float) -> dict:
    """Per-cell ⟨[Σ, Σ†]⟩ with Σ = ∫ K_a I_a dt.

    ⟨Σ† Σ⟩ = n_F uses C^>(t,t') with kernel F.
    ⟨Σ Σ†⟩_C> = n_{F*} uses the same greater function with F*.
    The current commutator in this X language is C^> − swapped, which
    reduces to the real difference n_{F*} − n_F.
    """
    from q8_m0_k1_kernel import project_current_kernel

    yk = project_current_kernel(x, k_a, dt)
    yks = project_current_kernel(x, np.conj(k_a), dt)
    n_F = complex(np.einsum("k,kcv,kcv->", w, np.conj(yk), yk))
    n_Fs = complex(np.einsum("k,kcv,kcv->", w, np.conj(yks), yks))
    return {
        "n_F": n_F,
        "n_Fstar": n_Fs,
        "commutator": n_Fs - n_F,
    }


def close_output_moments(x, w, mu_t, mean_d, times, dt, geo, prot) -> dict:
    """O(λ²) finite-time output n, m, V(θ). Six-k method, not a BZ claim."""
    rec = mode_waveforms(times, prot.omega0, prot.t_pulse, geo)
    F = np.asarray(rec["F"])
    e_perp = np.asarray(rec["e_perp"])
    k_a = np.asarray(rec["K_a_shape"])
    src = source_beta_nm(x, w, mu_t, k_a, dt)
    n_F = complex(src["n_source_shape"]["re"], src["n_source_shape"]["im"])
    m_F = complex(src["m_source_shape"]["re"], src["m_source_shape"]["im"])
    beta_F_src = complex(src["beta_shape"]["re"], src["beta_shape"]["im"])
    resp_F = response_to_complex_scalar(x, w, F, e_perp, dt, mean_d)
    resp_Fs = response_to_complex_scalar(x, w, np.conj(F), e_perp, dt, mean_d)
    beta_resp_F = mode_overlap(k_a, resp_F["total"], dt)
    beta_resp_Fs = mode_overlap(k_a, resp_Fs["total"], dt)
    beta_resp_F_state = mode_overlap(k_a, resp_F["state"], dt)
    beta_resp_Fs_state = mode_overlap(k_a, resp_Fs["state"], dt)
    beta_resp_F_contact = mode_overlap(k_a, resp_F["contact"], dt)
    beta_resp_Fs_contact = mode_overlap(k_a, resp_Fs["contact"], dt)
    pref = eta2_ncell(geo)
    # s1 = i η I_tot[F] ; ⟨s1† s1⟩ = +pref n_F ; ⟨s1 s1⟩ = i² pref m_F = −pref m_F
    n_source = pref * n_F
    m_source = -pref * m_F
    # A2 = i pref ∫ F ê · R[ê F*] ; B2 = i pref ∫ F ê · R[ê F]
    A2 = 1j * pref * beta_resp_Fs
    B2 = 1j * pref * beta_resp_F
    n_out = n_source
    m_out = m_source + B2
    comm_rec = sigma_commutator_per_cell(x, w, k_a, dt)
    comm_s = pref * comm_rec["commutator"]
    identity = 2.0 * np.real(A2) + comm_s
    A2_state = 1j * pref * beta_resp_Fs_state
    A2_contact = 1j * pref * beta_resp_Fs_contact
    bogoliubov_like = abs(1.0 + A2) ** 2 - abs(B2) ** 2
    theta = np.linspace(0.0, np.pi, 181, endpoint=True)
    v = 0.5 + float(np.real(n_out)) + np.real(np.exp(-2j * theta) * m_out)
    v_vac = np.full_like(v, 0.5)
    n_abs_m = float(np.real(n_out) - abs(m_out))
    gauss_det = float(np.real(n_out) * (np.real(n_out) + 1.0) - abs(m_out) ** 2)
    sqrt_pref = float(np.sqrt(pref))
    return {
        "mode_io_id": MODE_IO_ID,
        "time_window": TIME_WINDOW,
        "area": area_scaling_table(geo),
        "term_ledger": {
            **term_ledger_spec(),
            "n_in_vacuum": 0.0,
            "n_source_O_lambda2": pack_complex(n_source),
            "n_response_B2_dagger_B2_O_lambda4": "not_kept",
            "old_|alpha_minus|^2": "higher_order_not_used",
            "m_source_O_lambda2": pack_complex(m_source),
            "m_response_B2_O_lambda2": pack_complex(B2),
            "old_alpha_plus_alpha_minus": "higher_order_not_used",
            "contact_inside_B2_A2": True,
            "i_squared_on_anomalous_source": -1,
        },
        "prefactor_eta2_Ncell": pref,
        "n_F_per_cell_filter": pack_complex(n_F),
        "m_F_per_cell_filter": pack_complex(m_F),
        "beta_F_mean_filter": pack_complex(beta_F_src),
        "beta_resp_F": pack_complex(beta_resp_F),
        "beta_resp_Fstar": pack_complex(beta_resp_Fs),
        "A2": pack_complex(A2),
        "B2": pack_complex(B2),
        "n_out": pack_complex(n_out),
        "m_out": pack_complex(m_out),
        "beta_over_sqrt_Ncell": pack_complex(1j * sqrt_pref * beta_F_src),
        "beta_still_needs_sqrt_Ncell": True,
        "N_cell": "unset_not_equal_to_nk",
        "vacuum_V": 0.5,
        "V_theta": {
            "theta": theta.tolist(),
            "V": v.tolist(),
            "V_vacuum": v_vac.tolist(),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "theta_at_min": float(theta[int(np.argmin(v))]),
            "kind": "finite_time_bosonic_quadrature_variance_same_mode",
        },
        "commutator_check": {
            "expectation": "2 Re A2 + ⟨[s1,s1†]⟩ = 0 at kept order",
            "two_Re_A2": float(2.0 * np.real(A2)),
            "s1_commutator": pack_complex(comm_s),
            "residual": pack_complex(identity),
            "residual_abs": float(abs(identity)),
            "two_Re_A2_state_only": float(2.0 * np.real(A2_state)),
            "two_Re_A2_contact_only": float(2.0 * np.real(A2_contact)),
            "residual_state_plus_commutator_abs": float(
                abs(2.0 * np.real(A2_state) + comm_s)
            ),
            "residual_full_plus_commutator_abs": float(abs(identity)),
            "n_Fstar_minus_n_F_times_pref": pack_complex(comm_s),
            "n_Fstar_per_cell": pack_complex(comm_rec["n_Fstar"]),
            "bogoliubov_like_|1+A2|^2_|B2|^2": float(bogoliubov_like),
            "bogoliubov_like_plus_s1_commutator": float(bogoliubov_like + np.real(comm_s)),
            "abs_A2": float(abs(A2)),
            "abs_B2": float(abs(B2)),
            "kept_order_identity_residual_abs": float(abs(identity)),
            "kept_order_identity_closed": bool(abs(identity) < 1e-5),
            "unique_unclosed_item": (
                "O(λ²) identity 2 Re A2 + ⟨[s1,s1†]⟩ closes. "
                "1+|A2|²-|B2|² is a partial O(λ⁴) piece, not a predicted [b,b†]. "
                "|A2|~O(1) means omitted 4th-and-higher / other ports / resummation "
                "are not safely small. V stays on the O(λ²) Born ledger."
            ),
            "finite_Tp_not_all_outgoing_time": True,
            "other_ports_not_a_closed_oscillator": True,
            "do_not_clip_covariance": True,
        },
        "covariance_physicality": {
            "n_real": float(np.real(n_out)),
            "abs_m": float(abs(m_out)),
            "n_minus_abs_m": n_abs_m,
            "V_min": float(np.min(v)),
            "V_min_ge_0": bool(np.min(v) >= -1e-12),
            "n_nplus1_minus_abs_m2": gauss_det,
            "clipped": False,
        },
        "shape_metrics": rec["shape_metrics"],
        "F_at_t0": rec["F_at_t0"],
        "abs_F_t0_over_max": rec["abs_F_t0_over_max"],
        "squeezing_claim": "not_evaluated_sixk_method_only",
        "nk_not_BZ": True,
        "photon_output": "finite_time_same_mode_not_farfield",
        "g2": "not_run",
    }
