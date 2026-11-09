#!/usr/bin/env python3
"""Radiation slot chi_rad. Distinct from electronic response R^A.

Physical form of chi_rad is still a theory node. A 2026-09-18 vacuum-sheet
candidate is documented here and is not approved for photon-number evaluation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

CHI_STATUS = "unset_awaiting_theory"
CHI_ID = "chi_2d_sheet_to_freespace_mode_slot"
CHI_RAD_SLOT = "chi_rad"


@dataclass(frozen=True)
class RadiationKernelChi:
    """K_a(t) involves chi_rad(ω) φ(ω), once a theory form is supplied."""

    ident: str = CHI_ID
    slot_name: str = CHI_RAD_SLOT
    status: str = CHI_STATUS
    form: str = "not_chosen"
    gauge: str = "unset"
    green_function: str = "unset"
    sheet_to_3d: str = "unset"
    numerical_aperture: str = "unset"
    n_cell: str = "unset_not_equal_to_nk"
    spatial_area_S: str = "unset"
    units: str = "unset"
    frequency_dependence: str = "unset"
    not_electronic_response_R_A: bool = True
    note: str = (
        "Radiation interface only. Do not treat a constant g as derived chi_rad. "
        "R^A lives in q8_m0_response.py."
    )


class ChiNotChosenError(RuntimeError):
    """Raised if code tries to turn d_φ into photon numbers without chi_rad."""


def chi_rad_candidate_vacuum_sheet() -> dict:
    """Documented 2026-09-18 candidate. Not production, not photon numbers."""
    return {
        "status": "documented_candidate_not_approved_for_photon_eval",
        "geometry": "uniform_optically_thin_sheet_vacuum_both_sides_q0_normal",
        "E_emit": "E_emit(z,t) = -(Z0/2) J_sheet(t-|z|/c)",
        "a_src": "a_src(ω) = i sqrt(Z0/(4π ħ ω S)) I_tilde(ω),  I = S J_sheet",
        "K_t": "K(t) = i sqrt(Z0/(4π ħ S)) ∫_0^∞ dω φ(ω)/sqrt(ω) e^{+iωt}",
        "frequency_dependence": "1/sqrt(ω)_retained_not_pulled_out_as_constant",
        "S": "unset",
        "n_cell": "unset_not_equal_to_nk",
        "collection": "one_sided_detection_after_two_port_feedback_see_q8_m0_gsheet",
        "time_support": "electronic_data_ends_at_pulse_end_not_all_outgoing_time",
        "current_order": "zero_order_classical_pump_current_vs_full_coupled_current_unset",
        "evaluated": False,
        "photon_output": "not_run",
    }


def chi_spec() -> dict:
    rec = asdict(RadiationKernelChi())
    rec["ready_to_evaluate_photons"] = False
    rec["photon_output"] = "not_run"
    rec["g2"] = "not_run"
    rec["candidate_vacuum_sheet"] = chi_rad_candidate_vacuum_sheet()
    rec["electronic_response_slot"] = "R_A_separate_see_q8_m0_response"
    rec["shape_kernel_1_over_sqrt_omega"] = "implemented_finite_time_not_photon_number"
    rec["si_prefactor"] = "unset"
    rec["time_window"] = "finite_pulse_[0,Tp]_validation_not_farfield"
    return rec


def apply_chi_to_shape_kernel(_omega, _psi_or_dphi, *, _chi: RadiationKernelChi | None = None):
    """Blocked on purpose. A later approved chi_rad object would live here."""
    raise ChiNotChosenError(
        "chi_rad physical form is unset. Shape-only d_φ is not a photon number."
    )


def output_placeholder(d_phi, *, mean_b_phi=None) -> dict:
    """Ledger row: what chi_rad would multiply. Does not evaluate ⟨n⟩."""
    return {
        "shape_source_d_phi": d_phi,
        "shape_mean_b_phi": mean_b_phi,
        "chi_rad": chi_spec(),
        "chi": chi_spec(),
        "would_compute_if_chi_rad_and_Ncell_set": [
            "mean_field_in_mode",
            "source_contribution_to_mean_photon_number_O_lambda2_independent_input_vacuum",
            "some_second_order_covariances_from_C_JJ",
        ],
        "same_order_may_need_R_A": [
            "phase_sensitive_anomalous_moments",
            "normal_moments_with_non_vacuum_input",
        ],
        "not_available_from_C_JJ_alone": [
            "g2_without_extra_assumption",
            "squeezing_spectrum",
            "entanglement",
        ],
        "evaluated": False,
        "photon_output": "not_run",
        "gaussian_closure": "not_chosen",
    }
