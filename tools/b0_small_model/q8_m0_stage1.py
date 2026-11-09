#!/usr/bin/env python3
"""Stage-1 frozen theory interface (2026-09-21 human approval).

Strong classical P1 + weaker quantum EM perturbation.
Observables: β, n, m, V(θ). Not g2, entanglement, or pure BSV drive.
Time window: finite [0, Tp] validation, not far-field photon statistics.
"""
from __future__ import annotations

STAGE1_ID = "stage1_io_strong_P1_weak_quantum_perturbation_20260921"
STAGE1_STATUS = "stage1_gself_emergence_20260923"
THEORY_SCOPE = "strong_classical_P1_plus_dressed_ph_plus_two_port_photon_Gaussian"
PURE_BSV = "out_of_scope_until_validated"
OBSERVABLES = ("beta", "n_munu", "m_munu", "V_theta")
NOT_THIS_ROUND = ("g2", "entanglement_entropy", "non_gaussian_full", "pure_BSV_pump")
TIME_WINDOW = "finite_pulse_[0,Tp]_validation_not_farfield"
REFERENCE_MODE = "K1_R0_reference_perp_H2_pulse_matched"
NK192 = "not_run"
SQUEEZING_CLAIM = "not_evaluated_sixk_method_only"
BORN_LABEL = "O_lambda2_is_Born_perturbative_benchmark_not_main_theory"
THEORY_MAIN = "P1_dressed_particle_hole_photon_canonical_Gaussian"
EXPECTATION_RESOLVENT = "RA_xi_resolvent_classical_expectation_feedback_diagnostic"


def stage1_spec() -> dict:
    return {
        "stage1_id": STAGE1_ID,
        "status": STAGE1_STATUS,
        "born_label": BORN_LABEL,
        "theory_main": THEORY_MAIN,
        "expectation_resolvent": EXPECTATION_RESOLVENT,
        "theory_scope": THEORY_SCOPE,
        "pure_BSV": PURE_BSV,
        "observables": list(OBSERVABLES),
        "not_this_round": list(NOT_THIS_ROUND),
        "time_window": TIME_WINDOW,
        "reference_mode": REFERENCE_MODE,
        "nk96": "not_run",
        "nk192": NK192,
        "S": "not_fixed_physical_mode_area_not_a_normalization_box",
        "N_cell": "unset_not_equal_to_nk",
        "pauli_C_P_prototype": 28.020752514399163,
        "pauli_C_C_prototype": 55.744789878664065,
        "pauli_gate": "area_requirement_curve_until_continuum",
        "si_prefactor": "connected_n_m_use_eta2_Ncell_equals_alpha_fs_over_A_uc",
        "squeezing_claim": SQUEEZING_CLAIM,
        "unresolved_magnetic": "not_promoted",
        "quantum_input": "not_run",
        "photon_output": "finite_time_kernel_diagnostics_only",
        "g2": "not_run",
        "g_self": "two_port_vacuum_sheet_frozen_P1_local_continuum_feedback",
        "production_V": "blocked_until_BZ_covariant_commutator_linear_selfcons",
        "volterra": "approved_linear_IO_H2_channel_not_production_V",
        "gain_medium_claim": "not_evaluated_diagnostics_only",
        "runaway_language": (
            "finite_time_linear_self_consistent_runaway_amplification_"
            "not_a_statement_that_the_Volterra_equation_has_no_solution"
        ),
        "qualification_control": "covariant_kshift_AND_H2_projected_resolvent_error",
        "kappa_self_over_p1": "classical_seed_stress_test_scales_with_eps_not_universal",
        "unweighted_L2_A_sigma": "stress_diagnostic_not_quantum_feedback_strength",
        "sigma_max_L_ph": "global_worst_case_stress_not_born_passport",
        "old_gauge_I_state_plus_contact": "withdrawn_not_a_gauge",
        "born_role": "perturbative_benchmark",
        "production_V_blocked": True,
        "theory_standing": "phgauss_is_quantum_main_resolvent_is_expectation_diagnostic",
        "canonical_io_gate": "SS_dagger_minus_TT_dagger_plus_C_el_equals_I",
        "resummed_n_m": "no_quantum_physical_standing_until_canonical_commutator",
        "vacuum_rms": "provisional_self_consistency_stress_diagnostic",
        "vacuum_rms_not_a_linear_failure_gate": True,
        "nonlinear_quantum_feedback": "not_started",
        "T2_environment": "not_added_needs_matching_Langevin_and_human_review",
        "bz_adaptive": "paused_low_priority_defect_is_not_a_k_count_issue",
        "next_after_canonical": "continuum_Pauli_area_curve_then_BZ_then_H2_n_m_V",
        "fork_nonlinear_vs_dissipation": "deferred_until_canonical_io_passes",
        "missing_orthogonal_modes": "excluded_stage1n",
        "born_xi_R_mismatch": "excluded_stage1n_kubo",
        "quantum_IO_from_Kubo_resolvent": "not_automatic",
        "required_noise_matrix": "diagnostic_not_a_production_langevin_patch",
        "rpa_joint_linear": "approved_as_P1_dressed_ph_photon_canonical_Gaussian",
        "canonical_defect_order_audit": "confirmed_beyond_Born",
        "G_self": "not_an_input_kernel_must_emerge_from_photon_continuum",
        "delta_C": "diagnostic_not_a_langevin_patch",
        "quasi_boson_selfcons": "prototype_area_requirement_curve_S_not_fixed",
        "G_self_emergence": "approaches_Maxwell_IR_as_Tp_over_Tobs_kernel_independent_of_Ncell_diag",
        "nk_is_not_N_cell": True,
    }
