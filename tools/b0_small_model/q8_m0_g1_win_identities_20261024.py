"""G1 round 2: W_in identities + A1-A6 status table. Zero new quadrature nodes.

Read-only. Does not assemble U,V. Does not call rows_with_kernels.
Does not start K' or long propagation. Does not overwrite archived JSON.
Does not start a W_in numerical integral audit.
production_G1_pass=false.

Does not claim R_N(inf)=0 unconditionally.
Does not mix 2.59e-8 kernel Fourier with 6.013e-6 raw-vs-sheet.
Does not mix H2 two-pol with G1.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_win_identities_20261024"
OUTJSON = OUTDIR / "win_identities.json"
PROGRESS = OUTDIR / "progress.txt"
TOOLS = Path(__file__).resolve().parent

IO = ROOT / "results/q_line/g1_io_identity_20261020/io_identity.json"
A6FIX = ROOT / "results/q_line/g1_a6_fix_20261024/a6_fix.json"
AB12 = ROOT / "results/q_line/g1_ab12_residual_groups_20261019/ab12_residual_groups.json"
WIN = ROOT / "results/q_line/g1_window_vs_asymptotic_20261019/window_vs_asymptotic.json"
COMP = ROOT / "results/q_line/incoming_spectral_completeness_20261004/completeness_checks.json"
BUDGET = ROOT / "results/q_line/g1_nm_tables_20261017/error_budget_existing.json"
QSL = ROOT / "results/q_line/g1_q_slices_bcde_20261018/q_slices_bcde.json"
CCR = ROOT / "results/q_line/g1_gamma12_time_order_20261015/ccr_identity_quadratic.json"
RAD = ROOT / "results/q_line/g1_gamma12_time_order_20261015/undriven_radiated_match.json"

IC = TOOLS / "q8_m0_p1_h2_finite_kcv_input_close_20261005.py"
CF = TOOLS / "q8_m0_p1_h2_finite_kcv_caller_fix_20261005.py"
INC = TOOLS / "q8_m0_incoming_spectral_completeness_20261004.py"


def log(msg):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    prev = PROGRESS.read_text(encoding="utf-8") if PROGRESS.exists() else ""
    PROGRESS.write_text(prev + msg + "\n", encoding="utf-8")


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def snippet(path, start, end):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1 : end])


def four_window(ab12):
    return {float(row["cycles"]): row for row in ab12["slice_B1"]["four_window_table"]}


def win_identities(io, comp):
    """Formula + code map. No new nodes. Independent numerical audit not done."""
    b3 = io["W_in_bookkeeping"]
    return {
        "this_round_did_not_start_numerical_integral_audit": True,
        "B3_not_redone": True,
        "W_in_is_not_D": True,
        "D_is_not_radiative_damping": True,
        "gamma_is_radiative_feedback_not_D": True,
        "identities": [
            {
                "id": "W1_LR_independent_vacuum_ports",
                "formula": (
                    "Incoming a_L and a_R are two independent vacuum ports. "
                    "even=(L+R)/sqrt(2), odd=(L-R)/sqrt(2) is a later basis change. "
                    "even_bd L=R is not the operator identity a_L=a_R."
                ),
                "code": [
                    "q8_m0_p1_h2_finite_kcv_caller_fix_20261005.photon_port_indices",
                    "q8_m0_p1_h2_finite_kcv_caller_fix_20261005.even_odd_from_lr",
                ],
                "code_snippet": snippet(CF, 344, 365),
                "status_2A": "conditional",
                "evidence_kind": "mapping_only_not_numerically_verified",
                "proved_in_code": "L/R split by label; even/odd only after (L±R)/sqrt(2)",
                "production_code_verified": False,
                "independent_CCR_L_vs_R_continuum": "not_checked",
                "note": (
                    "Port-parity adjoint-row work shows even/odd dynamics decouple; "
                    "that is not a continuum W_in CCR audit of independent L/R vacua."
                ),
            },
            {
                "id": "W2_measure_eta_weights",
                "formula": (
                    "eta=sqrt(gamma/2pi); attach_weights puts sqrt(dnu) once into photon "
                    "columns; dark columns stay point mass 1; detector kernel has an extra "
                    "sqrt(2pi) relative to this eta; resolvent must not get another 2pi."
                ),
                "code": [
                    "q8_m0_p1_h2_finite_kcv_input_close_20261005.eta_of",
                    "q8_m0_p1_h2_finite_kcv_input_close_20261005.attach_weights",
                    "q8_m0_p1_h2_finite_kcv_input_close_20261005.input_quadrature",
                ],
                "eta_source": snippet(IC, 250, 251),
                "attach_source": snippet(IC, 922, 939),
                "status_2A": "conditional",
                "evidence_kind": "mapping_only_not_numerically_verified",
                "proved_in_code": "eta_of and attach_weights exist as written",
                "production_code_verified": False,
                "nodes_weights_counted_once": "not_checked",
                "discrete_vs_continuum_CCR": "not_checked",
                "IR_UV_endpoints": "not_checked",
            },
            {
                "id": "W3_incoming_completeness_and_dark",
                "formula": (
                    "IncomingBundle = n_photon_columns + n_point_columns. "
                    "Dark = ker(u*), discrete electronic DOF, not frequency-quadrature error. "
                    "Bright electrons are not independent incoming oscillators."
                ),
                "code": [
                    "q8_m0_incoming_spectral_completeness_20261004.IncomingBundle",
                    "q8_m0_incoming_spectral_completeness_20261004.assemble_incoming_bundle",
                ],
                "bundle_source": snippet(INC, 236, 260),
                "fixture": {
                    "all_checks_pass": comp["all_checks_pass"],
                    "status": comp["status"],
                    "continuum_BZ_claim": comp["continuum_BZ_claim"],
                    "missing": comp["M0_interface"]["missing"],
                },
                "status_2A": "conditional",
                "evidence_kind": "conditional_theory_only",
                "proved_in_code": "2-k fixture all_checks_pass; dark=ker(u*) by construction",
                "production_code_verified": False,
                "note": (
                    "Dropping one oscillator in the closed quadratic model breaks CCR O(1). "
                    "That does not prove production incoming columns are complete. "
                    "completeness_checks missing: full-BZ retarded electronic continuum, "
                    "complete positive-frequency photon packet integral, production n/m/V."
                ),
            },
            {
                "id": "W4_Win_not_D_not_gamma",
                "formula": (
                    "W_in = incoming vacuum in program coordinates. "
                    "D = electronic contact in I = u* b + u bdag - D A. "
                    "gamma = radiative Maxwell feedback Adot = Adot_f + gamma I. "
                    "These are three objects."
                ),
                "code": b3["objects"],
                "status_2A": "proved_in_code",
                "evidence_kind": "math_identity_proved",
                "proved_in_code": "inventory identity; B3 mapping 2026-10-19, not a new map",
                "production_code_verified": "classification_only",
                "numerical_integral_audit": "not_started",
            },
        ],
        "unverified_from_io_identity": b3["unverified"],
        "fluctuation_dissipation_warning": (
            "Do not test W_in vs D as a noise/dissipation pair."
        ),
    }


def a1a6_table(io, a6, ccr_exists):
    s1b = io["slice_1b"]
    s2a = io["slice_2a"]
    head = a6["omega_0.11_headline"]
    return {
        "theorem": s1b["statement"],
        "theorem_status": "implication_only_not_unconditional_fact",
        "quadratic_constraint": s1b["quadratic_constraint"],
        "marks_allowed": ["proved", "conditional_unverified", "fails"],
        "A1": {
            "content": s1b["A1"]["content"],
            "mark": "conditional_unverified",
            "evidence_kind": "math_identity_proved_on_closed_quadratic_only",
            "support": s1b["A1"]["support"],
            "source": s1b["A1"]["source"],
            "ccr_json_on_disk": bool(ccr_exists),
            "not_the_4p4e-15_covariance_vs_sqrt": True,
            "limitation": (
                "Closed quadratic S Sigma S^dag - Sigma ~ 4.442441202370322e-16 "
                "does not imply the production reduced equations plus initial history "
                "are the same model."
            ),
        },
        "A2": {
            "content": s1b["A2"]["content"],
            "mark": "proved",
            "evidence_kind": "production_code_verified_sheet_identity",
            "sheet_A_minus_Af_minus_i_gamma_I_over_omega": s1b["A2"][
                "sheet_A_minus_Af_minus_i_gamma_I_over_omega"
            ],
            "right_port_U0_minus_radiated": s1b["A2"]["right_port_U0_minus_radiated"],
            "static_U0U0dag_minus_I_2048_00": s1b["A2"]["static_U0U0dag_minus_I_2048_00"],
            "limitation": (
                "Sheet gamma-eta identity is verified. Full packet U0 Gram was not "
                "assembled this round (io_identity 2b is an isotropic L2 estimate)."
            ),
        },
        "A3": {
            "content": s1b["A3"]["content"],
            "mark": "conditional_unverified",
            "evidence_kind": "conditional_theory_only",
            "support": s1b["A3"]["support"],
            "limitation": (
                "Dropping a column breaks CCR in the closed quadratic model. "
                "That is not a proof that production incoming columns are complete."
            ),
        },
        "A4": {
            "content": s1b["A4"]["content"],
            "mark": "proved",
            "evidence_kind": "math_identity_proved",
            "phi0_analytic_zero": True,
            "K_has_no_t_to_minus_half_tail": True,
        },
        "A5": {
            "content": s1b["A5"]["content"],
            "mark": "conditional_unverified",
            "evidence_kind": "conditional_theory_only",
            "support": s1b["A5"]["support"],
            "not_rechecked_this_round": True,
            "limitation": (
                "Bounded oscillation of some columns to 20 cycles is not existence of "
                "U,V limits under the full incoming measure, and does not justify "
                "exchanging T->inf with the column integral."
            ),
        },
        "A6": {
            "content": s1b["A6"]["content"],
            "mark": "conditional_unverified",
            "evidence_kind": "numerical_gate_failed_identity_not_disproved",
            "two_a_numerical_gate": "fails",
            "A6_numerically_verified": False,
            "raw_minus_sheet_6e-6": s2a["gate_second_AI_spirit_correct_contrast"][
                "observed_mismatch"
            ],
            "timestep_1024_2048_mod_diff": s2a["gate_second_AI_spirit_correct_contrast"][
                "timestep_level"
            ],
            "kernel_fourier_complete_minus_analytic_exact": head[
                "abs_complete_minus_analytic_exact"
            ],
            "must_not_mix": (
                "2.59e-8 is 1D detector-kernel Fourier vs analytic. "
                "6.013e-6 is undriven adjoint raw vs radiated sheet. Different errors."
            ),
            "objects_not_equal_by_inventory": [
                "raw_undriven = int_0^T K I_0 (no independent b_free)",
                "radiated sheet = phi (sqrt(omega)/eta) (A-A_f), not full U0",
                "full U0 includes left-port transmission",
            ],
            "limitation": (
                "A6 is not closed. The 2a identification raw=sheet failed the "
                "timestep-level gate. That does not prove the tail identity false; "
                "it proves production objects are not identified with the theory pieces."
            ),
        },
    }


def error_budget(ab12, budget, qsl, io, a6):
    fw = four_window(ab12)
    r9, r11 = fw[9.0], fw[11.0]
    kq = budget["kernel_quadrature_9cycle_macro1024"]
    ts = budget["nine_cycle_IR_corrected_RN_00"]
    s2a = io["slice_2a"]
    head = a6["omega_0.11_headline"]
    return {
        "status": "inventory_of_existing_evidence_not_a_closed_budget",
        "R_N": {
            "cycles9_RN00": r9["RN00"],
            "cycles9_RN11": r9["RN11"],
            "cycles11_RN00": r11["RN00"],
            "cycles11_RN11": r11["RN11"],
            "window_11_minus_9_RN00": r11["RN00"] - r9["RN00"],
            "source": "ab12_residual_groups.json slice_B1 four_window_table",
            "R_A_9_prod_maxabs": 8.942758742199518e-07,
            "R_A_9_source": "kernel_correction_9cycles.json R_A_prod.maxabs (headline only; file not reloaded)",
            "R_A_11": "missing_in_small_json_not_invented",
        },
        "kernel_delta_RN_00": {
            "value": kq["delta_RN_00"],
            "RN_00_before": kq["RN_00_before"],
            "RN_00_after": kq["RN_00_after"],
            "not_the_leading_G1_leak": True,
            "source": "error_budget_existing.json",
        },
        "timestep": {
            "label": "conditional_error_estimate_not_strict_bound",
            "macro1024_RN00": ts["macro1024"],
            "macro2048_RN00": ts["macro2048"],
            "h4_two_point_extrapolation": ts["h4_two_point_extrapolation"],
            "note": ts["note"],
            "source": "error_budget_existing.json",
        },
        "four_window_Q0": {
            "Q0_9": qsl["windows"]["9"]["Q0"] if "windows" in qsl else r9["Q0"],
            "swing": qsl.get("four_window_Q0_swing"),
            "label": qsl.get("four_window_swing_label", "conditional_error_estimate_not_strict_bound"),
            "source": "q_slices_bcde.json / ab12",
        },
        "conservative_tail_bound_written_not_evaluated": {
            "formula": "||dR_N|| <= 2||U||||dU|| + ||dU||^2 + 2||V||||dV|| + ||dV||^2",
            "evaluated": False,
            "reason": "dU,dV as T->inf tails are not on disk as reusable columns",
        },
        "unapproved_sensitivity_delta_RN_9p60e-4": s2a["delta_RN_9p60e-4"],
        "a6_two_errors_not_the_same": {
            "kernel_fourier_2p59e-8": head["abs_complete_minus_analytic_exact"],
            "raw_minus_sheet_6p013e-6": head["archived_raw2048_minus_S"],
            "archived_raw_v": head["archived_raw_v"],
            "not_a_proof_that_6e-6_is_a_time_tail": True,
        },
        "do_not_claim_1e-5_is_proven_window_cutoff": True,
    }


def g1_decision():
    return {
        "can_apply_for_G1_pass": False,
        "production_G1_pass": False,
        "quasi_boson": "pending",
        "reason_class": "production_io_map_not_identified_with_theory",
        "decision_text": (
            "Cannot apply for G1 acceptance. The production input/output map is not "
            "identified with the theoretical mixed operator: raw_undriven, radiated "
            "sheet, and full U0 remain three objects. A6 is unverified; the 2a gate "
            "failed. W_in independent numerical audit was not done this round."
        ),
        "blocking_item": (
            "A6 exact boundary: undriven [0,T] adjoint ≠ radiated sheet ≠ full U0 "
            "(left-port transmission sits in U0). Measured gap "
            "max_abs_raw_minus_radiated = 6.013361711968586e-6 at omega=0.11 "
            "is not explained, and is not the 2.59e-8 kernel-Fourier remainder."
        ),
        "minimal_discriminating_evidence": {
            "what": (
                "Using already-saved undriven adjoint rows and the radiated sheet, "
                "decompose the complex 6.013e-6 (U and V) into (i) definitional "
                "mismatch of comparing int_0^T K I_0 to the full sheet versus "
                "(ii) quadrature/step error of the same object."
            ),
            "why_this_one": (
                "If (i), the production map is not the theoretical split and G1 "
                "cannot pass without a map correction (user decision, not a new "
                "quadrature). If (ii) at timestep level, A6 can be reopened as a "
                "controlled window remainder and a tail bound on dR_N becomes the "
                "next human decision. This splits 'window cutoff of a canonical map' "
                "from 'I/O map not corresponding'."
            ),
            "not": [
                "new omega nodes",
                "K'",
                "new U,V assembly",
                "new adjoint/vertex job unless user approves the quoted cost",
                "W_in continuum quadrature",
                "H2 +/-N",
            ],
            "if_cache_insufficient": {
                "quote": "11-cycle undriven vertices ~1870.5 s, rows cached class ~2622.7 s",
                "quote_class": "machine_time_quote_not_a_run_authorization",
                "user_decides": True,
            },
            "does_not_auto_expand": True,
        },
        "W_in_independent_numerical_audit": "not_completed_this_round_mapping_only",
        "H2_two_pol": "paper_physics_candidate_not_G1",
        "round3": "round3_not_needed_pending_user",
        "round3_note": (
            "K' is not the minimal discriminator. The block is object identification "
            "of undriven raw vs sheet vs U0, not a new finite-time measurement."
        ),
        "pause_after_round_2": True,
        "do_not_promote": [
            "n,m,Q",
            "repaired 4x4",
            "AB12 grouping",
            "A6 diagnostic fix",
            "H2 candidate",
        ],
    }


def q0_from_slices(qsl, fallback):
    block = qsl.get("slice_B_window_9p5_10", {})
    swing = block.get("four_window_Q0_swing", qsl.get("four_window_Q0_swing"))
    label = block.get(
        "four_window_swing_label",
        qsl.get("four_window_swing_label"),
    )
    q0_9 = fallback
    beat = block.get("four_beat")
    if isinstance(beat, list) and beat:
        q0_9 = beat[0].get("Q0", fallback)
    return {"Q0_9": q0_9, "swing": swing, "label": label}


def main():
    started = time.perf_counter()
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    log("win identities start")
    io = load_json(IO)
    log("loaded io_identity")
    a6 = load_json(A6FIX)
    log("loaded a6_fix")
    ab12 = load_json(AB12)
    log("loaded ab12")
    win = load_json(WIN)
    log("loaded window_vs_asymptotic")
    comp = load_json(COMP)
    log("loaded completeness")
    budget = load_json(BUDGET)
    log("loaded error_budget")
    qsl = load_json(QSL)
    log("loaded q_slices")

    fw = four_window(ab12)
    q0_info = q0_from_slices(qsl, fw[9.0]["Q0"])
    eb = error_budget(ab12, budget, qsl, io, a6)
    eb["four_window_Q0"] = {
        "Q0_9": q0_info["Q0_9"],
        "Q0_9_from_ab12": fw[9.0]["Q0"],
        "Q0_11_from_ab12": fw[11.0]["Q0"],
        "swing": q0_info["swing"],
        "label": q0_info["label"] or "conditional_error_estimate_not_strict_bound",
        "source": "q_slices_bcde.json four_window_Q0_swing; Q0 from ab12 four_window_table",
    }

    report = {
        "task": "G1 round 2 W_in identities + A1-A6 status + acceptance decision; no K'",
        "date": "2026-10-24",
        "production_G1_pass": False,
        "quasi_boson": "pending",
        "did_not_assemble_UV": True,
        "did_not_call_rows_with_kernels": True,
        "did_not_run_Kprime": True,
        "did_not_start_long_propagation": True,
        "did_not_start_Win_numerical_integral_audit": True,
        "did_not_add_quadrature_nodes": True,
        "did_not_change_Hamiltonian_P1_phi_eta_gamma_D": True,
        "did_not_overwrite_archived_json": True,
        "did_not_mix_2p59e-8_with_6p013e-6": True,
        "did_not_mix_H2_with_G1": True,
        "algebraic_1a_archived": io["slice_1a"]["algebraic_equals_time_split"],
        "production_map": io["slice_1a"]["strict"]["production_map"],
        "W_in_identities": win_identities(io, comp),
        "A1_to_A6": a1a6_table(io, a6, CCR.exists()),
        "error_budget_inventory": eb,
        "window_vs_asymptotic_cited": {
            "R_N_inf_equals_zero_then": win["b_out_inf_and_tail"]["R_N_inf_equals_zero"],
            "note": "Cited as inventory; this round does not reopen that write-up. Later 1b wrote the implication under A1-A6; A6 still unverified.",
        },
        "g1_acceptance": g1_decision(),
        "do_not_overwrite": [
            "a6_boundary_identity.json",
            "a6_fix.json",
            "io_identity.json",
            "window_vs_asymptotic.json",
            "ab12_residual_groups.json",
            "nm_tables.json",
            "eo_inv_dict.json",
            "q_cov4_fix.json",
            "q_slices_bcde.json",
            "kernel_correction_9cycles.*",
            "kernel_correction_11cycles.json",
            "precheck_sym_cred.json",
            "completeness_checks.json",
        ],
        "elapsed_s": time.perf_counter() - started,
        "provenance_sha256": provenance(
            [
                __file__,
                IO,
                A6FIX,
                AB12,
                WIN,
                COMP,
                BUDGET,
                QSL,
                IC,
                CF,
                INC,
            ]
        ),
    }
    write_new_json(OUTJSON, report)
    log(f"wrote {OUTJSON}")
    log("decision: cannot apply for G1 pass")


if __name__ == "__main__":
    main()
