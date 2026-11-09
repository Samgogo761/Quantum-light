"""G1 input/output combination sources: theory loop + read-only reconcile.

Does not assemble U,V. Does not call rows_with_kernels. Does not start K'.
Does not overwrite archived JSON. production_G1_pass=false.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_k1_packet import packet_on_grid

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_io_identity_20261020"
OUTJSON = OUTDIR / "io_identity.json"
PROGRESS = OUTDIR / "progress.txt"
TIME = ROOT / "results/q_line/g1_gamma12_time_order_20261015"
AB12 = ROOT / "results/q_line/g1_ab12_residual_groups_20261019/ab12_residual_groups.json"
WIN = ROOT / "results/q_line/g1_window_vs_asymptotic_20261019/window_vs_asymptotic.json"


def log(msg):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    prev = PROGRESS.read_text(encoding="utf-8") if PROGRESS.exists() else ""
    PROGRESS.write_text(prev + msg + "\n", encoding="utf-8")


def fnum(x):
    return float(np.real_if_close(x, tol=1e6))


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def trapz_weights(n, dt):
    w = np.full(n, dt, dtype=float)
    if n >= 2:
        w[0] *= 0.5
        w[-1] *= 0.5
    return w


def c2_window(t, t1, t2):
    """C^2 smootherstep: 1 on t<=t1, 0 on t>=t2."""
    w = np.ones_like(t, dtype=float)
    w[t >= t2] = 0.0
    mid = (t > t1) & (t < t2)
    s = (t[mid] - t1) / (t2 - t1)
    smooth = ((6.0 * s - 15.0) * s + 10.0) * s ** 3
    w[mid] = 1.0 - smooth
    return w


def f_on_times(times, dens, omega, chunk=64):
    out = np.empty(times.size, dtype=complex)
    for i0 in range(0, times.size, chunk):
        sl = times[i0 : i0 + chunk]
        phase = np.exp(1j * sl[:, None] * omega[None, :])
        out[i0 : i0 + chunk] = np.trapezoid(dens[None, :] * phase, omega, axis=1)
    return out


def delta_dens(omegas, times, g, wt, chunk=48):
    """Δ(ω)=(1/2π) ∫ g(t) e^{-iωt} dt, g=F(w-1)."""
    out = np.empty(omegas.size, dtype=complex)
    scale = wt / (2.0 * np.pi)
    for i0 in range(0, omegas.size, chunk):
        om = omegas[i0 : i0 + chunk]
        phase = np.exp(-1j * om[:, None] * times[None, :])
        out[i0 : i0 + chunk] = phase @ (g * scale)
    return out


def slice_1a():
    return {
        "question": (
            "Is U=U0+raw_driven(T)-raw_undriven(T) equivalent to slicing "
            "one output int K·field into three time segments?"
        ),
        "algebraic_equals_time_split": True,
        "algebraic_equals_time_split_meaning": (
            "Given U0 = complete undriven I/O map, "
            "U = (U0 - undr_T) + dr_T = undriven_outside_[0,T] + driven_[0,T]. "
            "This is one mixed output, not a sum of two independent Bogoliubov maps."
        ),
        "strict": {
            "production_map": (
                "U = U0 + raw_driven_u - raw_undriven_u; "
                "V = raw_driven_v - raw_undriven_v"
            ),
            "production_map_code": (
                "q8_m0_g1_boundary_sensitivity_20261015.py production_map "
                "(recorded, not written back)"
            ),
            "U0_code": "q8_m0_p1_h2_finite_kcv_input_close_20261005.u0_from_bundle",
            "U0_photon": "u0_j = phi(omega) * (pol @ ((sqrt(omega)/eta) * a_r))",
            "a_r": "left: full A (incident transmission + sheet); right: A - A_f",
            "dark_U0": "0 in gram_grouping.assemble",
            "raw_contains_direct_incident": True,
            "raw_incident_where": [
                "contract_initial: row @ (b, bdag, full A) at t=0",
                "contract_stacked time loop: forces @ free_A0 * phase on [0,T]",
            ],
            "kernel_split": {
                "F": "k_shape_on_times: int_0^inf phi/sqrt(omega) exp(+i omega t)",
                "K": "production detector = i * eta * F",
                "eta": "eta_of(gamma)=sqrt(gamma/2pi); no extra 2pi on resolvent",
            },
            "driven_minus_undriven_cancels_identical_vertices": (
                "If vertices identical, raw_driven=raw_undriven and U=U0. "
                "That cancellation is algebraic, not a proof that raw_undriven "
                "equals the radiated sheet."
            ),
        },
        "approximate": {
            "undr_T_versus_radiated_sheet": (
                "Identification raw_undriven(T) = windowed complete undriven "
                "radiation is numerical, tested in 2a (6e-6 at omega=0.11, weight 1)."
            ),
            "left_U0_has_transmission": (
                "Full U0 on the left is larger than the radiated sheet by design. "
                "raw_undriven=U0-out_U fails if U0 is the complete frequency map."
            ),
        },
        "unproved": [
            "U0 - undr_T equals int_{t not in [0,T]} K · undriven_outgoing",
            "6e-6 equals that closed-form window tail (not computed this round)",
            "adjoint pullback along two classical trajectories equals "
            "Heisenberg int K·deltaField of the mixed operator without remainder",
        ],
        "if_algebraic_split_failed": "would_stop_and_not_attribute_residual_to_tails",
        "did_not_fail_algebraically": True,
        "numerical_identification_is_2a": True,
    }


def slice_1b():
    return {
        "status": "proved_as_implication_under_A1_to_A6",
        "not_an_unconditional_physical_fact": True,
        "statement": (
            "If A1-A6 hold, then T->inf implies b(T)->b_dr, and the complete "
            "driven output is a normalized free outgoing mode, hence "
            "R_N(inf)=R_A(inf)=0."
        ),
        "quadratic_constraint": (
            "Finite-T linear combination need not be canonical. "
            "T->inf canonicity comes from the split reducing to the complete "
            "driven output, not from UU^dag-VV^dag being closed under subtraction."
        ),
        "A1": {
            "content": "quadratic generator Hermitian, pairing symmetric",
            "support": "S Sigma S^dag - Sigma = 4.442441202370322e-16",
            "source": "ccr_identity_quadratic.json; 线Q_G1独立对易守恒恒等式_20261015.md §2",
            "not_the_4p4e-15_covariance_vs_sqrt": True,
        },
        "A2": {
            "content": "gamma and eta=sqrt(gamma/2pi) on the same sheet",
            "sheet_A_minus_Af_minus_i_gamma_I_over_omega": 3.469625883049681e-17,
            "static_U0U0dag_minus_I_2048_00": 2.13e-12,
            "right_port_U0_minus_radiated": 9.813077866773595e-18,
            "source": "boundary_sensitivity.json; 线Q_G1动态映射分解_20261015.md",
        },
        "A3": {
            "content": "incoming columns complete; bright electrons not independent inputs; dark=ker(u*)",
            "support": "drop one oscillator column -> commutator O(1) in the closed quadratic model",
            "source": "ccr_identity_quadratic.json; incoming completeness 20261004",
        },
        "A4": {
            "content": "int |phi|^2 = 1, positive frequency only",
            "phi0_analytic_zero": True,
            "K_has_no_t_to_minus_half_tail": True,
        },
        "A5": {
            "content": "the T->inf limit exists",
            "support": "K has no t^{-1/2} tail (settled); Delta I to 20 cycles bounded oscillation (historical)",
            "stop_if_Delta_I_unbounded": True,
            "not_rechecked_this_round": True,
        },
        "A6": {
            "content": "t=0 matching of undriven window integral to radiated sheet, not to full U0",
            "numerical_support": "slice_2a",
            "if_2a_fails": "premise_not_numerically_verified",
        },
        "user_binding": "this_round_write_R_N_inf_equals_0_as_proved_under_A1_A6",
        "does_not_imply_finite_window_residual_is_a_proven_cutoff": True,
        "does_not_authorize_Kprime": True,
        "production_G1_pass": False,
    }


def slice_1c():
    return {
        "written_because_1a_algebraic": True,
        "column_form": "R_N(T) = -2 Re[b_dr, delta_b_T^dag] + [delta_b_T, delta_b_T^dag]",
        "delta_b_T": (
            "b_dr - b(T). Under the 1a split, this is the missing "
            "int_{t not in [0,T]} K · driven_field, currently replaced by "
            "undriven outside the window via U0 - undr_T."
        ),
        "delta_b_T_also_contains": (
            "the 2a undriven mismatch (A6), if that identification is imperfect"
        ),
        "keep_other_sources": ["U0 cross", "dark columns", "Maxwell thin-sheet"],
        "do_not_prelabel_1e-5_as_window_effect": True,
        "do_not_close_9p60e-4_as_set_out_to_zero": True,
    }


def slice_1d():
    return {
        "this_round_does_not_run_Kprime": True,
        "Kprime_is": "another finite-time measurement K*w",
        "Kprime_is_not": [
            "a legal normalized positive-frequency outgoing photon mode",
            "production W_out",
        ],
        "cannot_assume_compact_support_and_same_phi_with_phi0_zero": True,
        "if_diagnosed_later": {
            "R_N_Kprime": "||phi'_+||^2 - ||phi'_-||^2 - 1",
            "R_A_Kprime": "must_be_derived; negative frequency may enter V0'; not a prior",
            "static_u0_from_bundle_on_phi_plus": "must_say_how_negative_frequency_is_handled",
        },
        "future_gate_only": "||phi'_-||^2 <= 1e-8 is a later K' gate, not a run authorization",
    }


def slice_2a():
    rad = load_json(TIME / "undriven_radiated_match.json")
    bnd = load_json(TIME / "undriven_boundary_match.json")
    bsens = load_json(TIME / "boundary_sensitivity.json")
    p011 = rad["probes"]["0.11"]
    raw1024 = bnd["probes"]["1024"]["0.11"]["undriven_versus_U0"]["max_abs_raw_u"]
    raw2048 = bnd["probes"]["2048"]["0.11"]["undriven_versus_U0"]["max_abs_raw_u"]
    timestep_diff = abs(raw2048 - raw1024)
    mismatch = p011["max_abs_raw_minus_radiated"]
    lev = bsens["levels"]["2048"]
    gate_fail = mismatch > timestep_diff
    return {
        "did_not_reassemble_145944": True,
        "contrast_object": "radiated sheet phi*(sqrt(omega)/eta)*(A-A_f), not full U0",
        "full_U0_also_contains_direct_transmission_on_the_left": True,
        "cited_undriven_radiated_match": {
            "path": str(TIME / "undriven_radiated_match.json"),
            "weight": rad["weight"],
            "omega_0.11_max_abs_raw_minus_radiated": mismatch,
            "omega_0.11_max_abs_radiated": p011["max_abs_radiated"],
            "omega_0.11_max_abs_full_U0": p011["max_abs_full_U0"],
            "left_minus_right_raw": p011["left_minus_right_raw"],
            "probes": rad["probes"],
        },
        "cited_timestep_not_the_6e-6": {
            "path": str(TIME / "undriven_boundary_match.json"),
            "omega_0.11_raw_u_1024": raw1024,
            "omega_0.11_raw_u_2048": raw2048,
            "abs_diff": timestep_diff,
            "note": "6e-6 is not a timestep error; 1024 vs 2048 differ by ~2e-6",
        },
        "cited_dark_undriven": bnd["probes"]["2048"]["undriven_dark_8columns"],
        "cited_sheet_and_right_port": {
            "path": str(TIME / "boundary_sensitivity.json"),
            "sheet_A_minus_Af_minus_i_gamma_I_over_omega": lev[
                "sheet_A_minus_Af_minus_i_gamma_I_over_omega"
            ],
            "right_port_U0_minus_radiated": lev["right_port_U0_minus_radiated"],
            "left_radiated_minus_right_radiated": lev["left_radiated_minus_right_radiated"],
            "hypothesis_not_an_approved_replacement": bsens[
                "hypothesis_not_an_approved_replacement"
            ],
        },
        "cannot_read_as_out_of_window_closed_form_already_matches": True,
        "delta_RN_9p60e-4": {
            "value": 9.60e-4,
            "status": "unapproved_sensitivity_replace_undriven_integral_by_radiated_sheet",
            "close_only_if_1a_proves_it_equals_set_out_to_zero": True,
            "closed_this_round": False,
        },
        "gate_second_AI_spirit_correct_contrast": {
            "require": "residual at timestep level and shrinking at higher order",
            "observed_mismatch": mismatch,
            "timestep_level": timestep_diff,
            "passed": (not gate_fail),
        },
        "A6_numerically_verified": False,
        "stop_Kprime_because_2a": True,
    }


def slice_2b():
    log("2b start Fourier of K*w")
    prot = p1_a2_protocol(0.8)
    rec = packet_on_grid(prot.omega0, prot.t_pulse, n_harm=2)
    omega = np.asarray(rec["omega"], dtype=float)
    phi = np.asarray(rec["phi"])
    wo = np.sqrt(np.maximum(omega, 0.0))
    dens = np.zeros_like(phi)
    mask = wo > 0.0
    dens[mask] = phi[mask] / wo[mask]
    period = float(prot.period)
    t1 = 8.5 * period
    t2 = 9.0 * period
    extra = 80.0 * period
    dt = period / 256.0
    times = np.arange(t1, t2 + extra + 0.5 * dt, dt)
    F = f_on_times(times, dens, omega)
    log("2b F(t) on tail grid")
    w = c2_window(times, t1, t2)
    g = F * (w - 1.0)
    wt = trapz_weights(times.size, dt)
    pos = omega > 0.0
    om_pos = omega[pos]
    dlt_pos = delta_dens(om_pos, times, g, wt)
    om_neg = -om_pos[::-1]
    dlt_neg = delta_dens(om_neg, times, g, wt)
    phi_pos = phi[pos]
    phi_plus = phi_pos + dlt_pos * np.sqrt(om_pos)
    nrm_phi = fnum(np.trapezoid(np.abs(phi_pos) ** 2, om_pos))
    nrm_plus = fnum(np.trapezoid(np.abs(phi_plus) ** 2, om_pos))
    nrm_diff = fnum(np.trapezoid(np.abs(dlt_pos) ** 2 * om_pos, om_pos))
    nrm_minus = fnum(np.trapezoid(np.abs(dlt_neg) ** 2 * np.abs(om_neg), om_neg))
    # shorter tail for a cheap convergence check
    cut = times <= (t2 + 40.0 * period)
    dlt_m40 = delta_dens(om_neg, times[cut], g[cut], trapz_weights(int(cut.sum()), dt))
    nrm_minus_40 = fnum(
        np.trapezoid(np.abs(dlt_m40) ** 2 * np.abs(om_neg), om_neg)
    )
    future_gate = 1e-8
    out = {
        "protocol": "p1_a2_protocol(0.8)",
        "omega0": float(prot.omega0),
        "t_pulse": float(prot.t_pulse),
        "period": period,
        "window": "C2 smootherstep 8.5->9.0 drive cycles, 0 after 9.0",
        "t1": t1,
        "t2": t2,
        "tail_extra_cycles": 80.0,
        "dt": dt,
        "n_times": int(times.size),
        "int_abs_phi2_on_packet_grid": nrm_phi,
        "norm_phi_plus_sq": nrm_plus,
        "norm_phi_prime_minus_phi_sq": nrm_diff,
        "norm_phi_minus_sq": nrm_minus,
        "norm_phi_minus_sq_40cycle_tail": nrm_minus_40,
        "U0_phi_plus_U0dag_minus_I_estimate": nrm_plus - 1.0,
        "U0_estimate_kind": (
            "isotropic packet L2 on the k_shape omega grid; "
            "NOT an assembled production-node U0 Gram"
        ),
        "did_not_assemble_U0": True,
        "did_not_generate_vertices": True,
        "future_Kprime_gate_1e-8": future_gate,
        "future_Kprime_gate_passed": bool(nrm_minus <= future_gate),
        "this_round_still_does_not_run_Kprime": True,
        "note_even_if_gate_fails": "report only; K' is not started",
    }
    log("2b Fourier done")
    return out


def win_bookkeeping():
    ab = load_json(AB12)
    b3 = ab["slice_B3"]
    return {
        "status": "code_formula_mapping_only",
        "numerical_integral_audit": "not_started",
        "source": "ab12_residual_groups.json slice_B3",
        "W_in_is_not_D": b3["W_in_is_not_D"],
        "D_is_not_radiative_damping": b3["D_is_not_radiative_damping"],
        "gamma_is_radiative_feedback_not_D": b3["gamma_is_radiative_feedback_not_D"],
        "objects": b3["objects"],
        "unverified": [
            "nodes/weights counted once",
            "L/R x 2 polarizations",
            "IR/UV endpoints",
            "discrete commutator kernel vs continuum kernel",
        ],
        "do_not_submit_integral_job": True,
    }


def stage3_quote():
    win = load_json(WIN)
    cost = win["kernel_cost_quoted_from_archive"]
    return {
        "this_round_does_not_run": True,
        "nine_cycle_kernel_correction_elapsed_s_archived": cost[
            "nine_cycle_elapsed_s_from_json"
        ],
        "plan_unified_quote": {
            "nine_cycle_1024": "vertices ~1435 s + rows ~424 s + assemble; ~31-45 min",
            "note": "1435 s is the plan/first-AI vertex class, not a new measurement",
            "eleven_cycle_vertices_plus_rows_s": 2622.7,
            "eleven_cycle_including_assemble_s": 3102.7,
        },
        "vertices_on_disk": False,
        "vertices_2048_on_disk": False,
        "full_UV_assembly_2048": False,
        "if_approved_later": {
            "predict_against": "nine-cycle@1024 R_N(0,0)=1.048902e-6",
            "H_win": "R_N(K') ≈ ||phi'_+||^2 - ||phi'_-||^2 - 1",
            "D": "R_N(K) - R_N(K')",
            "do_not_use_1p7e-5_or_minus_1p6e-5": True,
            "K_rows_must_reproduce": "rows_kernel_correction_9cycles.npz",
            "new_directory": True,
            "even_if_H_win": [
                "do not declare G1 pass",
                "do not declare W_in audited",
            ],
        },
    }


def main():
    t0 = time.perf_counter()
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("start io-identity 20261020\n", encoding="utf-8")
    s1a = slice_1a()
    s1b = slice_1b()
    s1c = slice_1c()
    s1d = slice_1d()
    log("1a-1d theory written")
    s2a = slice_2a()
    log("2a cited existing JSON")
    s2b = slice_2b()
    report = {
        "task": "G1 input/output combination sources; theory loop + read-only; do not run K'",
        "date": "2026-10-20",
        "production_G1_pass": False,
        "quasi_boson": "pending",
        "did_not_assemble_UV": True,
        "did_not_call_rows_with_kernels": True,
        "did_not_run_Kprime": True,
        "did_not_start_long_propagation": True,
        "did_not_change_Hamiltonian_P1_phi_eta_gamma_D": True,
        "slice_1a": s1a,
        "slice_1b": s1b,
        "slice_1c": s1c,
        "slice_1d": s1d,
        "slice_2a": s2a,
        "slice_2b": s2b,
        "W_in_bookkeeping": win_bookkeeping(),
        "stage3_quote_not_run": stage3_quote(),
        "stop_Kprime": True,
        "stop_reason": (
            "2a: undriven integral vs radiated sheet is 6.013e-6 at omega=0.11, "
            "not at the 1024/2048 timestep level (~2e-6); A6 not numerically verified. "
            "K' remains blocked until 1+2 pass and explicit user approval."
        ),
        "flowchart": {
            "1a_algebraic": True,
            "1b_theorem_as_implication": True,
            "2a_gate": False,
            "2b_is_report_only": True,
            "ask_user_to_approve_Kprime": False,
        },
        "do_not_overwrite": [
            "window_vs_asymptotic.json",
            "ab12_residual_groups.json",
            "nm_tables.json",
            "kernel_correction_9cycles.*",
            "kernel_correction_11cycles.json",
            "q_slices_bcde.json",
            "q_cov4_fix.json",
        ],
        "elapsed_s": time.perf_counter() - t0,
        "provenance_sha256": provenance(
            [
                __file__,
                TIME / "undriven_radiated_match.json",
                TIME / "undriven_boundary_match.json",
                TIME / "boundary_sensitivity.json",
                TIME / "ccr_identity_quadratic.json",
                AB12,
                WIN,
            ]
        ),
    }
    write_new_json(OUTJSON, report)
    log(f"wrote {OUTJSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
