"""A6 boundary identity: exact split of undriven finite-T adjoint vs radiated sheet.

Read-only. Does not assemble U,V. Does not call rows_with_kernels.
Does not start K' or long propagation. Does not overwrite archived JSON.
production_G1_pass=false.

Cheap 1D work: Fourier split of the frozen detector kernel K(t) at the
archived probe frequencies. No adjoint rows, no vertices, no photon columns.
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
from q8_m0_k1_packet import normalize_phi, omega_grid, phi_unnormalized


def eta_of(gamma):
    return float(np.sqrt(float(gamma) / (2.0 * np.pi)))

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_a6_boundary_identity_20261021"
OUTJSON = OUTDIR / "a6_boundary_identity.json"
PROGRESS = OUTDIR / "progress.txt"
TIME = ROOT / "results/q_line/g1_gamma12_time_order_20261015"
IO = ROOT / "results/q_line/g1_io_identity_20261020/io_identity.json"
GAMMA = 0.0010807892763646022
CYCLES = 9.0
PAST_CYCLES = 40.0
FUTURE_CYCLES = 40.0
PROBES = (0.11, 0.25, 1.0, 1.5, 4.0)


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


def kernel_on_times(om, phi, eta, times, batch=48):
    weight = phi / np.sqrt(om)
    k = np.empty(times.size, complex)
    for s in range(0, times.size, batch):
        sl = slice(s, min(s + batch, times.size))
        phase = np.exp(1j * om[:, None] * times[sl][None, :])
        k[sl] = 1j * eta * np.trapezoid(weight[:, None] * phase, om, axis=0)
    return k


def fourier_K(k, times, wt, omega):
    phase = np.exp(-1j * float(omega) * times)
    return complex(np.dot(k * phase, wt))


def region_mask(times, lo, hi):
    return (times >= lo) & (times < hi)


def identity_from_code():
    return {
        "question": (
            "What is the exact relation between the undriven finite-T adjoint "
            "integral and the frequency-domain radiated sheet?"
        ),
        "must_not_require": "raw_[0,T] == radiated_sheet, ignoring initial and tails",
        "continuous_stationary_identity": {
            "all_time": (
                "S(omega) = int_{-inf}^{inf} dt K(t) (pol·I_stat(t)), "
                "I_stat(t)=I_0 e^{-i omega t}"
            ),
            "sheet_formula": "S = phi(omega) * (pol @ ((sqrt(omega)/eta) * (A-A_f)))",
            "sheet_equals_all_time_under": (
                "A-A_f = i gamma I / omega  (archived sheet gap 3.47e-17) "
                "and eta=sqrt(gamma/2pi), hence "
                "int_{-inf}^{inf} K e^{-i omega t} dt = 2 pi i eta phi / sqrt(omega)"
            ),
            "split": "S = raw_continuous_[0,T] + B_past + B_future",
            "B_past": "int_{-inf}^{0} dt K(t) (pol·I_stat(t))",
            "B_future": "int_{T}^{inf} dt K(t) (pol·I_stat(t))",
            "raw_continuous": "int_0^T dt K(t) (pol·I(t))",
        },
        "discrete_code_identity": {
            "exact_for_midpoint": (
                "raw = row(0)·(b, bdag, A)(0) + sum_n force_n · Delta A_f(t_n)"
            ),
            "this_equals": "the discrete trapezoid/midpoint of int_0^T K·I along the undriven trajectory",
            "not_equal_to": "B_past or the all-time sheet S",
            "row_T_is_zero": True,
            "no_stored_T_boundary_that_completes_S": True,
        },
        "four_objects": {
            "direct_incident": {
                "math": "A_f(t)=A_f(0) e^{-i omega t}",
                "in_raw": [
                    "contract_initial uses full A(0)=A_rad(0)+A_f(0)",
                    "time loop forces contract against Delta A_f = A_f(t+h)-A_f(t)",
                ],
                "in_full_U0": "left port a_r = A (incident transmission kept)",
                "in_radiated_sheet": "subtracted: field is A-A_f on every port",
            },
            "radiated_sheet": {
                "math": "phi*(sqrt(omega)/eta)*(A-A_f)",
                "code": "q8_m0_g1_radiated_match_20261015.radiated_target",
                "all_ports": True,
                "is_all_time_frequency_overlap_under_sheet_identity": True,
            },
            "t0_adjoint_initial": {
                "math": "row(0)·(b,bdag,A)(0)",
                "code": "contract_initial",
                "is_NOT_the_physical_past_integral_of_K": True,
                "meaning": (
                    "dependence of the [0,T] detector output on the dressed "
                    "initial state; part of raw_[0,T], not B_past"
                ),
            },
            "tT_omitted_tail": {
                "math": "int_{t>T} K(t) (pol·I(t)) dt",
                "code": "absent: adjoint starts from row(T)=0",
                "not_stored": True,
            },
        },
        "full_U0_is_not_S": {
            "left": "u0_from_bundle uses a_r=A (incident + sheet)",
            "right": "a_r=A-A_f, equals sheet at 9.81e-18",
            "omega_0.11": {"abs_U0": 1.0829210928336994, "abs_S": 0.18052090817873326},
        },
        "production_map_unchanged": (
            "U=U0+raw_driven-raw_undriven; V=raw_driven-raw_undriven. "
            "1a algebraic time split remains; A6 is the undriven identification."
        ),
        "R_N_tail_identity": {
            "delta_b_T": "b_inf - b(T)",
            "matrix_form": (
                "R_N(T) = R_N(inf) - 2 Herm[b_inf, delta_b_T^dag] "
                "+ [delta_b_T, delta_b_T^dag]"
            ),
            "Herm": "(M + M^dag)/2",
            "may_drop_R_N_inf_only_under_A1_A6": True,
            "this_round_does_not_set_R_N_inf_to_zero_inside_the_finite_T_identity": True,
        },
    }


def existing_data_recheck(rad, bnd, io):
    p11 = rad["probes"]["0.11"]
    b1024 = bnd["probes"]["1024"]["0.11"]["undriven_versus_U0"]
    b2048 = bnd["probes"]["2048"]["0.11"]["undriven_versus_U0"]
    abs_1024 = b1024["max_abs_raw_u"]
    abs_2048 = b2048["max_abs_raw_u"]
    abs_S = p11["max_abs_radiated"]
    complex_2048_minus_S = p11["max_abs_raw_minus_radiated"]
    moduli_diff = abs(abs_1024 - abs_2048)
    return {
        "radiated_match_already_is_complex_difference": True,
        "omega_0.11": {
            "abs_S": abs_S,
            "abs_full_U0": p11["max_abs_full_U0"],
            "abs_raw_1024": abs_1024,
            "abs_raw_2048": abs_2048,
            "complex_abs_raw2048_minus_S": complex_2048_minus_S,
            "moduli_abs_raw1024_minus_abs_raw2048": moduli_diff,
            "moduli_abs_raw1024_minus_abs_S": abs(abs_1024 - abs_S),
            "moduli_abs_raw2048_minus_abs_S": abs(abs_2048 - abs_S),
            "raw_v_2048": p11["max_abs_raw_v"],
            "initial_history_maxabs_2048": bnd["probes"]["2048"]["0.11"][
                "undriven_initial_history_maxabs"
            ],
            "force_maxabs_2048": bnd["probes"]["2048"]["0.11"]["undriven_force_maxabs"],
        },
        "can_verify_from_archived_JSON": {
            "complex_raw2048_minus_S": True,
            "complex_raw1024_minus_S": False,
            "complex_raw1024_minus_raw2048": False,
            "complex_vectors_themselves": False,
            "lower_bound_abs_z1024_minus_z2048": moduli_diff,
            "upper_bound_from_moduli_only": "useless: <= abs_1024+abs_2048 ~ 0.36",
        },
        "timestep_reading": {
            "moduli_diff_is_not_a_timestep_error_bound": True,
            "finer_step_is_farther_from_abs_S": bool(
                abs(abs_2048 - abs_S) > abs(abs_1024 - abs_S)
            ),
            "therefore_6e-6_is_not_shown_to_be_discretization_toward_S": True,
        },
        "off_packet_probes": {
            key: {
                "complex_abs_raw_minus_S": val["max_abs_raw_minus_radiated"],
                "abs_S": val["max_abs_radiated"],
                "note": "S~0 so the mismatch is the whole windowed raw",
            }
            for key, val in rad["probes"].items()
            if key != "0.11"
        },
        "cited_io_identity_2a_gate_was": io["slice_2a"]["gate_second_AI_spirit_correct_contrast"],
        "that_gate_compared_complex_6e-6_to_moduli_2e-6": True,
        "outside_AI_correction_accepted": (
            "do not treat 2.213e-6 as an upper bound on the 1024/2048 "
            "complex difference, and do not require raw==S"
        ),
    }


def kprime_norm_labels(io):
    s = io["slice_2b"]
    pos = s["norm_phi_prime_minus_phi_sq"]
    neg = s["norm_phi_minus_sq"]
    two_sided = pos + neg
    return {
        "did_not_rerun_Fourier": True,
        "did_not_run_Kprime": True,
        "source": "io_identity.json slice_2b, labels corrected here",
        "measure": (
            "L2(d omega) on the k_shape positive-frequency packet grid; "
            "negative frequencies use omega -> -omega of the same grid"
        ),
        "norm_phi_prime_minus_phi_sq_ARCHIVED_NAME": pos,
        "actual_integral": "int_{omega>0} |phi'_+ - phi|^2 d omega = int |dlt_pos|^2 omega d omega",
        "is_two_sided_L2": False,
        "relabel": "norm_phi_plus_minus_phi_sq",
        "norm_phi_minus_sq": {
            "value": neg,
            "integral": "int_{omega<0} |phi'_-|^2 d omega = int |dlt_neg|^2 |omega| d omega",
            "same_d_omega_measure": True,
            "tail_80_vs_40": [neg, s["norm_phi_minus_sq_40cycle_tail"]],
        },
        "two_sided_if_same_measure": {
            "norm_phi_prime_minus_phi_sq_TWO_SIDED": two_sided,
            "identity": "||phi'-phi||^2 = ||phi'_+-phi||^2 + ||phi'_-||^2",
            "holds_after_relabel": bool(two_sided + 1e-30 >= neg),
            "archived_names_violated_the_identity_because_left_side_was_positive_only": True,
        },
        "not_a_propagation_bug": True,
        "still_do_not_run_Kprime": True,
    }


def kernel_fourier_split():
    log("kernel Fourier split start")
    prot = p1_a2_protocol(0.8)
    period = float(prot.period)
    t_pulse = float(prot.t_pulse)
    t0 = 0.0
    t_T = CYCLES * period
    eta = eta_of(GAMMA)
    om = omega_grid(prot.omega0, t_pulse)
    om = om[om > 0.0]
    phi, nrm = normalize_phi(om, phi_unnormalized(om, 2.0 * prot.omega0, t_pulse))
    dt = period / 256.0
    t_min = -PAST_CYCLES * period
    t_max = t_T + FUTURE_CYCLES * period
    times = np.arange(t_min, t_max + 0.5 * dt, dt)
    log(f"kernel times {times.size}")
    k = kernel_on_times(om, phi, eta, times)
    wt = trapz_weights(times.size, dt)
    past = region_mask(times, t_min, t0)
    win = region_mask(times, t0, t_T)
    fut = region_mask(times, t_T, t_max + 0.25 * dt)
    probes = {}
    for omega in PROBES:
        i = int(np.argmin(np.abs(om - omega)))
        phi_w = complex(phi[i])
        analytic = 2.0j * np.pi * eta * phi_w / np.sqrt(omega)
        complete = fourier_K(k, times, wt, omega)
        b_past = fourier_K(k[past], times[past], wt[past], omega)
        b_win = fourier_K(k[win], times[win], wt[win], omega)
        b_fut = fourier_K(k[fut], times[fut], wt[fut], omega)
        pieces = b_past + b_win + b_fut
        abs_an = abs(analytic)
        abs_c = abs(complete)
        probes[f"{omega:.2f}"] = {
            "omega_grid_nearest": float(om[i]),
            "abs_phi": abs(phi_w),
            "analytic_all_time_2pi_i_eta_phi_over_sqrtw": {
                "abs": abs_an,
                "re": fnum(analytic.real),
                "im": fnum(analytic.imag),
            },
            "numerical_complete_tmin_tmax": {
                "abs": abs(complete),
                "re": fnum(complete.real),
                "im": fnum(complete.imag),
            },
            "window_0_to_T": {
                "abs": abs(b_win),
                "re": fnum(b_win.real),
                "im": fnum(b_win.imag),
            },
            "B_past": {"abs": abs(b_past), "re": fnum(b_past.real), "im": fnum(b_past.imag)},
            "B_future": {"abs": abs(b_fut), "re": fnum(b_fut.real), "im": fnum(b_fut.imag)},
            "abs_complete_minus_analytic": abs(complete - analytic),
            "abs_pieces_minus_complete": abs(pieces - complete),
            "abs_analytic_minus_window": abs(analytic - b_win),
            "abs_complete_minus_window": abs(complete - b_win),
            "rel_tail_over_analytic": (
                abs(analytic - b_win) / abs_an if abs_an > 0.0 else None
            ),
            "rel_tail_over_complete": (
                abs(complete - b_win) / abs_c if abs_c > 0.0 else None
            ),
        }
        log(f"omega {omega} tail/analytic {probes[f'{omega:.2f}']['rel_tail_over_analytic']}")
    dt_coarse = period / 128.0
    times_c = np.arange(t_min, t_max + 0.5 * dt_coarse, dt_coarse)
    k_c = kernel_on_times(om, phi, eta, times_c)
    wt_c = trapz_weights(times_c.size, dt_coarse)
    win_c = region_mask(times_c, t0, t_T)
    b_win_c = fourier_K(k_c[win_c], times_c[win_c], wt_c[win_c], 0.11)
    complete_c = fourier_K(k_c, times_c, wt_c, 0.11)
    log("kernel Fourier split done")
    return {
        "not_a_new_propagation": True,
        "not_adjoint_contraction": True,
        "assumes_stationary_harmonic_current": (
            "ratio (all_time - window)/all_time is independent of I_0; "
            "predicted sheet mismatch scale is that ratio times |S|"
        ),
        "does_not_include_electronic_transient_or_midpoint_discretization": True,
        "gamma_frozen": GAMMA,
        "eta": eta,
        "period": period,
        "t_pulse": t_pulse,
        "T_cycles": CYCLES,
        "T": t_T,
        "t_min_cycles": -PAST_CYCLES,
        "t_max_cycles": CYCLES + FUTURE_CYCLES,
        "dt_over_period": 1.0 / 256.0,
        "n_times": int(times.size),
        "n_omega": int(om.size),
        "phi_normalizer": float(nrm),
        "int_abs_phi2": fnum(np.trapezoid(np.abs(phi) ** 2, om)),
        "probes": probes,
        "omega_0.11_time_quadrature_128_vs_256": {
            "abs_window_128": abs(b_win_c),
            "abs_window_256": probes["0.11"]["window_0_to_T"]["abs"],
            "abs_complete_128": abs(complete_c),
            "abs_complete_256": probes["0.11"]["numerical_complete_tmin_tmax"]["abs"],
            "abs_diff_window": abs(b_win_c - complex(
                probes["0.11"]["window_0_to_T"]["re"],
                probes["0.11"]["window_0_to_T"]["im"],
            )),
        },
    }


def predicted_sheet_mismatch(fourier, rad):
    s = rad["probes"]["0.11"]["max_abs_radiated"]
    p = fourier["probes"]["0.11"]
    rel = p["rel_tail_over_analytic"]
    rel_c = p["rel_tail_over_complete"]
    pred_an = None if rel is None else rel * s
    pred_c = None if rel_c is None else rel_c * s
    observed = rad["probes"]["0.11"]["max_abs_raw_minus_radiated"]
    return {
        "abs_S": s,
        "observed_complex_abs_raw2048_minus_S": observed,
        "predicted_from_analytic_tail_ratio_times_abs_S": pred_an,
        "predicted_from_complete_tail_ratio_times_abs_S": pred_c,
        "ratio_observed_over_predicted_analytic": (
            None if not pred_an else observed / pred_an
        ),
        "reading": (
            "If predicted ~ observed, the 6e-6 is consistent with K Fourier "
            "tails of a stationary current (not a stepper bug). "
            "If predicted << observed, tails do not explain the mismatch. "
            "If predicted >> observed, something in the adjoint is cancelling tails. "
            "This is a scale test, not a proof that I(t) is stationary."
        ),
    }


def still_missing():
    return [
        "complex raw_1024 - raw_2048 at omega=0.11 (needs re-contraction of saved adjoint rows, 4 columns; not done)",
        "complex raw_1024 - S (radiated_match used 2048 only)",
        "explicit B_past and B_future contracted against the actual I_stat of generated_photon_block (this round used K Fourier ratios only)",
        "proof that the discrete undriven trajectory stays on I_stat e^{-i omega t}",
        "A5 existence of T->inf beyond historical bounded Delta I",
        "K' remains blocked",
    ]


def main():
    started = time.perf_counter()
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    log("A6 start")
    rad = load_json(TIME / "undriven_radiated_match.json")
    bnd = load_json(TIME / "undriven_boundary_match.json")
    io = load_json(IO)
    ident = identity_from_code()
    existing = existing_data_recheck(rad, bnd, io)
    norms = kprime_norm_labels(io)
    fourier = kernel_fourier_split()
    pred = predicted_sheet_mismatch(fourier, rad)
    report = {
        "task": "G1 A6 boundary identity; theory + existing data; no K'",
        "date": "2026-10-21",
        "production_G1_pass": False,
        "quasi_boson": "pending",
        "did_not_assemble_UV": True,
        "did_not_call_rows_with_kernels": True,
        "did_not_run_Kprime": True,
        "did_not_start_long_propagation": True,
        "did_not_change_Hamiltonian_P1_phi_eta_gamma_D": True,
        "did_not_overwrite_io_identity_or_window_vs_asymptotic": True,
        "archived_1a_1b": {
            "algebraic_time_split": "keep",
            "R_N_inf_equals_0": "implication under A1-A6 only; A6 not verified",
            "production_G1_pass": False,
        },
        "exact_identity": ident,
        "existing_data": existing,
        "kernel_fourier_split": fourier,
        "predicted_vs_observed_omega_0.11": pred,
        "kprime_norms": norms,
        "still_missing": still_missing(),
        "stop_Kprime": True,
        "stop_reason": (
            "A6 identity is now written; 6e-6 is not required to equal S, "
            "and is not a proven stepper error. K' still needs 1+2 and user approval."
        ),
        "elapsed_s": time.perf_counter() - started,
        "provenance_sha256": provenance([
            __file__,
            TIME / "undriven_radiated_match.json",
            TIME / "undriven_boundary_match.json",
            IO,
        ]),
    }
    write_new_json(OUTJSON, report)
    log(f"wrote {OUTJSON}")


if __name__ == "__main__":
    main()
