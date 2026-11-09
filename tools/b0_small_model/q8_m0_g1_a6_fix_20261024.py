"""A6 diagnostic fix: same-frequency analytic, closed-interval trapz, raw vs b_free.

Read-only. Does not assemble U,V. Does not call rows_with_kernels.
Does not start K' or long propagation. Does not overwrite archived JSON.
production_G1_pass=false.

Does not claim the remaining |complete-analytic| is a time tail.
Does not claim 6e-6 is explained.
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
OUTDIR = ROOT / "results/q_line/g1_a6_fix_20261024"
OUTJSON = OUTDIR / "a6_fix.json"
PROGRESS = OUTDIR / "progress.txt"
TIME = ROOT / "results/q_line/g1_gamma12_time_order_20261015"
OLD_A6 = ROOT / "results/q_line/g1_a6_boundary_identity_20261021/a6_boundary_identity.json"
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


def cpack(z):
    z = complex(z)
    return {"abs": abs(z), "re": fnum(z.real), "im": fnum(z.imag)}


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


def closed_slice(times, k, lo, hi, dt):
    sel = (times >= lo - 0.25 * dt) & (times <= hi + 0.25 * dt)
    ts = times[sel]
    return ts, k[sel], trapz_weights(ts.size, dt)


def operator_split():
    return {
        "raw_is_current_integral": True,
        "raw_is_not_independent_b_free": True,
        "observation_row": "q = K(u*, u, -D) in q8_m0_output_transpose_20261008",
        "b0": "b_free + int_R K(t) I_0(t) dt",
        "bT": "b0 + int_0^T K(t) [I_dr(t) - I_0(t)] dt",
        "b_inf_minus_bT_only_if_same_prehistory_and_full_readout": (
            "int_T^inf K(t) [I_dr(t) - I_0(t)] dt"
        ),
        "tail_to_explain": "driven minus undriven reference, not I_dr tail alone",
        "objects": {
            "raw_undriven": "int_0^T K I_0 along the undriven trajectory; no independent b_free",
            "radiated_sheet": "phi*(sqrt(omega)/eta)*(A-A_f); not full U0",
            "full_U0": "u0_from_bundle; left a_r=A keeps transmission, right A-A_f",
        },
        "do_not_collapse_raw_sheet_U0": True,
    }


def archived_uv(rad):
    probes = {}
    missing_complex_v = True
    for key, val in rad["probes"].items():
        probes[key] = {
            "max_abs_raw_minus_radiated": val["max_abs_raw_minus_radiated"],
            "max_abs_raw_v": val["max_abs_raw_v"],
            "max_abs_radiated": val["max_abs_radiated"],
            "max_abs_full_U0": val["max_abs_full_U0"],
            "complex_raw_minus_S_available": True,
            "complex_V_columns_available": False,
            "raw_v_is_max_abs_only": True,
        }
    return {
        "source": str(TIME / "undriven_radiated_match.json"),
        "identity_archived": rad["identity"],
        "full_U0_also_contains_direct_transmission_on_the_left": rad.get(
            "full_U0_also_contains_direct_transmission_on_the_left"
        ),
        "missing_complex_V_columns": missing_complex_v,
        "U_and_V_are_the_same_boundary_problem": True,
        "probes": probes,
        "do_not_treat_6e-6_as_time_tail": True,
        "omega_0.11_raw_minus_S_is_already_complex_abs": True,
    }


def kernel_fourier_fix():
    log("kernel Fourier fix start")
    prot = p1_a2_protocol(0.8)
    period = float(prot.period)
    t_pulse = float(prot.t_pulse)
    omega0 = float(prot.omega0)
    t0 = 0.0
    t_T = CYCLES * period
    eta = eta_of(GAMMA)
    om = omega_grid(omega0, t_pulse)
    om = om[om > 0.0]
    phi, nrm = normalize_phi(om, phi_unnormalized(om, 2.0 * omega0, t_pulse))
    dt = period / 256.0
    t_min = -PAST_CYCLES * period
    t_max = t_T + FUTURE_CYCLES * period
    times = np.arange(t_min, t_max + 0.5 * dt, dt)
    log(f"kernel times {times.size}")
    k = kernel_on_times(om, phi, eta, times)
    wt = trapz_weights(times.size, dt)
    past_m = region_mask(times, t_min, t0)
    win_m = region_mask(times, t0, t_T)
    fut_m = region_mask(times, t_T, t_max + 0.25 * dt)
    ts_p, ks_p, wt_p = closed_slice(times, k, t_min, t0, dt)
    ts_w, ks_w, wt_w = closed_slice(times, k, t0, t_T, dt)
    ts_f, ks_f, wt_f = closed_slice(times, k, t_T, t_max, dt)
    log(
        f"closed n past/win/fut {ts_p.size} {ts_w.size} {ts_f.size}"
    )
    probes = {}
    for omega in PROBES:
        i = int(np.argmin(np.abs(om - omega)))
        phi_nearest = complex(phi[i])
        phi_exact = complex(
            phi_unnormalized(np.array([omega]), 2.0 * omega0, t_pulse)[0] / nrm
        )
        analytic_old = 2.0j * np.pi * eta * phi_nearest / np.sqrt(omega)
        analytic = 2.0j * np.pi * eta * phi_exact / np.sqrt(omega)
        complete = fourier_K(k, times, wt, omega)
        b_past_old = fourier_K(k[past_m], times[past_m], wt[past_m], omega)
        b_win_old = fourier_K(k[win_m], times[win_m], wt[win_m], omega)
        b_fut_old = fourier_K(k[fut_m], times[fut_m], wt[fut_m], omega)
        pieces_old = b_past_old + b_win_old + b_fut_old
        b_past = fourier_K(ks_p, ts_p, wt_p, omega)
        b_win = fourier_K(ks_w, ts_w, wt_w, omega)
        b_fut = fourier_K(ks_f, ts_f, wt_f, omega)
        pieces = b_past + b_win + b_fut
        abs_an = abs(analytic)
        abs_c = abs(complete)
        probes[f"{omega:.2f}"] = {
            "omega_probe": float(omega),
            "omega_grid_nearest": float(om[i]),
            "abs_phi_nearest": abs(phi_nearest),
            "abs_phi_exact": abs(phi_exact),
            "abs_phi_nearest_minus_exact": abs(phi_nearest - phi_exact),
            "analytic_nearest_grid_phi_OLD": cpack(analytic_old),
            "analytic_exact_omega": cpack(analytic),
            "abs_analytic_old_minus_exact": abs(analytic_old - analytic),
            "numerical_complete_tmin_tmax": cpack(complete),
            "window_closed_0_to_T": cpack(b_win),
            "B_past_closed": cpack(b_past),
            "B_future_closed": cpack(b_fut),
            "window_mask_OLD": cpack(b_win_old),
            "abs_complete_minus_analytic_exact": abs(complete - analytic),
            "abs_complete_minus_analytic_old": abs(complete - analytic_old),
            "bookkeeping_old_pieces_minus_complete": abs(pieces_old - complete),
            "bookkeeping_only": True,
            "closed_pieces_minus_complete": abs(pieces - complete),
            "abs_analytic_exact_minus_window_closed": abs(analytic - b_win),
            "abs_complete_minus_window_closed": abs(complete - b_win),
            "rel_complete_minus_analytic_over_analytic": (
                abs(complete - analytic) / abs_an if abs_an > 0.0 else None
            ),
            "rel_complete_minus_window_over_complete": (
                abs(complete - b_win) / abs_c if abs_c > 0.0 else None
            ),
            "do_not_call_remaining_error_a_time_tail": True,
        }
        log(
            f"omega {omega} nearest {om[i]} "
            f"dphi {abs(phi_nearest - phi_exact)} "
            f"|c-an| {abs(complete - analytic)}"
        )
    log("kernel Fourier fix done")
    return {
        "not_a_new_propagation": True,
        "not_adjoint_contraction": True,
        "does_not_include_electronic_transient_or_midpoint_discretization": True,
        "gamma_frozen": GAMMA,
        "eta": eta,
        "period": period,
        "t_pulse": t_pulse,
        "omega0": omega0,
        "T_cycles": CYCLES,
        "T": t_T,
        "dt_over_period": 1.0 / 256.0,
        "n_times": int(times.size),
        "n_omega": int(om.size),
        "phi_normalizer_from_grid": float(nrm),
        "int_abs_phi2": fnum(np.trapezoid(np.abs(phi) ** 2, om)),
        "closed_n": {
            "past": int(ts_p.size),
            "window": int(ts_w.size),
            "future": int(ts_f.size),
        },
        "probes": probes,
        "same_frequency_rule": (
            "analytic uses phi_unnormalized(omega_probe)/nrm with nrm from "
            "normalize_phi on the production omega grid; not phi[argmin]"
        ),
        "closed_trapz_rule": (
            "each of [t_min,0], [0,T], [T,t_max] has its own trapz half-weights; "
            "junctions 0 and T are shared endpoints"
        ),
    }


def main():
    started = time.perf_counter()
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    log("A6 fix start")
    rad = load_json(TIME / "undriven_radiated_match.json")
    old = load_json(OLD_A6) if OLD_A6.exists() else None
    fourier = kernel_fourier_fix()
    p11 = fourier["probes"]["0.11"]
    report = {
        "task": "G1 A6 diagnostic fix; same-frequency analytic + closed trapz; no K'",
        "date": "2026-10-24",
        "production_G1_pass": False,
        "quasi_boson": "pending",
        "did_not_assemble_UV": True,
        "did_not_call_rows_with_kernels": True,
        "did_not_run_Kprime": True,
        "did_not_start_long_propagation": True,
        "did_not_overwrite_a6_boundary_identity_json": True,
        "did_not_change_Hamiltonian_P1_phi_eta_gamma_D": True,
        "did_not_claim_residual_is_time_tail": True,
        "operator_split": operator_split(),
        "archived_U_and_V": archived_uv(rad),
        "kernel_fourier_fix": fourier,
        "omega_0.11_headline": {
            "omega_grid_nearest": p11["omega_grid_nearest"],
            "abs_phi_nearest_minus_exact": p11["abs_phi_nearest_minus_exact"],
            "abs_analytic_old_minus_exact": p11["abs_analytic_old_minus_exact"],
            "abs_complete_minus_analytic_exact": p11["abs_complete_minus_analytic_exact"],
            "abs_complete_minus_analytic_old": p11["abs_complete_minus_analytic_old"],
            "bookkeeping_old_pieces_minus_complete": p11[
                "bookkeeping_old_pieces_minus_complete"
            ],
            "closed_pieces_minus_complete": p11["closed_pieces_minus_complete"],
            "archived_raw2048_minus_S": rad["probes"]["0.11"]["max_abs_raw_minus_radiated"],
            "archived_raw_v": rad["probes"]["0.11"]["max_abs_raw_v"],
            "not_a_proof_that_6e-6_is_a_time_tail": True,
        },
        "old_a6_json_cited_not_overwritten": bool(old is not None),
        "stop_Kprime": True,
        "elapsed_s": time.perf_counter() - started,
        "provenance_sha256": provenance(
            [
                __file__,
                TIME / "undriven_radiated_match.json",
                OLD_A6,
            ]
        ),
    }
    write_new_json(OUTJSON, report)
    log(f"wrote {OUTJSON}")


if __name__ == "__main__":
    main()
