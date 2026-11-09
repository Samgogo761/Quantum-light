#!/usr/bin/env python3
"""Stage-1b: same-mode I/O closure on six k-points. Local CPU.

1. Errata for the 2026-09-21 validation JSON (file kept unchanged).
2. Mode-waveform FD vs R^A (Re F, Im F) plus even remainder; one 400→800 dt check.
3. +N and −N O(λ²) n, m, V(θ) vs vacuum 1/2.

Not 96/192, not g2, not a BZ magnetic claim, not a squeezing claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import choose_probe_k, dump, propagate_cjj  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_mode_io import (  # noqa: E402
    MODE_IO_ID,
    close_output_moments,
    f_and_psi,
    f_vs_psi_shape_metrics,
    mode_waveforms,
    term_ledger_spec,
)
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_response import predict_delta_mu  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OLD_JSON = REPO / "results/q_line/stage1_io_20260921/stage1_validation.json"
OUT = REPO / "results/q_line/stage1b_mode_io_20260921"
MU = -0.009010620372198276
NTPC = 400
NTPC_FINE = 800
TARGET_MAX_KAPPA = 3e-5
THETA = np.linspace(0.0, np.pi, 181, endpoint=True)


def _rel(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-30))


def extra_fn(eps, h_unit, dt):
    hu = np.asarray(h_unit, dtype=float)
    nmax = hu.shape[0] - 1

    def fn(t):
        x = float(t) / dt
        i = int(np.floor(x))
        if i < 0:
            return eps * hu[0]
        if i >= nmax:
            return eps * hu[-1]
        f = x - i
        return eps * ((1.0 - f) * hu[i] + f * hu[i + 1])

    return fn


def run_one(*, geo, prot, kf, neel, ntpc, a_extra, save_full_x, save_mean_D, verbose=False):
    return propagate_cjj(
        geo, R0, prot, nt_per_cycle=ntpc, neel=neel, sign=+1, mu=MU,
        verbose=verbose, kf_frac=kf, save_full_x=save_full_x, a_extra=a_extra,
        save_mean_D=save_mean_D,
    )


def build_errata(old: dict, shape_801: dict) -> dict:
    par = old["fd_vs_RA"]["parallel"]["eps_scan"]
    perp = old["fd_vs_RA"]["perp"]["eps_scan"]
    nm = old["output_second_moments"]
    return {
        "parent_json": str(OLD_JSON.as_posix()),
        "parent_unmodified": True,
        "current_response_accepted": True,
        "scope_of_0.087pct_and_0.20pct": {
            "what": (
                "Overall max-norm relative error of the two Cartesian current "
                "components over all times, for two INPUT test-field directions."
            ),
            "not": "per-component accuracy of every weak output channel",
            "waveform": "carrier-free sin^2(pi t/Tp), not the H2 optical-mode A(t)",
            "epsilon_unit": "numerical kappa in Angstrom^{-1}, not SI vector potential",
            "linear_window": "observed for these two envelopes at eps <= 1e-4; not universal",
            "smallest_eps_rel_parallel": par[0]["rel_total"],
            "smallest_eps_rel_perp": perp[0]["rel_total"],
            "contact_off_rel_parallel": par[0]["rel_state_only"],
            "contact_off_rel_perp": perp[0]["rel_state_only"],
            "even_order_remainder": "not_in_parent_json_added_in_stage1b",
            "next_finite_tests": ["mode_Re_Im_waveform", "one_dt_halving_400_to_800"],
            "no_denser_k": True,
        },
        "K_vs_psi": {
            "parent_midpoint_raw_rel": nm["K_vs_psi_mid_rel"],
            "parent_interpretation_retracted": (
                "Raw |F-psi|/|psi|=2.16 is the known narrowband scale "
                "1/sqrt(omega_c)=sqrt(10)≈3.162, not an O(1) shape change."
            ),
            "independent_801_point_rebuild": shape_801,
            "keep_true_frequency_weight": True,
            "1.81pct_is_not_an_n_m_error_bar": True,
        },
        "term_order": {
            "parent_did_not_fold_alpha_into_V": True,
            "no_false_squeezing_from_parent": True,
            "parent_alpha_products_are_higher_order": True,
            "leading_m_response": "B2 = vacuum_input x emission_response interference",
            "do_not_unit_convert_old_alpha_into_V": True,
            "ledger": term_ledger_spec(),
        },
        "gates_note": {
            "fd_rel_shrinks_with_eps_parallel": (
                "Parent OR (smallest<last OR smallest<0.05) is too loose. "
                "Use the tabulated amplitude trend; the numbers did fall."
            ),
            "cpu_profile_35pct": (
                "35% is D's share of the profiled wall time, not a 35% surcharge "
                "on a previous cost. 2–3 min/domain is a short-fragment extrapolation."
            ),
        },
        "area": "Do not pick a numerical S. Connected n,m close with alpha_fs/A_uc.",
    }


def fd_mode_waveform(geo, prot, kf, r0, h_unit, name: str, eps: float) -> dict:
    times = np.asarray(r0["times"], dtype=float)
    dt = float(r0["dt"])
    x = np.asarray(r0["X_full"])
    w = np.asarray(r0["w_k"], dtype=float)
    mean_d = np.asarray(r0["mean_D_t"])
    mu0 = np.asarray(r0["mu_per_cell"])
    ntpc = int(r0["nt_per_cycle"])
    rp = run_one(geo=geo, prot=prot, kf=kf, neel=+1, ntpc=ntpc,
                 a_extra=extra_fn(eps, h_unit, dt), save_full_x=False, save_mean_D=False)
    rm = run_one(geo=geo, prot=prot, kf=kf, neel=+1, ntpc=ntpc,
                 a_extra=extra_fn(-eps, h_unit, dt), save_full_x=False, save_mean_D=False)
    mu_p = np.asarray(rp["mu_per_cell"])
    mu_m = np.asarray(rm["mu_per_cell"])
    fd = 0.5 * (mu_p - mu_m)
    even = 0.5 * (mu_p + mu_m - 2.0 * mu0)
    pred = predict_delta_mu(x, w, eps * np.asarray(h_unit, dtype=float), dt, mean_d)
    tot = pred["total"]
    max_fd = float(np.max(np.abs(fd)))
    max_even = float(np.max(np.abs(even)))
    return {
        "direction": name,
        "eps_on_waveform": float(eps),
        "max_abs_h": float(np.max(np.abs(h_unit))),
        "max_abs_delta_kappa": float(eps * np.max(np.abs(h_unit))),
        "h_t0_over_max": float(
            np.max(np.abs(h_unit[0])) / max(float(np.max(np.abs(h_unit))), 1e-30)
        ),
        "rel_total": _rel(fd, tot),
        "rel_state_only": _rel(fd, pred["state"]),
        "max_abs_fd": max_fd,
        "max_abs_pred": float(np.max(np.abs(tot))),
        "max_abs_even_remainder": max_even,
        "even_over_fd": max_even / max(max_fd, 1e-30),
        "contact_over_state": float(
            np.max(np.abs(pred["contact"])) / max(np.max(np.abs(pred["state"])), 1e-30)
        ),
        "initial_state": "field_free_H0_ground_kept_even_if_h_t0_nonzero",
        "nt_per_cycle": ntpc,
    }


def shape_on_801(geo, prot) -> dict:
    times = np.linspace(0.0, float(prot.t_pulse), 801)
    rec = f_and_psi(times, prot.omega0, prot.t_pulse)
    return f_vs_psi_shape_metrics(
        times, rec["F"], rec["psi"], rec["omega_c"], rec["omega"], rec["phi"]
    )


def moments_row(closed: dict) -> dict:
    return {
        "n_out": closed["n_out"],
        "m_out": closed["m_out"],
        "n_source": closed["term_ledger"]["n_source_O_lambda2"],
        "m_source": closed["term_ledger"]["m_source_O_lambda2"],
        "B2": closed["B2"],
        "A2": closed["A2"],
        "beta_over_sqrt_Ncell": closed["beta_over_sqrt_Ncell"],
        "V_min": closed["V_theta"]["min"],
        "V_max": closed["V_theta"]["max"],
        "theta_at_min": closed["V_theta"]["theta_at_min"],
        "vacuum_V": 0.5,
        "commutator_residual_abs": closed["commutator_check"]["residual_abs"],
        "two_Re_A2": closed["commutator_check"]["two_Re_A2"],
        "s1_commutator": closed["commutator_check"]["s1_commutator"],
        "bogoliubov_like": closed["commutator_check"]["bogoliubov_like_|1+A2|^2_|B2|^2"],
        "abs_A2": closed["commutator_check"]["abs_A2"],
        "kept_order_identity_closed": closed["commutator_check"]["kept_order_identity_closed"],
        "unique_unclosed_item": closed["commutator_check"]["unique_unclosed_item"],
        "covariance_physicality": closed["covariance_physicality"],
        "squeezing_claim": closed["squeezing_claim"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    kf = choose_probe_k(geo, R0, MU)["kf"]

    print("[stage1b] F vs psi shape (801, no electrons)", flush=True)
    shape_801 = shape_on_801(geo, prot)
    old = json.loads(OLD_JSON.read_text(encoding="utf-8"))
    errata = build_errata(old, shape_801)
    dump(OUT / "stage1_validation_errata.json", errata)
    dump(OLD_JSON.parent / "stage1_validation_errata.json", errata)

    print("[stage1b] unperturbed +N 400", flush=True)
    r_plus = run_one(geo=geo, prot=prot, kf=kf, neel=+1, ntpc=NTPC,
                     a_extra=None, save_full_x=True, save_mean_D=True, verbose=True)
    times = np.asarray(r_plus["times"])
    dt = float(r_plus["dt"])
    mw = mode_waveforms(times, prot.omega0, prot.t_pulse, geo)
    h_re = np.asarray(mw["h_re_F"])
    h_im = np.asarray(mw["h_im_F"])
    max_h = max(float(np.max(np.abs(h_re))), float(np.max(np.abs(h_im))), 1e-30)
    eps = TARGET_MAX_KAPPA / max_h
    print(f"[stage1b] mode FD Re/Im  eps={eps:.4e}  |F(0)|/max={mw['abs_F_t0_over_max']:.4f}", flush=True)
    fd_re = fd_mode_waveform(geo, prot, kf, r_plus, h_re, "mode_Re_F_perp", eps)
    fd_im = fd_mode_waveform(geo, prot, kf, r_plus, h_im, "mode_Im_F_perp", eps)

    print("[stage1b] dt-halving 800 on Re F", flush=True)
    r800 = run_one(geo=geo, prot=prot, kf=kf, neel=+1, ntpc=NTPC_FINE,
                   a_extra=None, save_full_x=True, save_mean_D=True, verbose=True)
    mw800 = mode_waveforms(np.asarray(r800["times"]), prot.omega0, prot.t_pulse, geo)
    fd_re_800 = fd_mode_waveform(
        geo, prot, kf, r800, np.asarray(mw800["h_re_F"]), "mode_Re_F_perp_ntpc800", eps
    )

    print("[stage1b] close +N moments", flush=True)
    closed_p = close_output_moments(
        r_plus["X_full"], r_plus["w_k"], r_plus["mu_per_cell"], r_plus["mean_D_t"],
        times, dt, geo, prot,
    )
    print("[stage1b] unperturbed -N 400", flush=True)
    r_minus = run_one(geo=geo, prot=prot, kf=kf, neel=-1, ntpc=NTPC,
                      a_extra=None, save_full_x=True, save_mean_D=True, verbose=True)
    closed_m = close_output_moments(
        r_minus["X_full"], r_minus["w_k"], r_minus["mu_per_cell"], r_minus["mean_D_t"],
        np.asarray(r_minus["times"]), float(r_minus["dt"]), geo, prot,
    )

    out = {
        "spec": {
            **stage1_spec(),
            "mode_io_id": MODE_IO_ID,
            "si_prefactor": "connected_n_m_use_eta2_Ncell_equals_alpha_fs_over_A_uc",
            "S": "not_a_free_parameter_for_connected_moments",
            "squeezing_claim": "not_evaluated_sixk_method_only",
            "photon_output": "finite_time_same_mode_not_farfield",
        },
        "protocol_id": prot.protocol_id,
        "n_k": 6,
        "nt_per_cycle": NTPC,
        "neel_pair": "method_validation_only_not_BZ_magnetic_claim",
        "errata_file": str((OUT / "stage1_validation_errata.json").as_posix()),
        "shape_801": shape_801,
        "shape_pulse_grid": closed_p["shape_metrics"],
        "F_t0": {
            "value": mw["F_at_t0"],
            "abs_over_max": mw["abs_F_t0_over_max"],
            "initial_state_convention": mw["initial_state_if_F_t0_nonzero"],
        },
        "mode_waveform_fd": {
            "target_max_kappa": TARGET_MAX_KAPPA,
            "eps": float(eps),
            "Re": fd_re,
            "Im": fd_im,
            "dt_halving_Re": {"ntpc400": fd_re, "ntpc800": fd_re_800},
        },
        "plus_N": moments_row(closed_p),
        "minus_N": moments_row(closed_m),
        "area": closed_p["area"],
        "term_ledger": closed_p["term_ledger"],
        "commutator_plus_N": closed_p["commutator_check"],
        "commutator_minus_N": closed_m["commutator_check"],
        "zero_coupling_vacuum": {
            "n": 0.0,
            "m": 0.0,
            "V": 0.5,
            "recovered_by_setting_pref_0": True,
        },
        "nk96": "not_run",
        "nk192": "not_run",
        "g2": "not_run",
        "unresolved_magnetic_promoted": False,
        "squeezing_claim": "not_evaluated_sixk_method_only",
    }
    dump(OUT / "stage1b_mode_io.json", {
        **out,
        "plus_N_full": {k: closed_p[k] for k in closed_p if k != "V_theta"},
        "minus_N_full": {k: closed_m[k] for k in closed_m if k != "V_theta"},
        "plus_N_V_theta": closed_p["V_theta"],
        "minus_N_V_theta": closed_m["V_theta"],
    })
    np.savez_compressed(
        OUT / "V_theta_both_neel.npz",
        theta=THETA,
        V_plus=np.asarray(closed_p["V_theta"]["V"]),
        V_minus=np.asarray(closed_m["V_theta"]["V"]),
        V_vacuum=np.full(THETA.shape, 0.5),
    )
    print(json.dumps({
        "wrote": str(OUT / "stage1b_mode_io.json"),
        "shape_801_mid_scaled": shape_801["scaled_midpoint_|F-psi_s|/|F|"],
        "shape_801_L2": shape_801["scaled_full_window_L2_|F-psi_s|/||F||"],
        "fd_rel_Re": fd_re["rel_total"],
        "fd_rel_Im": fd_im["rel_total"],
        "fd_rel_Re_800": fd_re_800["rel_total"],
        "even_over_fd_Re": fd_re["even_over_fd"],
        "V_min_plus": closed_p["V_theta"]["min"],
        "V_min_minus": closed_m["V_theta"]["min"],
        "comm_residual_plus": closed_p["commutator_check"]["residual_abs"],
        "squeezing_claim": "not_evaluated_sixk_method_only",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
