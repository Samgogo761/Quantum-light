#!/usr/bin/env python3
"""Stage-1 small-sample I/O validation. Local CPU. No 96/192. No g2.

1. Unperturbed sixk with X_full and ⟨D(t)⟩.
2. A_cl ± ε h along ∥ and ⊥; compare FD current to R^A (state + contact).
3. H2⊥ K(t)∝∫ φ/√ω e^{iωt} source β, n, m in kernel units.
4. Response α_± from real-linear R on Re/Im of K_a (no extra trajectory).
5. V(θ) only as kernel-unit diagnostic; squeezing not evaluated (S unset).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import choose_probe_k, dump, propagate_cjj  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_k1_chi_interface import ChiNotChosenError, apply_chi_to_shape_kernel  # noqa: E402
from q8_m0_k1_kernel import (  # noqa: E402
    k_vector_perp,
    response_mode_coefficients,
    source_beta_nm,
)
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_response import (  # noqa: E402
    ResponseNotChosenError,
    apply_response_to_output,
    predict_delta_mu,
    test_waveform_sin2_dir,
)
from q8_m0_s1_kernel import pack_complex, parallel_perp_basis  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1_io_20260921"
MU = -0.009010620372198276
NTPC = 400
EPS_LIST = (3e-5, 1e-4, 3e-4, 1e-3)
THETA = np.linspace(0.0, np.pi, 181, endpoint=True)


def _rel(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-30))


def run_one(*, geo, prot, kf, neel, a_extra, save_full_x, save_mean_D, verbose=False):
    return propagate_cjj(
        geo, R0, prot, nt_per_cycle=NTPC, neel=neel, sign=+1, mu=MU,
        verbose=verbose, kf_frac=kf, save_full_x=save_full_x, a_extra=a_extra,
        save_mean_D=save_mean_D,
    )


def _extra_fn(eps, h_unit, dt):
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


def fd_vs_response(geo, prot, kf, r0, ehat, name: str) -> dict:
    times = np.asarray(r0["times"], dtype=float)
    dt = float(r0["dt"])
    h_unit = test_waveform_sin2_dir(times, prot.t_pulse, ehat)
    x = np.asarray(r0["X_full"])
    w = np.asarray(r0["w_k"], dtype=float)
    mean_d = np.asarray(r0["mean_D_t"])
    rows = []
    for eps in EPS_LIST:
        rp = run_one(geo=geo, prot=prot, kf=kf, neel=+1, a_extra=_extra_fn(eps, h_unit, dt),
                     save_full_x=False, save_mean_D=False)
        rm = run_one(geo=geo, prot=prot, kf=kf, neel=+1, a_extra=_extra_fn(-eps, h_unit, dt),
                     save_full_x=False, save_mean_D=False)
        fd = 0.5 * (np.asarray(rp["mu_per_cell"]) - np.asarray(rm["mu_per_cell"]))
        pred = predict_delta_mu(x, w, eps * h_unit, dt, mean_d)
        tot = pred["total"]
        rows.append({
            "eps": float(eps),
            "rel_total": _rel(fd, tot),
            "rel_state_only": _rel(fd, pred["state"]),
            "max_abs_fd": float(np.max(np.abs(fd))),
            "max_abs_pred": float(np.max(np.abs(tot))),
            "max_abs_contact": float(np.max(np.abs(pred["contact"]))),
            "max_abs_state": float(np.max(np.abs(pred["state"]))),
            "contact_over_state": float(
                np.max(np.abs(pred["contact"])) / max(np.max(np.abs(pred["state"])), 1e-30)
            ),
        })
    return {"direction": name, "ehat": np.asarray(ehat, dtype=float).tolist(), "eps_scan": rows}


def kernel_nm_and_V(r0, geo, prot) -> dict:
    times = np.asarray(r0["times"], dtype=float)
    dt = float(r0["dt"])
    x = np.asarray(r0["X_full"])
    w = np.asarray(r0["w_k"], dtype=float)
    mu_t = np.asarray(r0["mu_per_cell"], dtype=float)
    mean_d = np.asarray(r0["mean_D_t"])
    krec = k_vector_perp(times, prot.omega0, prot.t_pulse, geo)
    k_a = np.asarray(krec["K_a"])
    src = source_beta_nm(x, w, mu_t, k_a, dt)
    ur, ui = np.real(k_a), np.imag(k_a)
    pr = predict_delta_mu(x, w, ur, dt, mean_d)
    pi = predict_delta_mu(x, w, ui, dt, mean_d)
    ru = pr["total"] + 1j * pi["total"]
    rus = pr["total"] - 1j * pi["total"]
    coef = response_mode_coefficients(ru, rus, k_a, dt)
    ap = complex(coef["alpha_plus_shape"]["re"], coef["alpha_plus_shape"]["im"])
    am = complex(coef["alpha_minus_shape"]["re"], coef["alpha_minus_shape"]["im"])
    n_src = complex(src["n_source_shape"]["re"], src["n_source_shape"]["im"])
    m_src = complex(src["m_source_shape"]["re"], src["m_source_shape"]["im"])
    n_resp = abs(am) ** 2
    m_resp = ap * am
    v_kernel = np.real(n_src) + np.real(np.exp(-2j * THETA) * m_src)
    vmin = float(np.min(v_kernel))
    return {
        "source": src,
        "response_coefficients": coef,
        "term_ledger": {
            "n_in_vacuum": "0_for_normal_occupation_of_b_in",
            "n_source_C": src["n_source_shape"],
            "n_response_vac_|alpha_minus|^2": pack_complex(n_resp),
            "m_source_C": src["m_source_shape"],
            "m_response_alpha_plus_alpha_minus": pack_complex(m_resp),
            "bare_b_in_added_to_alpha_plus": "not_added_SI_units_inconsistent",
            "vacuum_half": "not_mixed_into_kernel_units",
            "contact": "included_inside_R_predictions_used_for_alpha",
            "do_not_add_alpha_into_V": (
                "α_± use K_shape as A_in shape without SI; |α|~O(10^3) is not a bosonic gain. "
                "V_kernel below is SOURCE n,m only."
            ),
        },
        "V_kernel_theta": {
            "theta": THETA.tolist(),
            "V_source_only_without_vacuum_half": v_kernel.tolist(),
            "min": vmin,
            "n_source_real": float(np.real(n_src)),
            "m_source": pack_complex(m_src),
            "kind": "finite_time_source_kernel_units_not_bosonic_V",
        },
        "squeezing_claim": "not_evaluated",
        "K_vs_psi_mid_rel": float(
            abs(krec["K_shape"][len(times) // 2] - krec["psi"][len(times) // 2])
            / max(abs(krec["psi"][len(times) // 2]), 1e-30)
        ),
        "time_window": krec["time_window"],
        "si_prefactor": krec["si_prefactor"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    kf = choose_probe_k(geo, R0, MU)["kf"]
    e_par, e_perp = parallel_perp_basis(geo)
    print("[stage1] unperturbed sixk + X + D", flush=True)
    r0 = run_one(geo=geo, prot=prot, kf=kf, neel=+1, a_extra=None,
                 save_full_x=True, save_mean_D=True, verbose=True)
    chi_blocked = False
    try:
        apply_chi_to_shape_kernel(None, None)
    except ChiNotChosenError:
        chi_blocked = True
    r_blocked = False
    try:
        apply_response_to_output()
    except ResponseNotChosenError:
        r_blocked = True
    print("[stage1] FD vs R^A", flush=True)
    fd_par = fd_vs_response(geo, prot, kf, r0, e_par, "parallel_a2")
    fd_perp = fd_vs_response(geo, prot, kf, r0, e_perp, "perp_H2")
    print("[stage1] kernel n,m", flush=True)
    nm = kernel_nm_and_V(r0, geo, prot)
    gates = {
        "contact_included": all(
            row["max_abs_contact"] > 0.0
            for rec in (fd_par, fd_perp) for row in rec["eps_scan"]
        ),
        "fd_rel_shrinks_with_eps_parallel": (
            fd_par["eps_scan"][0]["rel_total"] < fd_par["eps_scan"][-1]["rel_total"]
        ),
        "smallest_eps_rel_parallel": fd_par["eps_scan"][0]["rel_total"],
        "smallest_eps_rel_perp": fd_perp["eps_scan"][0]["rel_total"],
        "K_differs_from_psi": nm["K_vs_psi_mid_rel"] > 1e-6,
        "chi_si_blocked": chi_blocked,
        "response_photon_blocked": r_blocked,
        "squeezing_claim": "not_evaluated",
        "unresolved_magnetic_promoted": False,
        "nk96": "not_run",
        "time_window": "finite_Tp_validation",
    }
    out = {
        "spec": stage1_spec(),
        "protocol_id": prot.protocol_id,
        "nt_per_cycle": NTPC,
        "n_k": 6,
        "neel": "+1_method_validation_only",
        "unperturbed_wall_s": float(r0["wall_seconds"]),
        "fd_vs_RA": {"parallel": fd_par, "perp": fd_perp},
        "output_second_moments": nm,
        "gates": gates,
        "photon_output": "not_run_as_farfield",
        "g2": "not_run",
        "quantum_input": "not_run",
    }
    dump(OUT / "stage1_validation.json", out)
    np.savez_compressed(
        OUT / "V_kernel_theta.npz",
        theta=THETA,
        V_kernel=np.asarray(nm["V_kernel_theta"]["V_source_only_without_vacuum_half"]),
    )
    print(json.dumps({
        "wrote": str(OUT / "stage1_validation.json"),
        "smallest_eps_rel_par": gates["smallest_eps_rel_parallel"],
        "smallest_eps_rel_perp": gates["smallest_eps_rel_perp"],
        "K_vs_psi": nm["K_vs_psi_mid_rel"],
        "n_source": nm["source"]["n_source_shape"],
        "squeezing_claim": "not_evaluated",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
