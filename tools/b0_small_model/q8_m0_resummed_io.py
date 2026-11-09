#!/usr/bin/env python3
"""Approved linear I/O: strong P1, linear R^A, causal (1-L)^{-1}.

Born A2/B2 stay a perturbative benchmark. Production V(θ) is not written.
H2 is a collection packet on the already-solved sheet field.
"""
from __future__ import annotations

import numpy as np

from q8_m0_adapt_bz import net_gross_complex, perk_born_ab
from q8_m0_feedback_op import apply_L, apply_R
from q8_m0_gauge_shift import GENERIC_DK_FRAC, compare_currents, shift_kf
from q8_m0_k1_kernel import trapz_weight
from q8_m0_mode_io import response_to_complex_scalar
from q8_m0_photon_op import analyze_phi, h2_phi, inner_phi, omega_weights, photon_omega_grid, synthesize_A
from q8_m0_response import contact_dmu_perk, prefix_state_dmu_perk
from q8_m0_s1_kernel import pack_complex
from q8_m0_volterra import (
    collect_A2_B2,
    filtered_source_moments,
    h2_kernels,
    two_port_output_moments,
    volterra_forward,
    volterra_response_to_complex_F,
)

RESUM_ID = "linear_feedback_resummation_h2_20260923"
EPS_TESTED = 3.0e-5


def perk_dressed_ab(x, d_perk, a_re, a_im, k_a, dt, pref):
    """Per-k A2/B2 from R_k[A_sheet] with a shared dressed field."""
    ire = prefix_state_dmu_perk(x, a_re, dt) + contact_dmu_perk(d_perk, a_re)
    iim = prefix_state_dmu_perk(x, a_im, dt) + contact_dmu_perk(d_perk, a_im)
    wt = trapz_weight(k_a.shape[0], dt)
    i_f = ire + 1j * iim
    i_fs = ire - 1j * iim
    beta_f = np.einsum("t,ta,tka->k", wt, k_a, i_f)
    beta_fs = np.einsum("t,ta,tka->k", wt, k_a, i_fs)
    return 1j * pref * beta_fs, 1j * pref * beta_f


def h2_channel(packed, geo, prot, gamma, pref) -> dict:
    """Born benchmark, (1+L), and (1-L)^{-1} on the approved H2 packet."""
    times = packed["times"]
    dt = packed["dt"]
    x = packed["X_full"]
    w = packed["w_k"]
    md = packed["mean_D_t"]
    hk = h2_kernels(times, prot.omega0, prot.t_pulse, geo)
    F, ehat, k_a = hk["F"], hk["e_perp"], hk["k_a"]
    a2_b, b2_b = perk_born_ab(x, packed["perk_D_t"], F, ehat, k_a, dt, pref)
    resp = response_to_complex_scalar(x, w, F, ehat, dt, md)
    c_r = collect_A2_B2(k_a, resp["total"], dt, pref)
    vol = volterra_response_to_complex_F(x, w, md, dt, gamma, F, ehat)
    c_ex = collect_A2_B2(k_a, vol["total"], dt, pref)
    hr = np.asarray(resp["h_re"], dtype=float)
    hi = np.asarray(resp["h_im"], dtype=float)
    hr_b = hr + apply_L(x, w, md, hr, dt, gamma)
    hi_b = hi + apply_L(x, w, md, hi, dt, gamma)
    i_bf = apply_R(x, w, md, hr_b, dt) + 1j * apply_R(x, w, md, hi_b, dt)
    c_bf = collect_A2_B2(k_a, i_bf, dt, pref)
    a_r, a_bf, a_ex = complex(c_r["A2"]), complex(c_bf["A2"]), complex(c_ex["A2"])
    b_r, b_ex = complex(c_r["B2"]), complex(c_ex["B2"])
    a2_d, b2_d = perk_dressed_ab(
        x, packed["perk_D_t"], vol["A_sheet_re"], vol["A_sheet_im"], k_a, dt, pref,
    )
    omega = photon_omega_grid(prot.t_pulse, prot.omega0, harm_max=12.0)
    dw = omega_weights(omega)
    ph2 = h2_phi(omega, prot.omega0, prot.t_pulse, ehat)
    a_src = synthesize_A(ph2, omega, times, dw)
    vphi = volterra_forward(x, w, md, dt, gamma, a_L_in=a_src)
    a_ex_ph = np.asarray(vphi["A_sheet"], dtype=float)
    a_born_ph = a_src + apply_L(x, w, md, a_src, dt, gamma)
    dphi = analyze_phi(a_ex_ph - a_born_ph, omega, times, dt)
    delta_ph = inner_phi(ph2, dphi, dw)
    return {
        "A2_born": pack_complex(a_r),
        "B2_born": pack_complex(b_r),
        "A2_born_field": pack_complex(a_bf),
        "A2_res": pack_complex(a_ex),
        "B2_res": pack_complex(b_ex),
        "dA2_res_minus_born": pack_complex(a_ex - a_r),
        "dA2_res_minus_born_field": pack_complex(a_ex - a_bf),
        "rel_A2_res_vs_born": float(abs(a_ex - a_r) / max(abs(a_r), 1e-30)),
        "delta_born_h2_ph": float(delta_ph),
        "A2_born_net": net_gross_complex(a2_b, w),
        "A2_res_net": net_gross_complex(a2_d, w),
        "B2_res_net": net_gross_complex(b2_d, w),
        "a2_born_k": a2_b,
        "b2_born_k": b2_b,
        "a2_res_k": a2_d,
        "b2_res_k": b2_d,
        "A_sheet_re": vol["A_sheet_re"],
        "A_sheet_im": vol["A_sheet_im"],
        "I_res": vol["total"],
        "finite": bool(vol["finite"]),
        "recon_res_minus_perk": float(abs(complex(np.dot(w, a2_d)) - a_ex)),
        "F": F,
        "e_perp": ehat,
        "k_a": k_a,
        "id": RESUM_ID,
    }


def covariant_residual(geo, p, prot, cache, kf, w, *, mu, neel, ntpc, delta_frac=None):
    dk = GENERIC_DK_FRAC if delta_frac is None else np.asarray(delta_frac, dtype=float)
    kf = np.asarray(kf, dtype=float)
    w = np.asarray(w, dtype=float)
    j0 = cache.assemble(kf, w)["mu_per_cell"]
    kf_p = shift_kf(geo, kf, delta_frac=dk)
    kf_m = shift_kf(geo, kf, delta_frac=-dk)
    cache.fill(geo, p, prot, kf_p, mu=mu, neel=neel, ntpc=ntpc)
    cache.fill(geo, p, prot, kf_m, mu=mu, neel=neel, ntpc=ntpc)
    jp = cache.assemble(kf_p, w)["mu_per_cell"]
    jm = cache.assemble(kf_m, w)["mu_per_cell"]
    dt = float(cache.dt)
    return {
        "delta_frac": np.asarray(dk, dtype=float).tolist(),
        "n_k": int(kf.shape[0]),
        "compare": compare_currents(j0, jp, jm, dt),
    }


def leave_one_out_scores(packed, geo, prot, gamma, pref, ch) -> np.ndarray:
    """|ΔA2_res| when one weight is dropped and the rest renormalized."""
    w = np.asarray(packed["w_k"], dtype=float)
    x = packed["X_full"]
    d = packed["perk_D_t"]
    md0 = packed["mean_D_t"]
    dt = packed["dt"]
    F, ehat, k_a = ch["F"], ch["e_perp"], ch["k_a"]
    a0 = complex(ch["A2_res"]["re"], ch["A2_res"]["im"])
    nk = int(w.size)
    out = np.zeros(nk, dtype=float)
    for i in range(nk):
        ww = w.copy()
        ww[i] = 0.0
        s = float(np.sum(ww))
        if s <= 1e-30:
            out[i] = abs(a0)
            continue
        ww = ww / s
        md = np.einsum("k,tkab->tab", ww, d)
        vol = volterra_response_to_complex_F(x, ww, md, dt, gamma, F, ehat)
        c = collect_A2_B2(k_a, vol["total"], dt, pref)
        out[i] = abs(complex(c["A2"]) - a0)
    return out


def resummed_moments(packed, ch, gamma, pref) -> dict:
    noise = filtered_source_moments(
        packed["X_full"], packed["w_k"], packed["mean_D_t"],
        packed["dt"], gamma, ch["k_a"],
    )
    n_src = pref * noise["n_F"]
    m_src = -pref * noise["m_F"]
    comm_s = pref * noise["commutator_F"]
    a2 = complex(ch["A2_res"]["re"], ch["A2_res"]["im"])
    b2 = complex(ch["B2_res"]["re"], ch["B2_res"]["im"])
    mom = two_port_output_moments(a2, b2, n_src, m_src, comm_s)
    born = two_port_output_moments(
        complex(ch["A2_born"]["re"], ch["A2_born"]["im"]),
        complex(ch["B2_born"]["re"], ch["B2_born"]["im"]),
        n_src, m_src, comm_s,
    )
    comm_ok = bool(abs(complex(mom["commutator"]["re"], mom["commutator"]["im"]) - 1.0) < 0.05)
    return {
        "n_src": pack_complex(n_src),
        "m_src": pack_complex(m_src),
        "comm_s": pack_complex(comm_s),
        "resummed": mom,
        "born_benchmark_same_noise": born,
        "commutator_pass": comm_ok,
        "covariance_not_patched": True,
        "squeezing_claim": "not_evaluated",
        "V_is_diagnostic_only": True,
    }


def vacuum_sheet_rms(packed, ch, gamma, pref, a_p1_peak: float) -> dict:
    """Provisional self-consistency stress diagnostic. Not a fail gate.

    H2-truncated two-port coherent-envelope proxy, not ⟨δA δA⟩ after a
    complete multimode I/O network. Must not by itself declare linear
    failure or open the nonlinear-vs-dissipation fork.
    """
    eta = float(np.sqrt(max(pref, 0.0)))
    ehat = np.asarray(ch["e_perp"], dtype=float)
    F = np.asarray(ch["F"])
    scale = eta * np.sqrt(2.0)
    a_in = scale * np.real(F)[:, None] * ehat[None, :]
    vol = volterra_forward(
        packed["X_full"], packed["w_k"], packed["mean_D_t"],
        packed["dt"], gamma, a_L_in=a_in,
    )
    a_s = np.asarray(vol["A_sheet"], dtype=float)
    rms_t = np.sqrt(np.sum(a_s * a_s, axis=1))
    max_rms = float(np.max(rms_t))
    mean_rms = float(np.sqrt(np.mean(rms_t * rms_t)))
    tested = float(EPS_TESTED)
    return {
        "eta": eta,
        "max_rms": max_rms,
        "mean_rms": mean_rms,
        "maxabs_A_sheet": float(np.max(np.abs(a_s))),
        "eps_tested_R": tested,
        "ratio_to_tested_eps": max_rms / max(tested, 1e-30),
        "a_p1_peak": float(a_p1_peak),
        "ratio_to_p1": max_rms / max(float(a_p1_peak), 1e-30),
        "within_tested_linear": bool(max_rms < 10.0 * tested),
        "small_vs_p1": bool(max_rms < 0.05 * float(a_p1_peak)),
        "exceeds_p1_stress": bool(max_rms > 0.10 * float(a_p1_peak)),
        "linear_selfcons_fail": False,
        "not_a_pass_fail_gate": True,
        "kind": "provisional_self_consistency_stress_diagnostic",
        "note": (
            "H2_truncated_two_port_coherent_envelope_proxy_not_full_multimode_"
            "second_order_and_not_a_linear_failure_gate"
        ),
    }
