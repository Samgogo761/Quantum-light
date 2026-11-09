#!/usr/bin/env python3
"""H2-projected linear resolvent vs Born, and H2→IR→H2 return.

σ_max(L_ph) is a worst-case transient amplification. Discrete L is strictly
causal / nilpotent, so (1-L)^{-1} exists on a finite window. What matters
for the H2 quantum output is

    P_H2 [(1-L)^{-1} - (1+L)]

and the first return ⟨H2|L²|H2⟩. No new physics, no T2, no DC cut.
"""
from __future__ import annotations

import numpy as np

from q8_m0_feedback_op import apply_L, apply_R, norm_dt
from q8_m0_mode_io import response_to_complex_scalar
from q8_m0_photon_op import (
    analyze_phi,
    apply_L_ph,
    inner_phi,
    norm_phi,
    synthesize_A,
)
from q8_m0_s1_kernel import pack_complex
from q8_m0_volterra import collect_A2_B2, volterra_forward, volterra_response_to_complex_F

H2RES_ID = "h2_projected_resolvent_20260923"


def apply_Ln(x, w, mean_d, f, dt, gamma, n: int):
    a = np.asarray(f, dtype=float)
    for _ in range(int(n)):
        a = apply_L(x, w, mean_d, a, dt, gamma)
    return a


def volterra_matches_resolvent(x, w, mean_d, a_src, dt, gamma) -> dict:
    """A_sheet = (1-L)^{-1} A_src  iff  A_sheet - A_src - L A_sheet = 0."""
    a_src = np.asarray(a_src, dtype=float)
    vol = volterra_forward(x, w, mean_d, dt, gamma, a_L_in=a_src)
    a_ex = np.asarray(vol["A_sheet"], dtype=float)
    resid = a_ex - a_src - apply_L(x, w, mean_d, a_ex, dt, gamma)
    den = max(norm_dt(a_ex, dt), 1e-30)
    return {
        "rel_l2": norm_dt(resid, dt) / den,
        "maxabs": float(np.max(np.abs(resid))),
        "finite": bool(vol["finite"]),
    }


def h2_return_ph(x, w, mean_d, phi, omega, times, dt, dw, gamma) -> dict:
    """⟨H2|L_ph^n|H2⟩ and ||L_ph^n H2|| for n=1,2,3."""
    nrm = norm_phi(phi, dw)
    fn = np.asarray(phi) / max(nrm, 1e-30)
    cur = fn
    out = {"phi_norm": nrm}
    for n in (1, 2, 3):
        cur = apply_L_ph(x, w, mean_d, cur, omega, times, dt, dw, gamma)
        ray = inner_phi(fn, cur, dw)
        gn = norm_phi(cur, dw)
        out[f"rayleigh_{n}"] = float(ray)
        out[f"gain_{n}"] = float(gn)
        out[f"off_h2_mass_{n}"] = float(max(gn * gn - ray * ray, 0.0))
    out["h2_to_ir_to_h2"] = float(out["rayleigh_2"])
    return out


def h2_resolvent_ph(x, w, mean_d, phi, omega, times, dt, dw, gamma) -> dict:
    """P_H2[(1-L)^{-1}-(1+L)] on the H2 photon packet."""
    nrm = norm_phi(phi, dw)
    fn = np.asarray(phi) / max(nrm, 1e-30)
    a_src = synthesize_A(fn, omega, times, dw)
    ident = volterra_matches_resolvent(x, w, mean_d, a_src, dt, gamma)
    vol = volterra_forward(x, w, mean_d, dt, gamma, a_L_in=a_src)
    a_ex = np.asarray(vol["A_sheet"], dtype=float)
    a_born = a_src + apply_L(x, w, mean_d, a_src, dt, gamma)
    phi_ex = analyze_phi(a_ex, omega, times, dt)
    phi_born = analyze_phi(a_born, omega, times, dt)
    phi_src = analyze_phi(a_src, omega, times, dt)
    diff = phi_ex - phi_born
    h2_ex = inner_phi(fn, phi_ex, dw)
    h2_born = inner_phi(fn, phi_born, dw)
    h2_diff = inner_phi(fn, diff, dw)
    return {
        "volterra_identity_rel_l2": ident["rel_l2"],
        "h2_component_QS": inner_phi(fn, phi_src, dw),
        "h2_component_exact": float(h2_ex),
        "h2_component_born": float(h2_born),
        "h2_component_diff": float(h2_diff),
        "norm_diff_ph": norm_phi(diff, dw),
        "rel_h2_component": float(abs(h2_diff) / max(abs(h2_ex), 1e-30)),
        "field_rel_l2": norm_dt(a_ex - a_born, dt) / max(norm_dt(a_ex, dt), 1e-30),
        "finite": bool(vol["finite"]),
    }


def h2_a2_resolvent(x, w, mean_d, F, ehat, k_a, dt, gamma, pref) -> dict:
    """A2 from R-only, R[(1+L)A], and exact linear I = R[(1-L)^{-1}A]."""
    resp_r = response_to_complex_scalar(x, w, F, ehat, dt, mean_d)
    c_r = collect_A2_B2(k_a, resp_r["total"], dt, pref)
    vol = volterra_response_to_complex_F(x, w, mean_d, dt, gamma, F, ehat)
    c_ex = collect_A2_B2(k_a, vol["total"], dt, pref)
    hr = np.asarray(resp_r["h_re"], dtype=float)
    hi = np.asarray(resp_r["h_im"], dtype=float)
    hr_b = hr + apply_L(x, w, mean_d, hr, dt, gamma)
    hi_b = hi + apply_L(x, w, mean_d, hi, dt, gamma)
    i_bf = apply_R(x, w, mean_d, hr_b, dt) + 1j * apply_R(x, w, mean_d, hi_b, dt)
    c_bf = collect_A2_B2(k_a, i_bf, dt, pref)
    a_r = complex(c_r["A2"])
    a_bf = complex(c_bf["A2"])
    a_ex = complex(c_ex["A2"])
    return {
        "A2_Ronly": pack_complex(a_r),
        "A2_born_field": pack_complex(a_bf),
        "A2_exact": pack_complex(a_ex),
        "dA2_exact_minus_Ronly": pack_complex(a_ex - a_r),
        "dA2_exact_minus_born_field": pack_complex(a_ex - a_bf),
        "rel_A2_exact_vs_Ronly": float(abs(a_ex - a_r) / max(abs(a_r), 1e-30)),
        "rel_A2_exact_vs_born_field": float(abs(a_ex - a_bf) / max(abs(a_ex), 1e-30)),
        "B2_Ronly": pack_complex(c_r["B2"]),
        "B2_exact": pack_complex(c_ex["B2"]),
        "finite": bool(vol["finite"]),
    }


def classify_h2_return(rayleigh_2: float, resolvent_h2: float, rel_a2: float) -> str:
    mag = max(abs(float(rayleigh_2)), abs(float(resolvent_h2)), abs(float(rel_a2)))
    if mag < 0.05:
        return "small_h2_return"
    if mag < 0.2:
        return "few_percent_to_0p2"
    if mag < 0.7:
        return "nonnegligible"
    return "O1"


def pack_h2_block(ret, res_ph, a2b) -> dict:
    klass = classify_h2_return(
        ret["h2_to_ir_to_h2"],
        res_ph["h2_component_diff"],
        a2b["rel_A2_exact_vs_born_field"],
    )
    return {
        "return": ret,
        "resolvent_ph": res_ph,
        "A2": a2b,
        "h2_class": klass,
        "id": H2RES_ID,
    }
