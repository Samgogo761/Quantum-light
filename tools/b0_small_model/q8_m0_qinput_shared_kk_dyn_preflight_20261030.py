#!/usr/bin/env python3
"""Shared-mode k,-k finite-coupling symmetry, cost probe, vacuum short ramp.

Not HHG, not a BSV pulse, not a magnetic selection rule.
Does not rewrite trial/preflight/diag/static json. No G1/H2, SV, BZ, or dense H.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry

MODEL_ID = "M0_shared_mode_k_minus_k_vacuum_preflight_v1"
G_ABS = sh.G_ABS
OMEGA0 = sh.OMEGA0
K_FRAC = list(sh.K_FRAC)
NEEL = sh.NEEL
NMAX_RUN = 8
NMAX_COST = 12
ORDER = 16
RAMP_WALL_CAP_S = 720.0
PREFIX_FRAC = 0.125
TAYLOR_HDT_MAX = 0.4


def cpack(z):
    z = complex(z)
    return {"re": float(z.real), "im": float(z.imag)}


def unpack_c(z):
    if isinstance(z, dict):
        return complex(float(z["re"]), float(z["im"]))
    return complex(z)


def quadrature_from_moments(n_c, m_c, n_max=None, p_top=0.0):
    """Lab-frame quadratures X_θ=(a e^{-iθ}+a† e^{iθ})/2.

    Vacuum variance is 1/4 only if ⟨[a,a†]⟩=1. Truncated Fock gives
    ⟨[a,a†]⟩=1-(n_max+1)p_{n_max}. Formula does not assume a Gaussian state.
    """
    n_c = float(np.real(n_c))
    m_c = complex(m_c)
    abs_m = float(abs(m_c))
    w_sq = n_c - abs_m
    comm_deficit = 0.0 if n_max is None else (int(n_max) + 1) * float(p_top)
    comm = 1.0 - comm_deficit
    vac = 0.25 * comm
    var_min = vac + 0.5 * w_sq
    var_max = vac + 0.5 * (n_c + abs_m)
    var_min_I = 0.25 + 0.5 * w_sq
    var_max_I = 0.25 + 0.5 * (n_c + abs_m)
    theta_min = float((0.5 * (np.angle(m_c) + np.pi)) % np.pi)
    xx = vac + 0.5 * n_c + 0.5 * m_c.real
    pp = vac + 0.5 * n_c - 0.5 * m_c.real
    xp = 0.5 * m_c.imag
    db = float(10.0 * np.log10(var_min_I / 0.25)) if var_min_I > 0.0 else None
    return {
        "convention": "X_theta=(a*exp(-i theta)+adag*exp(i theta))/2",
        "n_c": n_c,
        "m_c": cpack(m_c),
        "abs_m": abs_m,
        "W_sq": w_sq,
        "classical_P_bound_violated": bool(abs_m > n_c + 1e-18),
        "commutator_expect": comm,
        "commutator_deficit": comm_deficit,
        "n_max": None if n_max is None else int(n_max),
        "p_top": float(p_top),
        "theta_min": theta_min,
        "var_min": var_min,
        "var_max": var_max,
        "var_min_if_commutator_1": var_min_I,
        "var_max_if_commutator_1": var_max_I,
        "vacuum_var_if_commutator_1": 0.25,
        "squeeze_fraction_vs_vacuum_I": float(1.0 - var_min_I / 0.25),
        "squeeze_dB_vs_vacuum_I": db,
        "sigma_XP": [[float(xx), float(xp)], [float(xp), float(pp)]],
        "not_free_space_output_squeezing": True,
        "measured_while_coupled": True,
    }


def apply_poxp(v, wp, d1: int):
    V = np.asarray(v, complex).reshape(sh.EL36, sh.EL36, d1)
    V = np.einsum("ai,bj,ijn->abn", wp, wp, V, optimize=True)
    return V.ravel()


def apply_swap_h(model, v, s, Ds=None):
    d1 = model.d1
    w = sh.swap_electronic_factors(v, sh.EL36, d1)
    return sh.swap_electronic_factors(sh.apply_h(model, w, s, Ds=Ds), sh.EL36, d1)


def intertwiner_max_abs(h_left, h_right, u_fn, v):
    return float(np.max(np.abs(u_fn(h_left(v)) - h_right(u_fn(v)))))


def finite_coupling_symmetry(geo, k, n_max: int, rng):
    """Joint maps at finite s and finite g. Not an HHG selection rule."""
    wp = sh.wedge_p36()
    m_ng = sh.build_shared(geo, R0, k, NEEL, n_max=n_max, omega0=OMEGA0, g=G_ABS)
    m_mng = sh.build_shared(geo, R0, k, -NEEL, n_max=n_max, omega0=OMEGA0, g=G_ABS)
    m_n_mg = sh.build_shared(geo, R0, k, NEEL, n_max=n_max, omega0=OMEGA0, g=-G_ABS)
    m_mn_mg = sh.build_shared(geo, R0, k, -NEEL, n_max=n_max, omega0=OMEGA0, g=-G_ABS)
    d1 = m_ng.d1
    v = rng.normal(size=m_ng.dim) + 1j * rng.normal(size=m_ng.dim)
    v = v / np.linalg.norm(v)

    def u_el(x):
        return apply_poxp(x, wp, d1)

    def u_pi(x):
        return sh.apply_photon_parity(x, sh.EL36, d1)

    def u_full(x):
        return sh.apply_photon_parity(apply_poxp(x, wp, d1), sh.EL36, d1)

    rows = {}
    for s in (0.0, 0.4, 1.0):
        h_n = lambda x, s=s: sh.apply_h(m_ng, x, s)
        h_n_mg = lambda x, s=s: sh.apply_h(m_n_mg, x, s)
        h_mn = lambda x, s=s: apply_swap_h(m_mng, x, s)
        h_mn_mg = lambda x, s=s: apply_swap_h(m_mn_mg, x, s)
        rows[f"s_{s}"] = {
            "PoxP_H_N_g_vs_swap_H_minusN_g": intertwiner_max_abs(h_n, h_mn, u_el, v),
            "PoxP_H_N_g_vs_swap_H_minusN_minusg": intertwiner_max_abs(h_n, h_mn_mg, u_el, v),
            "Pi_H_N_g_vs_H_N_minusg": intertwiner_max_abs(h_n, h_n_mg, u_pi, v),
            "PoxP_Pi_H_N_g_vs_swap_H_minusN_minusg": intertwiner_max_abs(h_n, h_mn_mg, u_full, v),
            "PoxP_Pi_H_N_g_vs_swap_H_minusN_same_g": intertwiner_max_abs(h_n, h_mn, u_full, v),
            "H_N_g_vs_H_N_minusg": float(np.max(np.abs(h_n(v) - h_n_mg(v)))),
        }
    return {
        "n_max": n_max,
        "note": (
            "s=0: D(qsX)=I, g-sign is invisible. "
            "Finite s: Pi alone implements X→-X i.e. g→-g at fixed N. "
            "P⊗P alone maps to Swap H(-N,-g) because orbital inversion also reverses bonds. "
            "The joint map is (P⊗P) Pi : H(N,g) → Swap H(-N, same g). "
            "An extra g-flip on top of Pi double-counts A→-A. "
            "Not a quantum-driven HHG selection rule."
        ),
        "residuals": rows,
    }


def truncated_coherent(n_max: int, alpha=0.5):
    c = np.zeros(n_max + 1, complex)
    c[0] = 1.0
    for n in range(1, n_max + 1):
        c[n] = alpha * c[n - 1] / np.sqrt(n)
    nrm = float(np.linalg.norm(c))
    c = c / nrm
    n = np.arange(n_max + 1, dtype=float)
    a = np.diag(np.sqrt(np.arange(1, n_max + 1, dtype=float)), 1).astype(complex)
    mean_a = complex(np.vdot(c, a @ c))
    n_ph = float(np.vdot(c, n * c).real)
    return {
        "alpha": float(alpha),
        "n_max": n_max,
        "norm": 1.0,
        "pre_norm": nrm,
        "n_ph": n_ph,
        "n_ph_ideal_untruncated": float(abs(alpha) ** 2),
        "n_incoherent": n_ph - abs(mean_a) ** 2,
        "mean_a": cpack(mean_a),
        "top_mass": float(abs(c[-1]) ** 2),
    }


def vacuum_coeff(n_max: int):
    c = np.zeros(n_max + 1, complex)
    c[0] = 1.0
    return c


def ground_pair(geo, k):
    ek, mass_k, leak_k = sh.slater36(dyn.electron_ground_state(geo, R0, k, NEEL))
    emk, mass_mk, leak_mk = sh.slater36(dyn.electron_ground_state(geo, R0, -k, NEEL))
    return ek, emk, {
        "mass_k": mass_k, "mass_mk": mass_mk,
        "leak_k": leak_k, "leak_mk": leak_mk,
        "norm_k": float(np.linalg.norm(ek)), "norm_mk": float(np.linalg.norm(emk)),
    }


def spectral_radius_power(model, s: float, rng, n_iter: int = 24):
    v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    v = v / np.linalg.norm(v)
    lam = 0.0
    for _ in range(n_iter):
        hv = sh.apply_h(model, v, s)
        lam = complex(np.vdot(v, hv))
        nrm = float(np.linalg.norm(hv))
        if nrm <= 0.0:
            break
        v = hv / nrm
    return float(abs(lam)), float(nrm)


def taylor_step(model, v, s, dt, order=ORDER, Ds=None):
    Ds = dyn.peierls_at(model.q_units, s, model.n_max, Ds)
    nph = np.arange(model.d1, dtype=float)
    shift = float(np.mean(model.onsite_k) + np.mean(model.onsite_mk) + model.omega0 * np.mean(nph))
    term = np.asarray(v, complex).copy()
    out = term.copy()
    last = 0.0
    for k in range(1, order + 1):
        term = sh.apply_h(model, term, s, Ds=Ds) - shift * term
        term = (-1j * dt / k) * term
        out = out + term
        last = float(np.linalg.norm(term))
    out = np.exp(-1j * shift * dt) * out
    return out, last


def snapshot(model, psi, s, Ds=None):
    psi = np.asarray(psi, complex)
    nrm = float(np.linalg.norm(psi))
    ph = sh.photon_observables(psi, sh.EL36, model.d1)
    mean_a = ph["mean_a"]["re"] + 1j * ph["mean_a"]["im"]
    mean_aa = ph["mean_aa"]["re"] + 1j * ph["mean_aa"]["im"]
    n_ph = float(ph["n_ph"])
    n_inc = float(ph["n_incoherent"])
    Hv = sh.apply_h(model, psi, s, Ds=Ds)
    Jv = sh.apply_current(model, psi, s, Ds=Ds)
    mean_j = np.array([complex(np.vdot(psi, Jv[a])) / (nrm * nrm + 1e-30) for a in range(2)])
    energy = float(np.vdot(psi, Hv).real / (nrm * nrm + 1e-30))
    e_ph = model.omega0 * n_ph
    f_lin = -model.g * float(np.dot(model.e, mean_j.real))
    V = psi.reshape(sh.EL36, sh.EL36, model.d1)
    p_n = np.sum(np.abs(V) ** 2, axis=(0, 1)).real / (nrm * nrm + 1e-30)
    return {
        "norm": nrm,
        "n_e": 8.0,
        "n_ph": n_ph,
        "n_coherent": float(abs(mean_a) ** 2),
        "n_incoherent": n_inc,
        "mean_a": cpack(mean_a),
        "mean_aa": cpack(mean_aa),
        "m_incoherent": cpack(mean_aa - mean_a ** 2),
        "purity": float(ph["purity"]),
        "energy": energy,
        "energy_photon": float(e_ph),
        "energy_matter_plus_coupling": float(energy - e_ph),
        "mean_J": [cpack(z) for z in mean_j],
        "F_linear_minus_g_e_dot_J": float(f_lin),
        "top_fock_mass": float(p_n[-1]),
        "p_n": [float(x) for x in p_n],
        "s": float(s),
        "trace_photon": float(ph["trace"]),
        "quadrature": quadrature_from_moments(
            n_inc, mean_aa - mean_a ** 2, n_max=model.n_max, p_top=float(p_n[-1])),
    }


def dHds_expect(model, psi, s, eps=1e-6):
    hp = sh.apply_h(model, psi, s + eps)
    hm = sh.apply_h(model, psi, s - eps)
    n2 = float(np.vdot(psi, psi).real) + 1e-30
    return complex(np.vdot(psi, (hp - hm) / (2.0 * eps)) / n2)


def probe_apply(model, s, rng, n_repeat=3):
    v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    v = v / np.linalg.norm(v)
    t0 = time.perf_counter()
    hv = None
    for _ in range(n_repeat):
        hv = sh.apply_h(model, v, s)
    dt = (time.perf_counter() - t0) / n_repeat
    herm = float(abs(np.vdot(v, hv) - np.vdot(hv, v).conjugate()))
    return dt, herm, float(np.linalg.norm(hv))


def cost_probe(geo, k, n_max, env, rng):
    model = sh.build_shared(geo, R0, k, NEEL, n_max=n_max, omega0=OMEGA0, g=G_ABS)
    t_apply, herm, _ = probe_apply(model, 1.0, rng)
    rho, _ = spectral_radius_power(model, 1.0, rng)
    n_steps = dyn.n_steps_for(env.t_ramp, rho, requested=32, hdt_max=TAYLOR_HDT_MAX)
    dt = env.t_ramp / n_steps
    t0 = time.perf_counter()
    v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    v = v / np.linalg.norm(v)
    taylor_step(model, v, 0.4, dt)
    t_step = time.perf_counter() - t0
    est_ramp = t_step * n_steps
    state_bytes = model.dim * 16
    return {
        "n_max": n_max,
        "dim": model.dim,
        "n_group": int(model.q_units.size),
        "state_bytes_complex128": state_bytes,
        "dense_H_bytes_complex128": model.dim * model.dim * 16,
        "formed_dense_H": False,
        "apply_h_seconds": t_apply,
        "hermiticity_random": herm,
        "rho_power": rho,
        "planned_ramp_steps": n_steps,
        "planned_dt": dt,
        "timed_taylor_step_seconds": t_step,
        "estimated_full_ramp_seconds": est_ramp,
        "exceeds_12min": bool(est_ramp > RAMP_WALL_CAP_S),
    }, model, rho, n_steps


def propagate(model, psi, t0, t1, env, n_steps, clock, sample_every):
    dt = (t1 - t0) / n_steps
    samples = []
    work = 0.0
    worst_last = 0.0
    for j in range(n_steps):
        clock.check()
        tm = t0 + (j + 0.5) * dt
        s = env.s(tm)
        Ds = dyn.peierls_at(model.q_units, s, model.n_max)
        psi, last = taylor_step(model, psi, s, dt, Ds=Ds)
        worst_last = max(worst_last, last)
        take = sample_every and (j % sample_every == 0 or j == n_steps - 1)
        if take:
            obs = snapshot(model, psi, env.s(t0 + (j + 1) * dt), Ds=None)
            obs["t"] = t0 + (j + 1) * dt
            dhds = dHds_expect(model, psi, env.s(obs["t"]))
            dsdt = env.ds(obs["t"])
            obs["dHds_expect"] = cpack(dhds)
            obs["dsdt"] = float(dsdt)
            obs["power_dHds_dsdt"] = float((dhds * dsdt).real)
            samples.append(obs)
        if (j + 1) % 32 == 0 or j == n_steps - 1:
            nrm = float(np.linalg.norm(psi))
            if (not np.isfinite(psi).all()) or nrm > 2.0 or nrm < 0.5:
                raise RuntimeError(f"norm {nrm} at step {j}")
    if len(samples) >= 2:
        for a, b in zip(samples[:-1], samples[1:]):
            work += 0.5 * (a["power_dHds_dsdt"] + b["power_dHds_dsdt"]) * (b["t"] - a["t"])
    return psi, {
        "steps": n_steps, "dt": dt, "order": ORDER,
        "max_last_taylor_term": worst_last,
        "samples": samples,
        "work_trap_dHds_dsdt": float(work),
    }


def pack_samples(samples):
    if not samples:
        return {}
    keys = ("t", "s", "norm", "n_e", "n_ph", "n_coherent", "n_incoherent",
            "purity", "energy", "energy_photon", "energy_matter_plus_coupling",
            "F_linear_minus_g_e_dot_J", "top_fock_mass", "dsdt", "power_dHds_dsdt")
    tab = {k: [s[k] for s in samples] for k in keys}
    tab["mean_a"] = [s["mean_a"] for s in samples]
    tab["m_incoherent"] = [s["m_incoherent"] for s in samples]
    tab["mean_J"] = [s["mean_J"] for s in samples]
    if samples and "quadrature" in samples[0]:
        tab["W_sq"] = [s["quadrature"]["W_sq"] for s in samples]
        tab["var_min"] = [s["quadrature"]["var_min"] for s in samples]
        tab["theta_min"] = [s["quadrature"]["theta_min"] for s in samples]
        tab["abs_m"] = [s["quadrature"]["abs_m"] for s in samples]
    return tab


def vacuum_product(model, ek, emk):
    c = vacuum_coeff(model.n_max)
    psi = sh.product_state(ek, emk, c)
    nrm = float(np.linalg.norm(psi))
    if abs(nrm - 1.0) > 1e-12:
        psi = psi / nrm
    return psi, nrm


def run_preflight():
    geo = build_geometry()
    k = geo.k_cart(K_FRAC)
    env = dyn.Envelope(OMEGA0, ramp_cycles=1.0, flat_cycles=2.0)
    rng = np.random.default_rng(20261030)
    ek, emk, masses = ground_pair(geo, k)

    sym2 = finite_coupling_symmetry(geo, k, n_max=2, rng=rng)
    r0 = sym2["residuals"]["s_0.0"]
    r04 = sym2["residuals"]["s_0.4"]
    r1 = sym2["residuals"]["s_1.0"]
    step1_ok = (
        r0["H_N_g_vs_H_N_minusg"] < 1e-12
        and r0["PoxP_H_N_g_vs_swap_H_minusN_g"] < 1e-10
        and r0["PoxP_Pi_H_N_g_vs_swap_H_minusN_same_g"] < 1e-10
        and r1["Pi_H_N_g_vs_H_N_minusg"] < 1e-10
        and r1["PoxP_H_N_g_vs_swap_H_minusN_minusg"] < 1e-10
        and r1["PoxP_Pi_H_N_g_vs_swap_H_minusN_same_g"] < 1e-10
        and r04["PoxP_Pi_H_N_g_vs_swap_H_minusN_same_g"] < 1e-10
        and r1["PoxP_H_N_g_vs_swap_H_minusN_g"] > 1e-4
        and r1["PoxP_Pi_H_N_g_vs_swap_H_minusN_minusg"] > 1e-4
        and r1["H_N_g_vs_H_N_minusg"] > 1e-4
    )

    cost8, model8, rho8, n_up8 = cost_probe(geo, k, NMAX_RUN, env, rng)
    cost12, _model12, _rho12, _n12 = cost_probe(geo, k, NMAX_COST, env, rng)
    inputs = {
        "vacuum_nmax8": {
            "kind": "vacuum", "n_max": NMAX_RUN, "norm": 1.0,
            "n_ph": 0.0, "n_ph_ideal": 0.0, "top_mass": 0.0,
        },
        "coherent_alpha_0p5_truncated": {
            "nmax2": truncated_coherent(2),
            "nmax8": truncated_coherent(NMAX_RUN),
            "nmax12": truncated_coherent(NMAX_COST),
            "note": "truncated coherent n_ph is not 0.25 at small n_max; vacuum input is used below",
        },
        "electron_slater": masses,
    }

    dyn_block = {
        "attempted": False,
        "reason": None,
        "driven": None,
        "zero_coupling": None,
    }
    if not step1_ok:
        dyn_block["reason"] = "finite-coupling symmetry residual failed; no propagation"
    elif cost8["exceeds_12min"]:
        dyn_block["reason"] = (
            f"estimated full ramp {cost8['estimated_full_ramp_seconds']:.1f}s > 720s; "
            "prefix-only if prefix estimate is small"
        )

    prefix_steps = max(8, int(np.ceil(n_up8 * PREFIX_FRAC)))
    prefix_est = cost8["timed_taylor_step_seconds"] * prefix_steps
    allow_prefix = step1_ok and prefix_est <= 180.0
    allow_full = step1_ok and (not cost8["exceeds_12min"]) and cost8["estimated_full_ramp_seconds"] <= RAMP_WALL_CAP_S

    if allow_prefix:
        dyn_block["attempted"] = True
        clock = dyn.WallClock(RAMP_WALL_CAP_S)
        psi0, raw_n = vacuum_product(model8, ek, emk)
        s0 = snapshot(model8, psi0, 0.0)
        t_pref = PREFIX_FRAC * env.t_ramp
        try:
            psi, led_p = propagate(model8, psi0, 0.0, t_pref, env, prefix_steps, clock,
                                   sample_every=max(1, prefix_steps // 8))
            snap_p = snapshot(model8, psi, env.s(t_pref))
            elapsed_p = clock.elapsed()
            remain_est = cost8["timed_taylor_step_seconds"] * (n_up8 - prefix_steps)
            did_full = False
            led_f = None
            snap_f = None
            if allow_full and (elapsed_p + remain_est) <= RAMP_WALL_CAP_S:
                n_rest = n_up8 - prefix_steps
                psi, led_f = propagate(model8, psi, t_pref, env.t_ramp, env, n_rest, clock,
                                       sample_every=max(1, n_rest // 8))
                snap_f = snapshot(model8, psi, 1.0)
                did_full = True
            samples = list(led_p["samples"]) + (list(led_f["samples"]) if led_f else [])
            e_end = (snap_f or snap_p)["energy"]
            work = (led_p["work_trap_dHds_dsdt"]
                    + (led_f["work_trap_dHds_dsdt"] if led_f else 0.0))
            dyn_block["driven"] = {
                "g": G_ABS,
                "n_max": NMAX_RUN,
                "dim": model8.dim,
                "input_raw_product_norm": raw_n,
                "rho_power": rho8,
                "prefix_frac": PREFIX_FRAC,
                "t_prefix": t_pref,
                "t_ramp": env.t_ramp,
                "reached_ramp_end": did_full,
                "initial": s0,
                "after_prefix": snap_p,
                "after_ramp": snap_f,
                "delta_n_ph_prefix": snap_p["n_ph"] - s0["n_ph"],
                "delta_n_coherent_prefix": snap_p["n_coherent"] - s0["n_coherent"],
                "delta_n_incoherent_prefix": snap_p["n_incoherent"] - s0["n_incoherent"],
                "delta_energy": e_end - s0["energy"],
                "work_trap_dHds_dsdt": work,
                "energy_minus_work": (e_end - s0["energy"]) - work,
                "F_linear_initial": s0["F_linear_minus_g_e_dot_J"],
                "F_linear_after_prefix": snap_p["F_linear_minus_g_e_dot_J"],
                "F_linear_after_ramp": None if snap_f is None else snap_f["F_linear_minus_g_e_dot_J"],
                "prefix_ledger": {k: led_p[k] for k in ("steps", "dt", "order", "max_last_taylor_term")},
                "ramp_rest_ledger": None if led_f is None else {
                    k: led_f[k] for k in ("steps", "dt", "order", "max_last_taylor_term")},
                "time_series": pack_samples(samples),
                "wall_seconds": clock.elapsed(),
            }
            clock_z = dyn.WallClock(RAMP_WALL_CAP_S)
            model0 = sh.build_shared(geo, R0, k, NEEL, n_max=NMAX_RUN, omega0=OMEGA0, g=0.0)
            psi_z, _ = vacuum_product(model0, ek, emk)
            s0z = snapshot(model0, psi_z, 0.0)
            t_z = env.t_ramp if did_full else t_pref
            n_z = n_up8 if did_full else prefix_steps
            psi_z, led_z = propagate(model0, psi_z, 0.0, t_z, env, n_z, clock_z,
                                     sample_every=max(1, n_z // 8))
            sz = snapshot(model0, psi_z, env.s(t_z))
            dyn_block["zero_coupling"] = {
                "g": 0.0,
                "t_end": t_z,
                "initial": s0z,
                "final": sz,
                "delta_n_ph": sz["n_ph"] - s0z["n_ph"],
                "delta_energy": sz["energy"] - s0z["energy"],
                "delta_norm": sz["norm"] - s0z["norm"],
                "ledger": {k: led_z[k] for k in ("steps", "dt", "order", "max_last_taylor_term")},
                "wall_seconds": clock_z.elapsed(),
            }
            if dyn_block["reason"] is None and not did_full:
                dyn_block["reason"] = "prefix completed; full ramp not started under the 12 min cap"
        except TimeoutError as exc:
            dyn_block["reason"] = str(exc)
            dyn_block["driven"] = {"status": "wall_timeout", "error": str(exc),
                                   "wall_seconds": clock.elapsed()}
    elif dyn_block["reason"] is None:
        dyn_block["reason"] = f"prefix estimate {prefix_est:.1f}s too large; no propagation"

    return {
        "record_date": "2026-10-30",
        "model_id": MODEL_ID,
        "claim_level": "shared_mode_vacuum_short_ramp_preflight_not_magnetic_HHG",
        "not_free_space_BSV_pulse": True,
        "single_k_retained_as_propagator_benchmark_only": True,
        "did_not": [
            "rewrite trial.json", "rewrite preflight.json", "rewrite diag.json",
            "rewrite static.json", "G1", "H2", "HHG spectra", "SV scan",
            "semiclassical trajectories", "full BZ", "AI inverse design",
            "nmax=12 second propagation", "dense joint Hamiltonian",
        ],
        "definition": {
            "H": "omega0 n + He(k, X; N) + He(-k, X; N)",
            "X": "a + a^dagger, shared",
            "h8_frozen": True,
            "neel": NEEL,
            "k_fractional": K_FRAC,
            "g": G_ABS,
            "omega0": OMEGA0,
            "n_max_run": NMAX_RUN,
        },
        "finite_coupling_symmetry": {**sym2, "pass": step1_ok},
        "input_normalization": inputs,
        "cost": {"nmax8": cost8, "nmax12": cost12, "ramp_wall_cap_s": RAMP_WALL_CAP_S},
        "envelope": dyn.envelope_table(env),
        "dynamics": dyn_block,
        "not_a_selection_rule": True,
        "incoherent_occupation_is_not_squeezing": True,
        "files": {
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, sh.__file__, dyn.__file__)
        },
        "stopped": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path,
                    default=Path("results/q_line/qinput_a_shared_kk_dyn_preflight_20261030/preflight.json"))
    args = ap.parse_args(argv)
    if args.report.exists():
        print(f"report already exists: {args.report}", file=__import__("sys").stderr)
        return 2
    t0 = time.perf_counter()
    report = run_preflight()
    report["wall_seconds_total"] = time.perf_counter() - t0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    d = report["dynamics"]
    print(json.dumps({
        "model_id": report["model_id"],
        "sym_pass": report["finite_coupling_symmetry"]["pass"],
        "est_ramp_s_nmax8": report["cost"]["nmax8"]["estimated_full_ramp_seconds"],
        "est_ramp_s_nmax12": report["cost"]["nmax12"]["estimated_full_ramp_seconds"],
        "dyn_attempted": d["attempted"],
        "dyn_reason": d["reason"],
        "reached_ramp_end": None if not d.get("driven") else d["driven"].get("reached_ramp_end"),
        "n_ph_prefix": None if not d.get("driven") else d["driven"]["after_prefix"]["n_ph"],
        "n_coh_prefix": None if not d.get("driven") else d["driven"]["after_prefix"]["n_coherent"],
        "n_inc_prefix": None if not d.get("driven") else d["driven"]["after_prefix"]["n_incoherent"],
        "wall": report["wall_seconds_total"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
