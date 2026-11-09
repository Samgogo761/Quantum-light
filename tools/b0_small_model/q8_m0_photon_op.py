#!/usr/bin/env python3
"""Positive-frequency normalized-photon feedback operator L_ph.

L_ph = Q L S,  L = G_self R^A (open loop, real time).

S: φ(ω) → A(t) = Re ∫_{ω>0} dω [φ(ω)/√ω] e^{+iωt}
Q: A(t) → φ(ω) = √ω A^{(+)}(ω)

The photon inner product is ∫_{ω>0} dω (Re φ·Re ψ + Im φ·Im ψ),
i.e. the real form of ∫ |φ|² dω with ∫|φ|²=1 a normalized packet.

ω=0 is not a coordinate (not a normalizable propagating mode). No
ad-hoc DC cut is applied to L itself.

σ_max(L) and σ_max(L_ph) are worst-case stress diagnostics, not the Born
passport. Born for the H2 output is judged by a covariant k-shift test
plus the H2-projected resolvent error.
"""
from __future__ import annotations

import numpy as np

from q8_m0_feedback_op import (
    apply_L,
    apply_L_perk,
    apply_Ladj,
    classify_sigma,
    inner_dt,
    norm_dt,
)
from q8_m0_k1_packet import normalize_phi, phi_unnormalized
from q8_m0_response import contact_dmu, prefix_state_dmu

PH_OP_ID = "photon_norm_L_ph_Q_L_S_20260922"
INNER_PH = "int_domega_Re_phi_star_psi_omega_gt_0"
ANALYSIS_PREFACTOR = 1.0 / np.pi


def photon_omega_grid(tp: float, omega0: float, *, harm_max: float = 12.0, t_obs: float | None = None) -> np.ndarray:
    """Positive DFT frequencies 2π j / T_obs, j=1..J. Excludes ω=0.

    Default T_obs = Tp. A longer observation window only densifies the
    frequency grid; ω_max stays harm_max * ω0 if T_obs scales the cycle count.
    """
    T = float(tp) if t_obs is None else float(t_obs)
    dw = 2.0 * np.pi / T
    n_cycles = T * float(omega0) / (2.0 * np.pi)
    jmax = int(round(float(harm_max) * n_cycles))
    if jmax < 2:
        raise RuntimeError("photon omega grid too short")
    return dw * np.arange(1, jmax + 1, dtype=float)


def omega_weights(omega: np.ndarray) -> float:
    omega = np.asarray(omega, dtype=float)
    if omega.size < 2:
        raise RuntimeError("need at least two positive frequencies")
    dw = float(omega[1] - omega[0])
    if abs(dw) < 1e-18:
        raise RuntimeError("degenerate omega grid")
    return dw


def inner_phi(p, q, dw: float) -> float:
    p = np.asarray(p)
    q = np.asarray(q)
    return float(dw) * float(np.sum(np.real(p) * np.real(q) + np.imag(p) * np.imag(q)))


def norm_phi(p, dw: float) -> float:
    return float(np.sqrt(max(inner_phi(p, p, dw), 0.0)))


def _phase_plus(times, omega) -> np.ndarray:
    return np.exp(1j * np.asarray(times, dtype=float)[:, None] * np.asarray(omega, dtype=float)[None, :])


def synthesize_A(phi, omega, times, dw: float) -> np.ndarray:
    """S: complex φ(ω,a) → real A(t,a)."""
    phi = np.asarray(phi)
    omega = np.asarray(omega, dtype=float)
    wo = np.sqrt(omega)
    g = phi / wo[:, None]
    F = float(dw) * (_phase_plus(times, omega) @ g)
    return np.real(F)


def synthesize_adj(a_t, omega, times, dt: float) -> np.ndarray:
    """S† in (φ, dw) ↔ (A, dt) inner products."""
    a_t = np.asarray(a_t, dtype=float)
    omega = np.asarray(omega, dtype=float)
    phase = np.exp(-1j * np.asarray(times, dtype=float)[:, None] * omega[None, :])
    integ = float(dt) * (a_t.T @ phase).T
    return integ / np.sqrt(omega)[:, None]


def analyze_phi(a_t, omega, times, dt: float) -> np.ndarray:
    """Q: real A(t) → φ(ω) = √ω A^{(+)}(ω)."""
    a_t = np.asarray(a_t, dtype=float)
    omega = np.asarray(omega, dtype=float)
    phase = np.exp(-1j * np.asarray(times, dtype=float)[:, None] * omega[None, :])
    integ = float(dt) * (a_t.T @ phase).T
    return (np.sqrt(omega)[:, None] * float(ANALYSIS_PREFACTOR)) * integ


def analyze_adj(phi, omega, times, dw: float) -> np.ndarray:
    """Q†."""
    phi = np.asarray(phi)
    omega = np.asarray(omega, dtype=float)
    g = (np.sqrt(omega) * float(ANALYSIS_PREFACTOR))[:, None] * phi
    F = float(dw) * (_phase_plus(times, omega) @ g)
    return np.real(F)


def apply_L_ph(x, w, mean_d, phi, omega, times, dt, dw, gamma) -> np.ndarray:
    a_in = synthesize_A(phi, omega, times, dw)
    a_out = apply_L(x, w, mean_d, a_in, dt, gamma)
    return analyze_phi(a_out, omega, times, dt)


def apply_L_ph_adj(x, w, mean_d, psi, omega, times, dt, dw, gamma) -> np.ndarray:
    g_a = analyze_adj(psi, omega, times, dw)
    g_in = apply_Ladj(x, w, mean_d, g_a, dt, gamma)
    return synthesize_adj(g_in, omega, times, dt)


def qs_reconstruction(phi, omega, times, dt, dw) -> dict:
    a = synthesize_A(phi, omega, times, dw)
    back = analyze_phi(a, omega, times, dt)
    nrm = norm_phi(phi, dw)
    err = norm_phi(back - phi, dw) / max(nrm, 1e-30)
    return {
        "rel_l2": float(err),
        "overlap": inner_phi(phi, back, dw) / max(nrm * nrm, 1e-30),
    }


def h2_phi(omega, omega0: float, tp: float, e_perp) -> np.ndarray:
    raw = phi_unnormalized(omega, 2.0 * float(omega0), tp)
    phi, _ = normalize_phi(omega, raw)
    e = np.asarray(e_perp, dtype=float)
    e = e / max(float(np.linalg.norm(e)), 1e-30)
    return phi[:, None] * e[None, :]


def _orth_phi(cols, dw: float) -> np.ndarray:
    nω, _, p = cols.shape
    out = []
    for j in range(p):
        v = cols[:, :, j].copy()
        for u in out:
            v = v - inner_phi(u, v, dw) * u
        nrm = norm_phi(v, dw)
        if nrm < 1e-14:
            continue
        out.append(v / nrm)
    if not out:
        raise RuntimeError("phi orthonormalize produced no columns")
    stacked = np.zeros((nω, 2, len(out)), dtype=complex)
    for j, v in enumerate(out):
        stacked[:, :, j] = v
    return stacked


def _rotate_phi(cols, evecs) -> np.ndarray:
    return np.einsum("wap,pq->waq", cols, np.asarray(evecs, dtype=float))


def adjoint_residual_ph(x, w, mean_d, phi, psi, omega, times, dt, dw, gamma) -> float:
    lp = apply_L_ph(x, w, mean_d, phi, omega, times, dt, dw, gamma)
    left = inner_phi(psi, lp, dw)
    right = inner_phi(apply_L_ph_adj(x, w, mean_d, psi, omega, times, dt, dw, gamma), phi, dw)
    return float(abs(left - right) / max(abs(left), 1e-30))


def leading_singular_ph(
    x, w, mean_d, omega, times, dt, dw, gamma, phi_probe=None,
    *, p: int = 3, n_iter: int = 5, rng=None,
):
    """Largest singular values of L_ph in the photon inner product."""
    rng = np.random.default_rng(0 if rng is None else rng)
    nω = int(np.asarray(omega).shape[0])
    cols = np.zeros((nω, 2, p), dtype=complex)
    start = 0
    if phi_probe is not None:
        fp = np.asarray(phi_probe)
        nrm = norm_phi(fp, dw)
        if nrm > 0.0:
            cols[:, :, 0] = fp / nrm
            start = 1
    for j in range(start, p):
        cols[:, :, j] = rng.normal(size=(nω, 2)) + 1j * rng.normal(size=(nω, 2))
    V = _orth_phi(cols, dw)
    hist = []
    sigs = np.zeros(int(V.shape[2]), dtype=float)
    for it in range(n_iter):
        pcur = int(V.shape[2])
        W = np.zeros((nω, 2, pcur), dtype=complex)
        for j in range(pcur):
            W[:, :, j] = apply_L_ph(x, w, mean_d, V[:, :, j], omega, times, dt, dw, gamma)
        gram = np.zeros((pcur, pcur), dtype=float)
        for i in range(pcur):
            for j in range(i, pcur):
                gij = inner_phi(W[:, :, i], W[:, :, j], dw)
                gram[i, j] = gij
                gram[j, i] = gij
        evals, evecs = np.linalg.eigh(0.5 * (gram + gram.T))
        order = np.argsort(evals)[::-1]
        sigs = np.sqrt(np.maximum(evals[order], 0.0))
        hist.append(float(sigs[0]))
        V = _rotate_phi(V, evecs[:, order])
        W = _rotate_phi(W, evecs[:, order])
        V = _orth_phi(V, dw)
        if it >= 2 and abs(hist[-1] - hist[-2]) < 1e-3 * max(hist[-1], 1e-12):
            break
        U = np.zeros_like(W)
        for j in range(W.shape[2]):
            nrm = norm_phi(W[:, :, j], dw)
            U[:, :, j] = W[:, :, j] / nrm if nrm > 0.0 else W[:, :, j]
        Vnext = np.zeros_like(U)
        for j in range(U.shape[2]):
            Vnext[:, :, j] = apply_L_ph_adj(
                x, w, mean_d, U[:, :, j], omega, times, dt, dw, gamma,
            )
        V = _orth_phi(Vnext, dw)
    v0 = V[:, :, 0]
    u_raw = apply_L_ph(x, w, mean_d, v0, omega, times, dt, dw, gamma)
    s0 = norm_phi(u_raw, dw)
    u0 = u_raw / s0 if s0 > 0.0 else u_raw
    trial = [float(s) for s in sigs]
    if trial:
        trial[0] = float(s0)
    gain_probe = None
    rayleigh_probe = None
    if phi_probe is not None:
        nrm = norm_phi(phi_probe, dw)
        if nrm > 0.0:
            fn = np.asarray(phi_probe) / nrm
            lf = apply_L_ph(x, w, mean_d, fn, omega, times, dt, dw, gamma)
            gain_probe = norm_phi(lf, dw)
            rayleigh_probe = inner_phi(fn, lf, dw)
    a_v = synthesize_A(v0, omega, times, dw)
    a_lv = apply_L(x, w, mean_d, a_v, dt, gamma)
    return {
        "sigma_max": float(s0),
        "sigma_trial": trial,
        "n_iter": int(len(hist)),
        "sigma_hist": [float(s) for s in hist],
        "gain_h2_dir": None if gain_probe is None else float(gain_probe),
        "rayleigh_h2": None if rayleigh_probe is None else float(rayleigh_probe),
        "sigma_over_h2": (
            None if gain_probe in (None, 0.0)
            else float(s0 / max(float(gain_probe), 1e-30))
        ),
        "v_right": v0,
        "u_left": u0,
        "a_right": a_v,
        "a_out": a_lv,
        "inner": INNER_PH,
        "op_id": PH_OP_ID,
    }


def phi_spectrum(phi, omega) -> dict:
    phi = np.asarray(phi)
    omega = np.asarray(omega, dtype=float)
    dens = np.sum(np.abs(phi) ** 2, axis=1)
    tot = float(np.trapezoid(dens, omega))
    return {
        "omega": omega,
        "abs2": dens,
        "mass": tot,
        "omega_centroid": float(np.trapezoid(omega * dens, omega) / max(tot, 1e-30)),
        "low_frac_below_0p25_H2": float(
            np.trapezoid(dens[omega < 0.25 * 0.1], omega[omega < 0.25 * 0.1]) / max(tot, 1e-30)
        ),
        "h2_band_1p5_to_2p5_omega0": None,
    }


def h2_unweighted_gain(x, w, mean_d, phi, omega, times, dt, dw, gamma) -> dict:
    """Diagnostic: ||L Sφ||_dt / ||Sφ||_dt for the same packet."""
    a = synthesize_A(phi, omega, times, dw)
    la = apply_L(x, w, mean_d, a, dt, gamma)
    na = norm_dt(a, dt)
    return {
        "unweighted_dir_gain": norm_dt(la, dt) / max(na, 1e-30),
        "a_l2": na,
        "La_l2": norm_dt(la, dt),
    }


def rayleigh_perk_ph(u_phi, a_vk, omega, times, dt, dw) -> np.ndarray:
    """α_k = ⟨u, Q L_k S v⟩_ph = ⟨Q† u, L_k A_v⟩_dt."""
    qadj = analyze_adj(u_phi, omega, times, dw)
    return float(dt) * np.einsum("ta,tka->k", qadj, np.asarray(a_vk, dtype=float))


def apply_L_perk_from_A(x, d_perk, a_in, dt, gamma):
    return apply_L_perk(x, d_perk, a_in, dt, gamma)


def _split_norms(arr, sl) -> float:
    a = np.asarray(arr, dtype=float)
    return float(np.sqrt(np.sum(a[sl] * a[sl])))


def gauge_contact_cancel(x, w, mean_d, dt: float, times, ehat) -> dict:
    """Withdrawn as a gauge-fail criterion (kept as a labeled contrast).

    A sudden constant A with fixed Π0 is a quench, not a pure gauge
    transform. A slow sin² δA has E=-Ȧ≠0. Neither implies
    I_state+I_contact→0. The covariant test lives in q8_m0_gauge_shift.
    """
    ehat = np.asarray(ehat, dtype=float)
    ehat = ehat / max(float(np.linalg.norm(ehat)), 1e-30)
    nt = int(np.asarray(x).shape[0])
    h_const = np.broadcast_to(ehat, (nt, 2)).copy()
    tp = float(times[-1])
    ramp = np.sin(np.pi * np.asarray(times, dtype=float) / max(tp, 1e-30)) ** 2
    h_slow = ramp[:, None] * ehat[None, :]

    def _one(h, tag: str) -> dict:
        state = prefix_state_dmu(x, w, h, dt)
        contact = contact_dmu(mean_d, h)
        total = state + contact
        late = slice(int(0.75 * nt), None)
        st = _split_norms(state, late)
        ct = _split_norms(contact, late)
        tot = _split_norms(total, late)
        st_all = float(np.max(np.abs(state)))
        ct_all = float(np.max(np.abs(contact)))
        tot_all = float(np.max(np.abs(total)))
        return {
            "tag": tag,
            "late_|state|": st,
            "late_|contact|": ct,
            "late_|total|": tot,
            "late_cancel_ratio": float(tot / max(st + ct, 1e-30)),
            "max_|state|": st_all,
            "max_|contact|": ct_all,
            "max_|total|": tot_all,
            "max_cancel_ratio": float(tot_all / max(st_all + ct_all, 1e-30)),
        }

    return {
        "constant": _one(h_const, "constant"),
        "slow_sin2_envelope": _one(h_slow, "slow_sin2"),
        "ehat": ehat.tolist(),
    }


def lowest_bin_gains(x, w, mean_d, omega, times, dt, dw, gamma, ehat) -> dict:
    """Gain of the lowest positive ω bin vs H2 bin, photon vs unweighted."""
    omega = np.asarray(omega, dtype=float)
    ehat = np.asarray(ehat, dtype=float)
    ehat = ehat / max(float(np.linalg.norm(ehat)), 1e-30)
    nω = omega.size
    phi0 = np.zeros((nω, 2), dtype=complex)
    phi0[0] = ehat
    phi0 = phi0 / max(norm_phi(phi0, dw), 1e-30)
    lp = apply_L_ph(x, w, mean_d, phi0, omega, times, dt, dw, gamma)
    uw = h2_unweighted_gain(x, w, mean_d, phi0, omega, times, dt, dw, gamma)
    return {
        "omega_min": float(omega[0]),
        "photon_gain": norm_phi(lp, dw),
        "unweighted_gain": uw["unweighted_dir_gain"],
        "photon_over_unweighted": float(
            norm_phi(lp, dw) / max(uw["unweighted_dir_gain"], 1e-30)
        ),
    }


def pack_phi_spec(phi, omega, dw: float, times, dt: float) -> dict:
    phi = np.asarray(phi)
    omega = np.asarray(omega, dtype=float)
    dens = np.sum(np.abs(phi) ** 2, axis=1)
    tot = float(dw * np.sum(dens))
    a = synthesize_A(phi, omega, times, dw)
    nt = int(a.shape[0])
    e = np.sum(a * a, axis=1)
    etot = float(dt * np.sum(e))
    return {
        "omega_centroid": float(dw * np.sum(omega * dens) / max(tot, 1e-30)),
        "abs2_mass": tot,
        "low_frac_lt_0p25_H2": float(dw * np.sum(dens[omega < 0.025]) / max(tot, 1e-30)),
        "h2_band_frac": float(
            dw * np.sum(dens[(omega >= 0.075) & (omega <= 0.125)]) / max(tot, 1e-30)
        ),
        "a_energy_last20": float(dt * np.sum(e[int(0.8 * nt) :])) / max(etot, 1e-30),
        "a_energy_first_cycle": float(dt * np.sum(e[: max(nt // 8, 1)])) / max(etot, 1e-30),
        "a_end_norm": float(np.linalg.norm(a[-1])),
        "a_maxabs": float(np.max(np.abs(a))),
    }
