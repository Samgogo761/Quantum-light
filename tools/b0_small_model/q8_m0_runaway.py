#!/usr/bin/env python3
"""Runaway-source diagnostics for two-port G_self. Same physics, no new knobs.

Does not add T2, does not subtract DC, does not change P1, does not run a BZ.
Reports finite-time amplification of the linear self-consistent solve, not
'the Volterra equation has no solution'.
"""
from __future__ import annotations

import time

import numpy as np

from q8_m0_cjj import choose_probe_k
from q8_m0_classical import (
    initial_rho,
    k_grid,
    kappa_shift,
    trace_current,
)
from q8_m0_k1_kernel import trapz_weight
from q8_m0_model import h8, h8_and_current
from q8_m0_s1_kernel import parallel_perp_basis

RUNAWAY_ID = "runaway_source_diag_20260921"
RECHECK_ID = "runaway_recheck_20260922"
HALT_ABS_I = 1.0e8


def e_from_a_leftrect(a, dt: float) -> np.ndarray:
    """Legacy backward difference. E[0]=0, E[n]=-(A[n]-A[n-1])/dt.

    Off by one versus left-rect G_self, which updates A with I[n] on
    interval [n, n+1]. Kept only to reconstruct the 2026-09-21 residual.
    """
    a = np.asarray(a, dtype=float)
    e = np.zeros_like(a)
    if a.shape[0] >= 2:
        e[1:] = -(a[1:] - a[:-1]) / dt
    return e


def e_interval_forward(a, dt: float) -> np.ndarray:
    """E[n] = -(A[n+1]-A[n])/dt on [n, n+1]; E[last]=0.

    Matches left-rect G_self: (A_self[n+1]-A_self[n])/dt = γ I[n],
    hence E_self[n] = -γ I[n] on that interval.
    """
    a = np.asarray(a, dtype=float)
    e = np.zeros_like(a)
    if a.shape[0] >= 2:
        e[:-1] = -(a[1:] - a[:-1]) / dt
    return e


def work_from_e_j(e, j, dt: float) -> dict:
    """Legacy trapezoid ∫ E·J. Reconstructs the 2026-09-21 ledger only."""
    e = np.asarray(e, dtype=float)
    j = np.asarray(j, dtype=float)
    nt = e.shape[0]
    wt = trapz_weight(nt, dt)
    dens = np.sum(e * j, axis=1)
    w = float(np.dot(wt, dens))
    cum = np.zeros(nt, dtype=float)
    for n in range(1, nt):
        cum[n] = cum[n - 1] + 0.5 * dt * (dens[n - 1] + dens[n])
    return {
        "W": w,
        "density": dens,
        "cumulative": cum,
        "W_sign": int(np.sign(w)) if w != 0.0 else 0,
        "quadrature": "trapezoid_nodes",
    }


def left_riemann_work(e, j, dt: float) -> dict:
    """W = dt Σ_{n=0}^{N-2} E[n]·J[n] on the same intervals as G_self."""
    e = np.asarray(e, dtype=float)
    j = np.asarray(j, dtype=float)
    dens = np.sum(e * j, axis=1)
    cum = np.zeros_like(dens)
    if dens.size <= 1:
        w = 0.0
    else:
        w = float(dt * np.sum(dens[:-1]))
        cum[1:] = dt * np.cumsum(dens[:-1])
    return {
        "W": w,
        "density": dens,
        "cumulative": cum,
        "W_sign": int(np.sign(w)) if w != 0.0 else 0,
        "quadrature": "left_riemann_intervals_exclude_last_node",
    }


def a_self_leftrect(i, dt: float, gamma: float) -> np.ndarray:
    i = np.asarray(i, dtype=float)
    a = np.zeros_like(i)
    acc = np.zeros(2, dtype=float)
    for n in range(i.shape[0]):
        a[n] = gamma * acc
        acc = acc + dt * i[n]
    return a


def split_work(a_in, i, dt: float, gamma: float) -> dict:
    """Legacy mixed-time ledger (backward E_sheet vs E_self=-γ I_n)."""
    i = np.asarray(i, dtype=float)
    a_in = np.asarray(a_in, dtype=float)
    a_self = a_self_leftrect(i, dt, gamma)
    a_sheet = a_in + a_self
    e_in = e_from_a_leftrect(a_in, dt)
    e_self = -gamma * i
    e_sheet = e_from_a_leftrect(a_sheet, dt)
    w_el = work_from_e_j(e_sheet, i, dt)
    w_in = work_from_e_j(e_in, i, dt)
    w_self = work_from_e_j(e_self, i, dt)
    return {
        "A_self": a_self,
        "A_sheet": a_sheet,
        "E_in": e_in,
        "E_self": e_self,
        "E_sheet": e_sheet,
        "W_el": w_el["W"],
        "W_in": w_in["W"],
        "W_self": w_self["W"],
        "W_el_sign": w_el["W_sign"],
        "W_el_cumulative": w_el["cumulative"],
        "W_in_cumulative": w_in["cumulative"],
        "W_residual": float(w_el["W"] - w_in["W"] - w_self["W"]),
        "W_self_minus_gamma_I2": float(
            w_self["W"] + gamma * float(np.dot(trapz_weight(i.shape[0], dt), np.sum(i * i, axis=1)))
        ),
        "ledger": "legacy_backward_E_plus_node_Eself",
    }


def two_port_increment_flux(e_in, i, dt: float, gamma: float) -> dict:
    """(1/(2γ)) ∫ (|e-γi|² + |γi|² - |e|²) dt, left Riemann. Equals -W_el if γ>0."""
    e_in = np.asarray(e_in, dtype=float)
    i = np.asarray(i, dtype=float)
    e_self = -float(gamma) * i
    e_r = e_in + e_self
    e_l = e_self
    dens = np.sum(e_r * e_r, axis=1) + np.sum(e_l * e_l, axis=1) - np.sum(e_in * e_in, axis=1)
    if e_in.shape[0] <= 1:
        phi = 0.0
        cum = np.zeros(e_in.shape[0], dtype=float)
    else:
        if abs(gamma) < 1e-30:
            phi = 0.0
            cum = np.zeros(e_in.shape[0], dtype=float)
        else:
            phi = float(dt * np.sum(dens[:-1]) / (2.0 * gamma))
            cum = np.zeros(e_in.shape[0], dtype=float)
            cum[1:] = (dt / (2.0 * gamma)) * np.cumsum(dens[:-1])
    return {
        "Phi_out_minus_in": phi,
        "cumulative": cum,
        "gamma": float(gamma),
        "identity": "Phi = -W_el for gamma>0 with matching left-interval E,I",
    }


def split_work_interval(a_in, i, dt: float, gamma: float) -> dict:
    """One interval: forward ΔA and left-endpoint I. Residual is kept, not forced to 0."""
    i = np.asarray(i, dtype=float)
    a_in = np.asarray(a_in, dtype=float)
    a_self = a_self_leftrect(i, dt, gamma)
    a_sheet = a_in + a_self
    e_in = e_interval_forward(a_in, dt)
    e_self_a = e_interval_forward(a_self, dt)
    e_self_i = -float(gamma) * i
    e_sheet = e_interval_forward(a_sheet, dt)
    w_el = left_riemann_work(e_sheet, i, dt)
    w_in = left_riemann_work(e_in, i, dt)
    w_self = left_riemann_work(e_self_i, i, dt)
    w_self_a = left_riemann_work(e_self_a, i, dt)
    flux = two_port_increment_flux(e_in, i, dt, gamma)
    e_sum = e_in + e_self_i
    e_match = float(np.max(np.abs(e_sheet[:-1] - e_sum[:-1]))) if e_sheet.shape[0] > 1 else 0.0
    i2 = float(dt * np.sum(np.sum(i * i, axis=1)[:-1])) if i.shape[0] > 1 else 0.0
    return {
        "A_self": a_self,
        "A_sheet": a_sheet,
        "E_in": e_in,
        "E_self": e_self_i,
        "E_self_from_A": e_self_a,
        "E_sheet": e_sheet,
        "W_el": w_el["W"],
        "W_in": w_in["W"],
        "W_self": w_self["W"],
        "W_self_from_A": w_self_a["W"],
        "W_el_sign": w_el["W_sign"],
        "W_in_sign": w_in["W_sign"],
        "W_self_sign": w_self["W_sign"],
        "W_el_cumulative": w_el["cumulative"],
        "W_in_cumulative": w_in["cumulative"],
        "W_residual": float(w_el["W"] - w_in["W"] - w_self["W"]),
        "W_residual_from_A": float(w_el["W"] - w_in["W"] - w_self_a["W"]),
        "E_sheet_minus_Ein_Eself_maxabs": e_match,
        "W_self_plus_gamma_I2": float(w_self["W"] + float(gamma) * i2),
        "Phi_out_minus_in": flux["Phi_out_minus_in"],
        "Phi_plus_W_el": float(flux["Phi_out_minus_in"] + w_el["W"]),
        "ledger": "forward_interval_left_I",
        "gain_claim": "not_read_from_W_sign",
    }


def differential_energy_ledger(u, u0, kappa_p1, seed, a_self, i, i0, dt: float) -> dict:
    """ΔU vs E_P1·δI + δE·I0 + δE·δI. Residual kept. Not a gain claim.

    Node energies U=Tr(ρH). Fields from forward interval differences.
    Initial U(0)-U0(0) is stored; seed[0] need not vanish.
    """
    u = np.asarray(u, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    i = np.asarray(i, dtype=float)
    i0 = np.asarray(i0, dtype=float)
    di = i - i0
    e_p1 = e_interval_forward(np.asarray(kappa_p1, dtype=float), dt)
    e_seed = e_interval_forward(np.asarray(seed, dtype=float), dt)
    e_self = e_interval_forward(np.asarray(a_self, dtype=float), dt)
    e_delta = e_seed + e_self
    w_p1_di = left_riemann_work(e_p1, di, dt)["W"]
    w_de_i0 = left_riemann_work(e_delta, i0, dt)["W"]
    w_de_di = left_riemann_work(e_delta, di, dt)["W"]
    w_sum = w_p1_di + w_de_i0 + w_de_di
    du = float(u[-1] - u[0])
    du0 = float(u0[-1] - u0[0])
    ddu = du - du0
    return {
        "U_end_minus_start": du,
        "U0_end_minus_start": du0,
        "dU_minus_dU0": ddu,
        "U0_minus_U_at_t0": float(u[0] - u0[0]),
        "W_P1_dot_dI": w_p1_di,
        "W_dE_dot_I0": w_de_i0,
        "W_dE_dot_dI": w_de_di,
        "W_three_term_sum": w_sum,
        "residual_dU_minus_work": float(ddu - w_sum),
        "note": "deltaE·deltaI is only the last term; not the full electronic absorption.",
    }


def halt_index(i, ref_peak: float, *, ratio: float = 10.0, abs_cap: float = HALT_ABS_I) -> int:
    i = np.asarray(i, dtype=float)
    mag = np.linalg.norm(i, axis=1)
    cap = max(float(ratio) * max(float(ref_peak), 1e-30), 1.0)
    hit = np.where((mag > cap) | (mag > abs_cap) | ~np.isfinite(mag))[0]
    if hit.size == 0:
        return int(i.shape[0] - 1)
    return int(hit[0])


def growth_rate(i, times, n_lo: int, n_hi: int) -> dict:
    """Fit log ||I|| = a + Γ t on [n_lo, n_hi]."""
    i = np.asarray(i, dtype=float)
    t = np.asarray(times, dtype=float)
    mag = np.linalg.norm(i, axis=1)
    n_hi = min(n_hi, i.shape[0] - 1)
    n_lo = max(0, min(n_lo, n_hi - 2))
    sl = slice(n_lo, n_hi + 1)
    y = np.log(np.maximum(mag[sl], 1e-30))
    x = t[sl]
    x0 = x - x.mean()
    denom = float(np.dot(x0, x0))
    gamma = float(np.dot(x0, y - y.mean()) / max(denom, 1e-30))
    pred = y.mean() + gamma * x0
    resid = float(np.sqrt(np.mean((y - pred) ** 2)))
    return {
        "Gamma": gamma,
        "n_lo": int(n_lo),
        "n_hi": int(n_hi),
        "t_lo": float(t[n_lo]),
        "t_hi": float(t[n_hi]),
        "log_rms_resid": resid,
        "mag_lo": float(mag[n_lo]),
        "mag_hi": float(mag[n_hi]),
    }


def rfft_parseval_weights(nt: int) -> np.ndarray:
    """Weights so Σ w_k |X_k|² is Parseval-consistent for real rFFT."""
    nfreq = nt // 2 + 1
    w = np.ones(nfreq, dtype=float)
    if nt % 2 == 0:
        w[1:-1] = 2.0
    else:
        w[1:] = 2.0
    return w


def band_power(sig, dt: float, omega0: float) -> dict:
    """rFFT band fractions with one-sided Parseval weights. Does not remove DC.

    'low' is 0 <= ω < 0.25 ω0, not strict zero frequency.
    """
    sig = np.asarray(sig, dtype=float)
    nt = sig.shape[0]
    spec = np.fft.rfft(sig, axis=0)
    raw = np.sum(np.abs(spec) ** 2, axis=1)
    wt = rfft_parseval_weights(nt)
    pow_c = raw * wt
    freq = np.fft.rfftfreq(nt, d=dt)
    om = 2.0 * np.pi * freq
    tot = float(np.sum(pow_c))
    tot = max(tot, 1e-30)

    def frac(lo, hi):
        m = (om >= lo) & (om < hi)
        return float(np.sum(pow_c[m]) / tot)

    low = frac(0.0, 0.25 * omega0)
    h1 = frac(0.25 * omega0, 1.5 * omega0)
    h2 = frac(1.5 * omega0, 2.5 * omega0)
    high = frac(2.5 * omega0, np.inf)
    dc_bin = float(pow_c[0] / tot) if pow_c.size else 0.0
    i0 = int(np.argmax(pow_c[1:]) + 1) if pow_c.size > 1 else 0
    mean = np.mean(sig, axis=0)
    return {
        "frac_low_below_0.25_omega0": low,
        "frac_strict_dc_bin": dc_bin,
        "frac_0.25_to_1.5_omega0": h1,
        "frac_H2_1.5_to_2.5_omega0": h2,
        "frac_above_2.5_omega0": high,
        "peak_omega": float(om[i0]),
        "peak_over_omega0": float(om[i0] / max(omega0, 1e-30)),
        "mean": mean.tolist(),
        "mean_abs": float(np.linalg.norm(mean)),
        "total_power": tot,
        "parseval_weights": "dc_and_nyquist_1_interior_2",
        "low_band_note": "0<=omega<0.25*omega0_not_strict_DC",
    }


def band_power_unweighted_legacy(sig, dt: float, omega0: float) -> dict:
    """Unweighted rFFT |X|² used on 2026-09-21. For stock comparison only."""
    sig = np.asarray(sig, dtype=float)
    nt = sig.shape[0]
    spec = np.fft.rfft(sig, axis=0)
    pow_c = np.sum(np.abs(spec) ** 2, axis=1)
    freq = np.fft.rfftfreq(nt, d=dt)
    om = 2.0 * np.pi * freq
    tot = max(float(np.sum(pow_c)), 1e-30)

    def frac(lo, hi):
        m = (om >= lo) & (om < hi)
        return float(np.sum(pow_c[m]) / tot)

    return {
        "frac_low_below_0.25_omega0": frac(0.0, 0.25 * omega0),
        "frac_strict_dc_bin": float(pow_c[0] / tot) if pow_c.size else 0.0,
        "total_power": tot,
        "parseval_weights": "none_legacy_unweighted",
    }


def dc_ramp_account(i, a_self, dt: float, gamma: float) -> dict:
    """Endpoint identity: A_self is the integral of I. Not an instability proof."""
    i = np.asarray(i, dtype=float)
    a_self = np.asarray(a_self, dtype=float)
    mean = np.mean(i, axis=0)
    t = dt * (i.shape[0] - 1)
    ramp = gamma * mean * t
    a_end = a_self[-1]
    return {
        "mean_I": mean.tolist(),
        "gamma_mean_I_T": ramp.tolist(),
        "A_self_end": a_end.tolist(),
        "ramp_over_A_self_end": float(np.linalg.norm(ramp) / max(np.linalg.norm(a_end), 1e-30)),
        "note": (
            "gamma*<I>*T / A_self(T) tests the endpoint of the same integrator, "
            "not an independent runaway mechanism. Do not subtract DC."
        ),
    }


def kappa_p1_series(times, geo, prot, sign: int = +1) -> np.ndarray:
    return np.asarray(kappa_shift(times, geo, prot, sign), dtype=float)


def amplitude_compare(kappa_self, kappa_p1, geo, a_star: float) -> dict:
    """Compare self-field to P1 in Å^-1 and in dimensionless A* = κ a_NN."""
    ks = np.linalg.norm(np.asarray(kappa_self, dtype=float), axis=1)
    kp = np.linalg.norm(np.asarray(kappa_p1, dtype=float), axis=1)
    a_nn = float(geo.a_nn_mean_ang)
    ks_peak = float(np.max(ks)) if ks.size else 0.0
    kp_peak = float(np.max(kp)) if kp.size else 0.0
    ks_comp = float(np.max(np.abs(kappa_self))) if np.size(kappa_self) else 0.0
    return {
        "a_nn_ang": a_nn,
        "kappa_self_peak_mod_ang_inv": ks_peak,
        "kappa_self_peak_component_ang_inv": ks_comp,
        "kappa_p1_peak_mod_ang_inv": kp_peak,
        "ratio_kappa_self_mod_over_p1": float(ks_peak / max(kp_peak, 1e-30)),
        "Astar_self_peak_mod": float(ks_peak * a_nn),
        "Astar_p1_protocol": float(a_star),
        "Astar_p1_from_peak_kappa": float(kp_peak * a_nn),
        "note": (
            "A_self in the propagator is a Cartesian kappa shift in 1/Å. "
            "P1 A*=0.8 is dimensionless e a_NN A/ħ. Do not plot 0.8 on the kappa axis."
        ),
    }


def a_priori_windows(times, period: float) -> dict:
    t = np.asarray(times, dtype=float)
    return {
        "cycle1": int(np.searchsorted(t, period, side="right") - 1),
        "cycle2": int(np.searchsorted(t, 2.0 * period, side="right") - 1),
        "cycle4": int(np.searchsorted(t, 4.0 * period, side="right") - 1),
        "full": int(t.shape[0] - 1),
    }


def slim_bands(rec: dict) -> dict:
    keep = dict(rec)
    keep.pop("mean", None)
    return keep


def interval_report(i, a_in, times, dt, gamma, omega0) -> dict:
    """Full-window interval ledger. No amplitude halt mixed into t_halt."""
    wrk = split_work_interval(a_in, i, dt, gamma)
    n_end = int(i.shape[0] - 1)
    return {
        "n_end": n_end,
        "t_end": float(times[n_end]),
        "maxabs_I": float(np.max(np.abs(i))),
        "maxabs_A_self_ang_inv": float(np.max(np.abs(wrk["A_self"]))),
        "maxmod_A_self_ang_inv": float(np.max(np.linalg.norm(wrk["A_self"], axis=1))),
        "W_el": wrk["W_el"],
        "W_in": wrk["W_in"],
        "W_self": wrk["W_self"],
        "W_el_sign": wrk["W_el_sign"],
        "W_in_sign": wrk["W_in_sign"],
        "W_residual": wrk["W_residual"],
        "Phi_out_minus_in": wrk["Phi_out_minus_in"],
        "Phi_plus_W_el": wrk["Phi_plus_W_el"],
        "W_self_plus_gamma_I2": wrk["W_self_plus_gamma_I2"],
        "E_sheet_minus_Ein_Eself_maxabs": wrk["E_sheet_minus_Ein_Eself_maxabs"],
        "I_bands": slim_bands(band_power(i, dt, omega0)),
        "A_self_bands": slim_bands(band_power(wrk["A_self"], dt, omega0)),
        "dc_ramp": dc_ramp_account(i, wrk["A_self"], dt, gamma),
        "ledger": wrk["ledger"],
        "gain_claim": "not_evaluated",
    }


def window_report(i, a_in, times, dt, gamma, omega0, ref_peak) -> dict:
    """Legacy amplitude-halted report. Distinguishes diag window from propagation halt."""
    n_halt = halt_index(i, ref_peak)
    sl = slice(0, n_halt + 1)
    i_w = i[sl]
    a_w = a_in[sl]
    t_w = times[sl]
    wrk_legacy = split_work(a_w, i_w, dt, gamma)
    wrk = split_work_interval(a_w, i_w, dt, gamma)
    n_hi = max(n_halt // 2, 8)
    n_lo = max(n_hi // 4, 2)
    return {
        "n_diag_window": n_halt,
        "t_diag_window": float(times[n_halt]),
        "diag_window_before_end": bool(n_halt < i.shape[0] - 1),
        "n_halt": n_halt,
        "t_halt": float(times[n_halt]),
        "halted_before_end": bool(n_halt < i.shape[0] - 1),
        "maxabs_I": float(np.max(np.abs(i_w))),
        "maxabs_A_self": float(np.max(np.abs(wrk["A_self"]))),
        "W_el": wrk["W_el"],
        "W_in": wrk["W_in"],
        "W_self": wrk["W_self"],
        "W_el_sign": wrk["W_el_sign"],
        "W_residual": wrk["W_residual"],
        "W_residual_legacy": wrk_legacy["W_residual"],
        "Phi_out_minus_in": wrk["Phi_out_minus_in"],
        "Phi_plus_W_el": wrk["Phi_plus_W_el"],
        "W_self_plus_gamma_I2": wrk["W_self_plus_gamma_I2"],
        "growth": growth_rate(i_w, t_w, n_lo, n_hi),
        "I_bands": band_power(i_w, dt, omega0),
        "A_self_bands": band_power(wrk["A_self"], dt, omega0),
        "dc_ramp": dc_ramp_account(i_w, wrk["A_self"], dt, gamma),
        "W_el_at_half_halt": float(wrk["W_el_cumulative"][max(n_halt // 2, 0)]),
        "W_el_end": float(wrk["W_el_cumulative"][min(n_halt, wrk["W_el_cumulative"].size - 1)]),
        "gain_claim": "not_evaluated",
    }


def k_sets(geo, p, mu: float) -> dict:
    probe = choose_probe_k(geo, p, mu)
    kf3, _kc, _w = k_grid(geo, 3, shift=0.0)
    kf3s, _kc2, _w2 = k_grid(geo, 3, shift=0.5)
    generic = np.array([
        [0.11, 0.19], [0.23, 0.37], [0.41, 0.08],
        [0.29, 0.61], [0.07, 0.53], [0.33, 0.44],
    ], dtype=float)
    no_g = probe["kf"][1:]
    return {
        "probe6": {"kf": probe["kf"], "labels": probe["labels"], "n": 6},
        "generic6": {"kf": generic, "labels": [f"g{i}" for i in range(6)], "n": 6},
        "mp3": {"kf": kf3, "labels": ["mp3"] * len(kf3), "n": int(len(kf3))},
        "mp3_shift": {"kf": kf3s, "labels": ["mp3s"] * len(kf3s), "n": int(len(kf3s))},
        "probe_no_gamma": {"kf": no_g, "labels": probe["labels"][1:], "n": int(len(no_g))},
    }


def propagate_seed_self(
    geo,
    p,
    prot,
    *,
    kf,
    mu: float,
    neel: int,
    nt_per_cycle: int,
    seed,
    i_ref,
    gamma: float,
    sign: int = +1,
    verbose: bool = False,
    halt: float = HALT_ABS_I,
    w_k=None,
    save_perk_I: bool = False,
):
    """Full nonlinear H(k + A_cl + seed + A_self). A_self from δI = I - I_P1.

    Left-rect G_self matches Volterra. No T2, no DC subtraction.
    """
    kf = np.mod(np.asarray(kf, dtype=float), 1.0)
    kc = geo.k_cart(kf)
    n_k = int(kc.shape[0])
    if w_k is None:
        w = np.full(n_k, 1.0 / n_k)
    else:
        w = np.asarray(w_k, dtype=float).reshape(n_k)
        wsum = float(np.sum(w))
        if wsum <= 0.0:
            raise ValueError("w_k must sum to a positive weight")
        w = w / wsum
    rho, _e0 = initial_rho(geo, p, kc, neel, mu)
    nt = prot.n_cycles * nt_per_cycle
    dt = prot.period / nt_per_cycle
    times = np.arange(nt + 1) * dt
    seed = np.asarray(seed, dtype=float)
    if seed.shape != (nt + 1, 2):
        raise ValueError(f"seed shape {seed.shape} != ({nt + 1}, 2)")
    i_ref = np.asarray(i_ref, dtype=float)
    if i_ref.shape != (nt + 1, 2):
        raise ValueError("i_ref grid mismatch")
    i_t = np.zeros((nt + 1, 2), dtype=float)
    i_perk = np.zeros((nt + 1, n_k, 2), dtype=float) if save_perk_I else None
    u_t = np.zeros(nt + 1, dtype=float)
    kap_p1 = kappa_p1_series(times, geo, prot, sign)
    a_self = np.zeros(2, dtype=float)
    a_self_hist = np.zeros((nt + 1, 2), dtype=float)
    g = float(gamma)
    halted = False
    n_halt = nt
    t0 = time.perf_counter()
    for n in range(nt + 1):
        t = times[n]
        a_self_hist[n] = a_self
        kap = kc + kap_p1[n] + seed[n] + a_self
        h, j = h8_and_current(geo, p, kap, neel)
        jk = trace_current(rho, j)
        if save_perk_I:
            i_perk[n] = jk
        i_t[n] = w @ jk
        u_t[n] = float(np.real(np.einsum("k,kij,kji->", w, rho, h)))
        di = i_t[n] - i_ref[n]
        if n == nt:
            break
        if not np.isfinite(i_t[n]).all() or float(np.max(np.abs(i_t[n]))) > halt:
            halted = True
            n_halt = n
            i_t[n + 1 :] = i_t[n]
            u_t[n + 1 :] = u_t[n]
            a_self_hist[n + 1 :] = a_self
            if save_perk_I:
                i_perk[n + 1 :] = jk
            break
        tmid = t + 0.5 * dt
        seed_mid = 0.5 * (seed[n] + seed[n + 1])
        kmid = kc + kappa_shift(tmid, geo, prot, sign) + seed_mid + a_self
        hm = h8(geo, p, kmid, neel)
        em, wm = np.linalg.eigh(hm)
        ph = np.exp(-1j * em * dt)
        u = (wm * ph[:, None, :]) @ np.conj(np.swapaxes(wm, 1, 2))
        rho = u @ rho @ np.conj(np.swapaxes(u, 1, 2))
        a_self = a_self + g * dt * di
        if verbose and n % max(1, nt // 8) == 0:
            print(
                f"  seed-self {n}/{nt} |I|={np.linalg.norm(i_t[n]):.3e} "
                f"|As|={np.linalg.norm(a_self):.3e} {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    di_t = i_t - i_ref
    out = {
        "times": times,
        "dt": float(dt),
        "I": i_t,
        "dI": di_t,
        "I_ref": i_ref,
        "A_self": a_self_hist,
        "kappa_p1": kap_p1,
        "U": u_t,
        "seed": seed,
        "gamma": g,
        "n_k": n_k,
        "w_k": w,
        "halted": halted,
        "n_prop_halt": int(n_halt),
        "t_prop_halt": float(times[n_halt]),
        "n_halt": int(n_halt),
        "t_halt": float(times[n_halt]),
        "maxabs_I": float(np.max(np.abs(i_t))),
        "maxabs_dI": float(np.max(np.abs(di_t))),
        "maxabs_A_self": float(np.max(np.abs(a_self_hist))),
        "maxmod_A_self": float(np.max(np.linalg.norm(a_self_hist, axis=1))),
        "wall_seconds": time.perf_counter() - t0,
    }
    if save_perk_I:
        out["I_perk"] = i_perk
    return out


def seed_h2_re(times, omega0, tp, geo, eps: float) -> np.ndarray:
    from q8_m0_mode_io import mode_waveforms

    mw = mode_waveforms(times, omega0, tp, geo)
    F = np.asarray(mw["F"])
    e_perp = np.asarray(mw["e_perp"])
    h = np.real(F)
    peak = float(np.max(np.abs(h)))
    if peak < 1e-30:
        return np.zeros((times.shape[0], 2), dtype=float)
    return (eps / peak) * h[:, None] * e_perp[None, :]


def seed_slow_sin2(times, tp: float, geo, eps: float) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    e_par, e_perp = parallel_perp_basis(geo)
    u = t / tp
    w = np.sin(np.pi * u) ** 2
    w[(u < 0.0) | (u > 1.0)] = 0.0
    return eps * w[:, None] * e_perp[None, :]
