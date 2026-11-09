#!/usr/bin/env python3
"""Causal two-port Volterra on stored six-k R^A + G_self.

Solves, on the fluctuation fields only:

    I = ξ + R^A[A_loc]
    A_loc = A_L_in + A_R_in + γ ∫^{t-} I

R^A is real-linear (state prefix + contact). No scalar (1-χ(ω))^{-1}.
H2 is applied only after I(t) is known. Production V(θ) stays Born until
the weak-coupling and commutator diagnostics are read.

Left-rect in time matches prefix_state_dmu. Collection of H2 uses the same
trapezoid weights as close_output_moments so γ=0 recovers Born.
"""
from __future__ import annotations

import numpy as np

from q8_m0_k1_kernel import trapz_weight
from q8_m0_mode_io import mode_waveforms, response_to_complex_scalar
from q8_m0_s1_kernel import pack_complex

VOLTERRA_ID = "two_port_causal_volterra_20260921"
VOLTERRA_STATUS = "sixk_method_linear_IO_not_production_V"


def _as_nt2(arr, nt: int, name: str) -> np.ndarray:
    if arr is None:
        return np.zeros((nt, 2), dtype=float)
    out = np.asarray(arr, dtype=float)
    if out.shape != (nt, 2):
        raise ValueError(f"{name} shape {out.shape} != ({nt}, 2)")
    return out


def volterra_forward(
    x,
    w,
    mean_d,
    dt: float,
    gamma: float,
    a_L_in=None,
    a_R_in=None,
    xi=None,
):
    """Causal left-rect solve. Returns I, A_sheet, A_self, A_R_out, A_L_out."""
    x = np.asarray(x)
    w = np.asarray(w, dtype=float)
    md = np.asarray(mean_d, dtype=float)
    nt, nk = x.shape[0], x.shape[1]
    a_L_in = _as_nt2(a_L_in, nt, "a_L_in")
    a_R_in = _as_nt2(a_R_in, nt, "a_R_in")
    xi = _as_nt2(xi, nt, "xi")
    z = np.zeros((nk, 4, 4), dtype=complex)
    iacc = np.zeros(2, dtype=float)
    i_hist = np.zeros((nt, 2), dtype=float)
    a_sheet = np.zeros((nt, 2), dtype=float)
    a_self = np.zeros((nt, 2), dtype=float)
    g = float(gamma)
    for n in range(nt):
        a_s = g * iacc
        a = a_L_in[n] + a_R_in[n] + a_s
        a_self[n] = a_s
        a_sheet[n] = a
        tr = np.einsum("kacv,kcv->ka", np.conj(x[n]), z)
        i_state = -2.0 * np.imag(w @ tr)
        i_contact = -md[n] @ a
        i_now = xi[n] + i_state + i_contact
        i_hist[n] = i_now
        iacc = iacc + dt * i_now
        z = z + dt * np.einsum("a,kacv->kcv", a, x[n])
    return {
        "I": i_hist,
        "A_sheet": a_sheet,
        "A_self": a_self,
        "A_R_out": a_L_in + a_self,
        "A_L_out": a_R_in + a_self,
        "A_L_in": a_L_in,
        "A_R_in": a_R_in,
        "gamma": g,
        "dt": float(dt),
        "maxabs_I": float(np.max(np.abs(i_hist))),
        "maxabs_A_self": float(np.max(np.abs(a_self))),
        "maxabs_A_sheet": float(np.max(np.abs(a_sheet))),
        "finite": bool(np.isfinite(i_hist).all() and np.max(np.abs(i_hist)) < 1e8),
    }


def volterra_adjoint(
    x,
    w,
    mean_d,
    dt: float,
    gamma: float,
    g_I,
):
    """Packed-Euclidean reverse of I = T[ξ, A_L+A_R].

    L = Σ_n g_I[n] · I[n]. Returns dL/dξ, dL/dA_in (A_in = A_L+A_R).
    """
    x = np.asarray(x)
    w = np.asarray(w, dtype=float)
    md = np.asarray(mean_d, dtype=float)
    g_I = np.asarray(g_I, dtype=float)
    nt, nk = x.shape[0], x.shape[1]
    if g_I.shape != (nt, 2):
        raise ValueError(f"g_I shape {g_I.shape} != ({nt}, 2)")
    g_iacc = np.zeros(2, dtype=float)
    gz = np.zeros((nk, 4, 4), dtype=complex)
    g_xi = np.zeros((nt, 2), dtype=float)
    g_ain = np.zeros((nt, 2), dtype=float)
    g = float(gamma)
    for n in range(nt - 1, -1, -1):
        gA = dt * np.real(np.einsum("kcv,kacv->a", np.conj(gz), x[n]))
        gi = g_I[n] + dt * g_iacc
        g_xi[n] = gi
        gA = gA - md[n].T @ gi
        gz = gz - 2.0j * np.einsum("a,k,kacv->kcv", gi, w, x[n])
        g_ain[n] = gA
        g_iacc = g_iacc + g * gA
    return {"g_xi": g_xi, "g_Ain": g_ain}


def collect_overlap(k_a, signal, dt: float) -> complex:
    wt = trapz_weight(k_a.shape[0], dt)
    return complex(np.einsum("t,ta,ta->", wt, k_a, signal))


def h2_kernels(times, omega0, tp, geo) -> dict:
    mw = mode_waveforms(times, omega0, tp, geo)
    F = np.asarray(mw["F"])
    e_perp = np.asarray(mw["e_perp"])
    k_a = np.asarray(mw["K_a_shape"])
    return {"F": F, "e_perp": e_perp, "k_a": k_a, "mw": mw}


def drive_from_complex_F(F, ehat) -> dict:
    """Unscaled kappa shapes used by Born: A ~ (F* b + F b†) ê."""
    F = np.asarray(F)
    ehat = np.asarray(ehat, dtype=float)
    h_b = np.real(np.conj(F))[:, None] * ehat[None, :]  # coeff of b: Re(F*) = Re F
    # F* b + F b†, for a real test replacing b→1: 2 Re(F*) = 2 Re F.
    # Born splits R on Re F and Im F separately via response_to_complex_scalar.
    return {
        "h_re_F": np.real(F)[:, None] * ehat[None, :],
        "h_im_F": np.imag(F)[:, None] * ehat[None, :],
        "h_Fstar": np.real(np.conj(F))[:, None] * ehat[None, :],
    }


def volterra_response_to_complex_F(x, w, mean_d, dt, gamma, F, ehat) -> dict:
    """Real-linear T R on (Re F) ê and (Im F) ê, same split as Born."""
    F = np.asarray(F)
    ehat = np.asarray(ehat, dtype=float)
    hr = np.real(F)[:, None] * ehat[None, :]
    hi = np.imag(F)[:, None] * ehat[None, :]
    pr = volterra_forward(x, w, mean_d, dt, gamma, a_L_in=hr)
    pi = volterra_forward(x, w, mean_d, dt, gamma, a_L_in=hi)
    return {
        "total": pr["I"] + 1j * pi["I"],
        "I_re": pr["I"],
        "I_im": pi["I"],
        "A_sheet_re": pr["A_sheet"],
        "A_sheet_im": pi["A_sheet"],
        "maxabs_I": max(pr["maxabs_I"], pi["maxabs_I"]),
        "maxabs_A_self": max(pr["maxabs_A_self"], pi["maxabs_A_self"]),
        "finite": bool(pr["finite"] and pi["finite"]),
    }


def born_limit_currents(x, w, mean_d, dt, F, ehat) -> dict:
    return response_to_complex_scalar(x, w, F, ehat, dt, mean_d)


def collect_A2_B2(k_a, i_complex, dt, pref) -> dict:
    """A2 = i pref ∫ F ê · I[F*], B2 = i pref ∫ F ê · I[F].

    i_complex = I[Re F] + i I[Im F] = I[F] in the same sense as Born.
    I[F*] = I[Re F] - i I[Im F] = conj_parts, i.e. I_re - i I_im.
    """
    i_F = np.asarray(i_complex)
    i_Fs = np.real(i_F) - 1j * np.imag(i_F)
    beta_F = collect_overlap(k_a, i_F, dt)
    beta_Fs = collect_overlap(k_a, i_Fs, dt)
    return {
        "A2": 1j * pref * beta_Fs,
        "B2": 1j * pref * beta_F,
        "beta_F": beta_F,
        "beta_Fstar": beta_Fs,
    }


def noise_kernel_from_adjoint(x, w, mean_d, dt, gamma, k_a) -> np.ndarray:
    """K_eff(t) such that ∫ K · (T ξ) = Σ_n g_ξ[n] · ξ[n]."""
    wt = trapz_weight(k_a.shape[0], dt)
    g_I = np.real(wt[:, None] * np.asarray(k_a))
    # K is complex: n uses both Re and Im through Y=∫ K X.
    # Split: return complex K_eff = g_xi_re + i g_xi_im from two reverse runs.
    adj_r = volterra_adjoint(x, w, mean_d, dt, gamma, np.real(wt[:, None] * k_a))
    adj_i = volterra_adjoint(x, w, mean_d, dt, gamma, np.imag(wt[:, None] * k_a))
    return adj_r["g_xi"] + 1j * adj_i["g_xi"]


def project_with_g(x, g_kernel):
    """Y[k,cv] = Σ_{t,a} g[t,a] X[t,k,a,cv]  (g already includes time weights)."""
    return np.einsum("ta,tkacv->kcv", g_kernel, x)


def filtered_source_moments(x, w, mean_d, dt, gamma, k_a) -> dict:
    """⟨s†s⟩ and ⟨ss⟩ of collect(T ξ), X language, two-port G_self on."""
    k_a = np.asarray(k_a)
    k_eff = noise_kernel_from_adjoint(x, w, mean_d, dt, gamma, k_a)
    yk = project_with_g(x, k_eff)
    yks = project_with_g(x, np.conj(k_eff))
    n_F = complex(np.einsum("k,kcv,kcv->", w, np.conj(yk), yk))
    m_F = complex(np.einsum("k,kcv,kcv->", w, np.conj(yks), yk))
    n_Fs = complex(np.einsum("k,kcv,kcv->", w, np.conj(yks), yks))
    return {
        "n_F": n_F,
        "m_F": m_F,
        "n_Fstar": n_Fs,
        "commutator_F": n_Fs - n_F,
        "k_eff": k_eff,
    }


def two_port_output_moments(A2, B2, n_src, m_src, comm_s) -> dict:
    """b_R^{H2} = b_L + s + A2 (b_L + b_R) + B2 (b_L† + b_R†), vacuum ports.

    This is the same-mode (H2 packet) truncation of the incoming field.
    Continuum incoming is reported separately.
    """
    A2 = complex(A2)
    B2 = complex(B2)
    n_src = complex(n_src)
    m_src = complex(m_src)
    comm_s = complex(comm_s)
    # b = (1+A2) b_L + A2 b_R + B2 b_L† + B2 b_R† + s
    n = n_src + 2.0 * abs(B2) ** 2
    m = m_src + (1.0 + A2) * B2 + A2 * B2
    comm = (
        abs(1.0 + A2) ** 2
        + abs(A2) ** 2
        - 2.0 * abs(B2) ** 2
        + comm_s
    )
    theta = np.linspace(0.0, np.pi, 181, endpoint=True)
    v = 0.5 + float(np.real(n)) + np.real(np.exp(-2j * theta) * m)
    canonical = abs(comm - 1.0) < 0.05
    return {
        "n": pack_complex(n),
        "m": pack_complex(m),
        "commutator": pack_complex(comm),
        "commutator_abs_minus_1": float(abs(comm - 1.0)),
        "V_min": float(np.min(v)),
        "V_max": float(np.max(v)),
        "V_optical_valid": bool(canonical and np.min(v) >= -1e-12),
        "note": (
            "Same-mode two-port incoming (H2 packet on L and on R). "
            "If |[b,b†]-1| is not small, V is diagnostic, not optical variance."
        ),
    }


def continuum_incoming_from_adjoint(
    g_Ain,
    times,
    omega,
    ehat,
    pref: float,
):
    """Vacuum |B(ω)|² from A_in continuum, one port, after one adjoint.

    Unscaled A(t) = ∫ dω (e^{+iωt}/√ω a(ω) + h.c.). Prefactor pref as in Born.
    g_Ain is d collect_unscaled / d A_in, collect_unscaled = ∫ K_shape I dt
    so A2-like coefficients still multiply by i * pref outside, here we
    form c(ω), d(ω) for collect_unscaled then * pref for moments? 

    collect used in Born is i * pref * ∫ F I, while trapz K_a already is F ê
    and A2 = i pref * overlap. The adjoint g_Ain here is from g_I = wt Re/Im(K)
    so ⟨K, I⟩_trapz = ⟨g_Ain, A_in⟩. Then i pref * that overlap is the
    bosonic coefficient. For a(ω): A_in_shape = (e^{iωt}/√ω) ê, overlap
    α(ω) = ⟨g_Ain, u_ω⟩, bosonic piece i pref α(ω) is not quite right because
    pref already contains the mode amplitude squared for a normalized packet.

    Use the packet reconstruction check instead of an absolute continuum n:
    ∫ dω φ(ω) α(ω) should rebuild the H2 same-mode overlap.
    """
    times = np.asarray(times, dtype=float)
    omega = np.asarray(omega, dtype=float)
    ehat = np.asarray(ehat, dtype=float)
    g = np.asarray(g_Ain, dtype=float)
    g_dir = g @ ehat
    wo = np.sqrt(np.maximum(omega, 0.0))
    phase = np.exp(1j * times[:, None] * omega[None, :])
    # overlap_ω = ∫ dt g_dir(t) e^{iωt} / √ω
    dens = np.zeros_like(omega, dtype=complex)
    mask = wo > 0.0
    kernel = np.trapezoid(g_dir[:, None] * phase, times, axis=0)
    dens[mask] = kernel[mask] / wo[mask]
    dens[~mask] = 0.0
    return {
        "alpha_omega_unscaled": dens,
        "omega": omega,
        "l2": float(np.sqrt(np.trapezoid(np.abs(dens) ** 2, omega))),
    }


def gamma0_recovers_born(x, w, mean_d, dt, F, ehat, k_a, pref) -> dict:
    born = born_limit_currents(x, w, mean_d, dt, F, ehat)
    vol = volterra_response_to_complex_F(x, w, mean_d, dt, 0.0, F, ehat)
    err = float(np.max(np.abs(vol["total"] - born["total"])))
    peak = float(np.max(np.abs(born["total"])))
    c_born = collect_A2_B2(k_a, born["total"], dt, pref)
    c_vol = collect_A2_B2(k_a, vol["total"], dt, pref)
    return {
        "maxabs_I_vol_minus_born": err,
        "rel_maxabs": err / max(peak, 1e-30),
        "A2_born": pack_complex(c_born["A2"]),
        "A2_vol_gamma0": pack_complex(c_vol["A2"]),
        "B2_born": pack_complex(c_born["B2"]),
        "B2_vol_gamma0": pack_complex(c_vol["B2"]),
        "pass": bool(err / max(peak, 1e-30) < 1e-10),
    }
