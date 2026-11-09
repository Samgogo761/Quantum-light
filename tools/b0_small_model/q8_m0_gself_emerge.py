#!/usr/bin/env python3
"""G_self must emerge from the two-port photon continuum.

Prescribed c-number I_cell. Photon oscillators only. No hand-made G_self,
no p-h dynamics, no production V. Finite-time causal A(t0)=0, not bare 1/ω.

Diagnostic N_cell splits η_vac ∝ N^{-1/2} and g_el ∝ N^{+1/2}. The per-cell
Maxwell kernel must be independent of that N_cell.
"""
from __future__ import annotations

import numpy as np

from q8_m0_gsheet import a_self_from_current, gamma_self
from q8_m0_io_audit import F_from_phi, packet_basis
from q8_m0_mode_io import eta2_ncell

GSELF_EMERGE_ID = "photon_continuum_Gself_emergence_20260923"
NO_HAND_GSELF = "G_self_is_not_an_input_it_must_emerge"


def eta_frozen(geo) -> float:
    return float(np.sqrt(eta2_ncell(geo)))


def eta_split(geo, n_cell_diag: float) -> dict:
    """Diagnostic split. Production N_cell stays unset."""
    n = float(n_cell_diag)
    if n <= 0.0:
        raise ValueError("diagnostic N_cell must be positive")
    eta = eta_frozen(geo)
    return {
        "N_cell_diag": n,
        "eta_vac": eta / np.sqrt(n),
        "eta_drive": eta * np.sqrt(n),
        "eta_frozen": eta,
        "product_is_eta_frozen_sq": True,
        "production_N_cell": "unset_not_equal_to_nk",
    }


def two_port_F_list(phis, omega, times, dw: float) -> tuple[list, list]:
    """Each packet becomes an L oscillator and an R oscillator. Same F."""
    flist = []
    labs = []
    for i, phi in enumerate(phis):
        f = F_from_phi(phi, omega, times, dw)
        flist.extend([f, f])
        labs.extend([f"L_{i}", f"R_{i}"])
    return flist, labs


def emerge_A_from_I(f_list, i_cell, dt: float, eta_A: float, eta_drive: float) -> np.ndarray:
    """Eliminate photon EOM with prescribed I. Left-Riemann, A[0]=0."""
    i_cell = np.asarray(i_cell, dtype=float)
    nt = int(i_cell.shape[0])
    if not f_list:
        return np.zeros_like(i_cell)
    fr = np.stack([np.real(np.asarray(f, dtype=complex)) for f in f_list], axis=0)
    fi = np.stack([np.imag(np.asarray(f, dtype=complex)) for f in f_list], axis=0)
    n_ph = int(fr.shape[0])
    q = np.zeros(n_ph, dtype=float)
    p = np.zeros(n_ph, dtype=float)
    a = np.zeros((nt, 2), dtype=float)
    s2 = np.sqrt(2.0)
    for n in range(nt):
        a[n] = float(eta_A) * s2 * (fr[:, n].T @ q + fi[:, n].T @ p)
        q = q - s2 * float(eta_drive) * float(dt) * (fi[:, n] @ i_cell[n])
        p = p + s2 * float(eta_drive) * float(dt) * (fr[:, n] @ i_cell[n])
    return a


def maxwell_A_from_I(i_cell, dt: float, gamma: float) -> np.ndarray:
    return a_self_from_current(i_cell, dt, gamma)


def _slice_compare(a_ph, a_mx, i_cell, dt: float, gamma: float, sl) -> dict:
    a_ph = np.asarray(a_ph, dtype=float)[sl]
    a_mx = np.asarray(a_mx, dtype=float)[sl]
    i_cell = np.asarray(i_cell, dtype=float)[sl]
    den = float(np.max(np.abs(a_mx)))
    d_ph = np.zeros_like(a_ph)
    if a_ph.shape[0] > 1:
        d_ph[1:] = (a_ph[1:] - a_ph[:-1]) / float(dt)
    e_target = gamma * i_cell
    return {
        "A0": float(np.max(np.abs(a_ph[0]))),
        "maxabs_A_minus_Maxwell": float(np.max(np.abs(a_ph - a_mx))),
        "rel_maxabs_A": float(np.max(np.abs(a_ph - a_mx)) / max(den, 1e-30)),
        "l2_rel_A": float(np.linalg.norm(a_ph - a_mx) / max(np.linalg.norm(a_mx), 1e-30)),
        "maxabs_dA_minus_gamma_I_past": float(np.max(np.abs(d_ph[1:] - e_target[:-1]))) if a_ph.shape[0] > 1 else 0.0,
        "no_hand_G_self": True,
    }


def compare_A(a_ph, a_mx, i_cell, dt: float, gamma: float, *, n_pulse: int | None = None) -> dict:
    """Compare on the source window [0, Tp], not on the periodic T_obs image."""
    a_ph = np.asarray(a_ph, dtype=float)
    n_pulse = int(a_ph.shape[0] if n_pulse is None else n_pulse)
    sl = slice(0, n_pulse)
    rec = _slice_compare(a_ph, a_mx, i_cell, dt, gamma, sl)
    rec["n_pulse"] = n_pulse
    rec["n_total"] = int(a_ph.shape[0])
    return rec


def prescribed_I(times, ehat, *, kind: str = "sin2") -> np.ndarray:
    t = np.asarray(times, dtype=float)
    ehat = np.asarray(ehat, dtype=float)
    ehat = ehat / max(float(np.linalg.norm(ehat)), 1e-30)
    tp = float(t[-1] - t[0])
    if tp <= 0.0:
        env = np.ones_like(t)
    else:
        env = np.sin(np.pi * (t - t[0]) / tp) ** 2
    if kind == "sin2":
        amp = env
    elif kind == "sin2_carrier":
        w0 = 2.0 * np.pi * 8.0 / tp if tp > 0 else 0.05
        amp = env * np.sin(w0 * (t - t[0]))
    else:
        raise ValueError(kind)
    return amp[:, None] * ehat[None, :]


def emerge_on_basis(
    phis, omega, times, dw, i_cell, dt, geo, n_cell_diag: float, *, n_pulse: int | None = None,
) -> dict:
    flist, labs = two_port_F_list(phis, omega, times, dw)
    split = eta_split(geo, n_cell_diag)
    gamma = gamma_self(geo)
    a_ph = emerge_A_from_I(flist, i_cell, dt, split["eta_vac"], split["eta_drive"])
    a_mx = maxwell_A_from_I(i_cell, dt, gamma)
    cmp = compare_A(a_ph, a_mx, i_cell, dt, gamma, n_pulse=n_pulse)
    return {
        "n_packets": int(len(phis)),
        "n_ph": int(len(flist)),
        "labels_head": labs[:4],
        "split": split,
        "gamma_target": gamma,
        "compare": cmp,
        "A_ph": a_ph,
        "A_mx": a_mx,
        "no_hand_G_self": True,
    }


def long_window_bins(prot, e_perp, *, t_obs_over_tp: float, ntpc: int, harm_max: float = 8.0):
    """I on [0, Tp], photon grid on [0, T_obs]. ω=0 excluded; IR densifies with T_obs."""
    from q8_m0_io_audit import bin_packet
    from q8_m0_photon_op import omega_weights, photon_omega_grid

    tp = float(prot.t_pulse)
    dt = float(prot.period) / int(ntpc)
    t_obs = float(t_obs_over_tp) * tp
    times = np.arange(0.0, t_obs + 0.5 * dt, dt)
    n_pulse = int(np.searchsorted(times, tp + 0.5 * dt))
    i_cell = np.zeros((times.size, 2), dtype=float)
    i_cell[:n_pulse] = prescribed_I(times[:n_pulse], e_perp, kind="sin2")
    omega = photon_omega_grid(tp, prot.omega0, harm_max=float(harm_max), t_obs=t_obs)
    dw = omega_weights(omega)
    phis = [bin_packet(omega, dw, j, e_perp) for j in range(int(omega.size))]
    return {
        "times": times,
        "dt": dt,
        "i_cell": i_cell,
        "n_pulse": n_pulse,
        "omega": omega,
        "dw": dw,
        "phis": phis,
        "t_obs_over_tp": float(t_obs_over_tp),
        "n_omega": int(omega.size),
    }
