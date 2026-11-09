#!/usr/bin/env python3
"""1-k electron-photon Fock cut from the two-port sheet, not a cavity.

Incoming field is the interaction-picture packet operator
    δA(t) = √(η² N_cell) (F*(t) b + F(t) b†) ê_⊥
with b truncated to n=0..n_max. There is no extra ħω b†b: b is the
incoming free packet, not a standing cavity mode.

Self-field is the same G_self: A_self(t) = γ ∫ ⟨I⟩ dt (Ehrenfest, shared
c-number). That is a truncation of operator radiation reaction.

Hilbert space: C(8,4)=70 electronic dets ⊗ (n_max+1) photons.
This checks Born / Volterra on that truncated model. It does not
adjudicate the free-space continuum.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from q8_m0_classical import kappa_shift, p1_a2_protocol
from q8_m0_gsheet import gamma_self
from q8_m0_mode_io import eta2_ncell, mode_waveforms
from q8_m0_model import N_OCC, R0, current_ops, h8
from q8_m0_s1_kernel import pack_complex, parallel_perp_basis

FOCK_ID = "fock_1k_incoming_packet_plus_Ehrenfest_Gself_20260921"
N_DET = 70  # C(8,4)


def slater_list():
    return list(combinations(range(8), N_OCC))


def _index_map(dets):
    return {d: i for i, d in enumerate(dets)}


def hop_replace(det, p, q):
    lst = list(det)
    i = lst.index(p)
    sign = -1 if i % 2 else 1
    lst.pop(i)
    j = 0
    while j < len(lst) and lst[j] < q:
        j += 1
    sign *= -1 if j % 2 else 1
    lst.insert(j, q)
    return tuple(lst), sign


def onebody_to_ci(h1, dets, idx) -> np.ndarray:
    h1 = np.asarray(h1)
    n = len(dets)
    h70 = np.zeros((n, n), dtype=complex)
    for i, det in enumerate(dets):
        for p in det:
            h70[i, i] += h1[p, p]
        for p in det:
            for q in range(8):
                if q in det:
                    continue
                new, sign = hop_replace(det, p, q)
                j = idx[new]
                h70[j, i] += sign * h1[q, p]
    h70 = 0.5 * (h70 + h70.conj().T)
    return h70


def boson_ops(n_max: int):
    dim = n_max + 1
    b = np.zeros((dim, dim), dtype=complex)
    for n in range(n_max):
        b[n, n + 1] = np.sqrt(n + 1.0)
    bd = b.conj().T
    return b, bd


def slater_from_occupied(vv_occ, dets) -> np.ndarray:
    """Site-basis CI coefficients of a 4-electron Slater determinant."""
    vv_occ = np.asarray(vv_occ)
    psi = np.zeros(len(dets), dtype=complex)
    for i, det in enumerate(dets):
        psi[i] = np.linalg.det(vv_occ[list(det), :])
    nrm = float(np.linalg.norm(psi))
    if nrm < 1e-18:
        raise RuntimeError("Slater CI expansion vanished")
    return psi / nrm


def rdm1_from_ci(psi70, dets, idx) -> np.ndarray:
    """γ_{ab} = ⟨c†_a c_b⟩."""
    psi70 = np.asarray(psi70)
    rdm = np.zeros((8, 8), dtype=complex)
    for j, dj in enumerate(dets):
        cj = psi70[j]
        if abs(cj) < 1e-18:
            continue
        for p in dj:
            rdm[p, p] += np.conj(cj) * cj
            for q in range(8):
                if q in dj:
                    continue
                di, sign = hop_replace(dj, p, q)
                rdm[q, p] += np.conj(psi70[idx[di]]) * cj * sign
    return rdm


def gs_occupied_orbitals(h0) -> np.ndarray:
    _evals, vecs = np.linalg.eigh(h0)
    return vecs[:, :N_OCC]


def photon_field_matrix(F_t: complex, b, bd):
    """Hermitian M = F* b + F b†."""
    return np.conj(F_t) * b + F_t * bd


def evolve_midpoint_ci(psi, h70, dt: float):
    evals, u = np.linalg.eigh(h70)
    phase = np.exp(-1j * evals * dt)
    return u @ ((u.conj().T @ psi) * phase)


def current_ci(j1, psi70, dets, idx) -> np.ndarray:
    """⟨I_a⟩ from a 70-vector and one-body current (2,8,8)."""
    j1 = np.asarray(j1)
    rdm = rdm1_from_ci(psi70, dets, idx)
    out = np.zeros(2, dtype=float)
    for a in range(2):
        out[a] = float(np.real(np.einsum("ab,ab->", j1[a], rdm)))
    return out


def run_fock_1k(
    geo,
    *,
    n_max: int = 3,
    nt_per_cycle: int = 8,
    neel: int = 1,
    kfrac=(0.0, 0.0),
    ehrenfest_self: bool = True,
    incoming_scale: float = 1.0,
    verbose: bool = False,
) -> dict:
    """Short 1-k TDCI. incoming_scale multiplies the packet coupling (weak-coupling test)."""
    dets = slater_list()
    idx = _index_map(dets)
    n_det = len(dets)
    n_ph = n_max + 1
    prot = p1_a2_protocol(0.8)
    nt = prot.n_cycles * nt_per_cycle
    dt = prot.period / nt_per_cycle
    times = np.arange(nt + 1) * dt
    kc = geo.k_cart(np.asarray(kfrac, dtype=float))
    if kc.ndim == 1:
        kc = kc[None, :]
    k2 = kc[0]
    h0 = h8(geo, R0, k2, neel)
    vv_occ = gs_occupied_orbitals(h0)
    psi_el0 = slater_from_occupied(vv_occ, dets)
    b, bd = boson_ops(n_max)
    psi = np.zeros((n_ph, n_det), dtype=complex)
    psi[0] = psi_el0
    mw = mode_waveforms(times, prot.omega0, prot.t_pulse, geo)
    F = np.asarray(mw["F"])
    e_perp = np.asarray(mw["e_perp"])
    eta = float(np.sqrt(eta2_ncell(geo))) * float(incoming_scale)
    gamma = gamma_self(geo) if ehrenfest_self else 0.0
    a_self = np.zeros(2, dtype=float)
    i_hist = np.zeros((nt + 1, 2), dtype=float)
    di_hist = np.zeros((nt + 1, 2), dtype=float)
    i_p1_hist = np.zeros((nt + 1, 2), dtype=float)
    nph_hist = np.zeros(nt + 1, dtype=float)
    comm_op = b @ bd - bd @ b
    comm_vac = complex(np.vdot(np.array([1.0] + [0.0] * (n_ph - 1)), comm_op @ np.array([1.0] + [0.0] * (n_ph - 1))))
    psi_p1 = psi_el0.copy()

    def kappa_at(t, m_eig):
        a_cl = kappa_shift(t, geo, prot, +1)
        return k2 + a_cl + eta * float(m_eig) * e_perp + a_self

    def i_from_psi70(psi70, kap):
        j1 = current_ops(geo, R0, kap, neel)
        return current_ci(j1, psi70, dets, idx)

    for n in range(nt + 1):
        t = times[n]
        a_cl = kappa_shift(t, geo, prot, +1)
        i_p1 = i_from_psi70(psi_p1, k2 + a_cl)
        i_p1_hist[n] = i_p1
        Ft = complex(F[n])
        M = photon_field_matrix(Ft, b, bd)
        evals, u = np.linalg.eigh(M)
        psi_m = u.conj().T @ psi
        i_now = np.zeros(2, dtype=float)
        for alpha in range(n_ph):
            kap = kappa_at(t, evals[alpha])
            i_now += i_from_psi70(psi_m[alpha], kap)
        i_hist[n] = i_now
        di_hist[n] = i_now - i_p1
        pop = np.sum(np.abs(psi) ** 2, axis=1)
        nph_hist[n] = float(np.dot(np.arange(n_ph), pop))
        if n == nt:
            break
        tmid = t + 0.5 * dt
        Fmid = 0.5 * (F[n] + F[n + 1])
        Mmid = photon_field_matrix(complex(Fmid), b, bd)
        evals, u = np.linalg.eigh(Mmid)
        psi_m = u.conj().T @ psi
        for alpha in range(n_ph):
            kap = kappa_at(tmid, evals[alpha])
            h1 = h8(geo, R0, kap, neel)
            h70 = onebody_to_ci(h1, dets, idx)
            psi_m[alpha] = evolve_midpoint_ci(psi_m[alpha], h70, dt)
        psi = u @ psi_m
        nrm = np.linalg.norm(psi)
        if nrm > 0:
            psi = psi / nrm
        h_p1 = h8(geo, R0, k2 + kappa_shift(tmid, geo, prot, +1), neel)
        psi_p1 = evolve_midpoint_ci(psi_p1, onebody_to_ci(h_p1, dets, idx), dt)
        nrm_p = np.linalg.norm(psi_p1)
        if nrm_p > 0:
            psi_p1 = psi_p1 / nrm_p
        if ehrenfest_self:
            a_self = a_self + gamma * dt * di_hist[n]
        if verbose and n % max(1, nt // 4) == 0:
            print(f"  fock step {n}/{nt} nph={nph_hist[n]:.4f}", flush=True)

    pop = np.sum(np.abs(psi) ** 2, axis=1)
    return {
        "fock_id": FOCK_ID,
        "n_max": int(n_max),
        "n_det": n_det,
        "dim": int(n_ph * n_det),
        "nt": int(nt),
        "dt": float(dt),
        "kfrac": [float(kfrac[0]), float(kfrac[1])],
        "gs_occ": "energy_eigen_slater_expanded_in_site_dets",
        "ehrenfest_self": bool(ehrenfest_self),
        "incoming_scale": float(incoming_scale),
        "eta_1cell_scaled": eta,
        "gamma_self": float(gamma),
        "I": i_hist,
        "dI": di_hist,
        "I_p1": i_p1_hist,
        "n_photon_t": nph_hist,
        "n_photon_final": float(nph_hist[-1]),
        "photon_pop_final": pop.tolist(),
        "norm_final": float(np.linalg.norm(psi)),
        "truncated_Tr_commutator_minus_dim": float(np.real(np.trace(comm_op)) - n_ph),
        "vacuum_commutator": pack_complex(comm_vac),
        "cavity_omega_bdb": False,
        "not_free_space_continuum": True,
        "classical_J_not_in_G_self": True,
        "times": times,
    }
