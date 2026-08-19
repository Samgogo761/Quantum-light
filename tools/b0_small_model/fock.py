"""Exact Fock-space correlators for a quadratic (or general) few-orbital fermion model.

Hilbert space dimension 2^n_orb. Intended only for B0 gates (n_orb ≤ 8).
Convention matches wick.py / 完整理论 (7.5): J = Σ A_ij c_i† c_j.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def dim(n_orb: int) -> int:
    return 1 << n_orb


def apply_annihilation(psi: Array, orbital: int, n_orb: int) -> Array:
    """Jordan–Wigner: c_i |…⟩, parity from bits < i."""
    out = np.zeros_like(psi)
    bit = 1 << orbital
    for state in range(psi.size):
        amp = psi[state]
        if amp == 0.0:
            continue
        if state & bit == 0:
            continue
        parity = 0
        lower = state & (bit - 1)
        parity = bin(lower).count("1")
        sign = -1.0 if parity & 1 else 1.0
        out[state ^ bit] += sign * amp
    return out


def apply_creation(psi: Array, orbital: int, n_orb: int) -> Array:
    out = np.zeros_like(psi)
    bit = 1 << orbital
    for state in range(psi.size):
        amp = psi[state]
        if amp == 0.0:
            continue
        if state & bit:
            continue
        parity = bin(state & (bit - 1)).count("1")
        sign = -1.0 if parity & 1 else 1.0
        out[state ^ bit] += sign * amp
    return out


def annihilation_matrix(n_orb: int, orbital: int) -> Array:
    d = dim(n_orb)
    mat = np.zeros((d, d), dtype=complex)
    for src in range(d):
        e_src = np.zeros(d, dtype=complex)
        e_src[src] = 1.0
        mat[:, src] = apply_annihilation(e_src, orbital, n_orb)
    return mat


def creation_matrix(n_orb: int, orbital: int) -> Array:
    d = dim(n_orb)
    mat = np.zeros((d, d), dtype=complex)
    for src in range(d):
        e_src = np.zeros(d, dtype=complex)
        e_src[src] = 1.0
        mat[:, src] = apply_creation(e_src, orbital, n_orb)
    return mat


def bilinear_matrix(a: Array) -> Array:
    """Many-body matrix of Σ A_ij c_i† c_j."""
    n_orb = a.shape[0]
    d = dim(n_orb)
    mat = np.zeros((d, d), dtype=complex)
    for j in range(n_orb):
        for i in range(n_orb):
            coeff = a[i, j]
            if abs(coeff) < 1.0e-18:
                continue
            for src in range(d):
                e_src = np.zeros(d, dtype=complex)
                e_src[src] = 1.0
                tmp = apply_annihilation(e_src, j, n_orb)
                dst = apply_creation(tmp, i, n_orb)
                mat[:, src] += coeff * dst
    return mat


def quadratic_hamiltonian(h: Array) -> Array:
    return bilinear_matrix(h)


def thermal_density(h_mb: Array, beta: float) -> Array:
    evals, vecs = np.linalg.eigh(h_mb)
    shifted = evals - evals.min()
    w = np.exp(-beta * shifted)
    z = w.sum()
    return (vecs * (w / z)) @ vecs.conj().T


def connected_equal_time(rho_mb: Array, ja: Array, jb: Array) -> complex:
    ja_jb = ja @ jb
    mean_ab = np.trace(rho_mb @ ja_jb)
    mean_a = np.trace(rho_mb @ ja)
    mean_b = np.trace(rho_mb @ jb)
    return mean_ab - mean_a * mean_b


def two_time_greater(
    rho_mb_tprime: Array,
    ja: Array,
    jb: Array,
    u_mb: Array,
) -> complex:
    """⟨J_A(t) J_B(t')⟩_conn with U = U(t,t'), t ≥ t', closed system."""
    jb_rho = jb @ rho_mb_tprime
    evolved = u_mb @ jb_rho @ u_mb.conj().T
    corr = np.trace(ja @ evolved)
    mean_a = np.trace(ja @ (u_mb @ rho_mb_tprime @ u_mb.conj().T))
    mean_b = np.trace(jb @ rho_mb_tprime)
    return corr - mean_a * mean_b
