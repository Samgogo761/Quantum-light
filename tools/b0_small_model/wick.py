"""Quasifree/Wick current noise for quadratic fermions.

Implements 完整理论 §7.2–7.4 (2026-07-29):

    ρ_ij = ⟨c_j† c_i⟩                         (7.5)
    ⟨ΔJ_A ΔJ_B⟩ = Tr[A (I-ρ) B ρ]            (7.6)  equal-time Pauli
    C^>_AB(t,t') = Tr[A U (I-ρ') B ρ' U†]    (7.10) closed single-particle

J_X = Σ_{ij} X_ij c_i† c_j. This is the Gaussian-fermion route, not QRT.
Do not feed the production SBE T2 map into these formulas.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def onerdm_from_occupations(u: Array, occupations: Array) -> Array:
    """R = U f U† with ρ_ij = ⟨c_j† c_i⟩."""
    f = np.asarray(occupations, dtype=float)
    return (u * f) @ u.conj().T


def fermi_dirac(evals: Array, beta: float, mu: float = 0.0) -> Array:
    x = np.clip(beta * (np.asarray(evals, dtype=float) - mu), -60.0, 60.0)
    return 1.0 / (np.exp(x) + 1.0)


def thermal_onerdm(h: Array, beta: float, mu: float = 0.0) -> Array:
    evals, u = np.linalg.eigh(h)
    return onerdm_from_occupations(u, fermi_dirac(evals, beta, mu))


def equal_time_wick(a: Array, b: Array, rho: Array) -> complex:
    """Connected equal-time correlator (7.6)."""
    n = rho.shape[0]
    eye = np.eye(n, dtype=complex)
    return np.trace(a @ (eye - rho) @ b @ rho)


def mean_current(a: Array, rho: Array) -> complex:
    return np.trace(a @ rho)


def single_particle_u(h: Array, dt: float) -> Array:
    evals, u = np.linalg.eigh(h)
    phase = np.exp(-1j * evals * dt)
    return (u * phase) @ u.conj().T


def two_time_wick_greater(
    a: Array,
    b: Array,
    rho_tprime: Array,
    u_t_tprime: Array,
) -> complex:
    """Closed-system (7.10) for t ≥ t'. A, B are Schrödinger single-particle matrices."""
    n = rho_tprime.shape[0]
    eye = np.eye(n, dtype=complex)
    sandwiched = u_t_tprime @ (eye - rho_tprime) @ b @ rho_tprime @ u_t_tprime.conj().T
    return np.trace(a @ sandwiched)
