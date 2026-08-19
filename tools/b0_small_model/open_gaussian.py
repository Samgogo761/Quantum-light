"""Gaussian-preserving fermionic GKSL: 1RDM Lyapunov + two-time Wick.

Linear jumps L ∝ c_i and L ∝ c_i† keep the state quasifree, so Wick is exact.
Do not insert the production SBE T2 map here.
"""
from __future__ import annotations

import numpy as np

from qrt import _expm, vectorize, unvectorize
from wick import equal_time_wick

Array = np.ndarray


def heff(h: Array, gamma_out: Array, gamma_in: Array) -> Array:
    return h - 0.5j * (gamma_out + gamma_in)


def onerdm_superoperator(h: Array, gamma_out: Array, gamma_in: Array) -> Array:
    """Column-major vec(dρ/dt) = S vec(ρ) + vec(γ_in)."""
    n = h.shape[0]
    eye = np.eye(n, dtype=complex)
    gam = gamma_out + gamma_in
    return (
        -1j * np.kron(eye, h)
        + 1j * np.kron(h.T, eye)
        - 0.5 * np.kron(eye, gam)
        - 0.5 * np.kron(gam.T, eye)
    )


def propagate_onerdm(
    h: Array,
    gamma_out: Array,
    gamma_in: Array,
    rho0: Array,
    dt: float,
) -> Array:
    n = h.shape[0]
    superop = onerdm_superoperator(h, gamma_out, gamma_in)
    vec_in = vectorize(gamma_in)
    vec0 = vectorize(rho0)
    if np.linalg.norm(vec_in) < 1.0e-18:
        return unvectorize(_expm(superop * dt) @ vec0)
    # S ρ_ss + γ_in = 0
    rho_ss = unvectorize(-np.linalg.solve(superop, vec_in))
    delta = _expm(superop * dt) @ vectorize(rho0 - rho_ss)
    return unvectorize(delta) + rho_ss


def single_particle_g(h: Array, gamma_out: Array, gamma_in: Array, dt: float) -> Array:
    return _expm(-1j * heff(h, gamma_out, gamma_in) * dt)


def two_time_wick_greater_open(
    a: Array,
    b: Array,
    rho_tprime: Array,
    g_t_tprime: Array,
) -> complex:
    """C^> = ⟨A(t) B(t')⟩_conn with t ≥ t'. G replaces closed-system U."""
    n = rho_tprime.shape[0]
    eye = np.eye(n, dtype=complex)
    sandwiched = g_t_tprime @ (eye - rho_tprime) @ b @ rho_tprime @ g_t_tprime.conj().T
    return np.trace(a @ sandwiched)


def two_time_wick_lesser_open(
    a: Array,
    b: Array,
    rho_tprime: Array,
    g_t_tprime: Array,
) -> complex:
    """C^< = ⟨B(t') A(t)⟩_conn with t ≥ t'."""
    n = rho_tprime.shape[0]
    eye = np.eye(n, dtype=complex)
    sandwiched = g_t_tprime @ rho_tprime @ b @ (eye - rho_tprime) @ g_t_tprime.conj().T
    return np.trace(a @ sandwiched)


def two_time_wick_symmetric_open(
    a: Array,
    b: Array,
    rho_tprime: Array,
    g_t_tprime: Array,
) -> complex:
    c_gt = two_time_wick_greater_open(a, b, rho_tprime, g_t_tprime)
    c_lt = two_time_wick_lesser_open(a, b, rho_tprime, g_t_tprime)
    return 0.5 * (c_gt + c_lt)


def equal_time_from_onerdm(a: Array, b: Array, rho: Array) -> complex:
    return equal_time_wick(a, b, rho)
