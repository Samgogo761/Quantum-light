"""Explicit GKSL / quantum-regression for a two-level system.

Implements 完整理论 §7.5–7.6 (7.11)–(7.13). The auxiliary operator Bρ is
propagated without Hermitianization or renormalization.

This is not the production SBE T2 map (instantaneous-eigenbasis damping of 1RDM).
"""
from __future__ import annotations

import numpy as np


def _expm(mat: np.ndarray) -> np.ndarray:
    """Scaling-and-squaring Taylor expm for small non-Hermitian superoperators."""
    n = mat.shape[0]
    norm = float(np.linalg.norm(mat, ord=np.inf))
    s = max(0, int(np.ceil(np.log2(max(norm, 1.0e-16)))) + 1)
    a = mat / (2.0 ** s)
    term = np.eye(n, dtype=complex)
    total = np.eye(n, dtype=complex)
    for k in range(1, 18):
        term = term @ a / k
        total = total + term
    for _ in range(s):
        total = total @ total
    return total

Array = np.ndarray

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SP = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # |e⟩⟨g|
SM = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)  # |g⟩⟨e|


def vectorize(x: Array) -> Array:
    return x.reshape(-1, order="F")


def unvectorize(v: Array) -> Array:
    n = int(np.sqrt(v.size))
    return v.reshape((n, n), order="F")


def gksl_superoperator(h: Array, jumps: list[Array]) -> Array:
    """L vec(X) = vec(-i[H,X] + Σ (L X L† - 1/2 {L†L, X}))."""
    n = h.shape[0]
    eye = np.eye(n, dtype=complex)
    super_h = -1j * (np.kron(eye, h) - np.kron(h.T, eye))
    super_d = np.zeros((n * n, n * n), dtype=complex)
    for jump in jumps:
        ld = jump.conj().T @ jump
        super_d += np.kron(jump.conj(), jump)
        super_d -= 0.5 * (np.kron(eye, ld) + np.kron(ld.T, eye))
    return super_h + super_d


def propagate_operator(superop: Array, x0: Array, dt: float) -> Array:
    v = _expm(superop * dt) @ vectorize(x0)
    return unvectorize(v)


def qrt_two_time(
    superop: Array,
    rho_tprime: Array,
    a: Array,
    b: Array,
    dt: float,
) -> complex:
    """⟨A(t) B(t')⟩ = Tr[A G(t,t')(B ρ(t'))], t = t'+dt. Do not Hermitianize Bρ."""
    aux = b @ rho_tprime
    aux_t = propagate_operator(superop, aux, dt)
    return np.trace(a @ aux_t)


def qrt_two_time_lesser(
    superop: Array,
    rho_tprime: Array,
    a: Array,
    b: Array,
    dt: float,
) -> complex:
    """⟨B(t') A(t)⟩ = Tr[A G(t,t')(ρ(t') B)], t = t'+dt. Auxiliary is ρB, not Bρ."""
    aux = rho_tprime @ b
    aux_t = propagate_operator(superop, aux, dt)
    return np.trace(a @ aux_t)


def connected_qrt_two_time(
    superop: Array,
    rho_tprime: Array,
    a: Array,
    b: Array,
    dt: float,
) -> complex:
    corr = qrt_two_time(superop, rho_tprime, a, b, dt)
    rho_t = propagate_operator(superop, rho_tprime, dt)
    return corr - np.trace(a @ rho_t) * np.trace(b @ rho_tprime)


def connected_qrt_two_time_lesser(
    superop: Array,
    rho_tprime: Array,
    a: Array,
    b: Array,
    dt: float,
) -> complex:
    corr = qrt_two_time_lesser(superop, rho_tprime, a, b, dt)
    rho_t = propagate_operator(superop, rho_tprime, dt)
    return corr - np.trace(a @ rho_t) * np.trace(b @ rho_tprime)


def connected_qrt_symmetric(
    superop: Array,
    rho_tprime: Array,
    a: Array,
    b: Array,
    dt: float,
) -> complex:
    c_gt = connected_qrt_two_time(superop, rho_tprime, a, b, dt)
    c_lt = connected_qrt_two_time_lesser(superop, rho_tprime, a, b, dt)
    return 0.5 * (c_gt + c_lt)


def amplitude_damping_superop(omega: float, gamma: float) -> Array:
    h = 0.5 * omega * SZ
    return gksl_superoperator(h, [np.sqrt(gamma) * SM])
