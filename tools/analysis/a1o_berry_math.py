#!/usr/bin/env python3
"""Gauge-aware r comparisons for A1-O (Hamiltonian-gauge Berry from MMN)."""
from __future__ import annotations

import numpy as np


def polar_right_unitary(overlap: np.ndarray) -> np.ndarray:
    """Unitary U maximising Re Tr(overlap @ U)."""
    w, _s, vh = np.linalg.svd(overlap, full_matrices=True)
    return vh.conj().T @ w.conj().T


def align_overlap(mat: np.ndarray) -> np.ndarray:
    """Full-subspace polar. Do not use before finite-difference Berry: it sends
    a nearly-unitary M to I and wipes the antihermitian connection."""
    return mat @ polar_right_unitary(mat)


def phase_align_columns(mat: np.ndarray) -> np.ndarray:
    """Right-multiply by diagonal phases so diag(M) is real and >= 0.

    Removes per-band neighbor phases. Leaves off-diagonal Berry / r intact.
    """
    diag = np.diag(mat)
    phase = np.ones(mat.shape[1], dtype=complex)
    ok = np.abs(diag) > 1.0e-14
    phase[ok] = np.conj(diag[ok]) / np.abs(diag[ok])
    return mat * phase


def berry_from_pm_crystal(
    m_px: np.ndarray,
    m_mx: np.ndarray,
    m_py: np.ndarray,
    m_my: np.ndarray,
    delta: float,
    recip: np.ndarray,
) -> np.ndarray:
    """Symmetric FD along crystal k1,k2, then A_cart = recip^{-1} A_frac.

    ``recip`` rows are Cartesian b-vectors (Wannier90 nnkp convention, 1/Ang).
    """
    a1 = 1j * (m_px - m_mx) / (2.0 * delta)
    a2 = 1j * (m_py - m_my) / (2.0 * delta)
    a3 = np.zeros_like(a1)
    a_frac = np.stack([a1, a2, a3], axis=0)
    inv = np.linalg.inv(recip)
    r_ang = np.tensordot(inv, a_frac, axes=(1, 0))
    for alpha in range(3):
        r_ang[alpha] = 0.5 * (r_ang[alpha] + r_ang[alpha].conj().T)
    return r_ang


def window_indices(eig: np.ndarray, win_min: float, win_max: float) -> np.ndarray:
    return np.flatnonzero((eig >= win_min) & (eig <= win_max))


def energy_pi_q(n_window: int, n_q: int) -> tuple[np.ndarray, np.ndarray]:
    if n_window <= n_q:
        raise ValueError(f"window {n_window} too small for Q={n_q}")
    pi = np.arange(n_window - n_q)
    q = np.arange(n_window - n_q, n_window)
    return pi, q


def tb_weight_pi_q(u: np.ndarray, n_pi_wf: int, n_q_wf: int) -> tuple[np.ndarray, np.ndarray]:
    w_q = np.sum(np.abs(u[n_pi_wf : n_pi_wf + n_q_wf, :]) ** 2, axis=0)
    q = np.flatnonzero(w_q >= 0.5)
    pi = np.flatnonzero(w_q < 0.5)
    return pi, q


def frobenius_block(
    r_cart: np.ndarray, left: np.ndarray, right: np.ndarray, zero_diag: bool = False
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, axis in (("x", 0), ("y", 1), ("z", 2)):
        block = np.array(r_cart[axis][np.ix_(left, right)], copy=True)
        if zero_diag and left.size == right.size and np.array_equal(left, right):
            np.fill_diagonal(block, 0.0)
        out[f"F_{name}"] = float(np.linalg.norm(block))
        out[f"maxabs_{name}"] = float(np.max(np.abs(block))) if block.size else 0.0
        if min(block.shape) > 0:
            s = np.linalg.svd(block, compute_uv=False)
            out[f"svd1_{name}"] = float(s[0])
            out[f"svd_sum_{name}"] = float(np.sum(s))
        else:
            out[f"svd1_{name}"] = 0.0
            out[f"svd_sum_{name}"] = 0.0
    return out


def isolated_mask(eig: np.ndarray, gap_eV: float) -> np.ndarray:
    n = eig.size
    iso = np.ones(n, dtype=bool)
    if n == 0:
        return iso
    gaps_left = np.full(n, np.inf)
    gaps_right = np.full(n, np.inf)
    if n > 1:
        d = np.diff(eig)
        gaps_left[1:] = d
        gaps_right[:-1] = d
    iso &= np.minimum(gaps_left, gaps_right) >= gap_eV
    return iso


def isolated_pair_abs(
    r_axis: np.ndarray,
    eig: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    gap_eV: float,
    max_pairs: int = 12,
) -> list[dict]:
    iso = isolated_mask(eig, gap_eV)
    rows = []
    for i in left:
        if not iso[i]:
            continue
        for j in right:
            if not iso[j]:
                continue
            rows.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "e_i": float(eig[i]),
                    "e_j": float(eig[j]),
                    "abs_r": float(np.abs(r_axis[i, j])),
                }
            )
    rows.sort(key=lambda d: d["abs_r"], reverse=True)
    return rows[:max_pairs]


def interpolate_tb_hr(
    irvec: np.ndarray, ndegen: np.ndarray, h_r: np.ndarray, r_r: np.ndarray, kfrac: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.exp(2j * np.pi * (irvec @ kfrac)) / ndegen
    hk = np.tensordot(phase, h_r, axes=(0, 0))
    hk = 0.5 * (hk + hk.conj().T)
    rk = np.tensordot(phase, r_r, axes=(0, 0))
    evals, u = np.linalg.eigh(hk)
    r_cart = np.empty((3, evals.size, evals.size), dtype=complex)
    for alpha in range(3):
        r_w = rk[:, :, alpha]
        r_w = 0.5 * (r_w + r_w.conj().T)
        r_cart[alpha] = u.conj().T @ r_w @ u
    return np.asarray(evals.real), r_cart, u
