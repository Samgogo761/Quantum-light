"""Finite-bandwidth normalized harmonic modes F_n(ω).

FFT convention: e^{-i ω t} (numpy rfft / production mod_hhg).
Not a single FFT bin. Not an experimental detector.
∫ dω F_n*(ω) F_m(ω) = δ_nm on the discrete ω grid (trapezoid).
First set: n = 2, 5, 7, 9. H10 is excluded.
"""
from __future__ import annotations

import math

import numpy as np

HARMONICS = (2, 5, 7, 9)


def raised_cosine(omega: np.ndarray, center: float, half_width: float) -> np.ndarray:
    """Hann window on [center-half_width, center+half_width], else 0."""
    x = (omega - center) / half_width
    out = np.zeros_like(omega, dtype=float)
    mask = np.abs(x) <= 1.0
    out[mask] = 0.5 * (1.0 + np.cos(math.pi * x[mask]))
    return out


def build_modes(
    omega: np.ndarray,
    omega0: float,
    harmonics: tuple[int, ...] = HARMONICS,
    half_width_over_omega0: float = 0.45,
) -> dict[int, np.ndarray]:
    """Orthonormal F_n on the positive-frequency grid. Uses e^{-iωt} rfft bins."""
    dw = float(np.mean(np.diff(omega))) if omega.size > 1 else 1.0
    raw = []
    for n in harmonics:
        fn = raised_cosine(omega, n * omega0, half_width_over_omega0 * omega0)
        raw.append(fn)
    # Gram–Schmidt with discrete inner product ⟨f|g⟩ = Σ f* g dw
    modes: dict[int, np.ndarray] = {}
    orthonormal = []
    for n, vec in zip(harmonics, raw):
        v = vec.astype(complex)
        for prev in orthonormal:
            v = v - np.sum(np.conjugate(prev) * v) * dw * prev
        norm = math.sqrt(float(np.real(np.sum(np.conjugate(v) * v) * dw)))
        if norm < 1e-18:
            raise RuntimeError(f"mode H{n} collapsed under Gram-Schmidt")
        v = v / norm
        orthonormal.append(v)
        modes[n] = v
    return modes


def inner_product(f: np.ndarray, g: np.ndarray, dw: float) -> complex:
    return np.sum(np.conjugate(f) * g) * dw


def project_current(
    modes: dict[int, np.ndarray],
    jx_w: np.ndarray,
    jy_w: np.ndarray,
    dw: float,
) -> dict[int, complex]:
    """b_n proxy = ∫ F_n* (Jx + i Jy)/√2 dω. Classical-mixture amplitude, not â_out."""
    pol = (jx_w + 1j * jy_w) / math.sqrt(2.0)
    return {n: inner_product(fn, pol, dw) for n, fn in modes.items()}
