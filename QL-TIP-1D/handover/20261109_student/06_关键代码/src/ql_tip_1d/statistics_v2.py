"""Deterministic bright-BSV and thermal intensity averaging.

This module contains no electron dynamics.  It implements the frozen scalar
intensity distributions and piecewise-linear node weights used by the
continuation-v2 analysis.  Positive weights do not imply a non-negative
Glauber-Sudarshan P representation or quantum advantage.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np
from scipy.integrate import quad


def bsv_pdf(intensity_over_mu: float) -> float:
    """Bright single-mode squeezed-limit intensity density in I/mu."""
    x = float(intensity_over_mu)
    if x < 0.0:
        raise ValueError("Intensity must be non-negative")
    if x == 0.0:
        return math.inf
    return math.exp(-x / 2.0) / math.sqrt(2.0 * math.pi * x)


def thermal_pdf(intensity_over_mu: float) -> float:
    """Thermal intensity density in I/mu."""
    x = float(intensity_over_mu)
    if x < 0.0:
        raise ValueError("Intensity must be non-negative")
    return math.exp(-x)


def tail_bound(Imax_over_mu: float, state: str) -> float:
    """Return the exact omitted probability above a finite Imax."""
    x = float(Imax_over_mu)
    if x < 0.0:
        raise ValueError("Imax must be non-negative")
    if state == "bsv":
        return math.erfc(math.sqrt(x / 2.0))
    if state == "thermal":
        return math.exp(-x)
    raise ValueError(state)


def raw_moment(state: str, order: int) -> float:
    """Analytic raw moment of I/mu for the frozen weight laws."""
    if order < 0:
        raise ValueError("Moment order must be non-negative")
    if state == "bsv":
        return float(math.prod(range(1, 2 * order, 2)))
    if state == "thermal":
        return float(math.factorial(order))
    raise ValueError(state)


def _hat_bounds(nodes: np.ndarray, index: int, Imax: float) -> tuple[float, float]:
    """Return the support of the linear hat function for one node."""
    left = 0.0 if index == 0 else float(nodes[index - 1])
    right = float(Imax) if index == len(nodes) - 1 else float(nodes[index + 1])
    return left, right


def linear_weights(nodes: Iterable[float], state: str, Imax_over_mu: float,
                   epsabs: float = 1e-12, epsrel: float = 1e-11) -> np.ndarray:
    """Integrate p(I) times piecewise-linear hats over [0, Imax].

    The returned weights sum to 1-tail_bound, never to one after truncation.
    """
    node_values = np.asarray(list(nodes), dtype=float)
    if node_values.ndim != 1 or len(node_values) < 2 or node_values[0] != 0.0:
        raise ValueError("Nodes must be one-dimensional, start at zero, and contain at least two points")
    if np.any(np.diff(node_values) <= 0.0):
        raise ValueError("Nodes must be strictly increasing")
    Imax = float(Imax_over_mu)
    # Imax is the truncation point of the production piecewise-linear response.
    # It equals the last measured node: the final hat rises to 1 exactly at Imax
    # and the omitted probability above Imax is reported by tail_bound().
    if Imax < node_values[-1]:
        raise ValueError("Imax must not lie below the last response node")
    pdf: Callable[[float], float] = bsv_pdf if state == "bsv" else thermal_pdf if state == "thermal" else None  # type: ignore[assignment]
    if pdf is None:
        raise ValueError(state)
    weights = np.empty(len(node_values), dtype=float)
    for index, node in enumerate(node_values):
        left, right = _hat_bounds(node_values, index, Imax)
        if right <= left:
            weights[index] = 0.0
            continue
        if index == 0:
            def integrand(x: float) -> float:
                return pdf(x) * (right - x) / (right - node)
        elif index == len(node_values) - 1:
            def integrand(x: float) -> float:
                return pdf(x) * (x - left) / (node - left)
        else:
            def integrand(x: float) -> float:
                return pdf(x) * ((x - left) / (node - left) if x <= node else (right - x) / (right - node))
        value, _ = quad(integrand, left, right, epsabs=epsabs, epsrel=epsrel, limit=200,
                        points=[node] if left < node < right else None)
        weights[index] = value
    return weights


def linear_average(nodes: Iterable[float], yields: Iterable[float], weights: Iterable[float]) -> float:
    """Return sum_i w_i Y_i for the truncated piecewise-linear response."""
    node_values = np.asarray(list(nodes), dtype=float)
    y = np.asarray(list(yields), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if y.shape != node_values.shape or w.shape != node_values.shape:
        raise ValueError("Nodes, yields and weights must have equal one-dimensional shapes")
    if np.any(y < 0.0) or np.any(w < 0.0):
        raise ValueError("Yields and weights must be non-negative")
    return float(np.dot(w, y))


def linear_interpolate(nodes: Iterable[float], values: Iterable[float], query: Iterable[float]) -> np.ndarray:
    """Piecewise-linear interpolation in I/mu without extrapolation."""
    node_values = np.asarray(list(nodes), dtype=float)
    y = np.asarray(list(values), dtype=float)
    q = np.asarray(list(query), dtype=float)
    if np.any(q < node_values[0]) or np.any(q > node_values[-1]):
        raise ValueError("Query lies outside the response-node range")
    return np.interp(q, node_values, y)


def tight_quadrature_error(nodes: Iterable[float], yields: Iterable[float], state: str,
                           Imax_over_mu: float, loose: tuple[float, float] = (1e-8, 1e-8),
                           tight: tuple[float, float] = (1e-12, 1e-11)) -> float:
    """Estimate weight-integration error by repeating with tighter tolerances."""
    w_loose = linear_weights(nodes, state, Imax_over_mu, epsabs=loose[0], epsrel=loose[1])
    w_tight = linear_weights(nodes, state, Imax_over_mu, epsabs=tight[0], epsrel=tight[1])
    y = np.asarray(list(yields), dtype=float)
    return float(abs(np.dot(w_loose, y) - np.dot(w_tight, y)))


@dataclass(frozen=True)
class StatisticalResult:
    """One deterministic averaged yield and its explicit truncation bound."""

    state: str
    F_dc_V_per_nm: float
    Y_truncated: float
    tail_bound: float
    weights: tuple[float, ...]
    weight_sum: float
    quadrature_error: float

    @property
    def interval(self) -> tuple[float, float]:
        """Return the truncated lower audit value and upper value with tail."""
        return max(0.0, self.Y_truncated - self.quadrature_error), min(
            1.0, self.Y_truncated + self.quadrature_error + self.tail_bound)
