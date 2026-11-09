#!/usr/bin/env python3
"""Independent, static quantum-input preflight for the frozen M0 h8 model.

This is NOT the existing line-Q photon input/output model or a BSV experiment.
Four fermions occupy eight orbitals at one k. A single abstract photon mode
is Galerkin projected onto |0>,...,|n_max>. hbar=e=1, energy t_h, length Angstrom.
The free benchmark parameter g has units Angstrom^-1; no mode volume, pulse
history, or interacting initial state at nonzero coupling is selected here.

H = omega n + sum_{ij,delta} t_ij exp(i k.delta) c_i^dagger c_j
                  tensor P exp(i g scale (e.delta) X) P + onsite,
X=a+a^dagger. `scale` is an instantaneous real coupling multiplier, NOT a
prescribed pulse. There is no propagation routine. Galerkin projection of the
exponential is intentionally distinguished from exponentiating truncated X.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.special import eval_genlaguerre, gammaln

from q8_m0_model import M0Geometry, M0Params

MODEL_ID = "M0_single_k_four_fermion_quantum_input_static_preflight_v1"
DETS = tuple(combinations(range(8), 4))
DET_INDEX = {d: i for i, d in enumerate(DETS)}
N_ELECTRON = 4
ELECTRON_DIM = len(DETS)


@dataclass(frozen=True)
class Bond:
    row: int
    col: int
    delta: tuple[float, float]
    amplitude: complex


def _pair(out: list[Bond], row: int, col: int, delta, amp: complex) -> None:
    d = tuple(float(x) for x in np.asarray(delta)[:2])
    out.append(Bond(row, col, d, complex(amp)))
    out.append(Bond(col, row, (-d[0], -d[1]), complex(amp).conjugate()))


def hopping_bonds(geo: M0Geometry, p: M0Params) -> tuple[Bond, ...]:
    """Directed Fourier terms, including each adjoint exactly once.

    No bond is inferred from a numerical h8 derivative. This independent
    expansion covers NN, Kane-Mele NNN and all frozen interlayer bonds.
    """
    out: list[Bond] = []
    for layer in range(2):
        for spin in range(2):
            for delta in geo.nn_deltas_cart:
                _pair(out, 4 * layer + spin, 4 * layer + 2 + spin, delta, p.t_h)
        for sub in range(2):
            for spin in range(2):
                row = 4 * layer + 2 * sub + spin
                sign = (1 if sub == 0 else -1) * (1 if spin == 0 else -1)
                for nu, delta in geo.nnn_terms:
                    # 2 nu sin(k.delta) = -i nu exp(ik.delta) + h.c.
                    _pair(out, row, row, delta, -1j * p.lam_so * sign * nu)
    site = {"tA": 0, "tB": 2, "bA": 4, "bB": 6}
    amps = {"u": p.u, "v": p.v}
    for bond in geo.interlayer_bonds:
        for spin in range(2):
            _pair(out, site[bond["from"]] + spin, site[bond["to"]] + spin,
                  bond["delta_cart"], amps[bond["amplitude"]])
    return tuple(out)


def onsite(p: M0Params, neel: int) -> np.ndarray:
    if neel not in (-1, 1):
        raise ValueError("neel must be +1 or -1")
    return np.diag([neel * p.delta * (1 if layer == 0 else -1)
                    * (1 if spin == 0 else -1)
                    for layer in range(2) for sub in range(2) for spin in range(2)]).astype(complex)


def scalar_h_and_current(geo, p, k, neel):
    """Independent c-number h8 and j=-partial_k h in frozen code units."""
    k = np.asarray(k, dtype=float)
    h = onsite(p, neel)
    current = np.zeros((2, 8, 8), dtype=complex)
    for bond in hopping_bonds(geo, p):
        value = bond.amplitude * np.exp(1j * np.dot(k, bond.delta))
        h[bond.row, bond.col] += value
        current[:, bond.row, bond.col] += -1j * np.asarray(bond.delta) * value
    return h, current


@lru_cache(maxsize=64)
def fermion_bilinear(row: int, col: int) -> np.ndarray:
    """c_row^dagger c_col on the fixed four-particle determinant basis."""
    out = np.zeros((ELECTRON_DIM, ELECTRON_DIM), dtype=complex)
    for j, det in enumerate(DETS):
        if col not in det:
            continue
        work = list(det)
        pos = work.index(col)
        sign = (-1) ** pos
        work.pop(pos)
        if row in work:
            continue
        pos = sum(x < row for x in work)
        sign *= (-1) ** pos
        work.insert(pos, row)
        out[DET_INDEX[tuple(work)], j] = sign
    out.setflags(write=False)
    return out


def lift_one_body(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=complex)
    if h.shape != (8, 8):
        raise ValueError("expected an 8x8 one-body matrix")
    out = np.zeros((ELECTRON_DIM, ELECTRON_DIM), dtype=complex)
    for i, j in zip(*np.nonzero(h)):
        out += h[i, j] * fermion_bilinear(int(i), int(j))
    # Deliberately no Hermitization: a sign/adjoint error must remain visible.
    return out


def wedge_unitary(u: np.ndarray) -> np.ndarray:
    """Exterior-power representation Gamma(u), not dGamma(u)."""
    return np.array([[np.linalg.det(u[np.ix_(bra, ket)]) for ket in DETS]
                     for bra in DETS], dtype=complex)


def slater_state(occupied: np.ndarray) -> np.ndarray:
    """Unrenormalized exterior product; caller checks normalization."""
    if np.shape(occupied) != (8, 4):
        raise ValueError("expected four occupied orbitals")
    return np.array([np.linalg.det(occupied[list(det), :]) for det in DETS])


def boson_ops(n_max: int):
    if n_max < 0:
        raise ValueError("n_max must be nonnegative")
    a = np.diag(np.sqrt(np.arange(1, n_max + 1, dtype=float)), 1).astype(complex)
    return a, a.conj().T, np.diag(np.arange(n_max + 1, dtype=float))


def projected_peierls(q: float, n_max: int) -> np.ndarray:
    """<m| exp(i q X_infinity) |n>, using displacement D(iq).

    This compression is generally NOT unitary. Using exp(iq X_truncated)
    would silently choose a different finite-cutoff Hamiltonian.
    """
    if not np.isreal(q) or not np.isfinite(q) or n_max < 0:
        raise ValueError("real finite q and nonnegative cutoff required")
    q = float(q)
    out = np.empty((n_max + 1, n_max + 1), dtype=complex)
    for m in range(n_max + 1):
        for n in range(n_max + 1):
            lo, hi = min(m, n), max(m, n)
            # For alpha=iq, alpha and -alpha* coincide.
            out[m, n] = (np.exp(-0.5 * q * q + 0.5 * (gammaln(lo + 1) - gammaln(hi + 1)))
                         * (1j * q) ** (hi - lo)
                         * eval_genlaguerre(lo, hi - lo, q * q))
    return out


def a2_polarization(geo: M0Geometry) -> np.ndarray:
    e = np.asarray(geo.a2_cart, dtype=float)[:2]
    return e / np.linalg.norm(e)


def assemble_joint(geo, p, k, neel, *, n_max: int, omega: float,
                   g: float, scale: float = 1.0, polarization=None):
    """Static joint H and both probe currents on electron tensor photon.

    Current is -dH/d(probe kappa), at probe=0, from the very same bonds.
    omega>0; g is an abstract benchmark, not inferred experimental intensity.
    """
    if omega <= 0 or not np.isfinite([omega, g, scale]).all():
        raise ValueError("omega must be positive and all scalar parameters finite")
    e = a2_polarization(geo) if polarization is None else np.asarray(polarization, dtype=float)
    if e.shape != (2,) or not np.isclose(np.linalg.norm(e), 1.0, atol=1e-12):
        raise ValueError("polarization must be a real unit 2-vector")
    k = np.asarray(k, dtype=float)
    ip = np.eye(n_max + 1)
    _, _, n = boson_ops(n_max)
    h = np.kron(lift_one_body(onsite(p, neel)), ip)
    h += np.kron(np.eye(ELECTRON_DIM), omega * n)
    j = np.zeros((2,) + h.shape, dtype=complex)
    photon_cache = {}
    for bond in hopping_bonds(geo, p):
        q = float(g * scale * np.dot(e, bond.delta))
        if q not in photon_cache:
            photon_cache[q] = projected_peierls(q, n_max)
        phase = bond.amplitude * np.exp(1j * np.dot(k, bond.delta))
        term = np.kron(fermion_bilinear(bond.row, bond.col), photon_cache[q]) * phase
        h += term
        j += -1j * np.asarray(bond.delta)[:, None, None] * term
    return h, j


def coherent_coefficients(alpha: complex, n_max: int) -> np.ndarray:
    out = np.zeros(n_max + 1, dtype=complex)
    out[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for n in range(1, n_max + 1):
        out[n] = alpha * out[n - 1] / np.sqrt(n)
    return out


def squeezed_vacuum_coefficients(r: float, theta: float, n_max: int) -> np.ndarray:
    """S(z)|0>, z=r exp(i theta), unrenormalized truncated coefficients."""
    if r < 0:
        raise ValueError("r must be nonnegative")
    out = np.zeros(n_max + 1, dtype=complex)
    out[0] = 1.0 / np.sqrt(np.cosh(r))
    for n in range(2, n_max + 1, 2):
        out[n] = -np.exp(1j * theta) * np.tanh(r) * np.sqrt((n - 1) / n) * out[n - 2]
    return out


def state_diagnostics(coefficients: np.ndarray) -> dict:
    """Raw truncated moments plus mass loss; never silently renormalize."""
    c = np.asarray(coefficients, dtype=complex)
    n = np.arange(len(c), dtype=float)
    mass = float(np.vdot(c, c).real)
    aa = sum(np.conj(c[i]) * c[i + 2] * np.sqrt((i + 1) * (i + 2))
             for i in range(len(c) - 2))
    return {
        "retained_mass": mass,
        "omitted_mass": max(0.0, 1.0 - mass),
        "normalization_roundoff_above_one": max(0.0, mass - 1.0),
        "raw_mean_n": float(np.sum(n * np.abs(c) ** 2)),
        "raw_second_n": float(np.sum(n * n * np.abs(c) ** 2)),
        "raw_aa": {"real": float(np.real(aa)), "imag": float(np.imag(aa))},
        "top_two_mass": float(np.sum(np.abs(c[-2:]) ** 2)),
    }
