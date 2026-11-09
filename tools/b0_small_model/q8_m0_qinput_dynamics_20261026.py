#!/usr/bin/env python3
"""Short-range quantum-input dynamics for the A-baseline (method benchmark).

Scope
-----
Frozen M0 h8, ONE k, four fermions (70-dim), one projected photon mode
(n=0..n_max). Schroedinger picture, X = a + a^dagger, Peierls coupling
g*s(t). s(t) is a smooth switch-on/flat/switch-off envelope with s(t0)=0.

This is NOT a free-space BSV pulse, NOT a magnetic selection rule, and it
does not touch production G1/H2/U-V code. It exists to check that the exact
joint evolution is numerically trustworthy before any semiclassical
comparison is attempted.

Numerical strategy
------------------
H(t) = H_diag + sum_groups F_g (x) P e^{i q_g s(t) X} P,   q_g = g (e.delta).
Bonds are grouped by the scalar q, so one matvec costs one projected
displacement per group instead of one per bond.

Ramp segments: fixed-order Taylor propagator about the diagonal mean, so the
shifted spectrum has zero mean and ||H_shift dt|| stays O(1). The achieved
order is reported, never assumed.
Flat segment: H is time independent, so it is diagonalised once and every
source-side windowed observable is evaluated from the spectral expansion.
That part is exact up to the ramp propagator error.

Source-side observables (one window, one operator order)
--------------------------------------------------------
Jhat_{q,a} = int dt w(t) e^{i q omega0 t} J_a(t),   w = Hann on the flat segment
  <Jhat_{q,a}>   = sum_{nm} c_n^* c_m <n|J_a|m> W(q w0 + E_n - E_m)
  S_{q,ab}       = <Jhat_a^dag Jhat_b>
                 = c^dag [ (M^a . U)(M^b . V) ] c
                   U[n,m] = W(E_n - E_m - q w0),  V[m,l] = W(E_m - E_l + q w0)
with W(Om) = int_0^{T_w} sin^2(pi t/T_w) e^{i Om t} dt (closed form, stable).

S is a source-side two-time correlation integral. It is not a photon number
and must never be compared to |<J>|^2 without the connected part spelled out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import eval_genlaguerre, gammaln

import q8_m0_qinput_minimal_20261020 as qm
from q8_m0_model import R0, build_geometry

MODEL_ID = "M0_single_k_four_fermion_quantum_input_dynamics_trial_v1"
N_EL = qm.ELECTRON_DIM  # 70


# ---------------------------------------------------------------- envelope
@dataclass(frozen=True)
class Envelope:
    """sin^2 ramp, flat plateau, sin^2 fall. s(t0)=0 by construction.

    s(t) is an externally imposed coupling protocol, not a propagating pulse.
    Its time dependence does work on the system; that work is booked, not
    silently absorbed into an 'energy conservation' claim.
    """

    omega0: float
    ramp_cycles: float = 1.0
    flat_cycles: float = 2.0

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.omega0

    @property
    def t_ramp(self) -> float:
        return self.ramp_cycles * self.period

    @property
    def t_flat(self) -> float:
        return self.flat_cycles * self.period

    @property
    def t_total(self) -> float:
        return 2.0 * self.t_ramp + self.t_flat

    @property
    def plateau(self) -> tuple[float, float]:
        return (self.t_ramp, self.t_ramp + self.t_flat)

    def s(self, t: float) -> float:
        if t <= 0.0 or t >= self.t_total:
            return 0.0
        if t < self.t_ramp:
            return float(np.sin(0.5 * np.pi * t / self.t_ramp) ** 2)
        if t <= self.t_ramp + self.t_flat:
            return 1.0
        return float(np.sin(0.5 * np.pi * (self.t_total - t) / self.t_ramp) ** 2)

    def ds(self, t: float) -> float:
        """ds/dt. Nonzero only inside the two ramps."""
        if t <= 0.0 or t >= self.t_total:
            return 0.0
        if t < self.t_ramp:
            return float(0.5 * np.pi / self.t_ramp * np.sin(np.pi * t / self.t_ramp))
        if t <= self.t_ramp + self.t_flat:
            return 0.0
        return float(-0.5 * np.pi / self.t_ramp
                     * np.sin(np.pi * (self.t_total - t) / self.t_ramp))


# ---------------------------------------------------------------- operators
def projected_peierls_batch(qs, n_max: int) -> np.ndarray:
    """<m|exp(i q X_inf)|n> for a batch of real q; symmetric in (m,n).

    This is the Galerkin projection of the *infinite-space* displacement
    operator. It is not exp(i q P X P): the two differ by O(1) at finite
    cutoff (verified in the static preflight), so the choice must never be
    swapped silently inside a propagator.
    """
    qs = np.atleast_1d(np.asarray(qs, dtype=float))
    d1 = n_max + 1
    out = np.empty((len(qs), d1, d1), dtype=complex)
    q2 = qs * qs
    for m in range(d1):
        for n in range(m, d1):
            val = (np.exp(-0.5 * q2 + 0.5 * (gammaln(m + 1) - gammaln(n + 1)))
                   * (1j * qs) ** (n - m) * eval_genlaguerre(m, n - m, q2))
            out[:, m, n] = val
            out[:, n, m] = val
    return out


def onsite_determinant_energy(p, neel: int) -> np.ndarray:
    """Exchange onsite energy of each four-fermion determinant."""
    diag8 = np.real(np.diag(qm.onsite(p, neel)))
    return np.array([sum(diag8[i] for i in det) for det in qm.DETS], dtype=float)


@dataclass(frozen=True)
class JointModel:
    """Grouped joint Hamiltonian data for one (k, N, g, polarization, n_max)."""

    k: np.ndarray
    neel: int
    n_max: int
    omega0: float
    g: float
    e: np.ndarray
    diag: np.ndarray                    # (70*(n_max+1),)
    q_units: np.ndarray                 # (n_group,) q at s=1
    f_group: tuple[np.ndarray, ...]     # (70,70) electron blocks
    j_group: tuple[np.ndarray, ...]     # (2,70,70) current blocks, same grouping
    bonds: tuple

    @property
    def d1(self) -> int:
        return self.n_max + 1

    @property
    def dim(self) -> int:
        return N_EL * self.d1


def build_joint(geo, p, k, neel: int, *, n_max: int, omega0: float,
                g: float, polarization=None) -> JointModel:
    e = qm.a2_polarization(geo) if polarization is None else np.asarray(polarization, float)
    bonds = qm.hopping_bonds(geo, p)
    groups: dict[float, np.ndarray] = {}
    jgroups: dict[float, np.ndarray] = {}
    for bond in bonds:
        qu = round(float(g * np.dot(e, bond.delta)), 12)
        if qu not in groups:
            groups[qu] = np.zeros((N_EL, N_EL), dtype=complex)
            jgroups[qu] = np.zeros((2, N_EL, N_EL), dtype=complex)
        phase = bond.amplitude * np.exp(1j * np.dot(k, bond.delta))
        F = phase * qm.fermion_bilinear(bond.row, bond.col)
        groups[qu] += F
        jgroups[qu] += -1j * np.asarray(bond.delta)[:, None, None] * F
    keys = np.array(sorted(groups), dtype=float)
    diag = (onsite_determinant_energy(p, neel)[:, None]
            + omega0 * np.arange(n_max + 1, dtype=float)[None, :]).ravel()
    return JointModel(np.asarray(k, float), int(neel), int(n_max), float(omega0),
                      float(g), e, diag, keys,
                      tuple(groups[k_] for k_ in keys),
                      tuple(jgroups[k_] for k_ in keys), bonds)


def peierls_at(q_units: np.ndarray, s: float, n_max: int, Ds=None) -> np.ndarray:
    """P exp(i q s X) P for each group. q s = 0 is exactly I, never skipped hopping."""
    if Ds is not None:
        return Ds
    qs = np.asarray(q_units, dtype=float) * float(s)
    d1 = n_max + 1
    out = np.empty((len(qs), d1, d1), dtype=complex)
    zero = np.abs(qs) <= 1e-15
    if np.any(~zero):
        out[~zero] = projected_peierls_batch(qs[~zero], n_max)
    if np.any(zero):
        out[zero] = np.eye(d1, dtype=complex)
    return out


def apply_h(model: JointModel, v: np.ndarray, s: float, Ds=None) -> np.ndarray:
    """H(s) v. Hopping is always present; s=0 uses D=I, not a dropped kinetic term."""
    d1 = model.d1
    V = v.reshape(N_EL, d1)
    out = model.diag.reshape(N_EL, d1) * V
    if model.q_units.size:
        for F, D in zip(model.f_group, peierls_at(model.q_units, s, model.n_max, Ds)):
            out = out + F @ (V @ D)
    return out.ravel()


def apply_current(model: JointModel, v: np.ndarray, s: float, Ds=None) -> np.ndarray:
    """(J_0 v, J_1 v), same Peierls as H."""
    d1 = model.d1
    V = v.reshape(N_EL, d1)
    out = np.zeros((2, N_EL, d1), dtype=complex)
    if model.q_units.size:
        Ds = peierls_at(model.q_units, s, model.n_max, Ds)
        for J, D in zip(model.j_group, Ds):
            VD = V @ D
            out[0] += J[0] @ VD
            out[1] += J[1] @ VD
    return out.reshape(2, -1)


def dense_h(model: JointModel, s: float) -> np.ndarray:
    d1 = model.d1
    H = np.diag(model.diag).astype(complex)
    if model.q_units.size:
        for F, D in zip(model.f_group, peierls_at(model.q_units, s, model.n_max)):
            H = H + np.kron(F, D)
    return H


def dense_currents(model: JointModel, s: float) -> np.ndarray:
    """J_a = -dH/d kappa_a from the same grouped bonds, at coupling s."""
    d1 = model.d1
    out = np.zeros((2, N_EL * d1, N_EL * d1), dtype=complex)
    if model.q_units.size:
        for J, D in zip(model.j_group, peierls_at(model.q_units, s, model.n_max)):
            out[0] += np.kron(J[0], D)
            out[1] += np.kron(J[1], D)
    return out


# ---------------------------------------------------------------- propagator
def taylor_step(model: JointModel, v: np.ndarray, s: float, dt: float,
                order: int = 16, Ds=None) -> tuple[np.ndarray, float]:
    """exp(-i H(s) dt) v by fixed-order Taylor about the diagonal mean.

    Peierls is computed once per step, not once per Taylor term.
    The returned scalar is max|diag-shift| |dt|; the true ||H dt|| is larger
    by the hopping radius and is reported separately by a residual check.
    """
    Ds = peierls_at(model.q_units, s, model.n_max, Ds)
    shift = float(np.mean(model.diag))
    term = v.astype(complex, copy=True)
    out = term.copy()
    for k in range(1, order + 1):
        term = apply_h(model, term, s, Ds=Ds) - shift * term
        term = (-1j * dt / k) * term
        out = out + term
    out = np.exp(-1j * shift * dt) * out
    last = float(np.linalg.norm(term))
    return out, float(np.abs(model.diag - shift).max() * abs(dt)), last


def spectral_radius(model: JointModel, s: float = 1.0) -> float:
    ev = np.linalg.eigvalsh(dense_h(model, s))
    return float(max(abs(ev[0]), abs(ev[-1])))


def n_steps_for(t_span: float, rho: float, requested: int = 32, hdt_max: float = 0.4) -> int:
    dt_max = hdt_max / max(float(rho), 1e-9)
    need = int(np.ceil(abs(t_span) / dt_max))
    return max(int(requested), max(1, need))


def propagate_ramp(model: JointModel, psi: np.ndarray, t0: float, t1: float,
                   env: Envelope, n_steps: int, order: int = 16,
                   sample_every: int = 0, clock=None
                   ) -> tuple[np.ndarray, dict]:
    """Midpoint Taylor steps over a ramp segment; returns |psi(t1)> and ledger."""
    dt = (t1 - t0) / n_steps
    worst = 0.0
    samples = []
    work = 0.0
    e_prev = None
    for j in range(n_steps):
        if clock is not None:
            clock.check()
        tm = t0 + (j + 0.5) * dt
        s = env.s(tm)
        Ds = peierls_at(model.q_units, s, model.n_max)
        psi, hdt, last = taylor_step(model, psi, s, dt, order=order, Ds=Ds)
        worst = max(worst, hdt)
        if (j + 1) % 32 == 0 or j == n_steps - 1:
            if not np.isfinite(psi).all():
                raise RuntimeError(f"non-finite state at step {j}, dt={dt}, last_term={last}")
            nrm = float(np.linalg.norm(psi))
            if nrm > 2.0 or nrm < 0.5:
                raise RuntimeError(f"norm {nrm} left [0.5,2] at step {j}; Taylor step too large")
        if sample_every and (j % sample_every == 0 or j == n_steps - 1):
            obs = snapshot(model, psi, s, Ds=Ds)
            obs["t"] = t0 + (j + 1) * dt
            obs["s"] = env.s(obs["t"])
            samples.append(obs)
            e_now = obs["energy"]
            if e_prev is not None:
                work += e_now - e_prev
            e_prev = e_now
    return psi, {"steps": n_steps, "dt": dt, "order": order,
                 "max_abs_Hshift_dt": worst, "samples": samples,
                 "booked_Delta_energy_on_samples": work}


# ---------------------------------------------------------------- input states
def product_state(model: JointModel, occupied: np.ndarray, photon_coeff: np.ndarray
                  ) -> np.ndarray:
    """Slater electron state (x) truncated photon input; no renormalisation."""
    el = qm.slater_state(occupied)
    return np.kron(el, np.asarray(photon_coeff, dtype=complex))


def electron_ground_state(geo, p, k, neel: int) -> np.ndarray:
    from q8_m0_model import h8
    energies, vectors = np.linalg.eigh(h8(geo, p, k, neel))
    return vectors[:, :4]


# ---------------------------------------------------------------- windowed source observables
def hann_window_transform(omega, t_width: float):
    """W(Omega) = int_0^T sin^2(pi t/T) exp(i Omega t) dt.

    sin^2 = (1 - cos(2 pi t/T))/2 gives W = I(Om)/2 - I(Om+2pi/T)/4 - I(Om-2pi/T)/4
    with I(a) = int_0^T exp(i a t) dt. Stable at Omega=0 and at the 2pi/T poles.
    """
    om = np.atleast_1d(np.asarray(omega, dtype=float))
    T = float(t_width)
    wp = 2.0 * np.pi / T

    def iex(a):
        a = np.asarray(a, dtype=float)
        out = np.empty(a.shape, dtype=complex)
        small = np.abs(a) < 1e-12
        out[small] = T
        aa = a[~small]
        out[~small] = (np.exp(1j * aa * T) - 1.0) / (1j * aa)
        return out

    w = 0.5 * iex(om) - 0.25 * iex(om + wp) - 0.25 * iex(om - wp)
    if np.ndim(omega) == 0:
        return complex(w.reshape(-1)[0])
    return w


@dataclass
class SourceSpectra:
    """Windowed source-side amplitudes and correlations, one operator order."""

    omega0: float
    t_window: float
    harmonics: np.ndarray
    mean_J: np.ndarray          # (n_h, 2) complex  <Jhat_{q,a}>
    S: np.ndarray               # (n_h, 2, 2) complex  <Jhat_a^dag Jhat_b>
    connected: np.ndarray       # (n_h, 2, 2) complex  S - mean^dag mean
    diagnostics: dict = field(default_factory=dict)

    def intensity(self) -> np.ndarray:
        return np.real(np.einsum("qaa->q", self.S))


def source_spectra(model: JointModel, psi_flat: np.ndarray, currents: np.ndarray,
                   harmonics, t_window: float, s_flat: float = 1.0) -> SourceSpectra:
    """Evaluate <Jhat> and <Jhat^dag Jhat> on a constant-s segment.

    currents and s_flat must belong to the same Hamiltonian that generated
    the evolution on this window. Mixing H(s=1) with a state evolved at s=0
    is not a no-pump background.
    """
    H = dense_h(model, s_flat)
    energies, vecs = np.linalg.eigh(H)
    c = vecs.conj().T @ psi_flat
    M = np.stack([vecs.conj().T @ currents[a] @ vecs for a in range(2)])
    w0 = model.omega0
    harm = np.asarray(harmonics, dtype=float)
    n_h = harm.size
    mean_J = np.zeros((n_h, 2), dtype=complex)
    S = np.zeros((n_h, 2, 2), dtype=complex)
    dE = energies[:, None] - energies[None, :]
    for iq, q in enumerate(harm):
        # <Jhat_{q,a}>
        W_amp = hann_window_transform(q * w0 + dE.ravel(), t_window).reshape(dE.shape)
        for a in range(2):
            mean_J[iq, a] = np.einsum("n,nm,m->", c.conj(), M[a] * W_amp, c)
        # S_{q,ab} = c^dag (M^a U) (M^b V) c with U,V the two window factors
        U = hann_window_transform(dE - q * w0, t_window).reshape(dE.shape)   # (n,m)
        V = hann_window_transform(dE + q * w0, t_window).reshape(dE.shape)   # (m,l)
        for a in range(2):
            left = M[a] * U
            for b in range(2):
                right = M[b] * V
                S[iq, a, b] = np.einsum("n,nm,ml,l->", c.conj(), left, right, c)
    conn = S - np.einsum("qa,qb->qab", mean_J.conj(), mean_J)
    return SourceSpectra(w0, t_window, harm, mean_J, S, conn,
                         diagnostics={"n_eigen": energies.size,
                                      "eig_min": float(energies.min()),
                                      "eig_max": float(energies.max()),
                                      "norm_flat": float(np.linalg.norm(psi_flat)),
                                      "s_flat": float(s_flat)})


# ---------------------------------------------------------------- snapshots / states
def snapshot(model: JointModel, psi: np.ndarray, s: float, Ds=None) -> dict:
    psi = np.asarray(psi, dtype=complex)
    d1 = model.d1
    V = psi.reshape(N_EL, d1)
    nrm = float(np.linalg.norm(psi))
    p_n = np.sum(np.abs(V) ** 2, axis=0).real
    Hv = apply_h(model, psi, s, Ds=Ds)
    Jv = apply_current(model, psi, s, Ds=Ds)
    mean_J = [complex(np.vdot(psi, Jv[a])) for a in range(2)]
    return {
        "norm": nrm,
        "n_e": 4.0,
        "n_ph": float(np.dot(p_n, np.arange(d1))),
        "energy": float(np.vdot(psi, Hv).real),
        "mean_J_re": [float(z.real) for z in mean_J],
        "mean_J_im": [float(z.imag) for z in mean_J],
        "photon_even_mass": float(p_n[0::2].sum()),
        "top_two_mass": float(p_n[-2:].sum()),
        "p_n": p_n.tolist(),
        "s": float(s),
    }


def galerkin_displacement(alpha, n_max: int) -> np.ndarray:
    """<m|D_inf(α)|n>. Galerkin of the infinite-space displacement, not exp(α a†_P - α* a_P)."""
    a = complex(alpha)
    d1 = n_max + 1
    out = np.empty((d1, d1), dtype=complex)
    aa = abs(a) ** 2
    pref = np.exp(-0.5 * aa)
    for m in range(d1):
        for n in range(d1):
            if m >= n:
                out[m, n] = (pref * np.exp(0.5 * (gammaln(n + 1) - gammaln(m + 1)))
                             * (a ** (m - n)) * eval_genlaguerre(n, m - n, aa))
            else:
                out[m, n] = (pref * np.exp(0.5 * (gammaln(m + 1) - gammaln(n + 1)))
                             * ((-a.conjugate()) ** (n - m)) * eval_genlaguerre(m, n - m, aa))
    return out


def displaced_squeezed_coefficients(alpha, r: float, theta: float, n_max: int) -> np.ndarray:
    """Truncated D(α)S(ξ)|0>, unrenormalized. Interface only; this trial does not scan it."""
    sv = qm.squeezed_vacuum_coefficients(r, theta, n_max)
    return galerkin_displacement(alpha, n_max) @ sv


class WallClock:
    def __init__(self, limit_s: float):
        self.t0 = time.perf_counter()
        self.limit = float(limit_s)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def check(self) -> None:
        if self.elapsed() >= self.limit:
            raise TimeoutError(f"wall {self.limit:.1f}s exceeded")


S_FLOOR_PREDECLARED = 1e-16
K_FRAC = [0.173, 0.271]
TRIAL_OMEGA = 0.05
TRIAL_G = 0.02
TRIAL_NMAX = 8
TRIAL_NBAR = 0.25
HARMONICS = np.arange(0, 9, dtype=float)


def _cpack(z) -> dict:
    z = np.asarray(z)
    return {"re": np.real(z).tolist(), "im": np.imag(z).tolist()}


def _spectra_pack(sp: SourceSpectra) -> dict:
    intensity = np.real(np.einsum("qaa->q", sp.S))
    mean_int = np.sum(np.abs(sp.mean_J) ** 2, axis=1)
    return {
        "harmonics": sp.harmonics.tolist(),
        "t_window": sp.t_window,
        "mean_J": _cpack(sp.mean_J),
        "S": _cpack(sp.S),
        "connected": _cpack(sp.connected),
        "intensity_S": intensity.tolist(),
        "intensity_mean_J_sq": mean_int.tolist(),
        "S_floor_predeclared": S_FLOOR_PREDECLARED,
        "note": "intensity_S is source-side <Jhat^dag Jhat>; intensity_mean_J_sq is |<Jhat>|^2; do not call the gap quantum advantage",
        "diagnostics": sp.diagnostics,
    }


def envelope_table(env: Envelope, n: int = 401) -> dict:
    t = np.linspace(0.0, env.t_total, n)
    return {
        "t": t.tolist(),
        "s": [env.s(float(x)) for x in t],
        "ds": [env.ds(float(x)) for x in t],
        "t_ramp": env.t_ramp,
        "t_flat": env.t_flat,
        "t_total": env.t_total,
        "omega0": env.omega0,
        "ramp_cycles": env.ramp_cycles,
        "flat_cycles": env.flat_cycles,
        "protocol": "controlled_pulse_coupling_not_free_space_BSV",
    }


def build_trial_model(n_max: int, g: float, neel: int = 1) -> tuple:
    geo = build_geometry()
    k = geo.k_cart(K_FRAC)
    model = build_joint(geo, R0, k, neel, n_max=n_max, omega0=TRIAL_OMEGA, g=g)
    occupied = electron_ground_state(geo, R0, k, neel)
    return geo, k, model, occupied


def photon_input(kind: str, n_max: int, nbar: float):
    if kind == "vacuum":
        c = np.zeros(n_max + 1, dtype=complex)
        c[0] = 1.0
        meta = {"kind": "vacuum", "nbar_target": 0.0}
    elif kind == "coherent":
        alpha = complex(np.sqrt(nbar))
        c = qm.coherent_coefficients(alpha, n_max)
        meta = {"kind": "coherent", "alpha": {"re": float(alpha.real), "im": float(alpha.imag)},
                "nbar_target": nbar}
    elif kind == "squeezed_vacuum":
        r = float(np.arcsinh(np.sqrt(nbar)))
        c = qm.squeezed_vacuum_coefficients(r, 0.0, n_max)
        meta = {"kind": "squeezed_vacuum", "r": r, "theta": 0.0, "nbar_target": nbar,
                "role": "method_and_photon_parity_test_not_magnetic_scan"}
    else:
        raise ValueError(kind)
    meta["diagnostics"] = qm.state_diagnostics(c)
    return c, meta


def exact_constant_h(model: JointModel, psi: np.ndarray, t: float, s: float) -> np.ndarray:
    H = dense_h(model, s)
    ev, vecs = np.linalg.eigh(H)
    c = vecs.conj().T @ psi
    return vecs @ (np.exp(-1j * ev * t) * c)


def timestep_check(model: JointModel, psi: np.ndarray, env: Envelope,
                   rho: float, order: int, clock: WallClock, t_chk: float
                   ) -> tuple:
    """Short prefix at n vs 2n steps. Full first ramp is done once at the fine dt."""
    n_coarse = n_steps_for(t_chk, rho, requested=32, hdt_max=0.4)
    psi_a, led_a = propagate_ramp(model, psi, 0.0, t_chk, env, n_coarse, order, clock=clock)
    psi_b, led_b = propagate_ramp(model, psi, 0.0, t_chk, env, 2 * n_coarse, order, clock=clock)
    ov = complex(np.vdot(psi_b, psi_a))
    na, nb = float(np.linalg.norm(psi_a)), float(np.linalg.norm(psi_b))
    return {
        "t_check": t_chk,
        "n_steps": n_coarse,
        "n_steps_fine": 2 * n_coarse,
        "dt_coarse": led_a["dt"],
        "dt_fine": led_b["dt"],
        "overlap_abs": abs(ov) / (na * nb + 1e-30),
        "norm_coarse": na,
        "norm_fine": nb,
        "max_abs_Hshift_dt_coarse": led_a["max_abs_Hshift_dt"],
        "max_abs_Hshift_dt_fine": led_b["max_abs_Hshift_dt"],
        "rho": rho,
    }


def truncation_check(kind: str, nbar: float, env: Envelope, t_chk: float,
                     order: int, clock: WallClock) -> dict:
    """Same short prefix at n_max=6 vs 8. Static 7/7 is not this check."""
    out = {}
    for n_max in (6, 8):
        clock.check()
        _, _, model, occupied = build_trial_model(n_max, TRIAL_G)
        coeff, meta = photon_input(kind, n_max, nbar)
        psi = product_state(model, occupied, coeff)
        rho = spectral_radius(model, 1.0)
        n_steps = n_steps_for(t_chk, rho)
        psi1, led = propagate_ramp(model, psi, 0.0, t_chk, env, n_steps, order, clock=clock)
        snap = snapshot(model, psi1, env.s(t_chk))
        out[str(n_max)] = {"input": meta, "after_prefix": {k: snap[k] for k in
                           ("norm", "n_ph", "energy", "mean_J_re", "photon_even_mass", "top_two_mass")},
                           "max_abs_Hshift_dt": led["max_abs_Hshift_dt"], "rho": rho, "n_steps": n_steps}
    a, b = out["6"]["after_prefix"], out["8"]["after_prefix"]
    out["delta_n_ph"] = abs(a["n_ph"] - b["n_ph"])
    out["delta_mean_J"] = [abs(a["mean_J_re"][i] - b["mean_J_re"][i]) for i in range(2)]
    out["t_check"] = t_chk
    out["note"] = "P exp(iqX) P is kept; not replaced by exp(iq PXP)"
    return out


def pack_samples(samples: list) -> dict:
    if not samples:
        return {}
    keys = ("t", "s", "norm", "n_e", "n_ph", "energy", "photon_even_mass", "top_two_mass")
    tab = {k: [s[k] for s in samples] for k in keys}
    tab["mean_J_re"] = [s["mean_J_re"] for s in samples]
    tab["mean_J_im"] = [s["mean_J_im"] for s in samples]
    return tab


def run_zero_job(clock: WallClock, env: Envelope) -> dict:
    """g=0, H time-independent. Exact spectral evolution; Taylor only on a short prefix."""
    _, _, model, occupied = build_trial_model(TRIAL_NMAX, g=0.0)
    coeff, meta = photon_input("coherent", TRIAL_NMAX, TRIAL_NBAR)
    psi0 = product_state(model, occupied, coeff)
    s0 = snapshot(model, psi0, 0.0)
    t_end = env.t_total
    rho = spectral_radius(model, 0.0)
    psi_exact = exact_constant_h(model, psi0, t_end, 0.0)
    se = snapshot(model, psi_exact, 0.0)
    t_chk = min(env.period, 16.0)
    n_steps = n_steps_for(t_chk, rho)
    psi_tay, led = propagate_ramp(model, psi0, 0.0, t_chk, env, n_steps, order=16,
                                  sample_every=max(1, n_steps // 8), clock=clock)
    psi_ref = exact_constant_h(model, psi0, t_chk, 0.0)
    st = snapshot(model, psi_tay, 0.0)
    sr = snapshot(model, psi_ref, 0.0)
    ov = abs(complex(np.vdot(psi_ref, psi_tay))) / (sr["norm"] * st["norm"] + 1e-30)
    times = np.linspace(0.0, t_end, 17)
    series = []
    for t in times:
        snap = snapshot(model, exact_constant_h(model, psi0, float(t), 0.0), 0.0)
        snap["t"] = float(t)
        series.append(snap)
    return {
        "job": "zero_coupling_free",
        "g": 0.0,
        "input": meta,
        "dim": model.dim,
        "n_group": int(model.q_units.size),
        "rho": rho,
        "t_end": t_end,
        "initial": {k: s0[k] for k in ("norm", "n_ph", "energy", "mean_J_re", "photon_even_mass")},
        "exact_final": {k: se[k] for k in ("norm", "n_ph", "energy", "mean_J_re", "photon_even_mass")},
        "delta_n_ph_exact": abs(se["n_ph"] - s0["n_ph"]),
        "delta_energy_exact": abs(se["energy"] - s0["energy"]),
        "delta_norm_exact": abs(se["norm"] - s0["norm"]),
        "taylor_prefix": {
            "t_check": t_chk,
            "overlap_vs_exact": ov,
            "n_steps": n_steps,
            "dt": led["dt"],
            "max_abs_Hshift_dt": led["max_abs_Hshift_dt"],
            "taylor": {k: st[k] for k in ("norm", "n_ph", "energy")},
            "exact": {k: sr[k] for k in ("norm", "n_ph", "energy")},
        },
        "time_series_exact": pack_samples(series),
        "wall_seconds": clock.elapsed(),
        "claim": "conservation and propagator check only",
    }


def run_driven_job(kind: str, clock: WallClock, env: Envelope, n_steps_ramp: int,
                   do_trunc: bool) -> dict:
    _, _, model, occupied = build_trial_model(TRIAL_NMAX, g=TRIAL_G)
    coeff, meta = photon_input(kind, TRIAL_NMAX, TRIAL_NBAR)
    psi0 = product_state(model, occupied, coeff)
    s0 = snapshot(model, psi0, 0.0)
    rho = spectral_radius(model, 1.0)
    t_chk = min(0.25 * env.t_ramp, env.period / 4.0)
    dt_rep = timestep_check(model, psi0, env, rho, 16, clock, t_chk)
    n_up = n_steps_for(env.t_ramp, rho, requested=n_steps_ramp)
    psi_flat, led_up = propagate_ramp(model, psi0, 0.0, env.t_ramp, env, n_up, 16,
                                      sample_every=max(1, n_up // 8), clock=clock)
    snap_flat = snapshot(model, psi_flat, 1.0)
    currents = dense_currents(model, 1.0)
    driven = source_spectra(model, psi_flat, currents, HARMONICS, env.t_flat, s_flat=1.0)
    psi_bg = exact_constant_h(model, psi0, env.t_ramp, 0.0)
    bg = source_spectra(model, psi_bg, dense_currents(model, 0.0), HARMONICS, env.t_flat,
                        s_flat=0.0)
    n_dn = n_steps_for(env.t_ramp, rho, requested=n_steps_ramp)
    psi_end, led_dn = propagate_ramp(model, psi_flat, env.t_ramp + env.t_flat, env.t_total,
                                     env, n_dn, 16, sample_every=max(1, n_dn // 8),
                                     clock=clock)
    snap_end = snapshot(model, psi_end, 0.0)
    trunc = truncation_check(kind, TRIAL_NBAR, env, t_chk, 16, clock) if do_trunc else None
    work = snap_end["energy"] - s0["energy"]
    return {
        "job": kind,
        "g": TRIAL_G,
        "nbar_target": TRIAL_NBAR,
        "rho": rho,
        "input": meta,
        "dim": model.dim,
        "n_group": int(model.q_units.size),
        "initial": {k: s0[k] for k in ("norm", "n_ph", "energy", "mean_J_re", "photon_even_mass", "top_two_mass")},
        "after_ramp": {k: snap_flat[k] for k in ("norm", "n_ph", "energy", "mean_J_re", "photon_even_mass", "top_two_mass")},
        "after_fall": {k: snap_end[k] for k in ("norm", "n_ph", "energy", "mean_J_re", "photon_even_mass", "top_two_mass")},
        "external_work_Delta_energy": work,
        "work_note": "Delta <H> equals work by the explicit s(t) protocol, not a free-space pulse energy",
        "timestep": dt_rep,
        "ramp_ledger": {k: led_up[k] for k in ("steps", "dt", "order", "max_abs_Hshift_dt")},
        "fall_ledger": {k: led_dn[k] for k in ("steps", "dt", "order", "max_abs_Hshift_dt")},
        "time_series_ramp": pack_samples(led_up["samples"]),
        "time_series_fall": pack_samples(led_dn["samples"]),
        "driven_flat_spectra": _spectra_pack(driven),
        "nopump_flat_spectra": _spectra_pack(bg),
        "truncation": trunc,
        "wall_seconds": clock.elapsed(),
        "not_magnetic_HHG": True,
        "not_free_space_BSV": True,
    }


def next_stage_cost() -> dict:
    n_max = TRIAL_NMAX
    d1 = n_max + 1
    two_k = (N_EL ** 2) * d1
    bytes_dense = 16 * two_k * two_k
    return {
        "single_k_dim": N_EL * d1,
        "shared_mode_k_minus_k_dim": two_k,
        "dense_complex_matrix_bytes": bytes_dense,
        "dense_complex_matrix_GB": bytes_dense / 1024 ** 3,
        "recommendation": "matrix-free or sparse; do not assemble 44100 dense H",
        "not_authorized_this_round": ["k,-k dynamics", "full BZ", "G1", "H2", "U/V",
                                      "cross-harmonic entanglement", "AI optimization",
                                      "production BSV", "displaced-squeezed physics scan"],
    }


def run_trial(jobs, n_steps_ramp: int, wall_per_job: float, out_path: Path) -> dict:
    env = Envelope(TRIAL_OMEGA, ramp_cycles=1.0, flat_cycles=2.0)
    report = {
        "record_date": "2026-10-26",
        "model_id": MODEL_ID,
        "claim_level": "method_benchmark_not_magnetic_HHG",
        "not_free_space_BSV_pulse": True,
        "static_audit_accepted": True,
        "hamiltonian": "frozen_h8_not_reselected",
        "k_fractional": K_FRAC,
        "n_max": TRIAL_NMAX,
        "omega0": TRIAL_OMEGA,
        "g": TRIAL_G,
        "nbar": TRIAL_NBAR,
        "S_floor_predeclared": S_FLOOR_PREDECLARED,
        "relative_delta_definition": "|S_ex-S_sc|/max(S_ex,S_sc,S_floor); 1% is a strong-harmonic target, not a theorem",
        "envelope": envelope_table(env),
        "displaced_squeezed_interface": {
            "available": True,
            "scanned": False,
            "formula": "D(alpha) S(xi) |0>, Galerkin D, unrenormalized",
        },
        "jobs": {},
        "next_stage_cost": next_stage_cost(),
        "files": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                  for p in (qm.__file__, __file__)},
        "stopped": True,
    }
    for i, job in enumerate(jobs):
        clock = WallClock(wall_per_job)
        try:
            if job == "zero":
                report["jobs"]["zero"] = run_zero_job(clock, env)
            elif job == "coherent":
                report["jobs"]["coherent"] = run_driven_job("coherent", clock, env, n_steps_ramp,
                                                            do_trunc=True)
            elif job == "sv":
                report["jobs"]["squeezed_vacuum"] = run_driven_job("squeezed_vacuum", clock, env,
                                                                   n_steps_ramp, do_trunc=False)
            else:
                raise ValueError(job)
            report["jobs"][job if job != "sv" else "squeezed_vacuum"]["status"] = "completed"
        except TimeoutError as exc:
            key = {"zero": "zero", "coherent": "coherent", "sv": "squeezed_vacuum"}[job]
            report["jobs"][key] = {"status": "wall_timeout", "error": str(exc),
                                   "wall_seconds": clock.elapsed()}
        except Exception as exc:
            key = {"zero": "zero", "coherent": "coherent", "sv": "squeezed_vacuum"}[job]
            report["jobs"][key] = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                                   "wall_seconds": clock.elapsed()}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", action="append", choices=["zero", "coherent", "sv"],
                    help="repeatable; default is zero then coherent then sv")
    ap.add_argument("--n-steps-ramp", type=int, default=240)
    ap.add_argument("--wall-seconds", type=float, default=300.0)
    ap.add_argument("--report", type=Path,
                    default=Path("results/q_line/qinput_a_dynamics_trial_20261026/trial.json"))
    args = ap.parse_args(argv)
    jobs = args.job or ["zero", "coherent", "sv"]
    if args.report.exists():
        print(f"report already exists: {args.report}", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    report = run_trial(jobs, args.n_steps_ramp, args.wall_seconds, args.report)
    report["wall_seconds_total"] = time.perf_counter() - t0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in ("model_id", "claim_level", "wall_seconds_total")
                      if k in report}, ensure_ascii=False))
    for name, job in report["jobs"].items():
        print(name, job.get("status"), "wall", job.get("wall_seconds"))
    return 0


if __name__ == "__main__":
    sys.exit(main())


