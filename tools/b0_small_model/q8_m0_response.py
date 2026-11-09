#!/usr/bin/env python3
"""Electronic response slot on the strong classical pump. Not chi_rad.

R^A acts on a test δA. chi_rad maps a physical current onto a free-space mode.
This module does not evaluate photon numbers, g2, or a quantum pump.

Candidate implementation (prefix integrals, D) is an engineering identity check
on stored X(t). Filling the quantum I/O approximation remains a human node.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from q8_m0_model import (
    _interlayer_block,
    _layer_block,
    _phase,
    assemble,
    current_ops,
)

RESPONSE_ID = "R_A_electronic_on_strong_classical_pump"
RESPONSE_STATUS = "unset_awaiting_theory"


@dataclass(frozen=True)
class ElectronicResponseRA:
    ident: str = RESPONSE_ID
    status: str = RESPONSE_STATUS
    form: str = "not_chosen_for_quantum_output"
    convention: str = "I = -dH/dA, e=hbar=1, kappa = k+A"
    state_piece: str = "(i/hbar) theta(t-t') <[I_a^0(t), I_b^0(t')]>"
    contact_piece: str = "-<D_ab^0(t)> delta(t-t'), D=d2H/dA_a dA_b"
    vs_external_R_std: str = "R^A = -R^std - <D> delta  (sign from this I definition)"
    n_cell: str = "unset_not_equal_to_nk"
    gaussian_closure: str = "not_chosen"
    quantum_operator_replacement: str = (
        "replacing D and commutators by numerical expectations is a reduction; "
        "not an exact operator identity"
    )
    note: str = (
        "Interface plus identity checks on stored X. Do not treat a prefix "
        "kernel as approved quantum I/O. chi_rad is a different slot."
    )


class ResponseNotChosenError(RuntimeError):
    """Raised if code tries to close the quantum output with R^A."""


def response_spec() -> dict:
    rec = asdict(ElectronicResponseRA())
    rec["slot_name"] = "R_A"
    rec["not_chi_rad"] = True
    rec["photon_output"] = "not_run"
    rec["g2"] = "not_run"
    rec["ready_to_evaluate_photons"] = False
    rec["full_Y_does_not_invert_to_arbitrary_causal_R"] = True
    return rec


def apply_response_to_output(*_a, **_k):
    raise ResponseNotChosenError(
        "R^A is not approved as a quantum-output closure. "
        "Prefix/D checks on stored X are identity tests only."
    )


def d2h8(geo, p, k2, neel: int) -> np.ndarray:
    """∂²H/∂k_a∂k_b, shape (..., 2, 2, 8, 8). Exchange does not contribute."""
    k2 = np.asarray(k2, dtype=float)
    shp = k2.shape[:-1]
    d2f = np.zeros(shp + (2, 2), dtype=complex)
    d2g = np.zeros(shp + (2, 2), dtype=float)
    d2t0 = np.zeros(shp + (2, 2, 2, 2), dtype=complex)
    for delta in geo.nn_deltas_cart:
        ph = _phase(k2, delta)
        for a in range(2):
            for b in range(2):
                d2f[..., a, b] += (1j * delta[a]) * (1j * delta[b]) * ph
    for nu, bvec in geo.nnn_terms:
        arg = k2[..., 0] * bvec[0] + k2[..., 1] * bvec[1]
        s = np.sin(arg)
        for a in range(2):
            for b in range(2):
                d2g[..., a, b] += -2.0 * nu * bvec[a] * bvec[b] * s
    idx = {"A": 0, "B": 1}
    amp = {"u": p.u, "v": p.v}
    for bond in geo.interlayer_bonds:
        i, j = idx[bond["from"][1]], idx[bond["to"][1]]
        ph = amp[bond["amplitude"]] * _phase(k2, bond["delta_cart"])
        dlt = bond["delta_cart"]
        for a in range(2):
            for b in range(2):
                d2t0[..., a, b, i, j] += (1j * dlt[a]) * (1j * dlt[b]) * ph
    out = np.zeros(shp + (2, 2, 8, 8), dtype=complex)
    for a in range(2):
        for b in range(2):
            ht = _layer_block(d2f[..., a, b], d2g[..., a, b], 0.0, p.t_h, p.lam_so)
            out[..., a, b, :, :] = assemble(ht, ht, _interlayer_block(d2t0[..., a, b, :, :]))
    del neel
    return out


def d_ops(geo, p, k2, neel: int) -> np.ndarray:
    """D_ab = ∂²H/∂A_a∂A_b = ∂²H/∂k_a∂k_b at e = ħ = 1."""
    return d2h8(geo, p, k2, neel)


def d_ops_finite_difference(geo, p, k2, neel: int, *, eps: float = 1e-4) -> np.ndarray:
    """D_ab = -∂j_a/∂k_b from centered differences of current_ops."""
    k2 = np.asarray(k2, dtype=float)
    out = np.zeros(k2.shape[:-1] + (2, 2, 8, 8), dtype=complex)
    eye = np.zeros(k2.shape[:-1] + (2,), dtype=float)
    for b in range(2):
        shift = eye.copy()
        shift[..., b] = eps
        jp = current_ops(geo, p, k2 + shift, neel)
        jm = current_ops(geo, p, k2 - shift, neel)
        out[..., :, b, :, :] = -(jp - jm) / (2.0 * eps)
    return out


def mean_d_from_bases(d_ab: np.ndarray, vv: np.ndarray) -> np.ndarray:
    """Per-k occupied expectation ⟨D_ab⟩, shape (nk, 2, 2)."""
    return np.real(np.einsum("kiv,kabij,kjv->kab", vv.conj(), d_ab, vv))


def c_greater_from_x(xa: np.ndarray, xb: np.ndarray, w: np.ndarray) -> np.ndarray:
    """c^>_ab = Σ_k w_k Tr[X_a† X_b], xa/xb shape (nk, 2, 4, 4)."""
    return np.einsum("k,kacv,kbcv->ab", w, np.conj(xa), xb)


def commutator_from_c_greater(c_ab: np.ndarray) -> np.ndarray:
    """⟨[I_a(t), I_b(t')]⟩ per cell = c_ab(t,t') - c_ba(t',t) = c - c^*."""
    return c_ab - np.conj(c_ab)


def prefix_state_dmu_perk(x: np.ndarray, h: np.ndarray, dt: float) -> np.ndarray:
    """Per-k causal state piece. Shape (nt, nk, 2). Weighted sum recovers prefix_state_dmu."""
    x = np.asarray(x)
    h = np.asarray(h, dtype=float)
    nt, nk = x.shape[0], x.shape[1]
    z = np.zeros((nk, 4, 4), dtype=complex)
    dmu = np.zeros((nt, nk, 2), dtype=float)
    for n in range(nt):
        tr = np.einsum("kacv,kcv->ka", np.conj(x[n]), z)
        dmu[n] = -2.0 * np.imag(tr)
        z += dt * np.einsum("a,kacv->kcv", h[n], x[n])
    return dmu


def prefix_state_dmu(x: np.ndarray, w: np.ndarray, h: np.ndarray, dt: float) -> np.ndarray:
    """Causal state piece δμ_a = ∫ R_state h, left-Riemann on past times.

    x: (nt, nk, 2, 4, 4), h: (nt, 2) real, w: (nk,). ħ = 1.
    Z_k(t) = Σ_b ∫^{t-} dt' X_{b,k}(t') h_b(t').
    δμ_a(t) = -(2/ħ) Im Σ_k w_k Tr[X_a†(t) Z_k(t)].
    Does not include -<D> h. Complex h must be split into real/imag parts.
    """
    w = np.asarray(w, dtype=float)
    return np.einsum("k,tka->ta", w, prefix_state_dmu_perk(x, h, dt))


def two_time_state_dmu(x: np.ndarray, w: np.ndarray, h: np.ndarray, dt: float) -> np.ndarray:
    """Same state piece from explicit θ(t-t') two-time sums (small nt)."""
    x = np.asarray(x)
    h = np.asarray(h, dtype=float)
    w = np.asarray(w, dtype=float)
    nt = x.shape[0]
    dmu = np.zeros((nt, 2), dtype=float)
    for n in range(nt):
        z = np.zeros((x.shape[1], 4, 4), dtype=complex)
        for m in range(n):
            z += dt * np.einsum("a,kacv->kcv", h[m], x[m])
        tr = np.einsum("kacv,kcv->ka", np.conj(x[n]), z)
        dmu[n] = -2.0 * np.imag(w @ tr)
    return dmu


def contact_dmu_perk(d_perk: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Per-k -<D_ab(k,t)> h_b. d_perk (nt, nk, 2, 2), h (nt, 2). Shape (nt, nk, 2)."""
    h = np.asarray(h, dtype=float)
    d_perk = np.asarray(d_perk, dtype=float)
    return -np.einsum("tkab,tb->tka", d_perk, h)


def contact_dmu(mean_d: np.ndarray, h: np.ndarray) -> np.ndarray:
    """-<D_ab> h_b, mean_d (2,2) or (nt,2,2), h (nt,2)."""
    h = np.asarray(h, dtype=float)
    md = np.asarray(mean_d, dtype=float)
    if md.ndim == 2:
        return -np.einsum("ab,tb->ta", md, h)
    return -np.einsum("tab,tb->ta", md, h)


def predict_delta_mu(x, w, h, dt, mean_d_t) -> dict:
    """δ⟨I⟩ = state prefix + contact. h is the actual δA(t), shape (nt, 2)."""
    state = prefix_state_dmu(x, w, h, dt)
    contact = contact_dmu(mean_d_t, h)
    return {
        "state": state,
        "contact": contact,
        "total": state + contact,
    }


def test_waveform_sin2_dir(times, tp: float, ehat) -> np.ndarray:
    """h(0)=h(Tp)=0. Shape (nt, 2), peak |h|=1 along ehat."""
    t = np.asarray(times, dtype=float)
    ehat = np.asarray(ehat, dtype=float)
    ehat = ehat / np.linalg.norm(ehat)
    u = t / tp
    w = np.sin(np.pi * u) ** 2
    w[(u < 0.0) | (u > 1.0)] = 0.0
    return w[:, None] * ehat[None, :]
