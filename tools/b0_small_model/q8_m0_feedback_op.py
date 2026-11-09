#!/usr/bin/env python3
"""Open-loop feedback operator L = G_self R^A on frozen P1.

Born is (1-L)^{-1} ≈ 1+L. The control parameter is σ_max(L), not a
classical test-seed ratio κ_self/κ_P1 (that ratio scales with ε).

L maps a real δA(t) to the one-pass self-field G_self[R^A[δA]].
The Volterra closed loop is not applied. No T2, no DC cut, no production V.
"""
from __future__ import annotations

import numpy as np

from q8_m0_adapt_bz import left_rect_integral
from q8_m0_response import contact_dmu, contact_dmu_perk, prefix_state_dmu, prefix_state_dmu_perk
from q8_m0_volterra import volterra_adjoint

OP_ID = "feedback_op_L_Gself_RA_20260922"
INNER = "left_riemann_dt_sum_cartesian"


def inner_dt(a, b, dt: float) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(dt) * float(np.sum(a * b))


def norm_dt(a, dt: float) -> float:
    return float(np.sqrt(max(inner_dt(a, a, dt), 0.0)))


def apply_G(i_t, dt: float, gamma: float) -> np.ndarray:
    i_t = np.asarray(i_t, dtype=float)
    return left_rect_integral(i_t[:, None, :], dt, gamma)[:, 0, :]


def apply_Gadj(g_t, dt: float, gamma: float) -> np.ndarray:
    """Adjoint of left-rect G in the ⟨u,v⟩=dt Σ u·v inner product.

    (G† g)[m] = γ dt Σ_{n>m} g[n].
    """
    g = np.asarray(g_t, dtype=float)
    out = np.zeros_like(g)
    acc = np.zeros(g.shape[1:], dtype=float)
    coef = float(gamma) * float(dt)
    for n in range(g.shape[0] - 1, -1, -1):
        out[n] = coef * acc
        acc = acc + g[n]
    return out


def apply_R(x, w, mean_d, f, dt: float) -> np.ndarray:
    return predict_total(x, w, mean_d, f, dt)


def predict_total(x, w, mean_d, f, dt: float) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    state = prefix_state_dmu(x, w, f, dt)
    contact = contact_dmu(mean_d, f)
    return state + contact


def apply_Radj(x, w, mean_d, phi, dt: float) -> np.ndarray:
    """R† in the dt-sum inner product, via packed Euclidean reverse."""
    phi = np.asarray(phi, dtype=float)
    packed = volterra_adjoint(x, w, mean_d, dt, 0.0, float(dt) * phi)
    return packed["g_Ain"] / float(dt)


def apply_L(x, w, mean_d, f, dt: float, gamma: float) -> np.ndarray:
    """L f = G_self R^A[f]. Open loop."""
    return apply_G(apply_R(x, w, mean_d, f, dt), dt, gamma)


def apply_Ladj(x, w, mean_d, g, dt: float, gamma: float) -> np.ndarray:
    """L† = R† G_self†."""
    return apply_Radj(x, w, mean_d, apply_Gadj(g, dt, gamma), dt)


def apply_L_perk(x, d_perk, f, dt: float, gamma: float) -> np.ndarray:
    """Per-k open-loop self-field. L = Σ_k w_k L_k."""
    f = np.asarray(f, dtype=float)
    i_k = prefix_state_dmu_perk(x, f, dt) + contact_dmu_perk(d_perk, f)
    return left_rect_integral(i_k, dt, gamma)


def rayleigh_perk(u, v_k, dt: float) -> np.ndarray:
    """α_k = ⟨u, (L_k v)⟩. v_k is L_k v, shape (nt, nk, 2)."""
    u = np.asarray(u, dtype=float)
    vk = np.asarray(v_k, dtype=float)
    return float(dt) * np.einsum("ta,tka->k", u, vk)


def _orthonormalize(cols, dt: float) -> np.ndarray:
    """cols: (nt, 2, p) -> same, dt-orthonormal, drop null columns."""
    nt, _, p = cols.shape
    out = []
    for j in range(p):
        v = cols[:, :, j].copy()
        for u in out:
            v = v - inner_dt(u, v, dt) * u
        nrm = norm_dt(v, dt)
        if nrm < 1e-14:
            continue
        out.append(v / nrm)
    if not out:
        raise RuntimeError("orthonormalize produced no columns")
    stacked = np.zeros((nt, 2, len(out)), dtype=float)
    for j, v in enumerate(out):
        stacked[:, :, j] = v
    return stacked


def _rotate(cols, evecs) -> np.ndarray:
    return np.einsum("abp,pq->abq", cols, np.asarray(evecs, dtype=float))


def adjoint_residual(x, w, mean_d, f, g, dt: float, gamma: float) -> float:
    """|<g, L f> - <L† g, f>| / max(|<g,Lf>|, floor)."""
    lf = apply_L(x, w, mean_d, f, dt, gamma)
    left = inner_dt(g, lf, dt)
    right = inner_dt(apply_Ladj(x, w, mean_d, g, dt, gamma), f, dt)
    return float(abs(left - right) / max(abs(left), 1e-30))


def scale_residual(x, w, mean_d, f, dt: float, gamma: float, factor: float = 2.0) -> float:
    lf = apply_L(x, w, mean_d, f, dt, gamma)
    lf2 = apply_L(x, w, mean_d, float(factor) * np.asarray(f, dtype=float), dt, gamma)
    num = float(np.max(np.abs(lf2 - float(factor) * lf)))
    den = max(float(np.max(np.abs(lf))), 1e-30)
    return num / den


def leading_singular(x, w, mean_d, dt, gamma, f_probe=None, *, p: int = 3, n_iter: int = 5, rng=None):
    """Largest singular values of L by SVD subspace iteration in a p-plane.

    One probe slot is the H2-shaped f_probe if given. Remaining slots random.
    Each iteration: W = L V, Ritz on W†W, then V ← L† U.
    σ_max is independent of any classical test-seed amplitude.
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    x = np.asarray(x)
    nt = int(x.shape[0])
    cols = np.zeros((nt, 2, p), dtype=float)
    start = 0
    if f_probe is not None:
        fp = np.asarray(f_probe, dtype=float)
        nrm = norm_dt(fp, dt)
        if nrm > 0.0:
            cols[:, :, 0] = fp / nrm
            start = 1
    for j in range(start, p):
        cols[:, :, j] = rng.normal(size=(nt, 2))
    V = _orthonormalize(cols, dt)
    hist = []
    sigs = np.zeros(int(V.shape[2]), dtype=float)
    W = None
    for it in range(n_iter):
        pcur = int(V.shape[2])
        W = np.zeros((nt, 2, pcur), dtype=float)
        for j in range(pcur):
            W[:, :, j] = apply_L(x, w, mean_d, V[:, :, j], dt, gamma)
        gram = np.zeros((pcur, pcur), dtype=float)
        for i in range(pcur):
            for j in range(i, pcur):
                gij = inner_dt(W[:, :, i], W[:, :, j], dt)
                gram[i, j] = gij
                gram[j, i] = gij
        evals, evecs = np.linalg.eigh(0.5 * (gram + gram.T))
        order = np.argsort(evals)[::-1]
        sigs = np.sqrt(np.maximum(evals[order], 0.0))
        hist.append(float(sigs[0]))
        V = _rotate(V, evecs[:, order])
        W = _rotate(W, evecs[:, order])
        V = _orthonormalize(V, dt)
        if it >= 2 and abs(hist[-1] - hist[-2]) < 1e-3 * max(hist[-1], 1e-12):
            break
        U = np.zeros_like(W)
        for j in range(W.shape[2]):
            nrm = norm_dt(W[:, :, j], dt)
            U[:, :, j] = W[:, :, j] / nrm if nrm > 0.0 else W[:, :, j]
        Vnext = np.zeros_like(U)
        for j in range(U.shape[2]):
            Vnext[:, :, j] = apply_Ladj(x, w, mean_d, U[:, :, j], dt, gamma)
        V = _orthonormalize(Vnext, dt)
    v0 = V[:, :, 0]
    u_raw = apply_L(x, w, mean_d, v0, dt, gamma)
    s0 = norm_dt(u_raw, dt)
    u0 = u_raw / s0 if s0 > 0.0 else u_raw
    trial = [float(s) for s in sigs]
    if trial:
        trial[0] = float(s0)
    gain_probe = None
    rayleigh_probe = None
    if f_probe is not None:
        nrm = norm_dt(f_probe, dt)
        if nrm > 0.0:
            fn = np.asarray(f_probe, dtype=float) / nrm
            lf = apply_L(x, w, mean_d, fn, dt, gamma)
            gain_probe = norm_dt(lf, dt)
            rayleigh_probe = inner_dt(fn, lf, dt)
    return {
        "sigma_max": float(s0),
        "sigma_trial": trial,
        "n_iter": int(len(hist)),
        "sigma_hist": [float(s) for s in hist],
        "gain_h2_dir": None if gain_probe is None else float(gain_probe),
        "rayleigh_h2": None if rayleigh_probe is None else float(rayleigh_probe),
        "sigma_over_h2": (
            None if gain_probe in (None, 0.0)
            else float(s0 / max(float(gain_probe), 1e-30))
        ),
        "v_right": v0,
        "u_left": u0,
        "inner": INNER,
        "op_id": OP_ID,
    }


def classify_sigma(sigma: float) -> str:
    s = float(sigma)
    if s < 0.08:
        return "ll1_born_has_small_control_parameter"
    if s < 0.35:
        return "around_0p2_few_percent_second_pass"
    if s < 0.9:
        return "O1_but_below_unity"
    return "gtrsim_1_hidden_strong_direction"
