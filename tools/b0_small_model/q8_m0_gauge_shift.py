#!/usr/bin/env python3
"""Covariant k-shift gauge test: H, Π0 and k labels move together.

A constant δκ is a pure relabeling only if the initial projector is
Π0(k+δκ) and the drive uses H(k+δκ+κ_cl). Then J_δκ(t)−J_0(t) is a
BZ-quadrature error, not a physical current.

The older test (fixed Π0(k) plus a sudden constant A) is not a gauge
transform: it injects a kick. It is kept only as a labeled contrast.
"""
from __future__ import annotations

import numpy as np

from q8_m0_cjj import propagate_cjj
from q8_m0_classical import k_grid
from q8_m0_feedback_op import norm_dt

GAUGE_ID = "covariant_kshift_H_and_Pi0_20260923"
GENERIC_DK_FRAC = np.array([0.07, 0.03], dtype=float)


def shift_kf(geo, kf, *, delta_frac=None, delta_cart=None) -> np.ndarray:
    kf = np.asarray(kf, dtype=float)
    if delta_frac is not None:
        return np.mod(kf + np.asarray(delta_frac, dtype=float), 1.0)
    if delta_cart is None:
        raise ValueError("need delta_frac or delta_cart")
    kc = geo.k_cart(kf) + np.asarray(delta_cart, dtype=float)
    return np.mod(geo.k_frac(kc), 1.0)


def current_rel_l2(j_a, j_b, dt: float) -> float:
    ja = np.asarray(j_a, dtype=float)
    jb = np.asarray(j_b, dtype=float)
    return norm_dt(ja - jb, dt) / max(norm_dt(ja, dt), 1e-30)


def compare_currents(j0, jp, jm, dt: float) -> dict:
    j0 = np.asarray(j0, dtype=float)
    jp = np.asarray(jp, dtype=float)
    jm = np.asarray(jm, dtype=float)
    even = 0.5 * (jp + jm)
    odd = 0.5 * (jp - jm)
    return {
        "rel_l2_plus": current_rel_l2(j0, jp, dt),
        "rel_l2_minus": current_rel_l2(j0, jm, dt),
        "rel_l2_even_minus_j0": current_rel_l2(j0, even, dt),
        "rel_l2_odd_over_j0": norm_dt(odd, dt) / max(norm_dt(j0, dt), 1e-30),
        "maxabs_plus_minus_j0": float(np.max(np.abs(jp - j0))),
        "maxabs_minus_minus_j0": float(np.max(np.abs(jm - j0))),
    }


def run_covariant_pair(geo, p, prot, kf, w, *, mu, neel, ntpc, delta_frac, n_hold_cycles=0):
    kf = np.asarray(kf, dtype=float)
    w = np.asarray(w, dtype=float)
    kf_p = shift_kf(geo, kf, delta_frac=delta_frac)
    kf_m = shift_kf(geo, kf, delta_frac=-np.asarray(delta_frac, dtype=float))
    rec0 = propagate_cjj(
        geo, p, prot, nt_per_cycle=ntpc, neel=neel, sign=+1, mu=mu,
        verbose=False, kf_frac=kf, w_k=w, n_hold_cycles=n_hold_cycles,
    )
    recp = propagate_cjj(
        geo, p, prot, nt_per_cycle=ntpc, neel=neel, sign=+1, mu=mu,
        verbose=False, kf_frac=kf_p, w_k=w, n_hold_cycles=n_hold_cycles,
    )
    recm = propagate_cjj(
        geo, p, prot, nt_per_cycle=ntpc, neel=neel, sign=+1, mu=mu,
        verbose=False, kf_frac=kf_m, w_k=w, n_hold_cycles=n_hold_cycles,
    )
    dt = float(rec0["dt"])
    cmp = compare_currents(rec0["mu_per_cell"], recp["mu_per_cell"], recm["mu_per_cell"], dt)
    return {
        "kind": "covariant_H_and_Pi0",
        "delta_frac": np.asarray(delta_frac, dtype=float).tolist(),
        "n_k": int(kf.shape[0]),
        "compare": cmp,
        "wall_s": float(rec0["wall_seconds"] + recp["wall_seconds"] + recm["wall_seconds"]),
        "j0": rec0["mu_per_cell"],
        "jp": recp["mu_per_cell"],
        "jm": recm["mu_per_cell"],
        "times": rec0["times"],
        "dt": dt,
    }


def run_permutation_identity(geo, p, prot, nk: int, *, mu, neel, ntpc):
    """δk_frac = (1/nk, 0) permutes the MP grid. J must match."""
    kf, _, w = k_grid(geo, nk, shift=0.0)
    dk = np.array([1.0 / float(nk), 0.0])
    rec = run_covariant_pair(geo, p, prot, kf, w, mu=mu, neel=neel, ntpc=ntpc, delta_frac=dk)
    rec["kind"] = "grid_permutation_1_over_nk"
    rec["exact_set_map"] = True
    return rec


def run_sudden_fixed_pi0(geo, p, prot, kf, w, *, mu, neel, ntpc, delta_frac):
    """Contrast only: same Π0(k), sudden constant extra κ. Not a gauge."""
    kf = np.asarray(kf, dtype=float)
    w = np.asarray(w, dtype=float)
    dcart = geo.k_cart(np.asarray(delta_frac, dtype=float))

    def _const(_t, d=dcart):
        return d

    rec0 = propagate_cjj(
        geo, p, prot, nt_per_cycle=ntpc, neel=neel, sign=+1, mu=mu,
        verbose=False, kf_frac=kf, w_k=w,
    )
    recs = propagate_cjj(
        geo, p, prot, nt_per_cycle=ntpc, neel=neel, sign=+1, mu=mu,
        verbose=False, kf_frac=kf, w_k=w, a_extra=_const,
    )
    dt = float(rec0["dt"])
    return {
        "kind": "sudden_constant_A_fixed_Pi0_not_a_gauge",
        "delta_frac": np.asarray(delta_frac, dtype=float).tolist(),
        "n_k": int(kf.shape[0]),
        "rel_l2": current_rel_l2(rec0["mu_per_cell"], recs["mu_per_cell"], dt),
        "maxabs": float(np.max(np.abs(recs["mu_per_cell"] - rec0["mu_per_cell"]))),
        "note": "kick_plus_fixed_Pi0_not_used_as_gauge_fail",
        "wall_s": float(rec0["wall_seconds"] + recs["wall_seconds"]),
    }


def gauge_verdict(perm, generic_rows: list) -> dict:
    perm_rel = float(perm["compare"]["rel_l2_plus"])
    perm_ok = bool(perm_rel < 1e-8)
    rels = np.array([r["compare"]["rel_l2_plus"] for r in generic_rows], dtype=float)
    nks = np.array([r["n_k"] for r in generic_rows], dtype=float)
    improves = bool(len(rels) >= 2 and rels[-1] < 0.85 * rels[0])
    small = bool(np.max(rels) < 0.05)
    return {
        "permutation_rel_l2": perm_rel,
        "permutation_pass": perm_ok,
        "generic_rel_l2": rels.tolist(),
        "generic_n_k": nks.tolist(),
        "generic_improves_with_nk": improves,
        "generic_all_lt_0p05": small,
        "covariant_pass": bool(perm_ok and (improves or small)),
        "old_fixed_Pi0_criterion": "withdrawn_not_a_gauge",
        "note": (
            "Pass if a grid permutation leaves J invariant and a generic "
            "off-grid k-shift residual shrinks or is already small. "
            "A leftover O(1) residual that does not shrink is quadrature, "
            "not an automatic gauge failure of the Hamiltonian."
        ),
    }


def slim_gauge(rec: dict) -> dict:
    skip = {"j0", "jp", "jm", "times"}
    return {k: v for k, v in rec.items() if k not in skip}
