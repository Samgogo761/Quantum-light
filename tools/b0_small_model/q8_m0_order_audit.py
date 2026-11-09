#!/usr/bin/env python3
"""Canonical-defect order audit and required-noise diagnostic.

No new physics. Diagnostic only: scale γ → λγ, never patch ΔC into a bath.
"""
from __future__ import annotations

import numpy as np

from q8_m0_feedback_op import apply_Gadj, apply_L, apply_Radj
from q8_m0_io_audit import (
    F_from_phi,
    analyze_overlap_kernel,
    c_el_field,
    field_noise_kernel,
    h2_row_identity,
    h2_st_from_self,
)
from q8_m0_volterra import project_with_g as _project_with_g

ORDER_ID = "canonical_defect_order_audit_20260923"
DELTA_C_NOTE = "diagnostic_not_a_production_langevin_patch"
LAMBDAS = (0.0, 1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 0.25, 0.5, 1.0)


def field_noise_kernel_born(x, w, mean_d, dt, gamma, phi, omega, times, dw):
    """N = ⟨k, G ξ⟩, no R dressing."""
    k = analyze_overlap_kernel(phi, omega, times, dw)
    g_re = apply_Gadj(np.real(k), dt, gamma)
    g_im = apply_Gadj(np.imag(k), dt, gamma)
    return float(dt) * g_re + 1j * float(dt) * g_im


def field_noise_kernel_neumann(x, w, mean_d, dt, gamma, phi, omega, times, dw, n: int):
    """N = ⟨k, G I^{(n)}⟩ with I^{(n)} = Σ_{j=0}^{n-1} (R G)^j ξ."""
    k = analyze_overlap_kernel(phi, omega, times, dw)
    u_re = apply_Gadj(np.real(k), dt, gamma)
    u_im = apply_Gadj(np.imag(k), dt, gamma)

    def step(g):
        return apply_Gadj(apply_Radj(x, w, mean_d, g, dt), dt, gamma)

    def series(u):
        acc = np.array(u, dtype=float, copy=True)
        cur = np.array(u, dtype=float, copy=True)
        for _ in range(max(int(n), 1) - 1):
            cur = step(cur)
            acc = acc + cur
        return acc

    g_re = series(u_re)
    g_im = series(u_im)
    return float(dt) * g_re + 1j * float(dt) * g_im


def c_el_from_kernel(x, w, k_eff) -> complex:
    yk = _project_with_g(x, k_eff)
    yks = _project_with_g(x, np.conj(k_eff))
    n_f = complex(np.einsum("k,kcv,kcv->", w, np.conj(yk), yk))
    n_fs = complex(np.einsum("k,kcv,kcv->", w, np.conj(yks), yks))
    return n_fs - n_f


def c_el_born(x, w, mean_d, dt, gamma, phi, omega, times, dw) -> complex:
    return c_el_from_kernel(
        x, w, field_noise_kernel_born(x, w, mean_d, dt, gamma, phi, omega, times, dw),
    )


def c_el_neumann(x, w, mean_d, dt, gamma, phi, omega, times, dw, n: int) -> complex:
    return c_el_from_kernel(
        x, w, field_noise_kernel_neumann(
            x, w, mean_d, dt, gamma, phi, omega, times, dw, n,
        ),
    )


def a_self_neumann(x, w, mean_d, dt, gamma, F, n: int) -> dict:
    """A_self = (L + L² + … + L^n) A_in."""
    hr = np.real(np.asarray(F))
    hi = np.imag(np.asarray(F))

    def series(a0):
        acc = np.zeros_like(a0)
        cur = np.asarray(a0, dtype=float)
        for _ in range(int(n)):
            cur = apply_L(x, w, mean_d, cur, dt, gamma)
            acc = acc + cur
        return acc

    return {
        "A_self_re": series(hr),
        "A_self_im": series(hi),
        "finite": True,
        "maxabs_A_self": 0.0,
        "F": np.asarray(F),
    }


def prepare_drives_neumann(x, w, mean_d, dt, gamma, phis, omega, times, dw, n: int):
    out = []
    for phi in phis:
        F = F_from_phi(phi, omega, times, dw)
        rec = a_self_neumann(x, w, mean_d, dt, gamma, F, n)
        rec["phi"] = phi
        out.append(rec)
    return out


def h2_identity_at_gamma(
    x, w, mean_d, times, dt, gamma, omega, dw, phis, phi_h2, *, kind: str, neumann_n: int = 1,
):
    """kind: resummed | born | neumann. H2 row on R_out."""
    if kind == "resummed":
        from q8_m0_io_audit import prepare_drives

        drives = prepare_drives(x, w, mean_d, dt, gamma, phis, omega, times, dw)
        cel = complex(c_el_field(x, w, mean_d, dt, gamma, phi_h2, omega, times, dw)["C_el"])
    elif kind == "born":
        drives = prepare_drives_neumann(x, w, mean_d, dt, gamma, phis, omega, times, dw, 1)
        cel = c_el_born(x, w, mean_d, dt, gamma, phi_h2, omega, times, dw)
    elif kind == "neumann":
        drives = prepare_drives_neumann(
            x, w, mean_d, dt, gamma, phis, omega, times, dw, neumann_n,
        )
        cel = c_el_neumann(
            x, w, mean_d, dt, gamma, phi_h2, omega, times, dw, neumann_n,
        )
    else:
        raise ValueError(kind)
    rec = h2_row_identity(phis, drives, phi_h2, omega, times, dt, dw, cel)
    comm = float(rec["commutator_real"])
    s_rl = complex(rec["S_H2_RL"]["re"], rec["S_H2_RL"]["im"]) if rec["S_H2_RL"] else 0.0
    d_lin = 2.0 * float(np.real(s_rl - 1.0)) + float(np.real(cel))
    rec["kind"] = kind
    rec["gamma"] = float(gamma)
    rec["delta_can"] = comm - 1.0
    rec["delta_vac"] = float(rec["vac_piece_real"]) - 1.0
    rec["delta_lin_probe"] = d_lin
    rec["C_el_real"] = float(np.real(cel))
    rec["neumann_n"] = None if kind != "neumann" else int(neumann_n)
    return rec


def st_matrices(phis, drives, omega, times, dt, dw):
    n = len(phis)
    s = np.zeros((n, 2 * n), dtype=complex)
    t = np.zeros((n, 2 * n), dtype=complex)
    for nu, rec in enumerate(drives):
        for mu, phi_mu in enumerate(phis):
            row = h2_st_from_self(
                phi_mu, rec["F"], rec["A_self_re"], rec["A_self_im"],
                omega, times, dt, dw,
            )
            s[mu, 2 * nu] = row["S_RL"]
            t[mu, 2 * nu] = row["T_RL"]
            s[mu, 2 * nu + 1] = row["S_RR"]
            t[mu, 2 * nu + 1] = row["T_RR"]
    return s, t


def cel_matrix(x, w, mean_d, dt, gamma, phis, omega, times, dw, *, kind: str = "resummed", neumann_n: int = 1):
    ys = []
    yss = []
    for phi in phis:
        if kind == "resummed":
            k = field_noise_kernel(x, w, mean_d, dt, gamma, phi, omega, times, dw)
        elif kind == "born":
            k = field_noise_kernel_born(x, w, mean_d, dt, gamma, phi, omega, times, dw)
        else:
            k = field_noise_kernel_neumann(
                x, w, mean_d, dt, gamma, phi, omega, times, dw, neumann_n,
            )
        ys.append(_project_with_g(x, k))
        yss.append(_project_with_g(x, np.conj(k)))
    n = len(phis)
    c = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            nij = complex(np.einsum("k,kcv,kcv->", w, np.conj(ys[i]), ys[j]))
            nijs = complex(np.einsum("k,kcv,kcv->", w, np.conj(yss[i]), yss[j]))
            c[i, j] = nijs - nij
    return c


def nambu_deficit(s, t, c_el) -> dict:
    """ΔC = I − (S S† − T T† + C_el). Not a production patch."""
    s = np.asarray(s, dtype=complex)
    t = np.asarray(t, dtype=complex)
    c_el = np.asarray(c_el, dtype=complex)
    vac = s @ s.conj().T - t @ t.conj().T
    ident = vac + c_el
    n = int(s.shape[0])
    eye = np.eye(n, dtype=complex)
    d_c = eye - ident
    c_req = eye - vac
    return {
        "vac": vac,
        "ident": ident,
        "delta_C": d_c,
        "C_req": c_req,
        "fro_delta": float(np.linalg.norm(d_c)),
        "fro_C_req": float(np.linalg.norm(c_req)),
        "diag_delta": np.real(np.diag(d_c)).astype(float),
        "herm_delta": float(np.linalg.norm(d_c - d_c.conj().T)),
        "note": DELTA_C_NOTE,
    }


def packet_omega_centroid(phi, omega, dw: float) -> float:
    dens = np.sum(np.abs(np.asarray(phi)) ** 2, axis=1)
    tot = float(dw) * float(np.sum(dens))
    return float(dw * np.sum(np.asarray(omega, dtype=float) * dens) / max(tot, 1e-30))


def leading_power(lambdas, values, *, floor: float = 1e-14, lmin: float = 1e-4, lmax: float = 0.3):
    """Fit log|Δ| = p log λ + b on the small-λ window. None if underdetermined."""
    xs, ys = [], []
    used = []
    for lam, val in zip(lambdas, values):
        lam = float(lam)
        val = float(val)
        if lam < lmin or lam > lmax:
            continue
        if abs(val) < floor:
            continue
        xs.append(np.log(lam))
        ys.append(np.log(abs(val)))
        used.append(lam)
    if len(xs) < 2:
        return {"p": None, "n_fit": 0, "used": used, "r2": None}
    coef = np.polyfit(xs, ys, 1)
    p = float(coef[0])
    pred = np.polyval(coef, xs)
    ss_res = float(np.sum((np.asarray(ys) - pred) ** 2))
    ss_tot = float(np.sum((np.asarray(ys) - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    ratios = []
    pairs = [(float(a), float(b)) for a, b in zip(lambdas, values) if lmin <= a <= lmax and abs(b) >= floor]
    for (l1, v1), (l2, v2) in zip(pairs, pairs[1:]):
        if l2 > 1e-16 * l1:
            ratios.append(float(np.log(abs(v2) / abs(v1)) / np.log(l2 / l1)))
    return {
        "p": p,
        "p_eta_bookkeeping": 2.0 * p,
        "log_prefactor": float(coef[1]),
        "r2": float(r2),
        "n_fit": int(len(xs)),
        "used": used,
        "consecutive_p": ratios,
        "note": (
            "p is the power of diagnostic λ in γ→λγ. "
            "Formal η-bookkeeping λ_η~√γ so p_η=2p."
        ),
    }


def scale_table(lambdas, values) -> list[dict]:
    rows = []
    for lam, val in zip(lambdas, values):
        lam = float(lam)
        val = float(val)
        rec = {
            "lambda": lam,
            "delta": val,
            "abs_delta": abs(val),
            "over_l2": None if lam < 1e-16 else val / (lam ** 2),
            "over_l3": None if lam < 1e-16 else val / (lam ** 3),
            "over_l4": None if lam < 1e-16 else val / (lam ** 4),
        }
        rows.append(rec)
    return rows


def pack_cmatrix(m) -> dict:
    m = np.asarray(m)
    return {
        "re": np.real(m).astype(float).tolist(),
        "im": np.imag(m).astype(float).tolist(),
        "shape": list(m.shape),
    }


def slim_h2(rec: dict) -> dict:
    keep = (
        "kind", "gamma", "n_packets", "n_inputs", "sum_absS2", "sum_absT2",
        "vac_piece_real", "C_el_real", "commutator_real", "abs_minus_1",
        "delta_can", "delta_vac", "delta_lin_probe", "S_H2_RL", "T_H2_RL",
        "S_H2_RR", "T_H2_RR", "neumann_n",
    )
    return {k: rec[k] for k in keep if k in rec}
