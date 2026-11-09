#!/usr/bin/env python3
"""Finite n,m qualification: correct electronic CCR / dressed resolvent, +N close.

Does not overwrite u0_h2.json, input_close.json, or older completeness JSON.
Does not change gamma, D, initial state, Hamiltonian, P1, or the packet.
Not a production domain signal.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import (
    ColumnState,
    current,
    midpoint_step,
    real_frequency_kernel,
    stationary_photon_columns,
)
from q8_m0_continuum_ground_bridge_20261003 import (
    ContinuumGroundBridge,
    photon_resolvent,
)
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import (
    assemble_incoming_bundle,
    classify_dark_columns,
    dark_column_gaps,
    dark_projector,
    dark_subspace,
    maxabs,
)
from q8_m0_k1_chi_interface import CHI_STATUS
from q8_m0_k1_packet import normalize_phi, omega_grid, phi_unnormalized
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import (
    FREQ,
    OLD_JSON,
    TWO_K,
    WEIGHTS,
    advance_half_frame,
    even_odd_from_lr,
    free_A_increment,
    m0_sample,
    packed_state,
    photon_port_indices,
    sha,
    vertices_from_frame,
)
from q8_m0_s1_kernel import parallel_perp_basis

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_nm_qualify_20261005"
OLD_CONTRACT = REPO / "results/q_line/p1_h2_finite_kcv_contract_20261004/contract.json"
CALLER_FIX = REPO / "results/q_line/p1_h2_finite_kcv_caller_fix_20261005/caller_fix.json"
INPUT_LEDGER = REPO / "results/q_line/p1_h2_finite_kcv_input_ledger_20261005/input_ledger.json"
EIGHT_CYCLE = REPO / "results/q_line/p1_h2_finite_kcv_eight_cycle_20261005/eight_cycle.json"
OLD_INPUT_CLOSE = REPO / "results/q_line/p1_h2_finite_kcv_input_close_20261005/input_close.json"
OLD_U0_H2 = REPO / "results/q_line/p1_h2_finite_kcv_u0_h2_20261005/u0_h2.json"
NTPC = 1024
NTPC_FINE = 2048
N_TAIL_CYCLES = 1
A_STAR = 0.8
FROZEN_GAMMA = 0.001080789276
COMPLEX_Z = (0.3 + 0.2j, 1.0 + 0.4j, 2.5 + 0.15j)
CHUNK_FREQ = 800
N_PANEL = 64
N_PANEL_FINE = 96
NM_REL_TOL = 0.05
# H2 envelope lives in [0.02, 0.30]. n_h2=24 gave packet mass 0.731
# and U0 CCR 0.269; that is window density, not a new physical packet.
N_H2 = 48
N_H2_FINE = 96
# Lorentz window in t = arcsinh((ω-Δ)/Γ). Γ is the narrowest bright
# singular value of the cluster, not the total |u|². Neighbouring groups
# are cut at midpoints so one panel cannot swallow another.
T_MAX_MIN = 6.0
T_MAX_CAP = 12.0
T_TAIL_CAP = 40.0
JOIN_WIDTH = 80.0
N_TAIL_MIN = 12
N_TAIL_CAP = 64
OLD_JSON_KEEP = OLD_JSON + [
    "results/q_line/p1_h2_finite_kcv_caller_fix_20261005/caller_fix.json",
    "results/q_line/p1_h2_finite_kcv_input_ledger_20261005/input_ledger.json",
    "results/q_line/p1_h2_finite_kcv_eight_cycle_20261005/eight_cycle.json",
    "results/q_line/p1_h2_finite_kcv_input_close_20261005/input_close.json",
    "results/q_line/p1_h2_finite_kcv_u0_h2_20261005/u0_h2.json",
]
STAGE = os.environ.get("Q8_STAGE", "probe")


def progress(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with (OUT / "_progress.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def pack_c(m):
    m = np.asarray(m, complex)
    out = {
        "real": np.real(m).tolist(),
        "imag": np.imag(m).tolist(),
        "maxabs": float(np.max(np.abs(m))) if m.size else 0.0,
        "shape": list(m.shape),
    }
    if m.ndim == 2 and m.shape[0] == m.shape[1] and m.size:
        herm = 0.5 * (m + m.conj().T)
        evals = np.linalg.eigvalsh(herm)
        out["trace_real"] = float(np.trace(m).real)
        out["min_hermitian_eig"] = float(evals[0])
        out["anti_hermitian_maxabs"] = float(np.max(np.abs(m - herm)))
    return out


def errors(a, b):
    diff = maxabs(np.asarray(a) - np.asarray(b))
    scale = maxabs(b)
    return {
        "max_absolute": diff,
        "reference_max": scale,
        "max_relative": diff / max(scale, 1e-30),
    }


def unpack_c(rec):
    return np.asarray(rec["real"], float) + 1j * np.asarray(rec["imag"], float)


def matrix_diff(a, b, name):
    a = np.asarray(a, complex)
    b = np.asarray(b, complex)
    fro = float(np.linalg.norm(a - b))
    scale = float(np.linalg.norm(b))
    return {
        "name": name,
        "frobenius": fro,
        "frobenius_relative": fro / max(scale, 1e-30),
        "maxabs": maxabs(a - b),
        "reference_frobenius": scale,
        "reference_maxabs": maxabs(b),
    }


def moments_diff(a, b):
    dn = matrix_diff(unpack_c(a["n"]), unpack_c(b["n"]), "n")
    dm = matrix_diff(unpack_c(a["m"]), unpack_c(b["m"]), "m")
    du = matrix_diff(
        unpack_c(a["m_reference_source"]),
        unpack_c(b["m_reference_source"]),
        "U0VT",
    )
    return {
        "n": dn,
        "m": dm,
        "m_U0VT": du,
        "normal_ccr_a": float(a["normal_ccr"]),
        "normal_ccr_b": float(b["normal_ccr"]),
        "ok": bool(
            dn["frobenius_relative"] < NM_REL_TOL
            and dm["frobenius_relative"] < NM_REL_TOL
        ),
    }


def two_mode_quad_check(n_mat, m_mat):
    """Real covariance for n_ij=<a_i^dagger a_j>, m_ij=<a_i a_j>.

    Moment symmetrization below is diagnostic only: report its defect rather
    than treating it as a repair of noncanonical output rows.
    """
    n = np.asarray(n_mat, complex)
    m_raw = np.asarray(m_mat, complex)
    if n.shape != (2, 2) or m_raw.shape != (2, 2):
        raise ValueError("Expected two 2x2 moment matrices.")
    if not (np.all(np.isfinite(n)) and np.all(np.isfinite(m_raw))):
        raise ValueError("Moment matrices must be finite.")
    n_defect = maxabs(n - n.conj().T)
    m_defect = maxabs(m_raw - m_raw.T)
    n = 0.5 * (n + n.conj().T)
    m = 0.5 * (m_raw + m_raw.T)
    # a=(x+ip)/sqrt(2). Im(n_ij) belongs in x_i p_j, not in x_i x_j.
    xx = 0.5 * np.eye(2) + np.real(n + m)
    pp = 0.5 * np.eye(2) + np.real(n - m)
    xp = np.imag(n + m)
    cov = np.zeros((4, 4), float)
    for i in range(2):
        for j in range(2):
            cov[2 * i, 2 * j] = xx[i, j]
            cov[2 * i + 1, 2 * j + 1] = pp[i, j]
            cov[2 * i, 2 * j + 1] = xp[i, j]
            cov[2 * i + 1, 2 * j] = xp[j, i]
    omega = np.kron(np.eye(2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    M = 0.5 * ((cov + 1j * omega / 2.0) + (cov + 1j * omega / 2.0).conj().T)
    spec = np.linalg.eigvalsh(M)
    return {
        "moment_convention": "n_ij=<a_i^dagger a_j>; m_ij=<a_i a_j>",
        "covariance_x1_p1_x2_p2": cov.tolist(),
        "input_n_hermiticity_maxabs": n_defect,
        "input_m_symmetry_maxabs": m_defect,
        "symmetrized_for_diagnostic_only": True,
        "min_eig_cov_plus_i_omega_over_2": float(spec[0]),
        "robertson_diag": [
            float((n[i, i].real + 0.5) ** 2 - abs(m[i, i]) ** 2) for i in range(2)
        ],
        "not_a_squeezing_or_entanglement_certificate": True,
        "below_output_ccr_is_not_qualification": True,
    }


def unique_gaps(gaps):
    return np.unique(np.round(np.asarray(gaps, float), 10))


def trap_weights(x):
    x = np.asarray(x, float)
    w = np.zeros_like(x)
    if x.size < 2:
        return w
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def merge_nodes(oms, wts):
    key = np.round(np.asarray(oms, float), 12)
    order = np.argsort(key, kind="stable")
    key, oms, wts = key[order], np.asarray(oms, float)[order], np.asarray(wts, float)[order]
    out_o, out_w = [oms[0]], [wts[0]]
    for o, w, k in zip(oms[1:], wts[1:], key[1:]):
        if abs(k - np.round(out_o[-1], 12)) <= 1e-12:
            out_w[-1] += w
        else:
            out_o.append(o)
            out_w.append(w)
    return np.asarray(out_o, float), np.asarray(out_w, float)


def eta_of(gamma):
    return float(np.sqrt(float(gamma) / (2.0 * np.pi)))


def independent_field_conversion(gamma, sample, omega=0.1):
    eta = eta_of(gamma)
    ph = stationary_photon_columns(sample["gap"], sample["u"], sample["D"], gamma, [omega])
    ii = current(ph.initial, sample["u"], sample["D"])
    a = ph.initial.A
    af = ph.free_A0
    w = float(omega)
    lhs = 1j * 2.0 * np.pi * eta * ii / np.sqrt(w)
    rhs = np.sqrt(w) * (a - af) / eta
    wrong = np.sqrt(gamma) / eta
    return {
        "omega": w,
        "eta": eta,
        "sqrt_gamma": float(np.sqrt(gamma)),
        "ratio_sqrt_gamma_over_eta": float(wrong),
        "expect_sqrt_2pi": float(np.sqrt(2.0 * np.pi)),
        "i_2pi_eta_I_over_sqrtw_vs_sqrtw_AminusAf_over_eta_maxabs": maxabs(lhs - rhs),
        "wrong_sqrt_gamma_kernel_relative": float(wrong - 1.0),
        "A_free_one_pol": float(eta / np.sqrt(w)),
        "a_from_A_free": float(np.sqrt(w / gamma) * (eta / np.sqrt(w))),
        "expect_1_over_sqrt_2pi": float(1.0 / np.sqrt(2.0 * np.pi)),
    }


def detector_kernel_scalar(gamma, prot, nstep, h):
    """K(t)=i eta ∫ dω φ(ω)/sqrt(ω) e^{+iωt}, eta=sqrt(γ/2π)."""
    eta = eta_of(gamma)
    w0 = float(prot.omega0)
    tp = float(prot.t_pulse)
    om = omega_grid(w0, tp)
    om = om[om > 0.0]
    raw = phi_unnormalized(om, 2.0 * w0, tp)
    phi, nrm = normalize_phi(om, raw)
    t_mid = (np.arange(nstep) + 0.5) * h
    weight = phi / np.sqrt(om)
    k = np.empty(nstep, complex)
    batch = 64
    for s in range(0, nstep, batch):
        sl = slice(s, min(s + batch, nstep))
        phase = np.exp(1j * om[:, None] * t_mid[sl][None, :])
        k[sl] = 1j * eta * np.trapezoid(weight[:, None] * phase, om, axis=0)
    wc = 2.0 * w0
    k_pulled = np.empty(nstep, complex)
    for s in range(0, nstep, batch):
        sl = slice(s, min(s + batch, nstep))
        phase = np.exp(1j * om[:, None] * t_mid[sl][None, :])
        k_pulled[sl] = (1j * eta / np.sqrt(wc)) * np.trapezoid(
            phi[:, None] * phase, om, axis=0
        )
    pulse = t_mid <= tp
    tail = ~pulse
    extra = t_mid > (prot.n_cycles * prot.period)
    mass = np.abs(k) ** 2
    return {
        "k_scalar": k,
        "k_pulled_same_phi": k_pulled,
        "t_mid": t_mid,
        "eta": eta,
        "normalizer": float(nrm),
        "omega_c": float(wc),
        "keep_1_over_sqrt_omega": True,
        "prefactor_is_eta_not_sqrt_gamma": True,
        "pulled_center_is_not_the_kernel": True,
        "k_vs_pulled_maxabs": maxabs(k - k_pulled),
        "mass_pulse": float(np.trapezoid(mass[pulse], t_mid[pulse])) if np.any(pulse) else 0.0,
        "mass_after_pulse": float(np.trapezoid(mass[tail], t_mid[tail])) if np.any(tail) else 0.0,
        "mass_after_extra_cycle": float(np.trapezoid(mass[extra], t_mid[extra])) if np.any(extra) else 0.0,
        "n_omega": int(om.size),
        "phi_omega": om,
        "phi": phi,
    }


def phi_at(omega, prot, normalizer):
    om = np.asarray(omega, float)
    raw = phi_unnormalized(om, 2.0 * float(prot.omega0), float(prot.t_pulse))
    return raw / float(normalizer)


def cluster_width(gaps, u, gamma, dg):
    """Narrowest bright Lorentz width Γ = γ σ_min² / Δ of this gap cluster."""
    mask = np.abs(np.asarray(gaps, float) - float(dg)) < 1e-9
    block = np.asarray(u, complex)[:, mask]
    if block.size == 0:
        return 1e-16
    sv = np.linalg.svd(block, compute_uv=False)
    sv = sv[sv > 1e-12]
    if sv.size == 0:
        return 1e-16
    st2 = float(sv[-1] ** 2)
    return float(max(float(gamma) * st2 / max(abs(float(dg)), 1e-12), 1e-16))


def cluster_rank_width(gaps, u, gamma, dg):
    mask = np.abs(np.asarray(gaps, float) - float(dg)) < 1e-9
    block = np.asarray(u, complex)[:, mask]
    sv = np.linalg.svd(block, compute_uv=False) if block.size else np.zeros(0)
    bright_sv = sv[sv > 1e-12]
    widths = np.array(
        [
            float(gamma) * float(s * s) / max(abs(float(dg)), 1e-12)
            for s in bright_sv
        ],
        float,
    )
    return int(bright_sv.size), widths


def even_count(n, lo=8):
    n = max(int(n), int(lo))
    if n % 2:
        n += 1
    return n


def t_gauss_panel(center, width, t_a, t_b, n):
    """Gauss-Legendre in t=arcsinh((ω-Δ)/Γ) on (t_a, t_b).

    dω = Γ cosh(t) dt is counted once. Nodes are interior, so they do not
    sit on the pole or on the clipped ω-endpoints. Clipping is a t-range
    restriction, not dropping already-weighted nodes.
    """
    width = max(float(width), 1e-16)
    center = float(center)
    t_a = float(np.clip(t_a, -T_TAIL_CAP, T_TAIL_CAP))
    t_b = float(np.clip(t_b, -T_TAIL_CAP, T_TAIL_CAP))
    if t_b <= t_a + 1e-18:
        return np.zeros(0), np.zeros(0)
    n = even_count(n, 8)
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (t_b - t_a) * x + 0.5 * (t_a + t_b)
    om = center + width * np.sinh(t)
    wt = 0.5 * (t_b - t_a) * w * width * np.cosh(t)
    keep = om > 1e-10
    keep &= np.abs(om - center) > 1e-16
    return om[keep], wt[keep]


def sinh_panel(center, width, n=40, t_max=6.0, lo=None, hi=None):
    """Even Gauss-in-t panel; t=0 (the pole) is never a quadrature node."""
    width = max(float(width), 1e-16)
    center = float(center)
    t_max = float(t_max)
    t_a, t_b = -t_max, t_max
    if lo is not None:
        t_a = max(t_a, float(np.arcsinh((float(lo) - center) / width)))
    if hi is not None:
        t_b = min(t_b, float(np.arcsinh((float(hi) - center) / width)))
    return t_gauss_panel(center, width, t_a, t_b, n)


def gauss_segment(a, b, n):
    """Open Gauss-Legendre on (a, b). No endpoint samples."""
    a, b = float(a), float(b)
    if b <= a:
        return np.zeros(0), np.zeros(0)
    n = max(int(n), 2)
    x, w = np.polynomial.legendre.leggauss(n)
    om = 0.5 * (b - a) * x + 0.5 * (a + b)
    wt = 0.5 * (b - a) * w
    keep = om > 1e-10
    return om[keep], wt[keep]


def trap_segment(a, b, n):
    a, b = float(a), float(b)
    if b <= a:
        return np.zeros(0), np.zeros(0)
    n = max(int(n), 2)
    om = np.linspace(a, b, n)
    return om, trap_weights(om)


def geom_segment(a, b, n):
    a, b = float(a), float(b)
    if b <= a:
        return np.zeros(0), np.zeros(0)
    n = max(int(n), 2)
    om = np.geomspace(max(a, 1e-10), b, n)
    return om, trap_weights(om)


def bright_unique_gaps(gaps, u):
    out = []
    for dg in unique_gaps(gaps):
        mask = np.abs(np.asarray(gaps, float) - float(dg)) < 1e-9
        strength = float(np.sqrt(np.sum(np.abs(u[:, mask]) ** 2)))
        if strength > 1e-12:
            out.append(float(dg))
    return np.asarray(out, float)


def grouped_bright_gaps(gaps, u, gamma, *, join=JOIN_WIDTH):
    bright = bright_unique_gaps(gaps, u)
    widths = np.array([cluster_width(gaps, u, gamma, dg) for dg in bright], float)
    groups = []
    cur = [0]
    for i in range(1, bright.size):
        thresh = join * max(widths[i], widths[i - 1])
        if bright[i] - bright[i - 1] <= thresh:
            cur.append(i)
        else:
            groups.append(np.asarray(cur, int))
            cur = [i]
    groups.append(np.asarray(cur, int))
    return bright, widths, groups


def input_quadrature(gaps, u, gamma, *, n_panel=N_PANEL, n_h2=N_H2, omega_hi=12.0,
                     omega_lo=1.0e-3, h2_lo=0.02, h2_hi=0.30):
    """Disjoint finite-reference positive-frequency measure. dν counted once.

    Each bright Voronoi cell is one account: Lorentz core in t=arcsinh((ω-Δ)/Γ)
    plus both side tails in the same t, Gauss-Legendre, weight once. H2 packet
    density replaces only the tail pieces that overlap [h2_lo, h2_hi]; poles
    inside that window keep the sinh core. No inverted cells. No trap endpoint
    sitting on a near-pole value with a wide flat weight.
    """
    bright, widths, groups = grouped_bright_gaps(gaps, u, gamma)
    if bright.size == 0:
        raise RuntimeError("No bright gaps for the finite electronic sample.")
    chunks_o, chunks_w = [], []

    def add(om, wt):
        om = np.asarray(om, float)
        wt = np.asarray(wt, float)
        if om.size:
            chunks_o.append(om)
            chunks_w.append(wt)

    n_g = len(groups)
    g_left = np.array([float(bright[grp][0]) for grp in groups], float)
    g_right = np.array([float(bright[grp][-1]) for grp in groups], float)
    cuts = np.empty(n_g + 1, float)
    cuts[0] = float(omega_lo)
    cuts[-1] = float(omega_hi)
    for i in range(n_g - 1):
        cuts[i + 1] = 0.5 * (g_right[i] + g_left[i + 1])
    if np.any(np.diff(cuts) <= 0):
        raise RuntimeError("Inverted quadrature cells.")

    h2_lo, h2_hi = float(h2_lo), float(h2_hi)
    h2_span = max(h2_hi - h2_lo, 1e-12)

    def add_omega_split(dg, gwid, t0, t1):
        if t1 <= t0 + 1e-14:
            return
        o0 = float(dg) + float(gwid) * np.sinh(t0)
        o1 = float(dg) + float(gwid) * np.sinh(t1)
        if o1 < o0:
            o0, o1 = o1, o0
            t0, t1 = t1, t0
        marks = [o0]
        for x in (h2_lo, h2_hi):
            if o0 < x < o1:
                marks.append(x)
        marks.append(o1)
        for x0, x1 in zip(marks[:-1], marks[1:]):
            if x1 <= x0 + 1e-14:
                continue
            mid = 0.5 * (x0 + x1)
            if h2_lo <= mid <= h2_hi:
                n = even_count(max(8, int(round(int(n_h2) * (x1 - x0) / h2_span))), 8)
                add(*gauss_segment(x0, x1, n))
            else:
                ta = float(np.arcsinh((x0 - dg) / gwid))
                tb = float(np.arcsinh((x1 - dg) / gwid))
                if tb < ta:
                    ta, tb = tb, ta
                n = even_count(int(N_TAIL_MIN + 2.0 * (tb - ta)), N_TAIL_MIN)
                n = min(n, N_TAIL_CAP)
                add(*t_gauss_panel(dg, gwid, ta, tb, n))

    for i, grp in enumerate(groups):
        cell_lo, cell_hi = float(cuts[i]), float(cuts[i + 1])
        if cell_hi <= cell_lo:
            raise RuntimeError("Inverted group cell.")
        dgs = bright[grp]
        gs = widths[grp]
        pole_cuts = (
            np.concatenate(([cell_lo], 0.5 * (dgs[:-1] + dgs[1:]), [cell_hi]))
            if dgs.size > 1
            else np.array([cell_lo, cell_hi])
        )
        for j, dg in enumerate(dgs):
            a, b = float(pole_cuts[j]), float(pole_cuts[j + 1])
            if b <= a:
                raise RuntimeError("Inverted pole cell.")
            gwid = max(float(gs[j]), 1e-16)
            _rank, all_w = cluster_rank_width(gaps, u, gamma, dg)
            g_max = float(np.max(all_w)) if all_w.size else gwid
            t_left = float(np.clip(np.arcsinh((a - float(dg)) / gwid), -T_TAIL_CAP, T_TAIL_CAP))
            t_right = float(np.clip(np.arcsinh((b - float(dg)) / gwid), -T_TAIL_CAP, T_TAIL_CAP))
            half = max(min(float(dg) - a, b - float(dg)), 8.0 * gwid)
            t_cell = float(np.arcsinh(half / gwid))
            t_wide = float(np.arcsinh(8.0 * g_max / gwid))
            t_core = min(T_MAX_CAP, max(T_MAX_MIN, t_wide, min(t_cell, T_MAX_CAP)))
            t_core = min(t_core, max(-t_left, 0.0), max(t_right, 0.0))
            if t_core <= 1e-12:
                raise RuntimeError("Inverted sinh window.")
            add(*t_gauss_panel(float(dg), gwid, -t_core, t_core, n_panel))
            add_omega_split(float(dg), gwid, t_left, -t_core)
            add_omega_split(float(dg), gwid, t_core, t_right)
            om_left = float(dg) + gwid * np.sinh(t_left)
            om_right = float(dg) + gwid * np.sinh(t_right)
            if om_left > a + 1e-10:
                add(*gauss_segment(a, om_left, 12))
            if b > om_right + 1e-10:
                add(*gauss_segment(om_right, b, 12))

    oms = np.concatenate(chunks_o) if chunks_o else np.zeros(0)
    wts = np.concatenate(chunks_w) if chunks_w else np.zeros(0)
    oms, wts = merge_nodes(oms, wts)
    keep = (oms > 1e-10) & (wts > 0.0)
    for gval in unique_gaps(gaps):
        keep &= np.abs(oms - gval) > 1e-14
    oms, wts = oms[keep], wts[keep]
    if np.any(wts < 0):
        raise RuntimeError("Negative quadrature weight.")
    return oms, wts


def freq_chunks(oms, wts, size=CHUNK_FREQ):
    oms = np.asarray(oms, float)
    wts = np.asarray(wts, float)
    for i in range(0, oms.size, int(size)):
        yield oms[i : i + size], wts[i : i + size]


def a_r_out_from_column(a_col, free_a, labels, col):
    lab = labels[col]
    a = np.asarray(a_col, complex)[:, col]
    fa = np.asarray(free_a, complex)[:, col]
    if lab[0] == "dark_combination":
        return np.zeros(2, complex)
    side = lab[1]
    if side == "L":
        return a
    if side == "R":
        return a - fa
    raise ValueError(lab)


def oneside_bd(gaps, u, d, gamma, eta, om):
    k = real_frequency_kernel(gaps, u, d, om)
    af = (eta / np.sqrt(om)) * np.eye(2)
    a = np.linalg.solve(np.eye(2) + 1j * gamma * k / om, af)
    b = (u.T @ a) / (gaps - om)[:, None]
    bd = (u.conj().T @ a) / (gaps + om)[:, None]
    return b, bd, a


def even_bd(gaps, u, d, gamma, eta, om):
    """Even two-pol columns. L=R so even=√2×oneside; do not add L and R."""
    b, bd, a = oneside_bd(gaps, u, d, gamma, eta, om)
    s = np.sqrt(2.0)
    return s * b, s * bd, s * a


def f_from_bd(b, bd):
    """Real electronic coordinates of one incoming annihilator column set."""
    b = np.asarray(b, complex)
    bd = np.asarray(bd, complex)
    return np.vstack((b + bd, -1j * (b - bd))) / np.sqrt(2.0)


def dressed_electron_resolvent(bridge, z):
    """R_e + R_e C^T (I+q_a K)^{-1} q_a C R_e on original real coordinates."""
    z = complex(z)
    n = int(bridge.gaps.size)
    ident = np.eye(2 * n, dtype=complex)
    re = bridge.electron_action(z, ident)
    c = np.asarray(bridge.ci, complex)
    qa = photon_resolvent(z, bridge.gamma)
    k = bridge.electronic_kernel(z)
    mid = np.linalg.solve(np.eye(2, dtype=complex) + qa * k, qa * np.eye(2, dtype=complex))
    return re + re @ c.T @ mid @ c @ re, re


def electronic_measure_report(sample, gamma, oms, wts):
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    eta = eta_of(gamma)
    basis, info, near = dark_subspace(gaps, u)
    p_dark = dark_projector(basis)
    n = len(gaps)
    acc_bb = np.zeros((n, n), complex)
    acc_dd = np.zeros((n, n), complex)
    acc_bd = np.zeros((n, n), complex)
    acc_db = np.zeros((n, n), complex)
    acc_ii = np.zeros((2, 2), complex)
    skipped = 0
    for om, dw in zip(oms, wts):
        try:
            b, bd, a = even_bd(gaps, u, d, gamma, eta, float(om))
        except Exception:
            skipped += 1
            continue
        acc_bb += dw * (b @ b.conj().T)
        acc_dd += dw * (bd.conj() @ bd.T)
        acc_bd += dw * (b @ bd.conj().T)
        acc_db += dw * (bd.conj() @ b.T)
        ii = u.conj() @ b + u @ bd - d @ a
        acc_ii += dw * (ii @ ii.conj().T)
    ccr_incomplete = acc_bb + p_dark - np.eye(n)
    ccr_n = acc_bb - acc_dd + p_dark - np.eye(n)
    ccr_a = acc_bd - acc_db
    herm = 0.5 * (ccr_n + ccr_n.conj().T)
    evals = np.linalg.eigvalsh(herm)
    return {
        "n_nodes": int(oms.size),
        "sum_dnu": float(np.sum(wts)),
        "omega_min": float(oms[0]),
        "omega_max": float(oms[-1]),
        "covers_min_gap": bool(oms[0] < sample["min_direct_gap"] < oms[-1]),
        "covers_max_gap": bool(oms[0] < sample["max_direct_gap"] < oms[-1]),
        "skipped_poles": int(skipped),
        "n_dark": int(basis.shape[1]),
        "n_el": int(n),
        "near_degenerate_pairs": int(len(near)),
        "trace_photon_bb": float(np.trace(acc_bb).real),
        "trace_dd": float(np.trace(acc_dd).real),
        "trace_dark": float(np.trace(p_dark).real),
        "electronic_ccr_maxabs_incomplete_bb_only": maxabs(ccr_incomplete),
        "electronic_ccr_maxabs": maxabs(ccr_n),
        "electronic_ccr_normal_maxabs": maxabs(ccr_n),
        "electronic_ccr_anomalous_maxabs": maxabs(ccr_a),
        "electronic_ccr_min_eig": float(evals[0]),
        "electronic_ccr_max_eig": float(evals[-1]),
        "dark_leak_of_photon_bb": maxabs(p_dark @ acc_bb @ p_dark),
        "old_bb_only_flag_not_reused": True,
        "weight_once_sqrt_dnu_or_dnu_in_contraction": "even_two_pol_dnu_in_contraction",
        "one_sided_two_pol_not_LR_double_count": True,
        "even_is_sqrt2_oneside": True,
        "not_bare_photon_covariance": True,
        "info_ranks": [int(r["coupling_rank"]) for r in info],
    }


def resolvent_from_measure(sample, gamma, oms, wts, z_list):
    """Dressed electronic resolvent from the same even two-pol measure.

    Old Stieltjes-vs-bare-diag is kept as a diagnostic only.
    """
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    eta = eta_of(gamma)
    basis, info, near = dark_subspace(gaps, u)
    p_dark = dark_projector(basis)
    point_gaps = dark_column_gaps(info)
    n = len(gaps)
    bridge = ContinuumGroundBridge(gaps, u, d, gamma)
    rows = []
    for z in z_list:
        z = complex(z)
        acc_bb = np.zeros((n, n), complex)
        acc_r = np.zeros((2 * n, 2 * n), complex)
        skipped = 0
        for om, dw in zip(oms, wts):
            try:
                b, bd, a = even_bd(gaps, u, d, gamma, eta, float(om))
            except Exception:
                skipped += 1
                continue
            acc_bb += dw * (b @ b.conj().T) / (om - z)
            f = f_from_bd(b, bd)
            acc_r += dw * ((f @ f.conj().T) / (om - z) + (f.conj() @ f.T) / (om + z))
        for j in range(basis.shape[1]):
            v = basis[:, j : j + 1]
            f = f_from_bd(v, np.zeros_like(v))
            dg = float(point_gaps[j])
            acc_r += (f @ f.conj().T) / (dg - z) + (f.conj() @ f.T) / (dg + z)
        target = np.diag(1.0 / (gaps - z))
        recon = acc_bb + p_dark @ target @ p_dark
        bright = np.eye(n) - p_dark
        r_dressed, r_bare = dressed_electron_resolvent(bridge, z)
        qi = bridge.electron_current_resolvent(z)
        rows.append({
            "z": {"real": float(z.real), "imag": float(z.imag)},
            "electronic_stieltjes_vs_diag": errors(recon, target),
            "bright_stieltjes_vs_diag": errors(
                bright @ acc_bb @ bright, bright @ target @ bright
            ),
            "dressed_resolvent_vs_bridge": errors(acc_r, r_dressed),
            "dressed_vs_bare_resolvent": errors(r_dressed, r_bare),
            "spectral_vs_bare_resolvent": errors(acc_r, r_bare),
            "not_dressed_II_versus_bare_qi": True,
            "bare_diag_not_the_acceptance_target": True,
            "bare_qi_reported_not_used_as_jump_target": pack_c(qi),
            "skipped_poles": int(skipped),
        })
    return rows


def accumulate_uv(geo, sample, prot, photon, gamma, initial, n_point, k_scalar, nstep, h, pol):
    """Online midpoint rows for I_c and I_c*. Current from midpoint state."""
    state = initial.copy()
    frame = sample["frame"].copy()
    ncol = int(state.b.shape[1])
    acc_u = np.zeros((2, ncol), complex)
    acc_v = np.zeros((2, ncol), complex)
    n_pulse = int(round(prot.n_cycles * NTPC))
    row_u_pulse = acc_u.copy()
    started = time.perf_counter()
    ii = np.zeros((2, ncol), complex)
    for step in range(nstep):
        t = step * h
        for which, when in enumerate((t + h / 4.0, t + 3.0 * h / 4.0)):
            frame = advance_half_frame(
                geo, sample["kc"], prot, sample["neel"], frame, when, h
            )
            if which == 0:
                u, d = vertices_from_frame(
                    geo, sample["kc"], prot, sample["neel"], t + 0.5 * h, frame, sample["weights"]
                )
                fa = free_A_increment(photon, t, t + h, n_point)
                prev = state
                state = midpoint_step(state, u, d, h, gamma, free_A_increment=fa)
                mid = ColumnState(
                    0.5 * (prev.b + state.b),
                    0.5 * (prev.bdag + state.bdag),
                    0.5 * (prev.A + state.A),
                )
                ii = current(mid, u, d)
                src = pol @ ii
                acc_u += h * k_scalar[step] * src
                acc_v += h * k_scalar[step] * np.conj(src)
        if step + 1 == n_pulse:
            row_u_pulse = acc_u.copy()
        if (step + 1) % NTPC == 0:
            print(
                f"N={sample['neel']:+d} A={prot.a0_dimless:g} cycle {(step+1)//NTPC}/{nstep//NTPC}",
                flush=True,
            )
    return {
        "delta_u": acc_u,
        "v": acc_v,
        "delta_u_pulse": row_u_pulse,
        "delta_u_tail": acc_u - row_u_pulse,
        "n_pulse": n_pulse,
        "final_I_maxabs": maxabs(ii),
        "n_steps": int(nstep),
        "dt": float(h),
        "wall_seconds": time.perf_counter() - started,
        "ncol": ncol,
        "n_el": int(state.b.shape[0]),
        "bdag_not_set_to_b_conjugate": True,
    }


def independent_stationary_delta(photon, sample, k_scalar, h, nstep, n_point, geo):
    pol = np.vstack(parallel_perp_basis(geo))
    i0 = current(photon.initial, sample["u"], sample["D"])
    t_mid = (np.arange(nstep) + 0.5) * h
    n_ph = int(i0.shape[1])
    pred = np.zeros((2, n_ph + n_point), complex)
    hk = h * np.asarray(k_scalar, complex)
    omega = np.asarray(photon.omega, float)
    integ = np.empty(n_ph, complex)
    batch = 64
    for s in range(0, n_ph, batch):
        sl = slice(s, min(s + batch, n_ph))
        phase = np.exp(-1j * omega[sl][:, None] * t_mid[None, :])
        integ[sl] = phase @ hk
    pred[:, :n_ph] = pol @ (i0 * integ[None, :])
    return pred


def dt_halving_check(geo, sample, prot, gamma, freqs, k_maker, nstep_ref, h_ref):
    """Short prefix only. Same midpoint caller, dt and dt/2."""
    bundle = assemble_incoming_bundle(sample["gap"], sample["u"], sample["D"], gamma, freqs)
    n_point = int(bundle.n_point_columns)
    initial = packed_state(bundle)
    pol = np.vstack(parallel_perp_basis(geo))
    n_short = 16
    k_h = k_maker(n_short, h_ref)
    coarse = accumulate_uv(
        geo, sample, prot, bundle.photon, gamma, initial, n_point, k_h, n_short, h_ref, pol
    )
    h2 = h_ref / 2.0
    k_h2 = k_maker(2 * n_short, h2)
    fine = accumulate_uv(
        geo, sample, prot, bundle.photon, gamma, initial.copy(), n_point, k_h2, 2 * n_short, h2, pol
    )
    return {
        "n_short": n_short,
        "dt": float(h_ref),
        "dt_half": float(h2),
        "delta_u_coarse_vs_fine": errors(coarse["delta_u"], fine["delta_u"]),
        "v_coarse_vs_fine": errors(coarse["v"], fine["v"]),
        "reference_norm_delta_u": float(np.linalg.norm(fine["delta_u"])),
        "not_full_time_tail_bound": True,
    }


def midpoint_vertices(geo, sample, prot, nstep, h):
    frame = sample["frame"].copy()
    rec_u, rec_d = [], []
    for step in range(nstep):
        t = step * h
        for which, when in enumerate((t + h / 4.0, t + 3.0 * h / 4.0)):
            frame = advance_half_frame(
                geo, sample["kc"], prot, sample["neel"], frame, when, h
            )
            if which == 0:
                u, d = vertices_from_frame(
                    geo, sample["kc"], prot, sample["neel"], t + 0.5 * h, frame, sample["weights"]
                )
                rec_u.append(u)
                rec_d.append(d)
    return rec_u, rec_d


def accumulate_cached(
    photon, gamma, initial, n_point, k_scalar, vertices, nstep, h, pol, n_pulse_steps,
    snapshot_steps=None,
):
    state = initial.copy()
    ncol = int(state.b.shape[1])
    acc_u = np.zeros((2, ncol), complex)
    acc_v = np.zeros((2, ncol), complex)
    row_u_pulse = acc_u.copy()
    row_v_pulse = acc_v.copy()
    snaps = {}
    snap_map = {int(k): str(v) for k, v in dict(snapshot_steps or {}).items()}
    started = time.perf_counter()
    ii = np.zeros((2, ncol), complex)
    for step in range(nstep):
        u, d = vertices[step]
        fa = free_A_increment(photon, step * h, (step + 1) * h, n_point)
        prev = state
        state = midpoint_step(state, u, d, h, gamma, free_A_increment=fa)
        mid = ColumnState(
            0.5 * (prev.b + state.b),
            0.5 * (prev.bdag + state.bdag),
            0.5 * (prev.A + state.A),
        )
        ii = current(mid, u, d)
        src = pol @ ii
        acc_u += h * k_scalar[step] * src
        acc_v += h * k_scalar[step] * np.conj(src)
        if step + 1 == n_pulse_steps:
            row_u_pulse = acc_u.copy()
            row_v_pulse = acc_v.copy()
        if step + 1 in snap_map:
            snaps[snap_map[step + 1]] = {
                "delta_u": acc_u.copy(),
                "v": acc_v.copy(),
            }
        if (step + 1) % NTPC == 0:
            print(f"  columns cycle {(step+1)//NTPC}/{nstep//NTPC} ncol={ncol}", flush=True)
            progress(f"columns cycle {(step+1)//NTPC}/{nstep//NTPC} ncol={ncol}")
    return {
        "delta_u": acc_u,
        "v": acc_v,
        "delta_u_pulse": row_u_pulse,
        "delta_u_tail": acc_u - row_u_pulse,
        "v_pulse": row_v_pulse,
        "v_tail": acc_v - row_v_pulse,
        "snapshots": snaps,
        "final_I_maxabs": maxabs(ii),
        "n_steps": int(nstep),
        "n_pulse_steps": int(n_pulse_steps),
        "dt": float(h),
        "wall_seconds": time.perf_counter() - started,
        "ncol": ncol,
        "bdag_not_set_to_b_conjugate": True,
    }


def attach_weights(bundle, dnu_map):
    """sqrt(dν) into photon columns once. Dark columns stay point mass 1."""
    labs = bundle.photon.labels
    fac = np.ones(bundle.n_photon_columns, float)
    for j, lab in enumerate(labs):
        key = round(float(lab[0]), 12)
        if key not in dnu_map:
            raise KeyError(key)
        wt = float(dnu_map[key])
        if wt < 0.0:
            raise RuntimeError("Negative dν.")
        fac[j] = np.sqrt(wt)
    bundle.photon.initial.b *= fac[None, :]
    bundle.photon.initial.bdag *= fac[None, :]
    bundle.photon.initial.A *= fac[None, :]
    bundle.photon.free_A0 *= fac[None, :]
    return fac


def packet_mass_dnu(oms, wts, prot, nrm):
    ph = np.asarray(phi_at(oms, prot, nrm), complex)
    return float(np.sum(np.abs(ph) ** 2 * np.asarray(wts, float)))


def u0_packet_check(sample, gamma, oms, wts, pol, eta, phi_of, prot, nrm):
    """Stationary U0 on the same finite dν. Cheap; no eight-cycle."""
    dnu_map = {round(float(o), 12): float(w) for o, w in zip(oms, wts)}
    bundle = assemble_incoming_bundle(sample["gap"], sample["u"], sample["D"], gamma, oms)
    attach_weights(bundle, dnu_map)
    u0, _ = u0_from_bundle(bundle, pol, phi_of, eta)
    ident = np.eye(2, dtype=complex)
    gram = u0 @ u0.conj().T
    mass = packet_mass_dnu(oms, wts, prot, nrm)
    ccr = maxabs(gram - ident)
    return {
        "n_h2": int(N_H2),
        "n_nodes": int(np.asarray(oms).size),
        "n_photon_columns": int(bundle.n_photon_columns),
        "n_point_columns": int(bundle.n_point_columns),
        "packet_mass_dnu": mass,
        "packet_mass_deficit": float(1.0 - mass),
        "U0_packet_ccr": ccr,
        "U0_gram": pack_c(gram),
        "ok": bool(ccr < 0.05 and abs(mass - 1.0) < 0.05),
        "not_a_new_physical_packet": True,
        "h2_window_density_not_a_second_quadrature_object": True,
        "direct_transmission_port_conversion_unchanged": True,
    }


def u0_from_bundle(bundle, pol, phi_of, eta):
    ph = bundle.photon
    ncol = int(bundle.n_photon_columns + bundle.n_point_columns)
    u0 = np.zeros((2, ncol), complex)
    labels = tuple(ph.labels) + tuple(bundle.point_labels)
    free = np.concatenate(
        (ph.free_A0, np.zeros((2, bundle.n_point_columns), complex)), axis=1
    )
    a_all = np.concatenate((ph.initial.A, bundle.point.A), axis=1)
    for j in range(bundle.n_photon_columns):
        nu = float(ph.labels[j][0])
        a_r = a_r_out_from_column(a_all, free, labels, j)
        u0[:, j] = phi_of(nu) * (pol @ ((np.sqrt(nu) / eta) * a_r))
    return u0, labels


def moments_from(u, v):
    n = v.conj() @ v.T
    m = u @ v.T
    ident = np.eye(u.shape[0], dtype=complex)
    return {
        "U": pack_c(u),
        "V": pack_c(v),
        "n": pack_c(n),
        "m": pack_c(m),
        "m_reference_source": pack_c(np.zeros_like(m)),
        "normal_ccr": maxabs(u @ u.conj().T - v @ v.conj().T - ident),
        "anomalous_ccr": maxabs(u @ v.T - v @ u.T),
        "n_hermiticity": maxabs(n - n.conj().T),
        "n_min_eig": float(np.linalg.eigvalsh(0.5 * (n + n.conj().T))[0]) if n.size else 0.0,
        "m_symmetry": maxabs(m - m.T),
        "renormalized_output": False,
    }


def _checkpoint_moments(u0, snap_dU, snap_v):
    out = {}
    for name, parts in snap_dU.items():
        dUs = np.concatenate(parts, axis=1)
        vs = np.concatenate(snap_v[name], axis=1)
        rec = moments_from(u0 + dUs, vs)
        rec["m_reference_source"] = pack_c(u0 @ vs.T)
        rec["m_source_source"] = pack_c(dUs @ vs.T)
        out[name] = rec
    return out


def run_side(
    geo, sample, gamma, prot_drive, prot_ref, oms, wts, k_scalar, nstep, h, n_pulse, pol, eta, phi_of,
    keep_dark=True, snapshot_steps=None,
):
    dnu_map = {round(float(o), 12): float(w) for o, w in zip(oms, wts)}
    print(f"N={sample['neel']:+d}: cache vertices driven nstep={nstep}", flush=True)
    progress(f"N={sample['neel']:+d}: cache vertices driven nstep={nstep}")
    vu, vd = midpoint_vertices(geo, sample, prot_drive, nstep, h)
    vertices_d = list(zip(vu, vd))
    print(f"N={sample['neel']:+d}: cache vertices undriven", flush=True)
    progress(f"N={sample['neel']:+d}: cache vertices undriven")
    vu0, vd0 = midpoint_vertices(geo, sample, prot_ref, nstep, h)
    vertices_u = list(zip(vu0, vd0))
    u0_parts, dU_parts, v_parts = [], [], []
    undriven_parts, pred_parts = [], []
    tail_d, tail_u, pulse_d, pulse_u = [], [], [], []
    tail_v, pulse_v = [], []
    fac_all = []
    odd_vals, lr_vals = [], []
    n_photon = 0
    n_point = 0
    n_dark = 0
    meta_ok = True
    driven_wall = 0.0
    undriven_wall = 0.0
    final_I = 0.0
    snap_dU = {}
    snap_v = {}
    chunks = list(freq_chunks(oms, wts))
    for ic, (omc, wtc) in enumerate(chunks):
        include_dark = bool(keep_dark) and ic == 0
        bundle = assemble_incoming_bundle(sample["gap"], sample["u"], sample["D"], gamma, omc)
        classified = classify_dark_columns(bundle.dark_info, k_of_index=sample["k_of"])
        fac = attach_weights(bundle, dnu_map)
        fac_all.append(fac)
        n_ph = int(bundle.n_photon_columns)
        n_pt = int(bundle.n_point_columns) if include_dark else 0
        if include_dark:
            n_point = n_pt
            n_dark = int(sum(int(r["nullity"]) for r in classified))
            meta_ok = bool(
                n_point == len(bundle.point_gaps) == len(bundle.point_labels) == n_dark
            )
            initial = packed_state(bundle)
        else:
            initial = bundle.photon.initial.copy()
        n_photon += n_ph
        left, right = photon_port_indices(bundle.photon.labels)
        even_a, odd_a = even_odd_from_lr(bundle.photon.initial.A, left, right)
        odd_vals.append(maxabs(odd_a))
        lr_vals.append(maxabs(bundle.photon.initial.A[:, left] - bundle.photon.initial.A[:, right]))
        u0_chunk, _ = u0_from_bundle(bundle, pol, phi_of, eta)
        if not include_dark:
            u0_chunk = u0_chunk[:, :n_ph]
        print(
            f"N={sample['neel']:+d}: chunk {ic+1}/{len(chunks)} n_ph={n_ph} n_pt={n_pt}",
            flush=True,
        )
        progress(f"N={sample['neel']:+d}: chunk {ic+1}/{len(chunks)} n_ph={n_ph} n_pt={n_pt}")
        acc_d = accumulate_cached(
            bundle.photon, gamma, initial, n_pt, k_scalar, vertices_d, nstep, h, pol, n_pulse,
            snapshot_steps=snapshot_steps,
        )
        acc_u = accumulate_cached(
            bundle.photon, gamma, initial.copy(), n_pt, k_scalar, vertices_u, nstep, h, pol, n_pulse,
            snapshot_steps=snapshot_steps,
        )
        pred = independent_stationary_delta(
            bundle.photon, sample, k_scalar, h, nstep, n_pt, geo
        )
        u0_parts.append(u0_chunk)
        dU_parts.append(acc_d["delta_u"] - acc_u["delta_u"])
        v_parts.append(acc_d["v"] - acc_u["v"])
        undriven_parts.append(acc_u["delta_u"])
        pred_parts.append(pred)
        tail_d.append(acc_d["delta_u_tail"])
        tail_u.append(acc_u["delta_u_tail"])
        pulse_d.append(acc_d["delta_u_pulse"])
        pulse_u.append(acc_u["delta_u_pulse"])
        tail_v.append(acc_d["v_tail"] - acc_u["v_tail"])
        pulse_v.append(acc_d["v_pulse"] - acc_u["v_pulse"])
        driven_wall += acc_d["wall_seconds"]
        undriven_wall += acc_u["wall_seconds"]
        final_I = max(final_I, acc_d["final_I_maxabs"])
        for name, rec in acc_d.get("snapshots", {}).items():
            urec = acc_u["snapshots"][name]
            snap_dU.setdefault(name, []).append(rec["delta_u"] - urec["delta_u"])
            snap_v.setdefault(name, []).append(rec["v"] - urec["v"])
    u0 = np.concatenate(u0_parts, axis=1)
    dU = np.concatenate(dU_parts, axis=1)
    v = np.concatenate(v_parts, axis=1)
    undriven = np.concatenate(undriven_parts, axis=1)
    pred = np.concatenate(pred_parts, axis=1)
    dU_pulse = np.concatenate(pulse_d, axis=1) - np.concatenate(pulse_u, axis=1)
    dU_tail = np.concatenate(tail_d, axis=1) - np.concatenate(tail_u, axis=1)
    v_pulse = np.concatenate(pulse_v, axis=1)
    v_tail = np.concatenate(tail_v, axis=1)
    u = u0 + dU
    mom = moments_from(u, v)
    mom["m_reference_source"] = pack_c(u0 @ v.T)
    mom["m_source_source"] = pack_c(dU @ v.T)
    mom_pulse = moments_from(u0 + dU_pulse, v_pulse)
    mom_pulse["m_reference_source"] = pack_c(u0 @ v_pulse.T)
    mom["two_mode_quad"] = two_mode_quad_check(unpack_c(mom["n"]), unpack_c(mom["m"]))
    ident = np.eye(2, dtype=complex)
    u0_ccr = maxabs(u0 @ u0.conj().T - ident)
    fac_cat = np.concatenate(fac_all) if fac_all else np.ones(1)
    return {
        "neel": int(sample["neel"]),
        "n_el": sample["n_el"],
        "n_photon_columns": int(n_photon),
        "n_point_columns": n_point,
        "n_dark_classified": n_dark,
        "metadata_lengths_agree": bool(meta_ok),
        "n_chunks": int(len(chunks)),
        "weight_once_residual": sample["weight_once_residual"],
        "A_odd_maxabs": float(max(odd_vals) if odd_vals else 0.0),
        "A_L_minus_A_R_maxabs": float(max(lr_vals) if lr_vals else 0.0),
        "sqrt_dnu_min": float(np.min(fac_cat)),
        "sqrt_dnu_max": float(np.max(fac_cat)),
        "U0_packet_ccr": u0_ccr,
        "U0": pack_c(u0),
        "delta_U": pack_c(dU),
        "V": pack_c(v),
        "undriven_source_row_maxabs": maxabs(undriven),
        "undriven_vs_analytic_maxabs": maxabs(undriven - pred),
        "analytic_reference_norm": float(np.linalg.norm(pred)),
        "time_tail_over_pulse_fro": float(
            np.linalg.norm(dU_tail) / max(np.linalg.norm(dU_pulse), 1e-30)
        ),
        "v_tail_over_pulse_fro": float(
            np.linalg.norm(v_tail) / max(np.linalg.norm(v_pulse), 1e-30)
        ),
        "moments_pulse_only": mom_pulse,
        "moments_full_vs_pulse": moments_diff(mom, mom_pulse),
        "zero_pump_delta_U_maxabs": maxabs(dU) if abs(prot_drive.a0_dimless) < 1e-15 else None,
        "zero_pump_structure": {
            "delta_U_is_driven_minus_undriven": True,
            "V_is_driven_minus_undriven": True,
            "n_m_not_manually_zeroed": True,
            "undriven_V_increment_is_zero_by_same_structure": True,
            "zero_pump_identity_is_structural_not_a_second_run": True,
        },
        "moments": mom,
        "checkpoints": _checkpoint_moments(u0, snap_dU, snap_v),
        "driven_wall": driven_wall,
        "undriven_wall": undriven_wall,
        "final_I_driven": final_I,
        "kept_dark": bool(keep_dark and n_point > 0),
        "no_second_bright_vacuum": True,
        "bdag_not_set_to_b_conjugate": True,
    }


def measure_gate(meas, meas_ref, resolvent):
    cover = bool(meas["covers_min_gap"] and meas["covers_max_gap"])
    ccr_n = float(meas["electronic_ccr_normal_maxabs"])
    ccr_a = float(meas["electronic_ccr_anomalous_maxabs"])
    refine = abs(meas["trace_photon_bb"] - meas_ref["trace_photon_bb"]) / max(
        abs(meas_ref["trace_photon_bb"]), 1e-30
    )
    res_max = max(r["dressed_resolvent_vs_bridge"]["max_relative"] for r in resolvent)
    stieltjes_diag = max(r["electronic_stieltjes_vs_diag"]["max_relative"] for r in resolvent)
    ok = bool(
        cover
        and ccr_n < 0.05
        and ccr_a < 0.05
        and res_max < 0.05
        and refine < 0.05
        and int(meas.get("skipped_poles", 0)) == 0
    )
    return {
        "ok": ok,
        "ccr_normal": ccr_n,
        "ccr_anomalous": ccr_a,
        "resolvent_dressed": res_max,
        "stieltjes_vs_bare_diag_not_acceptance": stieltjes_diag,
        "refine": refine,
    }


def side_canonical_ok(side):
    if side is None:
        return False
    mom = side["moments"]
    return bool(
        mom["normal_ccr"] < 0.05
        and mom["anomalous_ccr"] < 0.05
        and mom["n_min_eig"] >= -1e-8
        and not mom["renormalized_output"]
    )


def flags_from(conv, meas, meas_ref, resolvent, plus, minus, kernel, gamma, extra=None):
    conv_ok = conv["i_2pi_eta_I_over_sqrtw_vs_sqrtw_AminusAf_over_eta_maxabs"] < 1e-12
    gate = measure_gate(meas, meas_ref, resolvent)
    extra = extra or {}
    out_p = None if plus is None else float(plus["moments"]["normal_ccr"])
    out_m = None if minus is None else float(minus["moments"]["normal_ccr"])
    nmin_p = None if plus is None else float(plus["moments"]["n_min_eig"])
    nmin_m = None if minus is None else float(minus["moments"]["n_min_eig"])
    plus_ok = side_canonical_ok(plus)
    minus_ok = side_canonical_ok(minus)
    dt_ok = bool(extra.get("dt_nm_ok", False))
    quad_ok = bool(extra.get("quad_nm_ok", False))
    time_checked = bool(extra.get("full_duration_time_checked", False))
    finite_ok = bool(
        gate["ok"]
        and plus_ok
        and dt_ok
        and quad_ok
        and extra.get("u0_ok", False)
        and not extra.get("renormalized", False)
    )
    return {
        "run_completed": True,
        "time_error_checked": time_checked,
        "time_error_small": dt_ok,
        "short_prefix_not_full_duration": True,
        "full_duration_time_qualified": dt_ok,
        "input_measure_complete": bool(gate["ok"]),
        "old_bb_only_flag_not_reused": True,
        "bare_diag_stieltjes_not_acceptance": True,
        "canonical_output_checked": plus_ok,
        "kernel_and_uv_wiring": True,
        "finite_nm_candidate_computed": plus is not None,
        "finite_nm_qualified": finite_ok and (minus_ok if minus is not None else False),
        "plus_only_this_round": minus is None,
        "plus_nm_qualified": bool(extra.get("plus_nm_qualified", False)),
        "kernel_prefactor_is_eta": bool(kernel["prefactor_is_eta_not_sqrt_gamma"]),
        "independent_field_conversion_ok": conv_ok,
        "gamma_matches_frozen": bool(abs(gamma - FROZEN_GAMMA) < 1e-12),
        "plus_keeps_dark": None if plus is None else bool(plus["n_point_columns"] > 0),
        "minus_keeps_dark": None if minus is None else bool(minus["n_point_columns"] > 0),
        "plus_metadata_agree": None if plus is None else bool(plus["metadata_lengths_agree"]),
        "minus_metadata_agree": None if minus is None else bool(minus["metadata_lengths_agree"]),
        "electronic_ccr_normal_maxabs": gate["ccr_normal"],
        "electronic_ccr_anomalous_maxabs": gate["ccr_anomalous"],
        "electronic_ccr_maxabs_incomplete_bb_only": float(
            meas["electronic_ccr_maxabs_incomplete_bb_only"]
        ),
        "measure_refine_rel_trace": float(gate["refine"]),
        "resolvent_dressed_max_relative": float(gate["resolvent_dressed"]),
        "stieltjes_vs_bare_diag_max_relative": float(gate["stieltjes_vs_bare_diag_not_acceptance"]),
        "plus_output_normal_ccr": out_p,
        "minus_output_normal_ccr": out_m,
        "plus_n_min_eig": nmin_p,
        "minus_n_min_eig": nmin_m,
        "dt_nm_frobenius_relative": extra.get("dt_nm_rel"),
        "quad_nm_frobenius_relative": extra.get("quad_nm_rel"),
        "old_json_not_overwritten": True,
        "chi_rad_placeholder_not_flipped": True,
        "two_k_not_a_domain_signal": True,
        "not_production_domain_signal": True,
        "n_m_are_finite_reference_only": True,
        "domain_signal": False,
        "squeezing": False,
        "entanglement": False,
    }


def endpoint_estimate(sample, gamma, oms, wts):
    """Finite extra IR/UV panels; not a hard physical cutoff."""
    lo, hi = float(oms[0]), float(oms[-1])
    extra_o, extra_w = [], []
    o1, w1 = geom_segment(1.0e-4, lo, 8)
    o2, w2 = trap_segment(hi, 24.0, 24)
    if o1.size:
        extra_o.append(o1)
        extra_w.append(w1)
    if o2.size:
        extra_o.append(o2)
        extra_w.append(w2)
    if not extra_o:
        return {"ok": True, "no_extra": True}
    xo = np.concatenate(extra_o)
    xw = np.concatenate(extra_w)
    meas_x = electronic_measure_report(sample, gamma, xo, xw)
    return {
        "omega_lo_extend": 1.0e-4,
        "omega_hi_extend": 24.0,
        "n_extra": int(xo.size),
        "extra_trace_bb": float(meas_x["trace_photon_bb"]),
        "extra_ccr_normal_maxabs": float(meas_x["electronic_ccr_normal_maxabs"]),
        "extra_ccr_anomalous_maxabs": float(meas_x["electronic_ccr_anomalous_maxabs"]),
        "base_omega_min": lo,
        "base_omega_max": hi,
        "not_a_physical_hard_cutoff": True,
        "extra_only_ccr_is_not_a_gate": True,
        "ok": bool(abs(meas_x["trace_photon_bb"]) < 0.05),
    }


def slim_moments(side):
    mom = side["moments"]
    return {
        "n": mom["n"],
        "m": mom["m"],
        "m_reference_source": mom["m_reference_source"],
        "m_source_source": mom.get("m_source_source"),
        "normal_ccr": mom["normal_ccr"],
        "anomalous_ccr": mom["anomalous_ccr"],
        "n_min_eig": mom["n_min_eig"],
        "two_mode_quad": mom.get("two_mode_quad"),
        "n_trace": mom["n"].get("trace_real"),
        "m_maxabs": mom["m"].get("maxabs"),
        "time_tail_over_pulse_fro": side.get("time_tail_over_pulse_fro"),
        "v_tail_over_pulse_fro": side.get("v_tail_over_pulse_fro"),
        "moments_full_vs_pulse": side.get("moments_full_vs_pulse"),
        "U0_packet_ccr": side.get("U0_packet_ccr"),
        "n_photon_columns": side.get("n_photon_columns"),
        "n_point_columns": side.get("n_point_columns"),
        "undriven_vs_analytic_maxabs": side.get("undriven_vs_analytic_maxabs"),
        "driven_wall": side.get("driven_wall"),
        "undriven_wall": side.get("undriven_wall"),
    }


def write_payload(payload, name):
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / name
    if out_json.exists():
        raise FileExistsError(f"Refuse overwrite of {name}")
    for path in (OLD_CONTRACT, CALLER_FIX, INPUT_LEDGER, EIGHT_CYCLE, OLD_INPUT_CLOSE, OLD_U0_H2):
        if not path.exists():
            raise FileNotFoundError(path)
    for rel in OLD_JSON_KEEP:
        if not (REPO / rel).exists():
            raise FileNotFoundError(f"Protected JSON missing: {rel}")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_json


def main() -> int:
    t0 = time.perf_counter()
    stage = str(STAGE).strip().lower()
    geo = build_geometry()
    prot_drive = p1_a2_protocol(A_STAR)
    prot_ref = p1_a2_protocol(0.0)
    gamma = float(gamma_self(geo))
    eta = eta_of(gamma)
    h = prot_drive.period / NTPC
    n_pulse = int(prot_drive.n_cycles) * NTPC
    nstep = n_pulse + N_TAIL_CYCLES * NTPC
    pol = np.vstack(parallel_perp_basis(geo))
    print(f"stage={stage} gamma={gamma:.12g} eta={eta:.12g} nstep={nstep}", flush=True)
    progress(f"start stage={stage} gamma={gamma:.12g} eta={eta:.12g} nstep={nstep}")
    sample = m0_sample(geo, TWO_K, WEIGHTS, +1)
    conv = independent_field_conversion(gamma, sample, 0.1)
    print("field conversion", conv["i_2pi_eta_I_over_sqrtw_vs_sqrtw_AminusAf_over_eta_maxabs"], flush=True)
    progress("field conversion done")
    oms, wts = input_quadrature(sample["gap"], sample["u"], gamma, n_panel=N_PANEL, n_h2=N_H2)
    oms_fine, wts_fine = input_quadrature(
        sample["gap"], sample["u"], gamma, n_panel=N_PANEL_FINE, n_h2=N_H2_FINE
    )
    meas = electronic_measure_report(sample, gamma, oms, wts)
    meas_fine = electronic_measure_report(sample, gamma, oms_fine, wts_fine)
    n_panel_used = int(N_PANEL)
    print(
        f"nodes={meas['n_nodes']} CCR_n={meas['electronic_ccr_normal_maxabs']:.4g} "
        f"CCR_a={meas['electronic_ccr_anomalous_maxabs']:.4g} "
        f"incomplete={meas['electronic_ccr_maxabs_incomplete_bb_only']:.4g} "
        f"tr={meas['trace_photon_bb']:.4g} tr_dd={meas['trace_dd']:.4g}",
        flush=True,
    )
    resolvent = resolvent_from_measure(sample, gamma, oms, wts, COMPLEX_Z)
    resolvent_fine = resolvent_from_measure(sample, gamma, oms_fine, wts_fine, COMPLEX_Z)
    ends = endpoint_estimate(sample, gamma, oms, wts)
    gate = measure_gate(meas, meas_fine, resolvent)
    gate_fine = measure_gate(meas_fine, meas, resolvent_fine)
    print(
        f"measure gate ok={gate['ok']} CCR_n={gate['ccr_normal']:.4g} "
        f"CCR_a={gate['ccr_anomalous']:.4g} dressed={gate['resolvent_dressed']:.4g} "
        f"bare_diag={gate['stieltjes_vs_bare_diag_not_acceptance']:.4g} refine={gate['refine']:.4g}",
        flush=True,
    )
    progress(
        f"measure gate ok={gate['ok']} CCR_n={gate['ccr_normal']:.4g} "
        f"dressed={gate['resolvent_dressed']:.4g}"
    )
    progress(f"kernel nstep={nstep}")
    kernel = detector_kernel_scalar(gamma, prot_drive, nstep, h)
    progress("kernel done")
    k_scalar = kernel.pop("k_scalar")
    kernel.pop("k_pulled_same_phi")
    kernel.pop("t_mid")
    kernel.pop("phi_omega")
    kernel.pop("phi")
    nrm = kernel["normalizer"]

    def phi_of(nu):
        return complex(np.asarray(phi_at(nu, prot_drive, nrm)).reshape(-1)[0])

    u0_chk = u0_packet_check(sample, gamma, oms, wts, pol, eta, phi_of, prot_drive, nrm)
    u0_fine = u0_packet_check(sample, gamma, oms_fine, wts_fine, pol, eta, phi_of, prot_drive, nrm)
    print(
        f"U0 packet mass={u0_chk['packet_mass_dnu']:.6g} CCR={u0_chk['U0_packet_ccr']:.4g} ok={u0_chk['ok']}",
        flush=True,
    )
    progress(
        f"U0 packet mass={u0_chk['packet_mass_dnu']:.6g} CCR={u0_chk['U0_packet_ccr']:.4g} ok={u0_chk['ok']}"
    )

    plus = None
    plus_dt = None
    plus_quad = None
    minus = None
    dt_cmp = None
    quad_cmp = None
    extra = {"u0_ok": bool(u0_chk["ok"]), "renormalized": False}
    stop_reason = None
    if not gate["ok"]:
        stop_reason = "CCR/dressed-resolvent/refine gate failed"
        print(f"STOP: {stop_reason}; skip eight-cycle and n,m", flush=True)
        progress(f"STOP {stop_reason}")
    elif not u0_chk["ok"]:
        stop_reason = "U0 packet CCR/mass failed"
        print(f"STOP: {stop_reason}; skip eight-cycle", flush=True)
        progress(f"STOP {stop_reason}")
    elif stage == "probe":
        stop_reason = "probe_only"
        progress("probe stage: skip +N")
    elif stage in ("plus", "minus"):
        progress("+N base ntpc=1024")
        plus = run_side(
            geo, sample, gamma, prot_drive, prot_ref, oms, wts, k_scalar, nstep, h, n_pulse, pol, eta, phi_of
        )
        if not side_canonical_ok(plus):
            stop_reason = "+N canonical CCR failed"
            print(f"STOP: {stop_reason}; skip dt/quad/-N", flush=True)
            progress(f"STOP {stop_reason}")
        else:
            h2 = prot_drive.period / NTPC_FINE
            n_pulse2 = int(prot_drive.n_cycles) * NTPC_FINE
            nstep2 = n_pulse2 + N_TAIL_CYCLES * NTPC_FINE
            progress(f"+N dt-halving full duration nstep={nstep2}")
            kernel2 = detector_kernel_scalar(gamma, prot_drive, nstep2, h2)
            k2 = kernel2["k_scalar"]
            plus_dt = run_side(
                geo, sample, gamma, prot_drive, prot_ref, oms, wts, k2, nstep2, h2, n_pulse2, pol, eta, phi_of
            )
            dt_cmp = moments_diff(plus["moments"], plus_dt["moments"])
            extra["full_duration_time_checked"] = True
            extra["dt_nm_ok"] = bool(dt_cmp["ok"] and side_canonical_ok(plus_dt))
            extra["dt_nm_rel"] = {
                "n": dt_cmp["n"]["frobenius_relative"],
                "m": dt_cmp["m"]["frobenius_relative"],
            }
            progress(
                f"dt n rel={dt_cmp['n']['frobenius_relative']:.4g} "
                f"m rel={dt_cmp['m']['frobenius_relative']:.4g} ok={dt_cmp['ok']}"
            )
            progress("+N input quadrature refine")
            kernel_q = detector_kernel_scalar(gamma, prot_drive, nstep, h)
            plus_quad = run_side(
                geo,
                sample,
                gamma,
                prot_drive,
                prot_ref,
                oms_fine,
                wts_fine,
                kernel_q["k_scalar"],
                nstep,
                h,
                n_pulse,
                pol,
                eta,
                phi_of,
            )
            quad_cmp = moments_diff(plus["moments"], plus_quad["moments"])
            extra["quad_nm_ok"] = bool(quad_cmp["ok"] and side_canonical_ok(plus_quad))
            extra["quad_nm_rel"] = {
                "n": quad_cmp["n"]["frobenius_relative"],
                "m": quad_cmp["m"]["frobenius_relative"],
            }
            progress(
                f"quad n rel={quad_cmp['n']['frobenius_relative']:.4g} "
                f"m rel={quad_cmp['m']['frobenius_relative']:.4g} ok={quad_cmp['ok']}"
            )
            plus_pass = bool(
                extra["dt_nm_ok"]
                and extra["quad_nm_ok"]
                and side_canonical_ok(plus)
            )
            extra["plus_nm_qualified"] = plus_pass
            extra["endpoint_ok"] = bool(ends["ok"])
            if stage == "minus":
                if not plus_pass:
                    stop_reason = "+N qualification failed; skip -N"
                    print(f"STOP: {stop_reason}", flush=True)
                    progress(f"STOP {stop_reason}")
                else:
                    progress("+N qualified; start -N at base settings")
                    sample_m = m0_sample(geo, TWO_K, WEIGHTS, -1)
                    minus = run_side(
                        geo,
                        sample_m,
                        gamma,
                        prot_drive,
                        prot_ref,
                        oms,
                        wts,
                        k_scalar,
                        nstep,
                        h,
                        n_pulse,
                        pol,
                        eta,
                        phi_of,
                    )
    elif stage == "minus_only":
        prev_path = OUT / "nm_qualify_plus.json"
        if not prev_path.exists():
            stop_reason = "minus_only needs nm_qualify_plus.json"
            print(f"STOP: {stop_reason}", flush=True)
            progress(f"STOP {stop_reason}")
        else:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            pf = prev.get("pass_flags", {})
            if not pf.get("plus_nm_qualified"):
                stop_reason = "plus_nm_qualified is false; skip -N"
                print(f"STOP: {stop_reason}", flush=True)
                progress(f"STOP {stop_reason}")
            else:
                pm = dict(prev["plus"])
                pm["renormalized_output"] = False
                plus = {
                    "moments": pm,
                    "n_point_columns": int(pm.get("n_point_columns") or 0),
                    "metadata_lengths_agree": True,
                    "U0": prev.get("plus_U0"),
                    "delta_U": prev.get("plus_delta_U"),
                    "V": prev.get("plus_V"),
                }
                extra["u0_ok"] = True
                extra["full_duration_time_checked"] = True
                extra["dt_nm_ok"] = True
                extra["quad_nm_ok"] = True
                extra["plus_nm_qualified"] = True
                extra["dt_nm_rel"] = pf.get("dt_nm_frobenius_relative")
                extra["quad_nm_rel"] = pf.get("quad_nm_frobenius_relative")
                extra["loaded_plus_from"] = "nm_qualify_plus.json"
                dt_cmp = prev.get("dt_compare")
                quad_cmp = prev.get("quad_compare")
                progress("-N only at final base settings")
                sample_m = m0_sample(geo, TWO_K, WEIGHTS, -1)
                minus = run_side(
                    geo,
                    sample_m,
                    gamma,
                    prot_drive,
                    prot_ref,
                    oms,
                    wts,
                    k_scalar,
                    nstep,
                    h,
                    n_pulse,
                    pol,
                    eta,
                    phi_of,
                )
    else:
        stop_reason = f"unknown stage {stage}"
        print(f"STOP: {stop_reason}", flush=True)

    flags = flags_from(conv, meas, meas_fine, resolvent, plus, minus, kernel, gamma, extra)
    flags["u0_packet_ok"] = bool(u0_chk["ok"])
    flags["u0_packet_ccr"] = float(u0_chk["U0_packet_ccr"])
    flags["packet_mass_dnu"] = float(u0_chk["packet_mass_dnu"])
    flags["stage"] = stage
    flags["stop_reason"] = stop_reason
    flags["endpoint_ok"] = bool(ends.get("ok", False))
    if plus is not None and minus is not None:
        flags["plus_minus_n_m"] = moments_diff(plus["moments"], minus["moments"])
        flags["canonical_output_checked"] = bool(
            side_canonical_ok(plus) and side_canonical_ok(minus)
        )
        flags["two_k_not_a_domain_signal"] = True
    plus_out = None
    if plus is not None:
        plus_out = plus["moments"] if stage == "minus_only" else slim_moments(plus)
    dump_plus_uv = bool(plus is not None and stage != "minus_only")
    payload = {
        "status": "p1_h2_finite_kcv_nm_qualify_plus_not_domain",
        "stage": stage,
        "run_completed": flags["run_completed"],
        "time_error_checked": flags["time_error_checked"],
        "input_measure_complete": flags["input_measure_complete"],
        "canonical_output_checked": flags["canonical_output_checked"],
        "finite_nm_qualified": flags["finite_nm_qualified"],
        "pass_flags": flags,
        "scope": {
            "same_two_k": True,
            "keep_dark_point_columns": True,
            "no_second_bright_electronic_vacuum": True,
            "plus_only_until_qualified": True,
            "eta": eta,
            "prefactor": "i eta int dω φ/√ω e^{+iωt}",
            "weight_convention": "sqrt(dν) into photon columns, dν counted once",
            "n_h2": int(N_H2),
            "n_h2_fine": int(N_H2_FINE),
            "ntpc": NTPC,
            "ntpc_fine": NTPC_FINE,
            "n_steps": nstep,
            "n_pulse_steps": n_pulse,
            "n_input_frequencies": int(oms.size),
            "n_input_frequencies_fine": int(oms_fine.size),
            "not_production_domain_signal": True,
            "two_k_not_a_domain_signal": True,
            "chi_rad_placeholder_status": CHI_STATUS,
            "do_not_flip_chi_rad_flag": True,
            "h2_window_density_not_a_second_quadrature_object": True,
        },
        "u0_packet": u0_chk,
        "u0_packet_fine": u0_fine,
        "detector_kernel": kernel,
        "independent_field_conversion": conv,
        "input_measure": meas,
        "input_measure_refined_panel": meas_fine,
        "n_panel_used": n_panel_used,
        "resolvent_complex": resolvent,
        "resolvent_complex_fine": resolvent_fine,
        "endpoint_estimate": ends,
        "measure_gate": gate,
        "gamma_self": gamma,
        "plus": plus_out,
        "plus_dt": None if plus_dt is None else slim_moments(plus_dt),
        "plus_quad": None if plus_quad is None else slim_moments(plus_quad),
        "dt_compare": dt_cmp,
        "quad_compare": quad_cmp,
        "minus": None if minus is None else slim_moments(minus),
        "plus_minus_compare": flags.get("plus_minus_n_m"),
        "plus_U": None if not dump_plus_uv else plus["moments"].get("U"),
        "plus_U0": None if not dump_plus_uv else plus.get("U0"),
        "plus_delta_U": None if not dump_plus_uv else plus.get("delta_U"),
        "plus_V": None if not dump_plus_uv else plus.get("V"),
        "old_json_not_overwritten": OLD_JSON_KEEP,
        "elapsed_s": time.perf_counter() - t0,
        "source_sha256": {
            str(p.relative_to(REPO)): sha(p)
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_p1_h2_finite_kcv_caller_fix_20261005.py",
                REPO / "tools/b0_small_model/q8_m0_p1_h2_finite_kcv_eight_cycle_20261005.py",
                REPO / "tools/b0_small_model/q8_m0_continuum_columns_20261004.py",
                REPO / "tools/b0_small_model/q8_m0_incoming_spectral_completeness_20261004.py",
                REPO / "tools/b0_small_model/q8_m0_gsheet.py",
            )
        },
        "eight_cycle_sha256": sha(EIGHT_CYCLE) if EIGHT_CYCLE.exists() else None,
        "old_input_close_sha256": sha(OLD_INPUT_CLOSE) if OLD_INPUT_CLOSE.exists() else None,
        "old_u0_h2_sha256": sha(OLD_U0_H2) if OLD_U0_H2.exists() else None,
        "production": False,
        "n_m_claimed_as_domain_signal": False,
        "gaussian_scope": "finite two-k linear bosonic contraction only",
        "stop_reason": stop_reason,
    }
    name = {
        "probe": "nm_qualify_probe.json",
        "plus": "nm_qualify_plus.json",
        "minus": "nm_qualify.json",
        "minus_only": "nm_qualify.json",
    }.get(stage, f"nm_qualify_{stage}.json")
    out_json = write_payload(payload, name)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "stage": stage,
                "run_completed": flags["run_completed"],
                "input_measure_complete": flags["input_measure_complete"],
                "canonical_output_checked": flags["canonical_output_checked"],
                "finite_nm_qualified": flags["finite_nm_qualified"],
                "eta": eta,
                "n_nodes": meas["n_nodes"],
                "electronic_ccr_normal": meas["electronic_ccr_normal_maxabs"],
                "electronic_ccr_anomalous": meas["electronic_ccr_anomalous_maxabs"],
                "dressed_resolvent": gate["resolvent_dressed"],
                "plus_n_trace": None if plus is None else plus["moments"]["n"].get("trace_real"),
                "plus_normal_ccr": None if plus is None else plus["moments"]["normal_ccr"],
                "dt_ok": extra.get("dt_nm_ok"),
                "quad_ok": extra.get("quad_nm_ok"),
                "u0_packet_ccr": u0_chk["U0_packet_ccr"],
                "stop_reason": stop_reason,
                "elapsed_s": payload["elapsed_s"],
                "out": str(out_json),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





