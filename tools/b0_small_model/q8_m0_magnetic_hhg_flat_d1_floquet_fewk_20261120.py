#!/usr/bin/env python3
"""Few-k Floquet occupation / coherence test on the frozen D1 flat-top drive.

Goal: not merely to find nonzero Floquet coherence, but to test whether the
off-diagonal Floquet block of rho quantitatively explains the non-stationary
magnetization-odd current seen in D1.

Does not change the Hamiltonian, does not introduce T2, and does not start
D2, Gamma192, squeezing, QED, G1/H2, or any server job.
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q8_m0_classical as cl
import q8_m0_magnetic_hhg_flat_d1_20261119 as d1
import q8_m0_magnetic_hhg_flat_d1_floquet_diag_20261120 as diag
import q8_m0_magnetic_hhg_source_20261114 as src
from q8_m0_model import (
    R0,
    build_geometry,
    current_ops,
    h8,
    op_P,
    op_sewing,
    sha256_file,
)
from q8_m0_s1_kernel import parallel_perp_basis

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/magnetic_hhg_flat_d1_floquet_fewk_20261120"
MU = d1.MU
NK_GRID = 48
NTPC = d1.NTPC
BUDGET_S = 120.0
SIGN = +1
T0_CYCLE = 3  # main Floquet stroboscopic origin = start of D1 rect6 window
HARD_DEGEN_ABS = 1e-10
NEAR_DEGEN_FRAC = 1e-4
SQRT2 = math.sqrt(2.0)

# Deterministic Gamma48 fractional indices (indexing='ij'): linear = i*nk + j.
# Two inversion-fixed points, one K/K' pair, two generic interior pairs.
K_SPEC = (
    {"name": "Gamma", "i": 0, "j": 0, "role": "inversion_fixed", "partner": "Gamma"},
    {"name": "M_b1", "i": 24, "j": 0, "role": "inversion_fixed_M", "partner": "M_b1"},
    {"name": "K", "i": 16, "j": 32, "role": "K_point", "partner": "Kp"},
    {"name": "Kp", "i": 32, "j": 16, "role": "K_prime", "partner": "K"},
    {"name": "G1", "i": 5, "j": 11, "role": "generic_interior", "partner": "G1p"},
    {"name": "G1p", "i": 43, "j": 37, "role": "generic_partner", "partner": "G1"},
    {"name": "G2", "i": 7, "j": 20, "role": "off_high_symmetry", "partner": "G2p"},
    {"name": "G2p", "i": 41, "j": 28, "role": "off_high_symmetry_partner", "partner": "G2"},
)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump(path, obj):
    d1.dump(path, obj)


def wrap_quasi(eps, w0):
    return (eps + 0.5 * w0) % w0 - 0.5 * w0


def circ_diff(e1, e2, w0):
    return wrap_quasi(e1 - e2, w0)


def fro_offdiag(mat):
    m = np.asarray(mat, dtype=complex).copy()
    np.fill_diagonal(m, 0.0)
    return float(np.linalg.norm(m))


def fro(mat):
    return float(np.linalg.norm(np.asarray(mat, dtype=complex)))


def maxabs(a):
    return float(np.max(np.abs(a))) if np.size(a) else 0.0


def midpoint_u(geo, p, kcart, neel, t, dt, prot, sign=SIGN):
    kmid = kcart + cl.kappa_shift(t + 0.5 * dt, geo, prot, sign)
    hm = h8(geo, p, kmid, neel)
    em, wm = np.linalg.eigh(hm)
    ph = np.exp(-1j * em * dt)
    return (wm * ph[:, None, :]) @ np.conj(np.swapaxes(wm, 1, 2))


def apply_u(u, rho):
    return u @ rho @ np.conj(np.swapaxes(u, 1, 2))


def sheet_current(rho, geo, p, kcart, neel, t, prot, sign=SIGN):
    kap = kcart + cl.kappa_shift(t, geo, prot, sign)
    jk = cl.trace_current(rho, current_ops(geo, p, kap, neel))
    return jk / geo.a_cell_ang2


def select_kpoints(geo, nk=NK_GRID):
    kf_all, kc_all, w_all = cl.k_grid_xy(geo, nk, 0.0, 0.0)
    recip = geo.recip()
    names = [s["name"] for s in K_SPEC]
    rows = []
    kf = np.zeros((len(K_SPEC), 2), dtype=float)
    kc = np.zeros((len(K_SPEC), 2), dtype=float)
    lin = np.zeros(len(K_SPEC), dtype=int)
    for n, spec in enumerate(K_SPEC):
        i, j = int(spec["i"]), int(spec["j"])
        idx = i * nk + j
        frac = np.array([(i + 0.0) / nk, (j + 0.0) / nk], dtype=float)
        cart = frac @ recip
        if maxabs(kf_all[idx] - frac) > 1e-14 or maxabs(kc_all[idx] - cart) > 1e-12:
            raise RuntimeError(f"k-grid mismatch at {spec['name']}: lin={idx}")
        inv_i = (nk - i) % nk
        inv_j = (nk - j) % nk
        partner_name = spec["partner"]
        partner_spec = next(s for s in K_SPEC if s["name"] == partner_name)
        if partner_spec["i"] != inv_i or partner_spec["j"] != inv_j:
            raise RuntimeError(f"inversion partner mismatch for {spec['name']}")
        rows.append(
            {
                "name": spec["name"],
                "role": spec["role"],
                "partner": partner_name,
                "i": i,
                "j": j,
                "linear_index": int(idx),
                "kf": [float(frac[0]), float(frac[1])],
                "kc": [float(cart[0]), float(cart[1])],
                "weight_if_full_grid": float(w_all[idx]),
                "inversion_fixed": bool(i == inv_i and j == inv_j),
            }
        )
        kf[n] = frac
        kc[n] = cart
        lin[n] = idx
    rationale = {
        "grid": "Gamma48 Monkhorst-Pack, k_grid_xy indexing=ij, shift=(0,0)",
        "n_points": 8,
        "not_only_high_symmetry": True,
        "contents": "2 inversion-fixed (Gamma, M along b1) + K/K' pair + two generic interior pairs",
        "linear_index": "i*nk + j",
        "inversion": "partner of (i,j) is ((nk-i)%nk, (nk-j)%nk) in fractional coordinates",
        "cannot_represent_BZ_sum": True,
    }
    return {"rows": rows, "names": names, "kf": kf, "kc": kc, "lin": lin, "rationale": rationale}


def union_clusters(eps, w0, thresh):
    n = len(eps)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    for i in range(n):
        for j in range(i + 1, n):
            if abs(circ_diff(eps[i], eps[j], w0)) < thresh:
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(v) for v in groups.values()]


def orthonormalize_columns(a):
    a = np.asarray(a, dtype=complex)
    if a.ndim != 2 or a.shape[1] == 0:
        return a
    q, r = np.linalg.qr(a, mode="reduced")
    diag = np.diag(r)
    ph = np.ones(diag.shape[0], dtype=complex)
    nz = np.abs(diag) > 0
    ph[nz] = diag[nz] / np.abs(diag[nz])
    return q * np.conj(ph)


def cluster_project_rho(phi, rho, clusters):
    """Block-diagonalize rho in the clustered Floquet subspaces."""
    rho_diag = np.zeros_like(rho)
    for clu in clusters:
        idx = np.array(clu, dtype=int)
        phi_c = orthonormalize_columns(phi[:, idx])
        proj = phi_c @ np.conj(phi_c.T)
        rho_diag += proj @ rho @ proj
    return rho_diag, rho - rho_diag


def eig_unitary(u):
    lam, vec = np.linalg.eig(u)
    # column-normalize
    nrm = np.linalg.norm(vec, axis=0, keepdims=True)
    nrm = np.where(nrm == 0, 1.0, nrm)
    vec = vec / nrm
    rec = maxabs(u @ vec - vec * lam)
    return lam, vec, rec


def analyze_uf(u, rho, prot, t0_period):
    w0 = float(prot.omega0)
    unit = maxabs(np.conj(u.T) @ u - np.eye(8))
    det_abs = float(abs(np.linalg.det(u)))
    lam, phi, rec = eig_unitary(u)
    eps = wrap_quasi(-np.angle(lam) / t0_period, w0)
    order = np.argsort(eps)
    lam = lam[order]
    phi = phi[:, order]
    eps = eps[order]
    rho_f = np.conj(phi.T) @ rho @ phi
    hard_th = HARD_DEGEN_ABS / t0_period
    near_th = NEAR_DEGEN_FRAC * w0
    hard_cl = union_clusters(eps, w0, hard_th)
    near_cl = union_clusters(eps, w0, near_th)
    rho_diag_h, rho_coh_h = cluster_project_rho(phi, rho, hard_cl)
    rho_diag_n, rho_coh_n = cluster_project_rho(phi, rho, near_cl)
    rho_f_h = np.conj(phi.T) @ rho_diag_h @ phi
    rho_f_n = np.conj(phi.T) @ rho_diag_n @ phi
    occ = np.real(np.diag(rho_f))
    return {
        "unitarity_maxabs": unit,
        "det_abs": det_abs,
        "eig_reconstruct_maxabs": rec,
        "lam": lam,
        "phi": phi,
        "eps": eps,
        "rho_f": rho_f,
        "occ": occ,
        "tr_rho": float(np.real(np.trace(rho))),
        "tr_rho_f": float(np.real(np.trace(rho_f))),
        "purity": float(np.real(np.trace(rho @ rho))),
        "offdiag_ungrouped": fro_offdiag(rho_f),
        "fro_rho_f": fro(rho_f),
        "offdiag_frac_ungrouped": (fro_offdiag(rho_f) / fro(rho_f) if fro(rho_f) else None),
        "hard_clusters": hard_cl,
        "near_clusters": near_cl,
        "n_hard_mult": int(sum(len(c) > 1 for c in hard_cl)),
        "n_near_mult": int(sum(len(c) > 1 for c in near_cl)),
        "offdiag_frac_hard": (fro_offdiag(rho_f_h) / fro(rho_f) if fro(rho_f) else None),
        "offdiag_frac_near": (fro_offdiag(np.conj(phi.T) @ rho_coh_n @ phi) / fro(rho_f) if fro(rho_f) else None),
        "coh_fro_near": fro(rho_coh_n),
        "diag_fro_near": fro(rho_diag_n),
        "rho_diag_near": rho_diag_n,
        "rho_coh_near": rho_coh_n,
        "rho_diag_hard": rho_diag_h,
        "rho_coh_hard": rho_coh_h,
        "hard_thresh_eps": hard_th,
        "near_thresh_eps": near_th,
    }


def build_period_u(geo, p, kcart, neel, t0, dt, prot, n_steps):
    nk = kcart.shape[0]
    eye = np.repeat(np.eye(8, dtype=complex)[None, :, :], nk, axis=0)
    u_full = eye.copy()
    u_early = eye.copy()
    u_late = eye.copy()
    n_half = n_steps // 2
    t = float(t0)
    for i in range(n_steps):
        u = midpoint_u(geo, p, kcart, neel, t, dt, prot)
        u_full = u @ u_full
        if i < n_half:
            u_early = u @ u_early
        else:
            u_late = u @ u_late
        t += dt
    return u_full, u_early, u_late


def propagate_selected(geo, p, kcart, neel, prot, need_half=True):
    rho, e0 = cl.initial_rho(geo, p, kcart, neel, MU)
    nt = prot.n_cycles * NTPC
    dt = prot.period / NTPC
    times = np.arange(nt + 1) * dt
    nk = kcart.shape[0]
    j_t = np.zeros((nt + 1, nk, 2), dtype=float)
    strobe = {}
    half = {}
    t0_idx = T0_CYCLE * NTPC
    half_idx = t0_idx + NTPC // 2
    for n in range(nt + 1):
        t = times[n]
        j_t[n] = sheet_current(rho, geo, p, kcart, neel, t, prot)
        if n % NTPC == 0:
            strobe[n // NTPC] = rho.copy()
        if need_half and n in (t0_idx, half_idx, t0_idx + NTPC):
            half[n] = rho.copy()
        if n == nt:
            break
        rho = apply_u(midpoint_u(geo, p, kcart, neel, t, dt, prot), rho)
    return {
        "times": times,
        "dt": dt,
        "J_sheet": j_t,
        "strobe": strobe,
        "half": half,
        "e0": e0,
    }


def propagate_from(rho0, geo, p, kcart, neel, prot, n0, n1):
    dt = prot.period / NTPC
    times = np.arange(n0, n1 + 1) * dt
    rho = rho0.copy()
    nk = kcart.shape[0]
    j_t = np.zeros((n1 - n0 + 1, nk, 2), dtype=float)
    for i, n in enumerate(range(n0, n1 + 1)):
        t = n * dt
        j_t[i] = sheet_current(rho, geo, p, kcart, neel, t, prot)
        if n == n1:
            break
        rho = apply_u(midpoint_u(geo, p, kcart, neel, t, dt, prot), rho)
    return times, j_t


def sewing_g(kc, kc_partner):
    g = np.asarray(kc_partner, dtype=float) + np.asarray(kc, dtype=float)
    return g


def hamiltonian_inversion_check(geo, p, kc, names):
    pmat = op_P()
    rows = []
    name_to_i = {n: i for i, n in enumerate(names)}
    for spec in K_SPEC:
        ik = name_to_i[spec["name"]]
        ip = name_to_i[spec["partner"]]
        k = kc[ik]
        kp = kc[ip]
        g = sewing_g(k, kp)
        hn = h8(geo, p, k[None, :], +1)[0]
        h_minus_cart = h8(geo, p, (-k)[None, :], -1)[0]
        mapped = pmat @ hn @ np.conj(pmat.T)
        err_cart = maxabs(mapped - h_minus_cart)
        h_partner = h8(geo, p, kp[None, :], -1)[0]
        if np.linalg.norm(g) > 1e-10:
            ug = op_sewing(geo, g)
            # H(k+G)=U_G^T H(k) U_G with k=-k_orig, k+G=k_partner
            h_from_partner = ug @ h_partner @ np.conj(ug.T)
        else:
            ug = np.eye(8, dtype=complex)
            h_from_partner = h_partner
        err_sewn = maxabs(mapped - h_from_partner)
        jn = current_ops(geo, p, k[None, :], +1)[0]
        j_minus_cart = current_ops(geo, p, (-k)[None, :], -1)[0]
        jmap = np.stack([pmat @ jn[a] @ np.conj(pmat.T) for a in range(2)], axis=0)
        err_j = maxabs(jmap + j_minus_cart)
        rows.append(
            {
                "name": spec["name"],
                "partner": spec["partner"],
                "G_cart": [float(g[0]), float(g[1])],
                "G_norm": float(np.linalg.norm(g)),
                "P_Hplus_P_minus_Hminus_minusk": err_cart,
                "P_Hplus_P_minus_sewn_Hminus_partner": err_sewn,
                "P_jplus_P_plus_jminus_minusk": err_j,
            }
        )
    return rows


def mapping_operators(u_late_plus, u_early_minus, pmat, ug):
    mapped = pmat @ u_late_plus @ np.conj(pmat.T)
    target = ug @ u_early_minus @ np.conj(ug.T)
    return maxabs(mapped - target)


def mapping_rho(rho_plus_half, rho_minus_t0, pmat, ug):
    # rho_minus(-k,t0) = U_G rho_minus(k',t0) U_G^T  vs  P rho_plus(k, t0+T/2) P^T
    mapped = pmat @ rho_plus_half @ np.conj(pmat.T)
    target = ug @ rho_minus_t0 @ np.conj(ug.T)
    return maxabs(mapped - target)


def pol_components(j, basis):
    return np.asarray(j, dtype=float) @ np.asarray(basis, dtype=float).T


def residual_pack(now, shift):
    now = np.asarray(now, dtype=float)
    shift = np.asarray(shift, dtype=float)
    res = shift - now
    num = d1.rms_vec(res)
    den = d1.rms_vec(now)
    corr = diag.vec_corr(now, shift)
    ratio = (d1.rms_vec(shift) / den) if den > 0 else None
    return {
        "rms_res": num,
        "rms_now": den,
        "rel": (num / den if den > 0 else None),
        "corr": corr,
        "rms_ratio": ratio,
    }


def time_index(t, t_target):
    """Index of t_target in an array that need not start at 0."""
    t = np.asarray(t, dtype=float)
    dt = float(np.mean(np.diff(t)))
    return int(round((float(t_target) - float(t[0])) / dt))


def slice_half_abs(t, prot, cycle_lo, cycle_hi):
    t = np.asarray(t, dtype=float)
    dt = float(np.mean(np.diff(t)))
    n_half = int(round(0.5 * prot.period / dt))
    i0 = time_index(t, cycle_lo * prot.period)
    i1 = time_index(t, cycle_hi * prot.period)
    i0 = max(i0, 0)
    i1 = min(i1, len(t) - n_half)
    return i0, i1, n_half


def rect_fourier_abs(times, signal, omega, t0, t1):
    t = np.asarray(times, dtype=float)
    sig = np.asarray(signal)
    i0 = max(time_index(t, t0), 0)
    i1 = min(time_index(t, t1), len(t) - 1)
    sl = slice(i0, i1 + 1)
    kern = np.exp(1j * omega * t[sl])
    if sig.ndim == 1:
        integ = np.trapezoid(sig[sl] * kern, t[sl])
    else:
        integ = np.trapezoid(sig[sl] * kern[:, None], t[sl], axis=0)
    return integ / (t1 - t0)


def half_period_split(t, j_tot, j_d, j_c, prot, cycle_lo, cycle_hi):
    i0, i1, n_half = slice_half_abs(t, prot, cycle_lo, cycle_hi)
    sl = slice(i0, i1)
    sls = slice(i0 + n_half, i1 + n_half)
    tot = residual_pack(j_tot[sl], j_tot[sls])
    dia = residual_pack(j_d[sl], j_d[sls])
    coh = residual_pack(j_c[sl], j_c[sls])
    res_t = j_tot[sls] - j_tot[sl]
    res_d = j_d[sls] - j_d[sl]
    res_c = j_c[sls] - j_c[sl]
    rec = maxabs(res_t - res_d - res_c)
    nt = d1.rms_vec(res_t) ** 2
    nd = d1.rms_vec(res_d) ** 2
    nc = d1.rms_vec(res_c) ** 2
    cross = float(np.mean(np.sum(res_d * res_c, axis=-1))) if res_t.ndim == 2 else float(np.mean(res_d * res_c))
    # 2 Re <d,c> with the same RMS convention: mean of inner products
    return {
        "cycle_lo": cycle_lo,
        "cycle_hi": cycle_hi,
        "total": tot,
        "diag": dia,
        "coh": coh,
        "residual_reconstruct_maxabs": rec,
        "res_ms_total": nt,
        "res_ms_diag": nd,
        "res_ms_coh": nc,
        "res_mean_inner_diag_coh": cross,
        "coh_ms_fraction": (nc / nt if nt > 0 else None),
        "diag_ms_fraction": (nd / nt if nt > 0 else None),
    }


def per_cycle_h2(t, j, prot, basis, cycle_lo, cycle_hi):
    jb = pol_components(j, basis)
    rows = []
    for cyc in range(cycle_lo, cycle_hi):
        t0 = cyc * prot.period
        t1 = (cyc + 1) * prot.period
        z = rect_fourier_abs(t, jb, 2 * prot.omega0, t0, t1)
        rows.append(
            {
                "cycle": cyc,
                "abs": [float(abs(x)) for x in np.asarray(z).reshape(-1)],
                "arg_deg": [float(np.angle(x, deg=True)) for x in np.asarray(z).reshape(-1)],
                "norm": float(np.linalg.norm(z)),
            }
        )
    norms = np.array([r["norm"] for r in rows], dtype=float)
    args = np.array([r["arg_deg"][0] for r in rows], dtype=float)
    unwrapped = np.unwrap(np.deg2rad(args))
    return {
        "rows": rows,
        "rel_std": float(np.std(norms) / np.mean(norms)) if np.mean(norms) > 0 else None,
        "max_over_min": float(np.max(norms) / np.min(norms)) if np.min(norms) > 0 else None,
        "phase_range_deg": float(np.rad2deg(np.max(unwrapped) - np.min(unwrapped))) if len(args) else None,
    }


def noninteger_frac(t, j, prot, t0, t1):
    t = np.asarray(t, dtype=float)
    t_abs = t - float(t[0])
    spec = diag.window_power_spectrum(
        t_abs, j, prot, t0 - float(t[0]), t1 - float(t[0]), kind="rect"
    )
    return {
        "noninteger_power_frac": spec["noninteger_power_frac"],
        "integer_power_frac_hw0p25": spec["integer_power_frac_hw0p25"],
        "peak_nu_over_omega0": spec["peak_nu_over_omega0"],
        "dnu_cycles": spec["dnu_cycles"],
    }


def pair_odd(jp_k, jm_partner):
    return 0.5 * (jp_k - jm_partner)


def same_odd(jp_k, jm_k):
    return 0.5 * (jp_k - jm_k)


def pack_occ(an):
    return {
        "unitarity_maxabs": an["unitarity_maxabs"],
        "det_abs": an["det_abs"],
        "eig_reconstruct_maxabs": an["eig_reconstruct_maxabs"],
        "tr_rho": an["tr_rho"],
        "purity": an["purity"],
        "occ": [float(x) for x in an["occ"]],
        "eps": [float(x) for x in an["eps"]],
        "offdiag_frac_ungrouped": an["offdiag_frac_ungrouped"],
        "offdiag_frac_hard": an["offdiag_frac_hard"],
        "offdiag_frac_near": an["offdiag_frac_near"],
        "n_hard_mult": an["n_hard_mult"],
        "n_near_mult": an["n_near_mult"],
        "hard_clusters": an["hard_clusters"],
        "near_clusters": an["near_clusters"],
        "coh_fro_near": an["coh_fro_near"],
        "diag_fro_near": an["diag_fro_near"],
    }


def make_figures(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "DejaVu Sans"],
            "font.size": 9,
            "axes.unicode_minus": False,
        }
    )
    names = summary["names"]
    # 1. coherence fractions
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(names))
    plus = [summary["floquet"][n]["plus"]["offdiag_frac_near"] or 0.0 for n in names]
    minus = [summary["floquet"][n]["minus"]["offdiag_frac_near"] or 0.0 for n in names]
    ax.bar(x - 0.18, plus, 0.36, label="+N")
    ax.bar(x + 0.18, minus, 0.36, label="-N")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("grouped off-diag |ρ^F| / |ρ^F|")
    ax.set_title("Near-grouped Floquet coherence fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figure_floquet_coherence.png", dpi=160)
    fig.savefig(OUT / "figure_floquet_coherence.pdf")
    plt.close(fig)

    # 2. Jodd split time series for G1 pair
    ts = summary["timeseries_G1"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    tau = np.array(ts["t_cycles"])
    ax.plot(tau, ts["tot_par"], lw=1.0, label="total")
    ax.plot(tau, ts["diag_par"], lw=1.0, label="Floquet diag")
    ax.plot(tau, ts["coh_par"], lw=1.0, label="Floquet coh")
    ax.axvspan(3, 9, color="0.85", zorder=0)
    ax.set_xlabel("t / T0")
    ax.set_ylabel("J^o_pair ∥  (sheet)")
    ax.set_title("G1/−G1 pair: odd current split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figure_jodd_split.png", dpi=160)
    fig.savefig(OUT / "figure_jodd_split.pdf")
    plt.close(fig)

    # 3. residual fractions
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = []
    coh_f = []
    diag_f = []
    ro_t = []
    ro_d = []
    for rec in summary["pair_metrics"]:
        labels.append(rec["pair"])
        coh_f.append(rec["rect6"]["coh_ms_fraction"] or 0.0)
        diag_f.append(rec["rect6"]["diag_ms_fraction"] or 0.0)
        ro_t.append(rec["rect6"]["total"]["rel"] or 0.0)
        ro_d.append(rec["rect6"]["diag"]["rel"] or 0.0)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, coh_f, 0.36, label="coh residual MS fraction")
    ax.bar(x + 0.18, diag_f, 0.36, label="diag residual MS fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("mean-square residual fraction")
    ax.set_title("Who carries J^o half-period residual? (pair odd, cycles 3–9)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figure_halfperiod_split.png", dpi=160)
    fig.savefig(OUT / "figure_halfperiod_split.pdf")
    plt.close(fig)

    # 4. per-cycle H2
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    h2 = summary["h2_G1"]
    cyc = [r["cycle"] + 0.5 for r in h2["total"]["rows"]]
    for key, lab in (("total", "total"), ("diag", "diag"), ("coh", "coh")):
        axs[0].plot(cyc, [r["norm"] for r in h2[key]["rows"]], "o-", ms=4, label=lab)
        axs[1].plot(cyc, [r["arg_deg"][0] for r in h2[key]["rows"]], "o-", ms=4, label=lab)
    axs[0].set_ylabel("|J2^o_pair|")
    axs[1].set_ylabel("arg ∥ (deg)")
    axs[1].set_xlabel("optical cycle")
    axs[0].set_title("G1 pair: per-cycle H2 odd source")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "figure_cycle_h2.png", dpi=160)
    fig.savefig(OUT / "figure_cycle_h2.pdf")
    plt.close(fig)


def write_tsv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    t_wall0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = cl.p1_a2_flat_282_protocol(0.8)
    p = R0
    basis = np.stack(parallel_perp_basis(geo), axis=0)
    ksel = select_kpoints(geo)
    names = ksel["names"]
    kc = ksel["kc"]
    name_to_i = {n: i for i, n in enumerate(names)}
    hashes = {
        "q8_m0_classical.py": sha256_file(HERE / "q8_m0_classical.py"),
        "q8_m0_model.py": sha256_file(HERE / "q8_m0_model.py"),
        "q8_m0_magnetic_hhg_flat_d1_20261119.py": sha256_file(
            HERE / "q8_m0_magnetic_hhg_flat_d1_20261119.py"
        ),
        "this": sha256_file(Path(__file__)),
    }
    dump(OUT / "kpoint_selection.json", {"rows": ksel["rows"], "rationale": ksel["rationale"], "hashes": hashes})

    inv_h = hamiltonian_inversion_check(geo, p, kc, names)
    dump(OUT / "hamiltonian_inversion_check.json", inv_h)

    runs = {}
    for neel in (+1, -1):
        if time.perf_counter() - t_wall0 > BUDGET_S:
            break
        print(f"propagate neel={neel:+d}  elapsed {time.perf_counter()-t_wall0:.1f}s", flush=True)
        runs[neel] = propagate_selected(geo, p, kc, neel, prot)

    if +1 not in runs or -1 not in runs:
        dump(
            OUT / "fewk_status.json",
            {
                "status": "timeout_before_both_domains",
                "elapsed": time.perf_counter() - t_wall0,
                "completed_neel": sorted(runs),
            },
        )
        print("stopped: budget exceeded before both domains finished")
        return 2

    dt = runs[+1]["dt"]
    times = runs[+1]["times"]
    t0 = T0_CYCLE * prot.period
    n0 = T0_CYCLE * NTPC
    n1 = 10 * NTPC  # end of plateau
    n_plat0 = 2 * NTPC

    uF = {}
    uE = {}
    uL = {}
    for neel in (+1, -1):
        u_full, u_early, u_late = build_period_u(geo, p, kc, neel, t0, dt, prot, NTPC)
        uF[neel] = u_full
        uE[neel] = u_early
        uL[neel] = u_late

    floquet = {}
    rho_split = {+1: {}, -1: {}}
    strobe_err = {}
    for ik, name in enumerate(names):
        floquet[name] = {}
        strobe_err[name] = {}
        for neel, tag in ((+1, "plus"), (-1, "minus")):
            rho_t0 = runs[neel]["strobe"][T0_CYCLE][ik]
            an = analyze_uf(uF[neel][ik], rho_t0, prot, prot.period)
            floquet[name][tag] = pack_occ(an)
            rho_split[neel][name] = {
                "diag": an["rho_diag_near"],
                "coh": an["rho_coh_near"],
                "phi": an["phi"],
                "eps": an["eps"],
                "rho_t0": rho_t0,
            }
            recs = []
            uf = uF[neel][ik]
            rho_s = rho_t0.copy()
            ufinv = np.conj(uf.T)
            for nstep in range(0, 8):
                cyc = T0_CYCLE + nstep
                if cyc not in runs[neel]["strobe"]:
                    continue
                direct = runs[neel]["strobe"][cyc][ik]
                recs.append({"cycle": cyc, "maxabs": maxabs(rho_s - direct)})
                # advance one period
                rho_s = uf @ rho_s @ ufinv
            strobe_err[name][tag] = recs
            # Floquet phase of off-diagonal rho_f
            phi = an["phi"]
            eps = an["eps"]
            rho_f0 = np.conj(phi.T) @ rho_t0 @ phi
            phase_err = []
            rho_s = rho_t0.copy()
            for nstep in range(0, 8):
                cyc = T0_CYCLE + nstep
                rho_f = np.conj(phi.T) @ rho_s @ phi
                pred = rho_f0 * np.exp(-1j * (eps[:, None] - eps[None, :]) * nstep * prot.period)
                phase_err.append({"cycle": cyc, "maxabs": maxabs(rho_f - pred)})
                rho_s = uf @ rho_s @ ufinv
            floquet[name][tag]["stroboscopic_rho_maxabs"] = recs
            floquet[name][tag]["rhoF_phase_maxabs"] = phase_err

    # Split currents on the plateau from t0, using the same midpoint stepper.
    j_diag = {}
    j_coh = {}
    j_full_plat = {}
    for neel in (+1, -1):
        rho_d = np.stack([rho_split[neel][n]["diag"] for n in names], axis=0)
        rho_c = np.stack([rho_split[neel][n]["coh"] for n in names], axis=0)
        rho_f = np.stack([rho_split[neel][n]["rho_t0"] for n in names], axis=0)
        _, jd = propagate_from(rho_d, geo, p, kc, neel, prot, n0, n1)
        _, jc = propagate_from(rho_c, geo, p, kc, neel, prot, n0, n1)
        _, jf = propagate_from(rho_f, geo, p, kc, neel, prot, n0, n1)
        j_diag[neel] = jd
        j_coh[neel] = jc
        j_full_plat[neel] = jf

    recon_err = maxabs(
        j_full_plat[+1] - j_diag[+1] - j_coh[+1]
    )
    recon_err = max(recon_err, maxabs(j_full_plat[-1] - j_diag[-1] - j_coh[-1]))
    # Compare reconstructed full current with the original plateau slice
    orig_slice_err = maxabs(j_full_plat[+1] - runs[+1]["J_sheet"][n0 : n1 + 1])

    pmat = op_P()
    mapping = []
    pair_metrics = []
    t_plat = times[n0 : n1 + 1]
    seen_pairs = set()
    for spec in K_SPEC:
        name = spec["name"]
        partner = spec["partner"]
        key = tuple(sorted((name, partner)))
        ik = name_to_i[name]
        ip = name_to_i[partner]
        g = sewing_g(kc[ik], kc[ip])
        ug = op_sewing(geo, g) if np.linalg.norm(g) > 1e-10 else np.eye(8, dtype=complex)
        u_map = mapping_operators(uL[+1][ik], uE[-1][ip], pmat, ug)
        u_map_rev = mapping_operators(uL[-1][ip], uE[+1][ik], pmat, np.conj(ug.T))
        rho_plus_half = runs[+1]["half"][n0 + NTPC // 2][ik]
        rho_minus_t0 = runs[-1]["strobe"][T0_CYCLE][ip]
        rho_map = mapping_rho(rho_plus_half, rho_minus_t0, pmat, ug)
        # current mapping on cycles 3-9 using original full currents
        i0, i1, n_half = diag.slice_half(times, prot, 3, 9)
        jp = runs[+1]["J_sheet"][:, ik]
        jm = runs[-1]["J_sheet"][:, ip]
        corr_map = diag.vec_corr(jp[i0 + n_half : i1 + n_half], -jm[i0:i1])
        mapping.append(
            {
                "k": name,
                "partner": partner,
                "G_norm": float(np.linalg.norm(g)),
                "U_late_plus_vs_U_early_minus": u_map,
                "U_late_minus_partner_vs_U_early_plus": u_map_rev,
                "rho_plus_half_vs_rho_minus_t0": rho_map,
                "corr_Jp_tphalf_minus_Jm_partner": corr_map,
            }
        )
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        # Pair-odd currents on plateau (from t0 split) and on full pulse (original)
        jp_t = j_full_plat[+1][:, ik]
        jm_p = j_full_plat[-1][:, ip]
        jd_p = j_diag[+1][:, ik]
        jd_m = j_diag[-1][:, ip]
        jc_p = j_coh[+1][:, ik]
        jc_m = j_coh[-1][:, ip]
        jo_t = pair_odd(jp_t, jm_p)
        jo_d = pair_odd(jd_p, jd_m)
        jo_c = pair_odd(jc_p, jc_m)
        jo_same_t = same_odd(jp_t, j_full_plat[-1][:, ik])
        hp = half_period_split(t_plat, jo_t, jo_d, jo_c, prot, 3, 9)
        hp_same = half_period_split(
            t_plat,
            jo_same_t,
            same_odd(jd_p, j_diag[-1][:, ik]),
            same_odd(jc_p, j_coh[-1][:, ik]),
            prot,
            3,
            9,
        )
        h2_t = per_cycle_h2(t_plat, jo_t, prot, basis, 3, 10)
        h2_d = per_cycle_h2(t_plat, jo_d, prot, basis, 3, 10)
        h2_c = per_cycle_h2(t_plat, jo_c, prot, basis, 3, 10)
        ni_t = noninteger_frac(t_plat, jo_t, prot, 3 * prot.period, 9 * prot.period)
        ni_d = noninteger_frac(t_plat, jo_d, prot, 3 * prot.period, 9 * prot.period)
        ni_c = noninteger_frac(t_plat, jo_c, prot, 3 * prot.period, 9 * prot.period)
        pair_metrics.append(
            {
                "pair": f"{name}/{partner}",
                "names": [name, partner],
                "inversion_fixed": bool(name == partner),
                "rect6": hp,
                "rect6_samek": hp_same,
                "h2_total": {"rel_std": h2_t["rel_std"], "max_over_min": h2_t["max_over_min"], "phase_range_deg": h2_t["phase_range_deg"]},
                "h2_diag": {"rel_std": h2_d["rel_std"], "max_over_min": h2_d["max_over_min"], "phase_range_deg": h2_d["phase_range_deg"]},
                "h2_coh": {"rel_std": h2_c["rel_std"], "max_over_min": h2_c["max_over_min"], "phase_range_deg": h2_c["phase_range_deg"]},
                "noninteger_total": ni_t,
                "noninteger_diag": ni_d,
                "noninteger_coh": ni_c,
                "rms_Jo_total": d1.rms_vec(jo_t),
                "rms_Jo_diag": d1.rms_vec(jo_d),
                "rms_Jo_coh": d1.rms_vec(jo_c),
            }
        )
        if name == "G1":
            g1_ts = {
                "t_cycles": (t_plat / prot.period).tolist(),
                "tot_par": pol_components(jo_t, basis)[:, 0].tolist(),
                "diag_par": pol_components(jo_d, basis)[:, 0].tolist(),
                "coh_par": pol_components(jo_c, basis)[:, 0].tolist(),
            }
            g1_h2 = {"total": h2_t, "diag": h2_d, "coh": h2_c}

    # Quantitative verdict helpers
    coh_fracs = [r["rect6"]["coh_ms_fraction"] for r in pair_metrics if r["rect6"]["coh_ms_fraction"] is not None]
    diag_rels = [r["rect6"]["diag"]["rel"] for r in pair_metrics if r["rect6"]["diag"]["rel"] is not None]
    tot_rels = [r["rect6"]["total"]["rel"] for r in pair_metrics if r["rect6"]["total"]["rel"] is not None]
    offdiag = [
        floquet[n][tag]["offdiag_frac_near"]
        for n in names
        for tag in ("plus", "minus")
        if floquet[n][tag]["offdiag_frac_near"] is not None
    ]
    map_rho = [m["rho_plus_half_vs_rho_minus_t0"] for m in mapping]
    map_u = [m["U_late_plus_vs_U_early_minus"] for m in mapping]
    map_j = [m["corr_Jp_tphalf_minus_Jm_partner"] for m in mapping if m["corr_Jp_tphalf_minus_Jm_partner"] is not None]

    mean_off = float(np.mean(offdiag)) if offdiag else None
    mean_coh_frac = float(np.mean(coh_fracs)) if coh_fracs else None
    mean_diag_rel = float(np.mean(diag_rels)) if diag_rels else None
    mean_tot_rel = float(np.mean(tot_rels)) if tot_rels else None
    mean_map_rho = float(np.mean(map_rho)) if map_rho else None
    mean_map_u = float(np.mean(map_u)) if map_u else None
    mean_map_j = float(np.mean(map_j)) if map_j else None

    q9a = "yes_significant" if (mean_off is not None and mean_off > 0.05) else "weak_or_absent"
    if mean_coh_frac is not None and mean_diag_rel is not None and mean_tot_rel is not None:
        if mean_coh_frac >= 0.7 and mean_diag_rel <= 0.5 * mean_tot_rel:
            q9b = "coh_dominates_quantitatively"
            hypo = "quantitatively_supported"
        elif mean_coh_frac <= 0.3:
            q9b = "coh_does_not_dominate"
            hypo = "weakened"
        else:
            q9b = "mixed_or_k_dependent"
            hypo = "still_indistinguishable"
    else:
        q9b = "insufficient"
        hypo = "still_indistinguishable"
    # Independent mapping-failure signs: large diag Ro, or failed U/rho map
    diag_not_covariant = bool(mean_diag_rel is not None and mean_diag_rel > 0.5)
    map_failed = bool(mean_map_u is not None and mean_map_u > 1e-6) or bool(
        mean_map_rho is not None and mean_map_rho > 1e-4
    )
    q9c = "independent_signs_present" if (diag_not_covariant or map_failed) else "no_independent_signs_at_few_k"

    elapsed = time.perf_counter() - t_wall0
    status = {
        "task": "floquet_occupation_few_k",
        "utc": utc_now(),
        "protocol": prot.label,
        "mu": MU,
        "ntpc": NTPC,
        "algorithm": cl.ALGORITHM,
        "t0_cycle": T0_CYCLE,
        "budget_s": BUDGET_S,
        "elapsed_s": elapsed,
        "timeout": elapsed > BUDGET_S,
        "inputs_unchanged": True,
        "hamiltonian_modified": False,
        "t2_introduced": False,
        "d2_started": False,
        "n_k": 8,
        "n_domains": 2,
        "current_split_reconstruct_maxabs": recon_err,
        "plateau_full_vs_original_maxabs": orig_slice_err,
        "mean_offdiag_frac_near": mean_off,
        "mean_coh_ms_fraction": mean_coh_frac,
        "mean_Ro_total": mean_tot_rel,
        "mean_Ro_diag": mean_diag_rel,
        "mean_U_map_maxabs": mean_map_u,
        "mean_rho_map_maxabs": mean_map_rho,
        "mean_current_map_corr": mean_map_j,
        "q9a_coherence": q9a,
        "q9b_explains_Jodd": q9b,
        "q9c_mapping": q9c,
        "hypothesis_H1": hypo,
        "cannot_represent_BZ_sum": True,
        "hashes": hashes,
        "names": names,
    }

    dump(OUT / "fewk_status.json", status)
    dump(OUT / "floquet_per_k.json", floquet)
    dump(OUT / "mapping_checks.json", mapping)
    dump(OUT / "pair_metrics.json", pair_metrics)
    dump(OUT / "stroboscopic_checks.json", strobe_err)

    tsv_rows = []
    for rec in pair_metrics:
        tsv_rows.append(
            {
                "pair": rec["pair"],
                "Ro_total": rec["rect6"]["total"]["rel"],
                "Ro_diag": rec["rect6"]["diag"]["rel"],
                "Ro_coh": rec["rect6"]["coh"]["rel"],
                "corr_total": rec["rect6"]["total"]["corr"],
                "corr_diag": rec["rect6"]["diag"]["corr"],
                "corr_coh": rec["rect6"]["coh"]["corr"],
                "coh_ms_fraction": rec["rect6"]["coh_ms_fraction"],
                "diag_ms_fraction": rec["rect6"]["diag_ms_fraction"],
                "Ro_samek_total": rec["rect6_samek"]["total"]["rel"],
                "h2_tot_rel_std": rec["h2_total"]["rel_std"],
                "h2_diag_rel_std": rec["h2_diag"]["rel_std"],
                "h2_coh_rel_std": rec["h2_coh"]["rel_std"],
                "h2_tot_phase_range": rec["h2_total"]["phase_range_deg"],
                "h2_diag_phase_range": rec["h2_diag"]["phase_range_deg"],
                "h2_coh_phase_range": rec["h2_coh"]["phase_range_deg"],
                "nonint_tot": rec["noninteger_total"]["noninteger_power_frac"],
                "nonint_diag": rec["noninteger_diag"]["noninteger_power_frac"],
                "nonint_coh": rec["noninteger_coh"]["noninteger_power_frac"],
                "rms_tot": rec["rms_Jo_total"],
                "rms_diag": rec["rms_Jo_diag"],
                "rms_coh": rec["rms_Jo_coh"],
            }
        )
    write_tsv(
        OUT / "pair_metrics.tsv",
        tsv_rows,
        list(tsv_rows[0].keys()) if tsv_rows else ["pair"],
    )

    np.savez_compressed(
        OUT / "fewk_currents.npz",
        times=times,
        names=np.array(names),
        kf=ksel["kf"],
        kc=kc,
        J_plus=runs[+1]["J_sheet"],
        J_minus=runs[-1]["J_sheet"],
        Jdiag_plus=j_diag[+1],
        Jdiag_minus=j_diag[-1],
        Jcoh_plus=j_coh[+1],
        Jcoh_minus=j_coh[-1],
        t0_cycle=np.int32(T0_CYCLE),
    )

    summary = {
        "names": names,
        "floquet": floquet,
        "pair_metrics": pair_metrics,
        "timeseries_G1": g1_ts,
        "h2_G1": g1_h2,
    }
    try:
        make_figures(summary)
        status["figures"] = True
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        status["figures"] = False
        status["figure_error"] = repr(exc)
        dump(OUT / "fewk_status.json", status)

    print("elapsed", elapsed, "budget", BUDGET_S)
    print("q9a", q9a, "q9b", q9b, "q9c", q9c, "H1", hypo)
    print("mean_off", mean_off, "mean_coh_frac", mean_coh_frac, "mean_Ro_diag", mean_diag_rel)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
