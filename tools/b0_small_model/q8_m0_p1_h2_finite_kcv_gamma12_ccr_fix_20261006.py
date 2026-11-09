#!/usr/bin/env python3
"""Γ12 CCR quadrature repair: core+tails, cheap clusters, bright CCR.

Does not overwrite bz_access.json or two-k qualify/tails JSON.
Does not change D, gamma, Hamiltonian, P1, packet, or initial state.
Does not start a Γ12 / half-shift pulse.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_classical import k_grid, p1_a2_protocol
from q8_m0_continuum_ground_bridge_20261003 import ContinuumGroundBridge, photon_resolvent
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import (
    dark_subspace,
    equal_gap_clusters,
)
from q8_m0_k1_chi_interface import CHI_STATUS
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import (
    quadrature_health,
    grid_sample,
)
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import TWO_K, WEIGHTS, m0_sample, sha
from q8_m0_s1_kernel import parallel_perp_basis

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_gamma12_ccr_fix_20261006"
PROTECTED = [
    REPO / "results/q_line/p1_h2_finite_kcv_bz_access_20261005/bz_access.json",
    REPO / "results/q_line/p1_h2_finite_kcv_nm_qualify_20261005/nm_qualify_plus.json",
    REPO / "results/q_line/p1_h2_finite_kcv_nm_tails_20261005/nm_tails.json",
    q.OLD_U0_H2,
    q.OLD_INPUT_CLOSE,
    q.EIGHT_CYCLE,
]
q.OUT = OUT
OMEGA_LO, OMEGA_HI = 1.0e-3, 12.0
WATCH_GAPS = (5.086678361098405, 2.0648643779)


def progress(msg):
    q.progress(msg)


def write_json(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    for p in PROTECTED:
        if p is not None and not Path(p).exists():
            raise FileNotFoundError(p)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def bright_basis(gaps, u):
    """Orthonormal bright columns, same kernel floor as dark_subspace."""
    gaps = np.asarray(gaps, float)
    u = np.asarray(u, complex)
    n = len(gaps)
    cols = []
    eps = float(np.finfo(float).eps)
    for idx in equal_gap_clusters(gaps):
        cluster = u[:, idx]
        n_cl = int(len(idx))
        _um, sval, vh = np.linalg.svd(cluster.conj(), full_matrices=True)
        s_full = np.zeros(n_cl)
        s_full[: len(sval)] = sval
        scale = max(float(s_full[0]) if n_cl else 0.0, 1.0)
        machine = 64.0 * eps * scale * max(n_cl, 2)
        floor = max(1e-12 * scale, 1e-12, machine)
        for j in np.flatnonzero(s_full > floor):
            v = np.zeros(n, complex)
            v[idx] = vh[j].conj()
            cols.append(v)
    if not cols:
        return np.zeros((n, 0), complex)
    B = np.column_stack(cols)
    qb, _ = np.linalg.qr(B, mode="reduced")
    return qb


def mass_report(oms, wts, omega_lo=OMEGA_LO, omega_hi=OMEGA_HI):
    length = float(omega_hi) - float(omega_lo)
    s = float(np.sum(wts))
    return {
        "n_nodes": int(np.asarray(oms).size),
        "sum_dnu": s,
        "interval_length": length,
        "mass_rel_err": abs(s - length) / max(length, 1e-30),
        "omega_min": float(oms[0]) if np.asarray(oms).size else None,
        "omega_max": float(oms[-1]) if np.asarray(oms).size else None,
        "negative_weights": bool(np.any(np.asarray(wts) < 0)),
        "overlap_vs_mass_are_separate": True,
    }


def cheap_diag_ccr(sample, gamma, oms, wts):
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    eta = q.eta_of(gamma)
    basis, info, _near = dark_subspace(gaps, u)
    n = len(gaps)
    if basis.size:
        qd, _ = np.linalg.qr(basis, mode="reduced")
        pdiag = np.sum(np.abs(qd) ** 2, axis=1)
    else:
        pdiag = np.zeros(n)
    acc_bb = np.zeros(n)
    acc_dd = np.zeros(n)
    skipped = 0
    t0 = time.perf_counter()
    for om, dw in zip(oms, wts):
        try:
            b, bd, _a = q.even_bd(gaps, u, d, gamma, eta, float(om))
        except Exception:
            skipped += 1
            continue
        acc_bb += float(dw) * np.sum(np.abs(b) ** 2, axis=1)
        acc_dd += float(dw) * np.sum(np.abs(bd) ** 2, axis=1)
    ccr = acc_bb - acc_dd + pdiag - 1.0
    order = np.argsort(np.abs(ccr))[::-1]
    top = []
    for i in order[:12]:
        top.append({
            "index": int(i),
            "gap": float(gaps[i]),
            "ccr_diag": float(ccr[i]),
            "bb": float(acc_bb[i]),
            "dd": float(acc_dd[i]),
            "p_dark": float(pdiag[i]),
        })
    clusters = []
    for dg in q.unique_gaps(gaps):
        mask = np.abs(gaps - dg) < 1e-9
        mag = float(np.max(np.abs(ccr[mask])))
        clusters.append({
            "gap": float(dg),
            "n": int(np.count_nonzero(mask)),
            "maxabs_diag": mag,
            "mean_diag": float(np.mean(ccr[mask])),
        })
    clusters.sort(key=lambda r: -r["maxabs_diag"])
    watch = []
    for g0 in WATCH_GAPS:
        j = int(np.argmin(np.abs(gaps - g0)))
        watch.append({
            "target_gap": float(g0),
            "nearest_index": j,
            "nearest_gap": float(gaps[j]),
            "ccr_diag": float(ccr[j]),
        })
    i_w = int(order[0])
    return {
        "n_el": n,
        "n_dark": int(basis.shape[1]),
        "n_bright_expected": n - int(basis.shape[1]),
        "skipped_poles": int(skipped),
        "maxabs_diag": float(np.max(np.abs(ccr))),
        "trace_bb": float(np.sum(acc_bb)),
        "trace_dd": float(np.sum(acc_dd)),
        "worst_index": i_w,
        "worst_gap": float(gaps[i_w]),
        "top_diag": top,
        "top_clusters": clusters[:10],
        "watch_gaps": watch,
        "elapsed_s": time.perf_counter() - t0,
        "did_not_form_full_nel_matrix": True,
    }


def bright_ccr(sample, gamma, oms, wts, batch=48):
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    eta = q.eta_of(gamma)
    dark, info, _near = dark_subspace(gaps, u)
    B = bright_basis(gaps, u)
    n_b = int(B.shape[1])
    acc_bb = np.zeros((n_b, n_b), complex)
    acc_dd = np.zeros((n_b, n_b), complex)
    acc_bd = np.zeros((n_b, n_b), complex)
    acc_db = np.zeros((n_b, n_b), complex)
    skipped = 0
    buf_b, buf_d = [], []
    t0 = time.perf_counter()

    def flush():
        nonlocal acc_bb, acc_dd, acc_bd, acc_db, buf_b, buf_d
        if not buf_b:
            return
        Bb = np.concatenate(buf_b, axis=1)
        Dd = np.concatenate(buf_d, axis=1)
        acc_bb += Bb @ Bb.conj().T
        acc_dd += Dd.conj() @ Dd.T
        acc_bd += Bb @ Dd.conj().T
        acc_db += Dd.conj() @ Bb.T
        buf_b, buf_d = [], []

    for om, dw in zip(oms, wts):
        try:
            b, bd, _a = q.even_bd(gaps, u, d, gamma, eta, float(om))
        except Exception:
            skipped += 1
            continue
        s = np.sqrt(float(dw))
        buf_b.append(B.conj().T @ (s * b))
        buf_d.append(B.T @ (s * bd))
        if len(buf_b) >= int(batch):
            flush()
    flush()
    ccr_n = acc_bb - acc_dd - np.eye(n_b)
    ccr_a = acc_bd - acc_db
    herm = 0.5 * (ccr_n + ccr_n.conj().T)
    evals = np.linalg.eigvalsh(herm) if n_b else np.zeros(0)
    return {
        "n_bright": n_b,
        "n_dark": int(dark.shape[1]),
        "n_el": int(len(gaps)),
        "bright_plus_dark": n_b + int(dark.shape[1]),
        "skipped_poles": int(skipped),
        "trace_photon_bb": float(np.trace(acc_bb).real),
        "trace_dd": float(np.trace(acc_dd).real),
        "trace_bb_minus_trace_dd": float(np.trace(acc_bb).real - np.trace(acc_dd).real),
        "electronic_ccr_normal_maxabs": q.maxabs(ccr_n),
        "electronic_ccr_anomalous_maxabs": q.maxabs(ccr_a),
        "electronic_ccr_min_eig": float(evals[0]) if evals.size else 0.0,
        "elapsed_s": time.perf_counter() - t0,
        "static_bright_space_only": True,
        "not_a_driven_low_rank_approx": True,
        "bd_transform": "B.T",
        "b_transform": "B.conj().T",
    }


def _ccr_from_cols(bb, dd):
    nmat = bb @ bb.conj().T - dd.conj() @ dd.T - np.eye(bb.shape[0])
    amat = bb @ dd.conj().T - dd.conj() @ bb.T
    return q.maxabs(nmat), q.maxabs(amat)


def canonical_three_mode_regression():
    """Complex 3-mode Bogoliubov fixture. Not a physical sample."""
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    qmat, _ = np.linalg.qr(raw)
    b_basis = qmat[:, :2]
    ch, sh = float(np.cosh(0.4)), float(np.sinh(0.4))
    b_b = ch * np.eye(2, dtype=complex)
    bd_b = sh * np.array([[0.0, 1.0], [1.0, 0.0]], complex)
    b = b_basis @ b_b
    bd = b_basis.conj() @ bd_b
    b_ok = b_basis.conj().T @ b
    bd_ok = b_basis.T @ bd
    bd_bad = b_basis.conj().T @ bd
    n_ok, a_ok = _ccr_from_cols(b_ok, bd_ok)
    n_bad, a_bad = _ccr_from_cols(b_ok, bd_bad)
    return {
        "correct_normal_maxabs": n_ok,
        "correct_anomalous_maxabs": a_ok,
        "wrong_conj_normal_maxabs": n_bad,
        "wrong_conj_anomalous_maxabs": a_bad,
        "correct_pass": bool(n_ok < 1e-12 and a_ok < 1e-12),
        "wrong_conj_is_larger": bool(max(n_bad, a_bad) > 1e-3),
        "complex_basis": True,
        "not_a_real_matrix_only_example": True,
    }


def mapped_bright_mats(sample, gamma, oms, wts, b_basis):
    """Static equivalent: b_B=U_B A/(Δ_B-ω), bd_B=D_B A/(Δ_B+ω)."""
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    eta = q.eta_of(gamma)
    b_basis = np.asarray(b_basis, complex)
    wgt = np.abs(b_basis) ** 2
    dg = (wgt.T @ np.asarray(gaps, float)) / np.maximum(np.sum(wgt, axis=0), 1e-30)
    u_b = b_basis.conj().T @ u.T
    d_b = b_basis.T @ u.conj().T
    n_b = int(b_basis.shape[1])
    acc_bb = np.zeros((n_b, n_b), complex)
    acc_dd = np.zeros((n_b, n_b), complex)
    acc_bd = np.zeros((n_b, n_b), complex)
    acc_db = np.zeros((n_b, n_b), complex)
    skipped = 0
    buf_b, buf_d = [], []

    def flush():
        nonlocal acc_bb, acc_dd, acc_bd, acc_db, buf_b, buf_d
        if not buf_b:
            return
        bb = np.concatenate(buf_b, axis=1)
        dd = np.concatenate(buf_d, axis=1)
        acc_bb += bb @ bb.conj().T
        acc_dd += dd.conj() @ dd.T
        acc_bd += bb @ dd.conj().T
        acc_db += dd.conj() @ bb.T
        buf_b, buf_d = [], []

    for om, dw in zip(oms, wts):
        try:
            _b, _bd, a = q.even_bd(gaps, u, d, gamma, eta, float(om))
        except Exception:
            skipped += 1
            continue
        s = np.sqrt(float(dw))
        buf_b.append(s * (u_b @ a) / (dg - float(om))[:, None])
        buf_d.append(s * (d_b @ a) / (dg + float(om))[:, None])
        if len(buf_b) >= 48:
            flush()
    flush()
    return acc_bb, acc_dd, acc_bd, acc_db, skipped, dg


def mapped_bright_ccr(sample, gamma, oms, wts, b_basis=None):
    t0 = time.perf_counter()
    gaps, u = sample["gap"], sample["u"]
    dark, _info, _n = dark_subspace(gaps, u)
    if b_basis is None:
        b_basis = bright_basis(gaps, u)
    acc_bb, acc_dd, acc_bd, acc_db, skipped, _dg = mapped_bright_mats(
        sample, gamma, oms, wts, b_basis
    )
    n_b = int(b_basis.shape[1])
    ccr_n = acc_bb - acc_dd - np.eye(n_b)
    ccr_a = acc_bd - acc_db
    herm = 0.5 * (ccr_n + ccr_n.conj().T)
    evals = np.linalg.eigvalsh(herm) if n_b else np.zeros(0)
    return {
        "n_bright": n_b,
        "n_dark": int(dark.shape[1]),
        "n_el": int(len(gaps)),
        "bright_plus_dark": n_b + int(dark.shape[1]),
        "skipped_poles": int(skipped),
        "trace_photon_bb": float(np.trace(acc_bb).real),
        "trace_dd": float(np.trace(acc_dd).real),
        "trace_bb_minus_trace_dd": float(np.trace(acc_bb).real - np.trace(acc_dd).real),
        "electronic_ccr_normal_maxabs": q.maxabs(ccr_n),
        "electronic_ccr_anomalous_maxabs": q.maxabs(ccr_a),
        "electronic_ccr_min_eig": float(evals[0]) if evals.size else 0.0,
        "elapsed_s": time.perf_counter() - t0,
        "static_mapped_UB_DB": True,
        "not_a_driven_low_rank_approx": True,
        "bd_transform": "B.T",
        "b_transform": "B.conj().T",
        "ccr_n": ccr_n,
        "ccr_a": ccr_a,
        "B": b_basis,
    }


def three_freq_map_vs_project(sample, gamma, oms, wts):
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    eta = q.eta_of(gamma)
    b_basis = bright_basis(gaps, u)
    wgt = np.abs(b_basis) ** 2
    dg = (wgt.T @ np.asarray(gaps, float)) / np.maximum(np.sum(wgt, axis=0), 1e-30)
    u_b = b_basis.conj().T @ u.T
    d_b = b_basis.T @ u.conj().T
    worst_b, worst_d, worst_gap = 0.0, 0.0, 0.0
    n_chk = 0
    for om, _dw in list(zip(oms, wts))[:: max(len(oms) // 3, 1)][:3]:
        b, bd, a = q.even_bd(gaps, u, d, gamma, eta, float(om))
        b_p = b_basis.conj().T @ b
        d_p = b_basis.T @ bd
        b_m = (u_b @ a) / (dg - float(om))[:, None]
        d_m = (d_b @ a) / (dg + float(om))[:, None]
        worst_b = max(worst_b, q.maxabs(b_p - b_m))
        worst_d = max(worst_d, q.maxabs(d_p - d_m))
        worst_gap = max(worst_gap, q.maxabs(np.asarray(gaps)[:, None] * b_basis - b_basis * dg[None, :]))
        n_chk += 1
    return {
        "n_checked": n_chk,
        "b_map_vs_project_maxabs": worst_b,
        "bd_map_vs_project_maxabs": worst_d,
        "gap_action_residual": worst_gap,
    }


def complex_basis_covariance(sample, gamma, oms, wts, seed=2):
    """Project the same even_bd columns with B and B R. Do not remap gaps."""
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    eta = q.eta_of(gamma)
    b0 = bright_basis(gaps, u)
    rng = np.random.default_rng(seed)
    n_b = int(b0.shape[1])
    raw = rng.normal(size=(n_b, n_b)) + 1j * rng.normal(size=(n_b, n_b))
    rot, _ = np.linalg.qr(raw)
    b1 = b0 @ rot
    acc = []
    for basis in (b0, b1):
        nmat = np.zeros((n_b, n_b), complex)
        amat = np.zeros((n_b, n_b), complex)
        skipped = 0
        for om, dw in zip(oms, wts):
            try:
                b, bd, _a = q.even_bd(gaps, u, d, gamma, eta, float(om))
            except Exception:
                skipped += 1
                continue
            s = np.sqrt(float(dw))
            bb = basis.conj().T @ (s * b)
            dd = basis.T @ (s * bd)
            nmat += bb @ bb.conj().T - dd.conj() @ dd.T
            amat += bb @ dd.conj().T - dd.conj() @ bb.T
        acc.append((nmat - np.eye(n_b), amat, skipped))
    n0, a0, sk = acc[0]
    n1, a1, _sk = acc[1]
    n_cov = rot.conj().T @ n0 @ rot
    a_cov = rot.conj().T @ a0 @ rot.conj()
    return {
        "normal_covariance_maxabs": q.maxabs(n1 - n_cov),
        "anomalous_covariance_maxabs": q.maxabs(a1 - a_cov),
        "complex_unitary_R": True,
        "rule": "N' = R† N R, A' = R† A R*",
        "used_projection_not_gap_remapping": True,
        "pass": bool(q.maxabs(n1 - n_cov) < 1e-10 and q.maxabs(a1 - a_cov) < 1e-10),
        "n0_maxabs": q.maxabs(n0),
        "a0_maxabs": q.maxabs(a0),
        "skipped_poles": int(sk),
        "n_freq": int(np.asarray(oms).size),
    }


def even_lr_spotcheck(sample, gamma, oms=(0.05, 0.10, 0.15, 1.0, 4.0)):
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    eta = q.eta_of(gamma)
    rows = []
    worst = 0.0
    for om in oms:
        b, bd, a = q.oneside_bd(gaps, u, d, gamma, eta, float(om))
        be, bde, ae = q.even_bd(gaps, u, d, gamma, eta, float(om))
        dn = q.maxabs(be - np.sqrt(2.0) * b)
        da = q.maxabs(ae - np.sqrt(2.0) * a)
        worst = max(worst, dn, da)
        rows.append({"omega": float(om), "even_vs_sqrt2_oneside": dn, "A_even_vs_sqrt2": da})
    return {"worst": worst, "machine_precision": bool(worst < 1e-12), "rows": rows}


def dressed_apply(bridge, z, v):
    z = complex(z)
    re_v = bridge.electron_action(z, v)
    qa = photon_resolvent(z, bridge.gamma)
    k = bridge.electronic_kernel(z)
    mid = np.linalg.solve(np.eye(2, dtype=complex) + qa * k, qa * (bridge.ci @ re_v))
    return re_v + bridge.electron_action(z, bridge.ci.T @ mid)


def spectral_apply(sample, gamma, oms, wts, z, v, bright_b, dark):
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    eta = q.eta_of(gamma)
    z = complex(z)
    acc = np.zeros_like(v, dtype=complex)
    skipped = 0
    for om, dw in zip(oms, wts):
        try:
            b, bd, _a = q.even_bd(gaps, u, d, gamma, eta, float(om))
        except Exception:
            skipped += 1
            continue
        f = q.f_from_bd(b, bd)
        acc += float(dw) * ((f @ (f.conj().T @ v)) / (om - z) + (f.conj() @ (f.T @ v)) / (om + z))
    if dark.size:
        point_gaps = []
        info = dark_subspace(gaps, u)[1]
        from q8_m0_incoming_spectral_completeness_20261004 import dark_column_gaps
        pg = dark_column_gaps(info)
        for j in range(dark.shape[1]):
            col = dark[:, j : j + 1]
            f = q.f_from_bd(col, np.zeros_like(col))
            dg = float(pg[j])
            acc += (f @ (f.conj().T @ v)) / (dg - z) + (f.conj() @ (f.T @ v)) / (dg + z)
    return acc, skipped


def projected_resolvent_check(sample, gamma, oms, wts, z_list):
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    bridge = ContinuumGroundBridge(gaps, u, d, gamma)
    B = bright_basis(gaps, u)
    dark, _info, _n = dark_subspace(gaps, u)
    rng = np.random.default_rng(0)
    n2 = 2 * len(gaps)
    vecs = []
    if B.shape[1]:
        v0 = np.zeros(n2, complex)
        v0[: len(gaps)] = B[:, 0]
        vecs.append(v0)
        if B.shape[1] > 1:
            v1 = np.zeros(n2, complex)
            v1[len(gaps) :] = B[:, 1]
            vecs.append(v1)
    vecs.append(rng.standard_normal(n2) + 1j * rng.standard_normal(n2))
    rows = []
    worst = 0.0
    t0 = time.perf_counter()
    for z in z_list:
        for i, v in enumerate(vecs):
            v = np.asarray(v, complex)
            v = v / max(np.linalg.norm(v), 1e-30)
            tgt = dressed_apply(bridge, z, v)
            spec, skipped = spectral_apply(sample, gamma, oms, wts, z, v, B, dark)
            err = q.errors(spec, tgt)
            worst = max(worst, err["max_relative"])
            rows.append({
                "z": {"real": float(complex(z).real), "imag": float(complex(z).imag)},
                "vec": i,
                "skipped_poles": int(skipped),
                **err,
            })
    return {
        "analytic_interface_available": True,
        "projected_resolvent_checked": True,
        "check_passed": bool(worst < 0.05),
        "worst_relative": worst,
        "n_vectors": len(vecs),
        "rows": rows,
        "elapsed_s": time.perf_counter() - t0,
        "did_not_form_2n_matrix": True,
    }


def blocked_cpu_timing(sample, gamma, oms, wts, n_freq=64):
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    eta = q.eta_of(gamma)
    B = bright_basis(gaps, u)
    oms = np.asarray(oms, float)[: int(n_freq)]
    wts = np.asarray(wts, float)[: int(n_freq)]
    t0 = time.perf_counter()
    cols = 0
    for om, dw in zip(oms, wts):
        b, bd, _a = q.even_bd(gaps, u, d, gamma, eta, float(om))
        _ = B.conj().T @ b
        cols += 2
    assemble_s = time.perf_counter() - t0
    per = assemble_s / max(len(oms), 1)
    return {
        "n_freq": int(len(oms)),
        "n_el": int(len(gaps)),
        "n_bright": int(B.shape[1]),
        "assemble_even_bd_and_bright_project_s": assemble_s,
        "per_freq_s": per,
        "not_a_full_pulse_budget": True,
        "not_from_16step_prefix_formula": True,
        "gpu_used": False,
    }


def scan_python_envs():
    cands = [
        sys.executable,
        r"C:\Users\26507\.workbuddy\binaries\python\envs\default\Scripts\python.exe",
        r"C:\Python314\python.exe",
        r"C:\Users\26507\AppData\Local\Python\pythoncore-3.14-64\python.exe",
    ]
    code = (
        "import json,sys\n"
        "mods={}\n"
        "for n in ('torch','cupy','numba'):\n"
        "    try:\n"
        "        m=__import__(n)\n"
        "        info={'import':True,'version':getattr(m,'__version__',None)}\n"
        "        if n=='torch' and hasattr(m,'cuda'):\n"
        "            info['cuda_is_available']=bool(m.cuda.is_available())\n"
        "            info['cuda_device_count']=int(m.cuda.device_count())\n"
        "        mods[n]=info\n"
        "    except Exception as e:\n"
        "        mods[n]={'import':False,'error':type(e).__name__}\n"
        "print(json.dumps({'exe':sys.executable,'ver':sys.version.split()[0],'mods':mods}))\n"
    )
    rows = []
    for exe in cands:
        if not exe or not Path(exe).exists():
            continue
        try:
            r = subprocess.run([exe, "-c", code], capture_output=True, text=True, timeout=20)
            rows.append(json.loads(r.stdout) if r.returncode == 0 else {"exe": exe, "error": r.stderr[-400:]})
        except Exception as exc:
            rows.append({"exe": exe, "error": repr(exc)})
    gpu = None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        gpu = r.stdout.strip() if r.returncode == 0 else r.stderr.strip()
    except Exception as exc:
        gpu = f"nvidia-smi_failed:{exc}"
    any_cuda = False
    for row in rows:
        t = row.get("mods", {}).get("torch", {})
        if t.get("cuda_is_available"):
            any_cuda = True
    return {
        "envs": rows,
        "nvidia_smi": gpu,
        "did_not_install_packages": True,
        "importable_is_not_cuda_backend": True,
        "gpu_backend_ready": any_cuda,
        "no_isolated_venv_created": True,
    }


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    prot = p1_a2_protocol(q.A_STAR)
    progress(f"gamma12 ccr fix start gamma={gamma:.12g}")

    two = m0_sample(geo, TWO_K, WEIGHTS, +1)
    oms2, wts2 = q.input_quadrature(two["gap"], two["u"], gamma)
    mass2 = mass_report(oms2, wts2)
    h2 = quadrature_health(oms2, wts2, two["gap"], two["u"], gamma)
    meas2 = q.electronic_measure_report(two, gamma, oms2, wts2)
    oms2f, wts2f = q.input_quadrature(two["gap"], two["u"], gamma, n_panel=q.N_PANEL_FINE, n_h2=q.N_H2_FINE)
    meas2f = q.electronic_measure_report(two, gamma, oms2f, wts2f)
    res2 = q.resolvent_from_measure(two, gamma, oms2, wts2, q.COMPLEX_Z)
    gate2 = q.measure_gate(meas2, meas2f, res2)
    pol = np.vstack(parallel_perp_basis(geo))
    eta = q.eta_of(gamma)
    kernel = q.detector_kernel_scalar(gamma, prot, 64, prot.period / q.NTPC)
    nrm = kernel["normalizer"]

    def phi_of(nu):
        return complex(np.asarray(q.phi_at(nu, prot, nrm)).reshape(-1)[0])

    u0_2 = q.u0_packet_check(two, gamma, oms2, wts2, pol, eta, phi_of, prot, nrm)
    progress(
        f"two-k nodes={h2['n_nodes']} mass_err={mass2['mass_rel_err']:.3g} "
        f"CCR_n={meas2['electronic_ccr_normal_maxabs']:.4g} gate={gate2['ok']} U0={u0_2['ok']}"
    )

    g12, inv12 = grid_sample(geo, 12, 0.0)
    oms12, wts12 = q.input_quadrature(g12["gap"], g12["u"], gamma)
    mass12 = mass_report(oms12, wts12)
    h12 = quadrature_health(oms12, wts12, g12["gap"], g12["u"], gamma)
    progress(f"gamma12 nodes={h12['n_nodes']} mass_err={mass12['mass_rel_err']:.3g} poles_in_h2={h12['n_poles_in_h2_window']}")
    lr12 = even_lr_spotcheck(g12, gamma)
    progress(f"even vs L/R worst={lr12['worst']:.3g}")
    cheap = cheap_diag_ccr(g12, gamma, oms12, wts12)
    progress(
        f"cheap diag maxabs={cheap['maxabs_diag']:.4g} worst_gap={cheap['worst_gap']:.6g} "
        f"tr_bb={cheap['trace_bb']:.4g} t={cheap['elapsed_s']:.2f}s"
    )

    meas12 = bright_ccr(g12, gamma, oms12, wts12)
    progress(
        f"bright CCR_n={meas12['electronic_ccr_normal_maxabs']:.4g} "
        f"CCR_a={meas12['electronic_ccr_anomalous_maxabs']:.4g} "
        f"n_b={meas12['n_bright']} t={meas12['elapsed_s']:.2f}s"
    )

    refine12 = None
    refine_status = "not_run"
    meas12f = None
    if meas12["electronic_ccr_normal_maxabs"] < 0.2:
        oms12f, wts12f = q.input_quadrature(g12["gap"], g12["u"], gamma, n_panel=q.N_PANEL_FINE)
        meas12f = bright_ccr(g12, gamma, oms12f, wts12f)
        refine12 = abs(meas12["trace_photon_bb"] - meas12f["trace_photon_bb"]) / max(
            abs(meas12f["trace_photon_bb"]), 1e-30
        )
        refine_status = "ran"
        progress(f"fine bright CCR_n={meas12f['electronic_ccr_normal_maxabs']:.4g} refine={refine12:.4g}")
    else:
        progress("fine panel not_run; bright CCR still above 0.2")

    proj = {
        "analytic_interface_available": True,
        "projected_resolvent_checked": False,
        "check_passed": False,
        "reason": "ccr_not_closed",
    }
    if meas12["electronic_ccr_normal_maxabs"] < 0.2:
        proj = projected_resolvent_check(g12, gamma, oms12, wts12, q.COMPLEX_Z)
        progress(f"projected resolvent worst_rel={proj['worst_relative']:.4g} pass={proj['check_passed']}")

    cover12 = bool(h12["n_nodes"] and oms12[0] < g12["min_direct_gap"] < oms12[-1] and oms12[0] < g12["max_direct_gap"] < oms12[-1])
    gate12 = {
        "ok": bool(
            cover12
            and meas12["electronic_ccr_normal_maxabs"] < 0.05
            and meas12["electronic_ccr_anomalous_maxabs"] < 0.05
            and refine12 is not None
            and refine12 < 0.05
            and int(meas12.get("skipped_poles", 0)) == 0
            and meas12["bright_plus_dark"] == meas12["n_el"]
        ),
        "ccr_normal": float(meas12["electronic_ccr_normal_maxabs"]),
        "ccr_anomalous": float(meas12["electronic_ccr_anomalous_maxabs"]),
        "refine": None if refine12 is None else float(refine12),
        "refine_status": refine_status,
        "resolvent_dressed": None if not proj.get("projected_resolvent_checked") else float(proj.get("worst_relative")),
        "analytic_interface_available": True,
        "projected_resolvent_checked": bool(proj.get("projected_resolvent_checked")),
        "check_passed": bool(proj.get("check_passed")),
        "analytic_dressed_bridge_used": False,
    }

    env = scan_python_envs()
    timing = blocked_cpu_timing(g12, gamma, oms12, wts12, n_freq=32)
    progress(f"blocked 32-freq assemble {timing['assemble_even_bd_and_bright_project_s']:.3f}s gpu_ready={env['gpu_backend_ready']}")

    start_plus = False
    stop_reason = None
    if not gate2["ok"] or not u0_2["ok"]:
        stop_reason = "two-k regression failed after quadrature change"
    elif not gate12["ok"]:
        stop_reason = "gamma12 input-measure gate failed"
    else:
        stop_reason = "CCR/access may pass but no GPU backend and no authorized full pulse"

    payload = {
        "status": "p1_h2_gamma12_ccr_core_tail_quadrature_not_domain",
        "two_k_archived": True,
        "did_not_start_pulse": True,
        "did_not_change_D_or_initial_or_gamma": True,
        "quadrature_rule": "voronoi_cell_t_gauss_core_and_tails_h2_gauss_on_tail_overlap",
        "mass": {"two_k": mass2, "gamma12": mass12},
        "two_k_regression": {
            "n_nodes": h2["n_nodes"],
            "old_n_nodes_576_generic": 576,
            "measure_gate": gate2,
            "u0_packet": {"ok": u0_2["ok"], "ccr": u0_2["U0_packet_ccr"], "mass": u0_2["packet_mass_dnu"]},
            "electronic_ccr_normal": meas2["electronic_ccr_normal_maxabs"],
            "electronic_ccr_anomalous": meas2["electronic_ccr_anomalous_maxabs"],
            "trace_photon_bb": meas2["trace_photon_bb"],
        },
        "gamma12_access": {
            "n_nodes": h12["n_nodes"],
            "n_el": g12["n_el"],
            "inversion": inv12,
            "quadrature_health": h12,
            "even_lr_spotcheck": lr12,
            "cheap_diag": cheap,
            "bright_ccr": meas12,
            "bright_ccr_fine": meas12f,
            "measure_gate": gate12,
            "projected_resolvent": proj,
            "kept_all_kcv_and_dark": True,
        },
        "short_test": {
            "blocked_cpu": timing,
            "gpu_env": env,
            "est_52h_is_not_measured_budget": True,
            "did_not_start_full_pulse": True,
        },
        "start_plusN": start_plus,
        "stop_reason": stop_reason,
        "chi_rad_placeholder_status": CHI_STATUS,
        "old_json_not_overwritten": [str(Path(p).relative_to(REPO)) for p in PROTECTED if p is not None],
        "elapsed_s": time.perf_counter() - t0,
        "production": False,
        "n_m_claimed_as_domain_signal": False,
    }
    out = write_json("gamma12_ccr_fix.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "two_k_gate": gate2["ok"],
        "two_k_nodes": h2["n_nodes"],
        "gamma12_gate": gate12["ok"],
        "gamma12_nodes": h12["n_nodes"],
        "cheap_maxabs": cheap["maxabs_diag"],
        "bright_CCR_n": meas12["electronic_ccr_normal_maxabs"],
        "refine_status": refine_status,
        "stop_reason": stop_reason,
        "elapsed_s": payload["elapsed_s"],
        "out": str(out),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
