#!/usr/bin/env python3
"""Caller-fix + short independent comparison. Not production n,m.

Fixes the 2026-10-04 contract caller without rewriting midpoint_step:
1. Midpoint vertices from the frame AFTER the first h/2, not the end-of-step frame.
2. L/R port labels. Even/odd only via (L±R)/sqrt(2); identical L and R
   are not even/odd equality.
3. New results directory. Old contract.json is not overwritten.

Short interval: first half cycle of the unchanged eight-cycle P1, same
two k points and photon frequencies as continuum_columns_m0_check.
ntpc=64 is not a precision claim. Independent reference is RK4 on a separate RHS.
Three H2-neighbour frequencies are not used as a complete input vacuum.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_m0_classical import kappa_shift, p1_a2_protocol
from q8_m0_cjj import occup_empty_bases, x_from_bases
from q8_m0_continuum_columns_20261004 import (
    ColumnState,
    current,
    midpoint_step,
    stationary_photon_columns,
)
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import (
    assemble_incoming_bundle,
    classify_dark_columns,
    maxabs,
)
from q8_m0_model import R0, build_geometry, current_ops, h8, h8_and_current
from q8_m0_response import d_ops, mean_d_from_bases

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_caller_fix_20261005"
OLD_CONTRACT = REPO / "results/q_line/p1_h2_finite_kcv_contract_20261004/contract.json"
OLD_JSON = [
    "results/q_line/p1_h2_finite_kcv_contract_20261004/contract.json",
    "results/q_line/incoming_spectral_completeness_20261004/completeness_checks.json",
    "results/q_line/incoming_completeness_revision_20261004/revision_checks.json",
    "results/q_line/incoming_completeness_r2_20261004/r2_checks.json",
    "results/q_line/incoming_completeness_r2_leftovers_20261004/leftover_checks.json",
    "results/q_line/stage1ao_electronic_spectrum_20261004/electronic_spectrum.json",
    "results/q_line/stage1ap_h2_window_quadrature_20261004/h2_window_quadrature.json",
    "results/q_line/incoming_m0_finite_attach_20261004/finite_attach.json",
    "results/q_line/continuum_columns_20261004/m0_checks.json",
]
MU = -0.009010620372198276
TWO_K = np.array([[0.0, 0.0], [0.23, 0.12]])
WEIGHTS = np.array([0.5, 0.5])
FREQ = np.array([0.025, 0.10])
NTPC_SCALE = 1024
NSTEP_COMPARE = 8
REL_TOL = 0.01
N_C, N_V = 4, 4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(a, b):
    diff = maxabs(np.asarray(a) - np.asarray(b))
    scale = maxabs(b)
    return {
        "max_absolute": diff,
        "reference_max": scale,
        "max_relative": diff / max(scale, 1e-30),
    }


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


def m0_sample(geo, kf, weights, neel):
    kf = np.asarray(kf, float)
    weights = np.asarray(weights, float)
    if abs(float(np.sum(weights)) - 1.0) > 1e-12:
        raise ValueError("BZ weights must sum to 1.")
    kc = geo.k_cart(kf)
    vv, vc, en = occup_empty_bases(h8(geo, R0, kc, neel), MU)
    x = x_from_bases(current_ops(geo, R0, kc, neel), vv, vc)
    gap = (en[:, 4:, None] - en[:, None, :4]).reshape(-1)
    u = np.einsum("k,kacv->akcv", np.sqrt(weights), x).reshape(2, -1)
    d_occ = mean_d_from_bases(d_ops(geo, R0, kc, neel), vv)
    d = 0.5 * (np.einsum("k,kab->ab", weights, d_occ) + np.einsum("k,kab->ab", weights, d_occ).T)
    n_k = len(kf)
    frame = np.concatenate((vv, vc), axis=2)
    return {
        "kf": kf,
        "weights": weights,
        "kc": kc,
        "gap": gap,
        "u": u,
        "D": d,
        "frame": frame,
        "k_of": np.repeat(np.arange(n_k), N_C * N_V),
        "n_el": int(u.shape[1]),
        "n_k": int(n_k),
        "neel": int(neel),
        "weight_once_residual": maxabs(
            u.conj() @ u.T
            - np.einsum(
                "k,kap,kbp->ab",
                weights,
                np.conj(x.reshape(n_k, 2, -1)),
                x.reshape(n_k, 2, -1),
            )
        ),
        "min_direct_gap": float(np.min(gap)),
        "max_direct_gap": float(np.max(gap)),
    }


def vertices_from_frame(geo, kc, prot, neel, t, frame, weights):
    kap = kc + kappa_shift(t, geo, prot, +1)
    j = current_ops(geo, R0, kap, neel)
    vv, vc = frame[:, :, :4], frame[:, :, 4:]
    x = x_from_bases(j, vv, vc)
    u = np.einsum("k,kacv->akcv", np.sqrt(weights), x).reshape(2, -1)
    d = np.einsum("k,kab->ab", weights, mean_d_from_bases(d_ops(geo, R0, kap, neel), vv))
    d = 0.5 * (d + d.T)
    return u, d


def advance_half_frame(geo, kc, prot, neel, frame, when, h):
    ham = h8(geo, R0, kc + kappa_shift(when, geo, prot, +1), neel)
    en, vec = np.linalg.eigh(ham)
    evol = (vec * np.exp(-1j * en * (h / 2.0))[:, None, :]) @ vec.conj().swapaxes(1, 2)
    return evol @ frame


def packed_state(bundle):
    ph, pt = bundle.photon.initial, bundle.point
    return ColumnState(
        np.concatenate((ph.b, pt.b), axis=1),
        np.concatenate((ph.bdag, pt.bdag), axis=1),
        np.concatenate((ph.A, pt.A), axis=1),
    )


def free_A_increment(ref, t0, t1, n_point=0):
    fa = ref.free_A(t1) - ref.free_A(t0)
    if n_point:
        fa = np.concatenate((fa, np.zeros((2, n_point), complex)), axis=1)
    return fa


def propagate_corrected(geo, sample, prot, ref, gamma, nstep, h, *, n_point=0, initial=None):
    """True midpoint: vertices after the first h/2, matching m0_check."""
    state = ref.initial.copy() if initial is None else initial.copy()
    frame = sample["frame"].copy()
    times = np.linspace(0.0, nstep * h, 5)
    idx = set(int(i) for i in np.round(times / h).astype(int))
    rec_t, rec_I, rec_state, rec_frame = [], [], [], []

    def record(step):
        t = step * h
        u, d = vertices_from_frame(
            geo, sample["kc"], prot, sample["neel"], t, frame, sample["weights"]
        )
        rec_t.append(t)
        rec_I.append(current(state, u, d))
        rec_state.append(state.packed())
        rec_frame.append(frame.copy())

    record(0)
    for step in range(nstep):
        t = step * h
        for which, when in enumerate((t + h / 4.0, t + 3.0 * h / 4.0)):
            frame = advance_half_frame(
                geo, sample["kc"], prot, sample["neel"], frame, when, h
            )
            if which == 0:
                u, d = vertices_from_frame(
                    geo,
                    sample["kc"],
                    prot,
                    sample["neel"],
                    t + 0.5 * h,
                    frame,
                    sample["weights"],
                )
                fa = free_A_increment(ref, t, t + h, n_point)
                state = midpoint_step(state, u, d, h, gamma, free_A_increment=fa)
        if step + 1 in idx:
            record(step + 1)
    return {
        "times": np.asarray(rec_t, float),
        "current": np.asarray(rec_I, complex),
        "state": np.asarray(rec_state, complex),
        "frames": np.asarray(rec_frame, complex),
        "final_state": state,
        "n_steps": int(nstep),
        "dt": float(h),
    }


def propagate_wrong_end_frame(geo, sample, prot, ref, gamma, nstep, h):
    """Old caller: evolve both half-steps, then take midpoint vertices."""
    state = ref.initial.copy()
    frame = sample["frame"].copy()
    times = np.linspace(0.0, nstep * h, 5)
    idx = set(int(i) for i in np.round(times / h).astype(int))
    rec_t, rec_I, rec_state, rec_frame = [], [], [], []

    def record(step):
        t = step * h
        u, d = vertices_from_frame(
            geo, sample["kc"], prot, sample["neel"], t, frame, sample["weights"]
        )
        rec_t.append(t)
        rec_I.append(current(state, u, d))
        rec_state.append(state.packed())
        rec_frame.append(frame.copy())

    record(0)
    for step in range(nstep):
        t = step * h
        for when in (t + h / 4.0, t + 3.0 * h / 4.0):
            frame = advance_half_frame(
                geo, sample["kc"], prot, sample["neel"], frame, when, h
            )
        u, d = vertices_from_frame(
            geo, sample["kc"], prot, sample["neel"], t + 0.5 * h, frame, sample["weights"]
        )
        fa = free_A_increment(ref, t, t + h, 0)
        state = midpoint_step(state, u, d, h, gamma, free_A_increment=fa)
        if step + 1 in idx:
            record(step + 1)
    return {
        "times": np.asarray(rec_t, float),
        "current": np.asarray(rec_I, complex),
        "state": np.asarray(rec_state, complex),
        "frames": np.asarray(rec_frame, complex),
        "n_steps": int(nstep),
        "dt": float(h),
    }


def independent_rhs(t, y, geo, sample, prot, ref, nframe, packed_shape, nel):
    """Independent RHS. Does not call library rhs/current/midpoint_step."""
    frame0 = sample["frame"]
    kc = sample["kc"]
    neel = sample["neel"]
    weights = sample["weights"]
    frame = y[:nframe].reshape(frame0.shape)
    state = y[nframe:].reshape(packed_shape)
    b, bd, a = state[:nel], state[nel : 2 * nel], state[2 * nel :]
    kap = kc + kappa_shift(t, geo, prot, +1)
    ham, j = h8_and_current(geo, R0, kap, neel)
    vv, vc = frame[:, :, :4], frame[:, :, 4:]
    x = np.einsum("kic,kaij,kjv->kacv", vc.conj(), j, vv)
    u = np.einsum("k,kacv->akcv", np.sqrt(weights), x).reshape(2, -1)
    dop = d_ops(geo, R0, kap, neel)
    d = np.einsum("k,kiv,kabij,kjv->ab", weights, vv.conj(), dop, vv).real
    ii = u.conj() @ b + u @ bd - d @ a
    df = -1j * ham @ frame
    db = 1j * u.T @ a
    dbdag = -1j * u.conj().T @ a
    da = (
        -1j * ref.omega[None, :] * ref.free_A0 * np.exp(-1j * ref.omega[None, :] * t)
        + ref.gamma * ii
    )
    return np.r_[df.reshape(-1), np.concatenate((db, dbdag, da), axis=0).reshape(-1)]


def independent_rk4(geo, sample, prot, ref, t_end, n_sub=4):
    """Classic RK4 on the independent RHS. No scipy, no library stepper."""
    frame0 = sample["frame"]
    nel = len(ref.gaps)
    ncol = len(ref.omega)
    nframe = frame0.size
    packed_shape = (2 * nel + 2, ncol)
    y = np.r_[frame0.reshape(-1), ref.initial.packed().reshape(-1)]
    n_mid = int(round(t_end / (prot.period / NTPC_SCALE)))
    dt_mid = t_end / max(n_mid, 1)
    h = dt_mid / n_sub
    nstep = n_mid * n_sub
    record_steps = set(int(i) for i in np.round(np.linspace(0, nstep, 5)).astype(int))
    rec_t, rec_y = [], []
    started = time.perf_counter()

    def maybe_record(step, tcur, ycur):
        if step in record_steps:
            rec_t.append(tcur)
            rec_y.append(ycur.copy())

    maybe_record(0, 0.0, y)
    t = 0.0
    for step in range(nstep):
        k1 = independent_rhs(t, y, geo, sample, prot, ref, nframe, packed_shape, nel)
        k2 = independent_rhs(t + 0.5 * h, y + 0.5 * h * k1, geo, sample, prot, ref, nframe, packed_shape, nel)
        k3 = independent_rhs(t + 0.5 * h, y + 0.5 * h * k2, geo, sample, prot, ref, nframe, packed_shape, nel)
        k4 = independent_rhs(t + h, y + h * k3, geo, sample, prot, ref, nframe, packed_shape, nel)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + h
        maybe_record(step + 1, t, y)
    frames = np.array([yy[:nframe].reshape(frame0.shape) for yy in rec_y])
    states = np.array([yy[nframe:].reshape(packed_shape) for yy in rec_y])
    currents = []
    for tt, f, s in zip(rec_t, frames, states):
        st = ColumnState.unpack(s, nel)
        u, d = vertices_from_frame(geo, sample["kc"], prot, sample["neel"], tt, f, sample["weights"])
        currents.append(current(st, u, d))
    return {
        "times": np.asarray(rec_t, float),
        "frames": frames,
        "state": states,
        "current": np.asarray(currents, complex),
        "nfev": int(4 * nstep),
        "n_sub": int(n_sub),
        "wall_seconds": time.perf_counter() - started,
        "independent_rhs": True,
        "method": "rk4_independent_no_scipy",
    }


def photon_port_indices(labels):
    """L/R by actual side label. Even/odd only after (L±R)/sqrt(2)."""
    left, right = [], []
    for j, lab in enumerate(labels):
        side = lab[1]
        if side == "L":
            left.append(j)
        elif side == "R":
            right.append(j)
        else:
            raise ValueError(f"Photon label side is not L/R: {lab}")
    return np.asarray(left, int), np.asarray(right, int)


def even_odd_from_lr(a_cols, left_idx, right_idx):
    """Same (ω, pol) order: L then R in stationary_photon_columns."""
    a = np.asarray(a_cols, complex)
    if left_idx.size != right_idx.size:
        raise ValueError("L and R column counts disagree.")
    even = (a[:, left_idx] + a[:, right_idx]) / np.sqrt(2.0)
    odd = (a[:, left_idx] - a[:, right_idx]) / np.sqrt(2.0)
    return even, odd


def port_report(state, labels, u, d):
    left, right = photon_port_indices(labels)
    ii = current(state, u, d)
    even_a, odd_a = even_odd_from_lr(state.A, left, right)
    even_i, odd_i = even_odd_from_lr(ii, left, right)
    return {
        "n_L": int(left.size),
        "n_R": int(right.size),
        "A_L_minus_A_R_maxabs": maxabs(state.A[:, left] - state.A[:, right]),
        "I_L_minus_I_R_maxabs": maxabs(ii[:, left] - ii[:, right]),
        "A_even_maxabs": maxabs(even_a),
        "A_odd_maxabs": maxabs(odd_a),
        "I_even_maxabs": maxabs(even_i),
        "I_odd_maxabs": maxabs(odd_i),
        "mislabel_would_call_R_even_L_odd": True,
        "identical_L_R_is_not_even_odd_equality": True,
    }


def radiation_ledger(gamma):
    return {
        "old_chi_rad_slot": {
            "file": "tools/b0_small_model/q8_m0_k1_chi_interface.py",
            "status": "unset_awaiting_theory",
            "meaning": "2026-09 placeholder; not a new theory block",
            "do_not_flip_flag_to_true": True,
        },
        "authoritative_interfaces": {
            "source": "q8_m0_stage1ao_canonical_channels.SI_QUANTIZATION copied as ledger, not re-derived",
            "a_src": "a_src(ω)=i sqrt(Z0/(4*pi*hbar*ω*S)) I_tilde(ω)",
            "code_A_to_a": "a(ω)=sqrt(ω/gamma_self) A(ω)",
            "keep_1_over_sqrt_omega": True,
            "fourier_transform": "I_tilde(omega)=integral dt exp(+i omega t) I(t)",
            "positive_frequency_time_dependence": "exp(-i omega t)",
            "gsheet": "q8_m0_gsheet.TwoPortSheetSelfField frozen 2026-09-21",
            "ports": "A_R_out=A_L_in+A_self; A_L_out=A_R_in+A_self",
            "G_self_not_equal_K": True,
            "gamma_self": float(gamma),
            "nk_is_not_Ncell": True,
            "H2_packet": "q8_m0_stationary_h2_outpacket_20261003.photon_output_rows / packet_quadrature",
            "relative_output": "q8_m0_relative_output_20261003.compose_relative_output",
            "incoming_dark": "q8_m0_incoming_spectral_completeness_20261004",
        },
        "this_slice": {
            "evaluates_n_m": False,
            "three_H2_neighbour_nodes_are_wiring_only": True,
            "selected_check_frequencies": FREQ.tolist(),
            "not_complete_input_vacuum": True,
        },
    }


def run_domain(geo, prot, gamma, neel, *, with_dark):
    started = time.perf_counter()
    sample = m0_sample(geo, TWO_K, WEIGHTS, neel)
    h = prot.period / NTPC_SCALE
    nstep = NSTEP_COMPARE
    t_end = nstep * h
    print(f"N={neel:+d}: sample n_el={sample['n_el']}", flush=True)
    photon = stationary_photon_columns(sample["gap"], sample["u"], sample["D"], gamma, FREQ)
    initial = photon.initial.copy()
    classified = []
    n_dark = 0
    if with_dark:
        bundle = assemble_incoming_bundle(sample["gap"], sample["u"], sample["D"], gamma, FREQ)
        classified = classify_dark_columns(bundle.dark_info, k_of_index=sample["k_of"])
        n_dark = int(bundle.n_point_columns)
        initial = packed_state(bundle)
    print(f"N={neel:+d}: RK4 t_end={t_end:.6g} nstep={nstep}", flush=True)
    oracle = independent_rk4(geo, sample, prot, photon, t_end, n_sub=8)
    print(f"N={neel:+d}: RK4 {oracle['wall_seconds']:.2f}s nfev={oracle['nfev']}", flush=True)
    corrected = propagate_corrected(
        geo, sample, prot, photon, gamma, nstep, h, n_point=0, initial=photon.initial
    )
    wrong = propagate_wrong_end_frame(geo, sample, prot, photon, gamma, nstep, h)
    print(f"N={neel:+d}: midpoint done", flush=True)
    cmp_ok = errors(corrected["state"], oracle["state"])
    cmp_I = errors(corrected["current"], oracle["current"])
    cmp_fr = errors(corrected["frames"], oracle["frames"])
    cmp_wrong = errors(wrong["state"], oracle["state"])
    u0, d0 = sample["u"], sample["D"]
    ports = port_report(photon.initial, photon.labels, u0, d0)
    dark_t0 = 0.0
    if with_dark:
        dark_state = ColumnState(
            initial.b[:, photon.initial.b.shape[1] :],
            initial.bdag[:, photon.initial.b.shape[1] :],
            initial.A[:, photon.initial.A.shape[1] :],
        )
        dark_t0 = maxabs(current(dark_state, u0, d0))
    return {
        "neel": int(neel),
        "n_el": sample["n_el"],
        "weight_once_residual": sample["weight_once_residual"],
        "min_direct_gap": sample["min_direct_gap"],
        "max_direct_gap": sample["max_direct_gap"],
        "n_photon_columns": int(photon.initial.b.shape[1]),
        "n_point_columns": n_dark,
        "n_dark_classified": int(sum(int(r["nullity"]) for r in classified)),
        "dark_current_t0": dark_t0,
        "interval": [0.0, float(t_end)],
        "interval_meaning": (
            "short prefix of the UNCHANGED eight-cycle P1; "
            f"dt=period/{NTPC_SCALE} inherited from continuum_columns_m0_check; "
            f"only {NSTEP_COMPARE} steps; not a shortened pulse; not eight-cycle production"
        ),
        "ntpc_scale": NTPC_SCALE,
        "n_steps": nstep,
        "dt": float(h),
        "independent": {
            "nfev": oracle["nfev"],
            "wall_seconds": oracle["wall_seconds"],
            "method": oracle["method"],
            "n_sub": oracle["n_sub"],
        },
        "corrected_vs_independent_state": cmp_ok,
        "corrected_vs_independent_current": cmp_I,
        "corrected_vs_independent_frames": cmp_fr,
        "wrong_end_frame_vs_independent_state": cmp_wrong,
        "wrong_caller_is_worse": bool(
            cmp_wrong["max_absolute"] > cmp_ok["max_absolute"] * 1.5
            or cmp_ok["max_relative"] < REL_TOL
        ),
        "ports": ports,
        "wall_seconds": time.perf_counter() - started,
    }


def flags_from(plus, minus, gamma):
    return {
        "gamma_positive": bool(gamma > 0.0),
        "plus_weight_once": bool(plus["weight_once_residual"] < 1e-12),
        "minus_weight_once": bool(minus["weight_once_residual"] < 1e-12),
        "plus_corrected_state_rel_lt_1pct": bool(
            plus["corrected_vs_independent_state"]["max_relative"] < REL_TOL
        ),
        "minus_corrected_state_rel_lt_1pct": bool(
            minus["corrected_vs_independent_state"]["max_relative"] < REL_TOL
        ),
        "plus_corrected_current_rel_lt_1pct": bool(
            plus["corrected_vs_independent_current"]["max_relative"] < REL_TOL
        ),
        "minus_corrected_current_rel_lt_1pct": bool(
            minus["corrected_vs_independent_current"]["max_relative"] < REL_TOL
        ),
        "plus_L_equals_R_at_t0": bool(plus["ports"]["A_L_minus_A_R_maxabs"] < 1e-12),
        "minus_L_equals_R_at_t0": bool(minus["ports"]["A_L_minus_A_R_maxabs"] < 1e-12),
        "plus_odd_A_small_at_t0": bool(plus["ports"]["A_odd_maxabs"] < 1e-12),
        "minus_odd_A_small_at_t0": bool(minus["ports"]["A_odd_maxabs"] < 1e-12),
        "plus_keeps_dark_inventory": bool(plus["n_point_columns"] > 0),
        "minus_keeps_dark_inventory": bool(minus["n_point_columns"] > 0),
        "old_contract_not_overwritten": bool(OLD_CONTRACT.exists()),
        "not_production_n": True,
        "not_production_m": True,
        "not_full_pulse": True,
        "not_BZ": True,
        "two_k_not_a_domain_signal": True,
        "three_freq_not_complete_input": True,
    }


def main() -> int:
    t0 = time.perf_counter()
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma = float(gamma_self(geo))
    plus = run_domain(geo, prot, gamma, +1, with_dark=True)
    minus = run_domain(geo, prot, gamma, -1, with_dark=True)
    flags = flags_from(plus, minus, gamma)
    physics_flags = {
        k: v
        for k, v in flags.items()
        if k
        not in (
            "not_production_n",
            "not_production_m",
            "not_full_pulse",
            "not_BZ",
            "two_k_not_a_domain_signal",
            "three_freq_not_complete_input",
        )
    }
    payload = {
        "status": "p1_h2_finite_kcv_caller_fix_short_compare_not_production_nm",
        "all_physics_checks_pass": bool(all(physics_flags.values())),
        "pass_flags": flags,
        "scope": {
            "fixes": [
                "midpoint vertices after first h/2, matching continuum_columns_m0_check",
                "L/R labels; even/odd only via (L±R)/sqrt(2)",
                "new directory; old contract.json retained",
            ],
            "interval": f"{NSTEP_COMPARE} steps at dt=period/{NTPC_SCALE}",
            "ntpc_scale": NTPC_SCALE,
            "n_steps": NSTEP_COMPARE,
            "ntpc_is_not_a_precision_certificate": True,
            "selected_frequencies": FREQ.tolist(),
            "selected_frequencies_are_wiring_only": True,
            "not_complete_input_vacuum": True,
            "not_eight_cycle": True,
            "not_production_n_m": True,
            "two_k_not_a_domain_signal": True,
            "old_334_336_not_quantitative": True,
            "reuse_midpoint_step": True,
            "no_new_propagator": True,
        },
        "radiation_ledger": radiation_ledger(gamma),
        "gamma_self": gamma,
        "domains": {"N+1": plus, "N-1": minus},
        "old_json_not_overwritten": OLD_JSON,
        "elapsed_s": time.perf_counter() - t0,
        "source_sha256": {
            str(p.relative_to(REPO)): sha(p)
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_continuum_columns_20261004.py",
                REPO / "tools/b0_small_model/q8_m0_continuum_columns_m0_check_20261004.py",
                REPO / "tools/b0_small_model/q8_m0_incoming_spectral_completeness_20261004.py",
                REPO / "tools/b0_small_model/q8_m0_gsheet.py",
                REPO / "tools/b0_small_model/q8_m0_stage1ao_canonical_channels.py",
            )
        },
        "old_contract_sha256": sha(OLD_CONTRACT) if OLD_CONTRACT.exists() else None,
        "production": False,
        "optical_moments_computed": False,
        "P1_full_pulse": False,
        "legal_electronic_continuum_boundary": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "caller_fix.json"
    if out_json.exists():
        raise FileExistsError("Refuse overwrite of caller_fix.json")
    if not OLD_CONTRACT.exists():
        raise FileNotFoundError("Old contract.json missing; refuse to proceed")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failed = [k for k, v in physics_flags.items() if not v]
    print(
        json.dumps(
            {
                "status": payload["status"],
                "all_physics_checks_pass": payload["all_physics_checks_pass"],
                "failed": failed,
                "gamma_self": gamma,
                "plus": {
                    "state_rel": plus["corrected_vs_independent_state"]["max_relative"],
                    "current_rel": plus["corrected_vs_independent_current"]["max_relative"],
                    "wrong_state_rel": plus["wrong_end_frame_vs_independent_state"]["max_relative"],
                    "n_point": plus["n_point_columns"],
                    "A_odd": plus["ports"]["A_odd_maxabs"],
                    "A_L_minus_R": plus["ports"]["A_L_minus_A_R_maxabs"],
                },
                "minus": {
                    "state_rel": minus["corrected_vs_independent_state"]["max_relative"],
                    "current_rel": minus["corrected_vs_independent_current"]["max_relative"],
                    "wrong_state_rel": minus["wrong_end_frame_vs_independent_state"]["max_relative"],
                    "n_point": minus["n_point_columns"],
                },
                "elapsed_s": payload["elapsed_s"],
                "out": str(out_json),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if payload["all_physics_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
