#!/usr/bin/env python3
"""Finite-k P1→H2 source contraction. Not production n,m,V.

Internal (k,c,v) is BZ quadrature / particle-hole coordinates, not a set of
external electronic baths. Photon columns stay true continuum in/out.
Dark kcv combinations are kept (P1=0 dark ⇏ P1-irrelevant). Continuum
electronic incoming channels are NOT required to enter this slice.

Two k points, ±N, frozen P1_a2 and frozen H2 packet window. chi_rad unset.
Two-k numbers are not a domain signal and not a k-grid claim.
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
from q8_m0_classical import _sin2_window, kappa_shift, p1_a2_protocol
from q8_m0_cjj import occup_empty_bases, x_from_bases
from q8_m0_continuum_columns_20261004 import (
    ColumnState,
    current,
    midpoint_step,
)
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import (
    assemble_incoming_bundle,
    classify_dark_columns,
    maxabs,
)
from q8_m0_k1_packet import PACKET_ID, PACKET_STATUS, phi_unnormalized
from q8_m0_model import R0, build_geometry, current_ops, h8
from q8_m0_response import d_ops, mean_d_from_bases
from q8_m0_s1_kernel import parallel_perp_basis

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_contract_20261004"
OLD_JSON = [
    "results/q_line/incoming_spectral_completeness_20261004/completeness_checks.json",
    "results/q_line/incoming_completeness_revision_20261004/revision_checks.json",
    "results/q_line/incoming_completeness_r2_20261004/r2_checks.json",
    "results/q_line/incoming_completeness_r2_leftovers_20261004/leftover_checks.json",
    "results/q_line/stage1ao_electronic_spectrum_20261004/electronic_spectrum.json",
    "results/q_line/stage1ap_h2_window_quadrature_20261004/h2_window_quadrature.json",
    "results/q_line/incoming_m0_finite_attach_20261004/finite_attach.json",
    "results/q_line/continuum_columns_20261004/m0_checks.json",
    "results/q_line/continuum_columns_20261004/summary.json",
    "results/q_line/continuum_columns_20261004/math_checks.json",
]
MU = -0.009010620372198276
TWO_K = np.array([[0.0, 0.0], [0.23, 0.12]])
WEIGHTS = np.array([0.5, 0.5])
NTPC = 64
N_C, N_V = 4, 4
POLE_FLOOR = 1e-8


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    """Weight once: u = sqrt(w) X. Original (k,c,v). Occupied D."""
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
        "weight_once_residual": maxabs(u.conj() @ u.T - np.einsum(
            "k,kap,kbp->ab", weights, np.conj(x.reshape(n_k, 2, -1)), x.reshape(n_k, 2, -1)
        )),
        "min_direct_gap": float(np.min(gap)),
    }


def h2_packet_nodes(prot, gaps):
    """Frozen H2 packet centre ± one envelope bin. Drop nodes on electronic poles."""
    w0 = float(prot.omega0)
    center = 2.0 * w0
    dw = w0 / float(prot.n_cycles)
    candidates = np.array([center - dw, center, center + dw], float)
    gaps = np.asarray(gaps, float)
    kept, dropped = [], []
    for w in candidates:
        dist = float(np.min(np.abs(gaps - w)))
        rec = {"omega": float(w), "min_pole_distance": dist, "is_center": bool(abs(w - center) < 1e-15)}
        if dist <= POLE_FLOOR:
            rec["kept"] = False
            dropped.append(rec)
        else:
            rec["kept"] = True
            kept.append(rec)
    if not kept:
        raise RuntimeError("All frozen H2 packet nodes sit on electronic poles; no clipping.")
    return {
        "center": center,
        "envelope_bin": float(dw),
        "packet_id": PACKET_ID,
        "packet_status": PACKET_STATUS,
        "kept": kept,
        "dropped_on_pole": dropped,
        "frequencies": [r["omega"] for r in kept],
    }


def packed_state(bundle):
    ph, pt = bundle.photon.initial, bundle.point
    return ColumnState(
        np.concatenate((ph.b, pt.b), axis=1),
        np.concatenate((ph.bdag, pt.bdag), axis=1),
        np.concatenate((ph.A, pt.A), axis=1),
    )


def split_counts(bundle):
    return int(bundle.n_photon_columns), int(bundle.n_point_columns)


def vertices_from_frame(geo, kc, prot, neel, t, frame, weights):
    kap = kc + kappa_shift(t, geo, prot, +1)
    j = current_ops(geo, R0, kap, neel)
    vv, vc = frame[:, :, :4], frame[:, :, 4:]
    x = x_from_bases(j, vv, vc)
    u = np.einsum("k,kacv->akcv", np.sqrt(weights), x).reshape(2, -1)
    d = np.einsum("k,kab->ab", weights, mean_d_from_bases(d_ops(geo, R0, kap, neel), vv))
    d = 0.5 * (d + d.T)
    return u, d


def evolve_frame(geo, kc, prot, neel, frame, t, h):
    """True midpoint frame: two half-steps, no averaging of fast X phases."""
    for when in (t + h / 4.0, t + 3.0 * h / 4.0):
        ham = h8(geo, R0, kc + kappa_shift(when, geo, prot, +1), neel)
        en, vec = np.linalg.eigh(ham)
        evol = (vec * np.exp(-1j * en * h / 2.0)[:, None, :]) @ vec.conj().swapaxes(1, 2)
        frame = evol @ frame
    return frame


def free_A_increment(bundle, t0, t1, n_point):
    fa = bundle.photon.free_A(t1) - bundle.photon.free_A(t0)
    if n_point:
        fa = np.concatenate((fa, np.zeros((2, n_point), complex)), axis=1)
    return fa


def propagate_driven(geo, sample, prot, bundle, gamma, nstep, h):
    """Reuse verified implicit midpoint. No new integrator."""
    n_ph, n_pt = split_counts(bundle)
    state = packed_state(bundle)
    frame = sample["frame"].copy()
    t_samples = np.linspace(0.0, nstep * h, 5)
    idx = set(int(i) for i in np.round(t_samples / h).astype(int))
    times, currents, photon_I, dark_I, frames = [], [], [], [], []

    def record(step):
        t = step * h
        u, d = vertices_from_frame(geo, sample["kc"], prot, sample["neel"], t, frame, sample["weights"])
        ii = current(state, u, d)
        times.append(t)
        currents.append(ii)
        photon_I.append(ii[:, :n_ph] if n_ph else np.zeros((2, 0), complex))
        dark_I.append(ii[:, n_ph:] if n_pt else np.zeros((2, 0), complex))
        frames.append(frame.copy())

    record(0)
    for step in range(nstep):
        t = step * h
        frame = evolve_frame(geo, sample["kc"], prot, sample["neel"], frame, t, h)
        u, d = vertices_from_frame(
            geo, sample["kc"], prot, sample["neel"], t + 0.5 * h, frame, sample["weights"]
        )
        fa = free_A_increment(bundle, t, t + h, n_pt)
        state = midpoint_step(state, u, d, h, gamma, free_A_increment=fa)
        if step + 1 in idx:
            record(step + 1)
    return {
        "times": np.asarray(times, float),
        "current": np.asarray(currents, complex),
        "photon_I": np.asarray(photon_I, complex) if photon_I[0].size else np.zeros((len(times), 2, 0), complex),
        "dark_I": np.asarray(dark_I, complex) if dark_I[0].size else np.zeros((len(times), 2, 0), complex),
        "frames": np.asarray(frames, complex),
        "final_state": state,
        "n_photon": n_ph,
        "n_point": n_pt,
        "n_steps": int(nstep),
        "dt": float(h),
    }


def trapezoid_weights(times):
    t = np.asarray(times, float)
    w = np.zeros_like(t)
    if t.size < 2:
        return w
    dt = np.diff(t)
    w[0] = 0.5 * dt[0]
    w[-1] = 0.5 * dt[-1]
    if t.size > 2:
        w[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    return w


def h2_source_rows(geo, prot, times, current_cols, phi_scale):
    """Same-side H2∥/H2⊥ source rows. chi_rad unset; not a photon number.

    Frozen Fourier: I(Ω)=∫ (W/Tp) exp(+i Ω t) dt. Packet φ lives on ω>0.
    Source kernel uses the frozen two-port sheet map a_src ∝ i I / sqrt(ω),
    then the H2 packet; overall chi_rad remains unset so this is a source
    contraction, not n or m.
    """
    e_par, e_perp = parallel_perp_basis(geo)
    pol = np.stack((e_par, e_perp), axis=0)
    t = np.asarray(times, float)
    i_cols = np.asarray(current_cols, complex)
    window = _sin2_window(t, prot.t_pulse)
    wgt = trapezoid_weights(t) * (window / prot.t_pulse)
    omega_c = 2.0 * prot.omega0
    phase = np.exp(1j * omega_c * t)
    filt = wgt * phase
    i_win = np.einsum("t,tac->ac", filt, i_cols)
    i_pol = pol @ i_win
    src = 1j * phi_scale * i_pol
    gram = src @ src.conj().T
    return {
        "source_row": src,
        "windowed_current_cartesian": i_win,
        "windowed_current_parallel_perp": i_pol,
        "gram": gram,
        "polarization": {"e_parallel": e_par.tolist(), "e_perp": e_perp.tolist()},
    }


def column_groups(bundle):
    n_ph, n_pt = split_counts(bundle)
    labels = list(bundle.photon.labels)
    groups = {
        "all": slice(0, n_ph + n_pt),
        "photon": slice(0, n_ph),
        "dark": slice(n_ph, n_ph + n_pt),
    }
    even_idx, odd_idx = [], []
    for j, lab in enumerate(labels):
        side = lab[1]
        if side == "R":
            even_idx.append(j)
        elif side == "L":
            odd_idx.append(j)
    return groups, even_idx, odd_idx, n_ph, n_pt


def gram_of(src, idx):
    if len(idx) == 0:
        return np.zeros((src.shape[0], src.shape[0]), complex)
    s = src[:, idx]
    return s @ s.conj().T


def run_domain(geo, prot, gamma, neel):
    t0 = time.perf_counter()
    sample = m0_sample(geo, TWO_K, WEIGHTS, neel)
    nodes = h2_packet_nodes(prot, sample["gap"])
    freqs = np.asarray(nodes["frequencies"], float)
    bundle = assemble_incoming_bundle(sample["gap"], sample["u"], sample["D"], gamma, freqs)
    classified = classify_dark_columns(bundle.dark_info, k_of_index=sample["k_of"])
    nstep = NTPC
    h = prot.period / NTPC
    driven = propagate_driven(geo, sample, prot, bundle, gamma, nstep, h)
    phi_raw = phi_unnormalized(np.array([2.0 * prot.omega0]), 2.0 * prot.omega0, prot.t_pulse)[0]
    phi_scale = complex(phi_raw)
    groups, even_idx, odd_idx, n_ph, n_pt = column_groups(bundle)
    src = h2_source_rows(geo, prot, driven["times"], driven["current"], phi_scale)
    photon_idx = list(range(n_ph))
    dark_idx = list(range(n_ph, n_ph + n_pt))
    grams = {
        "all": pack_c(src["gram"]),
        "photon_only": pack_c(gram_of(src["source_row"], photon_idx)),
        "dark_only": pack_c(gram_of(src["source_row"], dark_idx)),
        "even_photon": pack_c(gram_of(src["source_row"], even_idx)),
        "odd_photon": pack_c(gram_of(src["source_row"], odd_idx)),
    }
    i_dark = driven["dark_I"]
    dark_current_t0 = maxabs(i_dark[0]) if i_dark.size else 0.0
    dark_current_max = maxabs(i_dark) if i_dark.size else 0.0
    dark_became_bright = bool(dark_current_max > 1e-10 and dark_current_t0 < 1e-10)
    return {
        "neel": int(neel),
        "n_el": sample["n_el"],
        "n_k": sample["n_k"],
        "weight_once_residual": sample["weight_once_residual"],
        "min_direct_gap": sample["min_direct_gap"],
        "n_photon_columns": n_ph,
        "n_point_columns": n_pt,
        "n_dark_classified": int(sum(int(r["nullity"]) for r in classified)),
        "dark_spectral_types": clean_types(classified),
        "packet_nodes": nodes,
        "propagation": {
            "ntpc": NTPC,
            "dt": driven["dt"],
            "n_steps": driven["n_steps"],
            "times": driven["times"].tolist(),
            "interval": [0.0, float(nstep * h)],
            "interval_meaning": "first cycle of the UNCHANGED eight-cycle P1; not a shortened pulse",
        },
        "h2_source": {
            "chi_rad": "unset",
            "not_photon_number": True,
            "not_n": True,
            "not_m": True,
            "fourier": "window_fourier e^{+i Omega t} with W/Tp; packet centre 2 omega0",
            "packet_id": PACKET_ID,
            "phi_unnormalized_at_center": {
                "real": float(np.real(phi_scale)),
                "imag": float(np.imag(phi_scale)),
                "abs": float(abs(phi_scale)),
            },
            "windowed_current_cartesian": pack_c(src["windowed_current_cartesian"]),
            "windowed_current_parallel_perp": pack_c(src["windowed_current_parallel_perp"]),
            "grams": grams,
            "polarization": src["polarization"],
        },
        "dark_under_P1": {
            "current_at_t0": dark_current_t0,
            "current_max_over_samples": dark_current_max,
            "P1_zero_dark_not_always_irrelevant": True,
            "dark_current_appeared": dark_became_bright,
        },
        "wall_seconds": time.perf_counter() - t0,
        "_classified": classified,
        "_driven": driven,
        "_src": src,
    }


def clean_types(classified):
    types = {}
    for item in classified:
        types[item["spectral_type"]] = types.get(item["spectral_type"], 0) + int(item["nullity"])
    return types


def public_domain(rec):
    rec = dict(rec)
    rec.pop("_classified", None)
    rec.pop("_driven", None)
    rec.pop("_src", None)
    return rec


def flags_from(plus, minus, gamma):
    flags = {
        "gamma_positive": bool(gamma > 0.0),
        "plus_weight_once": bool(plus["weight_once_residual"] < 1e-12),
        "minus_weight_once": bool(minus["weight_once_residual"] < 1e-12),
        "plus_has_photon_columns": bool(plus["n_photon_columns"] > 0),
        "minus_has_photon_columns": bool(minus["n_photon_columns"] > 0),
        "plus_keeps_dark_columns": bool(plus["n_point_columns"] > 0),
        "minus_keeps_dark_columns": bool(minus["n_point_columns"] > 0),
        "plus_dark_t0_current_small": bool(plus["dark_under_P1"]["current_at_t0"] < 1e-10),
        "minus_dark_t0_current_small": bool(minus["dark_under_P1"]["current_at_t0"] < 1e-10),
        "plus_packet_nodes_kept": bool(len(plus["packet_nodes"]["frequencies"]) > 0),
        "minus_packet_nodes_kept": bool(len(minus["packet_nodes"]["frequencies"]) > 0),
        "same_photon_column_count": bool(plus["n_photon_columns"] == minus["n_photon_columns"]),
        "chi_rad_unset": True,
        "not_production_n": True,
        "not_production_m": True,
        "not_full_pulse": True,
        "not_BZ": True,
        "continuum_electronic_incoming_not_required": True,
        "two_k_not_a_domain_signal": True,
    }
    return flags


def main() -> int:
    t0 = time.perf_counter()
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma = float(gamma_self(geo))
    plus = run_domain(geo, prot, gamma, +1)
    minus = run_domain(geo, prot, gamma, -1)
    flags = flags_from(plus, minus, gamma)
    payload = {
        "status": "p1_h2_finite_kcv_contract_not_production_nm",
        "all_checks_pass": bool(all(flags.values())),
        "pass_flags": flags,
        "decision": {
            "internal_kcv": "BZ_quadrature_and_particle_hole_coordinates",
            "not_external_electronic_bath": True,
            "photon": "true_continuum_in_out_columns",
            "continuum_electronic_incoming": "deferred_low_rank_reduction_not_a_gate",
            "keep_P1_zero_dark_combinations": True,
            "literature_basis": [
                "Yue-Gaarde PRA 101 053411: SBE in Bloch-k for solid HHG",
                "Gardiner-Collett PRA 31 3761: continuum photon in/out",
            ],
            "not_a_claim_that_papers_prove_this_PTAFM_P1_model": True,
        },
        "definition": {
            "weight_once": "u = sqrt(w_k) X; C = u u^dagger; no second w",
            "labels": "original (k,c,v)",
            "D": "occupied mean_d_from_bases(d_ops)",
            "gamma": "q8_m0_gsheet.gamma_self",
            "protocol": prot.protocol_id,
            "k_points_fractional": TWO_K.tolist(),
            "weights": WEIGHTS.tolist(),
            "ntpc": NTPC,
            "not_photon_number": True,
            "chi_rad": "unset",
            "assemble_ground_or_photon_moments": "still_NotImplemented",
        },
        "gamma_self": gamma,
        "domains": {"N+1": public_domain(plus), "N-1": public_domain(minus)},
        "domain_difference_is_not_a_signal": True,
        "legal_electronic_continuum_boundary": False,
        "P1_full_pulse": False,
        "optical_moments_computed": False,
        "production": False,
        "old_json_not_overwritten": OLD_JSON,
        "elapsed_s": time.perf_counter() - t0,
        "source_sha256": {
            str(p.relative_to(REPO)): sha(p)
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_incoming_spectral_completeness_20261004.py",
                REPO / "tools/b0_small_model/q8_m0_continuum_columns_20261004.py",
                REPO / "tools/b0_small_model/q8_m0_gsheet.py",
                REPO / "tools/b0_small_model/q8_m0_k1_packet.py",
                REPO / "tools/b0_small_model/q8_m0_classical.py",
            )
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "contract.json"
    if out_json.exists():
        raise FileExistsError("Refuse overwrite of contract.json")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "all_checks_pass": payload["all_checks_pass"],
        "failed": [k for k, v in flags.items() if not v],
        "gamma_self": gamma,
        "plus": {
            "n_el": plus["n_el"],
            "n_photon": plus["n_photon_columns"],
            "n_point": plus["n_point_columns"],
            "dark_t0": plus["dark_under_P1"]["current_at_t0"],
            "dark_max": plus["dark_under_P1"]["current_max_over_samples"],
            "gram_all_trace": plus["h2_source"]["grams"]["all"]["trace_real"],
            "gram_dark_trace": plus["h2_source"]["grams"]["dark_only"]["trace_real"],
        },
        "minus": {
            "n_el": minus["n_el"],
            "n_photon": minus["n_photon_columns"],
            "n_point": minus["n_point_columns"],
            "dark_t0": minus["dark_under_P1"]["current_at_t0"],
            "dark_max": minus["dark_under_P1"]["current_max_over_samples"],
            "gram_all_trace": minus["h2_source"]["grams"]["all"]["trace_real"],
            "gram_dark_trace": minus["h2_source"]["grams"]["dark_only"]["trace_real"],
        },
        "elapsed_s": payload["elapsed_s"],
        "out": str(out_json),
    }, indent=2), flush=True)
    return 0 if payload["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())




