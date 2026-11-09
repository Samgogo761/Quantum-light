#!/usr/bin/env python3
"""Γ12 bright-basis conjugation fix, static recompute, propagation profile.

Does not overwrite gamma12_ccr_fix.json / n128 / two-k / bz_access JSON.
Does not change D, gamma, Hamiltonian, P1, packet, or initial state.
Does not start a full +N pulse from this file unless the measured budget says so.
"""
from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q8_m0_p1_h2_finite_kcv_gamma12_ccr_fix_20261006 as fx
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import ColumnState, current, midpoint_step, real_frequency_kernel
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import assemble_incoming_bundle
from q8_m0_k1_chi_interface import CHI_STATUS
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import free_A_increment, sha
from q8_m0_s1_kernel import parallel_perp_basis

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_gamma12_conj_prop_20261006"
OLD_FIX = REPO / "results/q_line/p1_h2_finite_kcv_gamma12_ccr_fix_20261006/gamma12_ccr_fix.json"
OLD_N128 = REPO / "results/q_line/p1_h2_finite_kcv_gamma12_ccr_fix_20261006/gamma12_ccr_n128.json"
BZ_JSON = REPO / "results/q_line/p1_h2_finite_kcv_bz_access_20261005/bz_access.json"
PROTECTED = [
    OLD_FIX, OLD_N128, BZ_JSON,
    REPO / "results/q_line/p1_h2_finite_kcv_nm_qualify_20261005/nm_qualify_plus.json",
    REPO / "results/q_line/p1_h2_finite_kcv_nm_tails_20261005/nm_tails.json",
    q.OLD_U0_H2, q.OLD_INPUT_CLOSE, q.EIGHT_CYCLE,
]
q.OUT = OUT
fx.OUT = OUT


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


def public_ccr(rec):
    skip = {"ccr_n", "ccr_a", "B"}
    out = {}
    for k, v in rec.items():
        if k in skip:
            continue
        if isinstance(v, np.ndarray):
            continue
        out[k] = v
    return out


def rss_bytes():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def u0_port_sum(sample, gamma, oms, wts, pol, prot, nrm):
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    acc = np.zeros((2, 2), complex)
    skipped = 0
    ph = np.asarray(q.phi_at(oms, prot, nrm), complex)
    t0 = time.perf_counter()
    for om, dw, phi in zip(oms, wts, ph):
        try:
            ker = real_frequency_kernel(gaps, u, d, float(om))
        except Exception:
            skipped += 1
            continue
        tmat = np.linalg.solve(np.eye(2) + 1j * gamma * ker / float(om), np.eye(2, dtype=complex))
        left = phi * (pol @ tmat)
        right = phi * (pol @ (tmat - np.eye(2)))
        urow = np.hstack((left, right))
        acc += float(dw) * (urow @ urow.conj().T)
    mass = q.packet_mass_dnu(oms, wts, prot, nrm)
    return {
        "n_nodes": int(np.asarray(oms).size),
        "packet_mass_dnu": mass,
        "packet_mass_deficit": float(1.0 - mass),
        "U0_packet_ccr": q.maxabs(acc - np.eye(2)),
        "skipped_poles": int(skipped),
        "used_2x2_T_not_giant_bundle": True,
        "elapsed_s": time.perf_counter() - t0,
        "ok": bool(abs(mass - 1.0) < 0.05 and q.maxabs(acc - np.eye(2)) < 0.05),
    }


def take_mixed(bundle, n_dark_keep):
    ph, pt = bundle.photon.initial, bundle.point
    nd = min(int(n_dark_keep), int(pt.b.shape[1]))
    st = ColumnState(
        np.concatenate((ph.b, pt.b[:, :nd]), axis=1),
        np.concatenate((ph.bdag, pt.bdag[:, :nd]), axis=1),
        np.concatenate((ph.A, pt.A[:, :nd]), axis=1),
    )
    return st, nd, int(ph.b.shape[1])


def profile_block(photon, gamma, initial, n_point, vertices, k_scalar, nstep, h, pol):
    state = initial.copy()
    ncol = int(state.b.shape[1])
    acc_u = np.zeros((2, ncol), complex)
    acc_v = np.zeros((2, ncol), complex)
    t_step = t_fb = t_contract = 0.0
    ii = np.zeros((2, ncol), complex)
    t0 = time.perf_counter()
    for step in range(nstep):
        u, d = vertices[step]
        fa = free_A_increment(photon, step * h, (step + 1) * h, n_point)
        t_a = time.perf_counter()
        prev = state
        state = midpoint_step(state, u, d, h, gamma, free_A_increment=fa)
        t_b = time.perf_counter()
        t_step += t_b - t_a
        mid = ColumnState(
            0.5 * (prev.b + state.b),
            0.5 * (prev.bdag + state.bdag),
            0.5 * (prev.A + state.A),
        )
        t_c = time.perf_counter()
        ii = current(mid, u, d)
        t_d = time.perf_counter()
        t_fb += t_d - t_c
        src = pol @ ii
        acc_u += h * k_scalar[step] * src
        acc_v += h * k_scalar[step] * np.conj(src)
        t_contract += time.perf_counter() - t_d
    return {
        "ncol": ncol,
        "n_point": int(n_point),
        "nstep": int(nstep),
        "wall_s": time.perf_counter() - t0,
        "step_s": t_step,
        "current_s": t_fb,
        "contract_s": t_contract,
        "per_col_step_s": (t_step + t_fb + t_contract) / max(nstep * ncol, 1),
        "final_I_maxabs": q.maxabs(ii),
        "delta_u_maxabs": q.maxabs(acc_u),
        "shared_vertices": True,
        "k_not_split_into_independent_drives": True,
    }


def rss_or_trace():
    rss = rss_bytes()
    cur, peak = tracemalloc.get_traced_memory()
    return {"rss_bytes": rss, "tracemalloc_current": int(cur), "tracemalloc_peak": int(peak)}


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    prot = p1_a2_protocol(q.A_STAR)
    pol = np.vstack(parallel_perp_basis(geo))
    eta = q.eta_of(gamma)
    progress(f"conj+prop start gamma={gamma:.12g}")

    three = fx.canonical_three_mode_regression()
    progress(f"3-mode correct n={three['correct_normal_maxabs']:.3g} a={three['correct_anomalous_maxabs']:.3g} "
             f"wrong a={three['wrong_conj_anomalous_maxabs']:.3g} pass={three['correct_pass']}")

    g12, inv12 = grid_sample(geo, 12, 0.0)
    oms96, wts96 = q.input_quadrature(g12["gap"], g12["u"], gamma, n_panel=96)
    oms128, wts128 = q.input_quadrature(g12["gap"], g12["u"], gamma, n_panel=128)
    progress(f"nodes 96={oms96.size} 128={oms128.size}")

    map_chk = fx.three_freq_map_vs_project(g12, gamma, oms96, wts96)
    progress(f"map vs project b={map_chk['b_map_vs_project_maxabs']:.3g} bd={map_chk['bd_map_vs_project_maxabs']:.3g}")

    stride = max(int(oms96.size // 60), 1)
    cov = fx.complex_basis_covariance(g12, gamma, oms96[::stride], wts96[::stride])
    progress(f"covariance N={cov['normal_covariance_maxabs']:.3g} A={cov['anomalous_covariance_maxabs']:.3g} pass={cov['pass']}")

    rec96 = fx.mapped_bright_ccr(g12, gamma, oms96, wts96)
    progress(
        f"n96 CCR_n={rec96['electronic_ccr_normal_maxabs']:.5g} "
        f"CCR_a={rec96['electronic_ccr_anomalous_maxabs']:.5g} "
        f"tr_bb-dd={rec96['trace_bb_minus_trace_dd']:.5g} t={rec96['elapsed_s']:.2f}s"
    )
    rec128 = fx.mapped_bright_ccr(g12, gamma, oms128, wts128)
    progress(
        f"n128 CCR_n={rec128['electronic_ccr_normal_maxabs']:.5g} "
        f"CCR_a={rec128['electronic_ccr_anomalous_maxabs']:.5g} "
        f"tr_bb-dd={rec128['trace_bb_minus_trace_dd']:.5g} t={rec128['elapsed_s']:.2f}s"
    )

    kernel = q.detector_kernel_scalar(gamma, prot, 64, prot.period / q.NTPC)
    nrm = kernel["normalizer"]
    u0_96 = u0_port_sum(g12, gamma, oms96, wts96, pol, prot, nrm)
    u0_128 = u0_port_sum(g12, gamma, oms128, wts128, pol, prot, nrm)
    progress(f"U0 n96 mass={u0_96['packet_mass_dnu']:.12f} ccr={u0_96['U0_packet_ccr']:.3g}")

    proj = fx.projected_resolvent_check(g12, gamma, oms96, wts96, q.COMPLEX_Z)
    progress(f"projected resolvent worst_rel={proj['worst_relative']:.4g} pass={proj['check_passed']}")

    refine_tr = abs(rec96["trace_bb_minus_trace_dd"] - rec128["trace_bb_minus_trace_dd"]) / max(
        abs(rec128["trace_bb_minus_trace_dd"]), 1e-30
    )
    gate96 = bool(
        rec96["electronic_ccr_normal_maxabs"] < 0.05
        and rec96["electronic_ccr_anomalous_maxabs"] < 0.05
        and rec128["electronic_ccr_normal_maxabs"] < 0.05
        and rec128["electronic_ccr_anomalous_maxabs"] < 0.05
        and refine_tr < 0.05
        and three["correct_pass"]
        and cov["pass"]
        and u0_96["ok"]
        and rec96["bright_plus_dark"] == rec96["n_el"]
    )
    progress(f"static gate96={gate96} refine_tr_bb_minus_dd={refine_tr:.4g}")

    h = prot.period / q.NTPC
    nstep = 16
    t_v0 = time.perf_counter()
    vertices = list(zip(*q.midpoint_vertices(geo, g12, prot, nstep, h)))
    t_vert = time.perf_counter() - t_v0
    k_scalar = q.detector_kernel_scalar(gamma, prot, nstep, h)["k_scalar"]
    progress(f"vertices {nstep} steps {t_vert:.3f}s")

    blocks = []
    for nfreq, ndark, tag in ((8, 32, "ph32_dark32"), (32, 128, "ph128_dark128")):
        t_pre = time.perf_counter()
        bundle = assemble_incoming_bundle(g12["gap"], g12["u"], g12["D"], gamma, oms96[:nfreq])
        st, nd, nph = take_mixed(bundle, ndark)
        t_pre = time.perf_counter() - t_pre
        mem0 = rss_or_trace()
        rec = profile_block(bundle.photon, gamma, st, nd, vertices, k_scalar, nstep, h, pol)
        rec["tag"] = tag
        rec["n_photon"] = nph
        rec["n_freq"] = int(nfreq)
        rec["preprocess_s"] = t_pre
        rec["vertices_s_shared"] = t_vert
        rec["memory_after"] = rss_or_trace()
        rec["memory_before"] = mem0
        blocks.append(rec)
        progress(f"{tag} ncol={rec['ncol']} wall={rec['wall_s']:.3f}s per_col_step={rec['per_col_step_s']:.3e}s")

    n_freq_full = int(oms96.size)
    n_ph_full = 4 * n_freq_full
    n_dark_full = int(rec96["n_dark"])
    ncol_full = n_ph_full + n_dark_full
    nstep_full = int(prot.n_cycles * q.NTPC + q.NTPC)
    per = max(blocks[-1]["per_col_step_s"], blocks[0]["per_col_step_s"])
    t_vert_per = t_vert / nstep
    est_s = nstep_full * (t_vert_per + per * ncol_full)
    est_h = est_s / 3600.0
    progress(f"extrapolate eight+tail ncol={ncol_full} nstep={nstep_full} -> {est_h:.2f} h")

    env = fx.scan_python_envs()
    start_plus = bool(gate96 and est_h < 3.0 and env.get("gpu_backend_ready"))
    if not gate96:
        stop_reason = "static n96 gate failed after conjugation fix"
    elif est_h >= 3.0:
        stop_reason = (
            f"measured-block extrapolation {est_h:.1f} h CPU for +N eight+tail; "
            "no GPU backend; not starting unbudgeted long job"
        )
    else:
        stop_reason = "budget ok on paper but GPU regression absent"

    payload = {
        "status": "p1_h2_gamma12_bright_conj_fix_and_profile_not_domain",
        "old_bright_anomalous_flag_superseded": True,
        "conjugation_fix": "bd_B = B.T @ bd; b_B = B.conj().T @ b",
        "working_grid": "n_panel=96",
        "n128_is_refine_only": True,
        "canonical_three_mode": three,
        "map_vs_project": map_chk,
        "complex_basis_covariance": {k: v for k, v in cov.items()},
        "n96": public_ccr(rec96),
        "n128": public_ccr(rec128),
        "refine_trace_bb_minus_dd": refine_tr,
        "u0": {"n96": u0_96, "n128": u0_128},
        "projected_resolvent_n96": proj,
        "static_gate_n96": gate96,
        "gamma12": {"n_el": g12["n_el"], "n_k": g12["n_k"], "inversion": inv12},
        "propagation_profile": {
            "nstep_short": nstep,
            "dt": h,
            "shared_vertices_once": True,
            "blocks": blocks,
            "n_freq_working": n_freq_full,
            "n_photon_full": n_ph_full,
            "n_dark_full": n_dark_full,
            "ncol_full": ncol_full,
            "nstep_eight_plus_one_tail": nstep_full,
            "est_full_seconds": est_s,
            "est_full_hours": est_h,
            "not_from_two_k_9s_scale": True,
            "not_from_32freq_assemble_only": True,
            "cache_vertices_reused_across_blocks": True,
        },
        "gpu_env": env,
        "start_plusN": start_plus,
        "stop_reason": stop_reason,
        "chi_rad_placeholder_status": CHI_STATUS,
        "old_json_not_overwritten": [str(Path(p).relative_to(REPO)) for p in PROTECTED if p is not None],
        "old_fix_sha256": sha(OLD_FIX) if OLD_FIX.exists() else None,
        "old_n128_sha256": sha(OLD_N128) if OLD_N128.exists() else None,
        "elapsed_s": time.perf_counter() - t0,
        "production": False,
        "n_m_claimed_as_domain_signal": False,
        "did_not_change_D_or_initial_or_gamma": True,
    }
    out = write_json("gamma12_conj_prop.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "gate96": gate96,
        "n96_CCR_n": rec96["electronic_ccr_normal_maxabs"],
        "n96_CCR_a": rec96["electronic_ccr_anomalous_maxabs"],
        "n128_CCR_n": rec128["electronic_ccr_normal_maxabs"],
        "n128_CCR_a": rec128["electronic_ccr_anomalous_maxabs"],
        "U0_ccr": u0_96["U0_packet_ccr"],
        "est_hours": est_h,
        "start_plusN": start_plus,
        "stop_reason": stop_reason,
        "out": str(out),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
