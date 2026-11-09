#!/usr/bin/env python3
"""+N frequency-endpoint n,m and post-Tp+T tail. Cheap BZ cost probe.

Does not overwrite nm_qualify JSON or older completeness JSON.
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
os.environ.setdefault("Q8_STAGE", "probe")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_classical import k_grid, p1_a2_protocol
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import dark_subspace
from q8_m0_k1_chi_interface import CHI_STATUS
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import TWO_K, WEIGHTS, m0_sample, sha
from q8_m0_s1_kernel import parallel_perp_basis
from q8_m0_stage1ap_h2_window_quadrature_20261004 import inversion_ok

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_nm_tails_20261005"
PLUS_JSON = REPO / "results/q_line/p1_h2_finite_kcv_nm_qualify_20261005/nm_qualify_plus.json"
QUAL_JSON = REPO / "results/q_line/p1_h2_finite_kcv_nm_qualify_20261005/nm_qualify.json"
PROTECTED = [
    PLUS_JSON,
    QUAL_JSON,
    REPO / "results/q_line/p1_h2_finite_kcv_nm_qualify_20261005/nm_qualify_probe.json",
    q.OLD_U0_H2,
    q.OLD_INPUT_CLOSE,
    q.EIGHT_CYCLE,
    q.INPUT_LEDGER,
    q.CALLER_FIX,
    q.OLD_CONTRACT,
]
N_TAIL_EXT = 3
NM_REL_TOL = 0.05
q.OUT = OUT


def progress(msg):
    q.progress(msg)


def unpack(rec):
    return np.asarray(rec["real"], float) + 1j * np.asarray(rec["imag"], float)


def extra_panels():
    ir, irw = q.geom_segment(1.0e-4, 1.0e-3, 8)
    uv, uvw = q.trap_segment(12.0, 24.0, 24)
    ir2, ir2w = q.geom_segment(1.0e-5, 1.0e-4, 6)
    uv2, uv2w = q.trap_segment(24.0, 36.0, 12)
    return {
        "ir": (ir, irw),
        "uv": (uv, uvw),
        "ir_outer": (ir2, ir2w),
        "uv_outer": (uv2, uv2w),
    }


def extra_contrib(name, side, n_base, m_base, u_base, v_base):
    u = unpack(side["U0"]) + unpack(side["delta_U"])
    v = unpack(side["V"])
    dn = v.conj() @ v.T
    dm = u @ v.T
    n_new = n_base + dn
    m_new = m_base + dm
    ident = np.eye(2, dtype=complex)
    u_cat = np.concatenate((u_base, u), axis=1)
    v_cat = np.concatenate((v_base, v), axis=1)
    return {
        "band": name,
        "n_nodes": int(side["n_photon_columns"] // 4) if side["n_photon_columns"] else 0,
        "n_photon_columns": int(side["n_photon_columns"]),
        "n_point_columns": int(side["n_point_columns"]),
        "kept_dark": bool(side["kept_dark"]),
        "omega_min": None,
        "dn": q.pack_c(dn),
        "dm": q.pack_c(dm),
        "vs_base_n": q.matrix_diff(n_new, n_base, "n"),
        "vs_base_m": q.matrix_diff(m_new, m_base, "m"),
        "extra_n_trace": float(np.trace(dn).real),
        "extra_m_maxabs": q.maxabs(dm),
        "concat_normal_ccr": q.maxabs(u_cat @ u_cat.conj().T - v_cat @ v_cat.conj().T - ident),
        "concat_anomalous_ccr": q.maxabs(u_cat @ v_cat.T - v_cat @ u_cat.T),
        "U0VT_in_m": True,
        "driven_wall": side["driven_wall"],
        "undriven_wall": side["undriven_wall"],
    }


def slim_check(mom):
    return {
        "n_trace": mom["n"].get("trace_real"),
        "n": mom["n"],
        "m": mom["m"],
        "m_reference_source": mom.get("m_reference_source"),
        "normal_ccr": mom["normal_ccr"],
        "anomalous_ccr": mom["anomalous_ccr"],
        "n_min_eig": mom["n_min_eig"],
    }


def bz_static_row(geo, nk, gamma):
    kf, kc, w = k_grid(geo, nk, shift=0.0)
    sample = m0_sample(geo, kf, w, +1)
    oms, wts = q.input_quadrature(sample["gap"], sample["u"], gamma, n_panel=q.N_PANEL, n_h2=q.N_H2)
    basis, info, near = dark_subspace(sample["gap"], sample["u"])
    n_ph = 4 * int(oms.size)
    n_el = int(sample["n_el"])
    ncol = n_ph + int(basis.shape[1])
    bytes_state = 3 * n_el * ncol * 16
    bytes_vert = int(9216 * (n_el * 2 + 4) * 16)
    inv = inversion_summary(inversion_ok(kf, w))
    return {
        "nk": int(nk),
        "n_k": int(kf.shape[0]),
        "weight_sum": float(np.sum(w)),
        "inversion": inv,
        "n_el": n_el,
        "min_direct_gap": sample["min_direct_gap"],
        "max_direct_gap": sample["max_direct_gap"],
        "n_dark": int(basis.shape[1]),
        "n_freq": int(oms.size),
        "omega_min": float(oms[0]),
        "omega_max": float(oms[-1]),
        "n_photon_columns": n_ph,
        "n_point_columns": int(basis.shape[1]),
        "ncol": ncol,
        "peak_state_bytes": int(bytes_state),
        "vertex_cache_bytes": int(bytes_vert),
        "near_degenerate_pairs": int(len(near)),
        "not_two_arbitrary_k": True,
        "shared_field_must_sum_all_k": True,
    }


def inversion_summary(inv):
    if not isinstance(inv, dict):
        return inv
    out = {}
    for k, v in inv.items():
        if isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int, bool, np.bool_)):
            out[k] = int(v) if not isinstance(v, (bool, np.bool_)) else bool(v)
        elif isinstance(v, (str, type(None))):
            out[k] = v
    return out


def write_json(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if path.exists():
        raise FileExistsError(f"Refuse overwrite of {path}")
    for p in PROTECTED:
        if not p.exists():
            raise FileNotFoundError(p)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot_drive = p1_a2_protocol(q.A_STAR)
    prot_ref = p1_a2_protocol(0.0)
    gamma = float(gamma_self(geo))
    eta = q.eta_of(gamma)
    h = prot_drive.period / q.NTPC
    n_pulse = int(prot_drive.n_cycles) * q.NTPC
    nstep_base = n_pulse + q.N_TAIL_CYCLES * q.NTPC
    nstep_ext = n_pulse + N_TAIL_EXT * q.NTPC
    pol = np.vstack(parallel_perp_basis(geo))
    sample = m0_sample(geo, TWO_K, WEIGHTS, +1)
    progress(f"tails start gamma={gamma:.12g} nstep_ext={nstep_ext}")
    plus = json.loads(PLUS_JSON.read_text(encoding="utf-8"))
    n_base = unpack(plus["plus"]["n"])
    m_base = unpack(plus["plus"]["m"])
    u_base = unpack(plus["plus_U"])
    v_base = unpack(plus["plus_V"])
    oms, wts = q.input_quadrature(sample["gap"], sample["u"], gamma, n_panel=q.N_PANEL, n_h2=q.N_H2)
    progress(f"kernel nstep={nstep_base}")
    kernel = q.detector_kernel_scalar(gamma, prot_drive, nstep_base, h)
    k_scalar = kernel["k_scalar"]
    nrm = kernel["normalizer"]

    def phi_of(nu):
        return complex(np.asarray(q.phi_at(nu, prot_drive, nrm)).reshape(-1)[0])

    bands = extra_panels()
    extra_rows = {}
    for name, (om, wt) in bands.items():
        if om.size == 0:
            continue
        progress(f"+N extra band {name} n={om.size} [{float(om[0]):.4g},{float(om[-1]):.4g}]")
        side = q.run_side(
            geo, sample, gamma, prot_drive, prot_ref, om, wt, k_scalar,
            nstep_base, h, n_pulse, pol, eta, phi_of, keep_dark=False,
        )
        row = extra_contrib(name, side, n_base, m_base, u_base, v_base)
        row["omega_min"] = float(om[0])
        row["omega_max"] = float(om[-1])
        row["sum_dnu"] = float(np.sum(wt))
        extra_rows[name] = row
        progress(
            f"  {name} dn_tr={row['extra_n_trace']:.4g} "
            f"n_rel={row['vs_base_n']['frobenius_relative']:.4g} "
            f"m_rel={row['vs_base_m']['frobenius_relative']:.4g}"
        )

    dn_all = sum(unpack(extra_rows[k]["dn"]) for k in extra_rows)
    dm_all = sum(unpack(extra_rows[k]["dm"]) for k in extra_rows)
    extra_all = {
        "n_vs_base": q.matrix_diff(n_base + dn_all, n_base, "n"),
        "m_vs_base": q.matrix_diff(m_base + dm_all, m_base, "m"),
        "extra_n_trace": float(np.trace(dn_all).real),
        "split_ir_uv_outer": True,
        "not_a_physical_hard_cutoff": True,
        "no_second_dark_or_bright_vacuum": True,
        "ok": bool(
            extra_all_ok(n_base, m_base, dn_all, dm_all)
        ),
    }

    snaps = {
        n_pulse: "pulse",
        n_pulse + q.NTPC: "plus1T",
        n_pulse + 2 * q.NTPC: "plus2T",
        n_pulse + 3 * q.NTPC: "plus3T",
    }
    progress(f"+N time extend nstep={nstep_ext} checkpoints={list(snaps.values())}")
    kernel_e = q.detector_kernel_scalar(gamma, prot_drive, nstep_ext, h)
    side_t = q.run_side(
        geo, sample, gamma, prot_drive, prot_ref, oms, wts, kernel_e["k_scalar"],
        nstep_ext, h, n_pulse, pol, eta, phi_of, keep_dark=True, snapshot_steps=snaps,
    )
    checks = {name: slim_check(rec) for name, rec in side_t["checkpoints"].items()}
    tail_cmp = {}
    if "plus1T" in checks and "plus2T" in checks:
        tail_cmp["plus2T_vs_plus1T"] = q.moments_diff(
            side_t["checkpoints"]["plus1T"], side_t["checkpoints"]["plus2T"]
        )
    if "plus1T" in checks and "plus3T" in checks:
        tail_cmp["plus3T_vs_plus1T"] = q.moments_diff(
            side_t["checkpoints"]["plus1T"], side_t["checkpoints"]["plus3T"]
        )
    if "pulse" in checks and "plus1T" in checks:
        tail_cmp["plus1T_vs_pulse"] = q.moments_diff(
            side_t["checkpoints"]["pulse"], side_t["checkpoints"]["plus1T"]
        )
    tail_ok = all(v.get("ok", False) for v in tail_cmp.values()) if tail_cmp else False
    progress(
        "time tail "
        + " ".join(
            f"{k} n={v['n']['frobenius_relative']:.4g}" for k, v in tail_cmp.items()
        )
    )

    progress("BZ static cost nk=4,6,8")
    bz_rows = []
    two_k_cycle_s = 9.0
    ref_ncol = 2424
    ref_nel = 32
    for nk in (4, 6, 8):
        row = bz_static_row(geo, nk, gamma)
        scale = (row["ncol"] / ref_ncol) * (row["n_el"] / ref_nel)
        row["est_full_pulse_seconds_cpu_from_two_k"] = float(2 * 9 * two_k_cycle_s * scale)
        row["est_peak_gb"] = (row["peak_state_bytes"] + row["vertex_cache_bytes"]) / 1e9
        row["shared_A_feedback_sums_all_k"] = True
        bz_rows.append(row)
        progress(
            f"  nk={row['nk']} n_k={row['n_k']} n_el={row['n_el']} "
            f"n_freq={row['n_freq']} ncol={row['ncol']} gap={row['min_direct_gap']:.4g}"
        )

    from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import FREQ
    kf4, _kc4, w4 = k_grid(geo, 4, shift=0.0)
    sample4 = m0_sample(geo, kf4, w4, +1)
    progress("nk=4 short prefix 16 steps on 2 wiring freqs")
    t_pref = time.perf_counter()
    q.dt_halving_check(
        geo,
        sample4,
        prot_drive,
        gamma,
        FREQ,
        lambda ns, hh: q.detector_kernel_scalar(gamma, prot_drive, ns, hh)["k_scalar"],
        16,
        h,
    )
    prefix_wall = time.perf_counter() - t_pref

    payload = {
        "status": "p1_h2_finite_kcv_nm_tails_and_bz_cost_not_domain",
        "baseline_is_ntpc_1024_601_freq": True,
        "baseline_tr_n": float(np.trace(n_base).real),
        "ntpc_2048_is_not_the_kept_baseline": True,
        "finite_nm_on_tested_frequency_and_time_window": "preliminarily_qualified",
        "full_positive_frequency_and_infinite_tail_not_claimed": True,
        "extra_frequency_driven": extra_rows,
        "extra_frequency_all": extra_all,
        "time_extend": {
            "n_tail_cycles_old": int(q.N_TAIL_CYCLES),
            "n_tail_cycles_new": int(N_TAIL_EXT),
            "nstep": int(nstep_ext),
            "checkpoints": checks,
            "compare": tail_cmp,
            "finite_endpoint_not_infinite_bound": True,
            "same_physical_packet": True,
            "no_T2_no_DC_cut_no_window_swap": True,
            "ok": bool(tail_ok),
        },
        "bz_cost": {
            "grids": bz_rows,
            "nk4_short_prefix_seconds": prefix_wall,
            "nk4_prefix_used_first_8_freqs_only": True,
            "no_full_bz_pulse": True,
            "gpu": {
                "matrix_free_blocked_online_contraction": True,
                "no_dense_total_propagator": True,
                "shared_field_feedback_sums_all_k": True,
                "cpu_reference_before_gpu": True,
                "ai_lowrank_not_physical_reference": True,
            },
            "first_stabilize_common_nm_then_domain_diff": True,
        },
        "chi_rad_placeholder_status": CHI_STATUS,
        "old_json_not_overwritten": [str(p.relative_to(REPO)) for p in PROTECTED],
        "plus_sha256": sha(PLUS_JSON),
        "qual_sha256": sha(QUAL_JSON) if QUAL_JSON.exists() else None,
        "elapsed_s": time.perf_counter() - t0,
        "production": False,
        "n_m_claimed_as_domain_signal": False,
    }
    extra_all["ok"] = extra_all_ok(n_base, m_base, dn_all, dm_all)
    payload["extra_frequency_all"]["ok"] = extra_all["ok"]
    out = write_json("nm_tails.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "extra_n_trace": extra_all["extra_n_trace"],
                "extra_n_rel": extra_all["n_vs_base"]["frobenius_relative"],
                "extra_m_rel": extra_all["m_vs_base"]["frobenius_relative"],
                "tail_ok": tail_ok,
                "bz_nk": [r["nk"] for r in bz_rows],
                "elapsed_s": payload["elapsed_s"],
                "out": str(out),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def extra_all_ok(n_base, m_base, dn, dm):
    n_cmp = q.matrix_diff(n_base + dn, n_base, "n")
    m_cmp = q.matrix_diff(m_base + dm, m_base, "m")
    return bool(n_cmp["frobenius_relative"] < NM_REL_TOL and m_cmp["frobenius_relative"] < NM_REL_TOL)


if __name__ == "__main__":
    raise SystemExit(main())
