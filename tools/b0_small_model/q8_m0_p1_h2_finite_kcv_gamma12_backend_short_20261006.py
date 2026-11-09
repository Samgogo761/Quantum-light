#!/usr/bin/env python3
"""Γ12 backend short test: mixed frequencies, dark, driven/undriven, CPU blocks, optional CUDA.

Does not rerun static 96/128 CCR. Does not overwrite old JSON.
Does not change D, gamma, Hamiltonian, P1, packet, or initial state.
Does not start a 9216-step full +N job.
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
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_classical import kappa_shift, p1_a2_protocol
from q8_m0_column_step_torch_20261006 import cpu_accumulate, gpu_probe
from q8_m0_continuum_columns_20261004 import ColumnState, stationary_photon_columns
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import (
    dark_subspace,
    stationary_dark_basis_columns,
)
from q8_m0_k1_chi_interface import CHI_STATUS
from q8_m0_k1_packet import normalize_phi, omega_grid, phi_unnormalized
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import (
    advance_half_frame,
    free_A_increment,
    sha,
    vertices_from_frame,
)
from q8_m0_s1_kernel import parallel_perp_basis

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("Q8_BACKEND_OUT", str(REPO / "results/q_line/p1_h2_finite_kcv_gamma12_backend_cuda_resume_20261006")))
SUMMARY = REPO / "results/q_line/p1_h2_finite_kcv_gamma12_conj_prop_20261006/gamma12_static_accept_summary.json"
PROTECTED = [
    SUMMARY,
    REPO / "results/q_line/p1_h2_finite_kcv_gamma12_conj_prop_20261006/gamma12_conj_prop.json",
    REPO / "results/q_line/p1_h2_finite_kcv_gamma12_ccr_fix_20261006/gamma12_ccr_fix.json",
    REPO / "results/q_line/p1_h2_finite_kcv_bz_access_20261005/bz_access.json",
    q.OLD_U0_H2, q.OLD_INPUT_CLOSE, q.EIGHT_CYCLE,
]
q.OUT = OUT
ATOL, RTOL = 1e-10, 1e-8
NSTEP = 16
N_DARK_KEEP = 64
PUMP_STEP0 = 2 * q.NTPC


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


def peak_wset():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", info.rss)), int(info.rss)
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
        GetCurrentProcess.restype = ctypes.c_void_p
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        getinfo = ctypes.windll.psapi.GetProcessMemoryInfo
        getinfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        getinfo.restype = wintypes.BOOL
        ok = getinfo(GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        if not ok:
            return None, None
        return int(pmc.PeakWorkingSetSize), int(pmc.WorkingSetSize)
    except Exception:
        return None, None


def arr_err(a, b, atol=ATOL, rtol=RTOL):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = q.maxabs(a - b)
    scale = max(q.maxabs(a), q.maxabs(b), 1e-30)
    elem = bool(np.all(np.abs(a - b) <= atol + rtol * np.abs(b)))
    return {
        "max_absolute": diff,
        "max_relative": diff / scale,
        "elem_atol_rtol_pass": elem,
        "reference_max": scale,
    }


def pick_band(oms, lo, hi, n):
    m = np.asarray(oms, float)
    m = m[(m >= float(lo)) & (m < float(hi))]
    if m.size == 0:
        return np.zeros(0)
    n = min(int(n), int(m.size))
    idx = np.unique(np.linspace(0, m.size - 1, n).round().astype(int))
    return m[idx]


def kernel_at_times(gamma, prot, t_mid):
    eta = q.eta_of(gamma)
    w0 = float(prot.omega0)
    tp = float(prot.t_pulse)
    om = omega_grid(w0, tp)
    om = om[om > 0.0]
    raw = phi_unnormalized(om, 2.0 * w0, tp)
    phi, nrm = normalize_phi(om, raw)
    t_mid = np.asarray(t_mid, float)
    weight = phi / np.sqrt(om)
    k = np.empty(t_mid.size, complex)
    batch = 64
    for s in range(0, t_mid.size, batch):
        sl = slice(s, min(s + batch, t_mid.size))
        phase = np.exp(1j * om[:, None] * t_mid[sl][None, :])
        k[sl] = 1j * eta * np.trapezoid(weight[:, None] * phase, om, axis=0)
    return k, float(nrm)


def vertices_window(geo, sample, prot, nstep, h, step0=0):
    frame = sample["frame"].copy()
    rec = []
    total = int(step0) + int(nstep)
    for step in range(total):
        t = step * h
        for which, when in enumerate((t + h / 4.0, t + 3.0 * h / 4.0)):
            frame = advance_half_frame(geo, sample["kc"], prot, sample["neel"], frame, when, h)
            if which == 0 and step >= int(step0):
                rec.append(vertices_from_frame(
                    geo, sample["kc"], prot, sample["neel"], t + 0.5 * h, frame, sample["weights"]
                ))
    return rec


def pack_photon_dark(photon, dark_state, n_dark):
    nd = min(int(n_dark), int(dark_state.b.shape[1]))
    st = ColumnState(
        np.concatenate((photon.initial.b, dark_state.b[:, :nd]), axis=1),
        np.concatenate((photon.initial.bdag, dark_state.bdag[:, :nd]), axis=1),
        np.concatenate((photon.initial.A, dark_state.A[:, :nd]), axis=1),
    )
    return st, nd


def slice_state(state, sl):
    return ColumnState(state.b[:, sl], state.bdag[:, sl], state.A[:, sl])


def compare_path(cpu, other, tag):
    rows = {}
    for name in ("b", "bdag", "A"):
        rows[name] = arr_err(getattr(cpu["state"], name), getattr(other["state"], name))
    rows["I"] = arr_err(cpu["final_I"], other["final_I"])
    rows["delta_u"] = arr_err(cpu["delta_u"], other["delta_u"])
    rows["v"] = arr_err(cpu["v"], other["v"])
    ok = all(r["elem_atol_rtol_pass"] for r in rows.values())
    return {"tag": tag, "ok": ok, "parts": rows}


def run_pair(photon, gamma, initial, n_pt, kd, ku, vd, vu, nstep, h, pol, t0):
    drec = cpu_accumulate(photon, gamma, initial, n_pt, kd, vd, nstep, h, pol, free_A_increment, t0=t0)
    urec = cpu_accumulate(photon, gamma, initial.copy(), n_pt, ku, vu, nstep, h, pol, free_A_increment, t0=t0)
    return drec, urec, {
        "delta_u": drec["delta_u"] - urec["delta_u"],
        "v": drec["v"] - urec["v"],
        "driven_wall_s": drec["wall_s"],
        "undriven_wall_s": urec["wall_s"],
        "sum_wall_s": drec["wall_s"] + urec["wall_s"],
    }


def main() -> int:
    t_all = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    prot_d = p1_a2_protocol(q.A_STAR)
    prot_u = p1_a2_protocol(0.0)
    pol = np.vstack(parallel_perp_basis(geo))
    h = prot_d.period / q.NTPC
    progress("backend short start; skip static CCR")

    g12, _inv = grid_sample(geo, 12, 0.0)
    oms, wts = q.input_quadrature(g12["gap"], g12["u"], gamma, n_panel=96)
    bands = {
        "low": pick_band(oms, 1e-3, 0.02, 8),
        "h2": pick_band(oms, 0.02, 0.30, 16),
        "mid": pick_band(oms, 0.40, 2.50, 16),
        "high": pick_band(oms, 6.0, 12.0, 8),
    }
    freq = np.concatenate([bands[k] for k in ("low", "h2", "mid", "high")])
    dnu = {round(float(o), 12): float(w) for o, w in zip(oms, wts)}
    wsel = np.array([dnu[round(float(o), 12)] for o in freq], float)
    progress(f"freq n={freq.size} low={bands['low'].size} h2={bands['h2'].size} mid={bands['mid'].size} high={bands['high'].size}")

    t_pre = time.perf_counter()
    photon = stationary_photon_columns(g12["gap"], g12["u"], g12["D"], gamma, freq)
    dark_basis, info, _n = dark_subspace(g12["gap"], g12["u"])
    dark_state, _labs, _pg = stationary_dark_basis_columns(g12["gap"], g12["u"], dark_basis, info)
    class _Bundle:
        def __init__(self, ph):
            self.photon = ph
            self.n_photon_columns = int(ph.initial.b.shape[1])

    q.attach_weights(_Bundle(photon), dnu)
    initial, n_pt = pack_photon_dark(photon, dark_state, N_DARK_KEEP)
    t_pre = time.perf_counter() - t_pre
    ncol = int(initial.b.shape[1])
    n_ph = int(photon.initial.b.shape[1])
    progress(f"assembled n_ph={n_ph} n_dark={n_pt} ncol={ncol} pre={t_pre:.2f}s")

    t_mid0 = (np.arange(NSTEP) + 0.5) * h
    t_midp = (PUMP_STEP0 + np.arange(NSTEP) + 0.5) * h
    k0, _ = kernel_at_times(gamma, prot_d, t_mid0)
    kp, _ = kernel_at_times(gamma, prot_d, t_midp)
    t_a_peak = float(np.max(np.linalg.norm(kappa_shift(t_midp, geo, prot_d, +1), axis=-1)))
    t_a_start = float(np.max(np.linalg.norm(kappa_shift(t_mid0, geo, prot_d, +1), axis=-1)))
    progress(f"|a(t)| max start={t_a_start:.4g} pump_win={t_a_peak:.4g}")

    t_v = time.perf_counter()
    vd0 = vertices_window(geo, g12, prot_d, NSTEP, h, 0)
    vu0 = vertices_window(geo, g12, prot_u, NSTEP, h, 0)
    vdp = vertices_window(geo, g12, prot_d, NSTEP, h, PUMP_STEP0)
    vup = vertices_window(geo, g12, prot_u, NSTEP, h, PUMP_STEP0)
    t_vert = time.perf_counter() - t_v
    progress(f"vertices 4 windows {t_vert:.2f}s")

    peak0, rss0 = peak_wset()
    d0, u0, m0 = run_pair(photon, gamma, initial, n_pt, k0, k0, vd0, vu0, NSTEP, h, pol, 0.0)
    dp, up, mp = run_pair(photon, gamma, initial, n_pt, kp, kp, vdp, vup, NSTEP, h, pol, PUMP_STEP0 * h)
    peak1, rss1 = peak_wset()
    progress(
        f"cpu start driven={d0['wall_s']:.3f}s undriven={u0['wall_s']:.3f}s; "
        f"pump driven={dp['wall_s']:.3f}s undriven={up['wall_s']:.3f}s"
    )

    n1 = (n_ph // 8) * 4
    if n1 < 4:
        n1 = min(4, n_ph)

    class PhSlice:
        def __init__(self, ph, sl):
            om = np.asarray(ph.omega, float)
            self.omega = om[sl]
            self.free_A0 = np.asarray(ph.free_A0, complex)[:, sl]

        def free_A(self, t):
            return self.free_A0 * np.exp(-1j * self.omega * float(t))[None, :]

    init_a = ColumnState(
        np.concatenate((initial.b[:, :n1], initial.b[:, n_ph:n_ph + n_pt]), axis=1),
        np.concatenate((initial.bdag[:, :n1], initial.bdag[:, n_ph:n_ph + n_pt]), axis=1),
        np.concatenate((initial.A[:, :n1], initial.A[:, n_ph:n_ph + n_pt]), axis=1),
    )
    init_b = slice_state(initial, slice(n1, n_ph))
    ph_a = PhSlice(photon, slice(0, n1))
    ph_b = PhSlice(photon, slice(n1, n_ph))
    # Strong-window algebraic backend check. Do not gate on a near-zero t=0 row.
    da, ua, ma = run_pair(ph_a, gamma, init_a, n_pt, kp, kp, vdp, vup, NSTEP, h, pol, PUMP_STEP0*h)
    db, ub, mb = run_pair(ph_b, gamma, init_b, 0, kp, kp, vdp, vup, NSTEP, h, pol, PUMP_STEP0*h)
    def merge_arr(a,b):
        return np.concatenate((a[:, :n1], b, a[:, n1:]), axis=1)
    def merged_path(a,b):
        return {"state": ColumnState(*(merge_arr(getattr(a["state"],k),getattr(b["state"],k)) for k in ("b","bdag","A"))),
                "final_I": merge_arr(a["final_I"],b["final_I"]),
                "delta_u": merge_arr(a["delta_u"],b["delta_u"]), "v": merge_arr(a["v"],b["v"])}
    block_vs_full = {
        "driven": compare_path(dp,merged_path(da,db),"pump_split_driven"),
        "undriven": compare_path(up,merged_path(ua,ub),"pump_split_undriven"),
        "delta_u": arr_err(mp["delta_u"],merge_arr(ma["delta_u"],mb["delta_u"])),
        "v": arr_err(mp["v"],merge_arr(ma["v"],mb["v"])),
    }
    block_ok = all(block_vs_full[k]["ok"] for k in ("driven","undriven")) and all(block_vs_full[k]["elem_atol_rtol_pass"] for k in ("delta_u","v"))
    progress(f"cpu strong-window column-split ok={block_ok} dU={block_vs_full['delta_u']['max_absolute']:.3g}")

    gpu = gpu_probe()
    gpu_cmp = {"ran": False, "ok": False, "probe": gpu}
    if gpu.get("cuda_is_available"):
        import torch
        from q8_m0_column_step_torch_20261006 import accumulate_torch
        device = torch.device("cuda:0")
        torch.cuda.synchronize()
        _ = accumulate_torch(photon, gamma, initial, n_pt, k0, vd0, 2, h, pol, free_A_increment, torch, device, t0=0.0)
        torch.cuda.synchronize()
        g_d0 = accumulate_torch(photon, gamma, initial, n_pt, k0, vd0, NSTEP, h, pol, free_A_increment, torch, device, t0=0.0)
        g_u0 = accumulate_torch(photon, gamma, initial.copy(), n_pt, k0, vu0, NSTEP, h, pol, free_A_increment, torch, device, t0=0.0)
        g_dp = accumulate_torch(photon, gamma, initial, n_pt, kp, vdp, NSTEP, h, pol, free_A_increment, torch, device, t0=PUMP_STEP0 * h)
        g_up = accumulate_torch(photon, gamma, initial.copy(), n_pt, kp, vup, NSTEP, h, pol, free_A_increment, torch, device, t0=PUMP_STEP0 * h)
        g_dU0 = {"state": g_d0["state"], "final_I": g_d0["final_I"], "delta_u": g_d0["delta_u"] - g_u0["delta_u"], "v": g_d0["v"] - g_u0["v"]}
        c_dU0 = {"state": d0["state"], "final_I": d0["final_I"], "delta_u": m0["delta_u"], "v": m0["v"]}
        # compare driven states and subtracted rows
        cmp0 = compare_path(
            {"state": d0["state"], "final_I": d0["final_I"], "delta_u": m0["delta_u"], "v": m0["v"]},
            {"state": g_d0["state"], "final_I": g_d0["final_I"], "delta_u": g_d0["delta_u"] - g_u0["delta_u"], "v": g_d0["v"] - g_u0["v"]},
            "start_driven_minus_undriven",
        )
        cmpp = compare_path(
            {"state": dp["state"], "final_I": dp["final_I"], "delta_u": mp["delta_u"], "v": mp["v"]},
            {"state": g_dp["state"], "final_I": g_dp["final_I"], "delta_u": g_dp["delta_u"] - g_up["delta_u"], "v": g_dp["v"] - g_up["v"]},
            "pump_algebraic_driven_minus_undriven",
        )
        gpu_cmp = {
            "ran": True,
            "ok": bool(cmp0["ok"] and cmpp["ok"] and compare_path(u0,g_u0,"undriven_start")["ok"] and compare_path(up,g_up,"undriven_pump")["ok"]),
            "undriven_start": compare_path(u0,g_u0,"undriven_start"),
            "undriven_pump": compare_path(up,g_up,"undriven_pump"),
            "timings": {tag:{key:rec.get(key) for key in ("wall_s","h2d_setup_s","loop_s","d2h_s","peak_allocated_bytes","peak_reserved_bytes")} for tag,rec in (("driven_start",g_d0),("undriven_start",g_u0),("driven_pump",g_dp),("undriven_pump",g_up))},
            "probe": gpu,
            "start": cmp0,
            "pump_window": cmpp,
            "gpu_driven_start_s": g_d0["wall_s"],
            "gpu_undriven_start_s": g_u0["wall_s"],
            "gpu_driven_pump_s": g_dp["wall_s"],
            "gpu_undriven_pump_s": g_up["wall_s"],
            "note_pump_window": "stationary initial at t=2T vertices; algebraic backend check, not physical trajectory from t=0",
        }
        progress(f"gpu compare ok={gpu_cmp['ok']}")
    else:
        progress(f"gpu skipped import={gpu.get('torch_import')} cuda={gpu.get('cuda_is_available')}")

    per = m0["sum_wall_s"] / max(NSTEP * ncol, 1)
    n_freq_full = int(oms.size)
    ncol_full = 4 * n_freq_full + int(dark_state.b.shape[1])
    nstep_full = int(prot_d.n_cycles * q.NTPC + q.NTPC)
    est_one = nstep_full * ncol_full * (m0["sum_wall_s"] / max(NSTEP * ncol, 1)) / 2.0
    # sum_wall is driven+undriven for nstep*ncol; one path ~ half
    est_one_from_pair = nstep_full * ncol_full * (m0["sum_wall_s"] / (NSTEP * ncol))
    est_pair_h = est_one_from_pair / 3600.0
    est_one_h = est_pair_h / 2.0
    progress(f"budget one-path ~{est_one_h:.2f}h pair ~{est_pair_h:.2f}h ncol_full={ncol_full}")

    start_plus = False
    stop_reason = (
        f"short-test pair extrapolation {est_pair_h:.1f} h; GPU ran={gpu_cmp['ran']} ok={gpu_cmp['ok']}; "
        "not starting 9216-step +N from this process"
    )
    payload = {
        "status": "p1_h2_gamma12_backend_short_not_domain",
        "static_audit_not_rerun": True,
        "static_summary": str(SUMMARY.relative_to(REPO)),
        "columns": {
            "bands": {k: int(v.size) for k, v in bands.items()},
            "n_freq": int(freq.size),
            "n_photon": n_ph,
            "n_dark": n_pt,
            "ncol": ncol,
            "freq_min": float(freq.min()),
            "freq_max": float(freq.max()),
            "h2_in_selection": bool(np.any((freq >= 0.02) & (freq < 0.30))),
            "not_only_lowest_freqs": True,
        },
        "pump": {
            "a_max_start_window": t_a_start,
            "a_max_pump_window": t_a_peak,
            "pump_step0": PUMP_STEP0,
            "pump_window_is_algebraic_not_physical_from_t0": True,
        },
        "cpu": {
            "preprocess_s": t_pre,
            "vertices_s": t_vert,
            "start": m0,
            "pump_window": mp,
            "peak_wset_before": peak0,
            "peak_wset_after": peak1,
            "working_set_after": rss1,
            "dark_assembled_once": True,
        },
        "column_split": {"ok": block_ok, "errors": block_vs_full, "n_split_photon": n1, "strong_window_not_near_zero": True},
        "gpu": gpu_cmp,
        "budget": {
            "nstep_short": NSTEP,
            "ncol_short": ncol,
            "ncol_full": ncol_full,
            "nstep_full": nstep_full,
            "est_one_path_hours_from_start_pair_wall": est_one_h,
            "est_driven_plus_undriven_hours": est_pair_h,
            "partition_not_hardcoded": True,
            "choose_by_peak_memory_concurrency_throughput_wall": True,
        },
        "start_plusN": start_plus,
        "stop_reason": stop_reason,
        "chi_rad_placeholder_status": CHI_STATUS,
        "old_json_not_overwritten": [str(Path(p).relative_to(REPO)) for p in PROTECTED if p is not None],
        "elapsed_s": time.perf_counter() - t_all,
        "production": False,
        "n_m_claimed_as_domain_signal": False,
    }
    # numpy in m0
    def scrub(x):
        if isinstance(x, dict):
            return {k: scrub(v) for k, v in x.items() if k not in ("delta_u", "v") or not isinstance(v, np.ndarray)}
        if isinstance(x, np.ndarray):
            return None
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x

    payload["cpu"]["start"] = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in m0.items() if k in ("driven_wall_s", "undriven_wall_s", "sum_wall_s")}
    payload["cpu"]["start"]["delta_u_maxabs"] = q.maxabs(m0["delta_u"])
    payload["cpu"]["start"]["v_maxabs"] = q.maxabs(m0["v"])
    payload["cpu"]["pump_window"] = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in mp.items() if k in ("driven_wall_s", "undriven_wall_s", "sum_wall_s")}
    payload["cpu"]["pump_window"]["delta_u_maxabs"] = q.maxabs(mp["delta_u"])
    payload["cpu"]["pump_window"]["v_maxabs"] = q.maxabs(mp["v"])
    payload["cpu"]["pump_window"]["algebraic_not_physical_from_t0"] = True

    np.savez_compressed(OUT / "backend_fixture.npz",
        b=initial.b,bdag=initial.bdag,A=initial.A,omega=photon.omega,free_A0=photon.free_A0,
        vd0_u=np.asarray([x[0] for x in vd0]),vd0_d=np.asarray([x[1] for x in vd0]),
        vu0_u=np.asarray([x[0] for x in vu0]),vu0_d=np.asarray([x[1] for x in vu0]),
        vdp_u=np.asarray([x[0] for x in vdp]),vdp_d=np.asarray([x[1] for x in vdp]),
        vup_u=np.asarray([x[0] for x in vup]),vup_d=np.asarray([x[1] for x in vup]),
        k0=k0,kp=kp,pol=pol,h=h,gamma=gamma,n_point=n_pt,t_pump=PUMP_STEP0*h,
        gaps=g12["gap"],u0=g12["u"],D0=g12["D"],oms=oms,wts=wts)
    out = write_json("gamma12_backend_short.json", payload)
    np.savez_compressed(
        OUT / "short_rows.npz",
        freq=freq,
        dU_start=m0["delta_u"],
        v_start=m0["v"],
        dU_pump=mp["delta_u"],
        v_pump=mp["v"],
    )
    print(json.dumps({
        "status": payload["status"],
        "ncol": ncol,
        "block_ok": block_ok,
        "gpu_ran": gpu_cmp["ran"],
        "gpu_ok": gpu_cmp["ok"],
        "est_pair_h": est_pair_h,
        "start_plusN": start_plus,
        "out": str(out),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
