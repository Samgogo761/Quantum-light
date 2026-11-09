"""Complex tail check and Gram-weighted column selection.

Reuses the stored window adjoints to select columns by their change in the
R_N Gram between 9 and 11 cycles. Propagates those columns to 20 cycles,
reports complex U,V per cycle, the zero-frequency part of delta I, and the
low-frequency spectrum. Replaces no production assembly.

Fixes 2026-10-17 (WorkBuddy handoff review):
- per-cycle Gram now uses the FULL U = U0 + ΔU (U0 from generated_photon_block
  was previously discarded), so U0 ΔU† + h.c. is no longer missing.
- The 12-cycle sliding mean on exactly 12 cycles of data degenerated to one
  full-window value; the window is now 4 cycles so a trend is visible.
- The spectrum now spans both negative and positive frequencies (delta I is
  complex; negative frequencies are not conjugate copies).
- Complex per-cycle U,V trajectories are saved to an .npz next to the JSON
  (complex .tolist() into plain JSON would have failed at write time).
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import ColumnState, current, midpoint_step
from q8_m0_g1_boundary_sensitivity_20261015 import contract_stacked, frequencies, stack_forces
from q8_m0_g1_dynamic_map_audit_20261015 import require_cache
from q8_m0_g1_gamma12_column_order_20261015 import CACHE
from q8_m0_g1_gamma12_ir_tail_20261015 import load_adjoint, vertices_progress
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import contract_initial
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
from q8_m0_g1_local_richardson_20261014 import _blend, _midpoint
from q8_m0_k1_packet import phi_unnormalized, omega_grid, normalize_phi
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_gamma12_time_order_20261015"
OUT = OUTDIR / "tail_complexity_macro1024.json"
TOP = 4


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def adjoint_path(cycles):
    if cycles == 9.0:
        path, _digest = require_cache(1024)
        return path
    label = f"{cycles:.1f}".replace(".", "p")
    return OUTDIR / f"adjoint_rows_macro1024_cycles{label}.npz"


def analytic_phi_zero(prot):
    """sin^2 envelope with n_cycles carrier cycles. Zero is exact when the
    pulse contains an integer number of carrier cycles."""
    return {
        "phi0_closed_form_is_zero": int(prot.n_cycles) > 0,
        "n_cycles": int(prot.n_cycles),
        "reason": "W(t)=sin^2(pi t/Tp) decomposes into exp(0), exp(±2 pi i t/Tp); each integral over Tp=n*period vanishes at omega=0 for an integer-cycle carrier",
    }


def gram_scores(u, v):
    normal = np.abs(u[0][None, :].conj() * u[:, :, None]).max(axis=(0, 1))
    return normal


def select_gram(cache, gamma, pol, phi_of, eta):
    rows9 = load_adjoint(adjoint_path(9.0))
    rows11 = load_adjoint(adjoint_path(11.0))
    b = np.array(np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976:].T)
    bd = np.array(np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976:].T)
    a = np.array(np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976:].T)
    du9, dv9 = contract_initial(rows9[0], b, bd, a)
    uu9, uv9 = contract_initial(rows9[2], b, bd, a)
    du11, dv11 = contract_initial(rows11[0], b, bd, a)
    uu11, uv11 = contract_initial(rows11[2], b, bd, a)
    u9, v9 = du9 - uu9, dv9 - uv9
    u11, v11 = du11 - uu11, dv11 - uv11
    score = np.zeros(1976)
    for j in range(1976):
        before = u9[:, j:j + 1] @ u9[:, j:j + 1].conj().T - v9[:, j:j + 1] @ v9[:, j:j + 1].conj().T
        after = u11[:, j:j + 1] @ u11[:, j:j + 1].conj().T - v11[:, j:j + 1] @ v11[:, j:j + 1].conj().T
        score[j] = np.max(np.abs(after - before))
    dark_index = np.argsort(score)[-TOP:][::-1]
    print("dark gram", dark_index, score[dark_index], flush=True)
    pieces, _edge = frequencies(cache, gamma)
    found = []
    sd9, su9 = stack_forces(rows9[1]), stack_forces(rows9[3])
    sd11, su11 = stack_forces(rows11[1]), stack_forces(rows11[3])
    for name, om, wt in pieces:
        print("gram", name, len(om), flush=True)
        for start in range(0, len(om), 200):
            stop = min(start + 200, len(om))
            ph, u0 = base.generated_photon_block(
                cache, gamma, om[start:stop], wt[start:stop], pol, phi_of, eta)
            def map_at(rows, sd, su):
                d = contract_stacked(rows[0], sd, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
                u = contract_stacked(rows[2], su, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
                return u0 + d[0] - u[0], d[1] - u[1]
            u9b, v9b = map_at(rows9, sd9, su9)
            u11b, v11b = map_at(rows11, sd11, su11)
            for j in range(u9b.shape[1]):
                before = np.outer(u9b[:, j], u9b[:, j].conj()) - np.outer(v9b[:, j], v9b[:, j].conj())
                after = np.outer(u11b[:, j], u11b[:, j].conj()) - np.outer(v11b[:, j], v11b[:, j].conj())
                value = float(np.max(np.abs(after - before)))
                found.append((value, {"omega": float(om[start + j // 4]), "weight": float(wt[start + j // 4]),
                                      "port": int(j % 4)}))
    found.sort(key=lambda item: item[0], reverse=True)
    bright = [item[1] for item in found[:TOP]]
    return {"dark_index": [int(i) for i in dark_index],
            "dark_gram_change": [float(score[i]) for i in dark_index],
            "bright": bright}


def column_current(prep, state, u, d, h, t):
    free = prep["free"] * (np.exp(-1j * prep["omega"] * (t + h)) - np.exp(-1j * prep["omega"] * t))
    new = midpoint_step(state, u, np.real(d), h, prep["gamma"], free_A_increment=free)
    mid = ColumnState(0.5 * (state.b + new.b), 0.5 * (state.bdag + new.bdag),
                      0.5 * (state.A + new.A))
    return prep["pol"] @ current(mid, u, np.real(d))


def propagate(prep, coarse_d, fine_d, coarse_u, fine_u, h, gamma, prot, eta):
    state_d, state_u = prep["initial"].copy(), prep["initial"].copy()
    acc = [np.zeros((2, prep["initial"].b.shape[1]), complex) for _ in range(4)]
    half = 0.5 * h
    n = len(coarse_d)
    times = np.empty(3 * n)
    for i in range(n):
        t = i * h
        times[3 * i:3 * i + 3] = (t + 0.25 * h, t + 0.5 * h, t + 0.75 * h)
    kernels, _ = kernel_at_times(gamma, prot, times)
    start = 8 * 1024
    di_steps = np.zeros((2, n - start, prep["initial"].b.shape[1]), complex)
    kernel_tail = np.empty(3 * (n - start), complex)
    per_cycle = []
    snaps = {}
    traj_u, traj_v = [], []
    for i in range(n):
        t = i * h
        k1, kc, k2 = kernels[3 * i:3 * i + 3]
        incoming_d, incoming_u = state_d, state_u
        for label, incoming, pair, coarse, fine in (
            ("d", incoming_d, acc[:2], coarse_d, fine_d),
            ("u", incoming_u, acc[2:], coarse_u, fine_u),
        ):
            u_c, d_c = coarse[i]
            coarse_state, du_c, dv_c = _midpoint(prep, incoming, u_c, d_c, h, t, kc)
            u1, d1 = fine[2 * i]
            u2, d2 = fine[2 * i + 1]
            mid, du1, dv1 = _midpoint(prep, incoming, u1, d1, half, t, k1)
            fine_state, du2, dv2 = _midpoint(prep, mid, u2, d2, half, t + half, k2)
            updated = _blend(fine_state, coarse_state)
            pair[0] += (4.0 * (du1 + du2) - du_c) / 3.0
            pair[1] += (4.0 * (dv1 + dv2) - dv_c) / 3.0
            cur = column_current(prep, incoming, u_c, d_c, h, t)
            if label == "d":
                state_d = updated
                cur_d = cur
            else:
                state_u = updated
                cur_u = cur
        done = i + 1
        if done > start:
            di_steps[:, done - start - 1] = cur_d - cur_u
        if done % 1024 == 0:
            cycle = done / 1024.0
            u = acc[0] - acc[2]
            v = acc[1] - acc[3]
            per_cycle.append({
                "cycle": cycle,
                "max_abs_u": float(np.max(np.abs(u))),
                "max_abs_v": float(np.max(np.abs(v))),
                "mean_abs_phase_u": float(np.mean(np.abs(np.angle(u[np.abs(u) > 1e-8])))) if np.any(np.abs(u) > 1e-8) else 0.0,
                "u_index": len(per_cycle),
            })
            if cycle in (9.0, 11.0, 14.0, 17.0, 20.0):
                snaps[str(cycle)] = (u.copy(), v.copy())
            traj_u.append(u.copy())
            traj_v.append(v.copy())
            print("cycle", cycle, per_cycle[-1]["max_abs_u"], flush=True)
    kernel_tail[:] = kernels[3 * start:]
    return per_cycle, snaps, di_steps, kernel_tail, traj_u, traj_v


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _saved, cache, t0, _meta = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    quiet = p1_a2_protocol(0.0)
    grid = omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, nrm = normalize_phi(grid, phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, nrm)
    eta = eta_of(gamma)
    selected = select_gram(cache, gamma, pol, phi_of, eta)
    print("selected", selected, flush=True)
    bright = selected["bright"]
    oms = np.array(sorted({item["omega"] for item in bright}))
    wts = np.array([next(item["weight"] for item in bright if item["omega"] == om) for om in oms])
    ph, u0_block = base.generated_photon_block(cache, gamma, oms, wts, pol, phi_of, eta)
    pick = []
    for item in bright:
        freq = int(np.where(np.isclose(oms, item["omega"]))[0][0])
        pick.append(4 * freq + item["port"])
    dark_ids = np.array(selected["dark_index"], int)
    b = np.array(np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976:].T)[:, dark_ids]
    bd = np.array(np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976:].T)[:, dark_ids]
    a = np.array(np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976:].T)[:, dark_ids]
    initial = ColumnState(
        np.concatenate((ph.initial.b[:, pick], b), axis=1),
        np.concatenate((ph.initial.bdag[:, pick], bd), axis=1),
        np.concatenate((ph.initial.A[:, pick], a), axis=1))
    free = np.concatenate((ph.free_A0[:, pick], np.zeros((2, len(dark_ids)), complex)), axis=1)
    omega = np.concatenate((ph.omega[pick], np.zeros(len(dark_ids))))
    prep = {"initial": initial, "free": free, "omega": omega, "gamma": gamma, "pol": pol}
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    h = float(prot.period) / 1024.0
    print("vertices", 20 * 1024, flush=True)
    coarse_d = vertices_progress(sample, geo, prot, 20 * 1024, h, "coarse driven")
    fine_d = vertices_progress(sample, geo, prot, 40 * 1024, 0.5 * h, "fine driven")
    coarse_u = vertices_progress(sample, geo, quiet, 20 * 1024, h, "coarse undriven")
    fine_u = vertices_progress(sample, geo, quiet, 40 * 1024, 0.5 * h, "fine undriven")
    per_cycle, snaps, di_steps, kernel_tail, traj_u, traj_v = propagate(
        prep, coarse_d, fine_d, coarse_u, fine_u, h, gamma, prot, eta)
    # U0 aligned to the selected column order (bright picks, then dark zeros).
    u0a = np.concatenate(
        (u0_block[:, pick], np.zeros((2, len(dark_ids)), complex)), axis=1)
    # 8 cycles onward. Step h. di is the driven-minus-undriven sheet current.
    mean_di = np.mean(di_steps, axis=1)
    # Sliding mean with a 4-cycle window (12 cycles of data): a 12-cycle
    # window would collapse to a single full-window value and hide any trend.
    window = 4 * 1024
    sliding = [float(np.max(np.abs(np.mean(di_steps[:, j:j + window], axis=1))))
               for j in range(0, di_steps.shape[1] - window + 1, 1024)]
    n_di = di_steps.shape[1]
    dw = 2.0 * np.pi / (n_di * h)
    # Complex delta I: negative frequencies carry independent information and
    # are NOT conjugate copies; report both sides of the spectrum.
    n_side = int(np.ceil(0.05 / dw))
    freqs = np.arange(-n_side, n_side + 1) * dw
    times_tail = (8 * 1024 + 0.5) * h + np.arange(n_di) * h
    phase = np.exp(-1j * freqs[:, None] * times_tail[None, :])
    top = []
    for j in range(di_steps.shape[2]):
        for pol_index in range(2):
            mag = np.abs(phase @ di_steps[pol_index, :, j])
            order = np.argsort(mag)[-5:][::-1]
            top.append({"pol": pol_index, "column": j,
                        "peaks": [(float(freqs[i]), float(mag[i])) for i in order],
                        "dc": float(abs(np.sum(di_steps[pol_index, :, j])))})
    gram_by_cycle = []
    for k, item in enumerate(per_cycle):
        u_full = u0a + traj_u[k]  # full U = U0 + ΔU; keep the U0ΔU† cross terms
        v_full = traj_v[k]
        gram_by_cycle.append({
            "cycle": item["cycle"],
            "normal_full": pack(u_full @ u_full.conj().T - v_full @ v_full.conj().T - np.eye(2)),
            "anomalous_full": pack(u_full @ v_full.T - v_full @ u_full.T),
            "cross_u0_du": pack(u0a @ traj_u[k].conj().T + traj_u[k] @ u0a.conj().T),
            "normal_delta_only": pack(traj_u[k] @ traj_u[k].conj().T - v_full @ v_full.conj().T),
        })
    npz_path = OUTDIR / "tail_complexity_macro1024.npz"
    np.savez_compressed(
        npz_path,
        traj_u=np.stack(traj_u), traj_v=np.stack(traj_v),
        u0_aligned=u0a, di_steps=di_steps, kernel_tail=kernel_tail,
        snap_cycles=np.array(sorted(float(k) for k in snaps)),
        **{f"snap_u_{k}": v[0] for k, v in snaps.items()},
        **{f"snap_v_{k}": v[1] for k, v in snaps.items()})
    write_new_json(OUT, {
        "phi_zero": analytic_phi_zero(prot),
        "selected_by_gram_change": selected,
        "per_cycle_complex_uv": gram_by_cycle,
        "snap_cycles": {key: {"u": pack(value[0]), "v": pack(value[1])} for key, value in snaps.items()},
        "delta_i_after_pulse": {
            "n_steps": int(di_steps.shape[1]),
            "mean_abs_by_column": float(np.max(np.abs(mean_di))),
            "sliding_4cycle_mean_max": sliding,
            "spectra_absfreq_below_0p05_both_sides": top,
        },
        "complex_trajectory_npz": npz_path.name,
        "gram_note": "normal_full/anomalous_full use U = U0 + ΔU (cross terms kept); normal_delta_only shown for comparison",
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
