"""Post-pulse spectrum on the two largest Gram-changing columns.

Only the two bright frequencies and the two dark modes that dominate the 9-11
cycle Gram change. Steps 8 through 20 cycles. No full 144-column propagation.

Fixes 2026-10-17 (WorkBuddy handoff review):
- 12-cycle sliding window on exactly 12 cycles of data degenerated to one
  full-window mean; window is now 4 cycles.
- Spectrum now covers negative and positive frequencies (delta I is complex;
  negative frequencies are not conjugate copies).
- Complex per-cycle snapshots and the full delta-I trajectory are saved to an
  .npz next to the JSON (complex .tolist() into plain JSON would have failed).
- per_cycle max_abs_u/v are ΔU/ΔV = driven minus undriven accumulations; the
  constant scattering background U0 is not included in this spectrum script.
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
from q8_m0_g1_gamma12_column_order_20261015 import CACHE
from q8_m0_g1_gamma12_ir_tail_20261015 import vertices_progress
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
from q8_m0_g1_local_richardson_20261014 import _blend, _midpoint
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/tail_spectrum_only.json"
BRIGHT = [(0.03513623463528209, 0.0017560790841957938, 3, 133)]
START = 8 * 1024
STOP = 20 * 1024


def column_current(prep, state, u, d, h, t):
    free = prep["free"] * (np.exp(-1j * prep["omega"] * (t + h)) - np.exp(-1j * prep["omega"] * t))
    new = midpoint_step(state, u, np.real(d), h, prep["gamma"], free_A_increment=free)
    mid = ColumnState(0.5 * (state.b + new.b), 0.5 * (state.bdag + new.bdag),
                      0.5 * (state.A + new.A))
    return prep["pol"] @ current(mid, u, np.real(d))


def propagate(prep, coarse_d, fine_d, coarse_u, fine_u, h, gamma, prot):
    state_d, state_u = prep["initial"].copy(), prep["initial"].copy()
    acc_d = [np.zeros((2, state_d.b.shape[1]), complex), np.zeros((2, state_d.b.shape[1]), complex)]
    acc_u = [np.zeros_like(acc_d[0]), np.zeros_like(acc_d[1])]
    half = 0.5 * h
    n = len(coarse_d)
    times = np.empty(3 * n)
    for i in range(n):
        t = i * h
        times[3 * i:3 * i + 3] = (t + 0.25 * h, t + 0.5 * h, t + 0.75 * h)
    kernels, _ = kernel_at_times(gamma, prot, times)
    snaps = {}
    di = []
    running = 0.0
    per_cycle = []
    for i in range(n):
        t = i * h
        k1, kc, k2 = kernels[3 * i:3 * i + 3]
        incoming_d, incoming_u = state_d, state_u
        for label, incoming, acc, coarse, fine in (
            ("d", incoming_d, acc_d, coarse_d, fine_d),
            ("u", incoming_u, acc_u, coarse_u, fine_u),
        ):
            u_c, d_c = coarse[i]
            coarse_state, du_c, dv_c = _midpoint(prep, incoming, u_c, d_c, h, t, kc)
            u1, d1 = fine[2 * i]
            u2, d2 = fine[2 * i + 1]
            mid, du1, dv1 = _midpoint(prep, incoming, u1, d1, half, t, k1)
            fine_state, du2, dv2 = _midpoint(prep, mid, u2, d2, half, t + half, k2)
            updated = _blend(fine_state, coarse_state)
            acc[0] += (4.0 * (du1 + du2) - du_c) / 3.0
            acc[1] += (4.0 * (dv1 + dv2) - dv_c) / 3.0
            cur = column_current(prep, incoming, u_c, d_c, h, t)
            if label == "d":
                state_d = updated
                cur_d = cur
            else:
                state_u = updated
                cur_u = cur
        done = i + 1
        if done > START:
            di.append(cur_d - cur_u)
        running = max(running, float(np.max(np.abs(cur_d - cur_u))))
        if done % 1024 == 0:
            cycle = done / 1024.0
            u = acc_d[0] - acc_u[0]
            v = acc_d[1] - acc_u[1]
            per_cycle.append({"cycle": cycle,
                              "max_abs_u": float(np.max(np.abs(u))),
                              "max_abs_v": float(np.max(np.abs(v))),
                              "max_abs_delta_I": running})
            running = 0.0
            if cycle in (9.0, 11.0, 14.0, 17.0, 20.0):
                snaps[str(cycle)] = (u.copy(), v.copy())
            if done % 4096 == 0:
                print("cycle", cycle, flush=True)
    return np.stack(di, axis=1), per_cycle, snaps


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _saved, cache, t0, _meta = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    quiet = p1_a2_protocol(0.0)
    eta = eta_of(gamma)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, nrm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, nrm)
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    h = float(prot.period) / 1024.0
    print("vertices", STOP, flush=True)
    coarse_d = vertices_progress(sample, geo, prot, STOP, h, "coarse driven")
    fine_d = vertices_progress(sample, geo, prot, 2 * STOP, 0.5 * h, "fine driven")
    coarse_u = vertices_progress(sample, geo, quiet, STOP, h, "coarse undriven")
    fine_u = vertices_progress(sample, geo, quiet, 2 * STOP, 0.5 * h, "fine undriven")
    saved = np.load(CACHE / "driven_u.npy", mmap_mode="r")
    if np.max(np.abs(coarse_d[9215][0] - saved[9215])) > 1e-8:
        raise AssertionError("Regenerated trajectory left the saved cache")
    out = {"bright": []}
    for omega, weight, port, dark in BRIGHT:
        ph, _u0 = base.generated_photon_block(cache, gamma, np.array([omega]), np.array([weight]), pol, phi_of, eta)
        sl = slice(port, port + 1)
        b = np.array(np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976 + dark].reshape(2304, 1))
        bd = np.array(np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976 + dark].reshape(2304, 1))
        a = np.array(np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976 + dark].reshape(2, 1))
        initial = ColumnState(
            np.concatenate((ph.initial.b[:, sl], b), axis=1),
            np.concatenate((ph.initial.bdag[:, sl], bd), axis=1),
            np.concatenate((ph.initial.A[:, sl], a), axis=1))
        free = np.concatenate((ph.free_A0[:, sl], np.zeros((2, 1), complex)), axis=1)
        om = np.concatenate((ph.omega[sl], np.zeros(1)))
        prep = {"initial": initial, "free": free, "omega": om, "gamma": gamma, "pol": pol}
        print("propagate", omega, dark, flush=True)
        di, per_cycle, snaps = propagate(prep, coarse_d, fine_d, coarse_u, fine_u, h, gamma, prot)
        mean = np.mean(di, axis=1)
        sliding = []
        window = 4 * 1024
        for j in range(0, di.shape[1] - window + 1, 1024):
            sliding.append(float(np.max(np.abs(np.mean(di[:, j:j + window], axis=1)))))
        dw = 2.0 * np.pi / (di.shape[1] * h)
        n_side = int(np.ceil(0.05 / dw))
        freqs = np.arange(-n_side, n_side + 1) * dw
        phase = np.exp(-1j * freqs[:, None] * ((START + 0.5) * h + np.arange(di.shape[1]) * h)[None, :])
        peaks = []
        for column in range(di.shape[2]):
            for p in range(2):
                mag = np.abs(phase @ di[p, :, column])
                order = np.argsort(mag)[-5:][::-1]
                peaks.append({"column": column, "pol": p,
                              "peaks": [(float(freqs[i]), float(mag[i])) for i in order],
                              "dc_sum": float(abs(np.sum(di[p, :, column])))})
        out["bright"].append({
            "omega": omega, "weight": weight, "port": port, "dark_index": dark,
            "per_cycle": per_cycle,
            "mean_abs_di": float(np.max(np.abs(mean))),
            "sliding_4cycle_mean": sliding,
            "spectra": peaks,
            "snapshot_maxabs": {k: {"max_abs_u": float(np.max(np.abs(v[0]))),
                                    "max_abs_v": float(np.max(np.abs(v[1])))}
                                for k, v in snaps.items()},
        })
        np.savez_compressed(
            ROOT / "results/q_line/g1_gamma12_time_order_20261015"
            / f"tail_spectrum_only_dark{dark}.npz",
            di=di,
            snap_cycles=np.array(sorted(float(k) for k in snaps)),
            **{f"snap_u_{k}": v[0] for k, v in snaps.items()},
            **{f"snap_v_{k}": v[1] for k, v in snaps.items()})
    write_new_json(OUT, {
        "note": "di_steps axis order is (pol, time, column); complex trajectories in tail_spectrum_only_dark{dark}.npz",
        "bright": out["bright"],
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
