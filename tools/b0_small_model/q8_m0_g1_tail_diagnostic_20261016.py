"""Tail diagnostic without a new full adjoint.

Reports the long-time size of the frozen detector kernel, checks that the
classical background has no phenomenological damping, and propagates the input
columns whose 9-to-11-cycle output changes are largest.
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
from q8_m0_k1_packet import phi_unnormalized
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_gamma12_time_order_20261015"
OUT = OUTDIR / "tail_diagnostic_macro1024.json"
CYCLES = 20
TOP = 4


def adjoint_path(cycles):
    if cycles == 9.0:
        path, _digest = require_cache(1024)
        return path
    label = f"{cycles:.1f}".replace(".", "p")
    return OUTDIR / f"adjoint_rows_macro1024_cycles{label}.npz"


def kernel_facts(gamma, prot, eta, normalizer):
    tp = float(prot.t_pulse)
    center = 2.0 * float(prot.omega0)
    probe = np.array([0.0, 1.0e-6, 1.0e-4, 1.0e-3, 0.01, 0.05, center])
    phi = phi_unnormalized(probe, center, tp) / normalizer
    phi_center = complex(phi[-1])
    samples = []
    for cycles in (9, 20, 40, 80):
        t = cycles * float(prot.period)
        # Fine low-frequency grid plus the packet grid. The production dw
        # cannot resolve a long-time endpoint oscillation.
        low = np.geomspace(1.0e-8, 0.02, 4000)
        packet = np.linspace(0.02, 0.6, 8000)
        om = np.unique(np.concatenate((low, packet)))
        weight = phi_unnormalized(om, center, tp) / normalizer / np.sqrt(om)
        value = 1j * eta * np.trapezoid(weight * np.exp(1j * om * t), om)
        endpoint = 1j * eta * complex(phi[0]) * np.sqrt(np.pi / t) * np.exp(0.25j * np.pi)
        samples.append({
            "cycles": cycles,
            "abs_K": abs(complex(value)),
            "sqrt_t_abs_K": abs(complex(value)) * np.sqrt(t),
            "abs_endpoint_formula": abs(complex(endpoint)),
        })
    return {
        "phi_defined_by_8_cycle_sin2_only": True,
        "phi_at_zero": abs(complex(phi[0])),
        "phi_at_h2": abs(phi_center),
        "phi_zero_over_phi_h2": abs(complex(phi[0]) / phi_center),
        "phi_over_sqrt_omega": [
            {"omega": float(w), "abs_phi": abs(complex(p)), "abs_phi_over_sqrt_omega": abs(complex(p)) / np.sqrt(w) if w else None}
            for w, p in zip(probe, phi) if w > 0
        ],
        "samples": samples,
        "background_has_no_T1_T2": True,
        "background_step": "advance_half_frame applies exp(-i H(k+kappa(t)) h/2); kappa is zero after t_pulse",
    }


def keep_top(found, score, payload):
    found.append((float(score), payload))
    found.sort(key=lambda item: item[0], reverse=True)
    del found[TOP:]


def rank(cache, gamma, pol, phi_of, eta):
    rows9 = load_adjoint(adjoint_path(9.0))
    rows11 = load_adjoint(adjoint_path(11.0))
    b = np.array(np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976:].T)
    bd = np.array(np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976:].T)
    a = np.array(np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976:].T)
    u9, v9 = contract_initial(rows9[0], b, bd, a)
    uu9, uv9 = contract_initial(rows9[2], b, bd, a)
    u11, v11 = contract_initial(rows11[0], b, bd, a)
    uu11, uv11 = contract_initial(rows11[2], b, bd, a)
    du, dv = (u11 - uu11) - (u9 - uu9), (v11 - uv11) - (v9 - uv9)
    dark_score = np.sum(np.abs(du) ** 2, axis=0) + np.sum(np.abs(dv) ** 2, axis=0)
    dark_index = np.argsort(dark_score)[-TOP:][::-1]
    stacked9 = (stack_forces(rows9[1]), stack_forces(rows9[3]))
    stacked11 = (stack_forces(rows11[1]), stack_forces(rows11[3]))
    pieces, _edge = frequencies(cache, gamma)
    found = []
    for name, om, wt in pieces:
        print("rank", name, len(om), flush=True)
        for start in range(0, len(om), 200):
            stop = min(start + 200, len(om))
            block_om, block_wt = om[start:stop], wt[start:stop]
            ph, u0 = base.generated_photon_block(
                cache, gamma, block_om, block_wt, pol, phi_of, eta)
            def raw(rows, stacked):
                return contract_stacked(
                    rows, stacked, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
            d9, u9p = raw(rows9[0], stacked9[0]), raw(rows9[2], stacked9[1])
            d11, u11p = raw(rows11[0], stacked11[0]), raw(rows11[2], stacked11[1])
            delta_u = (u0 + d11[0] - u11p[0]) - (u0 + d9[0] - u9p[0])
            delta_v = (d11[1] - u11p[1]) - (d9[1] - u9p[1])
            score = np.sum(np.abs(delta_u) ** 2, axis=0) + np.sum(np.abs(delta_v) ** 2, axis=0)
            for local in np.argsort(score)[-TOP:]:
                keep_top(found, score[local], {
                    "omega": float(block_om[local // 4]),
                    "weight": float(block_wt[local // 4]),
                    "port": int(local % 4),
                    "score": float(score[local]),
                })
    return {
        "dark_index_within_1976": [int(i) for i in dark_index],
        "dark_score": [float(dark_score[i]) for i in dark_index],
        "bright": [item[1] for item in found],
    }


def column_current(prep, state, u, d, h, t):
    free = prep["free"] * (np.exp(-1j * prep["omega"] * (t + h)) - np.exp(-1j * prep["omega"] * t))
    new = midpoint_step(state, u, np.real(d), h, prep["gamma"], free_A_increment=free)
    mid = ColumnState(0.5 * (state.b + new.b), 0.5 * (state.bdag + new.bdag),
                      0.5 * (state.A + new.A))
    return prep["pol"] @ current(mid, u, np.real(d))


def forward_pair(prep, coarse_d, fine_d, coarse_u, fine_u, h, gamma, prot):
    state_d = prep["initial"].copy()
    state_u = prep["initial"].copy()
    acc_d = [np.zeros((2, state_d.b.shape[1]), complex), np.zeros((2, state_d.b.shape[1]), complex)]
    acc_u = [np.zeros_like(acc_d[0]), np.zeros_like(acc_d[1])]
    half = 0.5 * h
    n = len(coarse_d)
    times = np.empty(3 * n)
    for i in range(n):
        t = i * h
        times[3 * i:3 * i + 3] = (t + 0.25 * h, t + 0.5 * h, t + 0.75 * h)
    kernels, _ = kernel_at_times(gamma, prot, times)
    per_cycle = []
    running = 0.0
    snaps = {}
    for i in range(n):
        t = i * h
        k1, kc, k2 = kernels[3 * i:3 * i + 3]
        incoming_d = state_d
        incoming_u = state_u
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
            current_now = column_current(prep, incoming, u_c, d_c, h, t)
            if label == "d":
                state_d = updated
                cur_d = current_now
            else:
                state_u = updated
                cur_u = current_now
        running = max(running, float(np.max(np.abs(cur_d - cur_u))))
        done = i + 1
        if done % 1024 == 0:
            cycle = done / 1024.0
            delta = acc_d[0] - acc_u[0]
            per_cycle.append({
                "cycle": cycle,
                "max_abs_delta_I": running,
                "max_abs_cumulative_U": float(np.max(np.abs(delta))),
                "max_abs_cumulative_V": float(np.max(np.abs(acc_d[1] - acc_u[1]))),
            })
            running = 0.0
            print("cycle", cycle, per_cycle[-1]["max_abs_cumulative_U"], flush=True)
        if done in {int(round(c * 1024)) for c in (9.0, 9.5, 10.0, 11.0)}:
            snaps[done] = (acc_d[0] - acc_u[0], acc_d[1] - acc_u[1])
    return per_cycle, snaps


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _saved, cache, t0, _meta = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    quiet = p1_a2_protocol(0.0)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, normalizer = base.q.normalize_phi(
        grid, base.q.phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, normalizer)
    eta = base.q.eta_of(gamma)
    facts = kernel_facts(gamma, prot, eta, normalizer)
    print("phi(0)/phi(H2)", facts["phi_zero_over_phi_h2"], flush=True)
    chosen = rank(cache, gamma, pol, phi_of, eta)
    print("chosen", chosen, flush=True)
    bright = chosen["bright"]
    oms = np.array(sorted({item["omega"] for item in bright}))
    wts = np.array([next(item["weight"] for item in bright if item["omega"] == om) for om in oms])
    ph, _u0 = base.generated_photon_block(cache, gamma, oms, wts, pol, phi_of, eta)
    pick = []
    for item in bright:
        freq = int(np.where(np.isclose(oms, item["omega"]))[0][0])
        pick.append(4 * freq + item["port"])
    dark_ids = np.array(chosen["dark_index_within_1976"], int)
    b = np.array(np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976:].T)[:, dark_ids]
    bd = np.array(np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976:].T)[:, dark_ids]
    a = np.array(np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976:].T)[:, dark_ids]
    initial = ColumnState(
        np.concatenate((ph.initial.b[:, pick], b), axis=1),
        np.concatenate((ph.initial.bdag[:, pick], bd), axis=1),
        np.concatenate((ph.initial.A[:, pick], a), axis=1),
    )
    free = np.concatenate((ph.free_A0[:, pick], np.zeros((2, len(dark_ids)), complex)), axis=1)
    omega = np.concatenate((ph.omega[pick], np.zeros(len(dark_ids))))
    prep = {"initial": initial, "free": free, "omega": omega, "gamma": gamma, "pol": pol}
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    h = float(prot.period) / 1024.0
    n = CYCLES * 1024
    print("vertices", n, flush=True)
    coarse_d = vertices_progress(sample, geo, prot, n, h, "coarse driven")
    fine_d = vertices_progress(sample, geo, prot, 2 * n, 0.5 * h, "fine driven")
    coarse_u = vertices_progress(sample, geo, quiet, n, h, "coarse undriven")
    fine_u = vertices_progress(sample, geo, quiet, 2 * n, 0.5 * h, "fine undriven")
    per_cycle, snaps = forward_pair(prep, coarse_d, fine_d, coarse_u, fine_u, h, gamma, prot)
    checks = {}
    for cycles in (9.0, 11.0):
        rows = load_adjoint(adjoint_path(cycles))
        bright_d = contract_stacked(
            rows[0], stack_forces(rows[1]), ph.initial.b[:, pick], ph.initial.bdag[:, pick],
            ph.initial.A[:, pick], ph.free_A0[:, pick], ph.omega[pick])
        bright_u = contract_stacked(
            rows[2], stack_forces(rows[3]), ph.initial.b[:, pick], ph.initial.bdag[:, pick],
            ph.initial.A[:, pick], ph.free_A0[:, pick], ph.omega[pick])
        dark_d = contract_initial(rows[0], b, bd, a)
        dark_u = contract_initial(rows[2], b, bd, a)
        adj = (np.concatenate(((bright_d[0] - bright_u[0]), (dark_d[0] - dark_u[0])), axis=1),
               np.concatenate(((bright_d[1] - bright_u[1]), (dark_d[1] - dark_u[1])), axis=1))
        got = snaps[int(round(cycles * 1024))]
        checks[f"{cycles:.1f}"] = {
            "max_abs_U": float(np.max(np.abs(got[0] - adj[0]))),
            "max_abs_V": float(np.max(np.abs(got[1] - adj[1]))),
        }
        print("check", cycles, checks[f"{cycles:.1f}"], flush=True)
    write_new_json(OUT, {
        "kernel": facts,
        "selected": chosen,
        "cumulative_by_cycle": per_cycle,
        "forward_minus_adjoint": checks,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
