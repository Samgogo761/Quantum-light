"""Forward check of a few bright and dark columns against the stored windows.

Regenerates the 8-cycle electronic trajectory through 11 sampling periods.
The fourth-order accumulator is compared with the adjoint rows at 9, 9.5, 10
and 11 periods. The driven-minus-undriven current is recorded after the pulse.
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
from q8_m0_g1_boundary_sensitivity_20261015 import contract_stacked, stack_forces
from q8_m0_g1_dynamic_map_audit_20261015 import require_cache
from q8_m0_g1_gamma12_column_order_20261015 import CACHE
from q8_m0_g1_gamma12_ir_tail_20261015 import load_adjoint, vertices_progress
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import contract_initial
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
from q8_m0_g1_local_richardson_20261014 import _blend, _midpoint
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_gamma12_time_order_20261015"
OUT = OUTDIR / "window_forward_check_macro1024.json"
MARKS = (9.0, 9.5, 10.0, 11.0)


def adjoint_path(cycles):
    if cycles == 9.0:
        path, _digest = require_cache(1024)
        return path
    label = f"{cycles:.1f}".replace(".", "p")
    return OUTDIR / f"adjoint_rows_macro1024_cycles{label}.npz"


def column_current(prep, state, u, d, h, t):
    free = prep["free"] * (np.exp(-1j * prep["omega"] * (t + h)) - np.exp(-1j * prep["omega"] * t))
    new = midpoint_step(state, u, np.real(d), h, prep["gamma"], free_A_increment=free)
    mid = ColumnState(0.5 * (state.b + new.b), 0.5 * (state.bdag + new.bdag),
                      0.5 * (state.A + new.A))
    return prep["pol"] @ current(mid, u, np.real(d))


def forward(prep, coarse, fine, h):
    state = prep["initial"].copy()
    acc_u = np.zeros((2, state.b.shape[1]), complex)
    acc_v = np.zeros_like(acc_u)
    half = 0.5 * h
    n = len(coarse)
    times = np.empty(3 * n)
    for i in range(n):
        t = i * h
        times[3 * i:3 * i + 3] = (t + 0.25 * h, t + 0.5 * h, t + 0.75 * h)
    kernels, _ = kernel_at_times(prep["gamma"], prep["drive"], times)
    snaps = {}
    envelope = []
    marks = {int(round(cycles * 1024)) for cycles in MARKS}
    for i, (u_c, d_c) in enumerate(coarse):
        incoming = state
        t = i * h
        k1, kc, k2 = kernels[3 * i:3 * i + 3]
        coarse_state, du_c, dv_c = _midpoint(prep, state, u_c, d_c, h, t, kc)
        u1, d1 = fine[2 * i]
        u2, d2 = fine[2 * i + 1]
        mid, du1, dv1 = _midpoint(prep, state, u1, d1, half, t, k1)
        fine_state, du2, dv2 = _midpoint(prep, mid, u2, d2, half, t + half, k2)
        state = _blend(fine_state, coarse_state)
        acc_u += (4.0 * (du1 + du2) - du_c) / 3.0
        acc_v += (4.0 * (dv1 + dv2) - dv_c) / 3.0
        done = i + 1
        if done in marks or (8 * 1024 <= done <= 11 * 1024 and done % 128 == 0):
            src = column_current(prep, incoming, u_c, d_c, h, t)
            envelope.append((float(done / 1024.0), src))
        if done in marks:
            snaps[done] = (acc_u.copy(), acc_v.copy())
        if i % 4000 == 0:
            print("forward", prep["name"], i, flush=True)
    return snaps, envelope


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _saved, cache, t0, _meta = load_rows()
    if abs(float(t0)) > 0:
        raise AssertionError(t0)
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    quiet = p1_a2_protocol(0.0)
    h = float(prot.period) / 1024.0
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    ph, _u0 = base.generated_photon_block(
        cache, gamma, np.array([0.11]), np.array([1.0]), pol, phi_of, eta)
    dark_b = np.array(np.load(CACHE / "b_cols.npy", mmap_mode="r")[-4:].T)
    dark_bd = np.array(np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-4:].T)
    dark_a = np.array(np.load(CACHE / "A_cols.npy", mmap_mode="r")[-4:].T)
    families = {
        "bright_0p11": {
            "initial": ColumnState(ph.initial.b, ph.initial.bdag, ph.initial.A),
            "free": ph.free_A0, "omega": ph.omega,
        },
        "dark4": {
            "initial": ColumnState(dark_b, dark_bd, dark_a),
            "free": np.zeros((2, 4), complex), "omega": np.zeros(4),
        },
    }
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    n = 11 * 1024
    print("vertices", n, flush=True)
    coarse_d = vertices_progress(sample, geo, prot, n, h, "coarse driven")
    fine_d = vertices_progress(sample, geo, prot, 2 * n, 0.5 * h, "fine driven")
    coarse_u = vertices_progress(sample, geo, quiet, n, h, "coarse undriven")
    fine_u = vertices_progress(sample, geo, quiet, 2 * n, 0.5 * h, "fine undriven")
    saved = np.load(CACHE / "driven_u.npy", mmap_mode="r")
    gap = float(np.max(np.abs(coarse_d[9215][0] - saved[9215])))
    print("vertex check", gap, flush=True)
    if gap > 1e-8:
        raise AssertionError("Regenerated trajectory left the saved cache")
    report = {"vertex_step_9215": gap, "families": {}}
    growing = False
    for name, spec in families.items():
        prep_d = {**spec, "gamma": gamma, "pol": pol, "drive": prot, "name": name + " driven"}
        prep_u = {**spec, "gamma": gamma, "pol": pol, "drive": quiet, "name": name + " undriven"}
        snap_d, env_d = forward(prep_d, coarse_d, fine_d, h)
        snap_u, env_u = forward(prep_u, coarse_u, fine_u, h)
        bins = []
        by_cycle = {}
        for (cycle, src_d), (_cycle_u, src_u) in zip(env_d, env_u):
            value = float(np.max(np.abs(src_d - src_u)))
            by_cycle.setdefault(np.floor(cycle * 2.0) / 2.0, []).append(value)
        for lo in (8.0, 8.5, 9.0, 9.5, 10.0, 10.5):
            values = by_cycle.get(lo, [])
            bins.append({"cycle": lo, "max_abs_delta_I": max(values) if values else None,
                         "samples": len(values)})
        late = [item["max_abs_delta_I"] for item in bins if item["cycle"] >= 10.0 and item["max_abs_delta_I"] is not None]
        early = bins[0]["max_abs_delta_I"]
        if late and early is not None and max(late) > 2.0 * early:
            growing = True
        windows = {}
        for cycles in MARKS:
            step = int(round(cycles * 1024))
            rows = load_adjoint(adjoint_path(cycles))
            if name == "dark4":
                adj_d = contract_initial(rows[0], dark_b, dark_bd, dark_a)
                adj_u = contract_initial(rows[2], dark_b, dark_bd, dark_a)
            else:
                stacked_d = stack_forces(rows[1])
                stacked_u = stack_forces(rows[3])
                adj_d = contract_stacked(
                    rows[0], stacked_d, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
                adj_u = contract_stacked(
                    rows[2], stacked_u, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
            fu, fv = snap_d[step][0] - snap_u[step][0], snap_d[step][1] - snap_u[step][1]
            au, av = adj_d[0] - adj_u[0], adj_d[1] - adj_u[1]
            windows[f"{cycles:.1f}"] = {
                "max_abs_forward_minus_adjoint_U": float(np.max(np.abs(fu - au))),
                "max_abs_forward_minus_adjoint_V": float(np.max(np.abs(fv - av))),
            }
            print(name, cycles, windows[f"{cycles:.1f}"], flush=True)
        report["families"][name] = {"windows": windows, "delta_I_envelope": bins}
    report["delta_I_grows_after_the_pulse"] = growing
    report["production_G1_pass"] = False
    report["provenance_sha256"] = provenance([__file__])
    write_new_json(OUT, report)


if __name__ == "__main__":
    main()
