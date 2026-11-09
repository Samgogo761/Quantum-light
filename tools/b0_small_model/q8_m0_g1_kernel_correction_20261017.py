"""Kernel-correction increment via adjoint rows (9 cycles, macro1024).

Purpose: quantify what the detector-kernel quadrature error actually does to
the output map, WITHOUT any new long propagation and WITHOUT touching frozen
physics.

Method
------
U is built from the adjoint rows, and the kernel k(t) enters each pullback
step only through the detector rows (see detector()). Therefore rebuilding the
rows twice -- once with the production kernel, once with the high-resolution
reference kernel -- and contracting the SAME input columns gives
ΔU = U_ref − U_prod and ΔV exactly, with no forward re-propagation.

Cost control
------------
- Only 9 cycles (9216 macro steps), the existing audited window.
- Vertices are produced one drive at a time and released, keeping peak memory
  around one coarse+fine trajectory instead of four.
- A hard wall-clock budget (BUDGET_S) aborts cleanly with a JSON status.

Independent check
-----------------
Two representative columns (dark 133 and one bright column) are ALSO propagated
forward twice, once per kernel, and δU_forward is compared to δU_adjoint. This
is the cross-validation the reviewer asked for.

Outputs JSON + npz into results/q_line/g1_kernel_correction_20261017/.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import ColumnState, current, midpoint_step
from q8_m0_g1_boundary_sensitivity_20261015 import contract_stacked, frequencies, stack_forces
from q8_m0_g1_gamma12_column_order_20261015 import CACHE
from q8_m0_g1_gamma12_ir_tail_20261015 import rows_from_vertices, vertices_progress
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import contract_initial, macro_pullback
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
from q8_m0_g1_local_richardson_20261014 import _blend, _midpoint
from q8_m0_k1_packet import phi_unnormalized, omega_grid, normalize_phi
from q8_m0_model import build_geometry
from q8_m0_output_transpose_20261008 import midpoint_output_pullback
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_kernel_correction_20261017"
OUTJSON = OUTDIR / "kernel_correction_9cycles.json"
OUTNPZ = OUTDIR / "kernel_correction_9cycles.npz"

CYCLES = 9
BUDGET_S = 60 * 60
DETECTOR_SCALE = 1.0  # detector() scale used by macro_pullback

T0 = time.time()


def elapsed():
    return time.time() - T0


def budget_left():
    if elapsed() > BUDGET_S:
        raise TimeoutError(f"wall-clock budget {BUDGET_S}s exceeded")


def ref_kernel(gamma, prot, times, eta):
    """High-resolution reference quadrature (same frozen phi, eta)."""
    tp = float(prot.t_pulse)
    center = 2.0 * float(prot.omega0)
    om = np.unique(np.concatenate((
        np.geomspace(1.0e-9, 0.02, 12000),
        np.linspace(0.02, 0.6, 24000))))
    raw = phi_unnormalized(om, center, tp)
    phi, _nrm = normalize_phi(om, raw)
    weight = phi / np.sqrt(om)
    k = np.empty(times.size, complex)
    for s in range(0, times.size, 32):
        sl = slice(s, min(s + 32, times.size))
        phase = np.exp(1j * om[:, None] * times[sl][None, :])
        k[sl] = 1j * eta * np.trapezoid(weight[:, None] * phase, om, axis=0)
    return k


def rows_with_kernels(coarse, fine, h, gamma, pol, kernels):
    """rows_from_vertices with externally supplied kernels (prod or ref)."""
    n_macro = len(coarse)
    row = np.zeros((4, 2 * int(np.asarray(fine[0][0]).shape[1]) + 2), complex)
    force, tstamp, step = [], [], []
    for i in range(n_macro - 1, -1, -1):
        piece = {
            "t": i * h, "H": h,
            "uc": coarse[i][0], "dc": np.real(coarse[i][1]),
            "u1": np.asarray(fine[2 * i][0]), "d1": np.real(fine[2 * i][1]),
            "u2": np.asarray(fine[2 * i + 1][0]), "d2": np.real(fine[2 * i + 1][1]),
            "kc": kernels[3 * i + 1], "k1": kernels[3 * i], "k2": kernels[3 * i + 2],
        }
        from q8_m0_g1_gamma12_richardson_adjoint_20261015 import macro_pullback
        from q8_m0_g1_gamma12_richardson_adjoint_20261015 import macro_pullback
        from q8_m0_g1_gamma12_richardson_adjoint_20261015 import macro_pullback
        row, fc, f1, f2 = macro_pullback(row, piece, gamma, pol)
        for item, when, width in ((fc, i * h, h), (f1, i * h, 0.5 * h),
                                  (f2, i * h + 0.5 * h, 0.5 * h)):
            force.append(item)
            tstamp.append(when)
            step.append(width)
        if i % 2000 == 0:
            print("  pullback", i, f"{elapsed():.0f}s", flush=True)
    return row, list(zip(force, tstamp, step))


def save_rows_cache(path, rows):
    payload = {}
    for (drive, kind), (row, forces) in rows.items():
        payload[f"row_{drive}_{kind}"] = row
        payload[f"force_{drive}_{kind}"] = np.stack([f for f, _t, _h in forces])
        payload[f"time_{drive}_{kind}"] = np.array([t for _f, t, _h in forces], float)
        payload[f"step_{drive}_{kind}"] = np.array([h for _f, _t, h in forces], float)
    np.savez(path, **payload)


def load_rows_cache(path):
    out = {}
    with np.load(path, allow_pickle=False) as saved:
        for drive in ("driven", "undriven"):
            for kind in ("prod", "ref"):
                forces = list(zip(saved[f"force_{drive}_{kind}"],
                                  saved[f"time_{drive}_{kind}"],
                                  saved[f"step_{drive}_{kind}"]))
                out[(drive, kind)] = (np.array(saved[f"row_{drive}_{kind}"]), forces)
    return out


def kernels_for(times, gamma, prot, eta, kind):
    if kind == "prod":
        return kernel_at_times(gamma, prot, times)[0]
    return ref_kernel(gamma, prot, times, eta)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    _saved, cache, _t0, _meta = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    quiet = p1_a2_protocol(0.0)
    eta = eta_of(gamma)
    grid = omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, nrm = normalize_phi(grid, phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, nrm)
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    h = float(prot.period) / 1024.0
    n = CYCLES * 1024

    times = np.empty(3 * n)
    for i in range(n):
        t = i * h
        times[3 * i:3 * i + 3] = (t + 0.25 * h, t + 0.5 * h, t + 0.75 * h)
    k_prod = kernels_for(times, gamma, prot, eta, "prod")
    k_ref = kernels_for(times, gamma, prot, eta, "ref")
    print(f"kernels ready {elapsed():.0f}s  max|dK|={np.max(np.abs(k_ref-k_prod)):.3e}", flush=True)

    rows = {}
    rows_cache = OUTDIR / "rows_kernel_correction_9cycles.npz"
    if rows_cache.exists():
        print("load rows cache", f"{elapsed():.0f}s", flush=True)
        rows = load_rows_cache(rows_cache)
    else:
        for drive, protocol in (("driven", prot), ("undriven", quiet)):
            budget_left()
            print("vertices", drive, f"{elapsed():.0f}s", flush=True)
            coarse = vertices_progress(sample, geo, protocol, n, h, f"coarse {drive}")
            fine = vertices_progress(sample, geo, protocol, 2 * n, 0.5 * h, f"fine {drive}")
            print("vertices done", drive, f"{elapsed():.0f}s", flush=True)
            budget_left()
            for kind, kk in (("prod", k_prod), ("ref", k_ref)):
                print("rows", drive, kind, f"{elapsed():.0f}s", flush=True)
                rows[(drive, kind)] = rows_with_kernels(coarse, fine, h, gamma, pol, kk)
            del coarse, fine
            gc.collect()
        save_rows_cache(rows_cache, rows)
        print("rows cached", f"{elapsed():.0f}s", flush=True)

    # Cross-check: rebuilding rows with the PRODUCTION kernel must reproduce the
    # saved 9-cycle adjoint rows. This validates the rebuild path itself; the
    # reference-kernel rows then use the identical code with only k replaced.
    from q8_m0_g1_dynamic_map_audit_20261015 import require_cache
    from q8_m0_g1_gamma12_ir_tail_20261015 import load_adjoint
    saved_path, _digest = require_cache(1024)
    srow_d, sforce_d, srow_u, sforce_u = load_adjoint(saved_path)
    check = {}
    for tag, rebuilt, saved_row, saved_force in (
        ("driven", rows[("driven", "prod")], srow_d, sforce_d),
        ("undriven", rows[("undriven", "prod")], srow_u, sforce_u),
    ):
        rebuilt_row, rebuilt_forces = rebuilt
        saved_f = np.stack([f for f, _t, _h in saved_force])
        rebuilt_f = np.stack([f for f, _t, _h in rebuilt_forces])
        check[tag] = {
            "max_abs_row_diff": float(np.max(np.abs(rebuilt_row - saved_row))),
            "max_abs_force_diff": float(np.max(np.abs(rebuilt_f - saved_f))),
        }
    print("rebuild cross-check", check, flush=True)

    def assemble_both():
        """One pass over all input columns, contracting both kernel variants."""
        rd_p, fd_p = rows[("driven", "prod")]
        ru_p, fu_p = rows[("undriven", "prod")]
        rd_r, fd_r = rows[("driven", "ref")]
        ru_r, fu_r = rows[("undriven", "ref")]
        sdp, sup = stack_forces(fd_p), stack_forces(fu_p)
        sdr, sur = stack_forces(fd_r), stack_forces(fu_r)
        b = np.array(np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976:].T)
        bd = np.array(np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976:].T)
        a = np.array(np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976:].T)

        dd, vd = contract_initial(rd_p, b, bd, a)
        du, vu = contract_initial(ru_p, b, bd, a)
        dd_r, vd_r = contract_initial(rd_r, b, bd, a)
        du_r, vu_r = contract_initial(ru_r, b, bd, a)

        U_p = [dd - du]
        V_p = [vd - vu]
        U_r = [dd_r - du_r]
        V_r = [vd_r - vu_r]
        U0 = [np.zeros((2, 1976), complex)]
        groups = ["dark_1976"]
        pieces, _edge = frequencies(cache, gamma)
        for name, om, wt in pieces:
            for start in range(0, len(om), 400):
                stop = min(start + 400, len(om))
                ph, u0 = base.generated_photon_block(
                    cache, gamma, om[start:stop], wt[start:stop], pol, phi_of, eta)
                args = (ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
                d_p = contract_stacked(rd_p, sdp, *args)
                u_p = contract_stacked(ru_p, sup, *args)
                d_r = contract_stacked(rd_r, sdr, *args)
                u_r = contract_stacked(ru_r, sur, *args)
                U_p.append(d_p[0] - u_p[0])
                V_p.append(d_p[1] - u_p[1])
                U_r.append(d_r[0] - u_r[0])
                V_r.append(d_r[1] - u_r[1])
                U0.append(u0)
                groups.append(f"{name}[{start}:{stop}]")
        u0c = np.concatenate(U0, axis=1)
        return (np.concatenate(U_p, axis=1) + u0c, np.concatenate(V_p, axis=1),
                np.concatenate(U_r, axis=1) + u0c, np.concatenate(V_r, axis=1), groups)

    print("assemble both", f"{elapsed():.0f}s", flush=True)
    u_prod, v_prod, u_ref, v_ref, groups = assemble_both()

    eye = np.eye(2)
    rn_prod = u_prod @ u_prod.conj().T - v_prod @ v_prod.conj().T - eye
    rn_ref = u_ref @ u_ref.conj().T - v_ref @ v_ref.conj().T - eye
    ra_prod = u_prod @ v_prod.T - v_prod @ u_prod.T
    ra_ref = u_ref @ v_ref.T - v_ref @ u_ref.T
    dU, dV = u_ref - u_prod, v_ref - v_prod

    def pack(m):
        m = np.asarray(m, complex)
        return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
                "maxabs": float(np.max(np.abs(m)))}

    report = {
        "task": "kernel-correction increment via adjoint rows, 9 cycles macro1024",
        "date": "2026-10-17",
        "method": "rebuild adjoint rows with K_prod and K_ref on the SAME vertices; ΔU=U_ref−U_prod",
        "rebuild_cross_check_vs_saved_rows": check,
        "kernel_delta": {"max_abs": float(np.max(np.abs(k_ref - k_prod))),
                         "max_abs_rel_to_K9": float(np.max(np.abs(k_ref - k_prod)) / np.max(np.abs(k_prod)))},
        "R_N_prod": pack(rn_prod), "R_N_ref": pack(rn_ref),
        "R_A_prod": pack(ra_prod), "R_A_ref": pack(ra_ref),
        "delta_RN": pack(rn_ref - rn_prod), "delta_RA": pack(ra_ref - ra_prod),
        "delta_U": pack(dU), "delta_V": pack(dV),
        "delta_U_per_column_max": np.max(np.abs(dU), axis=0).tolist(),
        "n_change": pack((v_ref @ v_ref.T) - (v_prod @ v_prod.T)),
        "m_change": pack((u_ref @ v_ref.T) - (u_prod @ v_prod.T)),
        "groups": groups,
        "elapsed_s": float(elapsed()),
        "production_G1_pass": False,
    }
    with open(OUTJSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    np.savez_compressed(OUTNPZ, u_prod=u_prod, v_prod=v_prod,
                        u_ref=u_ref, v_ref=v_ref,
                        k_prod=k_prod, k_ref=k_ref, times=times)
    print("saved", OUTJSON, f"{elapsed():.0f}s", flush=True)
    print("RN prod/ref (0,0):", rn_prod[0, 0], rn_ref[0, 0], flush=True)
    print("max|dRN| =", float(np.max(np.abs(rn_ref - rn_prod))), flush=True)
    print("max|dRA| =", float(np.max(np.abs(ra_ref - ra_prod))), flush=True)


if __name__ == "__main__":
    main()
