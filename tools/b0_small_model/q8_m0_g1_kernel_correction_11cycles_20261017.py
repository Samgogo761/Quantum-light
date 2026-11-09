"""11-cycle kernel-correction increment. New directory. Does not touch 9-cycle outputs.

Same method as q8_m0_g1_kernel_correction_20261017.py:
rebuild adjoint rows with K_prod and K_ref on the SAME 11-cycle vertices,
contract all input columns, report ΔR_N, ΔR_A and frozen n,m 2x2.

JSON stays small: no per-column ΔU dump.
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

import q8_m0_g1_kernel_correction_20261017 as k9
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_boundary_sensitivity_20261015 import contract_stacked, frequencies, stack_forces
from q8_m0_g1_gamma12_column_order_20261015 import CACHE
from q8_m0_g1_gamma12_ir_tail_20261015 import load_adjoint, vertices_progress
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import contract_initial
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
from q8_m0_k1_packet import phi_unnormalized, omega_grid, normalize_phi
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of
import q8_m0_gamma12_transpose_input_refine_20261008 as base

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_kernel_correction_11cycles_20261017"
OUTJSON = OUTDIR / "kernel_correction_11cycles.json"
OUTNPZ = OUTDIR / "kernel_correction_11cycles.npz"
ROWS = OUTDIR / "rows_kernel_correction_11cycles.npz"
PROGRESS = OUTDIR / "progress.txt"
SAVED11 = ROOT / "results/q_line/g1_gamma12_time_order_20261015/adjoint_rows_macro1024_cycles11p0.npz"

CYCLES = 11
BUDGET_S = 90 * 60
T0 = time.time()


def elapsed():
    return time.time() - T0


def prog(msg):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    line = f"{elapsed():.1f}s {msg}\n"
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def budget_left():
    if elapsed() > BUDGET_S:
        raise TimeoutError(f"wall-clock budget {BUDGET_S}s exceeded")


def pack(m):
    m = np.asarray(m, complex)
    return {
        "real": np.real(m).tolist(),
        "imag": np.imag(m).tolist(),
        "abs": np.abs(m).tolist(),
        "maxabs": float(np.max(np.abs(m))),
    }


def frozen_n(v):
    return np.conj(v) @ v.T


def frozen_m(u, v):
    return u @ v.T


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    PROGRESS.write_text("start 11-cycle kernel correction\n", encoding="utf-8")
    k9.T0 = T0
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
    prog("kernels")
    k_prod = k9.kernels_for(times, gamma, prot, eta, "prod")
    k_ref = k9.kernels_for(times, gamma, prot, eta, "ref")
    prog(f"kernels ready max|dK|={np.max(np.abs(k_ref-k_prod)):.3e}")

    rows = {}
    if ROWS.exists():
        prog("load rows cache")
        rows = k9.load_rows_cache(ROWS)
    else:
        for drive, protocol in (("driven", prot), ("undriven", quiet)):
            budget_left()
            prog(f"vertices {drive}")
            coarse = vertices_progress(sample, geo, protocol, n, h, f"coarse {drive}")
            fine = vertices_progress(sample, geo, protocol, 2 * n, 0.5 * h, f"fine {drive}")
            prog(f"vertices done {drive}")
            budget_left()
            for kind, kk in (("prod", k_prod), ("ref", k_ref)):
                prog(f"rows {drive} {kind}")
                rows[(drive, kind)] = k9.rows_with_kernels(coarse, fine, h, gamma, pol, kk)
            del coarse, fine
            gc.collect()
        k9.save_rows_cache(ROWS, rows)
        prog("rows cached")

    if not SAVED11.exists():
        raise FileNotFoundError(SAVED11)
    srow_d, sforce_d, srow_u, sforce_u = load_adjoint(SAVED11)
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
    prog(f"rebuild cross-check {check}")

    def assemble_both():
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
        U_p, V_p = [dd - du], [vd - vu]
        U_r, V_r = [dd_r - du_r], [vd_r - vu_r]
        U0 = [np.zeros((2, 1976), complex)]
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
            prog(f"assembled {name}")
        u0c = np.concatenate(U0, axis=1)
        return (np.concatenate(U_p, axis=1) + u0c, np.concatenate(V_p, axis=1),
                np.concatenate(U_r, axis=1) + u0c, np.concatenate(V_r, axis=1))

    prog("assemble both")
    u_prod, v_prod, u_ref, v_ref = assemble_both()
    eye = np.eye(2)
    rn_prod = u_prod @ u_prod.conj().T - v_prod @ v_prod.conj().T - eye
    rn_ref = u_ref @ u_ref.conj().T - v_ref @ v_ref.conj().T - eye
    ra_prod = u_prod @ v_prod.T - v_prod @ u_prod.T
    ra_ref = u_ref @ v_ref.T - v_ref @ u_ref.T
    n_prod, n_ref = frozen_n(v_prod), frozen_n(v_ref)
    m_prod, m_ref = frozen_m(u_prod, v_prod), frozen_m(u_ref, v_ref)
    report = {
        "task": "kernel-correction increment, 11 cycles macro1024",
        "date": "2026-10-17",
        "rebuild_cross_check_vs_saved_rows": check,
        "kernel_delta": {
            "max_abs": float(np.max(np.abs(k_ref - k_prod))),
            "max_abs_rel_to_K11": float(np.max(np.abs(k_ref - k_prod)) / max(np.max(np.abs(k_prod)), 1e-30)),
        },
        "R_N_prod": pack(rn_prod),
        "R_N_ref": pack(rn_ref),
        "R_A_prod": pack(ra_prod),
        "R_A_ref": pack(ra_ref),
        "delta_RN": pack(rn_ref - rn_prod),
        "delta_RA": pack(ra_ref - ra_prod),
        "n_prod_frozen": pack(n_prod),
        "n_ref_frozen": pack(n_ref),
        "dn_frozen": pack(n_ref - n_prod),
        "m_prod": pack(m_prod),
        "m_ref": pack(m_ref),
        "dm": pack(m_ref - m_prod),
        "cross_pol": {
            "dm_01_abs": float(abs((m_ref - m_prod)[0, 1])),
            "dn_01_abs": float(abs((n_ref - n_prod)[0, 1])),
            "m_prod_01_abs": float(abs(m_prod[0, 1])),
            "n_prod_01_abs": float(abs(n_prod[0, 1])),
        },
        "max_abs_dU": float(np.max(np.abs(u_ref - u_prod))),
        "max_abs_dV": float(np.max(np.abs(v_ref - v_prod))),
        "elapsed_s": float(elapsed()),
        "production_G1_pass": False,
        "does_not_overwrite_9cycle_outputs": True,
    }
    OUTJSON.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUTNPZ, u_prod=u_prod, v_prod=v_prod,
                        u_ref=u_ref, v_ref=v_ref, k_prod=k_prod, k_ref=k_ref, times=times)
    prog(f"saved {OUTJSON}")
    prog(f"RN prod/ref (0,0) {rn_prod[0,0]} {rn_ref[0,0]}")


if __name__ == "__main__":
    raise SystemExit(main())
