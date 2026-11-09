"""Output-window convergence at macro step 1024.

The drive stays 8 cycles. Only the post-drive propagation and the upper
limit of the same detector kernel change. Windows are 9, 9.5, 10 and 11
sampling periods. Each new adjoint is rebuilt from the end of its own
window. The infrared tail is contracted again at every window.
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
from q8_m0_g1_boundary_sensitivity_20261015 import (
    contract_stacked, frequencies, radiated_from, stack_forces,
)
from q8_m0_g1_dynamic_map_audit_20261015 import require_cache
from q8_m0_g1_gamma12_column_order_20261015 import CACHE
from q8_m0_g1_gamma12_ir_tail_20261015 import POLE, load_adjoint, rows_from_vertices, vertices_progress
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_gamma12_time_order_20261015"
OUT = OUTDIR / "window_convergence_macro1024.json"
WINDOWS = (9.0, 9.5, 10.0, 11.0)


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


class Gram:
    def __init__(self):
        self.uu = np.zeros((2, 2), complex)
        self.vv = np.zeros((2, 2), complex)
        self.uv = np.zeros((2, 2), complex)
        self.vu = np.zeros((2, 2), complex)
        self.n = 0

    def add(self, u, v):
        self.n += int(u.shape[1])
        self.uu += u @ u.conj().T
        self.vv += v @ v.conj().T
        self.uv += u @ v.T
        self.vu += v @ u.T

    def report(self, subtract_identity=False):
        normal = self.uu - self.vv
        if subtract_identity:
            normal = normal - np.eye(2)
        anomalous = self.uv - self.vu
        return {"n_columns": self.n, "normal": pack(normal), "anomalous": pack(anomalous)}


def tag(name):
    if name.startswith("infrared"):
        return "infrared"
    if name.startswith("tail"):
        pole = float(name.split(":", 1)[1])
        if abs(pole - POLE) < 1e-8:
            return "lowest_pole"
    return "other"


def cache_path(cycles):
    label = f"{cycles:.1f}".replace(".", "p")
    return OUTDIR / f"adjoint_rows_macro1024_cycles{label}.npz"


def save_rows(path, row_d, force_d, time_d, step_d, row_u, force_u, time_u, step_u):
    if path.exists():
        raise FileExistsError(path)
    np.savez(path, row_d=row_d, force_d=force_d, time_d=time_d, step_d=step_d,
             row_u=row_u, force_u=force_u, time_u=time_u, step_u=step_u)


def build_window(cycles, coarse_d, fine_d, coarse_u, fine_u, h, gamma, pol):
    path = cache_path(cycles)
    if path.exists():
        print("load", path.name, flush=True)
        return load_adjoint(path)
    n_macro = int(round(cycles * 1024))
    print("pullback", cycles, n_macro, flush=True)
    row_d, force_d, time_d, step_d = rows_from_vertices(
        coarse_d[:n_macro], fine_d[:2 * n_macro], h, gamma, pol)
    row_u, force_u, time_u, step_u = rows_from_vertices(
        coarse_u[:n_macro], fine_u[:2 * n_macro], h, gamma, pol)
    save_rows(path, row_d, force_d, time_d, step_d, row_u, force_u, time_u, step_u)
    return row_d, list(zip(force_d, time_d, step_d)), row_u, list(zip(force_u, time_u, step_u))


def contract_window(rows, pieces, cache, gamma, pol, phi_of, eta):
    stacked_d = stack_forces(rows[1])
    stacked_u = stack_forces(rows[3])
    grams = {name: Gram() for name in ("all", "infrared", "lowest_pole", "other")}
    columns = []
    for name, om, wt in pieces:
        group = tag(name)
        for start in range(0, len(om), 200):
            stop = min(start + 200, len(om))
            ph, u0 = base.generated_photon_block(
                cache, gamma, om[start:stop], wt[start:stop], pol, phi_of, eta)
            driven = contract_stacked(
                rows[0], stacked_d, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
            undriven = contract_stacked(
                rows[2], stacked_u, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
            u = u0 + driven[0] - undriven[0]
            v = driven[1] - undriven[1]
            grams["all"].add(u, v)
            grams[group].add(u, v)
            columns.append((u, v, driven[0], undriven[0], driven[1], undriven[1]))
        print("contract", name, len(om), flush=True)
    packed = tuple(np.concatenate([item[i] for item in columns], axis=1) for i in range(6))
    return {name: gram.report(name == "all") for name, gram in grams.items()}, packed


def probe(rows, cache, gamma, pol, phi_of, eta):
    ph, _u0 = base.generated_photon_block(
        cache, gamma, np.array([0.11]), np.array([1.0]), pol, phi_of, eta)
    stacked_u = stack_forces(rows[3])
    stacked_d = stack_forces(rows[1])
    undriven = contract_stacked(
        rows[2], stacked_u, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
    driven = contract_stacked(
        rows[0], stacked_d, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
    radiated = radiated_from(ph, phi_of, eta, pol)
    return {
        "undriven_minus_radiated": float(np.max(np.abs(undriven[0] - radiated))),
        "undriven_v": float(np.max(np.abs(undriven[1]))),
        "driven_minus_undriven_u": float(np.max(np.abs(driven[0] - undriven[0]))),
    }


def trajectories(sample, geo, prot, quiet, h):
    n_coarse = 11 * 1024
    print("vertices", n_coarse, flush=True)
    coarse_d = vertices_progress(sample, geo, prot, n_coarse, h, "coarse driven")
    fine_d = vertices_progress(sample, geo, prot, 2 * n_coarse, 0.5 * h, "fine driven")
    coarse_u = vertices_progress(sample, geo, quiet, n_coarse, h, "coarse undriven")
    fine_u = vertices_progress(sample, geo, quiet, 2 * n_coarse, 0.5 * h, "fine undriven")
    saved_u = np.load(CACHE / "driven_u.npy", mmap_mode="r")
    saved_d = np.load(CACHE / "driven_d.npy", mmap_mode="r")
    for index in (0, 9215):
        gap = max(
            float(np.max(np.abs(coarse_d[index][0] - saved_u[index]))),
            float(np.max(np.abs(np.real(coarse_d[index][1]) - saved_d[index]))),
        )
        print("vertex check", index, gap, flush=True)
        if gap > 1e-8:
            raise AssertionError(f"Extended trajectory left the saved cache at step {index}")
    return coarse_d, fine_d, coarse_u, fine_u


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
    if int(prot.n_cycles) != 8 or abs(float(prot.t_pulse) - 8.0 * float(prot.period)) > 1e-8:
        raise AssertionError("The frozen drive is not 8 cycles")
    h = float(prot.period) / 1024.0
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    pieces, edge = frequencies(cache, gamma)
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    need_new = any(not cache_path(cycles).exists() for cycles in WINDOWS if cycles != 9.0)
    coarse_d = fine_d = coarse_u = fine_u = None
    if need_new:
        coarse_d, fine_d, coarse_u, fine_u = trajectories(sample, geo, prot, quiet, h)
    levels = {}
    reference = None
    for cycles in WINDOWS:
        if cycles == 9.0:
            path, digest = require_cache(1024)
            rows = load_adjoint(path)
            identity = digest
        else:
            rows = build_window(cycles, coarse_d, fine_d, coarse_u, fine_u, h, gamma, pol)
            identity = "rebuilt_from_window_end"
        print("contract window", cycles, flush=True)
        grams, packed = contract_window(rows, pieces, cache, gamma, pol, phi_of, eta)
        level = {"rows": identity, "grams": grams, "probe_0p11_weight1": probe(
            rows, cache, gamma, pol, phi_of, eta)}
        if reference is None:
            reference = packed
            level["difference_from_9_cycles"] = None
        else:
            du = packed[0] - reference[0]
            dv = packed[1] - reference[1]
            level["difference_from_9_cycles"] = {
                "normal": pack(du @ du.conj().T + reference[0] @ du.conj().T + du @ reference[0].conj().T
                               - (dv @ dv.conj().T + reference[1] @ dv.conj().T + dv @ reference[1].conj().T)),
                "anomalous": pack(du @ reference[1].T + reference[0] @ dv.T + du @ dv.T
                                  - dv @ reference[0].T - reference[1] @ du.T - dv @ du.T),
                "driven_raw_u_change": float(np.max(np.abs(packed[2] - reference[2]))),
                "undriven_raw_u_change": float(np.max(np.abs(packed[3] - reference[3]))),
                "driven_minus_undriven_u_change": float(np.max(np.abs(
                    (packed[2] - packed[3]) - (reference[2] - reference[3])))),
            }
        levels[f"{cycles:.1f}"] = level
        print("window", cycles, grams["all"]["normal"]["maxabs"], flush=True)
    write_new_json(OUT, {
        "drive_cycles_held_at_8": True,
        "windows_are_sampling_periods_after_t0": WINDOWS,
        "infrared_recomputed_at_every_window": True,
        "infrared_upper_edge": edge,
        "macro_1024_nine_cycle_reference_is_the_recomputed_gram": True,
        "levels": levels,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
