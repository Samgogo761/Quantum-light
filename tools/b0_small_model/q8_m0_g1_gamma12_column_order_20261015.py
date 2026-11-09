"""Midpoint versus local Richardson on a few full-Gamma12 columns.

The electronic trajectory uses the same 12 by 12 grid as production.
Only four input frequencies are propagated. The ntpc=1024 vertices are the
saved production vertices; coarser vertices are regenerated with the same step.
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
from q8_m0_continuum_columns_20261004 import ColumnState
from q8_m0_g1_gamma12_tail_columns_20261014 import contribution, load_rows
from q8_m0_g1_local_richardson_20261014 import propagate_local_richardson
from q8_m0_g1_two_k_timestep_20261014 import propagate
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import advance_half_frame, vertices_from_frame
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007/arrays"
OUT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/column_richardson.json"
TARGETS = (0.5089945843, 0.247, 0.95, 3.0)
CYCLES = 9


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def vertices(sample, geo, prot, nstep, h):
    frame = sample["frame"].copy()
    out = []
    for step in range(nstep):
        t = step * h
        frame = advance_half_frame(geo, sample["kc"], prot, sample["neel"], frame, t + 0.25 * h, h)
        out.append(vertices_from_frame(
            geo, sample["kc"], prot, sample["neel"], t + 0.5 * h, frame, sample["weights"]))
        frame = advance_half_frame(geo, sample["kc"], prot, sample["neel"], frame, t + 0.75 * h, h)
    return out


def load_saved(name, nstep):
    u = np.load(CACHE / f"{name}_u.npy", mmap_mode="r")
    d = np.load(CACHE / f"{name}_d.npy", mmap_mode="r")
    return [(np.array(u[i]), np.array(d[i])) for i in range(nstep)]


def column_gram(prep, driven, undriven):
    du, dv = driven[0] - undriven[0], driven[1] - undriven[1]
    u = prep["u0"] + du
    v = dv
    return contribution(u, v)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _rows, cache, t0, _meta = load_rows()
    if abs(t0) > 0:
        raise AssertionError("Saved run does not start at t=0")
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    prot = p1_a2_protocol(0.8)
    quiet = p1_a2_protocol(0.0)
    h_saved = float(np.load(CACHE / "h.npy"))
    if abs(h_saved - float(prot.period) / 1024) > 1e-12:
        raise AssertionError("Saved step is not one 1024th of the period")
    n_saved = CYCLES * 1024
    check = vertices(sample, geo, prot, 2, h_saved)
    saved_u = np.load(CACHE / "driven_u.npy", mmap_mode="r")
    if np.max(np.abs(check[0][0] - saved_u[0])) > 1e-12 or np.max(np.abs(check[1][0] - saved_u[1])) > 1e-12:
        raise AssertionError("Regenerated vertices do not match the saved production vertices")
    k_saved = np.load(CACHE / "k.npy")
    times = (np.arange(n_saved) + 0.5) * h_saved
    k_check, _ = kernel_at_times(float(np.asarray(cache["gamma"])), prot, times)
    if np.max(np.abs(k_check - k_saved)) > 1e-10:
        raise AssertionError("Detector kernel does not match the saved production kernel")
    saved_oms = np.load(CACHE / "oms.npy")
    saved_wts = np.load(CACHE / "wts.npy")
    chosen = []
    for target in TARGETS:
        index = int(np.argmin(np.abs(saved_oms - target)))
        chosen.append((float(saved_oms[index]), float(saved_wts[index])))
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    ph, u0 = base.generated_photon_block(
        cache, gamma, np.array([o for o, _w in chosen]), np.array([w for _o, w in chosen]),
        pol, phi_of, base.q.eta_of(gamma))
    prep = {
        "initial": ColumnState(ph.initial.b, ph.initial.bdag, ph.initial.A),
        "free": ph.free_A0, "omega": ph.omega, "gamma": gamma, "pol": pol,
        "u0": u0, "drive": prot,
    }
    results = {}
    grids = {
        256: None,
        512: None,
        1024: {"driven": load_saved("driven", n_saved), "undriven": load_saved("undriven", n_saved)},
    }
    for ntpc in (256, 512):
        h = float(prot.period) / ntpc
        nstep = CYCLES * ntpc
        print("vertices", ntpc, flush=True)
        grids[ntpc] = {
            "driven": vertices(sample, geo, prot, nstep, h),
            "undriven": vertices(sample, geo, quiet, nstep, h),
        }
    for ntpc, grid_v in grids.items():
        h = float(prot.period) / ntpc
        print("midpoint", ntpc, flush=True)
        driven = propagate(prep, grid_v["driven"], h)
        undriven = propagate(prep, grid_v["undriven"], h)
        normal, anomalous = column_gram(prep, driven, undriven)
        results[f"midpoint_{ntpc}"] = {"normal": pack(normal), "anomalous": pack(anomalous)}
    for macro in (256, 512):
        print("richardson", macro, flush=True)
        h = float(prot.period) / macro
        driven = propagate_local_richardson(prep, grids[macro]["driven"], grids[macro * 2]["driven"], h)
        undriven = propagate_local_richardson(prep, grids[macro]["undriven"], grids[macro * 2]["undriven"], h)
        normal, anomalous = column_gram(prep, driven, undriven)
        results[f"richardson_{macro}"] = {"normal": pack(normal), "anomalous": pack(anomalous)}
    ref_n = np.array(results["midpoint_1024"]["normal"]["real"]) + 1j * np.array(results["midpoint_1024"]["normal"]["imag"])
    ref_a = np.array(results["midpoint_1024"]["anomalous"]["real"]) + 1j * np.array(results["midpoint_1024"]["anomalous"]["imag"])
    distance = {}
    for name, item in results.items():
        if name == "midpoint_1024":
            continue
        normal = np.array(item["normal"]["real"]) + 1j * np.array(item["normal"]["imag"])
        anomalous = np.array(item["anomalous"]["real"]) + 1j * np.array(item["anomalous"]["imag"])
        distance[name] = {"normal": pack(normal - ref_n), "anomalous": pack(anomalous - ref_a)}
        print(name, distance[name]["normal"]["maxabs"], distance[name]["anomalous"]["maxabs"], flush=True)
    write_new_json(OUT, {
        "frequencies": [o for o, _w in chosen], "cycles": CYCLES, "columns": 16,
        "levels": results, "distance_from_midpoint_1024": distance,
        "production_G1_pass": False, "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
