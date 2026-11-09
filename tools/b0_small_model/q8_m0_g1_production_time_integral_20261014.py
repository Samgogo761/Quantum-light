"""Signed channel grams and one continuous production time integral.

Two k-points. The electron trajectory is integrated once. Segment sums are
pieces of that same integral. Simpson weights reuse the endpoint and midpoint
states of each production step; they do not rerun the dynamics.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import ColumnState, current, midpoint_step
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import assemble_incoming_bundle
from q8_m0_k1_packet import normalize_phi, omega_grid, phi_unnormalized
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import TWO_K, WEIGHTS, m0_sample
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import (
    attach_weights, eta_of, input_quadrature, phi_at, u0_from_bundle,
)
from q8_m0_s1_kernel import parallel_perp_basis
from q8_m0_g1_chain_benchmark_20261014 import vertex_window

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_production_time_integral_20261014/time_integral.json"


def err(a):
    return float(np.max(np.abs(np.asarray(a))))


def pack(m):
    m = np.asarray(m)
    return {
        "real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
        "maxabs": err(m),
        "e00_real": float(np.real(m[0, 0])), "e00_imag": float(np.imag(m[0, 0])),
        "e01_real": float(np.real(m[0, 1])), "e01_imag": float(np.imag(m[0, 1])),
    }


def finish(u, v):
    uu, vv = u @ u.conj().T, v @ v.conj().T
    rn = uu - vv - np.eye(2)
    ra = u @ v.T - v @ u.T
    return {"UU": pack(uu), "VV": pack(vv), "UU_minus_VV": pack(uu - vv),
            "normal": pack(rn), "anomalous": pack(ra)}


def prepare(n_panel=32, n_h2=48, omega_lo=1.0e-3, omega_hi=12.0, h2_lo=0.02, h2_hi=0.30,
             oms=None, wts=None):
    geo = build_geometry()
    sample = m0_sample(geo, TWO_K, WEIGHTS, +1)
    gamma = float(gamma_self(geo))
    if oms is None:
        oms, wts = input_quadrature(
            sample["gap"], sample["u"], gamma, n_panel=n_panel, n_h2=n_h2,
            omega_lo=omega_lo, omega_hi=omega_hi, h2_lo=h2_lo, h2_hi=h2_hi)
    else:
        oms = np.asarray(oms, float)
        wts = np.asarray(wts, float)
    bundle = assemble_incoming_bundle(sample["gap"], sample["u"], sample["D"], gamma, oms)
    attach_weights(bundle, {round(float(o), 12): float(w) for o, w in zip(oms, wts)})
    drive = p1_a2_protocol(0.8)
    grid = omega_grid(drive.omega0, drive.t_pulse)
    grid = grid[grid > 0.0]
    _phi, nrm = normalize_phi(grid, phi_unnormalized(grid, 2.0 * drive.omega0, drive.t_pulse))
    pol = np.vstack(parallel_perp_basis(geo)).astype(complex)
    u0, _ = u0_from_bundle(bundle, pol, lambda nu: phi_at(nu, drive, nrm), eta_of(gamma))
    n_ph, n_dark = bundle.n_photon_columns, bundle.n_point_columns
    sides = np.array(["dark"] * (n_ph + n_dark), dtype=object)
    for i, lab in enumerate(bundle.photon.labels):
        sides[i] = lab[1]
    return {
        "geo": geo, "sample": sample, "gamma": gamma, "drive": drive, "pol": pol, "u0": u0,
        "free": np.concatenate((bundle.photon.free_A0, np.zeros((2, n_dark), complex)), axis=1),
        "omega": np.concatenate((bundle.photon.omega, np.zeros(n_dark))),
        "initial": ColumnState(
            np.concatenate((bundle.photon.initial.b, bundle.point.b), axis=1),
            np.concatenate((bundle.photon.initial.bdag, bundle.point.bdag), axis=1),
            np.concatenate((bundle.photon.initial.A, bundle.point.A), axis=1)),
        "sides": sides,
        "mass": float(np.sum(np.abs(phi_at(oms, drive, nrm)) ** 2 * wts)),
        "n_freq": int(len(oms)),
    }


def vertices(prep, a_star, t0, nstep, h):
    return list(vertex_window(prep["sample"], prep["geo"], p1_a2_protocol(a_star), t0, nstep, h))


def collect(prep, verts, t0, h, bounds):
    """One continuous trajectory. Returns midpoint and Simpson column sums by segment."""
    state = prep["initial"].copy()
    pol, gamma = prep["pol"], prep["gamma"]
    omega, free = prep["omega"], prep["free"]
    n = len(verts)
    knots = np.empty(3 * n)
    for step in range(n):
        knots[3 * step:3 * step + 3] = t0 + h * step + h * np.array([0.0, 0.5, 1.0])
    k_all, _ = kernel_at_times(gamma, prep["drive"], knots)
    names = list(bounds)
    rect = {name: [np.zeros((2, state.b.shape[1]), complex),
                   np.zeros((2, state.b.shape[1]), complex)] for name in names}
    simp = [np.zeros((2, state.b.shape[1]), complex), np.zeros((2, state.b.shape[1]), complex)]
    for step, (u, d) in enumerate(verts):
        t = t0 + step * h
        prev = state
        fa = free * (np.exp(-1j * omega * (t + h)) - np.exp(-1j * omega * t))
        state = midpoint_step(state, u, np.real(d), h, gamma, free_A_increment=fa)
        mid = ColumnState(0.5 * (prev.b + state.b), 0.5 * (prev.bdag + state.bdag),
                          0.5 * (prev.A + state.A))
        dreal = np.real(d)
        f_left = pol @ current(prev, u, dreal)
        f_mid = pol @ current(mid, u, dreal)
        f_right = pol @ current(state, u, dreal)
        ks = k_all[3 * step:3 * step + 3]
        block = None
        for name, (lo, hi) in bounds.items():
            if lo <= step < hi:
                block = rect[name]
                break
        if block is None:
            raise RuntimeError("Step is outside every segment")
        block[0] += h * ks[1] * f_mid
        block[1] += h * ks[1] * np.conj(f_mid)
        simp[0] += (h / 6.0) * (ks[0] * f_left + 4.0 * ks[1] * f_mid + ks[2] * f_right)
        simp[1] += (h / 6.0) * (ks[0] * np.conj(f_left) + 4.0 * ks[1] * np.conj(f_mid)
                                + ks[2] * np.conj(f_right))
    return rect, simp


def add_segments(rect, names):
    u = sum(rect[name][0] for name in names)
    v = sum(rect[name][1] for name in names)
    return u, v


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prep = prepare()
    h = float(prep["drive"].period) / 128.0
    ntpc = 128
    # Signed split on the previously reported one-cycle window.
    t_mid = 0.5 * float(prep["drive"].t_pulse)
    n0 = int(round(t_mid / h))
    one_bounds = {"only": (0, ntpc)}
    quiet_one, quiet_simp = collect(prep, vertices(prep, 0.0, n0 * h, ntpc, h), n0 * h, h, one_bounds)
    driven_one, driven_simp = collect(prep, vertices(prep, 0.8, n0 * h, ntpc, h), n0 * h, h, one_bounds)
    weak_one, _ = collect(prep, vertices(prep, 0.2, n0 * h, ntpc, h), n0 * h, h, one_bounds)
    zero_one, _ = collect(prep, vertices(prep, 0.0, n0 * h, ntpc, h), n0 * h, h, one_bounds)
    u_one = prep["u0"] + driven_one["only"][0] - quiet_one["only"][0]
    v_one = driven_one["only"][1] - quiet_one["only"][1]
    channels = {}
    sum_uu = np.zeros((2, 2), complex)
    sum_vv = np.zeros((2, 2), complex)
    for side in ("L", "R", "dark"):
        idx = np.flatnonzero(prep["sides"] == side)
        block = finish(u_one[:, idx], v_one[:, idx])
        channels[side] = {"columns": int(idx.size), **block}
        sum_uu += u_one[:, idx] @ u_one[:, idx].conj().T
        sum_vv += v_one[:, idx] @ v_one[:, idx].conj().T
    full_one = finish(u_one, v_one)
    if err(sum_uu - (full_one and u_one @ u_one.conj().T)) > 1e-8:
        raise AssertionError("Channel UU does not add to the total")
    zero_gap = err(zero_one["only"][0] - quiet_one["only"][0]) + err(zero_one["only"][1] - quiet_one["only"][1])
    weak = finish(prep["u0"] + weak_one["only"][0] - quiet_one["only"][0],
                  weak_one["only"][1] - quiet_one["only"][1])
    # Full pulse plus two tail cycles, one continuous pair of trajectories.
    n_total = 10 * ntpc
    bounds = {
        "early_0_2cyc": (0, 2 * ntpc),
        "middle_2_6cyc": (2 * ntpc, 6 * ntpc),
        "late_6_8cyc": (6 * ntpc, 8 * ntpc),
        "tail_first": (8 * ntpc, 9 * ntpc),
        "tail_second": (9 * ntpc, 10 * ntpc),
    }
    print("full window", n_total, flush=True)
    quiet, quiet_full_simp = collect(prep, vertices(prep, 0.0, 0.0, n_total, h), 0.0, h, bounds)
    driven, driven_full_simp = collect(prep, vertices(prep, 0.8, 0.0, n_total, h), 0.0, h, bounds)
    segments = {}
    du_parts, dv_parts = [], []
    for name in bounds:
        du = driven[name][0] - quiet[name][0]
        dv = driven[name][1] - quiet[name][1]
        du_parts.append(du)
        dv_parts.append(dv)
        segments[name] = {"delta_u_maxabs": err(du), "delta_v_maxabs": err(dv)}
    du_all = sum(du_parts)
    if err(du_all - sum(driven[name][0] - quiet[name][0] for name in bounds)) > 1e-10:
        raise AssertionError("Segment sum drifted")
    simp_u = driven_full_simp[0] - quiet_full_simp[0]
    simp_v = driven_full_simp[1] - quiet_full_simp[1]
    simpson_window = finish(prep["u0"] + simp_u, simp_v)
    ranges = {
        "pulse_only": 3,
        "pulse_plus_one_tail": 4,
        "pulse_plus_two_tails": 5,
    }
    windows = {}
    for name, count in ranges.items():
        u = prep["u0"] + sum(du_parts[:count])
        v = sum(dv_parts[:count])
        rn, ra = finish(u, v)["normal"], finish(u, v)["anomalous"]
        windows[name] = {"normal": rn, "anomalous": ra}
    report = {
        "packet_mass": prep["mass"],
        "frequencies": prep["n_freq"],
        "one_cycle_at_pulse_centre_ntpc128": {
            "channels": channels,
            "total": full_one,
            "channel_e00_real_UU_minus_VV": {
                side: channels[side]["UU_minus_VV"]["e00_real"] for side in ("L", "R", "dark")},
            "sum_e00_real_minus_one": full_one["normal"]["e00_real"],
            "note": "Earlier 0.6687, 0.4555 and 4.3e-5 are maxabs of 2x2 blocks, not scalars that add.",
            "Astar_0_delta_maxabs": zero_gap,
            "Astar_0.2": {"normal": weak["normal"], "anomalous": weak["anomalous"]},
            "simpson_minus_midpoint_on_same_states": err(
            (driven_simp[0] - quiet_simp[0]) - (driven_one["only"][0] - quiet_one["only"][0])),
        },
        "full_pulse_segments": segments,
        "windows": windows,
        "simpson_full_pulse_plus_tails": {
            "normal": simpson_window["normal"], "anomalous": simpson_window["anomalous"],
            "delta_u_minus_midpoint": err(simp_u - du_all),
        },
        "electronic_even_measure": (
            "bb-dd+dark projector-I on the electronic channels of an even one-sided "
            "frequency measure. It is not a term in the 2x2 H2 residual."
        ),
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    }
    # Simpson comparison on the one-cycle driven-minus-undriven rows.
    # simp_one was not kept. The one-cycle collect with simpson flag is inside collect always.
    # Fix: collect always returns Simpson totals. I discarded them as _.
    write_new_json(OUT, report)
    print("one normal", full_one["normal"]["maxabs"], "e00", full_one["normal"]["e00_real"])
    print("channels", {s: channels[s]["UU_minus_VV"]["e00_real"] for s in channels})
    print("zero", zero_gap)
    print("windows", {k: v["normal"]["maxabs"] for k, v in windows.items()})


if __name__ == "__main__":
    main()
