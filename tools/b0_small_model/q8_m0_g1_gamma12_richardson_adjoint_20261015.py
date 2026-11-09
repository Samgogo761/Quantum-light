"""Adjoint of one Richardson macro step, checked against the forward map.

The combination is three midpoint steps from the same incoming state:
S_out = (4*F - C)/3 and dY = (4*(dY1+dY2) - dYc)/3.
Each midpoint adjoint is the existing pullback. Old midpoint rows are not reused.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_g1_local_richardson_20261014 import propagate_local_richardson
from q8_m0_output_transpose_20261008 import midpoint_output_pullback
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/richardson_adjoint.json"


def detector(k, pol, h, scale):
    block = h * np.concatenate((k * pol, np.conjugate(k) * pol), axis=0)
    return scale * block


def macro_pullback(row_next, step, gamma, pol):
    half = 0.5 * step["H"]
    row_f, force_f2 = midpoint_output_pullback(
        (4.0 / 3.0) * row_next, step["u2"], step["d2"], half, gamma,
        detector(step["k2"], pol, half, 4.0 / 3.0))
    row_from_fine, force_f1 = midpoint_output_pullback(
        row_f, step["u1"], step["d1"], half, gamma,
        detector(step["k1"], pol, half, 4.0 / 3.0))
    row_from_coarse, force_c = midpoint_output_pullback(
        (-1.0 / 3.0) * row_next, step["uc"], step["dc"], step["H"], gamma,
        detector(step["kc"], pol, step["H"], -1.0 / 3.0))
    return row_from_fine + row_from_coarse, force_c, force_f1, force_f2


def contract_initial(row, b, bdag, a):
    ne = b.shape[0]
    yy = row[:, :ne] @ b + row[:, ne:2 * ne] @ bdag + row[:, 2 * ne:] @ a
    return yy[:2], np.conjugate(yy[2:])


def free_increment(free, omega, t, h):
    phase = np.exp(-1j * omega * t) * np.expm1(-1j * omega * h)
    return free * phase


def adjoint_output(steps, gamma, pol, b, bdag, a, free, omega):
    ne = b.shape[0]
    row = np.zeros((4, 2 * ne + 2), complex)
    forces = []
    for step in reversed(steps):
        row, force_c, force_f1, force_f2 = macro_pullback(row, step, gamma, pol)
        forces.append((step, force_c, force_f1, force_f2))
    raw_u, raw_v = contract_initial(row, b, bdag, a)
    for step, force_c, force_f1, force_f2 in forces:
        raw_c = force_c @ free_increment(free, omega, step["t"], step["H"])
        raw_1 = force_f1 @ free_increment(free, omega, step["t"], 0.5 * step["H"])
        raw_2 = force_f2 @ free_increment(free, omega, step["t"] + 0.5 * step["H"], 0.5 * step["H"])
        raw = raw_c + raw_1 + raw_2
        raw_u += raw[:2]
        raw_v += np.conjugate(raw[2:])
    return raw_u, raw_v


def build_steps(verts_c, verts_f, h, gamma, drive):
    n = len(verts_c)
    times = np.empty(3 * n)
    for i in range(n):
        t = i * h
        times[3 * i:3 * i + 3] = (t + 0.25 * h, t + 0.5 * h, t + 0.75 * h)
    kernels, _ = kernel_at_times(gamma, drive, times)
    steps = []
    for i, (uc, dc) in enumerate(verts_c):
        u1, d1 = verts_f[2 * i]
        u2, d2 = verts_f[2 * i + 1]
        kf1, kc, kf2 = kernels[3 * i:3 * i + 3]
        steps.append({
            "t": i * h, "H": h,
            "uc": uc, "dc": np.real(dc), "u1": u1, "d1": np.real(d1),
            "u2": u2, "d2": np.real(d2), "kc": kc, "k1": kf1, "k2": kf2,
        })
    return steps


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    from q8_m0_g1_gamma12_column_order_20261015 import (
        CYCLES, TARGETS, vertices, CACHE,
    )
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_continuum_columns_20261004 import ColumnState
    from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
    from q8_m0_model import build_geometry
    from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
    import q8_m0_gamma12_transpose_input_refine_20261008 as base

    _rows, cache, t0, _meta = load_rows()
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    prot = p1_a2_protocol(0.8)
    quiet = p1_a2_protocol(0.0)
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    saved_oms = np.load(CACHE / "oms.npy")
    saved_wts = np.load(CACHE / "wts.npy")
    chosen = []
    for target in TARGETS:
        index = int(np.argmin(np.abs(saved_oms - target)))
        chosen.append((float(saved_oms[index]), float(saved_wts[index])))
    packet = base.q.omega_grid(prot.omega0, prot.t_pulse)
    packet = packet[packet > 0.0]
    _phi, norm = base.q.normalize_phi(packet, base.q.phi_unnormalized(packet, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    ph, _u0 = base.generated_photon_block(
        cache, gamma, np.array([o for o, _w in chosen]), np.array([w for _o, w in chosen]),
        pol, phi_of, base.q.eta_of(gamma))
    prep = {
        "initial": ColumnState(ph.initial.b, ph.initial.bdag, ph.initial.A),
        "free": ph.free_A0, "omega": ph.omega, "gamma": gamma, "pol": pol, "drive": prot,
    }
    macro = 64
    h = float(prot.period) / macro
    nstep = CYCLES * macro
    print("vertices", nstep, flush=True)
    driven_c = vertices(sample, geo, prot, nstep, h)
    driven_f = vertices(sample, geo, prot, 2 * nstep, 0.5 * h)
    undriven_c = vertices(sample, geo, quiet, nstep, h)
    undriven_f = vertices(sample, geo, quiet, 2 * nstep, 0.5 * h)
    forward_d = propagate_local_richardson(prep, driven_c, driven_f, h)
    forward_u = propagate_local_richardson(prep, undriven_c, undriven_f, h)
    steps_d = build_steps(driven_c, driven_f, h, gamma, prot)
    steps_u = build_steps(undriven_c, undriven_f, h, gamma, prot)
    adj_d = adjoint_output(steps_d, gamma, pol, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
    adj_u = adjoint_output(steps_u, gamma, pol, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
    diff = max(
        float(np.max(np.abs(adj_d[0] - forward_d[0]))),
        float(np.max(np.abs(adj_d[1] - forward_d[1]))),
        float(np.max(np.abs(adj_u[0] - forward_u[0]))),
        float(np.max(np.abs(adj_u[1] - forward_u[1]))),
    )
    print("adjoint minus forward", diff, flush=True)
    write_new_json(OUT, {
        "macro_ntpc": macro, "cycles": CYCLES, "columns": int(ph.initial.b.shape[1]),
        "max_abs_difference": diff,
        "forward_matches_adjoint": diff < 1.0e-8,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
