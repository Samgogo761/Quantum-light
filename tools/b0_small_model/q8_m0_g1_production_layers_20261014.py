"""Where the production output residual first appears on two k-points.

Uses the production quadrature, stationary incoming columns, midpoint step
and H2 kernel. Does not change gamma, eta, D, or the packet. A short window
at the pulse centre is enough to separate the static row from the driven
current integral.
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
    attach_weights, electronic_measure_report, eta_of, input_quadrature, phi_at, u0_from_bundle,
)
from q8_m0_s1_kernel import parallel_perp_basis
from q8_m0_g1_chain_benchmark_20261014 import vertex_window

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_production_layers_20261014/production_layers_fullpulse.json"


def err(a):
    return float(np.max(np.abs(np.asarray(a))))


def pack(m):
    m = np.asarray(m)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(), "maxabs": err(m)}


def residual(u, v):
    rn = u @ u.conj().T - v @ v.conj().T - np.eye(u.shape[0])
    ra = u @ v.T - v @ u.T
    return pack(rn), pack(ra)


def gram(u, v):
    return pack(u @ u.conj().T - v @ v.conj().T), pack(u @ v.T - v @ u.T)


def propagate(initial, omega, free_a0, vertices, k_scalar, pol, gamma, h):
    state = initial.copy()
    acc_u = np.zeros((pol.shape[0], state.b.shape[1]), complex)
    acc_v = np.zeros_like(acc_u)
    for step, (u, d) in enumerate(vertices):
        t = step * h
        prev = state
        phase_now = np.exp(-1j * omega * t)
        phase_next = np.exp(-1j * omega * (t + h))
        fa = free_a0 * (phase_next - phase_now)
        state = midpoint_step(state, u, np.real(d), h, gamma, free_A_increment=fa)
        mid = ColumnState(0.5 * (prev.b + state.b), 0.5 * (prev.bdag + state.bdag),
                          0.5 * (prev.A + state.A))
        src = pol @ current(mid, u, np.real(d))
        acc_u += h * k_scalar[step] * src
        acc_v += h * k_scalar[step] * np.conj(src)
    return acc_u, acc_v


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    geo = build_geometry()
    sample = m0_sample(geo, TWO_K, WEIGHTS, +1)
    gamma = float(gamma_self(geo))
    eta = eta_of(gamma)
    oms, wts = input_quadrature(sample["gap"], sample["u"], gamma, n_panel=32,
                                omega_lo=1.0e-3, omega_hi=12.0)
    bundle = assemble_incoming_bundle(sample["gap"], sample["u"], sample["D"], gamma, oms)
    attach_weights(bundle, {round(float(o), 12): float(w) for o, w in zip(oms, wts)})
    prot = p1_a2_protocol(0.8)
    grid = omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, nrm = normalize_phi(grid, phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    pol = np.vstack(parallel_perp_basis(geo)).astype(complex)
    u0, labels = u0_from_bundle(bundle, pol, lambda nu: phi_at(nu, prot, nrm), eta)
    photon_labels = bundle.photon.labels
    n_ph = bundle.n_photon_columns
    n_dark = bundle.n_point_columns
    sides = [("L" if i < n_ph and photon_labels[i][1] == "L" else
              "R" if i < n_ph else "dark") for i in range(u0.shape[1])]
    free = np.concatenate((bundle.photon.free_A0, np.zeros((2, n_dark), complex)), axis=1)
    omega = np.concatenate((bundle.photon.omega, np.zeros(n_dark)))
    initial = ColumnState(
        np.concatenate((bundle.photon.initial.b, bundle.point.b), axis=1),
        np.concatenate((bundle.photon.initial.bdag, bundle.point.bdag), axis=1),
        np.concatenate((bundle.photon.initial.A, bundle.point.A), axis=1),
    )
    # Weight applied once: free amplitude is sqrt(w)*eta/sqrt(omega).
    weight_check = 0.0
    for j, lab in enumerate(photon_labels):
        nu, _side, pidx = lab
        expected = np.sqrt(wts[np.argmin(np.abs(oms - nu))]) * eta / np.sqrt(nu)
        weight_check = max(weight_check, abs(abs(free[int(pidx), j]) - expected))
    h = float(prot.period) / 32.0
    nstep = 8 * 32 + 32
    t0 = 0.0
    npre = int(round(t0 / h))
    driven_v = list(vertex_window(sample, geo, prot, npre * h, nstep, h))
    quiet_v = list(vertex_window(sample, geo, p1_a2_protocol(0.0), npre * h, nstep, h))
    times = npre * h + (np.arange(nstep) + 0.5) * h
    k_scalar, _ = kernel_at_times(gamma, prot, times)
    du, dv = propagate(initial, omega, free, driven_v, k_scalar, pol, gamma, h)
    uu, uv = propagate(initial, omega, free, quiet_v, k_scalar, pol, gamma, h)
    layers = {}
    for name, u, v in (
        ("U0_only", u0, np.zeros_like(u0)),
        ("U0_plus_undriven_integral", u0 + uu, uv),
        ("U0_plus_driven_integral", u0 + du, dv),
        ("driven_minus_undriven", u0 + du - uu, dv - uv),
    ):
        rn, ra = residual(u, v)
        layers[name] = {"normal": rn, "anomalous": ra}
    groups = {}
    u = u0 + du - uu
    v = dv - uv
    for side in ("L", "R", "dark"):
        idx = [i for i, s in enumerate(sides) if s == side]
        groups[side] = {"columns": len(idx)}
        if idx:
            gn, ga = gram(u[:, idx], v[:, idx])
            groups[side]["normal_without_identity"] = gn
            groups[side]["anomalous"] = ga
    total_n, total_a = gram(u, v)
    elec = electronic_measure_report(sample, gamma, oms, wts)
    report = {
        "k_points": 2,
        "electronic_modes": int(sample["n_el"]),
        "frequencies": int(len(oms)),
        "photon_columns": n_ph,
        "dark_columns": n_dark,
        "packet_mass": float(np.sum(np.abs(phi_at(oms, prot, nrm)) ** 2 * wts)),
        "sqrt_weight_once_maxabs": weight_check,
        "same_initial_columns_for_both_drives": True,
        "window": {"t0": npre * h, "steps": nstep, "h": h, "a_star": 0.8},
        "layers": layers,
        "driven_minus_undriven_channel_grams": groups,
        "driven_minus_undriven_sum_of_channel_grams": {"normal": total_n, "anomalous": total_a},
        "electronic_even_measure": {
            "normal": elec["electronic_ccr_normal_maxabs"],
            "anomalous": elec["electronic_ccr_anomalous_maxabs"],
            "dark": elec["n_dark"],
            "nodes": elec["n_nodes"],
        },
        "gamma12_residual_not_recomputed": True,
        "production_G1_pass": False,
        "provenance_sha256": provenance([
            __file__,
            Path(__file__).with_name("q8_m0_p1_h2_finite_kcv_input_close_20261005.py"),
            Path(__file__).with_name("q8_m0_continuum_columns_20261004.py"),
        ]),
    }
    write_new_json(OUT, report)
    for name, rec in layers.items():
        print(name, rec["normal"]["maxabs"], rec["anomalous"]["maxabs"])
    print("channels", {k: v["columns"] for k, v in groups.items()})
    print("weight", weight_check, "mass", report["packet_mass"], "freq", len(oms))
    print("electronic", report["electronic_even_measure"])


if __name__ == "__main__":
    main()
