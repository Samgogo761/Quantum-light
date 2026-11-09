"""Detector-kernel weight outside the 9-period output window.

K is built from the frozen H2 packet only. The observation length is not an
argument of that kernel. No electron propagation and no adjoint rebuild.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
from q8_m0_k1_packet import phi_unnormalized
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import (
    cluster_rank_width, cluster_width, phi_at,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/kdet_window_weight.json"
POLE = 0.0927933583
EDGES = (0.0, 8.0, 9.0, 9.5, 10.0, 11.0, 20.0, 40.0)


def trap(values, times):
    if times.size < 2:
        return 0.0
    return complex(np.trapezoid(values, times))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _rows, cache, t0, _meta = load_rows()
    if abs(float(t0)) > 0:
        raise AssertionError(t0)
    gamma = float(np.asarray(cache["gamma"]))
    prot = p1_a2_protocol(0.8)
    period = float(prot.period)
    samples = 64
    n = int(EDGES[-1] * samples)
    h = period / samples
    times = (np.arange(n) + 0.5) * h
    kernel, normalizer = kernel_at_times(gamma, prot, times)
    cycles = times / period
    mass = np.abs(kernel) ** 2
    total_l2 = float(np.trapezoid(mass, times))
    total_l1 = float(np.trapezoid(np.abs(kernel), times))
    bins = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        mask = (cycles >= lo) & (cycles < hi)
        tt, kk, mm = times[mask], kernel[mask], mass[mask]
        row = {
            "cycles": [lo, hi],
            "l1": abs(trap(np.abs(kk), tt)),
            "l2_mass": abs(trap(mm, tt)),
        }
        for name, omega in (("h2", 2.0 * float(prot.omega0)), ("pole", POLE)):
            row[f"harmonic_{name}"] = abs(trap(kk * np.exp(-1j * omega * tt), tt))
        bins.append(row)
    outside = cycles >= 9.0
    gaps = np.asarray(cache["gaps"], float)
    nearest = float(gaps[np.argmin(np.abs(gaps - POLE))])
    rank, widths = cluster_rank_width(gaps, cache["u0"], gamma, nearest)
    phi_kernel = phi_unnormalized(
        np.array([POLE, 0.10]), 2.0 * float(prot.omega0), float(prot.t_pulse)) / normalizer
    phi_u0 = np.asarray(phi_at(np.array([POLE, 0.10]), prot, normalizer), complex)
    write_new_json(OUT, {
        "kernel_definition": "K(t)=i eta ∫ phi(omega)/sqrt(omega) exp(+i omega t) d omega",
        "phi_uses_only_the_8_cycle_pulse": True,
        "observation_length_is_not_an_argument_of_K": True,
        "gamma": gamma,
        "period": period,
        "pulse_cycles": int(prot.n_cycles),
        "t_pulse": float(prot.t_pulse),
        "nine_periods": 9.0 * period,
        "two_pi_over_nine_periods": float(2.0 * np.pi / (9.0 * period)),
        "total_l1": total_l1,
        "total_l2_mass": total_l2,
        "outside_9_cycles_l1_fraction": abs(trap(np.abs(kernel[outside]), times[outside])) / total_l1,
        "outside_9_cycles_l2_fraction": abs(trap(mass[outside], times[outside])) / total_l2,
        "bins": bins,
        "pole_requested": POLE,
        "nearest_gap": nearest,
        "quadrature_width_formula": "gamma * sigma_min^2 / |gap|, static vertex SVD, sinh-panel scale",
        "not_a_measured_post_drive_decay_rate": True,
        "cluster_width_narrowest": float(cluster_width(gaps, cache["u0"], gamma, nearest)),
        "cluster_rank": int(rank),
        "cluster_widths": widths.tolist(),
        "phi_kernel_minus_phi_u0": [abs(complex(a - b)) for a, b in zip(phi_kernel, phi_u0)],
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })
    print("outside L2 fraction", abs(trap(mass[outside], times[outside])) / total_l2, flush=True)
    print("outside L1 fraction", abs(trap(np.abs(kernel[outside]), times[outside])) / total_l1, flush=True)
    print("width", float(cluster_width(gaps, cache["u0"], gamma, nearest)), "rank", int(rank), flush=True)


if __name__ == "__main__":
    main()
