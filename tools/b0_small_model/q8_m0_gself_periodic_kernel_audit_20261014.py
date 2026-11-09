"""Independent prescribed-current audit; does not alter production equations.

T_p = gamma = 1.  The positive DFT photon bins yield a periodic sawtooth
retarded kernel, not the flat causal kernel at finite T_obs.  All current
convolutions below are analytic or adaptive quadrature, avoiding the old
left-Riemann time discretization as a confounder.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import sici

from q8_m0_gself_emerge import emerge_A_from_I
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_continuum_periodic_audit_20261014"


def maxwell(t):
    return t / 2 - np.sin(2 * np.pi * t) / (4 * np.pi)


def periodic_limit(t, ratio):
    lag_moment = t * t / 4 + (np.cos(2 * np.pi * t) - 1) / (8 * np.pi**2)
    return maxwell(t) - 2 * lag_moment / ratio


def dft_response(t, ratio, cycles_cut):
    omega = 2 * np.pi * np.arange(1, int(ratio * cycles_cut) + 1) / ratio
    dw = 2 * np.pi / ratio
    w0 = 2 * np.pi
    om = omega[:, None]
    t = np.asarray(t)[None, :]
    conv = 0.5 * (1 - np.cos(om * t)) / om
    regular = np.abs(omega - w0) > 1e-10
    conv[regular] -= 0.5 * om[regular] * (
        np.cos(w0 * t) - np.cos(om[regular] * t)
    ) / (om[regular] ** 2 - w0**2)
    conv[~regular] -= 0.25 * t * np.sin(w0 * t)
    # gamma=1, 4 eta^2 = 2/pi, two ports included exactly once.
    return 2 / np.pi * dw * np.sum(conv / om, axis=0)


def continuum_response(t, omega_max):
    return quad(
        lambda s: 2 / np.pi * sici(omega_max * (t - s))[0] * np.sin(np.pi * s) ** 2,
        0.0, float(t), epsabs=1e-11, epsrel=1e-11, limit=500,
    )[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT / "periodic_kernel_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use --output with a new path: {args.output}")
    t = np.linspace(0, 1, 257)
    target = maxwell(t)
    rows = []
    for ratio in (1, 2, 4, 8, 16):
        a = dft_response(t, ratio, 32)
        predicted = periodic_limit(t, ratio)
        rows.append({
            "Tobs_over_Tp": ratio,
            "n_bins": ratio * 32,
            "error_vs_Maxwell_rel_max": float(np.max(np.abs(a - target)) / 0.5),
            "endpoint_relative_error": float(abs(a[-1] - target[-1]) / 0.5),
            "predicted_endpoint_relative_error": 1 / ratio,
            "error_vs_periodic_limit_rel_max": float(np.max(np.abs(a - predicted)) / 0.5),
        })
    fixed_ratio = []
    for cutoff in (8, 16, 32, 64, 128):
        a = dft_response(t, 2, cutoff)
        fixed_ratio.append({
            "cycles_cut": cutoff,
            "error_vs_Maxwell_rel_max": float(np.max(np.abs(a - target)) / 0.5),
            "error_vs_periodic_limit_rel_max": float(np.max(np.abs(a - periodic_limit(t, 2))) / 0.5),
        })
    continuum = []
    tt = np.linspace(0, 1, 33)
    for cutoff in (4, 8, 16, 32):
        a = np.array([continuum_response(x, 2 * np.pi * cutoff) for x in tt])
        continuum.append({
            "cycles_cut": cutoff,
            "continuous_frequency_error_vs_Maxwell_rel_max": float(np.max(np.abs(a - maxwell(tt))) / 0.5),
        })
    oscillator_checks = []
    for ratio in (1, 2):
        dt = 1 / 64
        times = np.arange(64 * ratio + 1) * dt
        current = np.zeros((times.size, 2))
        current[:65, 0] = np.sin(np.pi * times[:65]) ** 2
        dw = 2 * np.pi / ratio
        omega = dw * np.arange(1, 32 * ratio + 1)
        f = np.sqrt(dw / omega[:, None]) * np.exp(1j * omega[:, None] * times)
        packets = []
        for row in f:
            vector = np.column_stack((row, np.zeros_like(row)))
            packets.extend((vector, vector))
        a = emerge_A_from_I(packets, current, dt, 1 / np.sqrt(2 * np.pi), 1 / np.sqrt(2 * np.pi))
        kernel = 2 / np.pi * dw * np.sum(np.sin(omega[:, None] * times) / omega[:, None], axis=0)
        direct = dt * np.convolve(kernel, current[:, 0])[:times.size]
        oscillator_checks.append({
            "Tobs_over_Tp": ratio,
            "old_oscillator_vs_independent_discrete_kernel_maxabs": float(np.max(np.abs(a[:, 0] - direct))),
            "endpoint_error_vs_left_Riemann_Maxwell": float(abs(a[64, 0] - 0.5) / 0.5),
        })
    report = {
        "provenance_sha256": provenance([__file__,
            Path(__file__).with_name("q8_m0_gself_emerge.py")]),
        "kind": "prescribed_current_only_not_quantum_input_output_validation",
        "no_production_changes": True,
        "units": "Tp=gamma=1; I(t)=sin(pi*t)^2 on [0,1]",
        "finite_DFT_kernel": "K_J(tau)=(2*gamma/pi)*sum_j sin(2*pi*j*tau/Tobs)/j",
        "infinite_DFT_kernel_inside_period": "gamma*(1-2*tau/Tobs), 0<tau<Tobs",
        "finite_DFT_response": "gamma*integral I - (2*gamma/Tobs)*integral(t-s)I(s)",
        "continuum_cutoff_kernel": "(2*gamma/pi)*Si(omega_max*tau)",
        "window_scan": rows,
        "cutoff_scan_at_Tobs_2Tp": fixed_ratio,
        "true_continuous_frequency_cutoff_scan": continuum,
        "original_oscillator_implementation_checks": oscillator_checks,
        "initial_condition_scope": "Zero oscillator coordinates here mean zero source-generated self-field; no incoming free field is included or tested.",
        "quantum_noise_commutator_tested": False,
        "self_consistent_electronic_feedback_tested": False,
        "production_W_in_tested": False,
    }
    write_new_json(args.output, report)
    print(json.dumps({"window_scan": rows, "quantum_noise_commutator_tested": False}))


if __name__ == "__main__":
    main()
