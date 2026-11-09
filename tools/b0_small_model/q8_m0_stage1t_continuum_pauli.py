#!/usr/bin/env python3
"""Bounded IR/UV audit of Pauli depletion in the approved p-h photon model.

The photon continuum is represented by positive, nonuniform quadrature nodes.
Both free-space ports and both in-plane polarizations are retained.  This is a
two-k numerical qualification only: no physical area S is selected and no
quantum-output observable is promoted to a production result.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import propagate_cjj  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_mode_io import eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_phgauss import propagate_phi  # noqa: E402
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1s_continuum import (  # noqa: E402
    compare, field_from_kernel, photon_retarded_kernel,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1t_continuum_pauli_20260923"
MU = -0.009010620372198276
NTPC = 32
K_FRAC = np.array([[0.0, 0.0], [0.17, 0.41]])


def gauss_segment(lo, hi, n):
    x, w = leggauss(int(n))
    return lo + 0.5 * (hi - lo) * (x + 1.0), 0.5 * (hi - lo) * w


def continuum_waveforms(omega, weights, times, e_par, e_perp=None):
    """Each quadrature node is a canonical oscillator; F has sqrt(domega)."""
    om = np.asarray(omega, dtype=float)
    wt = np.asarray(weights, dtype=float)
    if np.any(om <= 0) or np.any(wt <= 0):
        raise ValueError("all photon frequencies and weights must be positive")
    phase = np.exp(1j * np.asarray(times)[:, None] * om[None, :])
    phase *= np.sqrt(wt / om)[None, :]
    waves = []
    polarizations = (e_par,) if e_perp is None else (e_par, e_perp)
    for pol in polarizations:
        for j in range(len(om)):
            f = phase[:, j, None] * np.asarray(pol, dtype=float)[None, :]
            waves.extend((f, f))  # distinct left and right vacuum ports
    return waves


def one_case(label, omega, weights, x, w_k, md, times, dt, eta, e_par, e_perp):
    t0 = time.perf_counter()
    # A quadrature must first represent Maxwell's retarded kernel on this
    # pulse.  A symplectic finite-oscillator calculation by itself cannot
    # qualify an under-resolved continuum.
    lag = dt * np.arange(len(times))
    gamma = 2.0 * np.pi * eta * eta
    k = photon_retarded_kernel(lag, omega, weights, gamma)
    prescribed = np.sin(np.pi * times / times[-1]) ** 2
    current = prescribed[:, None] * e_perp[None, :]
    target = field_from_kernel(current,
                               np.r_[0.0, np.full(len(times) - 1, gamma)], dt)
    from_photons = field_from_kernel(current, k, dt)
    kernel_check = compare(from_photons, target, current, dt, gamma)
    waves = continuum_waveforms(omega, weights, times, e_par, e_perp)
    ev = propagate_phi(x, w_k, md, waves, dt, eta, save_pauli=True)
    c_p = float(np.max(ev["Ncell_times_eps_Pauli_t"]))
    c_c = float(np.max(ev["Ncell_times_C_minus_I_t"]))
    kernel_qualified = bool(kernel_check["A_rel_max"] < 0.01 and
                            kernel_check["dA_rel_max"] < 0.01)
    return {
        "label": label,
        "n_frequency": int(len(omega)),
        "n_photon_oscillator": len(waves),
        "omega_min": float(omega[0]),
        "omega_max": float(omega[-1]),
        "both_inplane_polarizations": True,
        "both_ports": True,
        "C_P_max_t": c_p,
        "C_C_max_t": c_c,
        "N_B_max_t": float(np.max(ev["N_B_max_t"])),
        "kernel_check": kernel_check,
        "kernel_qualified_for_this_pulse": kernel_qualified,
        "Pauli_area_qualified": False,
        "Pauli_area_reason": "photon quadrature and time-step convergence required",
        "symplectic_maxabs": ev["symplectic_maxabs"],
        "symplectic_scaled_by_maxPhi_squared": float(
            ev["symplectic_maxabs"] /
            max(1.0, float(np.max(np.abs(ev["Phi"]))) ** 2)
        ),
        "hamiltonian_K_maxabs": ev["hamiltonian_K_maxabs"],
        "wall_seconds": time.perf_counter() - t0,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    e_par, e_perp = parallel_perp_basis(geo)
    rec = propagate_cjj(
        geo, R0, prot, nt_per_cycle=NTPC, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=K_FRAC, save_full_x=True, save_mean_D=True,
    )
    x, w_k, md = rec["X_full"], rec["w_k"], rec["mean_D_t"]
    times, dt = rec["times"], float(rec["dt"])
    eta = float(np.sqrt(eta2_ncell(geo)))
    omega0 = float(prot.omega0)
    rows = []
    for n in (8, 16, 32):
        omega, weights = gauss_segment(0.0, 8.0 * omega0, n)
        row = one_case(f"IR_{n}_UV8", omega, weights, x, w_k, md, times, dt,
                       eta, e_par, e_perp)
        rows.append(row)
        print(f"{row['label']}: CP={row['C_P_max_t']:.6g} "
              f"CC={row['C_C_max_t']:.6g} time={row['wall_seconds']:.2f}s", flush=True)

    # Change only the high-frequency interval while preserving the 0..8w0
    # quadrature, so that any difference really comes from the UV extension.
    low, low_w = gauss_segment(0.0, 8.0 * omega0, 16)
    high, high_w = gauss_segment(8.0 * omega0, 12.0 * omega0, 8)
    row = one_case("IR16_UV12", np.r_[low, high], np.r_[low_w, high_w],
                   x, w_k, md, times, dt, eta, e_par, e_perp)
    rows.append(row)
    print(f"{row['label']}: CP={row['C_P_max_t']:.6g} "
          f"CC={row['C_C_max_t']:.6g} time={row['wall_seconds']:.2f}s", flush=True)

    payload = {
        "status": "continuum_Pauli_two_k_diagnostic_not_area_verdict",
        "theory": "P1_dressed_particle_hole_photon_canonical_Gaussian",
        "n_k": 2,
        "k_frac": K_FRAC.tolist(),
        "w_k": np.asarray(w_k).tolist(),
        "ntpc": NTPC,
        "n_t": int(len(times)),
        "N_cell": "unset_not_equal_to_nk",
        "S": "not_fixed",
        "no_omega_zero_oscillator": True,
        "results": rows,
        "legacy_two_H2_C_P": 28.020752514399163,
        "legacy_two_H2_C_C": 55.744789878664065,
        "qualification_note": "All these small photon quadratures fail the "
                              "prescribed-current Maxwell kernel or time-step "
                              "gate; their large C_P/C_C are exploratory only. "
                              "No physical minimum area is inferred.",
    }
    (OUT / "stage1t_continuum_pauli.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {OUT / 'stage1t_continuum_pauli.json'}", flush=True)


if __name__ == "__main__":
    main()
