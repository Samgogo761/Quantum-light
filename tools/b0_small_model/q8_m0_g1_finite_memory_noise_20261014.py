"""Matched finite-spectrum quantum elimination fixture, NOT production vacuum.

Real M0 two-k electronic vertices, two photon ports and two polarizations.
Compare a full canonical Cayley evolution with elimination of the SAME
discrete photon block into its exact discrete memory kernel and free noise.
No continuum/Markov limit or asymptotic detector claim is made.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import json
import numpy as np
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import m0_sample, TWO_K, WEIGHTS
from q8_m0_model import build_geometry
from q8_m0_gsheet import gamma_self

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_io_crosscheck_review_20261014/finite_memory_noise.json"


def err(a):
    return float(np.max(np.abs(a)))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    geo = build_geometry()
    sample = m0_sample(geo, TWO_K, WEIGHTS, +1)
    gamma = float(gamma_self(geo))
    eta = np.sqrt(gamma / (2 * np.pi))
    freq = np.array([0.05, 0.2, 0.6, 1.2])
    weights = np.array([0.04, 0.1, 0.2, 0.2])
    om = np.repeat(freq, 4)  # Lx Ly Rx Ry, same measure on each port
    g = eta * np.sqrt(np.repeat(weights, 4) / om)
    nel, nph = sample["n_el"], len(om)
    ne, np_ = 2 * nel, 2 * nph
    je = np.kron(np.eye(nel), [[0., 1.], [-1., 0.]])
    jp = np.kron(np.eye(nph), [[0., 1.], [-1., 0.]])
    jt = np.kron(np.eye(nel + nph), [[0., 1.], [-1., 0.]])
    b = np.zeros((np_, np_))
    hfield = np.zeros((2, np_))
    for j, omega in enumerate(om):
        b[2*j:2*j+2, 2*j:2*j+2] = omega * np.array([[0., 1.], [-1., 0.]])
        hfield[j % 2, 2*j] = np.sqrt(2) * g[j]
    c = -jp @ hfield.T
    h = 0.5
    nstep = 64
    dph = np.linalg.inv(np.eye(np_) - 0.5 * h * b)
    e = dph @ (np.eye(np_) + 0.5 * h * b)
    gh = h * dph @ c
    fmid = hfield @ dph
    self_mid = 0.5 * h * fmid @ c
    nd = nel + nph
    z0 = np.zeros((2*nd, nd), complex)
    for j in range(nd):
        z0[2*j, j] = 1 / np.sqrt(2)
        z0[2*j+1, j] = -1j / np.sqrt(2)
    # The identical bare canonical coordinates in both arms are a test initial
    # condition, not a replacement for production stationary incoming vacuum.
    full = z0.copy()
    x = z0[:ne].copy()
    y0 = z0[ne:].copy()
    powers = [np.linalg.matrix_power(e, j) for j in range(nstep + 1)]
    past_kernels = [fmid @ powers[j] @ gh for j in range(nstep)]
    currents = []
    max_state = max_field = max_current = max_hamiltonian = 0.
    for step in range(nstep):
        tmid = (step + 0.5) * h
        # No pump: actual free electronic interaction-picture vertex.
        u = sample["u"] * np.exp(1j * sample["gap"] * tmid)
        m = np.empty((2, ne))
        m[:, ::2] = np.sqrt(2) * u.real
        m[:, 1::2] = np.sqrt(2) * u.imag
        l = -je @ m.T
        d = sample["D"]
        k = np.block([[np.zeros((ne, ne)), l @ hfield],
                      [c @ m, b - c @ d @ hfield]])
        max_hamiltonian = max(max_hamiltonian, err(k @ jt + jt @ k.T))
        full_new = np.linalg.solve(np.eye(2*nd) - 0.5*h*k,
                                   (np.eye(2*nd) + 0.5*h*k) @ full)
        full_mid = 0.5 * (full + full_new)
        af_mid = fmid @ powers[step] @ y0
        memory = np.zeros_like(af_mid)
        for j, ii in enumerate(currents):
            memory += past_kernels[step-1-j] @ ii
        amat = np.linalg.solve(np.eye(2) - self_mid @ (0.5*h*m@l - d),
                               af_mid + memory + self_mid @ m @ x)
        imat = m @ x + (0.5*h*m@l - d) @ amat
        x = x + h * l @ amat
        currents.append(imat)
        y = powers[step+1] @ y0
        for j, ii in enumerate(currents):
            y += powers[step-j] @ gh @ ii
        reduced = np.vstack((x, y))
        max_state = max(max_state, err(reduced - full_new))
        max_field = max(max_field, err(amat - hfield @ full_mid[ne:]))
        max_current = max(max_current, err(imat - (m @ full_mid[:ne] - d @ hfield @ full_mid[ne:])))
        full = full_new
    # End-time finite photon modes only; not the H2 far-field detector.
    out = np.zeros((2, np_), complex)
    for fi in range(len(freq)):
        for pol in range(2):
            j = 4*fi + 2 + pol  # R port in this oscillator fixture
            out[pol, 2*j] = 1 / np.sqrt(2 * len(freq))
            out[pol, 2*j+1] = 1j / np.sqrt(2 * len(freq))
    uu, vv = out @ y, out @ y.conj()
    rn = err(uu @ uu.conj().T - vv @ vv.conj().T - np.eye(2))
    ra = err(uu @ vv.T - vv @ uu.T)
    # i[Af(t),Af(0)] = H exp(Bt) C with precisely the same photon measure.
    noise_errors = []
    for tau in (0.1, 1., 3., 12.):
        rotation = np.zeros_like(b)
        for j, omega in enumerate(om):
            co, si = np.cos(omega * tau), np.sin(omega * tau)
            rotation[2*j:2*j+2, 2*j:2*j+2] = [[co, si], [-si, co]]
        free_t = hfield @ rotation @ y0
        free_0 = hfield @ y0
        comm = free_t @ free_0.conj().T - free_t.conj() @ free_0.T
        kernel = hfield @ rotation @ c
        noise_errors.append(err(1j * comm - kernel))
    missing_noise_ccr = err(uu[:, :nel] @ uu[:, :nel].conj().T
                             - vv[:, :nel] @ vv[:, :nel].conj().T - np.eye(2))
    report = {
        "status": "matched_finite_spectrum_memory_and_noise_fixture",
        "all_fixture_checks_pass": max(max_state, max_field, max_current, rn, ra,
                                        max(noise_errors), max_hamiltonian) < 2e-11,
        "electron_model": "existing real M0 two-k sample; no pump; exact gap phases",
        "electronic_modes": nel, "photon_modes": nph,
        "two_ports_and_two_polarizations": True,
        "frequency_nodes": freq.tolist(), "frequency_weights": weights.tolist(),
        "gamma_unchanged": gamma, "steps": nstep, "h": h,
        "full_vs_eliminated_state_maxabs": max_state,
        "full_vs_eliminated_midpoint_A_maxabs": max_field,
        "full_vs_eliminated_midpoint_current_maxabs": max_current,
        "generator_hamiltonian_defect": max_hamiltonian,
        "finite_photon_output_normal_ccr": rn,
        "finite_photon_output_anomalous_ccr": ra,
        "retarded_kernel_vs_free_noise_commutator_maxabs": max(noise_errors),
        "omitting_photon_input_columns_normal_ccr": missing_noise_ccr,
        "initial_state": "Identical bare canonical vacuum fixture in both arms; NOT production stationary in-vacuum.",
        "limits_not_tested": ["frequency spacing -> 0", "UV -> infinity", "IR limit",
                              "stationary incoming map", "finite-time H2 far-field output", "driven P1 production"],
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__,
            Path(__file__).with_name("q8_m0_p1_h2_finite_kcv_caller_fix_20261005.py"),
            Path(__file__).with_name("q8_m0_model.py")]),
    }
    assert report["all_fixture_checks_pass"], report
    write_new_json(OUT, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
