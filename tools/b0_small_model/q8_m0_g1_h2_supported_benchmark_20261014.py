"""H2-supported finite spectrum: packet commutator, same-spectrum output, bath limit.

The production packet phi is not renormalized on the subset. Four frequencies
from the earlier fixture are reported only to show that they carry no packet
weight. Bare oscillator vacuum is not the production incoming state.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import expm

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_gsheet import gamma_self
from q8_m0_k1_packet import normalize_phi, omega_grid, phi_unnormalized
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import (
    TWO_K, WEIGHTS, advance_half_frame, m0_sample, vertices_from_frame,
)
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of
from q8_m0_s1_kernel import parallel_perp_basis

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_h2_supported_benchmark_20261014/h2_supported_benchmark.json"
OLD_FREQ = np.array([0.05, 0.2, 0.6, 1.2])
OLD_W = np.array([0.04, 0.1, 0.2, 0.2])


def err(a):
    return float(np.max(np.abs(np.asarray(a))))


def production_phi(prot):
    omega = omega_grid(prot.omega0, prot.t_pulse)
    omega = omega[omega > 0.0]
    raw = phi_unnormalized(omega, 2.0 * prot.omega0, prot.t_pulse)
    phi, nrm = normalize_phi(omega, raw)
    w = np.zeros_like(omega)
    w[0] = 0.5 * (omega[1] - omega[0])
    w[-1] = 0.5 * (omega[-1] - omega[-2])
    w[1:-1] = 0.5 * (np.diff(omega[:-1]) + np.diff(omega[1:]))
    return omega, phi, w, nrm


def gauss_packet(prot, nrm, a, b, n):
    x, w = np.polynomial.legendre.leggauss(int(n))
    nodes = 0.5 * (b - a) * x + 0.5 * (a + b)
    wt = 0.5 * (b - a) * w
    order = np.argsort(nodes)
    nodes, wt = nodes[order], wt[order]
    phi = phi_unnormalized(nodes, 2.0 * prot.omega0, prot.t_pulse) / nrm
    return nodes, wt, phi


def bath(freqs, weights, gamma):
    eta = eta_of(gamma)
    om = np.repeat(np.asarray(freqs, float), 4)
    wt = np.repeat(np.asarray(weights, float), 4)
    g = eta * np.sqrt(wt / om)
    nph = len(om)
    bmat = np.zeros((2 * nph, 2 * nph))
    hfield = np.zeros((2, 2 * nph))
    for j, omega in enumerate(om):
        bmat[2 * j:2 * j + 2, 2 * j:2 * j + 2] = omega * np.array([[0.0, 1.0], [-1.0, 0.0]])
        hfield[j % 2, 2 * j] = np.sqrt(2.0) * g[j]
    jp = np.kron(np.eye(nph), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    return bmat, -jp @ hfield.T, hfield


def packet_rows(phi, weights):
    """L-port coefficients. Index 4*fi+pol is L; 4*fi+2+pol is R. Not renormalized."""
    n = len(phi)
    rows = np.zeros((2, 4 * n), complex)
    coeff = np.sqrt(weights) * phi
    for fi in range(n):
        rows[0, 4 * fi] = coeff[fi]
        rows[1, 4 * fi + 1] = coeff[fi]
    return rows


def amplitudes(y):
    q, p = y[0::2], y[1::2]
    return (q + 1j * p) / np.sqrt(2.0)


def vacuum_photons(nph):
    y0 = np.zeros((2 * nph, nph), complex)
    for j in range(nph):
        y0[2 * j, j] = 1.0 / np.sqrt(2.0)
        y0[2 * j + 1, j] = -1j / np.sqrt(2.0)
    return y0


def noise_versus_kernel(bmat, c, hfield, tau):
    rot = expm(bmat * float(tau))
    y0 = vacuum_photons(bmat.shape[0] // 2)
    free_t = hfield @ rot @ y0
    free_0 = hfield @ y0
    comm = free_t @ free_0.conj().T - free_t.conj() @ free_0.T
    kernel = hfield @ rot @ c
    return err(1j * comm - kernel)


def chi_matrix(omega, bmat, c, hfield, eps):
    dim = bmat.shape[0]
    resolvent = np.linalg.solve(-1j * (omega + 1j * eps) * np.eye(dim) - bmat, c)
    return hfield @ resolvent


def constant_drive_ratio(freqs, weights, gamma, t_end, nstep):
    bmat, c, hfield = bath(freqs, weights, gamma)
    h = float(t_end) / int(nstep)
    dph = np.linalg.inv(np.eye(bmat.shape[0]) - 0.5 * h * bmat)
    step = dph @ (np.eye(bmat.shape[0]) + 0.5 * h * bmat)
    kick = h * dph @ c
    y = np.zeros(bmat.shape[0])
    drive = np.array([1.0, 0.0])
    for _ in range(int(nstep)):
        y = step @ y + kick @ drive
    predicted = gamma * t_end * drive
    return err((hfield @ y - predicted) / max(err(predicted), 1e-30))


def quadratures(u):
    m = np.empty((2, 2 * u.shape[1]))
    m[:, ::2] = np.sqrt(2.0) * np.real(u)
    m[:, 1::2] = np.sqrt(2.0) * np.imag(u)
    return m


def run_pair(sample, mats, vertices, h, rows, k_of):
    bmat, c, hfield = mats
    nel = sample["n_el"]
    nph = bmat.shape[0] // 2
    ne, np_ = 2 * nel, 2 * nph
    nd = nel + nph
    je = np.kron(np.eye(nel), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    jt = np.kron(np.eye(nd), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    dph = np.linalg.inv(np.eye(np_) - 0.5 * h * bmat)
    e = dph @ (np.eye(np_) + 0.5 * h * bmat)
    gh = h * dph @ c
    fmid = hfield @ dph
    self_mid = 0.5 * h * fmid @ c
    z0 = np.zeros((2 * nd, nd), complex)
    for j in range(nd):
        z0[2 * j, j] = 1.0 / np.sqrt(2.0)
        z0[2 * j + 1, j] = -1j / np.sqrt(2.0)
    full = z0.copy()
    x = z0[:ne].copy()
    y0 = z0[ne:].copy()
    powers = [np.eye(np_)]
    currents = []
    max_state = max_packet = max_h = 0.0
    acc_ex = np.zeros((2, nd), complex)
    acc_mem = np.zeros_like(acc_ex)
    pol = np.vstack(parallel_perp_basis(build_geometry()))
    for step, (u, d) in enumerate(vertices):
        m = quadratures(u)
        l = -je @ m.T
        dreal = np.real(np.asarray(d, float))
        kmat = np.block([[np.zeros((ne, ne)), l @ hfield],
                         [c @ m, bmat - c @ dreal @ hfield]])
        max_h = max(max_h, err(jt @ kmat + kmat.T @ jt))
        full_new = np.linalg.solve(np.eye(2 * nd) - 0.5 * h * kmat,
                                   (np.eye(2 * nd) + 0.5 * h * kmat) @ full)
        power = powers[-1] @ e
        powers.append(power)
        af = fmid @ powers[step] @ y0
        memory = np.zeros_like(af)
        for j, ii in enumerate(currents):
            memory += (fmid @ powers[step - 1 - j] @ gh) @ ii
        amat = np.linalg.solve(np.eye(2) - self_mid @ (0.5 * h * m @ l - dreal),
                               af + memory + self_mid @ m @ x)
        imat = m @ x + (0.5 * h * m @ l - dreal) @ amat
        x = x + h * l @ amat
        currents.append(imat)
        y = powers[step + 1] @ y0
        for j, ii in enumerate(currents):
            y += powers[step - j] @ gh @ ii
        max_state = max(max_state, err(np.vstack((x, y)) - full_new))
        packet_ex = rows @ amplitudes(full_new[ne:])
        packet_mem = rows @ amplitudes(y)
        max_packet = max(max_packet, err(packet_ex - packet_mem))
        acc_ex += h * k_of(step) * (pol @ (m @ (0.5 * (full[:ne] + full_new[:ne]))
                                            - dreal @ hfield @ (0.5 * (full[ne:] + full_new[ne:]))))
        acc_mem += h * k_of(step) * (pol @ imat)
        full = full_new
    free = rows @ amplitudes(powers[-1] @ y0)
    driven_packet = rows @ amplitudes(y) - free
    return {
        "state": max_state,
        "packet_explicit_vs_memory": max_packet,
        "current_kernel_explicit_vs_memory": err(acc_ex - acc_mem),
        "driven_packet_minus_production_kernel": err(driven_packet - acc_mem),
        "driven_packet_maxabs": err(driven_packet),
        "production_kernel_integral_maxabs": err(acc_mem),
        "hamiltonian": max_h,
    }


def gap_vertices(sample, h, nstep):
    for step in range(nstep):
        u = sample["u"] * np.exp(1j * sample["gap"] * (step + 0.5) * h)
        yield u, sample["D"]


def frame_vertices(sample, geo, prot, t0, h, nstep):
    frame = sample["frame"].copy()
    npre = int(round(t0 / h))
    for step in range(npre + nstep):
        t = step * h
        frame = advance_half_frame(geo, sample["kc"], prot, sample["neel"], frame, t + 0.25 * h, h)
        if step >= npre:
            yield vertices_from_frame(geo, sample["kc"], prot, sample["neel"], t + 0.5 * h,
                                     frame, sample["weights"])
        frame = advance_half_frame(geo, sample["kc"], prot, sample["neel"], frame, t + 0.75 * h, h)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma = float(gamma_self(geo))
    fine_w, fine_phi, fine_wt, nrm = production_phi(prot)
    nodes, wt, phi = gauss_packet(prot, nrm, 0.08, 0.12, 24)
    mass = float(np.sum(np.abs(phi) ** 2 * wt))
    outside = (fine_w < 0.08) | (fine_w > 0.12)
    tail = float(np.sum(np.abs(fine_phi[outside]) ** 2 * fine_wt[outside]))
    old_phi = phi_unnormalized(OLD_FREQ, 2.0 * prot.omega0, prot.t_pulse) / nrm
    old_mass = float(np.sum(np.abs(old_phi) ** 2 * OLD_W))
    rows = packet_rows(phi, wt)
    gram = rows @ rows.conj().T
    mats = bath(nodes, wt, gamma)
    noise = [noise_versus_kernel(*mats, tau) for tau in (0.2, 1.0, 4.0)]
    sample = m0_sample(geo, TWO_K, WEIGHTS, +1)
    h = 0.5
    kcoef = 1j * eta_of(gamma) * wt * phi / np.sqrt(nodes)

    def k_of_factory(times):
        def k_of(step):
            phase = np.exp(1j * nodes * times[step])
            return np.sum(kcoef * phase)
        return k_of

    undriven_times = (np.arange(8) + 0.5) * h
    undriven = run_pair(sample, mats, list(gap_vertices(sample, h, 8)), h, rows,
                        k_of_factory(undriven_times))
    h_drive = float(prot.period) / 32.0
    t0 = 0.5 * float(prot.t_pulse)
    npre = int(round(t0 / h_drive))
    drive_times = t0 + (np.arange(8) + 0.5) * h_drive
    driven = run_pair(sample, mats, list(frame_vertices(
        sample, geo, prot, npre * h_drive, h_drive, 8)), h_drive, rows, k_of_factory(drive_times))
    target = 1j * gamma / 0.1 * np.eye(2)
    on_band = chi_matrix(0.1, *mats, 1e-4)
    scans = []
    for spacing, cutoff in ((0.05, 1.0), (0.025, 1.0), (0.05, 2.0), (0.025, 4.0)):
        grid = np.arange(spacing, cutoff + 0.5 * spacing, spacing)
        wts = np.full(grid.shape, spacing)
        bmat, c, hfield = bath(grid, wts, gamma)
        probe = 0.5 * cutoff
        resolved = chi_matrix(probe, bmat, c, hfield, 0.5 * spacing)
        tiny = chi_matrix(probe, bmat, c, hfield, 1e-8)
        markov = 1j * gamma / probe * np.eye(2)
        scans.append({
            "spacing": spacing, "cutoff": cutoff, "nodes": int(grid.size),
            "constant_drive_relative": constant_drive_ratio(grid, wts, gamma, 20.0, 80),
            "chi_eps_half_spacing_relative": err(resolved - markov) / err(markov),
            "chi_eps_1e-8_relative": err(tiny - markov) / err(markov),
            "noise_kernel": noise_versus_kernel(bmat, c, hfield, 1.0),
        })
    report = {
        "packet_definition": "production phi_unnormalized/normalizer; subset weights are Gauss on [0.08,0.12]; phi is not rescaled",
        "packet_mass": mass,
        "tail_mass_outside_0.08_0.12": tail,
        "old_four_frequency_mass": old_mass,
        "free_packet_gram_minus_mass_identity": err(gram - mass * np.eye(2)),
        "free_noise_matches_bath_kernel": noise,
        "undriven_bare_vacuum": undriven,
        "short_P1_bare_vacuum": driven,
        "packet_band_chi_versus_markov_at_0.1": err(on_band - target) / err(target),
        "broadband_scans": scans,
        "initial_state": "Bare canonical vacuum of this finite bath. Not the production stationary incoming state.",
        "output_map_verified": "Same packet row on L-port oscillator amplitudes, explicit Cayley versus its memory elimination. Production K_det integral is compared and is a different map when the bath is not the continuum.",
        "gamma12_residual_0.007439_localized": False,
        "production_G1_pass": False,
        "provenance_sha256": provenance([
            __file__,
            Path(__file__).with_name("q8_m0_k1_packet.py"),
            Path(__file__).with_name("q8_m0_p1_h2_finite_kcv_input_close_20261005.py"),
        ]),
    }
    write_new_json(OUT, report)
    print("mass", mass, "tail", tail, "old", old_mass)
    print("gram", report["free_packet_gram_minus_mass_identity"])
    print("undriven", undriven)
    print("driven", driven)
    print("scans", scans)


if __name__ == "__main__":
    main()
