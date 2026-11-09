"""Duhamel identity for the current finite bath, and the packet commutator.

The oscillator equation implemented by bath() is
alpha_dot = -i*omega*alpha + i*g*I_cartesian,
with alpha=(q+i*p)/sqrt(2) and g=eta*sqrt(weight/omega).
The previous 5.27e-4 versus 0.169 comparison did not use this phase or this current component.
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
from q8_m0_g1_h2_supported_benchmark_20261014 import (
    bath, gauss_packet, packet_rows, production_phi, quadratures,
)
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import TWO_K, WEIGHTS, m0_sample
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of
from q8_m0_s1_kernel import parallel_perp_basis

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_h2_duhamel_20261014/duhamel_identity.json"


def err(a):
    return float(np.max(np.abs(np.asarray(a))))


def generator(u, d, bmat, c, hfield, nel):
    ne = 2 * nel
    m = quadratures(u)
    je = np.kron(np.eye(nel), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    l = -je @ m.T
    dreal = np.real(np.asarray(d, float))
    return np.block([
        [np.zeros((ne, ne)), l @ hfield],
        [c @ m, bmat - c @ dreal @ hfield],
    ]), m, dreal


def propagate(vertices, h, bmat, c, hfield, nel, samples):
    ne = 2 * nel
    n = ne + bmat.shape[0]
    phi = np.eye(n)
    times = []
    currents = []
    nodes = np.linspace(0.0, 1.0, int(samples))
    weights = np.full(nodes.size, 1.0 / (nodes.size - 1))
    weights[0] *= 0.5
    weights[-1] *= 0.5
    for step, (u, d) in enumerate(vertices):
        kmat, m, dreal = generator(u, d, bmat, c, hfield, nel)
        step_map = expm(h * kmat)
        for s, node in enumerate(nodes):
            partial = expm(node * h * kmat) @ phi
            x, y = partial[:ne], partial[ne:]
            current = m @ x - dreal @ (hfield @ y)
            times.append(step * h + node * h)
            currents.append(weights[s] * h * current)
        phi = step_map @ phi
    return phi, np.asarray(times), currents


def alpha_rows(phi, ne, nph):
    out = np.zeros((nph, phi.shape[1]), complex)
    for j in range(nph):
        q = phi[ne + 2 * j]
        p = phi[ne + 2 * j + 1]
        out[j] = (q + 1j * p) / np.sqrt(2.0)
    return out


def alpha0_rows(n, ne, nph):
    out = np.zeros((nph, n), complex)
    for j in range(nph):
        out[j, ne + 2 * j] = 1.0 / np.sqrt(2.0)
        out[j, ne + 2 * j + 1] = 1j / np.sqrt(2.0)
    return out


def duhamel(phi, times, currents, om, g, ne):
    nph = len(om)
    alpha_f = alpha_rows(phi, ne, nph)
    alpha_0 = alpha0_rows(phi.shape[0], ne, nph)
    tf = float(times[-1])
    mode_residual = 0.0
    for j, omega in enumerate(om):
        phase = np.exp(1j * omega * times)
        i_comp = np.zeros(phi.shape[1], complex)
        for s, cur in enumerate(currents):
            i_comp += phase[s] * cur[j % 2]
        lhs = np.exp(1j * omega * tf) * alpha_f[j] - alpha_0[j]
        rhs = 1j * g[j] * i_comp
        mode_residual = max(mode_residual, err(lhs - rhs))
    return mode_residual, alpha_f, alpha_0


def packet_identity(phi, times, currents, om, g, rows, ne):
    nph = len(om)
    alpha_f = alpha_rows(phi, ne, nph)
    alpha_0 = alpha0_rows(phi.shape[0], ne, nph)
    tf = float(times[-1])
    lhs = np.zeros((2, phi.shape[1]), complex)
    rhs = np.zeros_like(lhs)
    for j, omega in enumerate(om):
        c = rows[:, j]
        if not np.any(c):
            continue
        phase = np.exp(1j * omega * times)
        i_comp = np.zeros(phi.shape[1], complex)
        for s, cur in enumerate(currents):
            i_comp += phase[s] * cur[j % 2]
        lhs += c[:, None] * (np.exp(1j * omega * tf) * alpha_f[j] - alpha_0[j])
        rhs += c[:, None] * (1j * g[j] * i_comp)
    unphased = rows @ (alpha_f - (np.exp(-1j * om * tf)[:, None] * alpha_0))
    return err(lhs - rhs), err(lhs), err(rhs), err(unphased)


def old_kernel_integral(times, currents, nodes, weights, phi, eta, pol):
    kcoef = 1j * eta * weights * phi / np.sqrt(nodes)
    acc_cart = np.zeros_like(currents[0], complex)
    acc_pol = np.zeros_like(acc_cart)
    # currents were already multiplied by quadrature weights*h
    for s, cur in enumerate(currents):
        k = np.sum(kcoef * np.exp(1j * nodes * times[s]))
        acc_cart += k * cur
        acc_pol += k * (pol @ cur)
    return acc_cart, acc_pol


def bogoliubov(phi, rows, om, ne, mass):
    nph = len(om)
    tf = 0.0  # replaced by caller through phased rows
    del tf
    # rows multiply alpha; phase is already included by the caller if wanted.
    # Here rows are coefficients of alpha_j(tf), and we multiply by exp(i omega tf) outside.
    n = phi.shape[0]
    form = np.zeros((2, n), complex)
    for j, omega in enumerate(om):
        # coefficient of alpha_j in the phased packet
        c = rows[:, j] * np.exp(1j * omega * 0.0)
        # placeholder, real phase passed by modifying rows before the call
        q = phi[ne + 2 * j]
        p = phi[ne + 2 * j + 1]
        form += c[:, None] * (q + 1j * p) / np.sqrt(2.0)
    nmodes = ne // 2 + nph
    u = np.zeros((2, nmodes), complex)
    v = np.zeros_like(u)
    for mode in range(nmodes):
        cq, cp = form[:, 2 * mode], form[:, 2 * mode + 1]
        u[:, mode] = cq / np.sqrt(2.0) - 1j * cp / np.sqrt(2.0)
        v[:, mode] = cq / np.sqrt(2.0) + 1j * cp / np.sqrt(2.0)
    normal = u @ u.conj().T - v @ v.conj().T
    anomalous = u @ v.T - v @ u.T
    return {
        "normal_minus_mass_identity": err(normal - mass * np.eye(2)),
        "anomalous": err(anomalous),
    }


def phased_rows(rows, om, tf):
    out = rows.copy()
    for j, omega in enumerate(om):
        out[:, j] *= np.exp(1j * omega * tf)
    return out


def symplectic_defect(phi):
    n = phi.shape[0] // 2
    j = np.kron(np.eye(n), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    return err(phi.T @ j @ phi - j)


def one_mode_identity():
    omega, g, eta = 0.1, 0.02, None
    del eta
    bmat = omega * np.array([[0.0, 1.0], [-1.0, 0.0]])
    # p_dot += sqrt(2)*g*I, matching bath()
    c = np.array([[0.0], [np.sqrt(2.0) * g]])
    h = 1.0e-3
    n = 400
    q = p = 0.0
    acc = 0.0
    for step in range(n):
        t = (step + 0.5) * h
        current = np.cos(t)
        k = np.array([[0.0, omega], [-omega, np.sqrt(2.0) * g * 0.0]])
        # inhomogeneous Cayley with frozen I
        inhom = np.array([0.0, np.sqrt(2.0) * g * current])
        mat = np.eye(2) - 0.5 * h * bmat
        q, p = np.linalg.solve(mat, (np.eye(2) + 0.5 * h * bmat) @ np.array([q, p]) + h * inhom)
        acc += h * np.exp(1j * omega * t) * current
    alpha = (q + 1j * p) / np.sqrt(2.0)
    lhs = np.exp(1j * omega * n * h) * alpha
    rhs = 1j * g * acc
    return err(lhs - rhs)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma = float(gamma_self(geo))
    eta = eta_of(gamma)
    _w, _phi, _wt, nrm = production_phi(prot)
    nodes, weights, phi = gauss_packet(prot, nrm, 0.08, 0.12, 24)
    mass = float(np.sum(np.abs(phi) ** 2 * weights))
    rows = packet_rows(phi, weights)
    mats = bath(nodes, weights, gamma)
    bmat, c, hfield = mats
    om = np.repeat(nodes, 4)
    g = eta * np.sqrt(np.repeat(weights, 4) / om)
    sample = m0_sample(geo, TWO_K, WEIGHTS, +1)
    pol = np.vstack(parallel_perp_basis(geo))
    h = float(prot.period) / 32.0
    t0 = 0.5 * float(prot.t_pulse)
    npre = int(round(t0 / h))
    vertices = []
    frame = sample["frame"].copy()
    from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import advance_half_frame, vertices_from_frame
    for step in range(npre + 8):
        t = step * h
        frame = advance_half_frame(geo, sample["kc"], prot, sample["neel"], frame, t + 0.25 * h, h)
        if step >= npre:
            vertices.append(vertices_from_frame(
                geo, sample["kc"], prot, sample["neel"], t + 0.5 * h, frame, sample["weights"]))
        frame = advance_half_frame(geo, sample["kc"], prot, sample["neel"], frame, t + 0.75 * h, h)
    phi_map, times, currents = propagate(vertices, h, bmat, c, hfield, sample["n_el"], 17)
    mode_residual, alpha_f, alpha_0 = duhamel(phi_map, times, currents, om, g, 2 * sample["n_el"])
    packet_gap, lhs_size, rhs_size, unphased_size = packet_identity(
        phi_map, times, currents, om, g, rows, 2 * sample["n_el"])
    cart, rotated = old_kernel_integral(times, currents, nodes, weights, phi, eta, pol)
    tf = float(times[-1] - times[0] + (times[1] - times[0]))
    # times cover [t_start, t_start+T]; phase uses the elapsed time from the initial sample.
    elapsed = float(times[-1] - times[0])
    # Rebuild packet LHS size is already elapsed-consistent inside packet_identity via times[-1].
    # The initial time in propagate is 0 relative. packet_identity uses times[-1] as tf.
    # vertices start at t0, but propagate's clock starts at 0. Phases are relative to the
    # start of this window, which is the correct Duhamel origin.
    bog = bogoliubov(phi_map, phased_rows(rows, om, float(times[-1])), om, 2 * sample["n_el"], mass)
    # bogoliubov currently ignores the phase argument inside and uses rows as alpha coefficients.
    # Pass phased rows and zero extra phase:
    bog = bogoliubov(phi_map, phased_rows(rows, om, float(times[-1])), om * 0.0, 2 * sample["n_el"], mass)
    report = {
        "equation": "d alpha/dt = -i*omega*alpha + i*g*I_cartesian, g=eta*sqrt(weight/omega), alpha=(q+ip)/sqrt(2)",
        "one_mode_prescribed_current_residual": one_mode_identity(),
        "coupled_mode_duhamel_maxabs": mode_residual,
        "packet_duhamel_maxabs": packet_gap,
        "phased_packet_maxabs": lhs_size,
        "current_integral_maxabs": rhs_size,
        "unphased_end_amplitude_maxabs": unphased_size,
        "production_kernel_on_cartesian_I": err(cart),
        "production_kernel_on_crystal_pol_I": err(rotated),
        "symplectic_defect": symplectic_defect(phi_map),
        "output_commutator": bog,
        "packet_mass": mass,
        "elapsed": elapsed,
        "tf_clock": tf,
        "initial_state": "Complete real canonical basis. Not production W_in.",
        "gamma12_residual_localized": False,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__, Path(__file__).with_name("q8_m0_g1_h2_supported_benchmark_20261014.py")]),
    }
    write_new_json(OUT, report)
    print(json_dump(report))


def json_dump(report):
    import json
    return json.dumps(report, indent=2)


if __name__ == "__main__":
    main()
