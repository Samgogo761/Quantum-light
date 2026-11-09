"""Small chain: stationary incoming state, memory history, short P1, H2 rows.

Uses the existing two-k M0 sample and the same four-frequency, two-port,
two-polarization measure as the finite-memory fixture. Does not overwrite
older results and does not change gamma, D, or the production update.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import (
    ColumnState, current, midpoint_step, real_frequency_kernel, stationary_photon_columns,
)
from q8_m0_gsheet import gamma_self
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import (
    TWO_K, WEIGHTS, advance_half_frame, m0_sample, vertices_from_frame,
)
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import a_r_out_from_column, eta_of, phi_at
from q8_m0_s1_kernel import parallel_perp_basis

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_chain_benchmark_20261014/chain_benchmark_v2.json"
FREQ = np.array([0.05, 0.2, 0.6, 1.2])
WEIGHTS_NU = np.array([0.04, 0.1, 0.2, 0.2])


def err(a):
    return float(np.max(np.abs(a)))


def crystal_pol(geo):
    return np.vstack(parallel_perp_basis(geo)).astype(complex)


def vertex_window(sample, geo, prot, t0, nstep, h):
    frame = sample["frame"].copy()
    npre = int(round(float(t0) / h))
    if abs(npre * h - float(t0)) > 1e-9:
        raise ValueError("t0 is not an integer number of steps")
    for step in range(npre + nstep):
        t = step * h
        frame = advance_half_frame(geo, sample["kc"], prot, sample["neel"], frame, t + 0.25 * h, h)
        if step >= npre:
            yield vertices_from_frame(
                geo, sample["kc"], prot, sample["neel"], t + 0.5 * h, frame, sample["weights"])
        frame = advance_half_frame(geo, sample["kc"], prot, sample["neel"], frame, t + 0.75 * h, h)


def propagate_columns(initial, ref, h, nstep, k_scalar, pol):
    state = initial.copy()
    acc_u = np.zeros((pol.shape[0], state.b.shape[1]), complex)
    acc_v = np.zeros_like(acc_u)
    for step in range(nstep):
        t = step * h
        prev = state
        fa = ref.free_A(t + h) - ref.free_A(t)
        state = midpoint_step(state, ref.vertices(t + 0.5 * h), ref.D0, h, ref.gamma, free_A_increment=fa)
        mid = ColumnState(0.5 * (prev.b + state.b), 0.5 * (prev.bdag + state.bdag), 0.5 * (prev.A + state.A))
        src = pol @ current(mid, ref.vertices(t + 0.5 * h), ref.D0)
        acc_u += h * k_scalar[step] * src
        acc_v += h * k_scalar[step] * np.conj(src)
    exact = ref.state(nstep * h)
    return {
        "state_error": max(err(state.b - exact.b), err(state.bdag - exact.bdag), err(state.A - exact.A)),
        "acc_u": acc_u, "acc_v": acc_v, "final_A_error_vs_free": err(state.A - ref.free_A(nstep * h)),
    }


def bare_and_material_rows(ref, pol, phi_of):
    material = np.zeros((2, ref.initial.b.shape[1]), complex)
    bare = np.zeros_like(material)
    labels = ref.labels
    for j, lab in enumerate(labels):
        nu, side, pidx = lab
        a_r = a_r_out_from_column(ref.initial.A, ref.free_A0, labels, j)
        material[:, j] = phi_of(nu) * (pol @ (np.sqrt(nu) / eta_of(ref.gamma) * a_r))
        if side == "L":
            bare[:, j] = phi_of(nu) * pol[:, int(pidx)]
    return material, bare


def oscillator_bath(freqs, weights, gamma):
    eta = eta_of(gamma)
    om = np.repeat(np.asarray(freqs, float), 4)
    wt = np.repeat(np.asarray(weights, float), 4)
    g = eta * np.sqrt(wt / om)
    nph = len(om)
    b = np.zeros((2 * nph, 2 * nph))
    hfield = np.zeros((2, 2 * nph))
    for j, omega in enumerate(om):
        b[2 * j:2 * j + 2, 2 * j:2 * j + 2] = omega * np.array([[0.0, 1.0], [-1.0, 0.0]])
        hfield[j % 2, 2 * j] = np.sqrt(2.0) * g[j]
    jp = np.kron(np.eye(nph), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    c = -jp @ hfield.T
    return b, c, hfield


def susceptibility(omega, b, c, hfield, eps):
    dim = b.shape[0]
    resolvent = np.linalg.solve(-1j * (omega + 1j * eps) * np.eye(dim) - b, c)
    return hfield @ resolvent


def finite_matrices(sample, gamma):
    b, c, hfield = oscillator_bath(FREQ, WEIGHTS_NU, gamma)
    nel, nph = sample["n_el"], b.shape[0] // 2
    ne = 2 * nel
    je = np.kron(np.eye(nel), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    jp = np.kron(np.eye(nph), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    jt = np.kron(np.eye(nel + nph), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    return {"b": b, "c": c, "hfield": hfield, "je": je, "jp": jp, "jt": jt, "ne": ne, "nph": nph}


def quadratures(u):
    m = np.empty((2, 2 * u.shape[1]))
    m[:, ::2] = np.sqrt(2.0) * np.real(u)
    m[:, 1::2] = np.sqrt(2.0) * np.imag(u)
    return m


def compare_explicit_memory(sample, mats, vertices, h):
    b, c, hfield = mats["b"], mats["c"], mats["hfield"]
    je, jt, ne = mats["je"], mats["jt"], mats["ne"]
    nph = mats["nph"]
    nd = sample["n_el"] + nph
    np_ = 2 * nph
    dph = np.linalg.inv(np.eye(np_) - 0.5 * h * b)
    e = dph @ (np.eye(np_) + 0.5 * h * b)
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
    max_state = max_field = max_current = max_h = 0.0
    currents = []
    h2_explicit = np.zeros((2, nd), complex)
    h2_memory = np.zeros_like(h2_explicit)
    pol = crystal_pol(build_geometry())
    for step, (u, d) in enumerate(vertices):
        m = quadratures(u)
        l = -je @ m.T
        kmat = np.block([[np.zeros((ne, ne)), l @ hfield], [c @ m, b - c @ np.asarray(d, float) @ hfield]])
        max_h = max(max_h, err(kmat @ jt + jt @ kmat.T))
        full_new = np.linalg.solve(np.eye(2 * nd) - 0.5 * h * kmat, (np.eye(2 * nd) + 0.5 * h * kmat) @ full)
        full_mid = 0.5 * (full + full_new)
        power = powers[-1] @ e
        powers.append(power)
        af_mid = fmid @ powers[step] @ y0
        memory = np.zeros_like(af_mid)
        for j, ii in enumerate(currents):
            memory += (fmid @ powers[step - 1 - j] @ gh) @ ii
        amat = np.linalg.solve(np.eye(2) - self_mid @ (0.5 * h * m @ l - d),
                               af_mid + memory + self_mid @ m @ x)
        imat = m @ x + (0.5 * h * m @ l - d) @ amat
        x = x + h * l @ amat
        currents.append(imat)
        y = powers[step + 1] @ y0
        for j, ii in enumerate(currents):
            y += powers[step - j] @ gh @ ii
        reduced = np.vstack((x, y))
        max_state = max(max_state, err(reduced - full_new))
        max_field = max(max_field, err(amat - hfield @ full_mid[ne:]))
        max_current = max(max_current, err(imat - (m @ full_mid[:ne] - d @ hfield @ full_mid[ne:])))
        # Same current-to-H2 map for both representations. Not the final oscillator coordinate.
        h2_explicit += h * (pol @ (m @ full_mid[:ne] - d @ hfield @ full_mid[ne:]))
        h2_memory += h * (pol @ imat)
        full = full_new
    final_mode = np.zeros((2, np_), complex)
    for fi in range(len(FREQ)):
        for pidx in range(2):
            j = 4 * fi + 2 + pidx
            final_mode[pidx, 2 * j] = 1.0 / np.sqrt(2.0 * len(FREQ))
            final_mode[pidx, 2 * j + 1] = 1j / np.sqrt(2.0 * len(FREQ))
    y_final = full[ne:]
    oscillator_readout = final_mode @ y_final
    return {
        "state": max_state, "field": max_field, "current": max_current,
        "hamiltonian": max_h,
        "h2_current_explicit_vs_memory": err(h2_explicit - h2_memory),
        "h2_current_vs_final_oscillator": err(h2_explicit - oscillator_readout),
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    geo = build_geometry()
    sample = m0_sample(geo, TWO_K, WEIGHTS, +1)
    gamma = float(gamma_self(geo))
    eta = eta_of(gamma)
    if np.min(np.abs(sample["gap"][:, None] - FREQ[None, :])) < 1e-8:
        raise AssertionError("A fixture frequency sits on an electronic gap")
    ref = stationary_photon_columns(sample["gap"], sample["u"], sample["D"], gamma, FREQ)
    pol = crystal_pol(geo)
    prot = p1_a2_protocol(0.8)
    h_orbit = 0.05
    n_orbit = 8
    k_orbit, _ = kernel_at_times(gamma, prot, (np.arange(n_orbit) + 0.5) * h_orbit)
    k_fine, _ = kernel_at_times(gamma, prot, (np.arange(2 * n_orbit) + 0.5) * (h_orbit / 2))
    coarse = propagate_columns(ref.initial, ref, h_orbit, n_orbit, k_orbit, pol)
    fine = propagate_columns(ref.initial, ref, h_orbit / 2, 2 * n_orbit, k_fine, pol)
    stripped = ColumnState(np.zeros_like(ref.initial.b), np.zeros_like(ref.initial.bdag), ref.free_A0.copy())
    bare_run = propagate_columns(stripped, ref, h_orbit, n_orbit, k_orbit, pol)
    from q8_m0_k1_packet import normalize_phi, omega_grid, phi_unnormalized
    grid = omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0]
    _phi, nrm = normalize_phi(grid, phi_unnormalized(grid, 2 * prot.omega0, prot.t_pulse))
    material, bare = bare_and_material_rows(ref, pol, lambda nu: phi_at(nu, prot, nrm))
    field_gap = 0.0
    phi_on_nodes = []
    for j, lab in enumerate(ref.labels):
        nu, side, pidx = lab
        mapped = pol @ (np.sqrt(nu) / eta * a_r_out_from_column(ref.initial.A, ref.free_A0, ref.labels, j))
        transmission = pol[:, int(pidx)] if side == "L" else np.zeros(2)
        field_gap = max(field_gap, err(mapped - transmission))
        phi_on_nodes.append(float(np.abs(phi_at(nu, prot, nrm))))
    t_mid = 0.5 * float(prot.t_pulse)
    k_mid, _ = kernel_at_times(gamma, prot, t_mid + (np.arange(n_orbit) + 0.5) * h_orbit)
    def collect(initial, t_start, kernel):
        state = initial.copy()
        acc = np.zeros((2, state.b.shape[1]), complex)
        for step in range(n_orbit):
            t = t_start + step * h_orbit
            prev = state
            u = ref.vertices(t + 0.5 * h_orbit)
            state = midpoint_step(state, u, ref.D0, h_orbit, ref.gamma,
                                  free_A_increment=ref.free_A(t + h_orbit) - ref.free_A(t))
            mid = ColumnState(0.5 * (prev.b + state.b), 0.5 * (prev.bdag + state.bdag),
                              0.5 * (prev.A + state.A))
            acc += h_orbit * kernel[step] * (pol @ current(mid, u, ref.D0))
        return acc
    history_mid = err(collect(ref.state(t_mid), t_mid, k_mid)
                      - collect(ColumnState(np.zeros_like(ref.initial.b), np.zeros_like(ref.initial.bdag),
                                            ref.free_A(t_mid)), t_mid, k_mid))
    undriven = list(vertex_window(sample, geo, p1_a2_protocol(0.0), 0.0, 4, h_orbit))
    picture = []
    for step, (u, _d) in enumerate(undriven):
        analytic = sample["u"] * np.exp(1j * sample["gap"] * (step + 0.5) * h_orbit)
        picture.append(err(u - analytic))
    mats = finite_matrices(sample, gamma)
    chi_rows = []
    for omega in (0.05, 0.2, 0.6, 1.2):
        got = susceptibility(omega, *oscillator_bath(FREQ, WEIGHTS_NU, gamma), 1e-6)
        target = 1j * gamma / omega * np.eye(2)
        chi_rows.append({"omega": omega, "relative_to_i_gamma_over_omega": err(got - target) / err(target)})
    comb_rows = []
    for nbin, tmax in ((8, 40.0), (16, 40.0), (32, 80.0), (64, 80.0)):
        nodes = np.pi * np.arange(1, nbin + 1) / tmax
        wts = np.full(nbin, nodes[0] if nbin == 1 else nodes[1] - nodes[0])
        got = susceptibility(0.3, *oscillator_bath(nodes, wts, gamma), 1e-6)
        target = 1j * gamma / 0.3 * np.eye(2)
        comb_rows.append({"nodes": nbin, "span": tmax, "spacing": float(wts[0]),
                          "relative_to_markov": err(got - target) / err(target)})
    h_drive = float(prot.period) / 32.0
    t_mid = 0.5 * float(prot.t_pulse)
    npre = int(round(t_mid / h_drive))
    driven_vertices = list(vertex_window(sample, geo, prot, npre * h_drive, 12, h_drive))
    undriven_vertices = list(vertex_window(sample, geo, p1_a2_protocol(0.0), npre * h_drive, 12, h_drive))
    driven = compare_explicit_memory(sample, mats, driven_vertices, h_drive)
    quiet = compare_explicit_memory(sample, mats, undriven_vertices, h_drive)
    report = {
        "gamma": gamma,
        "eta": eta,
        "electronic_modes": int(sample["n_el"]),
        "frequencies": FREQ.tolist(),
        "weights": WEIGHTS_NU.tolist(),
        "stationary_orbit_error_h": coarse["state_error"],
        "stationary_orbit_error_h_over_2": fine["state_error"],
        "orbit_error_ratio_coarse_over_fine": coarse["state_error"] / max(fine["state_error"], 1e-30),
        "A0_minus_Af_maxabs": err(ref.initial.A - ref.free_A0),
        "zero_history_changes_H2_u": err(coarse["acc_u"] - bare_run["acc_u"]),
        "zero_history_changes_H2_v": err(coarse["acc_v"] - bare_run["acc_v"]),
        "material_U0_minus_bare_transmission_maxabs": err(material - bare),
        "phi_abs_on_these_four_frequencies": phi_on_nodes,
        "right_going_map_minus_bare_transmission_before_phi": field_gap,
        "zero_history_changes_midpulse_H2": history_mid,
        "undriven_frame_minus_gap_phase_maxabs": picture,
        "finite_bath_versus_markov": chi_rows,
        "denser_bath_versus_markov": comb_rows,
        "explicit_versus_memory_undriven_frame": quiet,
        "explicit_versus_memory_short_P1": driven,
        "short_P1": {"h": h_drive, "steps": 12, "t0": npre * h_drive, "a_star": 0.8},
        "reading": {
            "U0": "The packet-weighted row is phi times the right-going map. On these four frequencies phi is ~1e-15, so a tiny packet-weighted difference does not mean the field maps agree. Compare the map before phi.",
            "history": "Starting the column from Af with b=bdag=0 drops the scattered part already stored in the stationary incoming state.",
            "finite_match": "Explicit oscillators and the memory kernel are the same discrete system. Their agreement is not the Markov gamma limit.",
            "final_oscillator": "The end-time photon coordinate is not the H2 current projection.",
        },
        "production_G1_pass": False,
        "covariance_module_sha256": provenance([
            Path(__file__).with_name("q8_m0_p1_h2_finite_kcv_input_close_20261005.py")]) ,
        "provenance_sha256": provenance([
            __file__,
            Path(__file__).with_name("q8_m0_g1_finite_memory_noise_20261014.py"),
            Path(__file__).with_name("q8_m0_continuum_columns_20261004.py"),
            Path(__file__).with_name("q8_m0_p1_h2_finite_kcv_caller_fix_20261005.py"),
        ]),
    }
    write_new_json(OUT, report)
    print("orbit", coarse["state_error"], fine["state_error"])
    print("history H2", report["zero_history_changes_H2_u"])
    print("U0-P", report["material_U0_minus_bare_transmission_maxabs"])
    print("A0-Af", report["A0_minus_Af_maxabs"])
    print("picture", picture)
    print("chi", chi_rows)
    print("comb", comb_rows)
    print("quiet", quiet)
    print("driven", driven)


if __name__ == "__main__":
    main()
