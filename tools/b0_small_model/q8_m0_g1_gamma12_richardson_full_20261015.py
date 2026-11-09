"""One fourth-order output for the full saved Gamma12 quadrature.

Rows are the adjoint of the Richardson step, checked separately against forward
propagation. They are not the stored midpoint rows. Macro step 512 uses the
saved 1024-step vertices as the fine grid.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_gamma12_column_order_20261015 import CACHE, CYCLES, vertices
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import contract_initial, macro_pullback
from q8_m0_g1_gamma12_tail_columns_20261014 import BASELINE, load_rows
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/richardson_full_macro512.json"


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def accumulate(row, forces, b, bdag, a, free, omega):
    raw_u, raw_v = contract_initial(row, b, bdag, a)
    for force, t, h in forces:
        phase = np.exp(-1j * omega * t) * np.expm1(-1j * omega * h)
        raw = force @ (free * phase)
        raw_u += raw[:2]
        raw_v += np.conjugate(raw[2:])
    return raw_u, raw_v


def build(sample, geo, prot, fine_name, h, gamma):
    n_macro = CYCLES * 512
    coarse = vertices(sample, geo, prot, n_macro, h)
    fine_u = np.load(CACHE / f"{fine_name}_u.npy", mmap_mode="r")
    fine_d = np.load(CACHE / f"{fine_name}_d.npy", mmap_mode="r")
    times = np.empty(3 * n_macro)
    for i in range(n_macro):
        t = i * h
        times[3 * i:3 * i + 3] = (t + 0.25 * h, t + 0.5 * h, t + 0.75 * h)
    kernels, _ = kernel_at_times(gamma, p1_a2_protocol(0.8), times)
    steps = []
    for i, (uc, dc) in enumerate(coarse):
        steps.append({
            "t": i * h, "H": h, "uc": uc, "dc": np.real(dc),
            "u1": np.array(fine_u[2 * i]), "d1": np.real(fine_d[2 * i]),
            "u2": np.array(fine_u[2 * i + 1]), "d2": np.real(fine_d[2 * i + 1]),
            "kc": kernels[3 * i + 1], "k1": kernels[3 * i], "k2": kernels[3 * i + 2],
        })
    ne = steps[0]["u1"].shape[1]
    row = np.zeros((4, 2 * ne + 2), complex)
    forces = []
    for step in reversed(steps):
        row, force_c, force_f1, force_f2 = macro_pullback(row, step, gamma, None)
        forces.append((force_c, step["t"], h))
        forces.append((force_f1, step["t"], 0.5 * h))
        forces.append((force_f2, step["t"] + 0.5 * h, 0.5 * h))
    return row, forces


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _rows, cache, t0, _meta = load_rows()
    if abs(t0) > 0:
        raise AssertionError(t0)
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    # macro_pullback needs pol. Patch the steps through a closure by binding pol
    # into detector via the existing function argument.
    h = float(p1_a2_protocol(0.8).period) / 512
    print("rows driven", flush=True)
    row_d, forces_d = build_with_pol(sample, geo, p1_a2_protocol(0.8), "driven", h, gamma, pol)
    print("rows undriven", flush=True)
    row_u, forces_u = build_with_pol(sample, geo, p1_a2_protocol(0.0), "undriven", h, gamma, pol)
    panels = []
    oms, wts, _counts = wing_quadrature(
        cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
        omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    with np.load(BASELINE, allow_pickle=False) as saved:
        if not (np.allclose(saved["oms"], oms) and np.allclose(saved["wts"], wts)):
            raise AssertionError("Quadrature nodes changed")
    packet = base.q.omega_grid(p1_a2_protocol(0.8).omega0, p1_a2_protocol(0.8).t_pulse)
    packet = packet[packet > 0.0]
    _phi, norm = base.q.normalize_phi(packet, base.q.phi_unnormalized(
        packet, 2.0 * p1_a2_protocol(0.8).omega0, p1_a2_protocol(0.8).t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, p1_a2_protocol(0.8), norm)
    eta = base.q.eta_of(gamma)
    nph = 4 * len(oms)
    raw = {key: np.empty((2, nph), complex) for key in ("du", "uu", "dv", "uv")}
    u0_all = np.empty((2, nph), complex)
    for start in range(0, len(oms), 400):
        stop = min(start + 400, len(oms))
        ph, u0 = base.generated_photon_block(
            cache, gamma, oms[start:stop], wts[start:stop], pol, phi_of, eta)
        sl = slice(4 * start, 4 * stop)
        u0_all[:, sl] = u0
        raw["du"][:, sl], raw["dv"][:, sl] = accumulate(row_d, forces_d, ph)
        raw["uu"][:, sl], raw["uv"][:, sl] = accumulate(row_u, forces_u, ph)
        print("freq", stop, flush=True)
    dark_b = np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976:].T
    dark_bd = np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976:].T
    dark_a = np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976:].T
    zeros = np.zeros((2, 1976), complex)
    omega0 = np.zeros(1976)
    dark = {}
    for label, row, forces in (("d", row_d, forces_d), ("u", row_u, forces_u)):
        dark[label] = accumulate_arrays(row, forces, dark_b, dark_bd, dark_a, zeros, omega0)
    u = np.concatenate((u0_all + raw["du"] - raw["uu"], dark["d"][0] - dark["u"][0]), axis=1)
    v = np.concatenate((raw["dv"] - raw["uv"], dark["d"][1] - dark["u"][1]), axis=1)
    normal = u @ u.conj().T - v @ v.conj().T - np.eye(2)
    anomalous = u @ v.T - v @ u.T
    print("RN", float(normal[0, 0].real), float(np.max(np.abs(normal))), flush=True)
    print("RA", float(np.max(np.abs(anomalous))), flush=True)
    write_new_json(OUT, {
        "macro_ntpc": 512, "fine_ntpc": 1024, "cycles": 9,
        "quadrature_nodes": int(len(oms)), "dark_columns": 1976,
        "normal": pack(normal), "anomalous": pack(anomalous),
        "production_G1_pass": False, "provenance_sha256": provenance([__file__]),
    })


def build_with_pol(sample, geo, prot, fine_name, h, gamma, pol):
    n_macro = CYCLES * 512
    coarse = vertices(sample, geo, prot, n_macro, h)
    fine_u = np.load(CACHE / f"{fine_name}_u.npy", mmap_mode="r")
    fine_d = np.load(CACHE / f"{fine_name}_d.npy", mmap_mode="r")
    times = np.empty(3 * n_macro)
    for i in range(n_macro):
        t = i * h
        times[3 * i:3 * i + 3] = (t + 0.25 * h, t + 0.5 * h, t + 0.75 * h)
    kernels, _ = kernel_at_times(gamma, p1_a2_protocol(0.8), times)
    ne = int(fine_u.shape[2])
    row = np.zeros((4, 2 * ne + 2), complex)
    forces = []
    for i in range(n_macro - 1, -1, -1):
        step = {
            "t": i * h, "H": h, "uc": coarse[i][0], "dc": np.real(coarse[i][1]),
            "u1": np.array(fine_u[2 * i]), "d1": np.real(fine_d[2 * i]),
            "u2": np.array(fine_u[2 * i + 1]), "d2": np.real(fine_d[2 * i + 1]),
            "kc": kernels[3 * i + 1], "k1": kernels[3 * i], "k2": kernels[3 * i + 2],
        }
        row, force_c, force_f1, force_f2 = macro_pullback(row, step, gamma, pol)
        forces.append((force_c, i * h, h))
        forces.append((force_f1, i * h, 0.5 * h))
        forces.append((force_f2, i * h + 0.5 * h, 0.5 * h))
    return row, forces


def accumulate(row, forces, ph):
    return accumulate_arrays(row, forces, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)


def accumulate_arrays(row, forces, b, bdag, a, free, omega):
    raw_u, raw_v = contract_initial(row, b, bdag, a)
    by_h = {}
    for force, t, h in forces:
        by_h.setdefault(h, []).append((force, t))
    for h, pairs in by_h.items():
        stacked = np.stack([force for force, _t in pairs])
        times = np.array([t for _force, t in pairs])
        phase = np.exp(-1j * omega * times[:, None]) * np.expm1(-1j * omega * h)
        raw = np.einsum("sab,bc,sc->ac", stacked, free, phase, optimize=True)
        raw_u = raw_u + raw[:2]
        raw_v = raw_v + np.conjugate(raw[2:])
    return raw_u, raw_v


if __name__ == "__main__":
    main()
