"""Split the fourth-order commutator into U0, cross, and dynamic pieces.

Loads the saved macro-2048 rows only after their sha256 matches the identity
record. Does not rebuild those rows and does not overwrite them.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_input_wing_20261008 as wing
import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import stationary_photon_columns
from q8_m0_g1_gamma12_column_order_20261015 import CACHE
from q8_m0_g1_gamma12_ir_tail_20261015 import POLE, load_adjoint, log_gauss
from q8_m0_g1_gamma12_richardson_2048_20261015 import final_frequencies
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import contract_initial
from q8_m0_g1_gamma12_richardson_corrected_20261015 import select_panels
from q8_m0_g1_gamma12_richardson_full_20261015 import accumulate_arrays
from q8_m0_g1_gamma12_tail_columns_20261014 import BASELINE, load_rows
from q8_m0_g1_time_model_identity_20261015 import sha256

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_gamma12_time_order_20261015"
OUT = OUTDIR / "dynamic_map_decomposition_macro2048.json"
EXPECTED = {
    1024: "a21e59b48574a5e0beaf599c747f392ce791dd55f6119a17348fbcd674b40736",
    2048: "c04da52436198758155b3f2351911afcc513ed3cf46a3cef96f97baa747b9bde",
}
NAMES = ("static", "cross", "du_du", "vv", "anomalous")


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def blank():
    return {name: np.zeros((2, 2), complex) for name in NAMES}


def require_cache(macro):
    path = OUTDIR / f"adjoint_rows_macro{macro}.npz"
    digest = sha256(path)
    if digest != EXPECTED[macro]:
        raise AssertionError(f"Adjoint cache {path.name} is not the recorded file")
    return path, digest


def loop_raw(row, forces, b, bdag, a, free, omega):
    raw_u, raw_v = contract_initial(row, b, bdag, a)
    for force, t, h in forces:
        phase = np.exp(-1j * omega * t) * np.expm1(-1j * omega * h)
        raw = force @ (free * phase)
        raw_u = raw_u + raw[:2]
        raw_v = raw_v + np.conjugate(raw[2:])
    return raw_u, raw_v


def split_raw(row, forces, ph):
    init_u, init_v = contract_initial(row, ph.initial.b, ph.initial.bdag, ph.initial.A)
    total_u, total_v = accumulate_arrays(
        row, forces, ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
    return init_u, init_v, total_u - init_u, total_v - init_v


def add_block(acc, u0, du, v):
    acc["static"] += u0 @ u0.conj().T
    acc["cross"] += u0 @ du.conj().T + du @ u0.conj().T
    acc["du_du"] += du @ du.conj().T
    acc["vv"] += v @ v.conj().T
    u = u0 + du
    acc["anomalous"] += u @ v.T - v @ u.T


def port_index(nfreq, side):
    cols = (0, 1) if side == "L" else (2, 3)
    return np.array([4 * i + k for i in range(nfreq) for k in cols], int)


def pack_acc(acc, subtract_identity=False):
    normal = acc["static"] + acc["cross"] + acc["du_du"] - acc["vv"]
    out = {
        "static_U0": pack(acc["static"]),
        "cross_U0_dU": pack(acc["cross"]),
        "dU_dU": pack(acc["du_du"]),
        "VV": pack(acc["vv"]),
        "dynamic_quadratic": pack(acc["du_du"] - acc["vv"]),
        "normal_without_I": pack(normal),
        "anomalous": pack(acc["anomalous"]),
        "antihermitian_cross": float(np.max(np.abs(acc["cross"] - acc["cross"].conj().T))),
    }
    if subtract_identity:
        out["static_minus_I"] = pack(acc["static"] - np.eye(2))
        out["normal_residual"] = pack(normal - np.eye(2))
    return out


def add_into(dst, src):
    for part, acc in src.items():
        bucket = dst.setdefault(part, blank())
        for item in NAMES:
            bucket[item] += acc[item]


def contract_frequencies(rows, cache, gamma, pol, phi_of, eta, oms, wts, label):
    total, left, right = blank(), blank(), blank()
    init_only, force_only, interference = blank(), blank(), blank()
    for start in range(0, len(oms), 100):
        stop = min(start + 100, len(oms))
        ph, u0 = base.generated_photon_block(
            cache, gamma, oms[start:stop], wts[start:stop], pol, phi_of, eta)
        di_u, di_v, df_u, df_v = split_raw(rows[0], rows[1], ph)
        ui_u, ui_v, uf_u, uf_v = split_raw(rows[2], rows[3], ph)
        du = (di_u + df_u) - (ui_u + uf_u)
        v = (di_v + df_v) - (ui_v + uf_v)
        du_i = di_u - ui_u
        du_f = df_u - uf_u
        v_i = di_v - ui_v
        v_f = df_v - uf_v
        add_block(total, u0, du, v)
        add_block(init_only, u0, du_i, v_i)
        add_block(force_only, np.zeros_like(u0), du_f, v_f)
        force_only["cross"] += u0 @ du_f.conj().T + du_f @ u0.conj().T
        force_only["anomalous"] += u0 @ v_f.T - v_f @ u0.T
        interference["du_du"] += du_i @ du_f.conj().T + du_f @ du_i.conj().T
        interference["vv"] += v_i @ v_f.conj().T + v_f @ v_i.conj().T
        interference["anomalous"] += (du_i @ v_f.T + du_f @ v_i.T
                                      - v_i @ du_f.T - v_f @ du_i.T)
        nfreq = stop - start
        for acc, side in ((left, "L"), (right, "R")):
            idx = port_index(nfreq, side)
            add_block(acc, u0[:, idx], du[:, idx], v[:, idx])
        if stop % 2000 == 0 or stop == len(oms):
            print(label, stop, float((total["static"] + total["cross"] + total["du_du"] - total["vv"] - np.eye(2))[0, 0].real), flush=True)
    return {"all": total, "photon_L": left, "photon_R": right,
            "initial_history": init_only, "free_field_forcing": force_only,
            "init_force_interference": interference}


def contract_dark(rows):
    dark_b = np.array(np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976:].T)
    dark_bd = np.array(np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976:].T)
    dark_a = np.array(np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976:].T)
    acc = blank()
    driven_u = np.zeros((2, 1976), complex)
    driven_v = np.zeros_like(driven_u)
    undriven_u = np.zeros_like(driven_u)
    undriven_v = np.zeros_like(driven_u)
    for start in range(0, 1976, 200):
        stop = min(start + 200, 1976)
        zeros = np.zeros((2, stop - start), complex)
        omega = np.zeros(stop - start)
        sl = slice(start, stop)
        driven_u[:, sl], driven_v[:, sl] = accumulate_arrays(
            rows[0], rows[1], dark_b[:, sl], dark_bd[:, sl], dark_a[:, sl], zeros, omega)
        undriven_u[:, sl], undriven_v[:, sl] = accumulate_arrays(
            rows[2], rows[3], dark_b[:, sl], dark_bd[:, sl], dark_a[:, sl], zeros, omega)
        print("dark", stop, flush=True)
    add_block(acc, np.zeros((2, 1976), complex), driven_u - undriven_u, driven_v - undriven_v)
    return acc


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    path, digest = require_cache(2048)
    rows = load_adjoint(path)
    _saved_rows, cache, t0, _meta = load_rows()
    if abs(t0) > 0:
        raise AssertionError(t0)
    gamma = float(np.asarray(cache["gamma"]))
    sample = stationary_photon_columns(cache["gaps"], cache["u0"], cache["D0"], gamma, np.array([0.2]))
    labels = [lab[1:] for lab in sample.labels]
    if labels != [("L", 0), ("L", 1), ("R", 0), ("R", 1)]:
        raise AssertionError(f"Photon port order changed: {labels}")
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    with np.load(BASELINE, allow_pickle=False) as saved:
        saved_oms, saved_wts = np.array(saved["oms"]), np.array(saved["wts"])
    tails, wing_panel, adjacent, core, big, identities = select_panels(
        cache, gamma, saved_oms, saved_wts)
    pieces = final_frequencies(saved_oms, saved_wts, tails, wing_panel, adjacent, core, big, identities)
    probe_om = np.array([float(saved_oms[0])])
    probe_wt = np.array([float(saved_wts[0])])
    ph, _u0 = base.generated_photon_block(cache, gamma, probe_om, probe_wt, pol, phi_of, eta)
    loop_u, loop_v = loop_raw(rows[0], rows[1], ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
    fast_u, fast_v = accumulate_arrays(rows[0], rows[1], ph.initial.b, ph.initial.bdag, ph.initial.A, ph.free_A0, ph.omega)
    formula_gap = max(float(np.max(np.abs(loop_u - fast_u))), float(np.max(np.abs(loop_v - fast_v))))
    print("loop minus einsum", formula_gap, flush=True)
    if formula_gap > 1e-8:
        raise AssertionError("Explicit contraction and einsum contraction disagree")
    groups = {}
    total = blank()
    for name, om, wt in pieces:
        print("piece", name, len(om), flush=True)
        got = contract_frequencies(rows, cache, gamma, pol, phi_of, eta, om, wt, name)
        key = "tails" if name.startswith("tail") else name
        if key not in groups:
            groups[key] = got
        else:
            add_into(groups[key], got)
        for item in NAMES:
            total[item] += got["all"][item]
    print("dark", flush=True)
    dark = contract_dark(rows)
    for item in NAMES:
        total[item] += dark[item]
    panels = []
    wing.wing_quadrature(cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
                         omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    panel = next(item for item in panels if item["kind"] == "sinh_tail"
                 and abs(item["pole"] - POLE) < 1e-10 and item["omega_hi"] < 0.05)
    ir_om, ir_wt = log_gauss(1.0e-8, panel["omega_lo"], 64)
    print("infrared", ir_om.size, flush=True)
    infrared = contract_frequencies(rows, cache, gamma, pol, phi_of, eta, ir_om, ir_wt, "infrared")
    with_ir = blank()
    for item in NAMES:
        with_ir[item] = total[item] + infrared["all"][item]
    photon = blank()
    split = blank()
    for group in groups.values():
        for item in NAMES:
            photon[item] += group["all"][item]
    for part in ("initial_history", "free_field_forcing", "init_force_interference"):
        summed = sum_named(groups, part)
        for item in NAMES:
            split[item] += summed[item]
    split_gap = max(float(np.max(np.abs(split[item] - photon[item]))) for item in NAMES)
    print("split gap", split_gap, flush=True)
    if split_gap > 1e-8:
        raise AssertionError("Initial, forcing and interference do not rebuild the photon Gram")
    write_new_json(OUT, {
        "macro_ntpc": 2048,
        "cache_sha256": digest,
        "photon_column_order": ["L0", "L1", "R0", "R1"],
        "explicit_loop_minus_einsum": formula_gap,
        "final_nine": {name: pack_acc(groups[name]["all"]) for name in groups},
        "dark": pack_acc(dark),
        "total_final_nine": pack_acc(total, subtract_identity=True),
        "ports_of_photon_columns": {
            "L": pack_acc(sum_named(groups, "photon_L")),
            "R": pack_acc(sum_named(groups, "photon_R")),
        },
        "initial_versus_forcing": {
            "initial_history": pack_acc(sum_named(groups, "initial_history"), subtract_identity=True),
            "free_field_forcing": pack_acc(sum_named(groups, "free_field_forcing")),
            "init_force_interference": pack_acc(sum_named(groups, "init_force_interference")),
        },
        "infrared_1e-8_log64": pack_acc(infrared["all"]),
        "final_nine_plus_infrared": pack_acc(with_ir, subtract_identity=True),
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })
    residual = with_ir["static"] + with_ir["cross"] + with_ir["du_du"] - with_ir["vv"] - np.eye(2)
    print("RN", float(residual[0, 0].real), float(np.max(np.abs(residual))), flush=True)


def sum_named(groups, part):
    acc = blank()
    for group in groups.values():
        for item in NAMES:
            acc[item] += group[part][item]
    return acc


if __name__ == "__main__":
    main()
