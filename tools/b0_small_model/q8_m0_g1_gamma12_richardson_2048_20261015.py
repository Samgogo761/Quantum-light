"""Fourth-order macro step 2048 on the final nine-panel quadrature.

The adjoint rows are built again. Stored midpoint rows are not reused.
Frequency bins, photon ports, dark columns, and nodes outside [1e-4, 12]
are contracted with those same rows.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_input_wing_20261008 as wing
import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_gamma12_column_order_20261015 import CACHE, CYCLES, vertices
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import macro_pullback
from q8_m0_g1_gamma12_richardson_corrected_20261015 import (
    checked_nodes, pack, select_panels,
)
from q8_m0_g1_gamma12_richardson_full_20261015 import accumulate, accumulate_arrays
from q8_m0_g1_gamma12_tail_columns_20261014 import (
    BASELINE, contribution, load_rows, refined_nodes,
)
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import gauss_segment, t_gauss_panel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/richardson_macro2048_final.json"
PREV = {
    512: ROOT / "results/q_line/g1_gamma12_time_order_20261015/richardson_corrected_macro512.json",
    1024: ROOT / "results/q_line/g1_gamma12_time_order_20261015/richardson_corrected_macro1024.json",
}
EDGES = (1.0e-4, 0.02, 0.13, 0.23, 0.264, 0.30, 1.0, 3.0, 12.0)


def rows_from_vertices(coarse, fine, h, gamma, pol):
    n_macro = len(coarse)
    if len(fine) != 2 * n_macro:
        raise ValueError("Fine vertices must be exactly two per coarse step")
    times = np.empty(3 * n_macro)
    for i in range(n_macro):
        t = i * h
        times[3 * i:3 * i + 3] = (t + 0.25 * h, t + 0.5 * h, t + 0.75 * h)
    kernels, _ = kernel_at_times(gamma, p1_a2_protocol(0.8), times)
    ne = int(np.asarray(fine[0][0]).shape[1])
    row = np.zeros((4, 2 * ne + 2), complex)
    forces = []
    for i in range(n_macro - 1, -1, -1):
        if i % 4000 == 0:
            print("pullback", i, flush=True)
        step = {
            "t": i * h, "H": h,
            "uc": coarse[i][0], "dc": np.real(coarse[i][1]),
            "u1": np.asarray(fine[2 * i][0]), "d1": np.real(fine[2 * i][1]),
            "u2": np.asarray(fine[2 * i + 1][0]), "d2": np.real(fine[2 * i + 1][1]),
            "kc": kernels[3 * i + 1], "k1": kernels[3 * i], "k2": kernels[3 * i + 2],
        }
        row, force_c, force_f1, force_f2 = macro_pullback(row, step, gamma, pol)
        forces.append((force_c, i * h, h))
        forces.append((force_f1, i * h, 0.5 * h))
        forces.append((force_f2, i * h + 0.5 * h, 0.5 * h))
    return row, forces


def contract_uv0(rows, cache, gamma, pol, phi_of, eta, oms, wts):
    oms = np.asarray(oms, float)
    wts = np.asarray(wts, float)
    if oms.size == 0:
        z = np.zeros((2, 0), complex)
        return z, z.copy(), z.copy()
    parts_u, parts_v, parts_0 = [], [], []
    for start in range(0, oms.size, 400):
        stop = min(start + 400, oms.size)
        ph, u0 = base.generated_photon_block(
            cache, gamma, oms[start:stop], wts[start:stop], pol, phi_of, eta)
        du, dv = accumulate(rows[0], rows[1], ph)
        uu, uv = accumulate(rows[2], rows[3], ph)
        parts_u.append(u0 + du - uu)
        parts_v.append(dv - uv)
        parts_0.append(u0)
        print("freq", stop, flush=True)
    return (np.concatenate(parts_u, 1), np.concatenate(parts_v, 1),
            np.concatenate(parts_0, 1))


def grams(u, v, u0):
    normal = u @ u.conj().T - v @ v.conj().T
    static = u0 @ u0.conj().T
    anomalous = u @ v.T - v @ u.T
    return normal, static, anomalous


def add_into(total, piece):
    for i in range(3):
        total[i] = total[i] + piece[i]


def blank():
    return [np.zeros((2, 2), complex) for _ in range(3)]


def band_name(omega):
    if omega < EDGES[0]:
        return f"below-{EDGES[0]:g}"
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        if lo <= omega < hi:
            return f"{lo:g}--{hi:g}"
    return f"above-{EDGES[-1]:g}"


def accumulate_photon(bins, ports, u, v, u0, oms):
    oms = np.asarray(oms, float)
    names = [band_name(float(o)) for o in oms]
    for name in dict.fromkeys(names):
        mask = np.repeat(np.array([item == name for item in names]), 4)
        add_into(bins.setdefault(name, blank()), grams(u[:, mask], v[:, mask], u0[:, mask]))
    left = np.zeros(u.shape[1], bool)
    right = np.zeros(u.shape[1], bool)
    for i in range(oms.size):
        left[4 * i:4 * i + 2] = True
        right[4 * i + 2:4 * i + 4] = True
    add_into(ports["photon_L"], grams(u[:, left], v[:, left], u0[:, left]))
    add_into(ports["photon_R"], grams(u[:, right], v[:, right], u0[:, right]))


def pack_triplet(triplet):
    normal, static, anomalous = triplet
    return {"normal": pack(normal), "static_U0": pack(static),
            "dynamic": pack(normal - static), "anomalous": pack(anomalous)}


def fit_orders(r512, r1024, r2048):
    d1 = r1024 - r2048
    d2 = r512 - r2048
    h6 = (d2 - 17.0 * d1) / 3024.0
    h4 = (d1 - 63.0 * h6) / 15.0
    limit = r2048 - h4 - h6
    h4_only = r2048 + (r2048 - r1024) / 15.0
    return limit, h4, h6, h4_only


def final_frequencies(saved_oms, saved_wts, tails, wing, adjacent, core, big, identities):
    removed = identities["tails"] | identities["wing"] | identities["adjacent"] | identities["core"]
    keep = np.ones(saved_oms.size, bool)
    keep[np.fromiter(removed, int, len(removed))] = False
    pieces = [("kept", saved_oms[keep], saved_wts[keep])]
    for panel in tails:
        if panel in big:
            nodes = checked_nodes(panel, *t_gauss_panel(
                panel["pole"], panel["gwid"], panel["t0"], panel["t1"],
                4 * int(panel["requested"])), 1e-8)
            pieces.append((f"tail4:{panel['pole']}", nodes[0], nodes[1]))
        else:
            nodes = checked_nodes(panel, *refined_nodes(panel), 1e-8)
            pieces.append((f"tail2:{panel['pole']}", nodes[0], nodes[1]))
    for name, panel, n in (("wing230", wing, 128), ("wing264", adjacent, 128)):
        nodes = checked_nodes(panel, *gauss_segment(panel["omega_lo"], panel["omega_hi"], n), 1e-8)
        pieces.append((name, nodes[0], nodes[1]))
    nodes = checked_nodes(core, *t_gauss_panel(
        core["pole"], core["gwid"], core["t0"], core["t1"], 1280), 1e-6)
    pieces.append(("core", nodes[0], nodes[1]))
    return pieces


def exterior_nodes(cache, gamma, saved_oms, final_oms):
    oms, wts, _counts = wing.wing_quadrature(
        cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
        omega_lo=1.0e-8, omega_hi=48.0)
    blocked = {round(float(o), 12) for o in saved_oms}
    blocked.update(round(float(o), 12) for o in final_oms)
    keep_o, keep_w = [], []
    for om, wt in zip(oms, wts):
        if 1.0e-4 <= float(om) <= 12.0:
            continue
        key = round(float(om), 12)
        if key in blocked:
            continue
        blocked.add(key)
        keep_o.append(float(om))
        keep_w.append(float(wt))
    return np.asarray(keep_o, float), np.asarray(keep_w, float)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _rows, cache, t0, _meta = load_rows()
    if abs(t0) > 0:
        raise AssertionError(t0)
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    h_saved = float(np.load(CACHE / "h.npy"))
    if abs(h_saved * 1024.0 - float(prot.period)) > 1e-8:
        raise AssertionError("Saved step does not match the production period")
    dark_b = np.array(np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976:].T)
    coupling = cache["u0"].conj() @ dark_b
    print("dark current", float(np.max(np.abs(coupling))), flush=True)
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
    final_o = np.concatenate([om for _name, om, _wt in pieces])
    extra_o, extra_w = exterior_nodes(cache, gamma, saved_oms, final_o)
    print("exterior", extra_o.size, float(np.sum(extra_w)) if extra_o.size else 0.0, flush=True)
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    check = vertices(sample, geo, prot, 2, h_saved)
    saved_u0 = np.load(CACHE / "driven_u.npy", mmap_mode="r")[0]
    if np.max(np.abs(check[0][0] - saved_u0)) > 1e-12:
        raise AssertionError("Regenerated vertices do not match the saved production vertices")
    h = 0.5 * h_saved
    n_macro = CYCLES * 2048
    quiet = p1_a2_protocol(0.0)
    print("rows driven", flush=True)
    coarse = vertices(sample, geo, prot, n_macro, h)
    fine = vertices(sample, geo, prot, 2 * n_macro, 0.5 * h)
    row_d = rows_from_vertices(coarse, fine, h, gamma, pol)
    del coarse, fine
    print("rows undriven", flush=True)
    coarse = vertices(sample, geo, quiet, n_macro, h)
    fine = vertices(sample, geo, quiet, 2 * n_macro, 0.5 * h)
    row_u = rows_from_vertices(coarse, fine, h, gamma, pol)
    del coarse, fine
    rows = (row_d[0], row_d[1], row_u[0], row_u[1])

    bins, ports = {}, {"photon_L": blank(), "photon_R": blank()}
    total = blank()
    groups = {}
    for name, om, wt in pieces:
        print("contract", name, om.size, flush=True)
        u, v, u0 = contract_uv0(rows, cache, gamma, pol, phi_of, eta, om, wt)
        got = grams(u, v, u0)
        add_into(total, got)
        accumulate_photon(bins, ports, u, v, u0, om)
        if name != "kept":
            groups[name] = pack_triplet(got)
    print("contract dark", flush=True)
    zeros = np.zeros((2, 1976), complex)
    dark = {}
    for label, row, forces in (("d", rows[0], rows[1]), ("u", rows[2], rows[3])):
        dark[label] = accumulate_arrays(row, forces, dark_b, np.array(np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976:].T),
                                        np.array(np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976:].T), zeros, np.zeros(1976))
    dark_u = dark["d"][0] - dark["u"][0]
    dark_v = dark["d"][1] - dark["u"][1]
    dark_g = grams(dark_u, dark_v, zeros)
    add_into(total, dark_g)
    band_sum = blank()
    for triplet in bins.values():
        add_into(band_sum, triplet)
    mismatch = max(float(np.max(np.abs(band_sum[i] + dark_g[i] - total[i]))) for i in range(3))
    if mismatch > 1e-8:
        raise AssertionError(f"Band plus dark does not rebuild the Gram: {mismatch}")
    normal = total[0] - np.eye(2)
    anomalous = total[2]
    print("RN", float(normal[0, 0].real), float(np.max(np.abs(normal))), flush=True)
    print("contract exterior", extra_o.size, flush=True)
    eu, ev, e0 = contract_uv0(rows, cache, gamma, pol, phi_of, eta, extra_o, extra_w)
    exterior = grams(eu, ev, e0)
    normal_ext = normal + exterior[0]
    anomalous_ext = anomalous + exterior[2]
    levels = {}
    for ntpc, path in PREV.items():
        block = json.loads(path.read_text(encoding="utf-8"))["final_nine_panels"]
        levels[ntpc] = {kind: np.array(block[kind]["real"], float) + 1j * np.array(block[kind]["imag"], float)
                        for kind in ("normal", "anomalous")}
    levels[2048] = {"normal": normal, "anomalous": anomalous}
    fitted = {}
    for kind in ("normal", "anomalous"):
        limit, h4, h6, h4_only = fit_orders(
            levels[512][kind], levels[1024][kind], levels[2048][kind])
        fitted[kind] = {"limit_h4_h6": pack(limit), "h4_coefficient_at_2048": pack(h4),
                        "h6_coefficient_at_2048": pack(h6),
                        "two_point_h4_from_1024_2048": pack(h4_only)}
    write_new_json(OUT, {
        "macro_ntpc": 2048, "fine_ntpc": 4096,
        "final_nine": {"normal": pack(normal), "anomalous": pack(anomalous)},
        "final_nine_plus_exterior": {"normal": pack(normal_ext), "anomalous": pack(anomalous_ext)},
        "order_fit": fitted,
        "bands": {name: pack_triplet(triplet) for name, triplet in sorted(bins.items())},
        "ports": {name: pack_triplet(triplet) for name, triplet in ports.items()},
        "dark": pack_triplet(dark_g),
        "replacement_groups_dynamic_only_recorded_as_full_gram": groups,
        "exterior_1e-8_to_48_outside_1e-4_12": {
            "nodes": int(extra_o.size),
            "weight_sum": float(np.sum(extra_w)) if extra_o.size else 0.0,
            "frequencies": extra_o.tolist(),
            **pack_triplet(exterior),
        },
        "dark_current_maxabs": float(np.max(np.abs(coupling))),
        "dark_columns": 1976,
        "bright_poles_inside_window": True,
        "stored_midpoint_rows_used": False,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })
    print("limit", float(fitted["normal"]["limit_h4_h6"]["real"][0][0]), flush=True)


if __name__ == "__main__":
    main()
