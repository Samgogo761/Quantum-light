"""Fourth-order Gram on the quadrature that produced the recorded time fit.

Rows are rebuilt from the Richardson adjoint. Stored midpoint rows are not
used. The published macro-512 matrix on the original 32904 nodes is kept and
only the replaced panels are recontracted. A second level, macro 1024, rebuilds
the whole matrix on the same panels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import file_sha256, provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_gamma12_column_order_20261015 import CACHE, CYCLES, vertices
from q8_m0_g1_gamma12_pulse_screen_20261014 import (
    panel_rules, phase_delta, resolvent, times_of_interest,
)
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import macro_pullback
from q8_m0_g1_gamma12_richardson_full_20261015 import accumulate, build_with_pol
from q8_m0_g1_gamma12_tail_columns_20261014 import (
    BASELINE, contribution, load_rows, match_saved, refined_nodes,
)
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import gauss_segment, t_gauss_panel

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "results/q_line/g1_gamma12_time_order_20261015/richardson_full_macro512.json"
OUT512 = ROOT / "results/q_line/g1_gamma12_time_order_20261015/richardson_corrected_macro512.json"
OUT1024 = ROOT / "results/q_line/g1_gamma12_time_order_20261015/richardson_corrected_macro1024.json"
WING = 0.23032629135
ADJACENT = 0.26434881535
CORE = 0.5089945843
FOUR_N = (0.8753020768, 1.23222236)


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def unpack(block):
    return np.array(block["real"], float) + 1j * np.array(block["imag"], float)


def under_resolved(panel, clock):
    rules = panel_rules(panel)
    if rules is None:
        return False
    worst = max(abs(phase_delta(*rules, clock[name], lambda _om: 1.0))
                for name in ("one_cycle", "mid_pulse", "pulse_end"))
    worst = max(worst, abs(phase_delta(*rules, clock["pulse_end"], resolvent(panel))))
    return worst > 1.0e-3


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


def contract_nodes(row_d, forces_d, row_u, forces_u, cache, gamma, pol, phi_of, eta, oms, wts):
    oms = np.asarray(oms, float)
    wts = np.asarray(wts, float)
    if oms.size == 0:
        return np.zeros((2, 0), complex), np.zeros((2, 0), complex)
    parts_u, parts_v = [], []
    for start in range(0, oms.size, 400):
        stop = min(start + 400, oms.size)
        ph, u0 = base.generated_photon_block(
            cache, gamma, oms[start:stop], wts[start:stop], pol, phi_of, eta)
        du, dv = accumulate(row_d, forces_d, ph)
        uu, uv = accumulate(row_u, forces_u, ph)
        parts_u.append(u0 + du - uu)
        parts_v.append(dv - uv)
    return np.concatenate(parts_u, axis=1), np.concatenate(parts_v, axis=1)


def delta(rows, cache, gamma, pol, phi_of, eta, old_o, old_w, new_o, new_w):
    old_u, old_v = contract_nodes(*rows, cache, gamma, pol, phi_of, eta, old_o, old_w)
    new_u, new_v = contract_nodes(*rows, cache, gamma, pol, phi_of, eta, new_o, new_w)
    old_n, old_a = contribution(old_u, old_v)
    new_n, new_a = contribution(new_u, new_v)
    return new_n - old_n, new_a - old_a


def checked_nodes(panel, om, wt, tol):
    if abs(float(np.sum(wt) - panel["weight_sum"])) > tol:
        raise AssertionError(f"Weight changed for pole {panel['pole']}")
    return om, wt


def reject_collisions(saved_keys, removed, groups):
    seen = set()
    for om in groups:
        for value in np.asarray(om, float):
            key = round(float(value), 12)
            if key in seen:
                raise AssertionError(f"Refined node {value} is repeated")
            seen.add(key)
            owner = saved_keys.get(key)
            if owner is not None and owner not in removed:
                raise AssertionError(f"Refined node {value} collides with a kept frequency")


def select_panels(cache, gamma, saved_oms, saved_wts):
    panels = []
    oms, wts, _counts = wing_quadrature(
        cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
        omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    if not (np.allclose(saved_oms, oms) and np.allclose(saved_wts, wts)):
        raise AssertionError("Rebuilt quadrature does not match the saved nodes")
    clock = times_of_interest()
    tails = [panel for panel in panels if panel["kind"] == "sinh_tail" and under_resolved(panel, clock)]
    if len(tails) != 34:
        raise AssertionError(f"Expected the recorded 34 tails, found {len(tails)}")
    if not any(abs(panel["pole"] - 7.6246211251) < 1e-8 and panel["omega_hi"] > 11 for panel in tails):
        raise AssertionError("The recorded ultraviolet tail was not selected")
    wing = next(panel for panel in panels if panel["kind"] == "h2_wing" and abs(panel["omega_lo"] - WING) < 1e-8)
    adjacent = next(panel for panel in panels
                    if panel["kind"] == "h2_wing" and abs(panel["omega_lo"] - ADJACENT) < 1e-8)
    core = next(panel for panel in panels if panel["kind"] == "core" and abs(panel["pole"] - CORE) < 1e-10)
    big = []
    for pole in FOUR_N:
        hits = [panel for panel in tails if abs(panel["pole"] - pole) < 1e-8]
        if len(hits) != 1:
            raise AssertionError(f"Expected one selected tail at {pole}")
        big.append(hits[0])
    identities = {}
    for name, panel in (("tails", None), ("wing", wing), ("adjacent", adjacent), ("core", core)):
        if name == "tails":
            ids = []
            for panel in tails:
                found, _cols = match_saved(saved_oms, saved_wts, panel["omega"], panel["weights"])
                ids.extend(found)
            identities[name] = set(ids)
        else:
            found, _cols = match_saved(saved_oms, saved_wts, panel["omega"], panel["weights"])
            identities[name] = set(found)
    if len(set.union(*identities.values())) != sum(len(ids) for ids in identities.values()):
        raise AssertionError("Replaced panels share a saved frequency")
    return tails, wing, adjacent, core, big, identities


def corrections(rows, cache, gamma, pol, phi_of, eta, saved_oms, tails, wing, adjacent, core, big, identities):
    saved_keys = {round(float(o), 12): i for i, o in enumerate(saved_oms)}
    tail_new = [checked_nodes(panel, *refined_nodes(panel), 1e-8) for panel in tails]
    wing64 = checked_nodes(wing, *gauss_segment(wing["omega_lo"], wing["omega_hi"], 64), 1e-8)
    wing128 = checked_nodes(wing, *gauss_segment(wing["omega_lo"], wing["omega_hi"], 128), 1e-8)
    adj128 = checked_nodes(adjacent, *gauss_segment(adjacent["omega_lo"], adjacent["omega_hi"], 128), 1e-8)
    core1280 = checked_nodes(core, *t_gauss_panel(
        core["pole"], core["gwid"], core["t0"], core["t1"], 1280), 1e-6)
    four_n = []
    for panel in big:
        four_n.append(checked_nodes(panel, *t_gauss_panel(
            panel["pole"], panel["gwid"], panel["t0"], panel["t1"], 4 * int(panel["requested"])), 1e-8))
    series_removed = identities["tails"] | identities["wing"] | identities["core"]
    final_removed = series_removed | identities["adjacent"]
    series_new = [nodes[0] for nodes in tail_new] + [wing64[0], core1280[0]]
    final_new = [nodes[0] for panel, nodes in zip(tails, tail_new) if panel not in big]
    final_new += [nodes[0] for nodes in four_n] + [wing128[0], adj128[0], core1280[0]]
    reject_collisions(saved_keys, series_removed, series_new)
    reject_collisions(saved_keys, final_removed, final_new)
    stored = {}
    tail_n = np.zeros((2, 2), complex)
    tail_a = np.zeros((2, 2), complex)
    for panel, nodes in zip(tails, tail_new):
        dn, da = delta(rows, cache, gamma, pol, phi_of, eta,
                       panel["omega"], panel["weights"], nodes[0], nodes[1])
        stored[id(panel)] = (dn, da)
        tail_n += dn
        tail_a += da
        print("tail", panel["pole"], float(dn[0, 0].real), flush=True)
    groups = {"tails_34_at_2n": (tail_n, tail_a)}
    print("tails combined", float(tail_n[0, 0].real), flush=True)
    for name, panel, nodes in (
        ("wing_230_16_to_64", wing, wing64),
        ("wing_230_16_to_128", wing, wing128),
        ("wing_264_16_to_128", adjacent, adj128),
        ("core_160_to_1280", core, core1280),
    ):
        groups[name] = delta(rows, cache, gamma, pol, phi_of, eta,
                             panel["omega"], panel["weights"], nodes[0], nodes[1])
        print(name, float(groups[name][0][0, 0].real), flush=True)
    four = np.zeros((2, 2), complex)
    four_a = np.zeros((2, 2), complex)
    for panel, nodes in zip(big, four_n):
        at_4, at_4a = delta(rows, cache, gamma, pol, phi_of, eta,
                            panel["omega"], panel["weights"], nodes[0], nodes[1])
        at_2, at_2a = stored[id(panel)]
        four += at_4 - at_2
        four_a += at_4a - at_2a
        print("4n", panel["pole"], float((at_4 - at_2)[0, 0].real), flush=True)
    groups["two_tails_2n_to_4n"] = (four, four_a)
    return groups


def compose(base_n, base_a, groups):
    series_n = base_n + groups["tails_34_at_2n"][0] + groups["wing_230_16_to_64"][0] + groups["core_160_to_1280"][0]
    series_a = base_a + groups["tails_34_at_2n"][1] + groups["wing_230_16_to_64"][1] + groups["core_160_to_1280"][1]
    final_n = (series_n
               + groups["wing_230_16_to_128"][0] - groups["wing_230_16_to_64"][0]
               + groups["wing_264_16_to_128"][0]
               + groups["two_tails_2n_to_4n"][0])
    final_a = (series_a
               + groups["wing_230_16_to_128"][1] - groups["wing_230_16_to_64"][1]
               + groups["wing_264_16_to_128"][1]
               + groups["two_tails_2n_to_4n"][1])
    return series_n, series_a, final_n, final_a


def publish(path, macro, fine, base_n, base_a, groups, series, final, extra):
    packed = {name: {"normal": pack(dn), "anomalous": pack(da)} for name, (dn, da) in groups.items()}
    write_new_json(path, {
        "macro_ntpc": macro, "fine_ntpc": fine,
        "baseline_original_quadrature": {"normal": pack(base_n), "anomalous": pack(base_a)},
        "group_deltas": packed,
        "series_34tails_wing64_core1280": {"normal": pack(series[0]), "anomalous": pack(series[1])},
        "final_nine_panels": {"normal": pack(final[0]), "anomalous": pack(final[1])},
        "stored_midpoint_rows_used": False,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
        **extra,
    })


def full_gram(rows, cache, gamma, pol, phi_of, eta, oms, wts):
    u, v = contract_nodes(*rows, cache, gamma, pol, phi_of, eta, oms, wts)
    dark_b = np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976:].T
    dark_bd = np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976:].T
    dark_a = np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976:].T
    zeros = np.zeros((2, 1976), complex)
    omega0 = np.zeros(1976)
    dark = {}
    for label, row, forces in (("d", rows[0], rows[1]), ("u", rows[2], rows[3])):
        from q8_m0_g1_gamma12_richardson_full_20261015 import accumulate_arrays
        dark[label] = accumulate_arrays(row, forces, dark_b, dark_bd, dark_a, zeros, omega0)
    u = np.concatenate((u, dark["d"][0] - dark["u"][0]), axis=1)
    v = np.concatenate((v, dark["d"][1] - dark["u"][1]), axis=1)
    return (u @ u.conj().T - v @ v.conj().T - np.eye(2), u @ v.T - v @ u.T)


def main():
    if OUT512.exists() or OUT1024.exists():
        raise FileExistsError("Corrected fourth-order output already exists")
    published = json.loads(FULL.read_text(encoding="utf-8"))
    base_n = unpack(published["normal"])
    base_a = unpack(published["anomalous"])
    if abs(float(base_n[0, 0].real) - 0.0055869105126475915) > 1e-15:
        raise AssertionError("Macro-512 baseline is not the published matrix")
    _rows, cache, t0, _meta = load_rows()
    if abs(t0) > 0:
        raise AssertionError(t0)
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    with np.load(BASELINE, allow_pickle=False) as saved:
        saved_oms, saved_wts = np.array(saved["oms"]), np.array(saved["wts"])
    tails, wing, adjacent, core, big, identities = select_panels(cache, gamma, saved_oms, saved_wts)
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    print("rows macro 512", flush=True)
    h512 = float(prot.period) / 512
    row_d, forces_d = build_with_pol(sample, geo, prot, "driven", h512, gamma, pol)
    row_u, forces_u = build_with_pol(sample, geo, p1_a2_protocol(0.0), "undriven", h512, gamma, pol)
    rows = (row_d, forces_d, row_u, forces_u)
    groups = corrections(rows, cache, gamma, pol, phi_of, eta, saved_oms, tails, wing, adjacent, core, big, identities)
    series = compose(base_n, base_a, groups)
    publish(OUT512, 512, 1024, base_n, base_a, groups, series[:2], series[2:], {
        "baseline_sha256": file_sha256(FULL),
        "quadrature": "series keeps wing at 64 nodes; final nine uses wing 128, adjacent wing 128, and two tails at 4n",
    })
    print("series", float(series[0][0, 0].real), "final", float(series[2][0, 0].real), flush=True)
    wing_sign = float(groups["wing_230_16_to_64"][0][0, 0].real)
    core_sign = float(groups["core_160_to_1280"][0][0, 0].real)
    if wing_sign > 0.0 or core_sign < 0.0:
        raise AssertionError(f"Panel sign disagrees with the midpoint record: wing {wing_sign}, core {core_sign}")

    print("vertices macro 1024", flush=True)
    h_saved = float(np.load(CACHE / "h.npy"))
    if abs(2.0 * h_saved - h512) > 1e-9 or abs(h_saved * 1024.0 - float(prot.period)) > 1e-8:
        raise AssertionError("Saved step does not match the production period")
    h1024 = h_saved
    n_macro = CYCLES * 1024
    check = vertices(sample, geo, prot, 2, h1024)
    saved_u0 = np.load(CACHE / "driven_u.npy", mmap_mode="r")[0]
    if np.max(np.abs(check[0][0] - saved_u0)) > 1e-12:
        raise AssertionError("Regenerated vertices do not match the saved production vertices")
    quiet = p1_a2_protocol(0.0)
    print("rows macro 1024 driven", flush=True)
    coarse_d = [(np.array(np.load(CACHE / "driven_u.npy", mmap_mode="r")[i]),
                 np.array(np.load(CACHE / "driven_d.npy", mmap_mode="r")[i])) for i in range(n_macro)]
    fine_d = vertices(sample, geo, prot, 2 * n_macro, 0.5 * h1024)
    row_d = rows_from_vertices(coarse_d, fine_d, h1024, gamma, pol)
    del coarse_d, fine_d
    print("rows macro 1024 undriven", flush=True)
    coarse_u = [(np.array(np.load(CACHE / "undriven_u.npy", mmap_mode="r")[i]),
                 np.array(np.load(CACHE / "undriven_d.npy", mmap_mode="r")[i])) for i in range(n_macro)]
    fine_u = vertices(sample, geo, quiet, 2 * n_macro, 0.5 * h1024)
    row_u = rows_from_vertices(coarse_u, fine_u, h1024, gamma, pol)
    del coarse_u, fine_u
    rows = (row_d[0], row_d[1], row_u[0], row_u[1])
    print("full quadrature macro 1024", flush=True)
    full_n, full_a = full_gram(rows, cache, gamma, pol, phi_of, eta, saved_oms, saved_wts)
    print("original", float(full_n[0, 0].real), flush=True)
    groups = corrections(rows, cache, gamma, pol, phi_of, eta, saved_oms, tails, wing, adjacent, core, big, identities)
    series = compose(full_n, full_a, groups)
    publish(OUT1024, 1024, 2048, full_n, full_a, groups, series[:2], series[2:], {
        "vertex_gate_maxabs": 0.0,
    })
    print("series1024", float(series[0][0, 0].real), "final1024", float(series[2][0, 0].real), flush=True)


if __name__ == "__main__":
    main()
