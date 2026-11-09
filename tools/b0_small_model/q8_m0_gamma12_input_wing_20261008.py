"""Double only the non-core quadrature panels. Core 160 and ntpc=4096 stay fixed.

Wing factor 1 must reproduce the saved n_panel=160 frequencies, weights, and
outputs. Wing factor 2 doubles the node count actually used on every panel
that is not the sinh resonance core, including H2-window direct Gauss wings,
sinh tails, and the outer 12-node direct segments. The original tail cap is
not reapplied after doubling. The frozen quadrature function is not edited.
"""
from pathlib import Path
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q

EDGES = (0.001, 0.02, 0.07, 0.13, 0.30, 1.0, 2.5, 6.0, 12.0)


def wing_quadrature(gaps, u, gamma, *, n_panel=160, wing_factor=1,
                     omega_lo=1.0e-3, omega_hi=12.0, panels=None):
    """Same partition as input_quadrature. Only non-core node counts scale.

    If panels is a list, each complete quadrature piece is appended before
    node merging. Those pieces are the integrals. A later frequency bin that
    keeps only some of their nodes is a node group, not another panel.
    """
    if n_panel != 160 or wing_factor not in (1, 2, 4):
        raise ValueError("This control is core 160 with wing factor 1, 2, or 4")
    bright, widths, groups = q.grouped_bright_gaps(gaps, u, gamma)
    if bright.size == 0:
        raise RuntimeError("No bright gaps for the finite electronic sample.")
    chunks_o, chunks_w = [], []
    counts = {"core_calls": 0, "core_nodes": 0,
              "h2_wing_calls": 0, "h2_wing_nodes": 0,
              "sinh_tail_calls": 0, "sinh_tail_nodes": 0,
              "outer_direct_calls": 0, "outer_direct_nodes": 0}

    def add(om, wt, kind, requested, o0, o1, pole, gwid=None, t0=None, t1=None):
        om = np.asarray(om, float)
        wt = np.asarray(wt, float)
        counts[kind + "_calls"] += 1
        counts[kind + "_nodes"] += int(requested)
        if om.size:
            chunks_o.append(om)
            chunks_w.append(wt)
            if panels is not None:
                panels.append({"kind": kind, "pole": float(pole),
                               "omega_lo": float(min(o0, o1)),
                               "omega_hi": float(max(o0, o1)),
                               "nodes": int(om.size),
                               "requested": int(requested),
                               "weight_sum": float(np.sum(wt)),
                               "gwid": None if gwid is None else float(gwid),
                               "t0": None if t0 is None else float(t0),
                               "t1": None if t1 is None else float(t1),
                               "omega": om.copy(), "weights": wt.copy()})

    n_g = len(groups)
    g_left = np.array([float(bright[grp][0]) for grp in groups], float)
    g_right = np.array([float(bright[grp][-1]) for grp in groups], float)
    cuts = np.empty(n_g + 1, float)
    cuts[0] = float(omega_lo)
    cuts[-1] = float(omega_hi)
    for i in range(n_g - 1):
        cuts[i + 1] = 0.5 * (g_right[i] + g_left[i + 1])
    if np.any(np.diff(cuts) <= 0):
        raise RuntimeError("Inverted quadrature cells.")
    h2_lo, h2_hi = 0.02, 0.30
    h2_span = h2_hi - h2_lo

    def add_omega_split(dg, gwid, t0, t1):
        if t1 <= t0 + 1e-14:
            return
        o0 = float(dg) + float(gwid) * np.sinh(t0)
        o1 = float(dg) + float(gwid) * np.sinh(t1)
        if o1 < o0:
            o0, o1 = o1, o0
            t0, t1 = t1, t0
        marks = [o0]
        for x in (h2_lo, h2_hi):
            if o0 < x < o1:
                marks.append(x)
        marks.append(o1)
        for x0, x1 in zip(marks[:-1], marks[1:]):
            if x1 <= x0 + 1e-14:
                continue
            mid = 0.5 * (x0 + x1)
            if h2_lo <= mid <= h2_hi:
                n = q.even_count(max(8, int(round(int(q.N_H2) * (x1 - x0) / h2_span))), 8)
                n *= wing_factor
                add(*q.gauss_segment(x0, x1, n), "h2_wing", n, x0, x1, dg)
            else:
                ta = float(np.arcsinh((x0 - dg) / gwid))
                tb = float(np.arcsinh((x1 - dg) / gwid))
                if tb < ta:
                    ta, tb = tb, ta
                n = q.even_count(int(q.N_TAIL_MIN + 2.0 * (tb - ta)), q.N_TAIL_MIN)
                n = min(n, q.N_TAIL_CAP)
                # Double the count that factor 1 actually used. Do not reapply the cap.
                n *= wing_factor
                add(*q.t_gauss_panel(dg, gwid, ta, tb, n), "sinh_tail", n, x0, x1, dg,
                    gwid, ta, tb)

    for i, grp in enumerate(groups):
        cell_lo, cell_hi = float(cuts[i]), float(cuts[i + 1])
        if cell_hi <= cell_lo:
            raise RuntimeError("Inverted group cell.")
        dgs = bright[grp]
        gs = widths[grp]
        pole_cuts = (np.concatenate(([cell_lo], 0.5 * (dgs[:-1] + dgs[1:]), [cell_hi]))
                     if dgs.size > 1 else np.array([cell_lo, cell_hi]))
        for j, dg in enumerate(dgs):
            a, b = float(pole_cuts[j]), float(pole_cuts[j + 1])
            if b <= a:
                raise RuntimeError("Inverted pole cell.")
            gwid = max(float(gs[j]), 1e-16)
            _rank, all_w = q.cluster_rank_width(gaps, u, gamma, dg)
            g_max = float(np.max(all_w)) if all_w.size else gwid
            t_left = float(np.clip(np.arcsinh((a - float(dg)) / gwid), -q.T_TAIL_CAP, q.T_TAIL_CAP))
            t_right = float(np.clip(np.arcsinh((b - float(dg)) / gwid), -q.T_TAIL_CAP, q.T_TAIL_CAP))
            half = max(min(float(dg) - a, b - float(dg)), 8.0 * gwid)
            t_cell = float(np.arcsinh(half / gwid))
            t_wide = float(np.arcsinh(8.0 * g_max / gwid))
            t_core = min(q.T_MAX_CAP, max(q.T_MAX_MIN, t_wide, min(t_cell, q.T_MAX_CAP)))
            t_core = min(t_core, max(-t_left, 0.0), max(t_right, 0.0))
            if t_core <= 1e-12:
                raise RuntimeError("Inverted sinh window.")
            core_lo = float(dg) + gwid * np.sinh(-t_core)
            core_hi = float(dg) + gwid * np.sinh(t_core)
            add(*q.t_gauss_panel(float(dg), gwid, -t_core, t_core, n_panel),
                "core", n_panel, core_lo, core_hi, float(dg), gwid, -t_core, t_core)
            add_omega_split(float(dg), gwid, t_left, -t_core)
            add_omega_split(float(dg), gwid, t_core, t_right)
            om_left = float(dg) + gwid * np.sinh(t_left)
            om_right = float(dg) + gwid * np.sinh(t_right)
            if om_left > a + 1e-10:
                add(*q.gauss_segment(a, om_left, 12 * wing_factor),
                    "outer_direct", 12 * wing_factor, a, om_left, float(dg))
            if b > om_right + 1e-10:
                add(*q.gauss_segment(om_right, b, 12 * wing_factor),
                    "outer_direct", 12 * wing_factor, om_right, b, float(dg))
    oms = np.concatenate(chunks_o) if chunks_o else np.zeros(0)
    wts = np.concatenate(chunks_w) if chunks_w else np.zeros(0)
    oms, wts = q.merge_nodes(oms, wts)
    keep = (oms > 1e-10) & (wts > 0.0)
    for gval in q.unique_gaps(gaps):
        keep &= np.abs(oms - gval) > 1e-14
    return oms[keep], wts[keep], counts


def matrix_change(new, old):
    delta = np.asarray(new) - np.asarray(old)
    return {"maxabs": float(np.max(np.abs(delta))),
            "relative_fro_using_finer": float(np.linalg.norm(delta) / max(np.linalg.norm(new), 1e-30)),
            "signed_delta_real": delta.real.tolist(),
            "signed_delta_imag": delta.imag.tolist()}


def band_sums(u, v, oms):
    """Node-location groups. Not complete quadrature panels."""
    omega = np.repeat(np.asarray(oms, float), 4)
    out = {}
    selections = {f"below-{EDGES[0]:g}": np.flatnonzero(omega < EDGES[0])}
    selections.update({f"{lo:g}--{hi:g}": np.flatnonzero((omega >= lo) & (omega < hi))
                       for lo, hi in zip(EDGES[:-1], EDGES[1:])})
    selections[f"above-{EDGES[-1]:g}"] = np.flatnonzero(omega >= EDGES[-1])
    selections["dark"] = np.arange(len(omega), u.shape[1])
    total = 0
    for name, ids in selections.items():
        ub, vb = u[:, ids], v[:, ids]
        normal = ub @ ub.conj().T - vb @ vb.conj().T
        out[name] = {"columns": int(len(ids)),
                     "trace_n": float(np.sum(np.abs(vb) ** 2)),
                     "normal_maxabs": float(np.max(np.abs(normal))),
                     "normal_real": normal.real.tolist(),
                     "normal_imag": normal.imag.tolist()}
        total += normal
    if not np.allclose(total - np.eye(2), u @ u.conj().T - v @ v.conj().T - np.eye(2),
                       atol=2e-9, rtol=1e-8):
        raise AssertionError("Frequency partition omitted or doubled columns")
    return out


def contract(cache, rows, gamma, pol, phi_of, eta, t0, oms, wts, old_nph, ndark):
    nph, total = 4 * len(oms), 4 * len(oms) + ndark
    vals = {k: np.empty((2, total), complex) for k in base.RAW_KEYS + ("u0",)}
    timings = {"stationary_assembly_s": 0.0, "driven_contraction_s": 0.0, "undriven_contraction_s": 0.0}
    for first in range(0, len(oms), 128):
        last = min(first + 128, len(oms))
        t = time.perf_counter()
        ph, u0 = base.generated_photon_block(cache, gamma, oms[first:last], wts[first:last], pol, phi_of, eta)
        timings["stationary_assembly_s"] += time.perf_counter() - t
        got, tt = base._contract(rows, (ph.initial.b, ph.initial.bdag, ph.initial.A),
                                 ph.omega, ph.free_A0, t0, 128, 2048)
        dest = slice(4 * first, 4 * last)
        for key in base.RAW_KEYS:
            vals[key][:, dest] = got[key]
        vals["u0"][:, dest] = u0
        for path in tt:
            timings[path + "_contraction_s"] += tt[path]
        if first == 0 or last == len(oms) or last % 2048 == 0:
            print(f"wing frequencies {last}/{len(oms)}", flush=True)
        del ph, got
    for first in range(0, ndark, 512):
        count = min(512, ndark - first)
        ids = np.arange(old_nph + first, old_nph + first + count)
        initial = tuple(np.array(cache[k][ids].T, copy=True) for k in ("b_cols", "bdag_cols", "A_cols"))
        om = np.array(cache["omega"][ids])
        free = np.array(cache["free_A0_cols"][ids].T)
        if np.any(om != 0) or np.any(free != 0):
            raise ValueError("Dark columns gained forcing")
        got, tt = base._contract(rows, initial, om, free, t0, 128, 2048)
        dest = slice(nph + first, nph + first + count)
        for key in base.RAW_KEYS:
            vals[key][:, dest] = got[key]
        vals["u0"][:, dest] = cache["U0_canonical"][:, ids]
        for path in tt:
            timings[path + "_contraction_s"] += tt[path]
    du = vals["driven_raw_u"] - vals["undriven_raw_u"]
    v = vals["driven_raw_v"] - vals["undriven_raw_v"]
    u = vals["u0"] + du
    vals.update(column_id=np.arange(total), oms=oms, wts=wts, U=u, V=v, delta_u=du,
                n=v.conj() @ v.T, m=u @ v.T, m_U0VT=vals["u0"] @ v.T, m_source_source=du @ v.T,
                normal_contribution=u @ u.conj().T - v @ v.conj().T,
                anomalous_contribution=u @ v.T - v @ u.T,
                U0_gram=vals["u0"] @ vals["u0"].conj().T)
    base.r1._finite_arrays(vals)
    return vals, timings


def run(wing_factor):
    out = base.OUT / f"wing_factor_{wing_factor}.json"
    if out.exists() or out.with_suffix(".npz").exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    manifest, cache, rows, info, identity = base.load_frozen()
    ref_meta = json.loads((base.OUT / "n_panel_160.json").read_text(encoding="utf-8"))
    ref_path = base.OUT / "n_panel_160.npz"
    if base.prep.sha256_file(ref_path) != ref_meta["row_sha256"]:
        raise ValueError("Saved core-160 output changed")
    predecessor = None
    if wing_factor == 2:
        gate = json.loads((base.OUT / "wing_factor_1.json").read_text(encoding="utf-8"))
        if not gate.get("reproduces_core_160"):
            raise ValueError("Wing factor 1 has not reproduced the saved core-160 output")
        predecessor = ref_path
    elif wing_factor == 4:
        gate = json.loads((base.OUT / "wing_factor_2.json").read_text(encoding="utf-8"))
        pred = base.OUT / "wing_factor_2.npz"
        if gate.get("wing_factor") != 2 or base.prep.sha256_file(pred) != gate["row_sha256"]:
            raise ValueError("Wing factor 2 artifact is missing or changed")
        if abs(float(gate["normal_ccr_maxabs"]) - 0.18415101795108058) > 1e-9:
            raise ValueError("Wing factor 2 is not the recorded control")
        predecessor = pred
    gamma = float(cache["gamma"])
    pol = np.array(cache["pol"], copy=True)
    prot = base.p1_a2_protocol(base.q.A_STAR)
    op = base.q.omega_grid(prot.omega0, prot.t_pulse)
    op = op[op > 0]
    _, norm = base.q.normalize_phi(op, base.q.phi_unnormalized(op, 2 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    t0 = float(manifest["time"].get("t0", 0.0))
    official, official_w = base.q.input_quadrature(cache["gaps"], cache["u0"], gamma, n_panel=160)
    oms, wts, counts = wing_quadrature(cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=wing_factor)
    if wing_factor == 1:
        node_check = base.comparison(oms, official)
        weight_check = base.comparison(wts, official_w)
        if not (node_check["allclose"] and weight_check["allclose"]):
            raise AssertionError("Wing factor 1 does not reproduce frozen quadrature")
        with np.load(ref_path, allow_pickle=False) as ref:
            saved_nodes = base.comparison(oms, ref["oms"])
            saved_weights = base.comparison(wts, ref["wts"])
        if not (saved_nodes["allclose"] and saved_weights["allclose"]):
            raise AssertionError("Wing factor 1 does not reproduce saved core-160 nodes")
    else:
        if len(oms) <= len(official):
            raise AssertionError("Wing doubling did not increase the node count")
        node_check = weight_check = None
    if np.any(wts <= 0) or np.any(np.diff(oms) <= 0) or len(np.unique(np.round(oms, 12))) != len(oms):
        raise ValueError("Quadrature ordering, weight, or frequency-key collision")
    ndark = int(manifest["physical_scope"]["n_dark_columns"])
    old_nph = int(manifest["physical_scope"]["n_photon_columns"])
    if ndark != 1976:
        raise ValueError("Dark space changed")
    print(f"wing factor {wing_factor}: {len(oms)} frequencies", flush=True)
    vals, timings = contract(cache, rows, gamma, pol, phi_of, eta, t0, oms, wts, old_nph, ndark)
    normal = vals["normal_contribution"] - np.eye(2)
    changes = {}
    with np.load(predecessor if predecessor is not None else ref_path, allow_pickle=False) as ref:
        changes["n"] = matrix_change(vals["n"], ref["n"])
        changes["m"] = matrix_change(vals["m"], ref["m"])
        changes["normal_residual"] = matrix_change(normal, ref["normal_contribution"] - np.eye(2))
        changes["anomalous"] = matrix_change(vals["anomalous_contribution"], ref["anomalous_contribution"])
        ref_u, ref_v, ref_om = ref["U"], ref["V"], ref["oms"]
        ref_bands = band_sums(ref_u, ref_v, ref_om)
    changes["compared_with"] = "wing_factor_2" if wing_factor == 4 else "core_160"
    bands = band_sums(vals["U"], vals["V"], oms)
    band_delta = {}
    for name in bands:
        band_delta[name] = {
            "trace_n": bands[name]["trace_n"],
            "trace_n_minus_core160": bands[name]["trace_n"] - ref_bands[name]["trace_n"],
            "normal_maxabs": bands[name]["normal_maxabs"],
            "normal_change_maxabs": float(np.max(np.abs(
                np.array(bands[name]["normal_real"]) + 1j * np.array(bands[name]["normal_imag"])
                - (np.array(ref_bands[name]["normal_real"]) + 1j * np.array(ref_bands[name]["normal_imag"]))))),
        }
    reproduces = wing_factor == 1 and all(changes[k]["maxabs"] <= 1e-8 for k in changes)
    base.r1._write_npz_atomic(out.with_suffix(".npz"), vals)
    meta = {
        "status": "noncore_wing_density_only",
        "wing_factor": wing_factor,
        "core_n_panel": 160,
        "ntpc": 4096,
        "nstep": info["nstep"],
        "h": info["h"],
        "identity": identity,
        "script_sha256": base.prep.sha256_file(Path(__file__)),
        "required_core160_row_sha256": ref_meta["row_sha256"],
        "node_check_against_frozen_function": node_check,
        "weight_check_against_frozen_function": weight_check,
        "panel_request_counts": counts,
        "n_nodes": int(len(oms)),
        "n_photon_columns": 4 * int(len(oms)),
        "n_dark_columns": ndark,
        "n_contracted_columns": int(vals["U"].shape[1]),
        "n_trace": float(np.trace(vals["n"]).real),
        "normal_ccr_maxabs": float(np.max(np.abs(normal))),
        "anomalous_maxabs": float(np.max(np.abs(vals["anomalous_contribution"]))),
        "U0_gram_defect": float(np.max(np.abs(vals["U0_gram"] - np.eye(2)))),
        "packet_mass": base.q.packet_mass_dnu(oms, wts, prot, norm),
        "changes_vs_core160": changes,
        "bands": band_delta,
        "reproduces_core_160": reproduces,
        "timings": timings,
        "total_wall_s": time.perf_counter() - started,
        "row_sha256": None,
        "core_nodes_not_doubled": True,
        "tail_cap_not_reapplied_after_doubling": True,
        "outer_12_node_segments_scaled": True,
        "endpoints_unchanged": [0.001, 12.0],
        "background_repropagated": False,
        "production_qualification": False,
        "physical_conclusion": False,
    }
    meta["row_sha256"] = base.prep.sha256_file(out.with_suffix(".npz"))
    base.prep.write_json_atomic(out, meta)
    print(json.dumps({k: meta[k] for k in (
        "wing_factor", "n_nodes", "n_trace", "normal_ccr_maxabs", "anomalous_maxabs",
        "reproduces_core_160", "total_wall_s")}, indent=2), flush=True)
    if wing_factor == 1 and not reproduces:
        raise AssertionError("Wing factor 1 output does not reproduce core 160")


def run_endpoint(which):
    """Move one frequency endpoint. Core 160 and wing factor 2 stay fixed."""
    if which == "low":
        omega_lo, omega_hi = 1.0e-4, 12.0
    elif which == "lower":
        omega_lo, omega_hi = 1.0e-5, 12.0
    elif which == "high":
        omega_lo, omega_hi = 1.0e-3, 24.0
    else:
        raise ValueError(which)
    out = base.OUT / f"endpoint_{which}.json"
    if out.exists() or out.with_suffix(".npz").exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    manifest, cache, rows, info, identity = base.load_frozen()
    pred_meta = json.loads((base.OUT / "wing_factor_2.json").read_text(encoding="utf-8"))
    pred = base.OUT / "wing_factor_2.npz"
    if base.prep.sha256_file(pred) != pred_meta["row_sha256"]:
        raise ValueError("Wing factor 2 artifact changed")
    gamma = float(cache["gamma"])
    pol = np.array(cache["pol"], copy=True)
    prot = base.p1_a2_protocol(base.q.A_STAR)
    op = base.q.omega_grid(prot.omega0, prot.t_pulse)
    op = op[op > 0]
    _, norm = base.q.normalize_phi(op, base.q.phi_unnormalized(op, 2 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    t0 = float(manifest["time"].get("t0", 0.0))
    oms, wts, counts = wing_quadrature(
        cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
        omega_lo=omega_lo, omega_hi=omega_hi)
    if np.any(wts <= 0) or np.any(np.diff(oms) <= 0):
        raise ValueError("Endpoint quadrature is not strictly ordered")
    ndark = int(manifest["physical_scope"]["n_dark_columns"])
    old_nph = int(manifest["physical_scope"]["n_photon_columns"])
    print(f"endpoint {which}: {len(oms)} frequencies, [{omega_lo:g}, {omega_hi:g}]", flush=True)
    vals, timings = contract(cache, rows, gamma, pol, phi_of, eta, t0, oms, wts, old_nph, ndark)
    normal = vals["normal_contribution"] - np.eye(2)
    u0, du, v = vals["u0"], vals["delta_u"], vals["V"]
    pump = u0 @ du.conj().T + du @ u0.conj().T + du @ du.conj().T - v @ v.conj().T
    with np.load(pred, allow_pickle=False) as ref:
        changes = {
            "n": matrix_change(vals["n"], ref["n"]),
            "m": matrix_change(vals["m"], ref["m"]),
            "normal_residual": matrix_change(normal, ref["normal_contribution"] - np.eye(2)),
            "anomalous": matrix_change(vals["anomalous_contribution"], ref["anomalous_contribution"]),
        }
        ref_bands = band_sums(ref["U"], ref["V"], ref["oms"])
    bands = band_sums(vals["U"], vals["V"], oms)
    band_delta = {}
    for name in bands:
        new_n = np.array(bands[name]["normal_real"]) + 1j * np.array(bands[name]["normal_imag"])
        old_n = np.array(ref_bands[name]["normal_real"]) + 1j * np.array(ref_bands[name]["normal_imag"])
        band_delta[name] = {
            "trace_n": bands[name]["trace_n"],
            "trace_n_minus_wing2": bands[name]["trace_n"] - ref_bands[name]["trace_n"],
            "normal_change_maxabs": float(np.max(np.abs(new_n - old_n))),
        }
    base.r1._write_npz_atomic(out.with_suffix(".npz"), vals)
    meta = {
        "status": "single_frequency_endpoint",
        "endpoint": which,
        "omega_lo": omega_lo,
        "omega_hi": omega_hi,
        "wing_factor": 2,
        "core_n_panel": 160,
        "ntpc": 4096,
        "n_nodes": int(len(oms)),
        "n_trace": float(np.trace(vals["n"]).real),
        "normal_ccr_maxabs": float(np.max(np.abs(normal))),
        "anomalous_maxabs": float(np.max(np.abs(vals["anomalous_contribution"]))),
        "normal_definition": "U @ U.conj().T - V @ V.conj().T - I",
        "anomalous_definition": "U @ V.T - V @ U.T",
        "U0_gram_defect": float(np.max(np.abs(u0 @ u0.conj().T - np.eye(2)))),
        "pump_cancellation_00": float(pump[0, 0].real),
        "changes_vs_wing2": changes,
        "bands": band_delta,
        "panel_request_counts": counts,
        "timings": timings,
        "total_wall_s": time.perf_counter() - started,
        "compared_with": "wing_factor_2",
        "background_repropagated": False,
        "production_qualification": False,
        "physical_conclusion": False,
        "row_sha256": None,
    }
    meta["row_sha256"] = base.prep.sha256_file(out.with_suffix(".npz"))
    base.prep.write_json_atomic(out, meta)
    print(json.dumps({k: meta[k] for k in (
        "endpoint", "n_nodes", "n_trace", "normal_ccr_maxabs", "anomalous_maxabs",
        "pump_cancellation_00", "total_wall_s")}, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wing-factor", type=int, choices=(1, 2, 4))
    parser.add_argument("--endpoint", choices=("low", "lower", "high"))
    args = parser.parse_args()
    if (args.wing_factor is None) == (args.endpoint is None):
        parser.error("Choose exactly one of --wing-factor or --endpoint")
    if args.endpoint:
        run_endpoint(args.endpoint)
    else:
        run(args.wing_factor)
