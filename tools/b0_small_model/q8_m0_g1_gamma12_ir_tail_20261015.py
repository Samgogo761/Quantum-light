"""Infrared edge of the lowest bright-pole tail, on saved fourth-order rows.

The production panel on [1e-4, 0.02] stays the reference. Each lower edge is
integrated either as its own panel below that edge, or by rebuilding
[omega_min, 0.02]. The two do not share nodes. One loose infrared node is
recorded only to explain the earlier single-node figure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_gamma12_column_order_20261015 import CACHE, CYCLES, load_saved, vertices
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import macro_pullback
from q8_m0_g1_gamma12_richardson_full_20261015 import accumulate
from q8_m0_g1_gamma12_tail_columns_20261014 import BASELINE, load_rows, match_saved
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import advance_half_frame, vertices_from_frame
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import gauss_segment, t_gauss_panel
import q8_m0_gamma12_input_wing_20261008 as wing

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_gamma12_time_order_20261015"
POLE = 0.0927933583
CUTOFFS = (1.0e-6, 1.0e-8, 1.0e-10)
NODE_COUNTS = (8, 16, 32, 64)
BASELINES = {
    1024: OUTDIR / "richardson_corrected_macro1024.json",
    2048: OUTDIR / "richardson_macro2048_final.json",
}


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def unpack(block):
    return np.array(block["real"], float) + 1j * np.array(block["imag"], float)


def vertices_progress(sample, geo, prot, nstep, h, label):
    frame = sample["frame"].copy()
    out = []
    for step in range(nstep):
        if step % 4000 == 0:
            print(label, step, flush=True)
        t = step * h
        frame = advance_half_frame(geo, sample["kc"], prot, sample["neel"], frame, t + 0.25 * h, h)
        out.append(vertices_from_frame(
            geo, sample["kc"], prot, sample["neel"], t + 0.5 * h, frame, sample["weights"]))
        frame = advance_half_frame(geo, sample["kc"], prot, sample["neel"], frame, t + 0.75 * h, h)
    return out


def rows_from_vertices(coarse, fine, h, gamma, pol):
    n_macro = len(coarse)
    if len(fine) != 2 * n_macro:
        raise ValueError("Fine vertices must be exactly two per coarse step")
    times = np.empty(3 * n_macro)
    for i in range(n_macro):
        t = i * h
        times[3 * i:3 * i + 3] = (t + 0.25 * h, t + 0.5 * h, t + 0.75 * h)
    kernels, _ = kernel_at_times(gamma, p1_a2_protocol(0.8), times)
    row = np.zeros((4, 2 * int(np.asarray(fine[0][0]).shape[1]) + 2), complex)
    force, time, step = [], [], []
    for i in range(n_macro - 1, -1, -1):
        if i % 4000 == 0:
            print("pullback", i, flush=True)
        piece = {
            "t": i * h, "H": h,
            "uc": coarse[i][0], "dc": np.real(coarse[i][1]),
            "u1": np.asarray(fine[2 * i][0]), "d1": np.real(fine[2 * i][1]),
            "u2": np.asarray(fine[2 * i + 1][0]), "d2": np.real(fine[2 * i + 1][1]),
            "kc": kernels[3 * i + 1], "k1": kernels[3 * i], "k2": kernels[3 * i + 2],
        }
        row, force_c, force_f1, force_f2 = macro_pullback(row, piece, gamma, pol)
        for item, when, width in (
            (force_c, i * h, h),
            (force_f1, i * h, 0.5 * h),
            (force_f2, i * h + 0.5 * h, 0.5 * h),
        ):
            force.append(item)
            time.append(when)
            step.append(width)
    return row, np.stack(force), np.asarray(time, float), np.asarray(step, float)


def save_rows(path, row_d, force_d, time_d, step_d, row_u, force_u, time_u, step_u):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez(path, row_d=row_d, force_d=force_d, time_d=time_d, step_d=step_d,
             row_u=row_u, force_u=force_u, time_u=time_u, step_u=step_u)


def load_adjoint(path):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: np.array(saved[key]) for key in saved.files}
    def forces(prefix):
        return list(zip(arrays[f"force_{prefix}"], arrays[f"time_{prefix}"], arrays[f"step_{prefix}"]))
    return arrays["row_d"], forces("d"), arrays["row_u"], forces("u")


def build_level(macro, sample, geo, prot, quiet, gamma, pol, h_saved):
    path = OUTDIR / f"adjoint_rows_macro{macro}.npz"
    if path.exists():
        print("load rows", macro, flush=True)
        return load_adjoint(path)
    h = h_saved * (1024 / macro)
    n_macro = CYCLES * macro
    print("rows", macro, "driven", flush=True)
    if macro == 1024:
        coarse = load_saved("driven", n_macro)
    else:
        coarse = vertices_progress(sample, geo, prot, n_macro, h, "coarse driven")
    fine = vertices_progress(sample, geo, prot, 2 * n_macro, 0.5 * h, "fine driven")
    row_d, force_d, time_d, step_d = rows_from_vertices(coarse, fine, h, gamma, pol)
    del coarse, fine
    print("rows", macro, "undriven", flush=True)
    if macro == 1024:
        coarse = load_saved("undriven", n_macro)
    else:
        coarse = vertices_progress(sample, geo, quiet, n_macro, h, "coarse undriven")
    fine = vertices_progress(sample, geo, quiet, 2 * n_macro, 0.5 * h, "fine undriven")
    row_u, force_u, time_u, step_u = rows_from_vertices(coarse, fine, h, gamma, pol)
    del coarse, fine
    save_rows(path, row_d, force_d, time_d, step_d, row_u, force_u, time_u, step_u)
    return (row_d, list(zip(force_d, time_d, step_d)), row_u, list(zip(force_u, time_u, step_u)))


def contract(rows, cache, gamma, pol, phi_of, eta, oms, wts):
    oms = np.asarray(oms, float)
    wts = np.asarray(wts, float)
    parts = {key: [] for key in ("u", "v", "u0", "free")}
    for start in range(0, oms.size, 400):
        stop = min(start + 400, oms.size)
        ph, u0 = base.generated_photon_block(
            cache, gamma, oms[start:stop], wts[start:stop], pol, phi_of, eta)
        du, dv = accumulate(rows[0], rows[1], ph)
        uu, uv = accumulate(rows[2], rows[3], ph)
        parts["u"].append(u0 + du - uu)
        parts["v"].append(dv - uv)
        parts["u0"].append(u0)
        parts["free"].append(ph.free_A0)
    return tuple(np.concatenate(parts[key], 1) for key in ("u", "v", "u0", "free"))


def gram(u, v, u0):
    normal = u @ u.conj().T - v @ v.conj().T
    return {"normal": normal, "static_U0": u0 @ u0.conj().T,
            "anomalous": u @ v.T - v @ u.T}


def port_detail(packed, oms):
    u, v, u0, free = packed
    nfreq = int(np.asarray(oms).size)
    left = np.array([4 * i + k for i in range(nfreq) for k in (0, 1)], int)
    right = np.array([4 * i + k for i in range(nfreq) for k in (2, 3)], int)
    out = {}
    for name, idx in (("all", slice(None)), ("photon_L", left), ("photon_R", right)):
        pieces = gram(u[:, idx], v[:, idx], u0[:, idx])
        out[name] = {key: pack(value) for key, value in pieces.items()}
    omega = np.asarray(oms, float)
    out["scales"] = {
        "min_omega": float(np.min(omega)),
        "max_abs_free_A": float(np.max(np.abs(free))),
        "free_A_times_sqrt_omega_max": float(np.max(np.abs(free) * np.sqrt(np.repeat(omega, 4)))),
        "max_abs_U0": float(np.max(np.abs(u0))),
        "max_abs_U": float(np.max(np.abs(u))),
        "max_abs_V": float(np.max(np.abs(v))),
    }
    return out


def log_gauss(a, b, n):
    x, w = np.polynomial.legendre.leggauss(int(n))
    la, lb = np.log(float(a)), np.log(float(b))
    u = 0.5 * (lb - la) * x + 0.5 * (la + lb)
    om = np.exp(u)
    wt = 0.5 * (lb - la) * w * om
    return om, wt


def rules(panel, lo, hi):
    ta = float(np.arcsinh((lo - panel["pole"]) / panel["gwid"]))
    tb = float(np.arcsinh((hi - panel["pole"]) / panel["gwid"]))
    return {
        "t_gauss": lambda n: t_gauss_panel(panel["pole"], panel["gwid"], ta, tb, n),
        "omega_gauss": lambda n: gauss_segment(lo, hi, n),
        "log_gauss": lambda n: log_gauss(lo, hi, n),
    }


def accept_weight(om, wt, span):
    if abs(float(np.sum(wt)) - span) > 1.0e-9:
        raise AssertionError(f"Quadrature weight {float(np.sum(wt))} is not the interval {span}")
    return om, wt


def matrix_from(detail, key="normal"):
    return unpack(detail["all"][key])


def study(rows, cache, gamma, pol, phi_of, eta, panel, saved_oms, saved_wts, baseline):
    old_ids, _cols = match_saved(saved_oms, saved_wts, panel["omega"], panel["weights"])
    old = contract(rows, cache, gamma, pol, phi_of, eta, panel["omega"], panel["weights"])
    old_detail = port_detail(old, panel["omega"])
    report = {"old_panel": {
        "nodes": int(panel["nodes"]), "weight": panel["weight_sum"],
        "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"],
        **old_detail,
    }, "cutoffs": {}}
    saved_keys = {round(float(o), 12) for o in saved_oms}
    old_keys = {round(float(o), 12) for o in panel["omega"]}
    for lo in CUTOFFS:
        span = float(panel["omega_lo"] - lo)
        added = {}
        for name, rule in rules(panel, lo, panel["omega_lo"]).items():
            added[name] = {}
            prev = None
            for n in NODE_COUNTS:
                om, wt = accept_weight(*rule(n), span)
                if np.any(om >= panel["omega_lo"]) or np.any(om <= lo):
                    raise AssertionError("Added interval left its open range")
                if any(round(float(o), 12) in saved_keys for o in om):
                    raise AssertionError("Added node collides with a saved frequency")
                detail = port_detail(contract(rows, cache, gamma, pol, phi_of, eta, om, wt), om)
                entry = {"nodes": int(om.size), "weight": float(np.sum(wt)), **detail}
                if prev is not None:
                    entry["step_from_previous"] = pack(matrix_from(detail) - matrix_from(prev))
                added[name][str(n)] = entry
                prev = detail
                print(lo, name, n, float(matrix_from(detail)[0, 0].real), flush=True)
        rebuilt = {}
        for n in (24, 48, 96):
            om, wt = accept_weight(*rules(panel, lo, panel["omega_hi"])["t_gauss"](n),
                                   float(panel["omega_hi"] - lo))
            for value in om:
                key = round(float(value), 12)
                if key in saved_keys and key not in old_keys:
                    raise AssertionError(f"Rebuilt node {value} hits a kept frequency")
            detail = port_detail(contract(rows, cache, gamma, pol, phi_of, eta, om, wt), om)
            delta = matrix_from(detail) - matrix_from(old_detail)
            rebuilt[str(n)] = {
                "nodes": int(om.size), "weight": float(np.sum(wt)),
                "delta_normal": pack(delta),
                "delta_anomalous": pack(matrix_from(detail, "anomalous") - matrix_from(old_detail, "anomalous")),
                "total_normal": pack(baseline["normal"] + delta),
                "total_anomalous": pack(baseline["anomalous"] + matrix_from(detail, "anomalous")
                                        - matrix_from(old_detail, "anomalous")),
                **detail,
            }
            print("rebuild", lo, n, float(delta[0, 0].real), flush=True)
        finest = added["log_gauss"]["64"]
        cross = matrix_from(rebuilt["96"]) - (
            matrix_from(old_detail) + matrix_from(finest))
        # Rebuilt gram should equal old gram plus the missing piece only if the
        # shared interval was not rediscretized. It was, so this difference is
        # that rediscretization, not a second copy of the infrared piece.
        report["cutoffs"][f"{lo:.0e}"] = {
            "missing_bandwidth": span,
            "added_below_edge": added,
            "rebuilt_whole_tail_t_gauss": rebuilt,
            "rebuilt96_minus_old_minus_log64": pack(
                (matrix_from(rebuilt["96"]) - matrix_from(finest)) - matrix_from(old_detail)),
            "cross_maxabs": float(np.max(np.abs(cross))),
        }
    om, wt = accept_weight(*rules(panel, 1.0e-8, panel["omega_hi"])["t_gauss"](24),
                           float(panel["omega_hi"] - 1.0e-8))
    lone = np.asarray(om) < panel["omega_lo"]
    lone_detail = port_detail(
        contract(rows, cache, gamma, pol, phi_of, eta, om[lone], wt[lone]), om[lone])
    report["single_node_inside_extended_t_panel"] = {
        "nodes": int(np.sum(lone)),
        "weight": float(np.sum(wt[lone])),
        "missing_bandwidth": float(panel["omega_lo"] - 1.0e-8),
        "weight_exceeds_missing_bandwidth": float(np.sum(wt[lone]) - (panel["omega_lo"] - 1.0e-8)),
        **lone_detail,
    }
    report["unchanged_frequency_ids"] = len(old_ids)
    return report


def main():
    _midpoint_rows, cache, t0, _meta = load_rows()
    if abs(t0) > 0:
        raise AssertionError(t0)
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    quiet = p1_a2_protocol(0.0)
    h_saved = float(np.load(CACHE / "h.npy"))
    if abs(h_saved * 1024.0 - float(prot.period)) > 1e-8:
        raise AssertionError("Saved step does not match the production period")
    geo = build_geometry()
    sample, _inv = grid_sample(geo, 12, 0.0, neel=1)
    check = vertices(sample, geo, prot, 2, h_saved)
    if np.max(np.abs(check[0][0] - np.load(CACHE / "driven_u.npy", mmap_mode="r")[0])) > 1e-12:
        raise AssertionError("Regenerated vertices do not match the saved production vertices")
    panels = []
    wing.wing_quadrature(cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
                         omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    panel = next(item for item in panels
                 if item["kind"] == "sinh_tail" and abs(item["pole"] - POLE) < 1e-10
                 and item["omega_hi"] < 0.05)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    with np.load(BASELINE, allow_pickle=False) as saved:
        saved_oms, saved_wts = np.array(saved["oms"]), np.array(saved["wts"])
    for macro in (1024, 2048):
        out = OUTDIR / f"richardson_ir_tail_macro{macro}.json"
        if out.exists():
            print("already published", macro, flush=True)
            continue
        rows = build_level(macro, sample, geo, prot, quiet, gamma, pol, h_saved)
        published = json.loads(BASELINES[macro].read_text(encoding="utf-8"))
        block = published["final_nine_panels"] if macro == 1024 else published["final_nine"]
        baseline = {kind: unpack(block[kind]) for kind in ("normal", "anomalous")}
        print("study", macro, flush=True)
        body = study(rows, cache, gamma, pol, phi_of, eta, panel, saved_oms, saved_wts, baseline)
        write_new_json(out, {
            "macro_ntpc": macro,
            "pole": panel["pole"], "gwid": panel["gwid"],
            "baseline_final_nine": {kind: pack(baseline[kind]) for kind in baseline},
            **body,
            "stored_midpoint_rows_used": False,
            "production_G1_pass": False,
            "provenance_sha256": provenance([__file__]),
        })
        print("published", macro, flush=True)


if __name__ == "__main__":
    main()
