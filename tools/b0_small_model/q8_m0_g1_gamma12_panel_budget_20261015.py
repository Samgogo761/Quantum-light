"""Panel-by-panel quadrature budget for the saved Gamma12 output.

Every current panel is replaced by the same rule with twice as many nodes.
The two coarse H2 wings are continued to 128 nodes and checked with a split rule.
Old caches are only read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_local_prepare_20261007 as prep
import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_gamma12_tail_columns_20261014 import (
    BASELINE, contribution, load_rows, match_saved, output_columns,
)
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import gauss_segment, t_gauss_panel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_panel_budget_20261015/panel_budget.json"
BATCH = 800


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def nodes_at(panel, n):
    if panel["kind"] in ("h2_wing", "outer_direct"):
        return gauss_segment(panel["omega_lo"], panel["omega_hi"], int(n))
    if panel["gwid"] is None or panel["t0"] is None or panel["t1"] is None:
        raise AssertionError(f"Panel has no t interval: {panel['kind']} {panel['pole']}")
    return t_gauss_panel(panel["pole"], panel["gwid"], panel["t0"], panel["t1"], int(n))


def contract_frequencies(rows, cache, gamma, pol, phi_of, eta, t0, oms, wts):
    parts_u, parts_v = [], []
    count = len(oms)
    for start in range(0, count, BATCH):
        stop = min(start + BATCH, count)
        u, v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, oms[start:stop], wts[start:stop])
        parts_u.append(u)
        parts_v.append(v)
        print(f"  contracted {stop}/{count}", flush=True)
    return np.concatenate(parts_u, axis=1), np.concatenate(parts_v, axis=1)


def successive(levels):
    out = {}
    keys = list(levels)
    for left, right in zip(keys, keys[1:]):
        out[f"{left}_to_{right}"] = {
            "normal": pack(levels[right][0] - levels[left][0]),
            "anomalous": pack(levels[right][1] - levels[left][1]),
        }
    return out


def wing_ladder(panel, rows, cache, gamma, pol, phi_of, eta, t0):
    levels = {}
    for n in (int(panel["requested"]), 32, 64, 128):
        if n < int(panel["requested"]):
            continue
        om, wt = nodes_at(panel, n)
        if abs(float(np.sum(wt) - (panel["omega_hi"] - panel["omega_lo"]))) > 1.0e-8:
            raise AssertionError(f"Flat-panel weight is not the interval length at n={n}")
        u, v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, om, wt)
        levels[str(n)] = contribution(u, v)
        print(" wing", panel["omega_lo"], n, flush=True)
    # Independent placement: two equal subintervals, 64 nodes each.
    mid = 0.5 * (panel["omega_lo"] + panel["omega_hi"])
    left, right = gauss_segment(panel["omega_lo"], mid, 64), gauss_segment(mid, panel["omega_hi"], 64)
    om = np.concatenate((left[0], right[0]))
    wt = np.concatenate((left[1], right[1]))
    u, v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, om, wt)
    split = contribution(u, v)
    return {
        "pole": panel["pole"], "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"],
        "steps": successive(levels),
        "split_64_plus_64_minus_single_128": {
            "normal": pack(split[0] - levels["128"][0]),
            "anomalous": pack(split[1] - levels["128"][1]),
        },
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    meta = json.loads(BASELINE.with_suffix(".json").read_text(encoding="utf-8"))
    if prep.sha256_file(BASELINE) != meta["row_sha256"]:
        raise ValueError("endpoint_low.npz hash changed")
    rows, cache, t0, _backward = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(base.q.A_STAR)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    panels = []
    oms, wts, counts = wing_quadrature(
        cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
        omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    with np.load(BASELINE, allow_pickle=False) as saved_file:
        saved_oms = np.array(saved_file["oms"])
        saved_wts = np.array(saved_file["wts"])
        saved_u = np.array(saved_file["U"])
        saved_v = np.array(saved_file["V"])
    if not (np.allclose(saved_oms, oms) and np.allclose(saved_wts, wts)):
        raise AssertionError("Rebuilt nodes differ from the saved quadrature")
    covered = np.zeros(len(saved_oms), dtype=bool)
    old = []
    for panel in panels:
        freq_ids, cols = match_saved(saved_oms, saved_wts, panel["omega"], panel["weights"])
        if np.any(covered[freq_ids]):
            raise AssertionError("Two panels claim the same saved frequency")
        covered[freq_ids] = True
        normal, anomalous = contribution(saved_u[:, cols], saved_v[:, cols])
        old.append((freq_ids, normal, anomalous))
    if not np.all(covered):
        raise AssertionError(f"Unassigned frequencies: {int(np.count_nonzero(~covered))}")
    targets = [panel for panel in panels if panel["kind"] == "h2_wing"
               and (abs(panel["omega_lo"] - 0.23032629135) < 1e-8
                    or abs(panel["omega_lo"] - 0.26434881535) < 1e-8)]
    if len(targets) != 2:
        raise AssertionError(f"Expected the two H2 wings, found {len(targets)}")
    print("wing ladders", flush=True)
    ladders = [wing_ladder(panel, rows, cache, gamma, pol, phi_of, eta, t0) for panel in targets]
    known_ladder = next(item for item in ladders if abs(item["omega_lo"] - 0.23032629135) < 1.0e-8)
    known = known_ladder["steps"]["16_to_32"]["normal"]["real"][0][0]
    if abs(known - (-0.0065009832787836785)) > 1.0e-12:
        raise AssertionError(f"Known wing step was not reproduced: {known}")
    new_o, new_w, owners = [], [], []
    for index, panel in enumerate(panels):
        om, wt = nodes_at(panel, 2 * int(panel["requested"]))
        gap = abs(float(np.sum(wt) - panel["weight_sum"]))
        if gap > 1.0e-6:
            raise AssertionError(f"Doubling changed the weight by {gap} at {panel['kind']} {panel['pole']}")
        new_o.append(om)
        new_w.append(wt)
        owners.append(np.full(om.size, index, dtype=int))
    new_o = np.concatenate(new_o)
    new_w = np.concatenate(new_w)
    owners = np.concatenate(owners)
    keys = np.round(new_o, 12)
    if len(np.unique(keys)) != len(keys):
        raise AssertionError("Doubled panels produced a repeated frequency")
    print("doubled frequencies", len(new_o), flush=True)
    new_u, new_v = contract_frequencies(rows, cache, gamma, pol, phi_of, eta, t0, new_o, new_w)
    if new_u.shape[1] != 4 * len(new_o):
        raise AssertionError("Column count does not match four columns per frequency")
    total_n = np.zeros((2, 2), complex)
    total_a = np.zeros((2, 2), complex)
    ledger = []
    by_kind = {}
    for index, panel in enumerate(panels):
        pick = np.flatnonzero(owners == index)
        cols = np.array([4 * int(i) + k for i in pick for k in range(4)], int)
        normal, anomalous = contribution(new_u[:, cols], new_v[:, cols])
        delta_n = normal - old[index][1]
        delta_a = anomalous - old[index][2]
        total_n += delta_n
        total_a += delta_a
        record = {
            "kind": panel["kind"], "pole": panel["pole"],
            "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"],
            "nodes": int(panel["nodes"]), "doubled_nodes": int(pick.size),
            "weight_sum": panel["weight_sum"],
            "normal_change_maxabs": float(np.max(np.abs(delta_n))),
            "anomalous_change_maxabs": float(np.max(np.abs(delta_a))),
            "normal_change": pack(delta_n), "anomalous_change": pack(delta_a),
        }
        ledger.append(record)
        bucket = by_kind.setdefault(panel["kind"], {"count": 0, "normal": np.zeros((2, 2), complex),
                                                    "anomalous": np.zeros((2, 2), complex),
                                                    "above_1e-6": 0, "above_1e-4": 0})
        bucket["count"] += 1
        bucket["normal"] += delta_n
        bucket["anomalous"] += delta_a
        if record["normal_change_maxabs"] > 1.0e-6 or record["anomalous_change_maxabs"] > 1.0e-6:
            bucket["above_1e-6"] += 1
        if record["normal_change_maxabs"] > 1.0e-4 or record["anomalous_change_maxabs"] > 1.0e-4:
            bucket["above_1e-4"] += 1
    notable = [row for row in ledger
               if row["normal_change_maxabs"] > 1.0e-6 or row["anomalous_change_maxabs"] > 1.0e-6]
    notable.sort(key=lambda row: max(row["normal_change_maxabs"], row["anomalous_change_maxabs"]), reverse=True)
    report = {
        "baseline_nodes": int(len(saved_oms)),
        "panel_count": len(panels),
        "request_counts": counts,
        "frequency_partition_complete": True,
        "h2_wing_ladders": ladders,
        "doubling_signed_sum": {"normal": pack(total_n), "anomalous": pack(total_a)},
        "by_kind": {kind: {"count": item["count"], "above_1e-6": item["above_1e-6"],
                           "above_1e-4": item["above_1e-4"],
                           "normal_sum": pack(item["normal"]), "anomalous_sum": pack(item["anomalous"])}
                    for kind, item in by_kind.items()},
        "panels_above_1e-6": notable,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    }
    write_new_json(OUT, report)
    print("signed", float(total_n[0, 0].real), float(np.max(np.abs(total_n))), float(np.max(np.abs(total_a))))
    print("above 1e-6", len(notable))


if __name__ == "__main__":
    main()
