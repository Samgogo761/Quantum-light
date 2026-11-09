"""Replace one Gamma12 tail inside the saved 4096-step output.

Old columns come from endpoint_low.npz. New columns are contracted with the
saved transpose rows. Other columns stay. The file hashes of that baseline
and of the backward rows are checked. The current code is accepted only if
it reproduces one saved column from the same panel.
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
import q8_m0_output_transpose_20261008 as transpose
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_g1_gamma12_pulse_screen_20261014 import (
    panel_rules, phase_delta, resolvent, times_of_interest,
)
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import t_gauss_panel

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007"
BACK = ROOT / "results/q_line/gamma12_output_transpose_20261008/ntpc_4096"
BASELINE = base.OUT / "endpoint_low.npz"
SCREEN = None
OUT = ROOT / "results/q_line/g1_gamma12_tail_column_20261014/uv_tail_replacement.json"


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def contribution(u, v):
    normal = u @ u.conj().T - v @ v.conj().T
    anomalous = u @ v.T - v @ u.T
    return normal, anomalous


def load_rows():
    meta = json.loads((BACK / "backward.json").read_text(encoding="utf-8"))
    path = BACK / "backward.npz"
    if prep.sha256_file(path) != meta["sha256"]:
        raise ValueError("Backward row file hash does not match its record")
    info = json.loads((base.REFINED / "prepared.json").read_text(encoding="utf-8"))
    rows = {}
    with np.load(path, allow_pickle=False) as z:
        for label in ("driven", "undriven"):
            row0, forcing = z[label + "_row0"], z[label + "_forcing"]
            if forcing.shape != (info["nstep"], 4, 2) or row0.shape != (4, 4610):
                raise ValueError("Unexpected backward-row shape")
            rows[label] = transpose.BackwardRows(
                row0, forcing, 2, int(info["nstep"]), float(info["h"]),
                meta["timing_s"][label])
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    needed = ("gaps", "u0", "D0", "gamma", "pol")
    cache = {}
    for name in needed:
        record = manifest["files"][name]
        file = SOURCE / record["path"]
        if prep.sha256_file(file) != record["sha256"]:
            raise ValueError(f"Frozen cache hash changed: {name}")
        cache[name] = np.load(file, allow_pickle=False)
    return rows, cache, float(manifest["time"].get("t0", 0.0)), meta


def output_columns(rows, cache, gamma, pol, phi_of, eta, t0, oms, wts):
    ph, u0 = base.generated_photon_block(cache, gamma, oms, wts, pol, phi_of, eta)
    got, _timing = base._contract(
        rows, (ph.initial.b, ph.initial.bdag, ph.initial.A),
        ph.omega, ph.free_A0, t0, 64, 2048)
    du = got["driven_raw_u"] - got["undriven_raw_u"]
    v = got["driven_raw_v"] - got["undriven_raw_v"]
    return u0 + du, v


def match_saved(saved_oms, saved_wts, oms, wts):
    index = {round(float(o), 12): i for i, o in enumerate(saved_oms)}
    freq_ids = []
    for om, wt in zip(oms, wts):
        i = index.get(round(float(om), 12))
        if i is None:
            raise AssertionError(f"Panel frequency {om} is absent from the saved quadrature")
        if abs(float(saved_wts[i] - wt)) > 1e-8:
            raise AssertionError(f"Saved weight at {om} is not this panel alone")
        freq_ids.append(i)
    if len(set(freq_ids)) != len(freq_ids):
        raise AssertionError("Panel frequencies are duplicated in the saved set")
    columns = np.array([4 * i + k for i in freq_ids for k in range(4)], int)
    return freq_ids, columns


def refined_nodes(panel):
    n = int(panel["requested"])
    om, wt = t_gauss_panel(panel["pole"], panel["gwid"], panel["t0"], panel["t1"], 2 * n)
    if abs(float(np.sum(wt) - panel["weight_sum"])) > 1e-8:
        raise AssertionError("Refined tail changed the integrated weight")
    return om, wt


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    meta = json.loads(BASELINE.with_suffix(".json").read_text(encoding="utf-8"))
    if prep.sha256_file(BASELINE) != meta["row_sha256"]:
        raise ValueError("endpoint_low.npz no longer matches its published hash")
    if abs(float(meta["normal_ccr_maxabs"]) - 0.007439260632963851) > 1e-15:
        raise ValueError("Baseline is not the recorded 0.007439 residual")
    rows, cache, t0, backward_meta = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(base.q.A_STAR)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    panels = []
    oms, wts, _counts = wing_quadrature(
        cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
        omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    if len(oms) != int(meta["n_nodes"]):
        raise AssertionError(f"Rebuilt quadrature has {len(oms)} nodes, baseline has {meta['n_nodes']}")
    clock = times_of_interest()

    def under_resolved(panel):
        rules = panel_rules(panel)
        if rules is None:
            return False
        worst = max(abs(phase_delta(*rules, clock[name], lambda _om: 1.0))
                    for name in ("one_cycle", "mid_pulse", "pulse_end"))
        worst = max(worst, abs(phase_delta(*rules, clock["pulse_end"], resolvent(panel))))
        return worst > 1.0e-3

    selected = [panel for panel in panels if panel["kind"] == "sinh_tail" and under_resolved(panel)]
    print("selected tails", len(selected), flush=True)
    if not any(abs(panel["pole"] - 7.6246211251) < 1e-8 and panel["omega_hi"] > 11 for panel in selected):
        raise AssertionError("The recorded ultraviolet tail was not selected")
    with np.load(BASELINE, allow_pickle=False) as saved:
        saved_oms = np.array(saved["oms"])
        saved_wts = np.array(saved["wts"])
        saved_u = np.array(saved["U"])
        saved_v = np.array(saved["V"])
        if not np.allclose(saved_oms, oms) or not np.allclose(saved_wts, wts):
            raise AssertionError("Rebuilt nodes differ from the saved baseline")
        probe = next(panel for panel in selected if abs(panel["pole"] - 7.6246211251) < 1e-8)
        freq_ids, columns = match_saved(saved_oms, saved_wts, probe["omega"], probe["weights"])
        got_u, got_v = output_columns(
            rows, cache, gamma, pol, phi_of, eta, t0,
            probe["omega"][:1], probe["weights"][:1])
        gate = float(max(np.max(np.abs(got_u - saved_u[:, columns[:4]])),
                         np.max(np.abs(got_v - saved_v[:, columns[:4]]))))
        print("reproduction gate", gate, flush=True)
        if gate > 1e-8:
            write_new_json(OUT, {"reproduction_gate_maxabs": gate, "accepted": False,
                                 "production_G1_pass": False,
                                 "provenance_sha256": provenance([__file__])})
            raise SystemExit("Saved column was not reproduced; no panel replacement published")
        claimed = []
        for panel in selected:
            _ids, _cols = match_saved(saved_oms, saved_wts, panel["omega"], panel["weights"])
            claimed.extend(_ids)
        if len(claimed) != len(set(claimed)):
            raise AssertionError("Selected tails share a saved frequency")
        removed = set(claimed)
        saved_keys = {round(float(o), 12): i for i, o in enumerate(saved_oms)}
        results = []
        total_n = np.zeros((2, 2), complex)
        total_a = np.zeros((2, 2), complex)
        for panel in selected:
            _old_ids, old_cols = match_saved(saved_oms, saved_wts, panel["omega"], panel["weights"])
            new_o, new_w = refined_nodes(panel)
            for om in new_o:
                key = round(float(om), 12)
                owner = saved_keys.get(key)
                if owner is not None and owner not in removed:
                    raise AssertionError(f"Refined node {om} collides with a kept frequency")
            new_u, new_v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, new_o, new_w)
            old_n, old_a = contribution(saved_u[:, old_cols], saved_v[:, old_cols])
            new_n, new_a = contribution(new_u, new_v)
            dn, da = new_n - old_n, new_a - old_a
            total_n += dn
            total_a += da
            results.append({
                "pole": panel["pole"], "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"],
                "old_nodes": int(panel["nodes"]), "new_nodes": int(new_o.size),
                "weight": panel["weight_sum"],
                "normal_change": pack(dn), "anomalous_change": pack(da),
            })
            print("panel", panel["pole"], "dN", float(np.max(np.abs(dn))),
                  "dA", float(np.max(np.abs(da))), flush=True)
        base_n = saved_u @ saved_u.conj().T - saved_v @ saved_v.conj().T - np.eye(2)
        base_a = saved_u @ saved_v.T - saved_v @ saved_u.T
    results.sort(key=lambda row: row["normal_change"]["maxabs"], reverse=True)
    report = {
        "baseline": "endpoint_low.npz",
        "baseline_normal_maxabs": float(np.max(np.abs(base_n))),
        "baseline_anomalous_maxabs": float(np.max(np.abs(base_a))),
        "reproduction_gate_maxabs": gate,
        "backward_sha256": backward_meta["sha256"],
        "panels": results,
        "combined_normal_change": pack(total_n),
        "combined_anomalous_change": pack(total_a),
        "residual_after_combined_replacement": {
            "normal": pack(base_n + total_n),
            "anomalous": pack(base_a + total_a),
        },
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    }
    write_new_json(OUT, report)
    print("combined", report["combined_normal_change"]["maxabs"],
          report["residual_after_combined_replacement"]["normal"]["maxabs"])


if __name__ == "__main__":
    main()
