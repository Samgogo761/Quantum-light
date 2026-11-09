"""Time extrapolation of the refined Gamma12 commutators.

All three saved step sizes use the same 32904-node baseline plus the same
tail replacements. Complex entries are extrapolated before the maximum
modulus is taken.
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
from q8_m0_g1_gamma12_pulse_screen_20261014 import panel_rules, phase_delta, resolvent, times_of_interest
from q8_m0_g1_gamma12_tail_columns_20261014 import (
    BASELINE, contribution, load_rows, match_saved, output_columns,
)
from q8_m0_g1_gamma12_two_tails_time_20261014 import rows_at
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import t_gauss_panel

ROOT = Path(__file__).resolve().parents[2]
LOW = base.OUT
OUT = ROOT / "results/q_line/g1_gamma12_tail_column_20261014/refined_time_extrapolation.json"


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def unpack(block):
    return np.array(block["real"], float) + 1j * np.array(block["imag"], float)


def under_resolved(panel, clock):
    rules = panel_rules(panel)
    if rules is None or panel["kind"] != "sinh_tail":
        return False
    worst = max(abs(phase_delta(*rules, clock[name], lambda _om: 1.0))
                for name in ("one_cycle", "mid_pulse", "pulse_end"))
    return max(worst, abs(phase_delta(*rules, clock["pulse_end"], resolvent(panel)))) > 1.0e-3


def panel_change(rows, cache, gamma, pol, phi_of, eta, t0, panel, saved, old_from_saved):
    new_o, new_w = t_gauss_panel(panel["pole"], panel["gwid"], panel["t0"], panel["t1"], 2 * int(panel["requested"]))
    if abs(float(np.sum(new_w) - panel["weight_sum"])) > 1.0e-8:
        raise AssertionError(f"Weight changed at pole {panel['pole']}")
    new_u, new_v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, new_o, new_w)
    new_n, new_a = contribution(new_u, new_v)
    if old_from_saved:
        _ids, cols = match_saved(saved["oms"], saved["wts"], panel["omega"], panel["weights"])
        old_n, old_a = contribution(saved["U"][:, cols], saved["V"][:, cols])
    else:
        old_u, old_v = output_columns(
            rows, cache, gamma, pol, phi_of, eta, t0, panel["omega"], panel["weights"])
        old_n, old_a = contribution(old_u, old_v)
    return new_n - old_n, new_a - old_a


def floor_from(fine, coarse):
    return fine + (fine - coarse) / 3.0


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows4096, cache, t0, _meta = load_rows()
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
    clock = times_of_interest()
    selected = [panel for panel in panels if under_resolved(panel, clock)]
    if len(selected) != 34:
        raise AssertionError(f"Panel selection changed: {len(selected)}")
    baselines = {}
    for ntpc in (1024, 2048):
        data = json.loads((LOW / f"lowend_time_matrix_ntpc_{ntpc}.json").read_text(encoding="utf-8"))
        if int(data["n_nodes"]) != 32904 or float(data["omega_lo"]) != 1.0e-4:
            raise AssertionError(f"ntpc {ntpc} is not the shared quadrature")
        baselines[ntpc] = {"normal": unpack(data["normal"]), "anomalous": unpack(data["anomalous"])}
    with np.load(BASELINE, allow_pickle=False) as saved_file:
        saved = {key: np.array(saved_file[key]) for key in ("oms", "wts", "U", "V")}
    if not (np.allclose(saved["oms"], oms) and np.allclose(saved["wts"], wts)):
        raise AssertionError("Saved 4096 nodes differ from the rebuilt quadrature")
    base_n = saved["U"] @ saved["U"].conj().T - saved["V"] @ saved["V"].conj().T - np.eye(2)
    base_a = saved["U"] @ saved["V"].T - saved["V"] @ saved["U"].T
    if abs(float(np.max(np.abs(base_n))) - 0.007439260632963851) > 1.0e-12:
        raise AssertionError("4096 baseline residual changed")
    baselines[4096] = {"normal": base_n, "anomalous": base_a}
    changes = {}
    for ntpc, rows in ((1024, rows_at(1024)), (2048, rows_at(2048)), (4096, rows4096)):
        total_n = np.zeros((2, 2), complex)
        total_a = np.zeros((2, 2), complex)
        print("ntpc", ntpc, flush=True)
        for panel in selected:
            dn, da = panel_change(rows, cache, gamma, pol, phi_of, eta, t0, panel, saved,
                                  old_from_saved=(ntpc == 4096))
            total_n += dn
            total_a += da
        changes[ntpc] = {"normal": total_n, "anomalous": total_a}
        print("  change", float(total_n[0, 0].real), float(np.max(np.abs(total_n))), flush=True)
    if abs(float(changes[4096]["normal"][0, 0].real) - (-0.00648327873568471)) > 1.0e-10:
        raise AssertionError("4096 tail sum does not reproduce the published replacement")
    refined = {ntpc: {name: baselines[ntpc][name] + changes[ntpc][name] for name in ("normal", "anomalous")}
               for ntpc in (1024, 2048, 4096)}
    floors = {
        "1024_2048": {name: floor_from(refined[2048][name], refined[1024][name]) for name in ("normal", "anomalous")},
        "2048_4096": {name: floor_from(refined[4096][name], refined[2048][name]) for name in ("normal", "anomalous")},
    }
    shift = {name: floors["2048_4096"][name] - floors["1024_2048"][name] for name in ("normal", "anomalous")}
    report = {
        "quadrature": "32904-node baseline plus the same 34 tail doublings at every step",
        "levels": {str(ntpc): {name: pack(refined[ntpc][name]) for name in ("normal", "anomalous")}
                   for ntpc in (1024, 2048, 4096)},
        "tail_changes": {str(ntpc): {name: pack(changes[ntpc][name]) for name in ("normal", "anomalous")}
                         for ntpc in (1024, 2048, 4096)},
        "richardson": {key: {name: pack(floors[key][name]) for name in ("normal", "anomalous")}
                       for key in floors},
        "floor_shift": {name: pack(shift[name]) for name in shift},
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    }
    write_new_json(OUT, report)
    for ntpc in (1024, 2048, 4096):
        print(ntpc, "RN", report["levels"][str(ntpc)]["normal"]["maxabs"],
              "RA", report["levels"][str(ntpc)]["anomalous"]["maxabs"])
    print("floor", report["richardson"]["2048_4096"]["normal"]["maxabs"],
          report["richardson"]["2048_4096"]["anomalous"]["maxabs"],
          "shift", report["floor_shift"]["normal"]["maxabs"])


if __name__ == "__main__":
    main()
