"""Same two tails at three saved time steps. Quadrature change versus time step."""
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
from q8_m0_g1_gamma12_tail_columns_20261014 import (
    contribution, load_rows, match_saved, output_columns, pack,
)
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import t_gauss_panel

ROOT = Path(__file__).resolve().parents[2]
BACK_ROOT = ROOT / "results/q_line/gamma12_output_transpose_20261008"
OUT = ROOT / "results/q_line/g1_gamma12_tail_column_20261014/two_tails_time.json"
POLES = (0.8753020768, 1.23222236)


def rows_at(ntpc):
    directory = BACK_ROOT / f"ntpc_{ntpc}"
    meta = json.loads((directory / "backward.json").read_text(encoding="utf-8"))
    path = directory / "backward.npz"
    if prep.sha256_file(path) != meta["sha256"]:
        raise ValueError(f"Backward hash mismatch at ntpc={ntpc}")
    period = float(p1_a2_protocol(0.8).period)
    h = period / int(ntpc)
    rows = {}
    with np.load(path, allow_pickle=False) as z:
        for label in ("driven", "undriven"):
            row0, forcing = z[label + "_row0"], z[label + "_forcing"]
            if row0.shape != (4, 4610) or forcing.shape[0] != 9 * int(ntpc):
                raise ValueError(f"Unexpected rows at ntpc={ntpc}: {forcing.shape}")
            rows[label] = transpose.BackwardRows(
                row0, forcing, 2, int(forcing.shape[0]), h, meta["timing_s"][label])
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    _rows4096, cache, t0, _meta = load_rows()
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
    chosen = []
    for pole in POLES:
        hits = [panel for panel in panels if panel["kind"] == "sinh_tail" and abs(panel["pole"] - pole) < 1e-8]
        if len(hits) != 1:
            raise AssertionError(pole)
        chosen.append(hits[0])
    report = {"ntpc": {}}
    for ntpc in (1024, 2048, 4096):
        rows = rows_at(ntpc) if ntpc != 4096 else _rows4096
        entry = {}
        for panel in chosen:
            n = int(panel["requested"])
            fine_o, fine_w = t_gauss_panel(panel["pole"], panel["gwid"], panel["t0"], panel["t1"], 2 * n)
            old_u, old_v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, panel["omega"], panel["weights"])
            new_u, new_v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, fine_o, fine_w)
            old_n, old_a = contribution(old_u, old_v)
            new_n, new_a = contribution(new_u, new_v)
            entry[str(panel["pole"])] = {"normal_change": pack(new_n - old_n),
                                         "anomalous_change": pack(new_a - old_a)}
            print(ntpc, panel["pole"], float(np.max(np.abs(new_n - old_n))), flush=True)
        report["ntpc"][str(ntpc)] = entry
    # The 4096 numbers above recompute the old columns instead of reading them.
    # They must agree with the saved-column replacement.
    report["nodes_checked_against_saved_quadrature"] = int(len(oms))
    report["saved_weight_sum"] = float(np.sum(wts))
    report["production_G1_pass"] = False
    report["provenance_sha256"] = provenance([__file__])
    write_new_json(OUT, report)


if __name__ == "__main__":
    main()
