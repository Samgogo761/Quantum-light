"""Second doubling, and three time steps, for the H2 wings that move."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_gamma12_tail_columns_20261014 import contribution, load_rows, output_columns
from q8_m0_g1_gamma12_two_tails_time_20261014 import rows_at
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import gauss_segment

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_tail_column_20261014/h2_wing_023_time.json"
TARGET = 0.23032629135


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


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
    wing_quadrature(cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
                    omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    panel = next(item for item in panels if item["kind"] == "h2_wing" and abs(item["omega_lo"] - TARGET) < 1e-8)
    levels = {}
    for ntpc, rows in ((1024, rows_at(1024)), (2048, rows_at(2048)), (4096, rows4096)):
        grams = {}
        prev = None
        for n in (16, 32, 64):
            om, wt = gauss_segment(panel["omega_lo"], panel["omega_hi"], n)
            u, v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, om, wt)
            got = contribution(u, v)
            entry = {"normal": pack(got[0]), "anomalous": pack(got[1])}
            if prev is not None:
                entry["normal_step"] = pack(got[0] - prev[0])
                entry["anomalous_step"] = pack(got[1] - prev[1])
                print(ntpc, n, float(np.max(np.abs(got[0] - prev[0]))), flush=True)
            grams[str(n)] = entry
            prev = got
        levels[str(ntpc)] = grams
    write_new_json(OUT, {
        "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"], "pole": panel["pole"],
        "levels": levels, "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })


if __name__ == "__main__":
    main()
