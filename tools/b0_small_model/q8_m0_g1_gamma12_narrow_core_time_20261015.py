"""Narrow-core correction at the two coarser saved time steps."""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_gamma12_tail_columns_20261014 import contribution, load_rows, output_columns
from q8_m0_g1_gamma12_two_tails_time_20261014 import rows_at
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import t_gauss_panel


def main():
    _rows, cache, t0, _meta = load_rows()
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
    panel = next(item for item in panels if item["kind"] == "core" and abs(item["pole"] - 0.5089945843) < 1e-10)
    for ntpc in (1024, 2048):
        rows = rows_at(ntpc)
        prev = None
        for n, om, wt in (
            (160, panel["omega"], panel["weights"]),
            (1280, *t_gauss_panel(panel["pole"], panel["gwid"], panel["t0"], panel["t1"], 1280)),
        ):
            u, v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, om, wt)
            got = contribution(u, v)
            if prev is not None:
                delta_n = got[0] - prev[0]
                delta_a = got[1] - prev[1]
                print(ntpc)
                print(" N", delta_n[0, 0].real, delta_n[0, 1], delta_n[1, 1].real)
                print(" A", delta_a[0, 1])
            prev = got


if __name__ == "__main__":
    main()
