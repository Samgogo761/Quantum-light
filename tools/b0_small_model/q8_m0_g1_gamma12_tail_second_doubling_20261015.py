"""Second doubling of the four tails still above 1e-6 after one doubling."""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_gamma12_tail_columns_20261014 import contribution, load_rows, output_columns
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import t_gauss_panel

POLES = (
    (0.8597387955, 0.73332602275),
    (1.437893666, 1.33856321465),
    (1.2392327633, 1.24269383715),
    (0.60691325, 0.6318625001),
)


def main():
    rows, cache, t0, _meta = load_rows()
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
    for pole, omega_lo in POLES:
        hits = [item for item in panels if item["kind"] == "sinh_tail"
                and abs(item["pole"] - pole) < 1e-8 and abs(item["omega_lo"] - omega_lo) < 1e-8]
        if len(hits) != 1:
            raise AssertionError(f"{pole}: {len(hits)} tails")
        panel = hits[0]
        prev = None
        for factor in (2, 4):
            om, wt = t_gauss_panel(panel["pole"], panel["gwid"], panel["t0"], panel["t1"], factor * int(panel["requested"]))
            u, v = output_columns(rows, cache, gamma, pol, phi_of, eta, t0, om, wt)
            got = contribution(u, v)
            if prev is not None:
                delta = got[0] - prev[0]
                print(pole, factor, float(delta[0, 0].real), float(np.max(np.abs(delta))), float(np.max(np.abs(got[1] - prev[1]))), flush=True)
            prev = got


if __name__ == "__main__":
    main()
