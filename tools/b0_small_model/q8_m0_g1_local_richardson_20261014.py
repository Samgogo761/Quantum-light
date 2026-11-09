"""Local Richardson step for the two-k diagnostic. Production midpoint stays as it is.

One macro step runs the existing midpoint once at H and twice at H/2, from the
same incoming state. The state and the H2 increments are combined as
(4*fine - coarse)/3. Negative substeps are not used.
"""
from __future__ import annotations

import numpy as np

from q8_m0_continuum_columns_20261004 import ColumnState, current, midpoint_step
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times


def _blend(fine, coarse):
    return ColumnState(
        (4.0 * fine.b - coarse.b) / 3.0,
        (4.0 * fine.bdag - coarse.bdag) / 3.0,
        (4.0 * fine.A - coarse.A) / 3.0,
    )


def _midpoint(prep, state, u, d, h, t, k_mid):
    prev = state
    free = prep["free"] * (np.exp(-1j * prep["omega"] * (t + h)) - np.exp(-1j * prep["omega"] * t))
    dreal = np.real(d)
    new = midpoint_step(state, u, dreal, h, prep["gamma"], free_A_increment=free)
    mid = ColumnState(0.5 * (prev.b + new.b), 0.5 * (prev.bdag + new.bdag),
                      0.5 * (prev.A + new.A))
    src = prep["pol"] @ current(mid, u, dreal)
    return new, h * k_mid * src, h * k_mid * np.conj(src)


def propagate_local_richardson(prep, verts_coarse, verts_fine, h_macro):
    if len(verts_fine) != 2 * len(verts_coarse):
        raise ValueError("Fine vertices must be exactly two per coarse step.")
    state = prep["initial"].copy()
    n = len(verts_coarse)
    times = np.empty(3 * n)
    for i in range(n):
        t = i * h_macro
        times[3 * i:3 * i + 3] = (t + 0.25 * h_macro, t + 0.5 * h_macro, t + 0.75 * h_macro)
    kernels, _ = kernel_at_times(prep["gamma"], prep["drive"], times)
    acc_u = np.zeros((2, state.b.shape[1]), complex)
    acc_v = np.zeros_like(acc_u)
    half = 0.5 * h_macro
    for i, (u_c, d_c) in enumerate(verts_coarse):
        t = i * h_macro
        k_f1, k_c, k_f2 = kernels[3 * i:3 * i + 3]
        coarse, du_c, dv_c = _midpoint(prep, state, u_c, d_c, h_macro, t, k_c)
        u1, d1 = verts_fine[2 * i]
        u2, d2 = verts_fine[2 * i + 1]
        mid, du1, dv1 = _midpoint(prep, state, u1, d1, half, t, k_f1)
        fine, du2, dv2 = _midpoint(prep, mid, u2, d2, half, t + half, k_f2)
        state = _blend(fine, coarse)
        acc_u += (4.0 * (du1 + du2) - du_c) / 3.0
        acc_v += (4.0 * (dv1 + dv2) - dv_c) / 3.0
    return acc_u, acc_v
