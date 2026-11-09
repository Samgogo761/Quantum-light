#!/usr/bin/env python3
"""Generate a minimal Wannier90-style tb.dat for a two-band hBN model.

The model matches the old script5 conv2 tight-binding convention:

    H11 = +M0
    H22 = -M0
    H12(k) = t * [1 + exp(i k.a1) + exp(-i k.a2)]
    H21(k) = conj(H12(k))

The diagonal position matrix follows the old model's sublattice positions:

    B = (2/3, 1/3, 1/2) in fractional lattice coordinates
    N = (1/3, 2/3, 1/2) in fractional lattice coordinates

Quantum-light reads this tb.dat format with lattice vectors in Angstrom,
Hamiltonian matrix elements in eV, and position matrix elements in Angstrom.
"""

from __future__ import annotations

import cmath
import json
import math
from pathlib import Path


T_EV = 2.33
M0_EV = 1.96
A0_ANG = 2.50
C_ANG = 20.0


def vec_add(*vecs):
    return tuple(sum(v[i] for v in vecs) for i in range(3))


def vec_scale(s, v):
    return tuple(s * x for x in v)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def frac_to_cart(frac, a1, a2, a3):
    return vec_add(vec_scale(frac[0], a1), vec_scale(frac[1], a2), vec_scale(frac[2], a3))


def make_model():
    sqrt3 = math.sqrt(3.0)
    a1 = (0.5 * A0_ANG, -0.5 * sqrt3 * A0_ANG, 0.0)
    a2 = (0.5 * A0_ANG,  0.5 * sqrt3 * A0_ANG, 0.0)
    a3 = (0.0, 0.0, C_ANG)

    b_pos = frac_to_cart((2.0 / 3.0, 1.0 / 3.0, 0.5), a1, a2, a3)
    n_pos = frac_to_cart((1.0 / 3.0, 2.0 / 3.0, 0.5), a1, a2, a3)

    rpts = [
        (0, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (-1, 0, 0),
        (0, 1, 0),
    ]

    h = {r: [[0.0j for _ in range(2)] for _ in range(2)] for r in rpts}
    h[(0, 0, 0)][0][0] = complex(M0_EV, 0.0)
    h[(0, 0, 0)][1][1] = complex(-M0_EV, 0.0)
    h[(0, 0, 0)][0][1] = complex(T_EV, 0.0)
    h[(0, 0, 0)][1][0] = complex(T_EV, 0.0)
    h[(1, 0, 0)][0][1] = complex(T_EV, 0.0)
    h[(0, -1, 0)][0][1] = complex(T_EV, 0.0)
    h[(-1, 0, 0)][1][0] = complex(T_EV, 0.0)
    h[(0, 1, 0)][1][0] = complex(T_EV, 0.0)

    rmat = {r: [[[0.0j for _ in range(3)] for _ in range(2)] for _ in range(2)] for r in rpts}
    for comp in range(3):
        rmat[(0, 0, 0)][0][0][comp] = complex(b_pos[comp], 0.0)
        rmat[(0, 0, 0)][1][1][comp] = complex(n_pos[comp], 0.0)

    return a1, a2, a3, b_pos, n_pos, rpts, h, rmat


def write_tb(path: Path):
    a1, a2, a3, b_pos, n_pos, rpts, h, rmat = make_model()
    lines = []
    lines.append("Generated hBN two-band model: t=2.33 eV, M0=1.96 eV, a0=2.50 Ang")
    for a in (a1, a2, a3):
        lines.append("%20.12f %20.12f %20.12f" % a)
    lines.append("%10d" % 2)
    lines.append("%10d" % len(rpts))
    lines.append(" ".join("%5d" % 1 for _ in rpts))

    for i, r in enumerate(rpts, start=1):
        lines.append("begin H R-point %d" % i)
        lines.append("%5d %5d %5d" % r)
        for n in range(2):
            for m in range(2):
                val = h[r][m][n]
                lines.append("%5d %5d %22.14e %22.14e" % (m + 1, n + 1, val.real, val.imag))

    for i, r in enumerate(rpts, start=1):
        lines.append("begin r R-point %d" % i)
        lines.append("%5d %5d %5d" % r)
        for n in range(2):
            for m in range(2):
                vals = rmat[r][m][n]
                lines.append(
                    "%5d %5d %22.14e %22.14e %22.14e %22.14e %22.14e %22.14e"
                    % (
                        m + 1,
                        n + 1,
                        vals[0].real,
                        vals[0].imag,
                        vals[1].real,
                        vals[1].imag,
                        vals[2].real,
                        vals[2].imag,
                    )
                )

    path.write_text("\n".join(lines) + "\n", encoding="ascii")

    summary = {
        "model": "hBN two-band gapped honeycomb",
        "source_convention": "script5 submod_TBmodel.f90 conv2",
        "t_eV": T_EV,
        "M0_eV": M0_EV,
        "direct_gap_eV": 2.0 * M0_EV,
        "a0_Ang": A0_ANG,
        "c_Ang": C_ANG,
        "lattice_Ang": {"a1": a1, "a2": a2, "a3": a3},
        "sublattice_positions_Ang": {"B": b_pos, "N": n_pos},
        "rpts": rpts,
        "gamma_eigenvalues_eV": gamma_eigs(rpts, h, a1, a2, a3),
        "K_1over3_1over3_eigenvalues_eV": kfrac_eigs((1.0 / 3.0, 1.0 / 3.0, 0.0), rpts, h),
    }
    path.with_name("hBN_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    return summary


def hk_fractional(kfrac, rpts, h):
    mat = [[0.0j for _ in range(2)] for _ in range(2)]
    for r in rpts:
        phase = cmath.exp(2j * math.pi * (kfrac[0] * r[0] + kfrac[1] * r[1] + kfrac[2] * r[2]))
        for m in range(2):
            for n in range(2):
                mat[m][n] += phase * h[r][m][n]
    return mat


def eig2_hermitian(mat):
    a = mat[0][0].real
    d = mat[1][1].real
    b = mat[0][1]
    tr = 0.5 * (a + d)
    rad = math.sqrt((0.5 * (a - d)) ** 2 + abs(b) ** 2)
    return [tr - rad, tr + rad]


def kfrac_eigs(kfrac, rpts, h):
    return eig2_hermitian(hk_fractional(kfrac, rpts, h))


def gamma_eigs(rpts, h, _a1, _a2, _a3):
    return kfrac_eigs((0.0, 0.0, 0.0), rpts, h)


def main():
    out = Path(__file__).resolve().with_name("hBN_model_tb.dat")
    summary = write_tb(out)
    print("Wrote", out)
    print("Direct gap at K fractional (1/3,1/3): %.6f eV" % (
        summary["K_1over3_1over3_eigenvalues_eV"][1]
        - summary["K_1over3_1over3_eigenvalues_eV"][0]
    ))
    print("Gamma eigenvalues:", summary["gamma_eigenvalues_eV"])


if __name__ == "__main__":
    main()
