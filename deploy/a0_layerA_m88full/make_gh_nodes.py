#!/usr/bin/env python3
"""Generate Gauss-Hermite Q-node manifests matching src/mod_quantum_light.f90.

Precision policy (CEP-π / absolute exactness):
  - All floats written as ES25.17 (not 16.8e).
  - Re(alpha), Im(alpha) are the sole source; I, phi derived from alpha.
  - Antipodes are enforced exactly before write:
        alpha_bar = -alpha,  w_bar = w,
        I_bar = I(alpha_bar),  phi_bar = mod(phi + pi, 2 pi).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

PI = math.pi
TWOPI = 2.0 * PI
ALPHA0_TOL = 1.0e-14

GH = {
    3: (
        np.array([-1.224744871391589, 0.0, 1.224744871391589]),
        np.array([0.2954089751509193, 1.181635900603677, 0.2954089751509193]),
    ),
    5: (
        np.array([-2.020182870456086, -0.9585724646138185, 0.0, 0.9585724646138185, 2.020182870456086]),
        np.array([0.01995324205904591, 0.3936193231522412, 0.9453087204829419, 0.3936193231522412, 0.01995324205904591]),
    ),
    7: (
        np.array([-2.651961356835233, -1.673551628767471, -0.8162878828589647, 0.0,
                  0.8162878828589647, 1.673551628767471, 2.651961356835233]),
        np.array([9.717812450995194e-4, 5.450729697312067e-2, 0.4256072526101278, 0.8102646175568073,
                  0.4256072526101278, 5.450729697312067e-2, 9.717812450995194e-4]),
    ),
}


def build_VQ_sv(r: float, theta_s: float):
    vx = 0.5 * (1.0 + math.exp(-2.0 * r))
    vp = 0.5 * (1.0 + math.exp(+2.0 * r))
    c = math.cos(0.5 * theta_s)
    s = math.sin(0.5 * theta_s)
    rot = np.array([[c, -s], [s, c]])
    Vdiag = np.diag([vx, vp])
    return rot @ Vdiag @ rot.T


def chol2(A):
    a11 = A[0, 0]
    a21 = 0.5 * (A[1, 0] + A[0, 1])
    a22 = A[1, 1]
    L = np.zeros((2, 2))
    L[0, 0] = math.sqrt(a11)
    L[1, 0] = a21 / L[0, 0]
    L[1, 1] = math.sqrt(a22 - L[1, 0] ** 2)
    return L


def alpha_to_drive(alpha: complex, I_bar: float, nbar: float):
    """Derive (I, phi) from alpha only (Fortran qlight_alpha_to_drive)."""
    I_mean = 2.0 * I_bar
    eabs2 = nbar + 1.0
    abs2 = abs(alpha) ** 2
    I = I_mean * abs2 / eabs2
    if abs2 <= ALPHA0_TOL * ALPHA0_TOL:
        return 0.0, 0.0
    phi = math.atan2(alpha.imag, alpha.real)
    if phi < 0.0:
        phi += TWOPI
    return I, phi


def _alpha_key(alpha: complex) -> tuple[float, float]:
    return (round(alpha.real, 12), round(alpha.imag, 12))


def enforce_antipodes(rows: list[tuple], I_bar: float, nbar: float) -> list[tuple]:
    """Force exact ᾱ=-α, w̄=w, and recompute I,φ from alpha."""
    # rows: (id, w, re, im, I, phi) — I/phi ignored and recomputed
    by_key: dict[tuple[float, float], int] = {}
    alphas = [complex(r[2], r[3]) for r in rows]
    weights = [float(r[1]) for r in rows]
    ids = [int(r[0]) for r in rows]

    for i, a in enumerate(alphas):
        key = _alpha_key(a)
        if key in by_key:
            raise ValueError(f"duplicate alpha key {key} for ids {by_key[key]}/{ids[i]}")
        by_key[key] = i

    used = set()
    out_alpha = list(alphas)
    out_w = list(weights)
    for i, a in enumerate(alphas):
        if i in used:
            continue
        if abs(a) <= ALPHA0_TOL:
            out_alpha[i] = 0.0 + 0.0j
            used.add(i)
            continue
        j = by_key.get(_alpha_key(-a))
        if j is None:
            raise ValueError(f"no antipode for id={ids[i]} alpha={a}")
        if j == i:
            raise ValueError(f"self-paired nonzero alpha id={ids[i]}")
        # Canonical: keep lower id as source; set partner = exact negation
        if ids[i] <= ids[j]:
            src, dst = i, j
        else:
            src, dst = j, i
        a_src = out_alpha[src]
        out_alpha[dst] = -a_src
        # Average weights then copy (GH should already match)
        w_mean = 0.5 * (out_w[src] + out_w[dst])
        out_w[src] = w_mean
        out_w[dst] = w_mean
        used.add(src)
        used.add(dst)

    if len(used) != len(rows):
        raise ValueError("antipode cover incomplete")

    # Renormalize weights after any averaging
    wsum = sum(out_w)
    out_w = [w / wsum for w in out_w]

    new_rows = []
    for i in range(len(rows)):
        a = out_alpha[i]
        I, phi = alpha_to_drive(a, I_bar, nbar)
        # Exact antipode phase relation for nonzero nodes
        if abs(a) > ALPHA0_TOL:
            # Ensure phi = atan2 path; partner will get mod(phi+pi)
            pass
        new_rows.append((ids[i], out_w[i], a.real, a.imag, I, phi))

    # Second pass: force phi_bar = mod(phi+pi, 2pi) and I_bar = I exactly
    by_key = {_alpha_key(complex(r[2], r[3])): idx for idx, r in enumerate(new_rows)}
    final = list(new_rows)
    seen = set()
    for i, row in enumerate(new_rows):
        if i in seen:
            continue
        a = complex(row[2], row[3])
        if abs(a) <= ALPHA0_TOL:
            # self: I=0, phi=0
            final[i] = (row[0], row[1], 0.0, 0.0, 0.0, 0.0)
            seen.add(i)
            continue
        j = by_key[_alpha_key(-a)]
        ia, ib = (i, j) if row[0] <= new_rows[j][0] else (j, i)
        ra, rb = final[ia], final[ib]
        phi_a = float(ra[5])
        phi_b = (phi_a + PI) % TWOPI
        I_a = float(ra[4])
        w_a = float(ra[1])
        # Keep alpha_b = -alpha_a (already), force I/w/phi
        final[ia] = (ra[0], w_a, ra[2], ra[3], I_a, phi_a)
        final[ib] = (rb[0], w_a, -ra[2], -ra[3], I_a, phi_b)
        seen.add(ia)
        seen.add(ib)
    return final


def make_sv_nodes(r: float, theta_deg: float, I_bar: float, gh_order: int):
    x, w = GH[gh_order]
    theta = math.radians(theta_deg)
    VQ = build_VQ_sv(r, theta)
    L = chol2(VQ)
    d = np.zeros(2)
    nbar = math.sinh(r) ** 2
    rows = []
    k = 0
    wsum = 0.0
    for i in range(len(x)):
        for j in range(len(x)):
            k += 1
            z = np.array([math.sqrt(2.0) * x[i], math.sqrt(2.0) * x[j]])
            ww = (w[i] / math.sqrt(PI)) * (w[j] / math.sqrt(PI))
            xi = d + L @ z
            alpha = complex(xi[0], xi[1]) / math.sqrt(2.0)
            I, phi = alpha_to_drive(alpha, I_bar, nbar)
            rows.append((k, ww, alpha.real, alpha.imag, I, phi))
            wsum += ww
    rows = [(i, ww / wsum, ra, ia, I, phi) for (i, ww, ra, ia, I, phi) in rows]
    return enforce_antipodes(rows, I_bar, nbar)


def write_manifest(path: Path, rows, r, theta_deg, gh_order, I_bar):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# quantum-light node manifest (shared ±N / quadrature)\n")
        f.write("# state_type = squeezed_vacuum\n")
        f.write("# sampling_mode = gauss_hermite\n")
        f.write("# precision = ES25.17\n")
        f.write("# antipodes = enforced (alpha_bar=-alpha; I,phi from alpha; phi_bar=mod(phi+pi,2pi))\n")
        f.write(f"# squeeze_r = {r}\n")
        f.write(f"# squeeze_theta_deg = {theta_deg}\n")
        f.write(f"# gh_order = {gh_order}\n")
        f.write(f"# I_bar = {I_bar}\n")
        f.write("# NOTE: equal-mean-intensity mapping; not experimental BSV calibration.\n")
        f.write(f"# n_nodes = {len(rows)}\n")
        f.write(f"# sum_weight = {sum(r[1] for r in rows):.16e}\n")
        f.write("# sample_id  weight  Re(alpha)  Im(alpha)  I_drive  phi_drive\n")
        for i, ww, ra, ia, I, phi in rows:
            f.write(
                f"{i:8d}{ww:25.17E}{ra:25.17E}{ia:25.17E}{I:25.17E}{phi:25.17E}\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--I-bar", type=float, default=1.0e11)
    ap.add_argument("--gh-order", type=int, default=3, choices=[3, 5, 7])
    ap.add_argument("--r", type=float, nargs="+", default=[0.5, 1.5, 2.5])
    ap.add_argument("--theta-deg", type=float, nargs="+", default=[0.0, 90.0, 180.0, 270.0])
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for r in args.r:
        for th in args.theta_deg:
            rows = make_sv_nodes(r, th, args.I_bar, args.gh_order)
            name = f"nodes_sv_r{str(r).replace('.','p')}_th{int(th)}_gh{args.gh_order}.dat"
            write_manifest(args.outdir / name, rows, r, th, args.gh_order, args.I_bar)
            print("wrote", args.outdir / name)


if __name__ == "__main__":
    main()
