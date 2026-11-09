#!/usr/bin/env python3
"""M64 P3 diagnostics: r(k) residual map, static J0(Nk), optional r_sym probe.

Does NOT require SBE rebuild. wsvec (use_ws_distance=.true.) full apply is
implemented as an optional slow path; default reports whether wsvec exists and
compares ndegen-only r(k) hermiticity.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_tb(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        line = fh.readline()
        a1 = np.array([float(x) for x in line.split()[:3]])
        a2 = np.array([float(x) for x in fh.readline().split()[:3]])
        a3 = np.array([float(x) for x in fh.readline().split()[:3]])
        nw = int(fh.readline().split()[0])
        nr = int(fh.readline().split()[0])
        ndegen = []
        while len(ndegen) < nr:
            ndegen.extend(int(x) for x in fh.readline().split())
        ndegen = np.asarray(ndegen[:nr], dtype=float)
        irvec = np.zeros((nr, 3), dtype=int)
        H = np.zeros((nr, nw, nw), dtype=np.complex128)
        for ir in range(nr):
            fh.readline()
            irvec[ir] = [int(x) for x in fh.readline().split()[:3]]
            for _ in range(nw * nw):
                m, n, rr, ri = fh.readline().split()[:4]
                H[ir, int(m) - 1, int(n) - 1] = float(rr) + 1j * float(ri)
        R = np.zeros((nr, nw, nw, 3), dtype=np.complex128)
        for ir in range(nr):
            fh.readline()
            fh.readline()
            for _ in range(nw * nw):
                f = fh.readline().split()
                m, n = int(f[0]) - 1, int(f[1]) - 1
                R[ir, m, n, 0] = float(f[2]) + 1j * float(f[3])
                R[ir, m, n, 1] = float(f[4]) + 1j * float(f[5])
                R[ir, m, n, 2] = float(f[6]) + 1j * float(f[7])
    A = np.vstack([a1, a2, a3])
    return A, irvec, ndegen, H, R


def interpolate(kfrac, irvec, ndegen, Op):
    """Op: (nr, nw, nw) or (nr, nw, nw, 3). ndegen-only Fourier (SBE default)."""
    phase = np.exp(2j * np.pi * (irvec @ kfrac)) / ndegen
    if Op.ndim == 3:
        return np.einsum("r,rij->ij", phase, Op)
    out = np.zeros(Op.shape[1:], dtype=np.complex128)
    for a in range(Op.shape[-1]):
        out[..., a] = np.einsum("r,rij->ij", phase, Op[..., a])
    return out


def antiherm_stats(Mat):
    diff = Mat - Mat.conj().T
    frob = np.linalg.norm(Mat)
    antif = np.linalg.norm(diff)
    # max element
    absd = np.abs(diff)
    idx = np.unravel_index(np.argmax(absd), absd.shape)
    return antif, (antif / frob if frob > 0 else 0.0), idx, absd[idx]


def static_j0(A, irvec, ndegen, H, Rmat, nk: int, nocc: int):
    Rc = irvec @ A
    j0 = np.zeros(3)
    w = 1.0 / (nk * nk)
    for ikx in range(nk):
        for iky in range(nk):
            kfrac = np.array([ikx / nk, iky / nk, 0.0])
            phase = np.exp(2j * np.pi * (irvec @ kfrac)) / ndegen
            Hk = np.einsum("r,rij->ij", phase, H)
            Hk = 0.5 * (Hk + Hk.conj().T)
            evals, evecs = np.linalg.eigh(Hk)
            Rk = np.zeros((H.shape[1], H.shape[1], 3), dtype=np.complex128)
            dH = np.zeros_like(Rk)
            for a in range(3):
                Rk[:, :, a] = np.einsum("r,rij->ij", phase, Rmat[..., a])
                dH[:, :, a] = np.einsum("r,rij->ij", 1j * Rc[:, a] * phase, H)
            for n in range(nocc):
                c = evecs[:, n]
                for a in range(3):
                    Ra = Rk[:, :, a]
                    Vk = dH[:, :, a] - 1j * (Ra @ Hk - Hk @ Ra)
                    j0[a] += w * np.vdot(c, Vk @ c).real
    return j0


def r_symmetrize(irvec, Rmat):
    """r_sym(R) = 0.5 * (r(R) + r(-R)^H) elementwise for each Cartesian."""
    index = {tuple(irvec[i]): i for i in range(len(irvec))}
    out = np.empty_like(Rmat)
    for ir in range(len(irvec)):
        jm = index[tuple(-irvec[ir])]
        for a in range(3):
            out[ir, :, :, a] = 0.5 * (Rmat[ir, :, :, a] + Rmat[jm, :, :, a].conj().T)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", type=Path, required=True)
    ap.add_argument("--wsvec", type=Path, default=None)
    ap.add_argument("--nocc", type=int, default=36)
    ap.add_argument("--nk-diag", type=int, default=20, help="mesh for r(k) residual scan")
    ap.add_argument("--nk-j0", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    A, irvec, ndegen, H, Rmat = read_tb(args.tb)
    nw = H.shape[1]
    print(f"Loaded tb nw={nw} nr={H.shape[0]}")

    if args.wsvec and args.wsvec.exists():
        head = args.wsvec.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
        print("wsvec present:")
        for line in head:
            print(" ", line)
        print("  NOTE: use_ws_distance=.true. per-(R,i,j) format; full apply deferred to Fortran path.")
        (args.outdir / "wsvec_header.txt").write_text("\n".join(head) + "\n", encoding="utf-8")
    else:
        print("wsvec: NOT PROVIDED")

    # accumulate |r-r†| per matrix element over mesh
    acc = np.zeros((nw, nw, 3))
    max_pair = []
    for ikx in range(args.nk_diag):
        for iky in range(args.nk_diag):
            kfrac = np.array([ikx / args.nk_diag, iky / args.nk_diag, 0.0])
            Rk = interpolate(kfrac, irvec, ndegen, Rmat)
            for a in range(3):
                diff = np.abs(Rk[:, :, a] - Rk[:, :, a].conj().T)
                acc[:, :, a] += diff

    # top residuals (sum over Cartesian)
    score = acc.sum(axis=2)
    flat = [(score[i, j], i, j) for i in range(nw) for j in range(i, nw)]
    flat.sort(reverse=True)
    with (args.outdir / "r_antiherm_top_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "i_wf", "j_wf", "sum_abs_antiherm_over_mesh", "mean_per_k"])
        for rank, (sc, i, j) in enumerate(flat[: args.top], 1):
            w.writerow([rank, i + 1, j + 1, f"{sc:.6e}", f"{sc/(args.nk_diag**2):.6e}"])
            max_pair.append((i + 1, j + 1, sc))
    print("Top r antiherm pairs (1-based WF):")
    for i, j, sc in max_pair[:10]:
        print(f"  WF {i:3d}-{j:3d}: {sc:.4e}")

    # Gamma stats before/after r_sym
    k0 = np.zeros(3)
    Rk = interpolate(k0, irvec, ndegen, Rmat)
    with (args.outdir / "r_gamma_antiherm.txt").open("w", encoding="utf-8") as f:
        for a, lab in enumerate("xyz"):
            antif, rel, idx, mx = antiherm_stats(Rk[:, :, a])
            line = f"r_{lab}(G): |anti|_F={antif:.4e} rel={rel:.4e} max@({idx[0]+1},{idx[1]+1})={mx:.4e}\n"
            f.write(line)
            print(line.strip())
        Rsym = r_symmetrize(irvec, Rmat)
        Rk2 = interpolate(k0, irvec, ndegen, Rsym)
        f.write("--- after r_sym(R)=[r(R)+r(-R)^H]/2 ---\n")
        print("after r_sym:")
        for a, lab in enumerate("xyz"):
            antif, rel, idx, mx = antiherm_stats(Rk2[:, :, a])
            line = f"r_{lab}(G): |anti|_F={antif:.4e} rel={rel:.4e} max@({idx[0]+1},{idx[1]+1})={mx:.4e}\n"
            f.write(line)
            print(line.strip())

    # static J0
    with (args.outdir / "static_J0.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nk", "J0x", "J0y", "J0z", "absJ0"])
        for nk in args.nk_j0:
            j0 = static_j0(A, irvec, ndegen, H, Rmat, nk, args.nocc)
            print(f"J0 nk={nk}: {j0} |J0|={np.linalg.norm(j0):.4e}")
            w.writerow([nk, f"{j0[0]:.6e}", f"{j0[1]:.6e}", f"{j0[2]:.6e}", f"{np.linalg.norm(j0):.6e}"])
        # also with r_sym
        Rsym = r_symmetrize(irvec, Rmat)
        for nk in args.nk_j0[:1]:
            j0 = static_j0(A, irvec, ndegen, H, Rsym, nk, args.nocc)
            print(f"J0 nk={nk} r_sym: {j0} |J0|={np.linalg.norm(j0):.4e}")
            w.writerow([f"{nk}_rsym", f"{j0[0]:.6e}", f"{j0[1]:.6e}", f"{j0[2]:.6e}", f"{np.linalg.norm(j0):.6e}"])

    print("Wrote", args.outdir)


if __name__ == "__main__":
    main()
