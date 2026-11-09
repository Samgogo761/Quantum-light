#!/usr/bin/env python3
"""Static occupied-band current J0(Nk) from Wannier tb.dat (no SBE propagation).

J0^a = sum_{k,n occ} w_k * v_nn^a(k)
where v = dH/dk - i[r,H] in Cartesian (same structure as SBE Peierls current).

This is a diagnostic for M64 |J(0)| bias. Uses ndegen only (no wsvec), matching SBE.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_tb(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        line = fh.readline()
        if "." in line:
            a1 = np.array([float(x) for x in line.split()[:3]])
            a2 = np.array([float(x) for x in fh.readline().split()[:3]])
            a3 = np.array([float(x) for x in fh.readline().split()[:3]])
            nw = int(fh.readline().split()[0])
        else:
            # no lattice in header — rare
            raise SystemExit("tb.dat missing lattice vectors")
        nr = int(fh.readline().split()[0])
        ndegen = []
        while len(ndegen) < nr:
            ndegen.extend(int(x) for x in fh.readline().split())
        ndegen = np.array(ndegen[:nr], dtype=float)
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
            fh.readline()  # R header
            for _ in range(nw * nw):
                f = fh.readline().split()
                m, n = int(f[0]) - 1, int(f[1]) - 1
                R[ir, m, n, 0] = float(f[2]) + 1j * float(f[3])
                R[ir, m, n, 1] = float(f[4]) + 1j * float(f[5])
                R[ir, m, n, 2] = float(f[6]) + 1j * float(f[7])
    # eV / Angstrom like Wannier; convert later if needed — keep eV·Ang for relative compare
    A = np.vstack([a1, a2, a3])  # Angstrom
    return A, irvec, ndegen, H, R


def cart_R(A, irvec):
    return irvec @ A  # (nr,3) Angstrom


def Hk_Rk(kfrac, A, irvec, ndegen, H, R):
    """kfrac in reciprocal crystal coords (fractional of b*)."""
    # k_cart · R = 2π kfrac · irvec
    phase = np.exp(2j * np.pi * (irvec @ kfrac)) / ndegen
    Hk = np.einsum("r,rij->ij", phase, H)
    Rk = np.einsum("r,rija->ija", phase, R)
    # dH/dk_cart: need R_cart
    Rc = cart_R(A, irvec)
    dH = np.zeros((3, H.shape[1], H.shape[2]), dtype=np.complex128)
    for a in range(3):
        dH[a] = np.einsum("r,rij->ij", (1j * Rc[:, a] * phase), H)
    return Hk, Rk, dH


def j0_mesh(A, irvec, ndegen, H, R, nk: int, nocc: int, ef_shift=0.0):
    nw = H.shape[1]
    # shift onsite by -EF if needed: H already includes EF convention of file;
    # for CrI3 SBE subtracts EF in Ha — here we just take lowest nocc bands after diag.
    j0 = np.zeros(3, dtype=float)
    # r antiherm diagnostic at Gamma
    k0 = np.array([0.0, 0.0, 0.0])
    Hk, Rk, dH = Hk_Rk(k0, A, irvec, ndegen, H, R)
    anti = []
    for a in range(3):
        Ra = Rk[:, :, a]
        diff = Ra - Ra.conj().T
        frob = np.linalg.norm(Ra)
        anti.append(np.linalg.norm(diff) / frob if frob > 0 else 0.0)

    w = 1.0 / (nk * nk)
    for ikx in range(nk):
        for iky in range(nk):
            kfrac = np.array([ikx / nk, iky / nk, 0.0])
            Hk, Rk, dH = Hk_Rk(kfrac, A, irvec, ndegen, H, R)
            # hermitize H lightly for eig
            Hk = 0.5 * (Hk + Hk.conj().T)
            evals, evecs = np.linalg.eigh(Hk)
            # occupied: lowest nocc
            for n in range(nocc):
                c = evecs[:, n]
                for a in range(3):
                    # v = dH - i[r,H]; diagonal in band basis
                    Ra = Rk[:, :, a]
                    comm = Ra @ Hk - Hk @ Ra
                    Vk = dH[a] - 1j * comm
                    # band-diagonal: c† V c
                    vnn = np.vdot(c, Vk @ c).real
                    j0[a] += w * vnn
    return j0, anti


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", type=Path, required=True)
    ap.add_argument("--nocc", type=int, required=True)
    ap.add_argument("--nk", type=int, nargs="+", default=[20, 40])
    args = ap.parse_args()
    A, irvec, ndegen, H, R = read_tb(args.tb)
    print(f"nwann={H.shape[1]} nrpts={H.shape[0]} nocc={args.nocc}")
    for nk in args.nk:
        j0, anti = j0_mesh(A, irvec, ndegen, H, R, nk, args.nocc)
        print(
            f"nk={nk:3d}  J0=({j0[0]:.4e},{j0[1]:.4e},{j0[2]:.4e})  |J0|={np.linalg.norm(j0):.4e}"
            f"  Gamma rel_antiherm_r=({anti[0]:.3e},{anti[1]:.3e},{anti[2]:.3e})"
        )


if __name__ == "__main__":
    main()
