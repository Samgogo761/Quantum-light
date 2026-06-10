"""Compare Wannier TB eigenvalues at selected fractional k points.

This reads only the Hamiltonian block of a Wannier90 *_tb.dat file and skips the
position matrix block. Energies are kept in eV as stored in the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


KPTS = {
    "G": (0.0, 0.0, 0.0),
    "K1": (1.0 / 3.0, 1.0 / 3.0, 0.0),
    "K2": (2.0 / 3.0, 1.0 / 3.0, 0.0),
    "K3": (2.0 / 3.0, 2.0 / 3.0, 0.0),
    "M1": (0.5, 0.0, 0.0),
    "M2": (0.0, 0.5, 0.0),
}


def _read_ints_until(fh, n: int) -> list[int]:
    vals: list[int] = []
    while len(vals) < n:
        line = fh.readline()
        if not line:
            raise EOFError("Unexpected EOF while reading degeneracies")
        vals.extend(int(x) for x in line.split())
    return vals[:n]


def read_hamiltonian_blocks(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        line = fh.readline()
        if "." in line:
            fh.readline()
            fh.readline()
            nwann = int(fh.readline().split()[0])
        else:
            nwann = int(line.split()[0])
        nrpts = int(fh.readline().split()[0])
        ndegen = np.array(_read_ints_until(fh, nrpts), dtype=float)

        irvec = np.zeros((nrpts, 3), dtype=int)
        h_r = np.zeros((nrpts, nwann, nwann), dtype=np.complex128)
        for ir in range(nrpts):
            fh.readline()
            parts = fh.readline().split()
            irvec[ir, :] = [int(parts[0]), int(parts[1]), int(parts[2])]
            for _ in range(nwann * nwann):
                m, n, rr, ri = fh.readline().split()[:4]
                h_r[ir, int(m) - 1, int(n) - 1] = float(rr) + 1j * float(ri)
    return irvec, ndegen, h_r


def eigvals_at(irvec: np.ndarray, ndegen: np.ndarray, h_r: np.ndarray, kfrac: tuple[float, float, float]) -> np.ndarray:
    k = np.array(kfrac, dtype=float)
    phase = np.exp(2j * np.pi * (irvec @ k)) / ndegen
    hk = np.tensordot(phase, h_r, axes=(0, 0))
    hk = 0.5 * (hk + hk.conj().T)
    return np.linalg.eigvalsh(hk)


def invert_k(kfrac: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple((-x) % 1.0 for x in kfrac)


def diff_summary(ep: np.ndarray, em: np.ndarray, window: int) -> tuple[float, float, float, float, float]:
    diff = ep[:window] - em[:window]
    shift = float(np.mean(diff))
    resid = diff - shift
    return (
        float(np.max(np.abs(diff))),
        float(np.sqrt(np.mean(diff * diff))),
        float(np.mean(np.abs(diff))),
        shift,
        float(np.max(np.abs(resid))),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plus", required=True, type=Path)
    parser.add_argument("--minus", required=True, type=Path)
    parser.add_argument("--window", type=int, default=112, help="Number of sorted bands to compare")
    args = parser.parse_args()

    print(f"Reading +N TB: {args.plus}")
    ir_p, nd_p, h_p = read_hamiltonian_blocks(args.plus)
    print(f"  +N: nwann={h_p.shape[1]}, nrpts={h_p.shape[0]}")
    print(f"Reading -N TB: {args.minus}")
    ir_m, nd_m, h_m = read_hamiltonian_blocks(args.minus)
    print(f"  -N: nwann={h_m.shape[1]}, nrpts={h_m.shape[0]}")

    if h_p.shape != h_m.shape:
        raise SystemExit(f"Shape mismatch: {h_p.shape} vs {h_m.shape}")
    if not np.array_equal(ir_p, ir_m):
        print("WARNING: R-vector lists differ between the two TB files.")
    if not np.allclose(nd_p, nd_m):
        print("WARNING: degeneracy lists differ between the two TB files.")

    print("\n# Same-k comparison: E_+N(k) vs E_-N(k)")
    print("# kpoint max_abs_diff_eV rms_diff_eV mean_abs_diff_eV mean_shift_eV max_resid_after_shift_eV")
    for name, kfrac in KPTS.items():
        ep = eigvals_at(ir_p, nd_p, h_p, kfrac)
        em = eigvals_at(ir_m, nd_m, h_m, kfrac)
        maxd, rmsd, meand, shift, resid = diff_summary(ep, em, args.window)
        print(
            f"{name:3s} {maxd:16.8e} "
            f"{rmsd:16.8e} {meand:16.8e} {shift:16.8e} {resid:16.8e}"
        )

    print("\n# Inversion-paired comparison: E_+N(k) vs E_-N(-k)")
    print("# kpoint minus_k max_abs_diff_eV rms_diff_eV mean_abs_diff_eV mean_shift_eV max_resid_after_shift_eV")
    for name, kfrac in KPTS.items():
        kminus = invert_k(kfrac)
        ep = eigvals_at(ir_p, nd_p, h_p, kfrac)
        em = eigvals_at(ir_m, nd_m, h_m, kminus)
        maxd, rmsd, meand, shift, resid = diff_summary(ep, em, args.window)
        print(
            f"{name:3s} ({kminus[0]:.6f},{kminus[1]:.6f},{kminus[2]:.6f}) "
            f"{maxd:16.8e} {rmsd:16.8e} {meand:16.8e} {shift:16.8e} {resid:16.8e}"
        )


if __name__ == "__main__":
    main()
