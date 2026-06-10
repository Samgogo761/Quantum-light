"""Convert a wannier90 spin matrix (.spn, Bloch basis) into the on-site S_z
matrix in the Wannier basis, for the SBE solver's `spin_sz_file` input.

Pipeline
--------
The .spn file stores sigma matrix elements in the Bloch (ab-initio) band basis:
    S^a_{mn}(k) = <psi_mk| sigma_a |psi_nk>,  a = x,y,z.
The wannier90 gauge maps Bloch -> Wannier via U_total(k) = U_dis(k) @ U(k):
    U_dis : (num_bands x num_wann)   [disentanglement; identity if none]
    U     : (num_wann  x num_wann)   [MLWF gauge]
The k-resolved spin operator in the Wannier gauge is
    Sz_w(k) = U_total(k)^dagger  ( sigma_z/2 )(k)  U_total(k).
The solver uses a single, k-independent on-site (R=0) matrix, which is the BZ
average:
    Sz_wann_{ab} = (1/Nk) sum_k Sz_w(k)_{ab}.
(This drops R/=0 spin hopping, which is tiny because spin is local. Good enough;
the solver's equilibrium <S_z> self-check is the ultimate validation.)

Required wannier90 / pw2wannier90 flags (server re-run)
------------------------------------------------------
  pw2wannier90 input:  write_spn = .true.      write_spn_formatted = .true.
  wannier90    input:  write_u_matrices = .true.
This yields  seedname.spn (formatted),  seedname_u.mat,  seedname_u_dis.mat.
(.spn note: factor 1/2 -> set --half to convert Pauli sigma to spin S=sigma/2.)

Output
------
  Sz_wannier.dat   with lines "m n Re Im"  (1-based; only |entry|>tol kept).
Point the solver at it:   &spin  spin_sz_file = "Sz_wannier.dat" /
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

_CPLX = re.compile(r"\(\s*([\-0-9.eE+]+)\s*,\s*([\-0-9.eE+]+)\s*\)")


def _tokens_to_complex(text: str) -> np.ndarray:
    """Parse a stream that is either '(re,im) (re,im) ...' or 're im re im ...'."""
    paren = _CPLX.findall(text)
    if paren:
        return np.array([float(a) + 1j * float(b) for a, b in paren], dtype=complex)
    vals = np.array([float(x) for x in text.split()], dtype=float)
    if vals.size % 2 != 0:
        raise ValueError("Odd number of reals; cannot pair into complex.")
    return vals[0::2] + 1j * vals[1::2]


def read_spn_formatted(path: Path):
    """Return Sz_bloch[nk, nb, nb] (only the z component, already sigma_z)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    # line 0 = header/comment; line 1 = "num_bands num_kpts"
    nb, nk = (int(x) for x in lines[1].split()[:2])
    body = "\n".join(lines[2:])
    flat = _tokens_to_complex(body)
    ntri = nb * (nb + 1) // 2
    expect = 3 * ntri * nk
    if flat.size != expect:
        raise ValueError(f".spn token count {flat.size} != expected {expect} "
                         f"(nb={nb}, nk={nk}); check formatting.")
    flat = flat.reshape(nk, ntri, 3)         # per k, per triangle entry, (x,y,z)
    sz = np.zeros((nk, nb, nb), dtype=complex)
    # unpack upper triangle: m outer (1..nb), n inner (1..m)
    for ik in range(nk):
        c = 0
        for m in range(nb):
            for n in range(m + 1):
                val = flat[ik, c, 2]         # z component
                sz[ik, n, m] = val
                sz[ik, m, n] = np.conjugate(val)
                c += 1
    return nb, nk, sz


def read_umat(path: Path):
    """Read a wannier90 *_u.mat / *_u_dis.mat (formatted). Returns U[nk, nrow, ncol]."""
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    # header line 0; line 1 = "nk d1 d2"
    nk, d1, d2 = (int(x) for x in lines[1].split()[:3])
    # For _u.mat:      d1=num_wann (rows), d2=num_wann (cols)
    # For _u_dis.mat:  d1=num_wann, d2=num_bands  -> stored U_opt is (num_bands,num_wann)
    nums = _tokens_to_complex("\n".join(lines[2:]))
    per_k = d1 * d2
    # each k block is preceded by a kpoint line (3 reals) -> 3 reals = not complex pairs.
    # Robust approach: re-parse counting real tokens, skipping 3 reals (kpt) per block.
    reals = np.array([float(x) for x in "\n".join(lines[2:]).split()], dtype=float)
    out = []
    idx = 0
    for _ in range(nk):
        idx += 3                              # skip kx ky kz
        block = reals[idx: idx + 2 * per_k]
        idx += 2 * per_k
        cx = block[0::2] + 1j * block[1::2]
        out.append(cx.reshape((d2, d1)).T)    # column-major (rows fastest) -> (d1,d2)... see note
    return nk, d1, d2, np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spn", required=True, type=Path)
    ap.add_argument("--umat", required=True, type=Path, help="seedname_u.mat")
    ap.add_argument("--udis", type=Path, default=None,
                    help="seedname_u_dis.mat (omit if no disentanglement)")
    ap.add_argument("--half", action="store_true",
                    help="multiply by 1/2 (sigma_z -> S_z); use if .spn stores Pauli sigma")
    ap.add_argument("--out", type=Path, default=Path("Sz_wannier.dat"))
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()

    nb, nk, sz_bloch = read_spn_formatted(args.spn)
    print(f".spn: num_bands={nb}, num_kpts={nk}")
    if args.half:
        sz_bloch = 0.5 * sz_bloch

    nk_u, nw, nw2, U = read_umat(args.umat)
    print(f"_u.mat: nk={nk_u}, shape per k = ({nw},{nw2})")
    if args.udis is not None:
        nk_d, dw, db, Udis = read_umat(args.udis)
        print(f"_u_dis.mat: nk={nk_d}, shape per k = ({dw},{db})")
        # U_total(k) = U_dis(k)[nb,nw] @ U(k)[nw,nw]
        Utot = np.array([_combine(Udis[k], U[k], nb, nw) for k in range(nk)])
    else:
        Utot = U  # num_bands == num_wann

    if Utot.shape[1] != nb:
        print(f"WARNING: U_total rows ({Utot.shape[1]}) != num_bands ({nb}); "
              f"band/window alignment may be off. Verify with the solver self-check.")

    # Sz_wann = (1/Nk) sum_k U(k)^H sigma_z(k) U(k)
    nwann = Utot.shape[2]
    Sz_w = np.zeros((nwann, nwann), dtype=complex)
    for k in range(nk):
        Sz_w += Utot[k].conj().T @ sz_bloch[k] @ Utot[k]
    Sz_w /= nk

    # --- self-checks ---
    herm = float(np.max(np.abs(Sz_w - Sz_w.conj().T)))
    diag = np.real(np.diag(Sz_w))
    print("\nSelf-checks:")
    print(f"  Hermiticity max|Sz - Sz^H|   = {herm:.2e}")
    print(f"  diagonal range               = [{diag.min():.4f}, {diag.max():.4f}] (expect within +-0.5)")
    print(f"  trace (sum of diag)          = {np.real(np.trace(Sz_w)):.4f}")
    print(f"  first 10 diagonal <S_z>      = {np.round(diag[:10], 3)}")

    with args.out.open("w", encoding="utf-8") as f:
        f.write(f"# S_z in Wannier basis (on-site, BZ-averaged). nwann={nwann}\n")
        f.write("# m n Re Im\n")
        for m in range(nwann):
            for n in range(nwann):
                v = Sz_w[m, n]
                if abs(v) > args.tol:
                    f.write(f"{m+1} {n+1} {v.real:.10e} {v.imag:.10e}\n")
    print(f"\n[written] {args.out}  (set &spin spin_sz_file to this)")


def _combine(Udis_k, U_k, nb, nw):
    """U_total(k) = U_dis(k)[nb,nw] @ U(k)[nw,nw]. read_umat gives (num_wann,num_bands)."""
    Udis_bn = Udis_k.T if Udis_k.shape == (nw, nb) else Udis_k   # -> (nb, nw)
    return Udis_bn @ U_k


if __name__ == "__main__":
    main()
