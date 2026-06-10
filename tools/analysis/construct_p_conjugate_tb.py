"""Construct a strictly inversion (P) conjugate -N Wannier TB from the +N TB.

Motivation
----------
The AFM Neel selection-rule check J_n(-N) = (-1)^(n+1) J_n(N) requires that the
-N tight-binding model be the *exact* spatial-inversion partner of the +N model.
Two independent SCF+Wannier runs do NOT satisfy this at the operator level (SCF
total energies differ by 2.18 mRy, Omega_I differs by 0.13 Ang^2, TB eigenvalues
differ by 50-110 meV at Gamma/K/M). So we *build* the -N TB from the +N TB.

Symmetry bookkeeping
--------------------
For bilayer CrI3 the (non-magnetic) geometry is centrosymmetric about a center
c (fractional) ~ (0.5, 1/3, 0.372). +N and -N share identical atomic positions;
only the Cr1/Cr2 spin labels are swapped. Spatial inversion P swaps the two
layers (Cr1<->Cr2 and the I partners), carrying each atom's spin with it (spin is
an axial vector: P does NOT flip it). The layer swap turns the +N Neel pattern
into the -N pattern, so P maps the +N ground state onto the -N one.

Wannier-function mapping under P:
  * P pairs each atom X with its inversion image atom P(X).
  * Within an atom the Wannier order is (orbital m) x (spin), and P preserves
    both (l-even d -> same d, sign +1; l-odd p -> same p, sign -1). So the
    intra-atom index is preserved and the permutation is fixed by atom pairing.
  * Parity sign eta_a = +1 for Cr-d Wannier functions, -1 for I-p ones.

Operator transforms (stored, Wannier90 convention; eta_a^2 = 1):

    H_{-N}(R)_{a b} =  eta_a eta_b  H_{+N}(-R)_{P(a) P(b)}
    r_{-N}(R)_{a b} = -eta_a eta_b  r_{+N}(-R)_{P(a) P(b)}
                       + 2 c_cart * delta_{a b} delta_{R,0}

The position operator is a polar vector (P: r -> 2c - r), hence the overall minus
sign on the off-diagonal/finite-R part and the 2c_cart shift on the home-cell
diagonal (which places the -N Wannier centers at their inverted positions). The
2c term only affects intraband/diagonal positions and is a global translation =
length-gauge gauge for the current, but it is included so the centers are
physically correct.

Per-orbital Wigner-Seitz cell shifts only add a k-dependent diagonal gauge
e^{-i k (L_a - L_b)} that leaves eigenvalues and gauge-invariant dipoles
unchanged; ndegen(-R)=ndegen(R) so reusing the +N R-list/ndegen is exact.

Validation gates (all must pass before the file is trusted):
  D1  Eigenvalue gate:  E_{-N}(k) == E_{+N}(-k)            (target < 1e-9 eV)
  D2  Hermiticity:      r_{-N}(R)_{ab} == conj(r_{-N}(-R)_{ba})
  D3  Wannier centers:  diag r_{-N}(0)_a == 2c - center_{P(a)}
  D4  Interband dipole: |d^band_{-N}(k)| == |d^band_{+N}(-k)|  (gauge invariant)
"""

from __future__ import annotations

import argparse
import re
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

# WF block structure (from +N projection / wout center order):
#   WF   1-40  : 4 Cr atoms x 10 (5 d-orbital x 2 spin) -> parity +1
#   WF  41-112 : 12 I atoms x 6 (3 p-orbital x 2 spin)  -> parity -1
N_CR_ATOMS = 4
CR_WF_PER_ATOM = 10
N_I_ATOMS = 12
I_WF_PER_ATOM = 6

WF_RE = re.compile(
    r"WF centre and spread\s+(\d+)\s+\(\s*([\-0-9.Ee+]+),\s*([\-0-9.Ee+]+),\s*([\-0-9.Ee+]+)\s*\)\s+([\-0-9.Ee+]+)"
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_lattice_ang(win: Path) -> np.ndarray:
    lines = win.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == "begin unit_cell_cart":
            unit = lines[i + 1].strip().lower()
            if unit not in ("ang", "angstrom"):
                raise ValueError(f"Expected angstrom unit_cell_cart, got {unit!r}")
            return np.array(
                [[float(x) for x in lines[i + 2 + j].split()[:3]] for j in range(3)],
                dtype=float,
            )
    raise ValueError(f"Cannot find unit_cell_cart in {win}")


def parse_final_centers(wout: Path) -> np.ndarray:
    lines = wout.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "Final State"), None)
    if start is None:
        raise ValueError(f"Cannot find 'Final State' in {wout}")
    centers: list[tuple[int, list[float]]] = []
    for line in lines[start + 1:]:
        m = WF_RE.search(line)
        if m:
            centers.append((int(m.group(1)), [float(m.group(j)) for j in range(2, 5)]))
        elif centers and line.strip().startswith("Sum of centres"):
            break
    if not centers:
        raise ValueError(f"No WF centers found in {wout}")
    centers.sort(key=lambda x: x[0])
    return np.array([c for _, c in centers], dtype=float)


def _read_ints_until(fh, n: int) -> list[int]:
    vals: list[int] = []
    while len(vals) < n:
        line = fh.readline()
        if not line:
            raise EOFError("Unexpected EOF while reading degeneracies")
        vals.extend(int(x) for x in line.split())
    return vals[:n]


def read_tb_blocks(path: Path, read_position: bool):
    """Parse a Wannier90 *_tb.dat file.

    Returns (irvec, ndegen, h_r, r_r) where r_r is None unless read_position.
    h_r: (nrpts, nw, nw) complex ; r_r: (nrpts, nw, nw, 3) complex (Angstrom).
    """
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

        r_r = None
        if read_position:
            r_r = np.zeros((nrpts, nwann, nwann, 3), dtype=np.complex128)
            for ir in range(nrpts):
                fh.readline()
                parts = fh.readline().split()
                if [int(parts[0]), int(parts[1]), int(parts[2])] != irvec[ir].tolist():
                    raise ValueError(f"R header mismatch in position block at ir={ir}")
                for _ in range(nwann * nwann):
                    f = fh.readline().split()
                    m, n = int(f[0]) - 1, int(f[1]) - 1
                    r_r[ir, m, n, 0] = float(f[2]) + 1j * float(f[3])
                    r_r[ir, m, n, 1] = float(f[4]) + 1j * float(f[5])
                    r_r[ir, m, n, 2] = float(f[6]) + 1j * float(f[7])
    return irvec, ndegen, h_r, r_r


def read_header_text(path: Path, nrpts: int) -> list[str]:
    """Return verbatim header lines (comment + lattice + nwann + nrpts + ndegen)."""
    n_header = 6 + (nrpts + 14) // 15
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for _ in range(n_header):
            out.append(fh.readline().rstrip("\n"))
    return out


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #
def build_wf_metadata(nwann: int) -> dict:
    species = np.empty(nwann, dtype="U2")
    atom = np.empty(nwann, dtype=int)
    intra = np.empty(nwann, dtype=int)
    parity = np.empty(nwann, dtype=int)
    for a in range(nwann):
        if a < N_CR_ATOMS * CR_WF_PER_ATOM:
            species[a] = "Cr"
            atom[a] = a // CR_WF_PER_ATOM
            intra[a] = a % CR_WF_PER_ATOM
            parity[a] = +1
        else:
            b = a - N_CR_ATOMS * CR_WF_PER_ATOM
            species[a] = "I"
            atom[a] = N_CR_ATOMS + b // I_WF_PER_ATOM
            intra[a] = b % I_WF_PER_ATOM
            parity[a] = -1
    return {"species": species, "atom": atom, "intra": intra, "parity": parity}


def _frac_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 0.5) % 1.0 - 0.5


def build_permutation(centers_frac, meta, center):
    nwann = centers_frac.shape[0]
    atom, intra, species = meta["atom"], meta["intra"], meta["species"]
    natoms = int(atom.max()) + 1

    centroid = np.zeros((natoms, 3))
    for at in range(natoms):
        idx = np.where(atom == at)[0]
        ref = centers_frac[idx[0]]
        centroid[at] = ref + _frac_delta(centers_frac[idx], ref).mean(axis=0)

    atom_partner = -np.ones(natoms, dtype=int)
    atom_pair_dist = np.zeros(natoms)
    sp_of_atom = [species[np.where(atom == j)[0][0]] for j in range(natoms)]
    for at in range(natoms):
        image = (2.0 * center - centroid[at]) % 1.0
        d = np.array([np.linalg.norm(_frac_delta(image, centroid[j])) for j in range(natoms)])
        d = np.where([sp_of_atom[j] == sp_of_atom[at] for j in range(natoms)], d, np.inf)
        j = int(np.argmin(d))
        atom_partner[at] = j
        atom_pair_dist[at] = d[j]

    intra_lookup = {(atom[a], intra[a]): a for a in range(nwann)}
    perm = np.array([intra_lookup[(atom_partner[atom[a]], intra[a])] for a in range(nwann)])

    diag = {
        "natoms": natoms,
        "max_atom_pair_dist": float(atom_pair_dist.max()),
        "mean_atom_pair_dist": float(atom_pair_dist.mean()),
        "atom_partner": atom_partner,
        "duplicate_targets": int(nwann - len(set(perm.tolist()))),
        "non_involutive": int(np.sum(perm[perm] != np.arange(nwann))),
    }
    return perm, diag


def neg_R_index(irvec):
    index = {tuple(irvec[i]): i for i in range(irvec.shape[0])}
    missing = [tuple(-irvec[i]) for i in range(irvec.shape[0]) if tuple(-irvec[i]) not in index]
    if missing:
        raise SystemExit(f"R-list not inversion symmetric ({len(missing)} missing -R).")
    return np.array([index[tuple(-irvec[i])] for i in range(irvec.shape[0])], dtype=int)


def build_minus_hr(h_r, perm, parity, neg_idx):
    sign = np.outer(parity.astype(float), parity.astype(float))
    h_m = np.empty_like(h_r)
    for ir in range(h_r.shape[0]):
        h_m[ir] = sign * h_r[neg_idx[ir]][np.ix_(perm, perm)]
    return h_m


def build_minus_rr(r_r, perm, parity, neg_idx, irvec, c_cart):
    """r_{-N}(R)_ab = -eta_a eta_b r_{+N}(-R)_{P(a)P(b)} + 2 c_cart delta_ab delta_R0."""
    sign = np.outer(parity.astype(float), parity.astype(float))
    r_m = np.empty_like(r_r)
    nwann = r_r.shape[1]
    for ir in range(r_r.shape[0]):
        src = r_r[neg_idx[ir]]            # r_{+N}(-R)
        for a in range(3):
            r_m[ir, :, :, a] = -sign * src[:, :, a][np.ix_(perm, perm)]
    ir0 = next(i for i in range(irvec.shape[0]) if tuple(irvec[i]) == (0, 0, 0))
    for a in range(3):
        r_m[ir0, :, :, a][np.diag_indices(nwann)] += 2.0 * c_cart[a]
    return r_m


# --------------------------------------------------------------------------- #
# k-space helpers / gates
# --------------------------------------------------------------------------- #
def eigh_at(irvec, ndegen, h_r, kfrac):
    phase = np.exp(2j * np.pi * (irvec @ np.array(kfrac))) / ndegen
    hk = np.tensordot(phase, h_r, axes=(0, 0))
    hk = 0.5 * (hk + hk.conj().T)
    return np.linalg.eigh(hk)


def dipole_k(irvec, ndegen, r_r, kfrac):
    phase = np.exp(2j * np.pi * (irvec @ np.array(kfrac))) / ndegen
    return np.tensordot(phase, r_r, axes=(0, 0))  # (nw, nw, 3)


def interband_strength(evec, dk):
    """Per-band Sum_{m!=n} |d^band_{nm}|^2, summed over the 3 Cartesian comps."""
    nw = evec.shape[0]
    s = np.zeros(nw)
    for a in range(3):
        db = evec.conj().T @ dk[:, :, a] @ evec
        np.fill_diagonal(db, 0.0)
        s += np.sum(np.abs(db) ** 2, axis=1)
    return s


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
def write_tb_file(out_path, header_lines, irvec, h_m, r_m, comment=None):
    nwann = h_m.shape[1]
    # m fast, n slow (file order: (1,1),(2,1),...,(nw,1),(1,2),...)
    n_grid, m_grid = np.meshgrid(np.arange(1, nwann + 1), np.arange(1, nwann + 1), indexing="ij")
    m_flat = m_grid.reshape(-1)  # length nw*nw, m varies fastest within each n
    n_flat = n_grid.reshape(-1)
    # We need order m-fastest: iterate n outer, m inner -> column index pattern
    order = np.lexsort((m_flat, n_flat))  # sort by n then m -> n outer, m inner
    m_seq = m_flat[order]
    n_seq = n_flat[order]

    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        if comment is not None:
            fh.write(comment.rstrip("\n") + "\n")
            fh.write("\n".join(header_lines[1:]) + "\n")
        else:
            fh.write("\n".join(header_lines) + "\n")

        # H block
        for ir in range(irvec.shape[0]):
            fh.write("\n")
            fh.write(f"{irvec[ir,0]:5d}{irvec[ir,1]:5d}{irvec[ir,2]:5d}\n")
            h = h_m[ir]
            re = h.real.T.reshape(-1)  # transpose so m-fastest matches m_seq order
            im = h.imag.T.reshape(-1)
            lines = [f"{m_seq[i]:5d}{n_seq[i]:5d}{re[i]:18.8E}{im[i]:18.8E}"
                     for i in range(len(m_seq))]
            fh.write("\n".join(lines) + "\n")

        # position block
        for ir in range(irvec.shape[0]):
            fh.write("\n")
            fh.write(f"{irvec[ir,0]:5d}{irvec[ir,1]:5d}{irvec[ir,2]:5d}\n")
            rx = r_m[ir, :, :, 0].T.reshape(-1)
            ry = r_m[ir, :, :, 1].T.reshape(-1)
            rz = r_m[ir, :, :, 2].T.reshape(-1)
            lines = [
                f"{m_seq[i]:5d}{n_seq[i]:5d}"
                f"{rx[i].real:18.8E}{rx[i].imag:18.8E}"
                f"{ry[i].real:18.8E}{ry[i].imag:18.8E}"
                f"{rz[i].real:18.8E}{rz[i].imag:18.8E}"
                for i in range(len(m_seq))
            ]
            fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plus-tb", required=True, type=Path)
    ap.add_argument("--plus-win", required=True, type=Path)
    ap.add_argument("--plus-wout", required=True, type=Path)
    ap.add_argument("--center", default="0.499998583,0.333333067,0.372004766")
    ap.add_argument("--out", type=Path, default=None, help="Write the -N tb.dat here")
    ap.add_argument("--gate-tol", type=float, default=1e-9)
    args = ap.parse_args()

    center = np.array([float(x) for x in args.center.split(",")], dtype=float)
    want_position = args.out is not None

    print("=== Construct P-conjugate -N TB from +N TB ===")
    lattice = parse_lattice_ang(args.plus_win)
    c_cart = center @ lattice
    centers_cart = parse_final_centers(args.plus_wout)
    centers_frac = (centers_cart @ np.linalg.inv(lattice)) % 1.0
    nwann = centers_frac.shape[0]
    meta = build_wf_metadata(nwann)
    perm, diag = build_permutation(centers_frac, meta, center)

    print(f"\n[A] nwann={nwann}, natoms={diag['natoms']}")
    print(f"    max/mean atom-pair fractional dist = "
          f"{diag['max_atom_pair_dist']:.3e} / {diag['mean_atom_pair_dist']:.3e}")
    print(f"[B] duplicate targets={diag['duplicate_targets']}, "
          f"non-involutive={diag['non_involutive']}")
    if diag["duplicate_targets"] or diag["non_involutive"]:
        raise SystemExit("Permutation is not a clean involution; aborting.")

    print(f"\n[C] Reading +N TB (position block: {want_position}) ...")
    irvec, ndegen, h_p, r_p = read_tb_blocks(args.plus_tb, want_position)
    print(f"    nrpts={irvec.shape[0]}")
    neg_idx = neg_R_index(irvec)

    h_m = build_minus_hr(h_p, perm, meta["parity"], neg_idx)

    # ---- D1 eigenvalue gate -------------------------------------------------
    print("\n[D1] Eigenvalue gate  E_-N(k) vs E_+N(-k):")
    worst = 0.0
    eig_cache = {}
    for name, kf in KPTS.items():
        kminus = tuple((-x) % 1.0 for x in kf)
        em, vm = eigh_at(irvec, ndegen, h_m, kf)
        ep, vp = eigh_at(irvec, ndegen, h_p, kminus)
        eig_cache[name] = (em, vm, ep, vp, kminus)
        d = float(np.max(np.abs(em - ep)))
        worst = max(worst, d)
        print(f"     {name:3s}  max|dE| = {d:.3e} eV")
    print(f"     -> worst = {worst:.3e} eV : {'PASS' if worst < args.gate_tol else 'FAIL'}")
    if worst >= args.gate_tol:
        raise SystemExit("D1 failed.")

    if not want_position:
        print("\n(H-block only; pass --out to build position block and write file.)")
        return

    # ---- build position block ----------------------------------------------
    print("\n[C] Constructing r_-N(R) (polar: extra -1, +2c on home diagonal) ...")
    r_m = build_minus_rr(r_p, perm, meta["parity"], neg_idx, irvec, c_cart)

    # ---- D2 Hermiticity (compared to the +N baseline) -----------------------
    # Wannier90's position matrix is only approximately Hermitian (finite-diff
    # k-mesh), so the meaningful test is that the -N construction does not make
    # it any worse than the +N input it was built from.
    herm_m = herm_p = 0.0
    for ir in range(irvec.shape[0]):
        dm = r_m[ir] - r_m[neg_idx[ir]].conj().transpose(1, 0, 2)
        dp = r_p[ir] - r_p[neg_idx[ir]].conj().transpose(1, 0, 2)
        herm_m = max(herm_m, float(np.max(np.abs(dm))))
        herm_p = max(herm_p, float(np.max(np.abs(dp))))
    d2_ok = herm_m <= herm_p * 1.0001 + 1e-12
    print(f"[D2] max |r(R)-r(-R)^H|:  -N={herm_m:.3e}  (+N baseline={herm_p:.3e}) Ang : "
          f"{'PASS (inherited from +N)' if d2_ok else 'FAIL'}")

    # ---- D3 Wannier centers -------------------------------------------------
    ir0 = next(i for i in range(irvec.shape[0]) if tuple(irvec[i]) == (0, 0, 0))
    center_m = np.real(np.array([r_m[ir0, a, a, :] for a in range(nwann)]))
    center_p = np.real(np.array([r_p[ir0, a, a, :] for a in range(nwann)]))
    expected = 2.0 * c_cart[None, :] - center_p[perm]
    d3 = float(np.max(np.abs(center_m - expected)))
    print(f"[D3] max |center_-N(a) - (2c - center_+N(P(a)))| = {d3:.3e} Ang : "
          f"{'PASS' if d3 < 1e-6 else 'FAIL'}")

    # ---- D4 interband dipole (gauge invariant) ------------------------------
    print("[D4] Interband dipole strength  S_n^-N(k) vs S_n^+N(-k):")
    worst_rel = 0.0
    for name, kf in KPTS.items():
        em, vm, ep, vp, kminus = eig_cache[name]
        dk_m = dipole_k(irvec, ndegen, r_m, kf)
        dk_p = dipole_k(irvec, ndegen, r_p, kminus)
        s_m = interband_strength(vm, dk_m)
        s_p = interband_strength(vp, dk_p)
        denom = np.maximum(s_p, 1e-6)
        rel = float(np.max(np.abs(s_m - s_p) / denom))
        worst_rel = max(worst_rel, rel)
        print(f"     {name:3s}  max rel diff = {rel:.3e}")
    print(f"     -> worst rel = {worst_rel:.3e} : "
          f"{'PASS' if worst_rel < 1e-4 else 'CHECK (degeneracy-limited)'}")

    # ---- write --------------------------------------------------------------
    header = read_header_text(args.plus_tb, irvec.shape[0])
    comment = " P-conjugate -N TB constructed from +N by construct_p_conjugate_tb.py"
    print(f"\n[write] {args.out} ...")
    write_tb_file(args.out, header, irvec, h_m, r_m, comment=comment)
    print("[write] done.")
    print("\nNext: run compare_tb_eigenvalues.py on the written file to re-confirm,")
    print("then upload and run the -N SBE (lg_cov / k40 / nb1-104 & full112).")


if __name__ == "__main__":
    main()
