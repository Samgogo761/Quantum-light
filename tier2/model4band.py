"""4-band minimal model for bilayer AFM-CrI3 (Tier-2 quantum-signature engine).

Route 1: a manifestly PT-symmetric model. We build H(k;N) REAL for all k in an
orbital(x)spin basis, so PT = K (complex conjugation) is an exact antiunitary
symmetry by construction => the Berry curvature vanishes pointwise (the defining
feature of the user's PT-symmetric AFM), while the quantum metric survives.

    H(k;N) = v sin kx (tau_x s0)            intra Dirac (real)
           + v sin ky (tau_y sy)            intra Dirac, real combination i(x)i
           + [m + b(2-cos kx-cos ky)] (tau_z s0)   Wilson mass / gap
           + Nz (tau_z sz)                  staggered (AFM) Neel exchange, real

tau = orbital pseudospin, s = real spin. tau_y(x)sy is real (imag(x)imag), so the
whole H is real symmetric. Magnetic-domain reversal is N -> -N; the symmetry
operators (PT, and the inversion P giving the N-parity relation) are found by an
explicit numerical search over the 16 Pauli bilinears, so we never hand-guess them.

AFM analogue of the SSH toy model in Ilin et al. (arXiv:2604.20388). Parameters are
placeholders; fit_from_wf.py (route 2) will set them from the 112-WF model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

S0 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"0": S0, "x": SX, "y": SY, "z": SZ}


def op(tau, sig):
    return np.kron(tau, sig)


# all 16 Pauli bilinears tau_i (x) s_j, labelled, for the symmetry search
BILINEARS = {f"t{a}s{b}": op(PAULI[a], PAULI[b]) for a in "0xyz" for b in "0xyz"}


@dataclass
class Model4Band:
    v: float = 1.0
    m: float = 0.3
    b: float = 0.5
    Nz: float = 0.4              # Neel exchange (AFM along z); domain reversal = -Nz
    lam: float = 0.0            # DEPRECATED non-magnetic inversion-breaking term. Setting
                                #   lam>0 breaks inversion N-INDEPENDENTLY (even harmonics
                                #   appear but are N-even, and N=0 centrosymmetry is lost).
                                #   Gate B (2026-06-25) showed J_o=(J(+N)-J(-N))/2 = 1e-15:
                                #   this orbital(x)spin model is the WRONG structure. The
                                #   correct minimal model needs tau=LAYER, inversion = layer
                                #   swap (tau_x), Neel = tau_z(x)spin (odd under layer swap),
                                #   and a PT-symmetric DOUBLED Dirac (single Dirac has Berry).
                                #   See model4band_bilayer.py (next iteration).

    def H(self, kx, ky, Nz=None):
        nz = self.Nz if Nz is None else Nz
        h = self.v * np.sin(kx) * op(SX, S0)
        h += self.v * np.sin(ky) * op(SY, SY)
        h += (self.m + self.b * (2 - np.cos(kx) - np.cos(ky))) * op(SZ, S0)
        h += nz * op(SZ, SZ)
        h += self.lam * np.sin(kx) * op(S0, SZ)
        return h

    def velocity(self, kx, ky):
        vx = (self.v * np.cos(kx) * op(SX, S0) + self.b * np.sin(kx) * op(SZ, S0)
              + self.lam * np.cos(kx) * op(S0, SZ))
        vy = self.v * np.cos(ky) * op(SY, SY) + self.b * np.sin(ky) * op(SZ, S0)
        return vx, vy

    def bands(self, kx, ky, Nz=None):
        return np.linalg.eigvalsh(self.H(kx, ky, Nz))


# ---------- symmetry verification (no hand-guessed operators) ----------

def find_antiunitary_fixingk(model, nk=9):
    """Search for U s.t. U H(k)* U^-1 = H(k) at the SAME k (an antiunitary U*K that
    keeps k fixed -> PT-type). Return list of (label, max_dev) for the best matches."""
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    hk = [model.H(kx, ky) for kx in ks for ky in ks]
    out = []
    for lab, U in BILINEARS.items():
        Ui = U.conj().T
        dev = max(np.max(np.abs(U @ h.conj() @ Ui - h)) for h in hk)
        out.append((lab, dev))
    out.sort(key=lambda x: x[1])
    return out


def find_inversion_nparity(model, nk=9):
    """Search for unitary P s.t. P H(k;+N) P^-1 = H(-k;-N) (magnetic-domain reversal as
    spatial inversion). Return sorted (label, max_dev)."""
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    pairs = [(kx, ky) for kx in ks for ky in ks]
    out = []
    for lab, P in BILINEARS.items():
        Pi = P.conj().T
        dev = 0.0
        for kx, ky in pairs:
            lhs = P @ model.H(kx, ky, Nz=model.Nz) @ Pi
            rhs = model.H(-kx, -ky, Nz=-model.Nz)
            dev = max(dev, np.max(np.abs(lhs - rhs)))
        out.append((lab, dev))
    out.sort(key=lambda x: x[1])
    return out


def berry_curvature_check(model, nk=24):
    """PT => occupied-manifold Berry curvature = 0. Kubo sum over occ->unocc
    (inter-manifold, so no intra-occupied degeneracy in the denominator).
    For a real Hamiltonian the velocity matrix elements are real => Omega = 0
    exactly; this returns max|Omega| as the numerical check."""
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    nocc = 2
    maxom = 0.0
    for kx in ks:
        for ky in ks:
            E, V = np.linalg.eigh(model.H(kx, ky))
            vx, vy = model.velocity(kx, ky)
            vxb = V.conj().T @ vx @ V
            vyb = V.conj().T @ vy @ V
            om = 0.0
            for n in range(nocc):
                for mb in range(nocc, 4):
                    de = E[mb] - E[n]
                    if abs(de) > 1e-9:
                        om += -2.0 * np.imag(vxb[n, mb] * vyb[mb, n]) / de ** 2
            maxom = max(maxom, abs(om))
    return maxom


def find_inversion_fixedN(model, nk=9):
    """Search for unitary P' with P' H(k;N) P'^{-1} = H(-k;N) (inversion at FIXED N).
    For genuine magnetic inversion-breaking this must NOT exist => best dev should be
    well away from 0. Returns sorted (label, dev)."""
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    pairs = [(kx, ky) for kx in ks for ky in ks]
    out = []
    for lab, P in BILINEARS.items():
        Pi = P.conj().T
        dev = max(np.max(np.abs(P @ model.H(kx, ky) @ Pi - model.H(-kx, -ky)))
                  for kx, ky in pairs)
        out.append((lab, dev))
    out.sort(key=lambda x: x[1])
    return out


def find_neven_unitary(model, nk=9):
    """Search for unitary W with W H(k;N) W^{-1} = H(k;-N) at the SAME k. If it exists,
    the current is N-even and the N-parity physics is absent => must NOT exist."""
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    pairs = [(kx, ky) for kx in ks for ky in ks]
    out = []
    for lab, W in BILINEARS.items():
        Wi = W.conj().T
        dev = max(np.max(np.abs(W @ model.H(kx, ky, Nz=model.Nz) @ Wi
                                - model.H(kx, ky, Nz=-model.Nz)))
                  for kx, ky in pairs)
        out.append((lab, dev))
    out.sort(key=lambda x: x[1])
    return out


def gap_value(model, nk=41):
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    return min(model.bands(kx, ky)[2] - model.bands(kx, ky)[1] for kx in ks for ky in ks)


if __name__ == "__main__":
    mdl = Model4Band()
    pt = find_antiunitary_fixingk(mdl)
    pp = find_inversion_nparity(mdl)
    om = berry_curvature_check(mdl)
    gap = gap_value(mdl)
    print("=" * 64)
    print("4-band PT-symmetric AFM model - symmetry verification (auto-search)")
    print("=" * 64)
    print(f"  params: v={mdl.v}, m={mdl.m}, b={mdl.b}, Nz={mdl.Nz}")
    print(f"  best PT-type (U H* U^-1 = H @ same k):  {pt[0][0]}  dev={pt[0][1]:.2e}")
    print(f"     (next: {pt[1][0]} dev={pt[1][1]:.2e},  {pt[2][0]} dev={pt[2][1]:.2e})")
    print(f"  best inversion P (P H(k,N) P^-1 = H(-k,-N)): {pp[0][0]}  dev={pp[0][1]:.2e}")
    print(f"     (next: {pp[1][0]} dev={pp[1][1]:.2e},  {pp[2][0]} dev={pp[2][1]:.2e})")
    print(f"  occupied Berry curvature max|Omega| = {om:.2e}  (PT => ~0)")
    print(f"  direct gap = {gap:.4f}")
    inv = find_inversion_fixedN(mdl)
    wev = find_neven_unitary(mdl)
    print(f"  NO fixed-N inversion P'? best dev = {inv[0][1]:.2e} ({inv[0][0]})  (must be >>0)")
    print(f"  NO N-even unitary W?     best dev = {wev[0][1]:.2e} ({wev[0][0]})  (must be >>0)")
    ok = (pt[0][1] < 1e-9 and pp[0][1] < 1e-9 and om < 1e-6 and gap > 1e-3
          and inv[0][1] > 1e-3 and wev[0][1] > 1e-3)
    print("  GATE A: " + ("PASS" if ok else "FAIL"))
