"""Canonical PT-symmetric 2D AFM-TI 4-band model (MnBi2Te4 / Gao-2023 family).

basis = orbital tau (x) spin s.

    H(k;N) = M(k) (tau_z s0) + v sin kx (tau_x sx) + v sin ky (tau_x sy)
             + N (tau_z sz)
    M(k)   = m + b(2 - cos kx - cos ky)         Wilson mass (lattice gap)

The orbital Dirac (tau_x sx, tau_x sy) carries opposite spin chirality so the two
spin sectors have opposite Berry curvature -> PT protected (occupied Tr Omega = 0).
The staggered exchange N(tau_z sz) is the Neel order: it breaks P and T individually
but preserves PT, and N->-N is the magnetic-domain reversal (the P-conjugate).

Verified by auto-search (no hand-guessed operators) against the full gate battery:
PT exists; an N-flipping inversion P exists; NO fixed-N inversion and NO N-even
unitary (so the x/y current is genuinely N-odd in its even harmonics); gapped;
occupied Berry curvature ~0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model4band import (S0, SX, SY, SZ, op, find_antiunitary_fixingk,
                        find_inversion_nparity, find_inversion_fixedN,
                        find_neven_unitary, gap_value)

_T = {"zs0": op(SZ, S0), "xsx": op(SX, SX), "xsy": op(SX, SY), "zsz": op(SZ, SZ), "ysz": op(SY, SZ),
      "zs0_": op(SZ, S0)}


@dataclass
class BilayerAFM:
    v: float = 1.0
    m: float = 0.5           # trivial-insulator window, clean gap
    b: float = 0.3
    Nz: float = 0.3          # Neel exchange; domain reversal = -Nz

    def H(self, kx, ky, Nz=None):
        n = self.Nz if Nz is None else Nz
        M = self.m + self.b * (2 - np.cos(kx) - np.cos(ky))
        return (M * _T["zs0"] + self.v * np.sin(kx) * _T["xsx"]
                + self.v * np.sin(ky) * _T["xsy"] + n * _T["ysz"])

    def velocity(self, kx, ky):
        vx = self.v * np.cos(kx) * _T["xsx"] + self.b * np.sin(kx) * _T["zs0"]
        vy = self.v * np.cos(ky) * _T["xsy"] + self.b * np.sin(ky) * _T["zs0"]
        return vx, vy

    def bands(self, kx, ky, Nz=None):
        return np.linalg.eigvalsh(self.H(kx, ky, Nz))


def berry_max(model, nk=24):
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    mx = 0.0
    for kx in ks:
        for ky in ks:
            E, V = np.linalg.eigh(model.H(kx, ky))
            vx, vy = model.velocity(kx, ky)
            vxb = V.conj().T @ vx @ V; vyb = V.conj().T @ vy @ V
            om = 0.0
            for nn in range(2):
                for mb in range(2, 4):
                    de = E[mb] - E[nn]
                    if abs(de) > 1e-9:
                        om += -2 * np.imag(vxb[nn, mb] * vyb[mb, nn]) / de ** 2
            mx = max(mx, abs(om))
    return mx


# ---- generic vectorised propagator (Peierls) + N-odd current dynamics check ----
def run_J(model, Nz, comp="x", A0=0.6, w0=0.2, ncyc=10, dt=0.05, nk=20):
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    KX, KY = np.meshgrid(ks, ks, indexing="ij")
    kx = KX.ravel(); ky = KY.ravel(); nK = kx.size
    Tend = 2 * np.pi / w0 * ncyc
    nt = int(Tend / dt); tg = np.arange(nt) * dt
    env = np.where(np.abs(tg - Tend / 2) < Tend / 2, np.cos(np.pi * (tg - Tend / 2) / Tend) ** 2, 0)
    A = A0 * env * np.sin(w0 * (tg - Tend / 2))

    def Hs(a):
        kxa = kx + a
        M = model.m + model.b * (2 - np.cos(kxa) - np.cos(ky))
        return (M[:, None, None] * _T["zs0"] + (model.v * np.sin(kxa))[:, None, None] * _T["xsx"]
                + (model.v * np.sin(ky))[:, None, None] * _T["xsy"] + Nz * _T["ysz"])

    def Vs(a):
        kxa = kx + a
        if comp == "x":
            return (model.v * np.cos(kxa))[:, None, None] * _T["xsx"] + (model.b * np.sin(kxa))[:, None, None] * _T["zs0"]
        return (model.v * np.cos(ky))[:, None, None] * _T["xsy"] + (model.b * np.sin(ky))[:, None, None] * _T["zs0"]

    _, V = np.linalg.eigh(Hs(0.0)); occ = V[:, :, :2]
    rho = occ @ occ.conj().transpose(0, 2, 1)
    J = np.zeros(nt)
    for it in range(nt):
        H = Hs(A[it])
        def c(r): return -1j * (H @ r - r @ H)
        k1 = c(rho); k2 = c(rho + .5 * dt * k1); k3 = c(rho + .5 * dt * k2); k4 = c(rho + dt * k3)
        rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        J[it] = -np.real(np.einsum("kij,kji->", rho, Vs(A[it]))) / nK
    return tg, J


if __name__ == "__main__":
    mdl = BilayerAFM()
    pt = find_antiunitary_fixingk(mdl); pp = find_inversion_nparity(mdl)
    inv = find_inversion_fixedN(mdl); wev = find_neven_unitary(mdl)
    om = berry_max(mdl); gap = gap_value(mdl)
    print("=" * 64)
    print("Canonical PT-AFM 4-band (MnBi2Te4 family) - gate battery")
    print("=" * 64)
    print(f"  params v={mdl.v} m={mdl.m} b={mdl.b} N={mdl.Nz}")
    print(f"  PT exists?           {pt[0][0]} dev={pt[0][1]:.2e}   (need ~0)")
    print(f"  N-flip inversion P?  {pp[0][0]} dev={pp[0][1]:.2e}   (need ~0)")
    print(f"  Berry occ max|Omega| {om:.2e}   (PT => ~0)")
    print(f"  gap                  {gap:.4f}")
    print(f"  NO fixed-N inversion {inv[0][1]:.2e} ({inv[0][0]})   (need >>0)")
    print(f"  NO N-even unitary    {wev[0][1]:.2e} ({wev[0][0]})   (need >>0)")
    static_ok = pt[0][1] < 1e-9 and pp[0][1] < 1e-9 and om < 1e-6 and gap > 1e-3 and inv[0][1] > 1e-3 and wev[0][1] > 1e-3
    print(f"  GATE A (static): {'PASS' if static_ok else 'FAIL'}")
    # dynamics: N-odd current must be nonzero (even harmonics flip with N)
    print("\n  Dynamics Gate B (N-odd current must be physical, not machine zero):")
    for comp in ("x", "y"):
        t, Jp = run_J(mdl, +mdl.Nz, comp); _, Jm = run_J(mdl, -mdl.Nz, comp)
        Jo = .5 * (Jp - Jm); Je = .5 * (Jp + Jm)
        r = np.max(np.abs(Jo)) / max(np.max(np.abs(Je)), 1e-30)
        print(f"    J_{comp}: max|J_o(N-odd)|={np.max(np.abs(Jo)):.3e}  max|J_e|={np.max(np.abs(Je)):.3e}  |Jo|/|Je|={r:.3e}")
