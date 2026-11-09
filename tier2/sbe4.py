"""4-band SBE forward solver (Peierls/velocity gauge, exact for the lattice model).

For the FULL 4-band lattice model there is no band truncation, so the Peierls
substitution k -> k + A(t) is exact and is the simplest correct gauge (the
velocity-gauge truncation problems that forced lg_cov in the 112-band Wannier
solver do not arise here). This is the forward engine that Phase B's QRT will reuse.

    drho_k/dt = -i [H(k+A(t)), rho_k] + (T2 dephasing in instantaneous eigenbasis)
    J_a(t)    = -(1/Nk) sum_k Tr[ rho_k(t) v_a(k+A(t)) ]

Dynamics-level GATE A (selection rule): with linear field along x,
    even harmonics vanish at N=0 (P-symmetric), and J_n(-N) = (-1)^{n+1} J_n(N).
"""

from __future__ import annotations

import numpy as np

from model4band import Model4Band, op, S0, SX, SY, SZ


def cos2_field(t, A0, w0, ncyc):
    """Vector potential A(t), cos^2 envelope, linearly polarized along x."""
    T = 2 * np.pi / w0 * ncyc
    env = np.where(np.abs(t - T / 2) < T / 2, np.cos(np.pi * (t - T / 2) / T) ** 2, 0.0)
    return A0 * env * np.sin(w0 * (t - T / 2))


# constant 4x4 building blocks (vectorised model)
_M1 = op(SX, S0)            # tau_x s0   (coeff v sin kx)
_M2 = op(SY, SY)            # tau_y sy   (coeff v sin ky)
_M3 = op(SZ, S0)            # tau_z s0   (coeff Wilson mass)
_M4 = op(SZ, SZ)            # tau_z sz   (coeff Nz)
_M5 = op(S0, SZ)            # tau0 sz    (coeff lam sin kx, inversion-breaking)


def run(model, Nz, A0=0.6, w0=0.2, ncyc=10, dt=0.05, nk=24, T2=None, n_deph=4):
    """Propagate (fully vectorised over k) and return (t, Jx). Field || x."""
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    KX, KY = np.meshgrid(ks, ks, indexing="ij")
    kx = KX.ravel(); ky = KY.ravel()
    nK = kx.size
    v, m, b = model.v, model.m, model.b
    s_ky = np.sin(ky); c_ky_term = -np.cos(ky)

    Tend = 2 * np.pi / w0 * ncyc
    nt = int(Tend / dt)
    tgrid = np.arange(nt) * dt
    A = cos2_field(tgrid, A0, w0, ncyc)

    lam = model.lam

    def Hstack(a):
        kxa = kx + a
        c1 = v * np.sin(kxa)
        c3 = m + b * (2 - np.cos(kxa) + c_ky_term)
        c5 = lam * np.sin(kxa)
        return (c1[:, None, None] * _M1 + (v * s_ky)[:, None, None] * _M2
                + c3[:, None, None] * _M3 + Nz * _M4 + c5[:, None, None] * _M5)

    def Vxstack(a):
        kxa = kx + a
        return ((v * np.cos(kxa))[:, None, None] * _M1
                + (b * np.sin(kxa))[:, None, None] * _M3
                + (lam * np.cos(kxa))[:, None, None] * _M5)

    # initial rho = occupied projector at each k (batched eigh)
    _, V = np.linalg.eigh(Hstack(0.0))
    occ = V[:, :, :2]
    rho = occ @ occ.conj().transpose(0, 2, 1)

    gamma = 0.0 if T2 is None else 1.0 / T2
    Jx = np.zeros(nt)
    eye = np.eye(4, dtype=complex)
    offmask = (1.0 - eye)

    def comm(H, r):
        return -1j * (H @ r - r @ H)

    for it in range(nt):
        a = A[it]
        H = Hstack(a)
        k1 = comm(H, rho)
        k2 = comm(H, rho + 0.5 * dt * k1)
        k3 = comm(H, rho + 0.5 * dt * k2)
        k4 = comm(H, rho + dt * k3)
        rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if gamma > 0 and (it % n_deph == 0):
            damp = np.exp(-gamma * n_deph * dt)
            _, U = np.linalg.eigh(H)
            Ud = U.conj().transpose(0, 2, 1)
            rb = Ud @ rho @ U
            rb = rb * (eye + damp * offmask)
            rho = U @ rb @ Ud
        Jx[it] = -np.real(np.einsum("kij,kji->", rho, Vxstack(a))) / nK
    return tgrid, Jx, w0


def hhg(t, J, w0, window=True):
    n = J.size
    w = np.hanning(n) if window else np.ones(n)
    Jf = np.fft.rfft(J * w)
    freqs = np.fft.rfftfreq(n, d=(t[1] - t[0])) * 2 * np.pi   # angular
    order = freqs / w0
    return order, Jf


def harmonic_amp(order, Jf, n, half=0.4):
    sel = np.abs(order - n) <= half
    if not sel.any():
        return 0.0 + 0j
    idx = np.where(sel)[0]
    return Jf[idx[np.argmax(np.abs(Jf[idx]))]]


if __name__ == "__main__":
    mdl = Model4Band()
    print("4-band SBE selection-rule test (field || x, T2 off for clean check)")
    t, Jp, w0 = run(mdl, Nz=+mdl.Nz)
    _, Jm, _ = run(mdl, Nz=-mdl.Nz)
    _, J0, _ = run(mdl, Nz=0.0)
    op, Fp = hhg(t, Jp, w0)
    _, Fm = hhg(t, Jm, w0)
    o0, F0 = hhg(t, J0, w0)
    print(f"{'H':>3} {'|J(+N)|':>11} {'|J(-N)|':>11} {'|J(0)|':>11} {'parity':>6} "
          f"{'J(-N)/J(N)':>11} {'expect':>7}")
    for n in range(1, 8):
        ap = harmonic_amp(op, Fp, n); am = harmonic_amp(op, Fm, n); a0 = harmonic_amp(o0, F0, n)
        ratio = (am / ap).real if abs(ap) > 1e-14 else float("nan")
        exp = -1 if n % 2 == 0 else +1   # J_n(-N)=(-1)^{n+1}J_n(N)
        par = "EVEN" if n % 2 == 0 else "odd"
        print(f"{n:>3} {abs(ap):>11.3e} {abs(am):>11.3e} {abs(a0):>11.3e} {par:>6} "
              f"{ratio:>11.3f} {exp:>7}")
    # gate: even harmonics vanish at N=0, and J_n(-N)=(-1)^{n+1}J_n(N)
    ev0 = max(abs(harmonic_amp(o0, F0, n)) for n in (2, 4, 6))
    od0 = max(abs(harmonic_amp(o0, F0, n)) for n in (1, 3, 5))
    print(f"\n  N=0 even/odd ratio = {ev0/od0:.2e}  (expect <<1: even forbidden at N=0)")
