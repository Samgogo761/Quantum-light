"""Auditable CPU primitives. Internal length, energy and time are atomic units.

The production finite-difference Hamiltonian and Crank-Nicolson propagator use
the same kinetic operator. FFT propagation is a separate free-particle check.
No qmsolve code is copied.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.constants import physical_constants, c, epsilon_0, h, electron_volt
from scipy.linalg import eigh_tridiagonal
from scipy.linalg.lapack import zgttrf, zgttrs
from scipy.optimize import brentq

A0_NM = physical_constants['Bohr radius'][0] * 1e9
EH_EV = physical_constants['Hartree energy in eV'][0]
TAU_FS = physical_constants['atomic unit of time'][0] * 1e15
FIELD_V_M = physical_constants['atomic unit of electric field'][0]


@dataclass(frozen=True)
class Grid:
    x_au: np.ndarray
    dx_au: float
    well_cells: int

    @property
    def x_nm(self):
        return self.x_au * A0_NM


def well_grid(well_cells=48, xmin_nm=-2.0, xmax_nm=2.0, width_nm=0.135):
    """Cell centers; both well interfaces are cell faces, exact width at every dx.

    Dirichlet zero ghost nodes lie one step beyond the first and last centers.
    Requested limits are rounded outward to cell faces.
    """
    if well_cells < 2 or xmin_nm >= -width_nm or xmax_nm <= 0:
        raise ValueError('Invalid well grid')
    dx_nm = width_nm / well_cells
    nl, nr = int(np.ceil(-xmin_nm / dx_nm)), int(np.ceil(xmax_nm / dx_nm))
    x_nm = (np.arange(nl + nr) - nl + 0.5) * dx_nm
    return Grid(x_nm / A0_NM, dx_nm / A0_NM, well_cells)


def surface_potential(x_au, width_nm=0.135, depth_ev=10.8):
    x_nm = np.asarray(x_au) * A0_NM
    return np.where((x_nm >= -width_nm) & (x_nm <= 0), -depth_ev / EH_EV, 0.)


def hamiltonian_diagonals(grid, potential_au):
    diag = 1 / grid.dx_au**2 + np.asarray(potential_au)
    off = np.full(len(diag) - 1, -0.5 / grid.dx_au**2)
    return diag, off


def bound_states(grid, potential_au, depth_ev=10.8):
    diag, off = hamiltonian_diagonals(grid, potential_au)
    energies, vectors = eigh_tridiagonal(
        diag, off, select='v', select_range=(-depth_ev / EH_EV - 1e-9, 0.),
        lapack_driver='stebz', tol=1e-13)
    vectors /= np.sqrt(grid.dx_au)
    for k in range(vectors.shape[1]):
        if vectors[len(vectors) // 2, k] < 0:
            vectors[:, k] *= -1
    return energies, vectors


def analytic_well(width_nm=0.135, depth_ev=10.8):
    """Even ground state of full-width a symmetric finite square well.

    z tan(z) = sqrt(z0^2-z^2), z0=(a/2)*sqrt(2 V0).
    Count is exact for z0<pi/2 used here: one even state, no odd state.
    """
    half = width_nm / A0_NM / 2
    depth = depth_ev / EH_EV
    z0 = half * np.sqrt(2 * depth)
    z = brentq(lambda z: z*np.tan(z) - np.sqrt(z0*z0-z*z),
               1e-14, min(z0, np.pi/2)-1e-14, xtol=5e-15)
    k, kappa = z / half, np.sqrt(z0*z0-z*z) / half
    norm_integral = half + np.sin(2*k*half)/(2*k) + np.cos(k*half)**2/kappa
    return {'energy_au': -0.5*kappa*kappa, 'z0': z0, 'z': z,
            'k_au': k, 'kappa_au': kappa, 'half_width_au': half,
            'amplitude_per_sqrt_bohr': 1/np.sqrt(norm_integral),
            'bound_state_count': 1 if z0 < np.pi/2 else None}


def analytic_wavefunction(x_au, width_nm=0.135, depth_ev=10.8):
    p = analytic_well(width_nm, depth_ev)
    y = np.abs(np.asarray(x_au) + p['half_width_au'])
    d, k, kap = p['half_width_au'], p['k_au'], p['kappa_au']
    # clip exponent to avoid overflows in the unused branch of np.where
    return p['amplitude_per_sqrt_bohr'] * np.where(
        y <= d, np.cos(k*y), np.cos(k*d)*np.exp(-kap*np.maximum(y-d, 0)))


class CrankNicolson:
    """Midpoint CN for a real or complex diagonal potential and FD kinetic term.

    Static potentials factor once. The time-dependent method refactors at each
    midpoint and supports -i W CAP. Returns the unrenormalized wavefunction.
    """
    def __init__(self, grid, dt_au, potential_au):
        self.grid, self.dt = grid, float(dt_au)
        self.off_h = -0.5 / grid.dx_au**2
        self.kin_diag = 1 / grid.dx_au**2
        self._factor(np.asarray(potential_au, dtype=complex))

    def _factor(self, potential):
        self.hdiag = self.kin_diag + potential
        n = len(potential)
        aoff = np.full(n-1, 0.5j*self.dt*self.off_h, dtype=complex)
        adiag = 1 + 0.5j*self.dt*self.hdiag
        self.factor = zgttrf(aoff.copy(), adiag.copy(), aoff.copy())
        if self.factor[-1] != 0:
            raise RuntimeError(f'zgttrf info={self.factor[-1]}')

    def step(self, psi, potential_midpoint_au=None):
        if potential_midpoint_au is not None:
            self._factor(np.asarray(potential_midpoint_au, dtype=complex))
        rhs = (1-0.5j*self.dt*self.hdiag)*psi
        rhs[:-1] -= 0.5j*self.dt*self.off_h*psi[1:]
        rhs[1:] -= 0.5j*self.dt*self.off_h*psi[:-1]
        out, info = zgttrs(*self.factor[:-1], rhs[:, None])
        if info:
            raise RuntimeError(f'zgttrs info={info}')
        return out[:, 0]


def norm(psi, dx_au):
    return float(np.vdot(psi, psi).real * dx_au)


def integrated_near_field(x_au, beta=3., ell_nm=20.):
    """Integral_0^x xi(s) ds in bohr; derivative is xi on vacuum side."""
    xp = np.maximum(np.asarray(x_au), 0.)
    ell = ell_nm / A0_NM
    return xp - (beta-1)*ell*np.expm1(-xp/ell)


def enhancement(x_au, beta=3., ell_nm=20.):
    xp = np.maximum(np.asarray(x_au), 0.)
    return np.where(np.asarray(x_au) >= 0, 1+(beta-1)*np.exp(-xp/(ell_nm/A0_NM)), 0.)


def pulse(t_au, intensity_w_cm2, wavelength_nm=800., tau_intensity_fs=15., cep=0.):
    """E in atomic units; positive E exerts negative force on the electron."""
    e0 = np.sqrt(2*intensity_w_cm2*1e4/(c*epsilon_0))/FIELD_V_M
    photon_ev = h*c/(wavelength_nm*1e-9)/electron_volt
    omega_au = photon_ev / EH_EV
    return e0*np.exp(-2*np.log(2)*(np.asarray(t_au)*TAU_FS/tau_intensity_fs)**2)*np.cos(omega_au*t_au+cep)


def fft_free_step(psi, dx_au, dt_au):
    """Periodic spectral exp(-i k^2 dt / 2), unitary NumPy FFT convention."""
    k = 2*np.pi*np.fft.fftfreq(len(psi), d=dx_au)
    return np.fft.ifft(np.exp(-0.5j*k*k*dt_au)*np.fft.fft(psi))
