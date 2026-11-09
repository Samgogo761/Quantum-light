"""Static-Hamiltonian energy-window spectra with independent test helpers.

No dense matrix is formed by :func:`window_spectrum`; production work consists
of two complex tridiagonal solves per energy.  Energies passed to and returned
from that function are in eV, while the Hamiltonian and wavefunction use atomic
units.  The returned spectrum is per eV.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.linalg.lapack import zgttrf, zgttrs

from .core import EH_EV


def kernel_hartree(delta_hartree: np.ndarray, eta_hartree: float) -> np.ndarray:
    """Normalised window kernel 2 eta^3/[pi(z^2+eta^2)^2]."""
    z = np.asarray(delta_hartree)
    return 2.0 * eta_hartree**3 / (np.pi * (z**2 + eta_hartree**2) ** 2)


def window_spectrum(
    diag_au: np.ndarray,
    off_au: np.ndarray,
    phi: np.ndarray,
    dx_au: float,
    energies_eV: np.ndarray,
    eta_eV: float,
) -> np.ndarray:
    """Evaluate the static energy-window spectrum on a tridiagonal grid.

    Args:
        diag_au: Real static-Hamiltonian diagonal in Hartree.
        off_au: Real nearest-neighbour off-diagonal in Hartree.
        phi: Masked, unnormalised wavefunction in bohr^(-1/2).
        dx_au: Uniform grid spacing in bohr.
        energies_eV: Total static-Hamiltonian energies in eV.
        eta_eV: Window-kernel parameter in eV (not an instrument FWHM).

    Returns:
        Non-negative spectral density per eV.

    Raises:
        RuntimeError: If a tridiagonal factorisation or solve fails.
        ValueError: If shapes are inconsistent or a result is non-finite.
    """
    diag = np.asarray(diag_au, dtype=float)
    off = np.asarray(off_au, dtype=float)
    state = np.asarray(phi, dtype=complex)
    energies = np.asarray(energies_eV, dtype=float)
    if diag.ndim != 1 or off.shape != (len(diag) - 1,) or state.shape != diag.shape:
        raise ValueError("Inconsistent tridiagonal Hamiltonian/wavefunction shapes")
    if dx_au <= 0.0 or eta_eV <= 0.0:
        raise ValueError("dx and eta must be positive")
    eta = eta_eV / EH_EV
    prefactor = 2.0 * eta**3 * dx_au / np.pi
    result_hartree = np.empty(len(energies), dtype=float)
    for index, energy_eV in enumerate(energies):
        energy = energy_eV / EH_EV
        off_complex = off.astype(complex)
        factor_plus = zgttrf(off_complex.copy(), diag - energy + 1j * eta, off_complex.copy())
        if factor_plus[-1] != 0:
            raise RuntimeError(f"First zgttrf failed at E={energy_eV}: info={factor_plus[-1]}")
        u, info = zgttrs(*factor_plus[:-1], state[:, None])
        if info:
            raise RuntimeError(f"First zgttrs failed at E={energy_eV}: info={info}")
        factor_minus = zgttrf(off_complex.copy(), diag - energy - 1j * eta, off_complex.copy())
        if factor_minus[-1] != 0:
            raise RuntimeError(f"Second zgttrf failed at E={energy_eV}: info={factor_minus[-1]}")
        v, info = zgttrs(*factor_minus[:-1], u)
        if info:
            raise RuntimeError(f"Second zgttrs failed at E={energy_eV}: info={info}")
        result_hartree[index] = prefactor * np.vdot(v[:, 0], v[:, 0]).real
    result = result_hartree / EH_EV
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite energy-window spectrum")
    return result


def direct_small_matrix_spectrum(
    diag_au: np.ndarray,
    off_au: np.ndarray,
    phi: np.ndarray,
    dx_au: float,
    energies_eV: np.ndarray,
    eta_eV: float,
) -> np.ndarray:
    """Independent eigendecomposition plus kernel convolution for small matrices."""
    eigenvalues, eigenvectors = eigh_tridiagonal(np.asarray(diag_au), np.asarray(off_au))
    # eigh vectors have Euclidean norm one, whereas physical norm includes dx.
    weights = dx_au * np.abs(eigenvectors.conj().T @ np.asarray(phi)) ** 2
    delta = (np.asarray(energies_eV)[:, None] - eigenvalues[None, :] * EH_EV) / EH_EV
    broadened_hartree = kernel_hartree(delta, eta_eV / EH_EV) @ weights
    return broadened_hartree / EH_EV


def integrate_above(energies_eV: np.ndarray, spectrum_per_eV: np.ndarray, threshold_eV: float) -> float:
    """Trapezoidally integrate above a threshold, inserting its interpolated value."""
    energy = np.asarray(energies_eV, dtype=float)
    spectrum = np.asarray(spectrum_per_eV, dtype=float)
    if energy.ndim != 1 or spectrum.shape != energy.shape or len(energy) < 2:
        raise ValueError("A one-dimensional energy/spectrum pair with at least two points is required")
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("Energy grid must be strictly increasing")
    if threshold_eV >= energy[-1]:
        return 0.0
    if threshold_eV <= energy[0]:
        return float(np.trapezoid(spectrum, energy))
    index = int(np.searchsorted(energy, threshold_eV))
    interpolated = np.interp(threshold_eV, energy[index - 1 : index + 1], spectrum[index - 1 : index + 1])
    x = np.concatenate(([threshold_eV], energy[index:]))
    y = np.concatenate(([interpolated], spectrum[index:]))
    return float(np.trapezoid(y, x))


def spectrum_l1_difference(
    energies_eV: np.ndarray,
    first_per_eV: np.ndarray,
    second_per_eV: np.ndarray,
) -> float:
    """Return integral |S1-S2| dE, preserving the absolute probability scale."""
    return float(np.trapezoid(np.abs(np.asarray(first_per_eV) - np.asarray(second_per_eV)), energies_eV))


def free_gaussian_fd_reference(
    energies_eV: np.ndarray,
    p0_au: float,
    sigma_au: float,
    dx_au: float,
    eta_eV: float,
    quadrature_order: int = 200,
) -> np.ndarray:
    """Analytic Gaussian momentum distribution with the discrete FD dispersion.

    Gauss-Hermite quadrature integrates the known momentum distribution against
    exactly the same broadening kernel as the resolvent implementation.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    momenta = p0_au + nodes / (np.sqrt(2.0) * sigma_au)
    kinetic_hartree = (1.0 - np.cos(momenta * dx_au)) / dx_au**2
    eta_hartree = eta_eV / EH_EV
    energies_hartree = np.asarray(energies_eV)[:, None] / EH_EV
    delta = energies_hartree - kinetic_hartree[None, :]
    broadened = kernel_hartree(delta, eta_hartree) @ (weights / np.sqrt(np.pi))
    return broadened / EH_EV


def write_spectrum_csv(
    path: Path,
    run_id: str,
    energies_eV: np.ndarray,
    spectrum_per_eV: np.ndarray,
    eta_eV: float,
    u_infinity_eV: float,
) -> None:
    """Save both corrected total energy and asymptotic kinetic energy."""
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["E_corr_eV", "K_infinity_eV", "S_per_eV", "eta_eV", "run_id"])
        for energy, value in zip(energies_eV, spectrum_per_eV):
            writer.writerow([f"{energy:.17g}", f"{energy - u_infinity_eV:.17g}", f"{value:.17g}", f"{eta_eV:.17g}", run_id])
