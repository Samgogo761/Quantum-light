"""Independent acceptance tests for the solo-v1 static-energy spectrum."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ql_tip_1d.bias_v1 import dc_field_au, dc_potential_au, make_grid, right_mask, static_potential_au
from ql_tip_1d.core import A0_NM, EH_EV, CrankNicolson, Grid, bound_states, hamiltonian_diagonals, norm, surface_potential, well_grid
from ql_tip_1d.spectrum_v1 import direct_small_matrix_spectrum, free_gaussian_fd_reference, integrate_above, spectrum_l1_difference, window_spectrum


def test_embedded_protocol_config_constructs_grid_and_static_potential() -> None:
    """Regression: generated run configs nest model fields under protocol."""
    protocol_path = Path(__file__).resolve().parents[1] / "configs/protocol_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = {"protocol": protocol, "numerics": protocol["numerics"], "F_dc_V_per_nm": 1.0}
    grid = make_grid(config)
    potential = static_potential_au(grid, config)
    assert potential.shape == grid.x_au.shape
    assert potential[-1] < 0.0


def test_small_matrix_direct_eigendecomposition() -> None:
    """Two solves agree with direct eigh and the identical broadening kernel."""
    rng = np.random.default_rng(20261104)
    n = 160
    x = np.arange(n) * 0.1
    diag = 1.0 / 0.1**2 + 0.03 * np.cos(x / 7.0)
    off = np.full(n - 1, -0.5 / 0.1**2)
    phi = np.exp(-((x - 6.0) / 1.5) ** 2 + 0.7j * x)
    energies = np.linspace(-50.0, 300.0, 3501)
    eta = 0.05 * EH_EV
    direct = direct_small_matrix_spectrum(diag, off, phi, 0.1, energies, eta)
    filtered = window_spectrum(diag, off, phi, 0.1, energies, eta)
    difference = spectrum_l1_difference(energies, direct, filtered)
    norm_integral = np.trapezoid(direct, energies)
    assert difference / norm_integral < 1e-8
    assert abs(np.trapezoid(filtered, energies) - norm(phi, 0.1)) / norm(phi, 0.1) < 2e-3


def test_constant_potential_shift_preserves_spectrum() -> None:
    """H->H+C and E->E+C must not change the absolute spectrum."""
    n = 120
    dx = 0.15
    x = np.arange(n) * dx
    diag = 1.0 / dx**2 + 0.02 * np.sin(x / 4.0)
    off = np.full(n - 1, -0.5 / dx**2)
    phi = np.exp(-((x - 8.0) / 2.0) ** 2 + 0.4j * x)
    energies = np.linspace(-0.1, 0.8, 180)
    eta = 0.07 * EH_EV
    shift = 0.137
    base = window_spectrum(diag, off, phi, dx, energies, eta)
    shifted = window_spectrum(diag + shift, off, phi, dx, energies + shift * EH_EV, eta)
    np.testing.assert_allclose(base, shifted, rtol=2e-11, atol=2e-13)


def test_free_right_gaussian_known_fd_momentum_distribution() -> None:
    """The filtered spectrum matches the known FD-dispersion Gaussian within 1%."""
    dx = 0.4
    n = 2048
    x = (np.arange(n) - n // 2) * dx
    sigma = 4.0
    p0 = 0.8
    phi = (2.0 * np.pi * sigma**2) ** (-0.25) * np.exp(-((x + 12.0) ** 2) / (4.0 * sigma**2) + 1j * p0 * (x + 12.0))
    diag = np.full(n, 1.0 / dx**2)
    off = np.full(n - 1, -0.5 / dx**2)
    energies = np.linspace(-5.0, 40.0, 901)
    eta = 0.03 * EH_EV
    filtered = window_spectrum(diag, off, phi, dx, energies, eta)
    reference = free_gaussian_fd_reference(energies, p0, sigma, dx, eta)
    difference = spectrum_l1_difference(energies, filtered, reference)
    assert difference / norm(phi, dx) < 0.01
    threshold = 0.2 * EH_EV
    y_filtered = integrate_above(energies, filtered, threshold)
    y_reference = integrate_above(energies, reference, threshold)
    assert abs(y_filtered - y_reference) / y_reference < 0.01


def test_dc_outgoing_packet_conserves_total_static_energy_spectrum() -> None:
    """A DC force changes kinetic/potential energy, not the H_dc spectrum."""
    dx_nm = 0.05
    x_nm = np.arange(-10.0, 80.0, dx_nm) + dx_nm / 2.0
    grid = Grid(x_nm / A0_NM, dx_nm / A0_NM, 0)
    potential = dc_potential_au(grid.x_au, 1.0)
    diag, off = hamiltonian_diagonals(grid, potential)
    sigma = 1.0 / A0_NM
    p0 = 0.5
    centre = 20.0 / A0_NM
    initial = (2.0 * np.pi * sigma**2) ** (-0.25) * np.exp(-((grid.x_au - centre) ** 2) / (4.0 * sigma**2) + 1j * p0 * grid.x_au)
    initial /= np.sqrt(norm(initial, grid.dx_au))
    cn = CrankNicolson(grid, 0.1, potential)
    final = initial.copy()
    for _ in range(827):
        final = cn.step(final)
    energies = np.linspace(-1.0, 15.0, 321)
    eta = 0.1
    before = window_spectrum(diag, off, initial, grid.dx_au, energies, eta)
    after = window_spectrum(diag, off, final, grid.dx_au, energies, eta)
    assert spectrum_l1_difference(energies, before, after) < 0.01
    energy_before = np.real(np.vdot(initial, diag * initial) * grid.dx_au)
    energy_before += np.real(np.vdot(initial[:-1], off * initial[1:]) * 2.0 * grid.dx_au)
    energy_after = np.real(np.vdot(final, diag * final) * grid.dx_au)
    energy_after += np.real(np.vdot(final[:-1], off * final[1:]) * 2.0 * grid.dx_au)
    assert abs(energy_after - energy_before) < 1e-8


def test_dc_force_direction_and_ground_state_mask_leakage() -> None:
    """Positive F extracts toward +x; a masked bound tail is not high yield."""
    x = np.linspace(0.5, 3.0, 20) / A0_NM
    assert np.all(dc_field_au(x, 1.0) < 0.0)
    potential = dc_potential_au(x, 1.0)
    assert np.all(np.diff(potential) < 0.0)
    grid = well_grid(24, xmin_nm=-2.0, xmax_nm=20.0)
    surface = surface_potential(grid.x_au)
    _, states = bound_states(grid, surface)
    phi = right_mask(grid) * states[:, 0]
    diag, off = hamiltonian_diagonals(grid, surface)
    energies = np.linspace(-2.0, 5.0, 141)
    spectrum = window_spectrum(diag, off, phi, grid.dx_au, energies, 0.2)
    assert norm(phi, grid.dx_au) < 1e-12
    assert integrate_above(energies, spectrum, 0.7170816) < 1e-12
