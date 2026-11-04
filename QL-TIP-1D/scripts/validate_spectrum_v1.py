"""Save quantitative independent P1 spectrum-method acceptance evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from ql_tip_1d.bias_v1 import dc_potential_au, right_mask, write_json_new
from ql_tip_1d.core import A0_NM, EH_EV, CrankNicolson, Grid, bound_states, hamiltonian_diagonals, norm, surface_potential, well_grid
from ql_tip_1d.spectrum_v1 import direct_small_matrix_spectrum, free_gaussian_fd_reference, integrate_above, spectrum_l1_difference, window_spectrum


def main() -> int:
    """Run independent method tests and write their absolute numerical errors."""
    start = time.perf_counter()
    output: dict[str, object] = {"method": "static H_dc resolvent energy-window filter", "tests": {}}
    tests: dict[str, object] = output["tests"]  # type: ignore[assignment]

    dx = 0.1
    n = 160
    x = np.arange(n) * dx
    diag = 1.0 / dx**2 + 0.03 * np.cos(x / 7.0)
    off = np.full(n - 1, -0.5 / dx**2)
    phi = np.exp(-((x - 6.0) / 1.5) ** 2 + 0.7j * x)
    energies = np.linspace(-50.0, 300.0, 3501)
    eta = 0.05 * EH_EV
    direct = direct_small_matrix_spectrum(diag, off, phi, dx, energies, eta)
    filtered = window_spectrum(diag, off, phi, dx, energies, eta)
    l1 = spectrum_l1_difference(energies, direct, filtered)
    tests["small_matrix"] = {"N": n, "relative_L1": l1 / norm(phi, dx), "spectrum_integral": float(np.trapezoid(filtered, energies)),
                             "masked_norm": norm(phi, dx), "pass": l1 / norm(phi, dx) < 1e-8}

    shift = 0.137
    shift_energies = np.linspace(-0.1, 0.8, 180)
    shift_eta = 0.07 * EH_EV
    base = window_spectrum(diag, off, phi, dx, shift_energies, shift_eta)
    shifted = window_spectrum(diag + shift, off, phi, dx, shift_energies + shift * EH_EV, shift_eta)
    shift_error = float(np.max(np.abs(base - shifted)))
    tests["constant_shift"] = {"shift_hartree": shift, "max_absolute_spectrum_difference_per_eV": shift_error,
                               "pass": shift_error < 1e-10}

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
    threshold = 0.2 * EH_EV
    y_filter = integrate_above(energies, filtered, threshold)
    y_reference = integrate_above(energies, reference, threshold)
    free_l1 = spectrum_l1_difference(energies, filtered, reference) / norm(phi, dx)
    free_y_error = abs(y_filter - y_reference) / y_reference
    tests["free_gaussian"] = {"eta_eV": eta, "relative_L1": free_l1, "Y_filter": y_filter,
                              "Y_reference": y_reference, "relative_Y_error": free_y_error,
                              "pass": free_l1 < 0.01 and free_y_error < 0.01}

    dx_nm = 0.05
    x_nm = np.arange(-10.0, 80.0, dx_nm) + dx_nm / 2.0
    grid = Grid(x_nm / A0_NM, dx_nm / A0_NM, 0)
    potential = dc_potential_au(grid.x_au, 1.0)
    diag, off = hamiltonian_diagonals(grid, potential)
    sigma = 1.0 / A0_NM
    centre = 20.0 / A0_NM
    initial = (2.0 * np.pi * sigma**2) ** (-0.25) * np.exp(-((grid.x_au - centre) ** 2) / (4.0 * sigma**2) + 0.5j * grid.x_au)
    initial /= np.sqrt(norm(initial, grid.dx_au))
    final = initial.copy()
    cn = CrankNicolson(grid, 0.1, potential)
    for _ in range(827):
        final = cn.step(final)
    energies = np.linspace(-1.0, 15.0, 321)
    before = window_spectrum(diag, off, initial, grid.dx_au, energies, 0.1)
    after = window_spectrum(diag, off, final, grid.dx_au, energies, 0.1)
    dc_l1 = spectrum_l1_difference(energies, before, after)
    energy_before = float(np.real(np.vdot(initial, diag * initial) * grid.dx_au + 2.0 * np.vdot(initial[:-1], off * initial[1:]) * grid.dx_au))
    energy_after = float(np.real(np.vdot(final, diag * final) * grid.dx_au + 2.0 * np.vdot(final[:-1], off * final[1:]) * grid.dx_au))
    tests["dc_outgoing"] = {"relative_L1": dc_l1, "total_energy_initial_hartree": energy_before,
                            "total_energy_final_hartree": energy_after, "absolute_total_energy_error_hartree": abs(energy_after - energy_before),
                            "pass": dc_l1 < 0.01 and abs(energy_after - energy_before) < 1e-8}

    grid = well_grid(24, xmin_nm=-2.0, xmax_nm=20.0)
    surface = surface_potential(grid.x_au)
    _, states = bound_states(grid, surface)
    phi = right_mask(grid) * states[:, 0]
    diag, off = hamiltonian_diagonals(grid, surface)
    energies = np.linspace(-2.0, 5.0, 141)
    spectrum = window_spectrum(diag, off, phi, grid.dx_au, energies, 0.2)
    bound_y = integrate_above(energies, spectrum, 0.7170816000000001)
    tests["bound_mask_leakage"] = {"masked_norm": norm(phi, grid.dx_au), "Y_H": bound_y,
                                   "pass": bound_y < 1e-12}

    dx_nm = 0.02
    x_nm = np.arange(0.0, 20.0, dx_nm) + dx_nm / 2.0
    grid = Grid(x_nm / A0_NM, dx_nm / A0_NM, 0)
    diag, off = hamiltonian_diagonals(grid, np.zeros(len(x_nm)))
    sigma = 1.0 / A0_NM
    centre = 10.0 / A0_NM
    p0 = np.sqrt(2.0 * 0.2 / EH_EV)
    phi = (2.0 * np.pi * sigma**2) ** (-0.25) * np.exp(-((grid.x_au - centre) ** 2) / (4.0 * sigma**2) + 1j * p0 * grid.x_au)
    phi /= np.sqrt(norm(phi, grid.dx_au))
    energies = np.linspace(-2.0, 5.0, 281)
    low_results = {}
    for eta_eV in [0.2, 0.1, 0.05]:
        spectrum = window_spectrum(diag, off, right_mask(grid) * phi, grid.dx_au, energies, eta_eV)
        low_results[str(eta_eV)] = integrate_above(energies, spectrum, 0.7170816000000001)
    tests["low_energy_kernel_tail"] = {"packet_mean_energy_eV": 0.2, "Y_H_by_eta_eV": low_results,
                                       "note": "Finite-eta kernel tail is an error-floor diagnostic, not a physical high-energy channel."}

    output["wall_seconds"] = time.perf_counter() - start
    output["status"] = "PASS" if all(value.get("pass", True) for value in tests.values()) else "FAIL"  # type: ignore[union-attr]
    path = BASE / "results/bias_v1/spectrum_method_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(path, output)
    print(json.dumps(output, indent=2))
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
