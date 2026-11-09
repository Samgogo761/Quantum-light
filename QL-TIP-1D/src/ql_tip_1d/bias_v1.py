"""Bias-protocol primitives for the solo-v1 QL-TIP-1D study.

The historical :mod:`ql_tip_1d.core` remains byte-frozen.  This module only
constructs potentials, masks and time envelopes; propagation reuses its CN
implementation.  Internal arrays use atomic units unless a name says otherwise.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import A0_NM, EH_EV, FIELD_V_M, TAU_FS, integrated_near_field, surface_potential, well_grid, Grid


def sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without changing it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    """Write UTF-8 JSON, refusing to replace an existing scientific record."""
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Not JSON serializable: {type(item)!r}")

    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False, default=convert)
        handle.write("\n")


def make_grid(config: dict[str, Any]) -> Grid:
    """Build the cell-centred grid with both well interfaces on cell faces."""
    numerics = config["numerics"]
    model = config["protocol"]["model"]
    return well_grid(
        well_cells=int(numerics["well_cells"]),
        xmin_nm=float(numerics["xmin_nm"]),
        xmax_nm=float(numerics["xmax_nm"]),
        width_nm=float(model["well_width_nm"]),
    )


def dc_field_au(x_au: np.ndarray, F_dc_V_per_nm: float, L_dc_nm: float = 20.0) -> np.ndarray:
    """Return the screened static electric field in atomic units.

    Positive ``F_dc_V_per_nm`` denotes extraction toward +x, hence E_x<0.
    """
    x_nm = np.asarray(x_au) * A0_NM
    field_si = -float(F_dc_V_per_nm) * 1e9 / (1.0 + np.maximum(x_nm, 0.0) / L_dc_nm) ** 2
    return np.where(x_nm >= 0.0, field_si / FIELD_V_M, 0.0)


def dc_potential_au(x_au: np.ndarray, F_dc_V_per_nm: float, L_dc_nm: float = 20.0) -> np.ndarray:
    """Return the electron static potential energy, continuous at the surface."""
    x_bohr = np.maximum(np.asarray(x_au), 0.0)
    length_bohr = L_dc_nm / A0_NM
    field_au = float(F_dc_V_per_nm) * 1e9 / FIELD_V_M
    value = -field_au * length_bohr * x_bohr / (x_bohr + length_bohr)
    return np.where(np.asarray(x_au) >= 0.0, value, 0.0)


def static_potential_au(grid: Grid, config: dict[str, Any]) -> np.ndarray:
    """Combine the finite well and DC potential without CAP or optical field."""
    model = config["protocol"]["model"]
    return surface_potential(grid.x_au, model["well_width_nm"], model["well_depth_eV"]) + dc_potential_au(
        grid.x_au,
        float(config["F_dc_V_per_nm"]),
        float(model["L_dc_nm"]),
    )


def cap_arrays(grid: Grid, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Construct separate cubic left/right absorbing potentials in atomic units."""
    x_nm = grid.x_nm
    numerics = config["numerics"]
    left0, left1 = map(float, numerics["left_cap_interval_nm"])
    right0, right1 = map(float, numerics["right_cap_interval_nm"])
    strength = float(numerics["cap_strength_au"])
    power = float(numerics["cap_power"])
    left = strength * (np.clip((left1 - x_nm) / (left1 - left0), 0.0, 1.0)) ** power
    right = strength * (np.clip((x_nm - right0) / (right1 - right0), 0.0, 1.0)) ** power
    return left, right


def smootherstep(time_fs: np.ndarray | float, start_fs: float, end_fs: float) -> np.ndarray:
    """Evaluate 10s^3-15s^4+6s^5 with constant endpoint extensions."""
    t = np.asarray(time_fs, dtype=float)
    s = np.clip((t - start_fs) / (end_fs - start_fs), 0.0, 1.0)
    return s**3 * (10.0 + s * (-15.0 + 6.0 * s))


def right_mask(grid: Grid, start_nm: float = 2.0, end_nm: float = 4.0) -> np.ndarray:
    """Return the unnormalised right-wavepacket mask used by the spectrum."""
    x_nm = grid.x_nm
    phase = np.clip((x_nm - start_nm) / (end_nm - start_nm), 0.0, 1.0)
    return np.sin(0.5 * np.pi * phase) ** 2


def energy_grid_ev(config: dict[str, Any], intensity_over_mu: float, step_eV: float | None = None) -> np.ndarray:
    """Construct the protocol's initial total-static-energy grid."""
    protocol = config["protocol"]
    model = protocol["model"]
    optics = protocol["optics"]
    spectrum = protocol["spectrum"]
    F_dc = float(config["F_dc_V_per_nm"])
    u_inf_eV = -F_dc * float(model["L_dc_nm"])
    mu = float(optics["mu_I_apex_W_per_cm2"])
    wavelength_um = float(optics["wavelength_nm"]) / 1000.0
    up_eV = 9.337e-14 * mu * intensity_over_mu * wavelength_um**2
    lower = u_inf_eV - 2.0
    upper = max(30.0, 20.0 * up_eV + abs(u_inf_eV) + 10.0)
    step = float(spectrum["energy_step_base_eV"] if step_eV is None else step_eV)
    count = int(np.ceil((upper - lower) / step))
    return lower + step * np.arange(count + 1)


def optical_potential_au(grid: Grid, config: dict[str, Any], time_fs: float,
                         spatial: np.ndarray | None = None) -> float | np.ndarray:
    """Return the midpoint optical interaction multiplied by integrated G(x)."""
    from scipy.constants import c, epsilon_0, electron_volt, h
    from .core import FIELD_V_M

    optics = config["protocol"]["optics"]
    intensity = float(config["I_apex_W_per_cm2"]) / float(config["protocol"]["model"]["intensity_enhancement_apex_over_incident"])
    e0 = np.sqrt(2.0 * intensity * 1e4 / (c * epsilon_0)) / FIELD_V_M
    tau = float(optics["intensity_envelope_fwhm_fs"])
    photon_eV = h * c / (float(optics["wavelength_nm"]) * 1e-9) / electron_volt
    omega_au = photon_eV / EH_EV
    time_au = time_fs / TAU_FS
    field = e0 * np.exp(-2.0 * np.log(2.0) * (time_fs / tau) ** 2) * np.cos(
        omega_au * time_au + float(config["CEP_rad"])
    )
    if spatial is None:
        spatial = integrated_near_field(grid.x_au, ell_nm=float(config["protocol"]["model"]["ell_nm"]))
    return field * spatial


def up_eV(I_apex_W_per_cm2: float, wavelength_nm: float = 1600.0) -> float:
    """Calculate ponderomotive energy using the frozen protocol convention."""
    return 9.337e-14 * I_apex_W_per_cm2 * (wavelength_nm / 1000.0) ** 2
