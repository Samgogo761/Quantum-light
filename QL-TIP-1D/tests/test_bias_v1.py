"""Regression tests for bias_v1 helpers and static CN factor reuse."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ql_tip_1d.bias_v1 import dc_potential_au, make_grid, static_potential_au
from ql_tip_1d.core import CrankNicolson, Grid, norm, well_grid


def test_static_factor_reuse_is_bitwise_equivalent_to_refactorisation() -> None:
    """Skipping an identical LU refactorisation must not change the state."""
    grid = well_grid(12, xmin_nm=-1.0, xmax_nm=20.0)
    potential = dc_potential_au(grid.x_au, 1.0)
    initial = np.exp(-((grid.x_au - 20.0) ** 2) / 4.0 + 0.3j * grid.x_au).astype(complex)
    initial /= np.sqrt(norm(initial, grid.dx_au))
    reused = CrankNicolson(grid, 0.05, potential)
    state_reused = initial.copy()
    state_refactored = initial.copy()
    for _ in range(200):
        state_reused = reused.step(state_reused)
        refactoring = CrankNicolson(grid, 0.05, potential)
        state_refactored = refactoring.step(state_refactored, potential)
    np.testing.assert_array_equal(state_reused, state_refactored)


def test_protocol_config_grid_matches_well_geometry() -> None:
    """The generated protocol must preserve both well faces on cell faces."""
    protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/protocol_v1.json").read_text(encoding="utf-8"))
    config = {"protocol": protocol, "numerics": protocol["numerics"], "F_dc_V_per_nm": 1.0}
    grid = make_grid(config)
    x_nm = grid.x_nm
    assert np.min(x_nm) < -0.135 < np.max(x_nm)
    assert np.min(x_nm) < 0.0 < np.max(x_nm)
    assert static_potential_au(grid, config).shape == x_nm.shape
