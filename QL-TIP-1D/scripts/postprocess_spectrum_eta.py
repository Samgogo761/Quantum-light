"""Reuse a saved 100 fs state to test additional spectrum resolutions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from ql_tip_1d.bias_v1 import energy_grid_ev, make_grid, right_mask, sha256, write_json_new
from ql_tip_1d.core import hamiltonian_diagonals
from ql_tip_1d.spectrum_v1 import integrate_above, window_spectrum, write_spectrum_csv


def main() -> int:
    """Calculate eta/energy-step controls from a previously saved complex state."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    result_dir = BASE / "results/bias_v1" / args.run_id
    raw_dir = BASE / "data/raw/bias_v1" / args.run_id
    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    state_path = raw_dir / "state_100fs.npz"
    start = time.perf_counter()
    with np.load(state_path) as saved:
        psi = saved["psi"].astype(complex)
        potential = saved["static_H_potential_au"]
        dx_au = float(saved["dx_au"])
    grid = make_grid(config)
    diag, off = hamiltonian_diagonals(grid, potential)
    phi = right_mask(grid) * psi
    energies = energy_grid_ev(config, float(config["I_over_mu"]), args.step)
    spectrum = window_spectrum(diag, off, phi, dx_au, energies, args.eta)
    u_infinity = -float(config["F_dc_V_per_nm"]) * float(config["protocol"]["model"]["L_dc_nm"])
    csv_path = result_dir / f"spectrum_{args.label}.csv"
    write_spectrum_csv(csv_path, args.run_id, energies, spectrum, args.eta, u_infinity)
    e_star = float(config["protocol"]["spectrum"]["E_star_eV"])
    offset = max(0.1 * e_star, float(config["protocol"]["spectrum"]["eta_base_eV"]))
    record = {
        "run_id": args.run_id, "label": args.label, "eta_eV": args.eta, "energy_step_eV": args.step,
        "mask_bounds_nm": [2.0, 4.0], "energy_range_eV": [float(energies[0]), float(energies[-1])],
        "masked_norm": float(np.vdot(phi, phi).real * dx_au),
        "spectrum_integral": float(np.trapezoid(spectrum, energies)),
        "Y_H_main": integrate_above(energies, spectrum, e_star),
        "Y_H_lower": integrate_above(energies, spectrum, e_star - offset),
        "Y_H_upper": integrate_above(energies, spectrum, e_star + offset),
        "wall_seconds": time.perf_counter() - start,
        "state_sha256": sha256(state_path), "source_sha256": sha256(Path(__file__)),
        "spectrum_csv_sha256": sha256(csv_path),
    }
    output = result_dir / f"eta_control_{args.label}.json"
    write_json_new(output, record)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
