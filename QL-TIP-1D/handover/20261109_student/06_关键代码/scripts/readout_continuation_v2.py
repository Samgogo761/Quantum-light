"""Generic saved-state readout for continuation-v2; never evolves a wavefunction."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from ql_tip_1d.bias_v1 import make_grid, right_mask, sha256, write_json_new
from ql_tip_1d.continuation_v2 import (BASE, THRESHOLDS_EV, V2_RESULTS, base_energy_range,
                                       project_path, readout_id)
from ql_tip_1d.core import A0_NM, EH_EV, hamiltonian_diagonals, norm
from ql_tip_1d.spectrum_v1 import integrate_above, window_spectrum


def source_hashes() -> dict[str, str]:
    """Hash the active readout implementation and its numerical dependencies."""
    paths = [
        PROJECT / "scripts/readout_continuation_v2.py",
        PROJECT / "src/ql_tip_1d/continuation_v2.py",
        PROJECT / "src/ql_tip_1d/core.py",
        PROJECT / "src/ql_tip_1d/bias_v1.py",
        PROJECT / "src/ql_tip_1d/spectrum_v1.py",
    ]
    if (PROJECT / "configs/continuation_v2/protocol_v2.json").exists():
        paths.append(PROJECT / "configs/continuation_v2/protocol_v2.json")
    return {str(path.relative_to(PROJECT)).replace("\\", "/"): sha256(path) for path in paths}


def load_readout_spec(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Build a readout spec from an explicit saved run configuration."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = {
        "kind": "manual",
        "run_id": str(config_path.parent.name),
        "config_path": str(config_path.relative_to(PROJECT)).replace("\\", "/"),
        "state_path": args.state_path,
        "F_dc_V_per_nm": float(config["F_dc_V_per_nm"]),
        "I_over_mu": float(config["I_over_mu"]),
        "CEP_rad": float(config["CEP_rad"]),
        "expected_readout_fs": float(args.expected_readout_fs),
        "eta_eV": float(args.eta), "step_eV": float(args.step),
        "mask_start_nm": float(args.mask_start), "mask_end_nm": float(args.mask_end),
        "energy_min_eV": args.energy_min, "energy_max_eV": args.energy_max,
        "energy_max_mode": "explicit" if args.energy_max is not None else "base",
        "thresholds_eV": list(THRESHOLDS_EV),
    }
    spec["readout_id"] = readout_id(spec)
    return spec


def run_readout(spec: dict[str, Any], output_dir: Path | None = None) -> dict[str, Any]:
    """Evaluate one saved state at one fixed spectrum configuration."""
    start = time.perf_counter()
    config_path = project_path(spec["config_path"])
    state_path = project_path(spec["state_path"])
    resolved_output = (V2_RESULTS / "readouts" / spec["readout_id"]) if output_dir is None else output_dir
    cached = resolved_output / "summary.json"
    if cached.exists():
        previous = json.loads(cached.read_text(encoding="utf-8"))
        if (previous.get("state_sha256") == sha256(state_path)
                and previous.get("readout_id") == spec["readout_id"]):
            return {**previous, "reused": True}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(round(float(spec["F_dc_V_per_nm"]) * 1000)) != int(round(float(config["F_dc_V_per_nm"]) * 1000)):
        raise ValueError("Readout/spec F mismatch")
    if abs(float(spec["I_over_mu"]) - float(config["I_over_mu"])) > 1e-12:
        raise ValueError("Readout/spec intensity mismatch")
    if abs(float(spec["CEP_rad"]) - float(config["CEP_rad"])) > 1e-12:
        raise ValueError("Readout/spec CEP mismatch")
    with np.load(state_path) as state:
        x_au = state["x_au"]
        psi = state["psi"].astype(complex)
        potential = state["static_H_potential_au"]
        dx_au = float(state["dx_au"])
        readout_fs = float(state["readout_fs"])
    if abs(readout_fs - float(spec["expected_readout_fs"])) > 1e-9:
        raise ValueError(f"State readout time {readout_fs} != expected {spec['expected_readout_fs']}")
    grid = make_grid(config)
    if len(grid.x_au) != len(x_au) or not np.allclose(grid.x_au, x_au, rtol=0.0, atol=1e-13):
        raise ValueError("Saved state x grid does not match its run config")
    if abs(grid.dx_au - dx_au) > 1e-13:
        raise ValueError("Saved state dx does not match its run config")
    energies = None
    if spec.get("energy_min_eV") is not None and spec.get("energy_max_eV") is not None:
        lower, upper = float(spec["energy_min_eV"]), float(spec["energy_max_eV"])
    else:
        lower, upper = base_energy_range(config)
        if spec.get("energy_max_mode") == "extended":
            e_star = THRESHOLDS_EV[1]
            upper += max(10.0, 0.5 * (upper - e_star))
    step = float(spec["step_eV"])
    count = int(np.ceil((upper - lower) / step))
    energies = lower + step * np.arange(count + 1)
    mask = right_mask(grid, float(spec["mask_start_nm"]), float(spec["mask_end_nm"]))
    phi = mask * psi
    diag, off = hamiltonian_diagonals(grid, potential)
    spectrum = window_spectrum(diag, off, phi, dx_au, energies, float(spec["eta_eV"]))
    masked_norm = norm(phi, dx_au)
    integral = float(np.trapezoid(spectrum, energies))
    yields = {f"Y_at_E_{name}": integrate_above(energies, spectrum, threshold)
              for name, threshold in zip(["low", "main", "high"], THRESHOLDS_EV)}
    elapsed = time.perf_counter() - start
    output_dir = resolved_output
    if output_dir.exists():
        raise FileExistsError(f"Incomplete readout refuses overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    config_record = {
        "schema": "continuation-v2-readout",
        "readout_id": spec["readout_id"],
        **spec,
        "state_sha256": sha256(state_path),
        "run_config_sha256": sha256(config_path),
        "source_sha256": source_hashes(),
        "actual_readout_fs": readout_fs,
        "energy_range_eV": [float(energies[0]), float(energies[-1])],
        "energy_points": len(energies),
    }
    write_json_new(output_dir / "config.json", config_record)
    with (output_dir / "spectrum.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["E_corr_eV", "K_infinity_eV", "S_per_eV", "eta_eV", "run_id"])
        u_inf = -float(config["F_dc_V_per_nm"]) * float(config["protocol"]["model"]["L_dc_nm"])
        for energy, value in zip(energies, spectrum):
            writer.writerow([f"{energy:.17g}", f"{energy - u_inf:.17g}", f"{value:.17g}",
                             f"{spec['eta_eV']:.17g}", spec["run_id"]])
    summary = {
        "schema": "continuation-v2-readout-summary",
        "readout_id": spec["readout_id"], "kind": spec["kind"], "run_id": spec["run_id"],
        "F_dc_V_per_nm": float(spec["F_dc_V_per_nm"]), "I_over_mu": float(spec["I_over_mu"]),
        "CEP_rad": float(spec["CEP_rad"]), "actual_readout_fs": readout_fs,
        "eta_eV": float(spec["eta_eV"]), "step_eV": step,
        "mask_bounds_nm": [float(spec["mask_start_nm"]), float(spec["mask_end_nm"])],
        "energy_range_eV": [float(energies[0]), float(energies[-1])], "energy_points": len(energies),
        "thresholds_eV": list(THRESHOLDS_EV), **yields,
        "masked_norm": masked_norm, "spectrum_integral": integral,
        "integral_over_masked_norm": integral / masked_norm if masked_norm else None,
        "state_sha256": sha256(state_path), "run_config_sha256": sha256(config_path),
        "source_sha256": source_hashes(), "wall_seconds": elapsed,
        "execution_status": "PASS", "scientific_gate_status": "NOT_EVALUATED",
    }
    write_json_new(output_dir / "summary.json", summary)
    (output_dir / "run_log.txt").write_text(
        f"readout_id={spec['readout_id']}\nstate={state_path}\nwall_seconds={elapsed:.17g}\n"
        f"status=PASS\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    """CLI for a single explicit readout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--mask-start", type=float, default=2.0)
    parser.add_argument("--mask-end", type=float, default=4.0)
    parser.add_argument("--energy-min", type=float, default=None)
    parser.add_argument("--energy-max", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--expected-readout-fs", type=float, default=100.0)
    args = parser.parse_args()
    spec = load_readout_spec(args.config_path, args)
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        allowed = (V2_RESULTS / "readouts").resolve()
        if allowed not in output.parents:
            raise ValueError("Readout output must stay under results/bias_v2/readouts")
    else:
        output = None
    summary = run_readout(spec, output)
    return 0 if summary["execution_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
