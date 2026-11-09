"""Shared, side-effect-free plan helpers for the continuation-v2 execution."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT.parent
MANIFEST_PATH = PROJECT / "plans/20261105_continuation_v2/TASK_MANIFEST.json"
V2_RESULTS = PROJECT / "results/bias_v2"
V2_RAW = PROJECT / "data/raw/bias_v2"
V2_CONFIGS = PROJECT / "configs/continuation_v2"
THRESHOLDS_EV = (0.5170816, 0.7170816, 0.9170816)
MU_APEX = 1.5e12
READOUT_ETA = 0.05
READOUT_STEP = 0.0125
CONTROL_ETA = 0.025
CONTROL_STEP = 0.00625
GRID_STEP_ETA005 = 0.00625
GRID_STEP_ETA0025 = 0.003125


def sha256(path: str | Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest() -> dict[str, Any]:
    """Load the frozen continuation plan manifest."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def tag(value: float) -> str:
    """Make a compact, unique numeric tag without locale-dependent formatting."""
    return f"{value:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def readout_id(spec: dict[str, Any]) -> str:
    """Build a deterministic readout identifier from all physical choices."""
    base = f"ro2_{spec['run_id']}_t{tag(float(spec['expected_readout_fs']))}"
    base += f"_eta{tag(float(spec['eta_eV']))}_dE{tag(float(spec['step_eV']))}"
    base += f"_m{tag(float(spec['mask_start_nm']))}-{tag(float(spec['mask_end_nm']))}"
    if spec.get("energy_max_mode") == "extended":
        base += "_emaxplus"
    if spec.get("energy_min_eV") is not None and spec.get("energy_max_eV") is not None:
        base += f"_er{tag(float(spec['energy_min_eV']))}-{tag(float(spec['energy_max_eV']))}"
    return base


def base_energy_range(config: dict[str, Any], intensity_over_mu: float | None = None) -> tuple[float, float]:
    """Return the frozen starting total-energy range in eV."""
    model = config["protocol"]["model"]
    optics = config["protocol"]["optics"]
    F = float(config["F_dc_V_per_nm"])
    u_inf = -F * float(model["L_dc_nm"])
    intensity = float(config.get("I_over_mu", 0.0) if intensity_over_mu is None else intensity_over_mu)
    up = 9.337e-14 * MU_APEX * intensity * (float(optics["wavelength_nm"]) / 1000.0) ** 2
    lower = u_inf - 2.0
    upper = max(30.0, 20.0 * up + abs(u_inf) + 10.0)
    return lower, upper


def readout_plan(manifest: dict[str, Any] | None = None, *, include_new: bool = False) -> list[dict[str, Any]]:
    """Construct the complete R1 readout table without running any science."""
    m = load_manifest() if manifest is None else manifest
    specs: list[dict[str, Any]] = []

    def add(item: dict[str, Any], kind: str, eta: float, step: float,
            mask: tuple[float, float] = (2.0, 4.0), readout_fs: float | None = None,
            energy_max_mode: str = "base", note: str = "") -> None:
        specs.append({
            "kind": kind,
            "run_id": item["run_id"],
            "config_path": item["config"],
            "state_path": item["state"],
            "F_dc_V_per_nm": float(item["F_dc_V_per_nm"]),
            "I_over_mu": float(item["I_over_mu"]),
            "CEP_rad": float(item["CEP_rad"]),
            "expected_readout_fs": float(item.get("expected_readout_fs", 100.0)
                                         if readout_fs is None else readout_fs),
            "eta_eV": eta, "step_eV": step,
            "mask_start_nm": mask[0], "mask_end_nm": mask[1],
            "energy_max_mode": energy_max_mode,
            "thresholds_eV": list(THRESHOLDS_EV),
            "note": note,
        })

    for item in m["existing_primary_states"]:
        add(item, "primary", READOUT_ETA, READOUT_STEP)
        add(item, "primary_eta_control", CONTROL_ETA, CONTROL_STEP)
        if abs(float(item["CEP_rad"])) < 1e-14:
            add(item, "grid_step_eta005", READOUT_ETA, GRID_STEP_ETA005)
            add(item, "grid_step_eta0025", CONTROL_ETA, GRID_STEP_ETA0025)
    for item in m["existing_half_phase_states"]:
        add(item, "half_phase", READOUT_ETA, READOUT_STEP)
    for item in m["existing_numerical_control_states"]:
        add(item, "numerical_control", READOUT_ETA, READOUT_STEP)
    for item in m["existing_dc_states"]:
        add(item, "dc_only", READOUT_ETA, READOUT_STEP)
    for item in m["existing_late_states"]:
        add(item, "late_time", READOUT_ETA, READOUT_STEP)
    primary_by_id = {item["run_id"]: item for item in m["existing_primary_states"]}
    for run_id in ["bias_p1_F0.000_I1_phi000_r02", "bias_p1_F1.000_I1_phi000_r02"]:
        add(primary_by_id[run_id], "mask_control", READOUT_ETA, READOUT_STEP, mask=(3.0, 5.0))
    for run_id in ["bias_p2_F0.000_I4_phi000_r02", "bias_p2_F1.000_I4_phi000_r02"]:
        add(primary_by_id[run_id], "energy_range_control", READOUT_ETA, READOUT_STEP,
            energy_max_mode="extended")
    # The four new R1 states become readable only after repair-tdse.  Their
    # configurations live under bias_v2, never in the frozen bias_v1 tree.
    if include_new:
        for task in m["planned_repair_tdse"]:
            config_rel = f"results/bias_v2/runs/{task['run_id']}/config.json"
            state_rel = f"data/raw/bias_v2/{task['run_id']}/state_100fs.npz"
            if not project_path(config_rel).is_file() or not project_path(state_rel).is_file():
                continue
            parent_config = json.loads(project_path(task["parent_config"]).read_text(encoding="utf-8"))
            item = {"run_id": task["run_id"], "config": config_rel, "state": state_rel,
                    "F_dc_V_per_nm": task["overrides_json_pointer"].get(
                        "/F_dc_V_per_nm", parent_config["F_dc_V_per_nm"]),
                    "I_over_mu": task["overrides_json_pointer"]["/I_over_mu"],
                    "CEP_rad": task["overrides_json_pointer"]["/CEP_rad"],
                    "expected_readout_fs": 100.0}
            add(item, "new_repair_control", READOUT_ETA, READOUT_STEP)
            add(item, "new_repair_control_eta", CONTROL_ETA, CONTROL_STEP)
    for spec in specs:
        spec["readout_id"] = readout_id(spec)
    if len({spec["readout_id"] for spec in specs}) != len(specs):
        raise ValueError("Duplicate continuation-v2 readout ID")
    return specs


def _derive_observation(parent_rel: str, run_id: str, intensity: float, cep: float,
                        phase: str, baseline: str = "") -> dict[str, Any]:
    """Derive a new laser task from a frozen parent config without editing it."""
    parent_path = project_path(parent_rel)
    config = json.loads(parent_path.read_text(encoding="utf-8"))
    config["I_over_mu"] = float(intensity)
    config["I_apex_W_per_cm2"] = float(intensity) * MU_APEX
    config["CEP_rad"] = float(cep)
    config["late_readout"] = False
    config["full_spectrum_controls"] = False
    config["parent_config_path"] = parent_rel
    config["parent_config_sha256"] = sha256(parent_path)
    config["protocol_sha256"] = sha256(PROJECT / "configs/protocol_v1.json")
    config["continuation"] = {"schema": "continuation-v2", "task_id": run_id,
                              "phase": phase, "baseline_run_id": baseline}
    config["skip_legacy_spectrum"] = True
    return {"run_id": run_id, "config": config, "parent_config": parent_rel,
            "parent_config_sha256": config["parent_config_sha256"], "baseline_run_id": baseline}


def high_intensity_tasks() -> list[dict[str, Any]]:
    """Return four 8/32mu probes and six highest-intensity controls."""
    tasks = []
    for intensity, phase in [(8.0, "R2_probe"), (32.0, "R2_probe")]:
        for F in [0.0, 1.0]:
            parent = f"results/bias_v1/bias_p1_F{F:.3f}_I1_phi000_r02/config.json"
            tasks.append(_derive_observation(parent, f"bias_v2_probe_F{F:g}_I{intensity:g}_phi000_r01",
                                             intensity, 0.0, phase))
    for F in [0.0, 1.0]:
        for label, key in [("dx048", "well_cells"), ("dt005", "dt_au")]:
            parent = f"results/bias_v1/bias_p1_F{F:.3f}_I1_phi000_{label}_r02/config.json"
            tasks.append(_derive_observation(parent, f"bias_v2_probe_F{F:g}_I32_phi000_{label}_r01",
                                             32.0, 0.0, "R2_control"))
    for label in ["box900", "caphalf"]:
        parent = f"results/bias_v1/bias_p1_F1.000_I1_phi000_{label}_r02/config.json"
        tasks.append(_derive_observation(parent, f"bias_v2_probe_F1_I32_phi000_{label}_r01",
                                         32.0, 0.0, "R2_control"))
    parent = "results/bias_v1/bias_p1_F0.000_I1_phi000_box900_r02/config.json"
    tasks.append(_derive_observation(parent, "bias_v2_probe_F0_I32_phi000_box900_r01",
                                     32.0, 0.0, "R2_control"))
    parent = "results/bias_v1/bias_p1_F0.000_I1_phi000_caphalf_r02/config.json"
    tasks.append(_derive_observation(parent, "bias_v2_probe_F0_I32_phi000_caphalf_r01",
                                     32.0, 0.0, "R2_control"))
    return tasks


def library_tasks() -> list[dict[str, Any]]:
    """Return every missing production phase for the two-bias response library."""
    nodes = [8.0, 32.0, 12.0, 20.0, 0.25, 0.5, 2.0, 1.0, 4.0]
    phases = [(0, 0.0), (90, math.pi / 2.0), (180, math.pi), (270, 3.0 * math.pi / 2.0)]
    tasks = []
    for F in [0.0, 1.0]:
        for intensity in nodes:
            for degrees, cep in phases:
                if intensity == 1.0:
                    existing = (f"bias_p1_F{F:.3f}_I1_phi000_r02" if degrees == 0
                                else f"bias_p2_F{F:.3f}_I1_phi{degrees:03d}_r02")
                elif intensity == 4.0:
                    existing = (f"bias_p2_F{F:.3f}_I4_phi000_r02" if degrees == 0
                                else f"bias_p2_F{F:.3f}_I4_phi{degrees:03d}_r02")
                elif intensity in {8.0, 32.0} and degrees == 0:
                    existing = f"bias_v2_probe_F{F:g}_I{intensity:g}_phi000_r01"
                    tasks.append({"run_id": existing, "existing": True, "F_dc_V_per_nm": F,
                                  "I_over_mu": intensity, "CEP_rad": cep})
                    continue
                else:
                    existing = ""
                if existing:
                    tasks.append({"run_id": existing, "existing": True, "F_dc_V_per_nm": F,
                                  "I_over_mu": intensity, "CEP_rad": cep})
                    continue
                parent = f"results/bias_v1/bias_p1_F{F:.3f}_I1_phi000_r02/config.json"
                run_id = f"bias_v2_lib_F{F:g}_I{intensity:g}_phi{degrees:03d}_r01"
                task = _derive_observation(parent, run_id, intensity, cep, "R3_library")
                tasks.append({**task, "existing": False, "F_dc_V_per_nm": F,
                              "I_over_mu": intensity, "CEP_rad": cep})
    return tasks


def v2_readout_specs() -> list[dict[str, Any]]:
    """Add readouts for currently available v2 probe/library states."""
    specs = []
    for directory in sorted((PROJECT / "results/bias_v2/runs").glob("*")):
        config_path = directory / "config.json"
        state_path = PROJECT / "data/raw/bias_v2" / directory.name / "state_100fs.npz"
        if not config_path.is_file() or not state_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        base = {"run_id": directory.name,
                "config_path": str(config_path.relative_to(PROJECT)).replace("\\", "/"),
                "state_path": str(state_path.relative_to(PROJECT)).replace("\\", "/"),
                "F_dc_V_per_nm": float(config["F_dc_V_per_nm"]),
                "I_over_mu": float(config["I_over_mu"]), "CEP_rad": float(config["CEP_rad"]),
                "expected_readout_fs": 100.0, "thresholds_eV": list(THRESHOLDS_EV)}
        for eta, step, kind in [(READOUT_ETA, READOUT_STEP, "v2_probe")]:
            spec = {**base, "kind": kind, "eta_eV": eta, "step_eV": step,
                    "mask_start_nm": 2.0, "mask_end_nm": 4.0, "energy_max_mode": "base"}
            spec["readout_id"] = readout_id(spec)
            specs.append(spec)
        if (config["I_over_mu"] == 32.0 and abs(config["CEP_rad"]) < 1e-14
                and not any(label in directory.name for label in ["dx048", "dt005", "box900", "caphalf"])):
            for eta, step, kind in [(CONTROL_ETA, CONTROL_STEP, "v2_probe_eta")]:
                spec = {**base, "kind": kind, "eta_eV": eta, "step_eV": step,
                        "mask_start_nm": 2.0, "mask_end_nm": 4.0, "energy_max_mode": "base"}
                spec["readout_id"] = readout_id(spec)
                specs.append(spec)
            spec = {**base, "kind": "v2_probe_emax", "eta_eV": READOUT_ETA, "step_eV": READOUT_STEP,
                    "mask_start_nm": 2.0, "mask_end_nm": 4.0, "energy_max_mode": "extended"}
            spec["readout_id"] = readout_id(spec)
            specs.append(spec)
            spec = {**base, "kind": "v2_probe_grid", "eta_eV": READOUT_ETA, "step_eV": GRID_STEP_ETA005,
                    "mask_start_nm": 2.0, "mask_end_nm": 4.0, "energy_max_mode": "base"}
            spec["readout_id"] = readout_id(spec)
            specs.append(spec)
    return specs


def midpoint_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Build four-phase TDSE tasks at the two planned arithmetic midpoints."""
    tasks = []
    for item in plan.get("intervals", []):
        intensity = float(item["midpoint_I_over_mu"])
        for F in [0.0, 1.0]:
            parent = f"results/bias_v1/bias_p1_F{F:.3f}_I1_phi000_r02/config.json"
            for degrees, cep in [(0, 0.0), (90, math.pi / 2.0), (180, math.pi), (270, 3.0 * math.pi / 2.0)]:
                run_id = f"bias_v2_mid_F{F:g}_I{intensity:g}_phi{degrees:03d}_r01"
                task = _derive_observation(parent, run_id, intensity, cep, "R3_midpoint")
                task.update({"F_dc_V_per_nm": F, "I_over_mu": intensity, "CEP_rad": cep})
                tasks.append(task)
    return tasks


def s4_control_tasks(intensity: float) -> list[dict[str, Any]]:
    """Four matched dx/dt controls at the S4 representative intensity, CEP0."""
    tasks = []
    for F in [0.0, 1.0]:
        for label in ["dx048", "dt005"]:
            parent = f"results/bias_v1/bias_p1_F{F:.3f}_I1_phi000_{label}_r02/config.json"
            run_id = f"bias_v2_s4_F{F:g}_I{intensity:g}_phi000_{label}_r01"
            task = _derive_observation(parent, run_id, intensity, 0.0, "R3_S4_control")
            task.update({"F_dc_V_per_nm": F, "I_over_mu": intensity, "CEP_rad": 0.0})
            tasks.append(task)
    return tasks


def phase8_tasks(conditions: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """Four extra CEP values for each 4->8 phase-validation condition."""
    tasks = []
    for F, intensity in conditions:
        parent = f"results/bias_v1/bias_p1_F{F:.3f}_I1_phi000_r02/config.json"
        for degrees in [45, 135, 225, 315]:
            cep = math.radians(degrees)
            run_id = f"bias_v2_ph8_F{F:g}_I{intensity:g}_phi{degrees:03d}_r01"
            task = _derive_observation(parent, run_id, intensity, cep, "R3_phase8")
            task.update({"F_dc_V_per_nm": F, "I_over_mu": intensity, "CEP_rad": cep})
            tasks.append(task)
    return tasks


def project_path(relative: str | Path) -> Path:
    """Resolve a path recorded relative to the QL-TIP-1D project root."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT / path


def repair_tasks(manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return only the four mandatory R1 laser propagations."""
    m = load_manifest() if manifest is None else manifest
    tasks = []
    for item in m["planned_repair_tdse"]:
        base = item["parent_config"]
        parent = json.loads(project_path(base).read_text(encoding="utf-8"))
        prep_cfg_path = item["preparation"]["config"]
        prep_summary_path = item["preparation"]["summary"]
        prep_state_path = item["preparation"]["state"]
        prep_config = json.loads(project_path(prep_cfg_path).read_text(encoding="utf-8"))
        prep_summary = json.loads(project_path(prep_summary_path).read_text(encoding="utf-8"))
        config = json.loads(json.dumps(parent))
        for key, value in item["overrides_json_pointer"].items():
            config[key.lstrip("/")] = value
        config["I_over_mu"] = float(config["I_apex_W_per_cm2"]) / MU_APEX
        config["prep_config_path"] = prep_cfg_path
        config["prep_summary_sha256"] = sha256(project_path(prep_summary_path))
        config["prep_config_sha256"] = sha256(project_path(prep_cfg_path))
        config["parent_config_path"] = base
        config["parent_config_sha256"] = sha256(project_path(base))
        config["continuation"] = {
            "schema": "continuation-v2",
            "task_id": item["run_id"],
            "baseline_run_id": item["baseline_run_id"],
            "actual_preparation_fs": item["preparation"]["actual_preparation_fs"],
            "prep_run_id": item["preparation"]["run_id"],
        }
        preparation = item["preparation"]["actual_preparation_fs"]
        config["preparation"]["ramp_start_fs"] = float(preparation["ramp_start_fs"])
        config["preparation"]["ramp_end_fs"] = float(preparation["ramp_end_fs"])
        config["preparation"]["wait_end_fs"] = float(preparation["prep_end_fs"])
        config["preparation"]["ramp_duration_fs"] = float(preparation["ramp_end_fs"] - preparation["ramp_start_fs"])
        config["preparation"]["wait_duration_fs"] = float(preparation["prep_end_fs"] - preparation["ramp_end_fs"])
        config["protocol_sha256"] = sha256(PROJECT / "configs/protocol_v1.json")
        tasks.append({
            "run_id": item["run_id"],
            "config": config,
            "parent_config": base,
            "parent_config_sha256": config["parent_config_sha256"],
            "baseline_run_id": item["baseline_run_id"],
            "prep_config_path": prep_cfg_path,
            "prep_config": prep_config,
            "prep_summary": prep_summary,
            "prep_state_path": prep_state_path,
            "prep_state_sha256": item["preparation"]["state_sha256"] if "state_sha256" in item["preparation"] else config["prep_state_sha256"],
        })
    return tasks
