#!/usr/bin/env python3
"""Run local k-grid scans for the hBN two-band model benchmark.

This script is intended for the Windows/MSYS local workstation used in this
project. It creates run folders under examples/hBN_model_tb/runs and executes
the repository-local hhg_sbe.exe with a Yue/Gaarde-style hBN input.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
SOLVER = ROOT / "hhg_sbe.exe"
BASH = Path(r"C:\msys64\usr\bin\bash.exe")


def to_msys(path: Path) -> str:
    p = path.resolve()
    drive = p.drive.rstrip(":").lower()
    rest = str(p)[3:].replace("\\", "/")
    return f"/{drive}/{rest}"


def input_text(nk: int, t2_fs: float, ncyc: float) -> str:
    return f"""&crystal
  a1_ang = 1.2500000000, -2.1650635095, 0.0000000000
  a2_ang = 1.2500000000,  2.1650635095, 0.0000000000
  a3_ang = 0.0000000000,  0.0000000000, 20.0000000000
  E_fermi_eV = 0.0
  SOC = 0
  wannier_tb_file = "../../hBN_model_tb.dat"
  wannier_hr_file = ""
  wannier_r_file  = ""
/

&kgrid
  nkx = {nk}
  nky = {nk}
/

&bands
  nv_orig = 1
  nb_start = 1
  nb_end = 2
/

&laser
  wvl_nm = 1600.0
  intensity_Wcm2 = 3.5e12
  theta_deg = 0.0
  phi_cep_deg = 90.0
  ncyc = {ncyc:.8f}
  env_type = 2
  ellipticity = 0.0
  delta_phase_deg = 90.0
/

&laser2
  wvl_nm_2 = 0.0
  intensity_Wcm2_2 = 0.0
/

&external_field
  use_external_A = .false.
  external_A_file = ""
/

&timestep
  dt = 0.50
  n_dt_deph = 5
/

&dephasing
  T2_cycles = -1.0
  T2_fs = {t2_fs:.8e}
/

&bsv
  bsv_enabled = .false.
  bsv_n_samples = 1
  bsv_mean_intensity = 0.0
  bsv_seed = 42
/

&method
  gauge_method = 'lg_cov'
/

&diagnostics
  run_pcenter_check = .false.
  stop_after_diagnostics = .false.
/

&output
  save_geometry = .true.
  save_occupation = .false.
  occ_stride = 0
  occ_band_resolved = .false.
  save_coherence = .false.
/

&spin
  spin_current = .false.
  spin_sz_file = ""
  spin_order = 'interleaved'
/
"""


def run_case(nk: int, t2_fs: float, ncyc: float, prepare_only: bool, tag: str) -> Path:
    label = f"lgcov_k{nk}_T2_{str(t2_fs).replace('.', 'p')}fs"
    if tag:
        label += "_" + tag
    run_dir = EXAMPLE_DIR / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input.nml").write_text(input_text(nk, t2_fs, ncyc), encoding="ascii")
    if prepare_only:
        return run_dir

    if not SOLVER.exists():
        raise FileNotFoundError(f"Missing solver executable: {SOLVER}")
    if not BASH.exists():
        raise FileNotFoundError(f"Missing MSYS bash: {BASH}")

    cmd = (
        "export HOME=/tmp; "
        "export PATH=/mingw64/bin:/usr/bin:$PATH; "
        f"cd '{to_msys(run_dir)}' && '{to_msys(SOLVER)}' input.nml > run.log 2>&1"
    )
    env = os.environ.copy()
    env["HOME"] = r"C:\tmp"
    print(f"Running nk={nk}, T2={t2_fs} fs in {run_dir}")
    subprocess.run([str(BASH), "-lc", cmd], check=True, env=env)
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", nargs="+", type=int, default=[30, 60, 90])
    parser.add_argument("--t2-fs", type=float, default=5.0)
    parser.add_argument("--ncyc", type=float, default=11.0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    for nk in args.k:
        run_case(nk, args.t2_fs, args.ncyc, args.prepare_only, args.tag)


if __name__ == "__main__":
    main()
