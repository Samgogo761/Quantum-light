"""Independent SI finite-well bisection and checks of saved Stage 1 artifacts.

Does not import ql_tip_1d, its transcendental solver, or its eigensolver.
Does not perform driven propagation. Output is exclusive-create JSON.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import csv
import hashlib
import json
import math
import sys
import numpy as np
import scipy
from scipy.constants import hbar, electron_mass, electron_volt


def finite_well_si():
    half_width_m = 0.135e-9 / 2
    depth_joule = 10.8 * electron_volt
    z0 = half_width_m * math.sqrt(2 * electron_mass * depth_joule) / hbar
    lo, hi = 0.0, z0
    for _ in range(100):
        z = (lo + hi) / 2
        residual = z * math.tan(z) - math.sqrt(z0 * z0 - z * z)
        if residual > 0:
            hi = z
        else:
            lo = z
    k = z / half_width_m
    kappa = math.sqrt(z0 * z0 - z * z) / half_width_m
    energy_ev = -hbar * hbar * kappa * kappa / (2 * electron_mass * electron_volt)
    norm_integral_m = (half_width_m + math.sin(2 * k * half_width_m) / (2 * k)
                       + math.cos(k * half_width_m) ** 2 / kappa)
    amplitude_squared_per_m = 1 / norm_integral_m
    right_tail = amplitude_squared_per_m * math.cos(k * half_width_m) ** 2 / (2 * kappa)
    return {
        'width_nm': 0.135, 'depth_eV': 10.8,
        'constants_SI': {'hbar_J_s': hbar, 'electron_mass_kg': electron_mass,
                         'electron_volt_J': electron_volt},
        'method': '100 bisection iterations in SI, independent of production code',
        'z0': z0, 'z': z, 'dimensionless_equation_residual': residual,
        'energy_eV': energy_ev,
        'bound_state_count': 1 if 0 < z0 < math.pi / 2 else None,
        'analytic_vacuum_probability': right_tail,
        'analytic_total_outside_probability': 2 * right_tail,
        'normalization_constant_per_sqrt_m': math.sqrt(amplitude_squared_per_m),
    }


def check_saved_stage1(project, run_id):
    raw = project / 'data/raw' / run_id
    results = project / 'results' / run_id
    meta = json.loads((results / 'metadata.json').read_text(encoding='utf-8'))
    summary = json.loads((results / 'summary.json').read_text(encoding='utf-8'))
    # Only unit conversion constants are taken from recorded metadata.
    a0_nm = meta['constants']['bohr_nm']
    time_au_fs = meta['constants']['time_au_fs']
    saved = np.load(raw / 'bound_cells192.npz')
    x, p = saved['x_au'], saved['psi0']
    dx = float(x[1] - x[0])
    energy, potential = saved['energies_au'][0], saved['potential_au']
    padded = np.pad(p, (1, 1))
    hp = -(padded[2:] - 2 * padded[1:-1] + padded[:-2]) / (2 * dx * dx) + potential * p
    checks = {
        'run_id': run_id,
        'finest_eigen_residual_L2_au': float(np.sqrt(np.sum(abs(hp - energy * p) ** 2) * dx)),
        'finest_norm': float(np.sum(abs(p) ** 2) * dx),
        'finest_initial_right_half_probability': float(np.sum(abs(p[x > 0]) ** 2) * dx),
    }
    saved = np.load(raw / 'nofield_dt0.05.npz')
    dx = float(saved['x_au'][1] - saved['x_au'][0])
    p0, pf = saved['psi0'], saved['psi_final']
    energy, dt, steps = saved['energy_au'], saved['actual_dt_au'], saved['steps']
    cn_phase = complex((1 - .5j * dt * energy) / (1 + .5j * dt * energy)) ** int(steps)
    checks['saved_nofield_L2_error_to_cn_formula'] = float(np.sqrt(np.sum(abs(pf - cn_phase * p0) ** 2) * dx))
    checks['saved_nofield_L2_error_to_exact_exp'] = float(np.sqrt(np.sum(abs(pf - np.exp(-1j * energy * dt * steps) * p0) ** 2) * dx))
    saved = np.load(raw / 'free_gaussian_cn.npz')
    x = saved['x_au']
    dx = float(x[1] - x[0])
    density = abs(saved['psi_final']) ** 2
    probability = float(sum(density) * dx)
    mean = float(sum(x * density) * dx / probability)
    variance = float(sum((x - mean) ** 2 * density) * dx / probability)
    packet = meta['config']['free_packet']
    time_au = packet['duration_fs'] / time_au_fs
    sigma_au = packet['density_sigma_nm'] / a0_nm
    target_mean = packet['x0_nm'] / a0_nm + packet['momentum_au'] * time_au
    target_variance = sigma_au ** 2 + time_au ** 2 / (4 * sigma_au ** 2)
    checks['saved_free_cn_centroid_error_nm'] = (mean - target_mean) * a0_nm
    checks['saved_free_cn_sigma_error_nm'] = (math.sqrt(variance) - math.sqrt(target_variance)) * a0_nm
    with (results / 'manifest.csv').open(encoding='utf-8', newline='') as stream:
        manifest = list(csv.DictReader(stream))
    checks['saved_manifest_file_count'] = len(manifest)
    checks['saved_manifest_hashes_ok'] = all(
        hashlib.sha256((project / row['path']).read_bytes()).hexdigest() == row['sha256']
        for row in manifest)
    checks['independent_SI_vs_production_analytic_difference_eV'] = (
        finite_well_si()['energy_eV'] - summary['analytic']['energy_eV'])
    return checks


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1-run-id', default='stage1_20261103_r2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {
        'checked_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'finite_well_SI': finite_well_si(),
        'saved_artifacts': check_saved_stage1(project, args.stage1_run_id),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(serialized)
    print(serialized)


if __name__ == '__main__':
    main()
