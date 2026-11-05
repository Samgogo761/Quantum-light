"""Metadata only: inventory and hash checks; never imports scientific project code.

Run from any cwd. --build writes TASK_MANIFEST.json once; --inventory writes the
final documentation inventory once; --check verifies frozen inputs and outputs.
Existing files are never overwritten. Revision requires a new versioned folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPO = PROJECT.parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def build():
    old_list = PROJECT / 'results/bias_v1/local_artifact_hashes.csv'
    with old_list.open(encoding='utf-8-sig', newline='') as stream:
        expected = {r['path'].replace('\\', '/'): r for r in csv.DictReader(stream)}
    inputs = {}

    def file(rel):
        if rel not in inputs:
            path = PROJECT / rel
            actual = sha(path)
            if rel not in expected or actual != expected[rel]['sha256']:
                raise ValueError(f'Missing historical hash or changed input: {rel}')
            inputs[rel] = {'path': rel, 'bytes': path.stat().st_size,
                           'sha256': actual, 'historical_hash_match': True}
        return rel

    def run(rid, state='state_100fs.npz'):
        conf = file(f'results/bias_v1/{rid}/config.json')
        summ = file(f'results/bias_v1/{rid}/summary.json')
        provenance = file(f'results/bias_v1/{rid}/provenance.json')
        c = json.loads((PROJECT / conf).read_text(encoding='utf-8'))
        s = json.loads((PROJECT / summ).read_text(encoding='utf-8'))
        assert s['status'] == 'PASS', rid
        result = {'run_id': rid, 'config': conf, 'summary': summ, 'provenance': provenance,
                  'state': file(f'data/raw/bias_v1/{rid}/{state}'),
                  'F_dc_V_per_nm': c['F_dc_V_per_nm'],
                  'I_over_mu': c.get('I_over_mu'), 'CEP_rad': c['CEP_rad']}
        if state == 'state_prep.npz':
            result['actual_preparation_fs'] = {
                k: s[k] for k in ('ramp_start_fs', 'ramp_end_fs', 'prep_end_fs')}
        else:
            result['expected_readout_fs'] = 125.0 if '125' in state else 100.0
        return result

    def primary(f, intensity, angle):
        stage = 'p1' if intensity == 1 and angle == 0 else 'p2'
        return f'bias_{stage}_F{f:.3f}_I{intensity}_phi{angle:03d}_r02'

    main = [run(primary(f, i, a)) for f in (0, 1) for i in (1, 4)
            for a in (0, 90, 180, 270)]
    halves = [run(primary(1, 4, a)) for a in (45, 135, 225, 315)]
    controls = [run(f'bias_p1_F{f:.3f}_I1_phi000_{ctrl}_r02')
                for f in (0, 1) for ctrl in ('dx048', 'dt005', 'box900', 'caphalf')]
    controls += [run(f'bias_p2_F1.000_I4_phi000_{ctrl}_r02')
                 for ctrl in ('dx048', 'dt005', 'box900', 'caphalf')]
    dc = [run(f'bias_p1_dc_F{f:.3f}_r02') for f in (0, 1)]
    late = [run(primary(f, 1, 0), 'state_125fs.npz') for f in (0, 1)]
    repairs = []
    for label, f, intensity, ctrl in [
        ('F0_I4_dx048', 0, 4, 'dx048'), ('F0_I4_dt005', 0, 4, 'dt005'),
        ('F1_I1_ramp080', 1, 1, 'ramp080'), ('F1_I1_wait040', 1, 1, 'wait040')]:
        prep = run(f'bias_p1_prep_F{f:.3f}_{ctrl}_r02', 'state_prep.npz')
        parent = f'bias_p1_F0.000_I1_phi000_{ctrl}_r02' if f == 0 else primary(1, 1, 0)
        overrides = {'/I_over_mu': intensity,
                     '/I_apex_W_per_cm2': intensity * 1.5e12,
                     '/CEP_rad': 0.0, '/late_readout': False,
                     '/full_spectrum_controls': False,
                     '/prep_state_path': prep['state'],
                     '/prep_summary_path': prep['summary'],
                     '/prep_state_sha256': inputs[prep['state']]['sha256']}
        if f == 1:
            ramp, wait = (80.0, 20.0) if ctrl == 'ramp080' else (40.0, 40.0)
            overrides.update({'/preparation/ramp_duration_fs': ramp,
                              '/preparation/wait_duration_fs': wait,
                              '/preparation/ramp_start_fs': -100.0 - wait - ramp,
                              '/preparation/ramp_end_fs': -100.0 - wait,
                              '/preparation/wait_end_fs': -100.0})
        repairs.append({'run_id': f'bias_v2_fix_{label}_r01', 'status': 'planned_not_run',
                        'parent_config': file(f'results/bias_v1/{parent}/config.json'),
                        'preparation': prep,
                        'overrides_json_pointer': overrides,
                        'baseline_run_id': primary(f, intensity, 0),
                        'new_results_dir': f'results/bias_v2/runs/bias_v2_fix_{label}_r01',
                        'new_raw_dir': f'data/raw/bias_v2/bias_v2_fix_{label}_r01'})
    probes = [{'run_id': f'bias_v2_probe_F{f}_I{i}_phi000_r01',
               'status': 'planned_not_run', 'F_dc_V_per_nm': f, 'I_over_mu': i,
               'I_apex_W_per_cm2': i * 1.5e12, 'CEP_rad': 0.0,
               'requires': 'matching R1 PASS; 32mu also requires both 8mu probes usable'}
              for i in (8, 32) for f in (0, 1)]
    result = {
        'schema': 'ql-tip-1d-continuation-plan-v2', 'date': '2026-11-05',
        'status': 'planned_not_run', 'path_base': 'QL-TIP-1D project root',
        'scientific_computation_performed_by_this_builder': False,
        'inherited_result_commit': '41e26d1', 'inherited_review_commit': '641bdef',
        'source_hash_list': {'path': str(old_list.relative_to(PROJECT)).replace('\\', '/'),
                             'sha256': sha(old_list)},
        'scope': 'R0/R1 repair then gated R2/R3; two biases only; local default',
        'new_runner_exists': False,
        'manifest_is_not_an_executable_config': True,
        'config_derivation_rules': [
            'Copy the parent in memory; save only to a new v2 path, never edit the parent.',
            'Apply explicit JSON-pointer overrides, then bind a new full protocol and its SHA.',
            'Synchronize nested protocol preparation/numerics with actual top-level effective values.',
            'Retain old protocol SHA separately as parent identity; never pass it off as the new SHA.',
            'Resolve project-relative paths explicitly; verify prep state/config/summary hashes.',
            'Recompute I_inc=I_apex/9; do not apply field enhancement twice.',
            'Old automatically produced eta=.1 spectra are diagnostics; new main readout is separate.'
        ],
        'frozen_thresholds_eV': [0.5170816, 0.7170816, 0.9170816],
        'candidate_readout': {'eta_eV': 0.05, 'step_eV': 0.0125,
                             'control_eta_eV': 0.025, 'control_step_eV': 0.00625,
                             'requires_gate_before_production_freeze': True},
        'budget': {'global_tdse_attempts': 240, 'global_science_hours': 72,
                   'continuation_new_tdse_attempts': 112, 'continuation_science_hours': 40,
                   'repair_new_tdse_attempts': 12, 'repair_science_hours': 8,
                   'mandatory_repair_tdse': 4, 'maximum_batch_tdse': 8,
                   'single_event_minutes': 90},
        'existing_primary_states': main, 'existing_half_phase_states': halves,
        'existing_numerical_control_states': controls, 'existing_dc_states': dc,
        'existing_late_states': late, 'planned_repair_tdse': repairs,
        'planned_reusable_high_intensity_probes': probes,
        'readout_sequence': ['existing saved states', 'four repair TDSE',
                             'new repair saved states', 'R1 gate'],
        'initial_library_I_over_mu': [0, .25, .5, 1, 2, 4, 8, 12, 20, 32],
        'initial_library_production_tdse_including_reuse': 72,
        'initial_library_new_production_tdse_including_four_probes': 56,
        'maximum_nonzero_nodes_per_bias': 12, 'maximum_I_over_mu': 64,
        'input_files': sorted(inputs.values(), key=lambda x: x['path'])
    }
    save_new(HERE / 'TASK_MANIFEST.json', result)
    print(f'Wrote planned manifest; {len(inputs)} historical files hash verified. No science run.')


def inventory():
    names = ['EXECUTION_TASKBOOK.md', 'START_HERE.md', 'TASK_MANIFEST.json',
             'PLAN_REVIEW.md', 'build_plan_manifest.py']
    evidence = ['reports/STATUS_REVIEW_20261105.md',
                'plans/20261104_solo_v1/EXECUTION_TASKBOOK.md',
                'results/bias_v1/P2_pilot.json', 'results/bias_v1/P2_error_budget.json']
    paths = [HERE / n for n in names] + [PROJECT / n for n in evidence]
    rows = [{'path': p.relative_to(REPO).as_posix(), 'bytes': p.stat().st_size,
             'sha256': sha(p), 'role': 'new_plan' if p.parent == HERE else 'read_only_evidence'}
            for p in paths]
    save_new(HERE / 'file_inventory.json', {'status': 'documentation_only', 'files': rows})
    print(f'Wrote inventory of {len(rows)} explicit files; no science run.')


def check():
    manifest = json.loads((HERE / 'TASK_MANIFEST.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'planned_not_run'
    assert len(manifest['existing_primary_states']) == 16
    assert len(manifest['existing_numerical_control_states']) == 12
    assert len(manifest['planned_repair_tdse']) == 4
    for item in manifest['input_files']:
        assert sha(PROJECT / item['path']) == item['sha256'], item['path']
    inv = json.loads((HERE / 'file_inventory.json').read_text(encoding='utf-8'))
    for item in inv['files']:
        assert sha(REPO / item['path']) == item['sha256'], item['path']
    print(f"PASS: {len(manifest['input_files'])} old input hashes; {len(inv['files'])} inventory hashes; planned cardinalities. No scientific validation implied.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--build', action='store_true')
    group.add_argument('--inventory', action='store_true')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.build:
        build()
    elif args.inventory:
        inventory()
    else:
        check()
