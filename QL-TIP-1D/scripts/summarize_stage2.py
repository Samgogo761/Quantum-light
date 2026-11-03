"""Audit saved 800 nm runs only. Does not propagate or fit power-law slopes."""
from pathlib import Path
import csv
import hashlib
import json

BASE = Path(__file__).resolve().parents[1]
PREFIX = 'stage2_20261103_'
CONTROL_NAMES = ['dx96', 'dt01', 'box70', 'cap_shift', 'cap_half',
                 'time100', 'detector3', 'start60']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_csv(path, names, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(names); writer.writerows(rows)

def main():
    out = BASE/'results/stage2_audit_20261104'
    out.mkdir(exist_ok=False)
    origin = BASE/'data/origin/stage2_audit_20261104'
    origin.mkdir(exist_ok=False)
    runs = {}
    source_paths = [p for d in ['src', 'scripts', 'validation'] for p in (BASE/d).rglob('*.py')]
    source_map = {sha(p):p.relative_to(BASE).as_posix() for p in source_paths}
    all_hashes = 0
    for folder in sorted((BASE/'results').glob(PREFIX+'*')):
        if not (folder/'summary.json').exists():
            raise ValueError(f'Incomplete run: {folder.name}')
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        with (folder/'manifest.csv').open(encoding='utf-8') as f:
            for item in csv.DictReader(f):
                if sha(BASE/item['path']) != item['sha256']:
                    raise ValueError(f"Changed saved artifact: {item['path']}")
                all_hashes += 1
        resolved = {}
        for name, h in meta['source_sha256'].items():
            if h not in source_map:
                raise ValueError(f'Missing historical source {name} {h}')
            resolved[name] = source_map[h]
        name = folder.name.removeprefix(PREFIX)
        runs[name] = dict(summary=summary, metadata=meta, source_resolved=resolved)
    base = runs['baseline']['summary']; y0 = base['yield_right']
    floor = abs(runs['nofield']['summary']['yield_right'])
    comparisons = []
    for name in CONTROL_NAMES:
        s = runs[name]['summary']
        delta = s['yield_right']-y0
        comparisons.append(dict(case=name, yield_right=s['yield_right'],
            absolute_change=abs(delta), relative_change=delta/y0,
            pass_5percent=bool(abs(delta/y0)<.05 and s['numeric_sanity_pass'] and s['plateau_pass'])))
    all_single_sanity = all(r['summary']['numeric_sanity_pass'] for r in runs.values())
    passed = all(c['pass_5percent'] for c in comparisons) and floor < 1e-10 and all_single_sanity
    result = dict(date='2026-11-04', scope='800nm zero-bias baseline single-intensity numerical validation',
        baseline_intensity_inc_W_cm2=1e11, baseline_intensity_apex_W_cm2=9e11,
        baseline_yield=y0, nofield_signed_yield=runs['nofield']['summary']['yield_right'],
        controls=comparisons, gate_b_at_baseline_pass=passed,
        max_observed_absolute_control_change=max(c['absolute_change'] for c in comparisons),
        max_observed_relative_control_change=max(abs(c['relative_change']) for c in comparisons),
        saved_runs=len(runs), validated_manifest_entries=all_hashes,
        cpu_propagation_seconds=sum(r['summary']['propagation_seconds'] for r in runs.values()),
        source_resolution={PREFIX+k:r['source_resolved'] for k,r in runs.items()},
        limitations=['Only I_inc=1e11 has full control suite.',
            'I_inc=5e10 and 2e11 are unrefined pilot points, not slope-fit data.',
            '800nm curve refinement stopped on 2026-11-04 user direction change.',
            'No 1600nm, static bias, electron spectrum, CEP pi, BSV or thermal results.',
            'Observed parameter differences are convergence evidence, not statistical confidence intervals.'])
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    write_csv(origin/'convergence.csv',
        ['case_id','nx','dx_nm','dt_au','x_max_nm','cap_start_nm','yield','relative_delta','norm_budget_error'],
        [[PREFIX+name, runs[name]['summary']['nx'], runs[name]['summary']['dx_nm'],
          runs[name]['summary']['actual_dt_au'], runs[name]['metadata']['config']['xmax_nm'],
          runs[name]['metadata']['config']['right_cap_start_nm'], runs[name]['summary']['yield_right'],
          (runs[name]['summary']['yield_right']-y0)/y0, runs[name]['summary']['max_total_budget_error']]
         for name in ['baseline']+CONTROL_NAMES+['nofield']])
    write_csv(origin/'coherent_pilot_points_not_a_validated_curve.csv',
        ['I_inc_W_cm2','I_apex_W_cm2','yield_right','max_observed_numeric_change','fully_refined','run_id'],
        [[runs[name]['metadata']['config']['intensity_W_cm2'],
          9*runs[name]['metadata']['config']['intensity_W_cm2'], runs[name]['summary']['yield_right'],
          result['max_observed_absolute_control_change'] if name=='baseline' else '',
          int(name=='baseline'), PREFIX+name] for name in ['I5e10','baseline','I2e11']])
    log=['Saved-run audit, generated 2026-11-04 from metadata and completed summaries.',
         'This is not a captured process stdout log. Process exit codes were not retained here.',
         'No propagation or new intensity runs were launched by this audit.']
    for name,r in runs.items():
        meta,s=r['metadata'],r['summary']
        log += ['',PREFIX+name, 'started_at='+meta['time_iso8601'],
                'config_sha256='+meta['config_sha256'],
                'propagation_seconds='+str(s['propagation_seconds']),
                'numeric_sanity_pass='+str(s['numeric_sanity_pass']),
                'right_yield='+str(s['yield_right']),
                'source_resolution='+json.dumps(r['source_resolved'])]
    (BASE/'logs/stage2_saved_runs_audit_20261104.txt').write_text('\n'.join(log)+'\n',encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
