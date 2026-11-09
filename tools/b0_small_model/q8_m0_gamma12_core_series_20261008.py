"""Compare the three core quadratures and their signed node-group sums.

The frequency edges below select nodes by location. They are not complete
quadrature panels, and a large group sum is not by itself an integration error.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

BASE = Path(__file__).resolve().parents[2]/'results/q_line/gamma12_output_transpose_input_refine_20261008'


def pack(x):
    return dict(real=np.asarray(x).real.tolist(), imag=np.asarray(x).imag.tolist())


def main():
    target = BASE/'core_series.json'
    if target.exists():
        raise FileExistsError(target)
    levels, arrays, bands = {}, {}, {}
    edges = [.001, .02, .07, .13, .30, 1., 2.5, 6., 12.]
    for p in (96, 128, 160):
        meta = json.loads((BASE/f'n_panel_{p}.json').read_text(encoding='utf-8'))
        path = BASE/f'n_panel_{p}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['row_sha256']:
            raise ValueError('Array identity mismatch')
        with np.load(path, allow_pickle=False) as z:
            u, v, om = z['U'], z['V'], z['oms']
            arrays[p] = dict(n=z['n'], m=z['m'], normal=z['normal_contribution']-np.eye(2),
                             anomalous=z['anomalous_contribution'])
            bands[p] = {}
            omega = np.repeat(om, 4)
            selections = {f'{lo:g}--{hi:g}': np.flatnonzero((omega>=lo)&(omega<hi))
                          for lo, hi in zip(edges[:-1], edges[1:])}
            selections['dark'] = np.arange(len(omega), u.shape[1])
            for name, ids in selections.items():
                ub, vb = u[:, ids], v[:, ids]
                bands[p][name] = dict(normal=ub@ub.conj().T-vb@vb.conj().T,
                                     anomalous=ub@vb.T-vb@ub.T,
                                     trace_n=float(np.sum(abs(vb)**2)), columns=len(ids))
            nsum = sum(b['normal'] for b in bands[p].values())-np.eye(2)
            if not np.allclose(nsum, arrays[p]['normal'], atol=2e-9, rtol=1e-8):
                raise AssertionError('Frequency partition omitted/doubled columns')
        levels[p] = {k:meta[k] for k in ('n_nodes','n_contracted_columns','n_trace',
                        'normal_ccr_maxabs','anomalous_maxabs','total_wall_s','row_sha256')}
    steps = {}
    for a, b in ((96,128),(128,160)):
        rec = {}
        for key in arrays[a]:
            delta = arrays[b][key]-arrays[a][key]
            rec[key] = dict(maxabs=float(np.max(abs(delta))),
                            relative_fro=float(np.linalg.norm(delta)/max(np.linalg.norm(arrays[b][key]),1e-30)),
                            signed_delta=pack(delta))
        rec['band_changes'] = {name: dict(
            normal_maxabs=float(np.max(abs(bands[b][name]['normal']-bands[a][name]['normal']))),
            anomalous_maxabs=float(np.max(abs(bands[b][name]['anomalous']-bands[a][name]['anomalous']))),
            trace_n_delta=bands[b][name]['trace_n']-bands[a][name]['trace_n']) for name in bands[a]}
        steps[f'{a}_to_{b}'] = rec
    result = dict(status='three_core_levels_not_convergence_proof', levels=levels, steps=steps,
                  ntpc=4096, n_h2=48, unchanged_tail_panels=True, unchanged_frequency_interval=[.001,12.],
                  normal_matrices={p:pack(arrays[p]['normal']) for p in arrays},
                  anomalous_matrices={p:pack(arrays[p]['anomalous']) for p in arrays},
                  physical_qualification=False, gauss_node_count_not_time_richardson=True)
    with target.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'levels':levels,'step_summary':{s:{k:{i:v for i,v in r.items() if i!='signed_delta'}
                                                       for k,r in rec.items() if k!='band_changes'}
                                                    for s,rec in steps.items()}},indent=2))


if __name__=='__main__':
    main()
