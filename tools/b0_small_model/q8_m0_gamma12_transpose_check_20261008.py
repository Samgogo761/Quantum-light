"""Check exact discrete output transpose against frozen Gamma12 forward rows.

New candidate only: unchanged input columns, time vertices, U0 and detector.
All outputs live in a new directory; no old propagation files are modified.
"""
from pathlib import Path
import argparse
import json
import os
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import q8_m0_gamma12_local_prepare_20261007 as prep
import q8_m0_gamma12_local_runner_20261007 as local
import q8_m0_gamma12_runner_20261006 as r1
import q8_m0_output_transpose_20261008 as tr

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007'
OLD = ROOT / 'results/q_line/gamma12_output_diagnostic_20261007'
OUT = ROOT / 'results/q_line/gamma12_output_transpose_20261008'
ATOL, RTOL = 1e-10, 1e-8


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return dict(maxabs=float(np.max(abs(a-b))),
                relative_fro=float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30)),
                allclose=bool(np.allclose(a, b, atol=ATOL, rtol=RTOL)))


def setup(ntpc, manifest, cache):
    data = local.block_inputs(manifest, cache, [0])
    extra = dict(source_manifest=manifest['manifest_sha256'], ntpc=ntpc,
                 execution_code_sha256=prep.execution_hashes())
    if ntpc != 1024:
        p = OLD / f'time_{ntpc}'
        info = json.loads((p/'prepared.json').read_text(encoding='utf-8'))
        if info['execution_code_sha256'] != prep.execution_hashes():
            raise ValueError('Frozen source code differs from refined cache')
        if info['source_manifest'] != manifest['manifest_sha256']:
            raise ValueError('Refined cache source identity differs')
        for name, rec in info['files'].items():
            if prep.sha256_file(p/rec['path']) != rec['sha256']:
                raise ValueError('Refined cache changed: '+name)
        data['vertices'] = {path: (np.load(p/f'arrays/{path}_u.npy', mmap_mode='r'),
                                  np.load(p/f'arrays/{path}_d.npy', mmap_mode='r'))
                            for path in ('driven', 'undriven')}
        data['k'] = np.load(p/'arrays/k.npy')
        data['h'], data['nstep'] = info['h'], info['nstep']
        extra['refined_prepared_sha256'] = prep.sha256_file(p/'prepared.json')
    extra['transpose_code_sha256'] = prep.sha256_file(Path(tr.__file__))
    return data, extra


def get_backward(ntpc, data, identity):
    directory = OUT/f'ntpc_{ntpc}'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/'backward.npz'
    meta_file = directory/'backward.json'
    if target.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding='utf-8'))
        if meta['identity'] != identity or prep.sha256_file(target) != meta['sha256']:
            raise ValueError('Backward cache identity/hash mismatch')
        with np.load(target, allow_pickle=False) as z:
            rows = {p: tr.BackwardRows(z[p+'_row0'], z[p+'_forcing'], data['pol'].shape[0],
                                      data['nstep'], data['h'], meta['timing_s'][p])
                    for p in ('driven', 'undriven')}
        return rows, meta
    if target.exists() or meta_file.exists():
        raise FileExistsError('Incomplete backward artifact; preserve for diagnosis')
    rows = {}
    for path in ('driven', 'undriven'):
        print(f'ntpc={ntpc}: reverse {path}', flush=True)
        u, d = data['vertices'][path]
        rows[path] = tr.backward_rows(u, d, data['k'], data['pol'], data['h'], data['gamma'])
    r1._write_npz_atomic(target, {p+'_'+key: getattr(rec, key)
                                for p, rec in rows.items() for key in ('row0', 'forcing')})
    meta = dict(identity=identity, sha256=prep.sha256_file(target),
                timing_s={p: rec.elapsed_s for p, rec in rows.items()})
    prep.write_json_atomic(meta_file, meta)
    return rows, meta


def run(ntpc, full, manifest, cache):
    start = time.perf_counter()
    directory = OUT/f'ntpc_{ntpc}'
    label = 'full' if full else 'probe'
    result = directory/(label+'.json')
    if result.exists() or result.with_suffix('.npz').exists():
        raise FileExistsError(result)
    if full:
        gate = json.loads((directory/'probe.json').read_text(encoding='utf-8'))
        if not gate['all_checks_pass']:
            raise ValueError('Real 144-column comparison failed')
        if ntpc != 1024:
            gate0 = json.loads((OUT/'ntpc_1024/full.json').read_text(encoding='utf-8'))
            if not gate0['all_checks_pass']:
                raise ValueError('Full 1024 comparison failed')
        ids = np.arange(manifest['physical_scope']['n_columns'], dtype=np.int64)
    else:
        with np.load(OLD/f'time_{ntpc}/rows.npz', allow_pickle=False) as z:
            ids = z['column_id']
    data, identity = setup(ntpc, manifest, cache)
    rows, backward_meta = get_backward(ntpc, data, identity)
    collected = {p: {k: np.empty((2, len(ids)), complex) for k in ('raw_u', 'raw_v')}
                 for p in ('driven', 'undriven')}
    block_times = {p: 0. for p in collected}
    block_size = 512
    for pos in range(0, len(ids), block_size):
        chosen = ids[pos:pos+block_size]
        # Every selected column retains all electronic k coordinates.
        args = [np.array(cache[key][chosen].T, copy=True)
                for key in ('b_cols', 'bdag_cols', 'A_cols')]
        omega = np.array(cache['omega'][chosen])
        free = np.array(cache['free_A0_cols'][chosen].T, copy=True)
        for p in collected:
            ans = tr.contract_columns(rows[p], *args, omega, free,
                                      t0=data['t0'], frequency_block=128, time_block=2048)
            for key in ('raw_u', 'raw_v'):
                collected[p][key][:, pos:pos+len(chosen)] = ans[key]
            block_times[p] += ans['wall_s']
        if full and ((pos//block_size)%16 == 0):
            print(f'ntpc={ntpc}: contracted {pos+len(chosen)}/{len(ids)} columns', flush=True)
    d, u = collected['driven'], collected['undriven']
    du, v = d['raw_u']-u['raw_u'], d['raw_v']-u['raw_v']
    u0 = np.array(cache['U0_canonical'][:, ids], copy=True)
    U = u0+du
    nm = dict(column_id=ids, u0=u0, U=U, V=v, delta_u=du,
              driven_raw_u=d['raw_u'], driven_raw_v=d['raw_v'],
              undriven_raw_u=u['raw_u'], undriven_raw_v=u['raw_v'],
              n=v.conj()@v.T, m=U@v.T,
              m_U0VT=u0@v.T, m_source_source=du@v.T,
              normal_contribution=U@U.conj().T-v@v.conj().T,
              anomalous_contribution=U@v.T-v@U.T)
    reference_path = SRC/'merged.npz' if full else OLD/f'time_{ntpc}/rows.npz'
    checks = {}
    # Finer full outputs have no independent forward reference yet.
    if not full or ntpc == 1024:
        with np.load(reference_path, allow_pickle=False) as z:
            for key in ('u0', 'U', 'V', 'delta_u', 'driven_raw_u', 'driven_raw_v',
                        'undriven_raw_u', 'undriven_raw_v', 'n', 'm'):
                checks[key] = compare(nm[key], z[key])
            ref_u, ref_v = z['U'], z['V']
            checks['normal_contribution'] = compare(nm['normal_contribution'],
                                                    ref_u@ref_u.conj().T-ref_v@ref_v.conj().T)
            checks['anomalous_contribution'] = compare(nm['anomalous_contribution'],
                                                       ref_u@ref_v.T-ref_v@ref_u.T)
    r1._finite_arrays(nm)
    r1._write_npz_atomic(result.with_suffix('.npz'), nm)
    meta = dict(ntpc=ntpc, nstep=data['nstep'], n_columns=len(ids), identity=identity,
                comparison=checks, all_checks_pass=all(v['allclose'] for v in checks.values()) if checks else None,
                reference_kind='saved_forward' if checks else 'finer_full_no_forward_reference',
                n_trace=float(np.trace(nm['n']).real),
                normal_ccr_maxabs=float(np.max(abs(nm['normal_contribution']-np.eye(2)))) if full else None,
                anomalous_maxabs=float(np.max(abs(nm['anomalous_contribution']))),
                partial_columns_only=not full, physical_qualification=False,
                wall_s=time.perf_counter()-start, contraction_wall_s=block_times,
                backward=backward_meta, row_sha256=prep.sha256_file(result.with_suffix('.npz')),
                reference_sha256=prep.sha256_file(reference_path) if checks else None,
                atol=ATOL, rtol=RTOL)
    prep.write_json_atomic(result, meta)
    print(json.dumps({k: meta[k] for k in ('ntpc', 'n_columns', 'all_checks_pass', 'n_trace', 'normal_ccr_maxabs', 'wall_s')}), flush=True)
    if checks and not meta['all_checks_pass']:
        raise AssertionError('Output transpose differs from saved forward rows')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ntpc', nargs='+', type=int, choices=(1024, 2048, 4096), required=True)
    ap.add_argument('--full', action='store_true')
    args = ap.parse_args()
    manifest, cache = local._load(SRC)
    for ntpc in args.ntpc:
        run(ntpc, args.full, manifest, cache)


if __name__ == '__main__':
    main()
