"""Bounded r1 regression and explicit r0 raw-row migration; no long propagation."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import q8_m0_gamma12_runner_20261006 as r

ROOT = r.REPO
OLD = ROOT / 'results/q_line/p1_h2_finite_kcv_gamma12_runner_continuous_1024_20261006'
FIX = ROOT / 'results/q_line/p1_h2_finite_kcv_gamma12_backend_cuda_resume_20261006'

def independent_u0(cache, manifest):
    """Explicit port formula, independent of runner/U0 assembly code."""
    q = r._backend_modules().q
    from q8_m0_classical import p1_a2_protocol
    prot = p1_a2_protocol(.8)
    omega = q.omega_grid(prot.omega0, prot.t_pulse)
    omega = omega[omega > 0]
    _, norm = q.normalize_phi(omega, q.phi_unnormalized(omega, 2*prot.omega0, prot.t_pulse))
    eta = q.eta_of(float(cache['gamma']))
    expected = np.zeros_like(cache['A'])
    for j, label in enumerate(manifest['columns']['records']):
        if label['kind'] == 'dark':
            continue
        nu = float(label['frequency'])
        out_a = cache['A'][:, j].copy()
        if label['port'] == 'R':
            out_a -= cache['free_A0'][:, j]
        expected[:, j] = q.phi_at(nu, prot, norm)*np.sqrt(nu)/eta*(cache['pol'] @ out_a)
    return expected


def same(a, b):
    error = r._array_error(a, b)
    assert error['allclose'], error
    return error


def merge(manifest_path, block_paths, output, window=None):
    args = ['merge', '--run-root', str(manifest_path.parent), '--manifest', str(manifest_path),
            '--output', str(output), '--require-all']
    if window:
        args += ['--window', window]
    for p in block_paths:
        args += ['--block', str(p)]
    assert r.main(args) == 0


def reject_test(root, manifest, manifest_path, base_paths, checks):
    directory = root/'rejected_cases'
    directory.mkdir()
    m0=json.loads(base_paths[0].read_text(encoding='utf-8'))
    m1=json.loads(base_paths[1].read_text(encoding='utf-8'))
    def altered(name, original, changes, arrays=None):
        meta=copy.deepcopy(original)
        meta.update(changes)
        origin=base_paths[0] if original is m0 else base_paths[1]
        src=origin.parent/original['artifact']['array_path']
        if arrays is not None:
            src=directory/(name+'.npz')
            r._write_npz_atomic(src, arrays)
        meta['artifact']={'array_path':str(src),'sha256':r._sha256_file(src)}
        dest=directory/(name+'.json')
        r._write_json_atomic(dest,meta)
        return dest
    def refused(name, paths, all_cols=False):
        try:
            r._validate_block_metadata(manifest,manifest_path,paths,require_all=all_cols,window=None)
        except ValueError as exc:
            checks[name]={'rejected':True,'message':str(exc)}
        else:
            raise AssertionError('Not rejected: '+name)
    pump=altered('pump_window',m1,r._block_identity(manifest,'pump'))
    refused('mixed_window_without_cli_window',[base_paths[0],pump])
    bad_time=altered('bad_nstep',m0,{'nstep':m0['nstep']+1})
    refused('time_mismatch',[bad_time])
    bad_code=copy.deepcopy(m0['execution_code_sha256'])
    bad_code[next(iter(bad_code))]='not-this-code'
    foreign=altered('foreign_execution',m0,{'execution_code_sha256':bad_code})
    refused('block_execution_mismatch',[foreign])
    payload=r._load_npz(base_paths[0].parent/m0['artifact']['array_path'])
    payload['u0']=payload['undriven_raw_u'].copy()
    wrong_u0=altered('source_as_u0',m0,{},payload)
    refused('source_integral_masquerading_as_u0',[wrong_u0])
    stale=copy.deepcopy(manifest)
    stale['execution_code_sha256']=bad_code
    try:
        r._check_execution(stale)
    except ValueError as exc:
        checks['worker_execution_mismatch']={'rejected':True,'message':str(exc)}
    else:
        raise AssertionError('Stale worker execution accepted')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    out=args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    protected={str(p.relative_to(ROOT)):r._sha256_file(p) for p in OLD.rglob('*') if p.is_file()}
    code=r._source_code_hashes()
    report={'runner_version':r.RUNNER_VERSION,'execution_code_sha256':code,'no_long_propagation':True}
    # New prepare path and CPU fixture split/merge, then real CPU/GPU workers.
    fr=out/'fixture'
    assert r.main(['prepare','--run-root',str(fr),'--fixture',str(FIX/'backend_fixture.npz')]) == 0
    mp=fr/'manifest.json'
    manifest=r._read_manifest(mp)
    _,cache=r._load_manifest_cache(mp,manifest)
    report['fixture_u0_independent']=same(cache['U0_canonical'],independent_u0(cache,manifest))
    assert r.main(['validate','--run-root',str(fr),'--manifest',str(mp),'--backend','cpu',
                   '--window','start','--block-sizes','64,128,64','--reference',str(FIX/'short_rows.npz')]) == 0
    c1=json.loads((fr/'validation_c1/c1_validation.json').read_text(encoding='utf-8'))
    c1_paths=[Path(x) for x in c1['block_metadata']]
    merged=fr/'merged_split.npz'
    merge(mp,c1_paths,merged) # no --window: consistent windows inferred safely
    for backend in ('cpu','cuda'):
        assert r.main(['worker','--run-root',str(fr),'--manifest',str(mp),'--backend',backend,
                       '--window','start','--block-id',backend,'--column-start','0','--column-stop','256']) == 0
    cpu=fr/'blocks/block_cpu_00000000_00000256.json'
    gpu=fr/'blocks/block_cuda_00000000_00000256.json'
    assert r.main(['compare','--run-root',str(out),'--left',str(cpu),'--right',str(gpu),'--output',str(fr/'cpu_gpu.json')]) == 0
    assert r.main(['compare','--run-root',str(out),'--left',str(gpu),'--right',str(merged),'--output',str(fr/'direct_split.json')]) == 0
    report['fixture_cpu_gpu']=json.loads((fr/'cpu_gpu.json').read_text(encoding='utf-8'))['ok']
    report['fixture_direct_split']=json.loads((fr/'direct_split.json').read_text(encoding='utf-8'))['ok']
    report['rejection_tests']={}
    reject_test(out,manifest,mp,c1_paths,report['rejection_tests'])
    del cache
    # Explicit migration reuses reviewed propagation rows; it is not a new run.
    old_mp=OLD/'manifest.json'
    old_man=r._read_manifest(old_mp)
    old_cache_path=OLD/old_man['shared_cache']['path']
    assert r._sha256_file(old_cache_path)==old_man['shared_cache']['sha256']
    for key,value in old_man['code_sha256'].items():
        name=key.replace('\\','/')
        if Path(name).name != Path(r.__file__).name:
            assert r._sha256_file(ROOT/name)==value, ('Changed legacy physics/backend dependency',name)
    arr=r._load_npz(old_cache_path)
    new_root=out/'continuous_reprocessed'
    man=copy.deepcopy(old_man)
    man['cache_provenance']={'kind':'reviewed_r0_cache_reused_without_propagation',
                             'manifest_path':str(old_mp),'manifest_sha256':old_man['manifest_sha256'],
                             'generator_code_sha256':old_man['code_sha256']}
    new_mp=r._write_manifest_and_cache(new_root,arr,man,force=False)
    man=r._read_manifest(new_mp)
    u0=independent_u0(arr,man)
    _,patched=r._load_manifest_cache(new_mp,man)
    report['continuous_u0_independent']=same(patched['U0_canonical'],u0)
    del arr,patched
    block_dir=new_root/'blocks';block_dir.mkdir()
    old_names=['block_gpu256a_00000000_00000256','block_gpu512_00000256_00000768',
               'block_gpu256b_00000768_00001024','block_gpu1024direct_patch_00000000_00001024']
    new_paths=[]; unchanged=[]
    for name in old_names:
        source_meta=OLD/'blocks'/(name+'.json')
        old_meta=json.loads(source_meta.read_text(encoding='utf-8'))
        source_array=source_meta.parent/old_meta['artifact']['array_path']
        assert r._sha256_file(source_array)==old_meta['artifact']['sha256']
        data=r._load_npz(source_array)
        ids=data['column_id']
        fingerprints={key:r._sha256_bytes(data[key].tobytes()) for key in data if key!='u0'}
        data['u0']=u0[:,ids]
        target_array=block_dir/(name+'.npz')
        r._write_npz_atomic(target_array,data)
        assert all(r._sha256_bytes(data[key].tobytes())==value for key,value in fingerprints.items())
        unchanged.append({'block':name,'all_raw_arrays_unchanged':True})
        meta=copy.deepcopy(old_meta)
        meta.update(r._block_identity(man,'continuous'))
        meta.update(manifest_sha256=man['manifest_sha256'],u0_definition='u0=canonical stationary output',
                    artifact={'array_path':target_array.name,'sha256':r._sha256_file(target_array)})
        meta['repair_provenance']={'kind':'U0_only_postprocessing_no_propagation',
                                  'parent_meta':str(source_meta),'parent_meta_sha256':r._sha256_file(source_meta),
                                  'parent_artifact_sha256':old_meta['artifact']['sha256'],
                                  'propagation_runner_version':old_meta['runner_version'],
                                  'old_execution_hash_record_is_not_a_runtime_attestation':True}
        new_meta=block_dir/(name+'.json');r._write_json_atomic(new_meta,meta);new_paths.append(new_meta)
        del data
    merged=new_root/'merged_corrected.npz'
    merge(new_mp,new_paths[:3],merged,'continuous')
    assert r.main(['compare','--run-root',str(out),'--left',str(new_paths[3]),'--right',str(merged),
                   '--output',str(new_root/'direct_vs_split_corrected.json')]) == 0
    result=r._load_npz(merged)
    previous=r._load_npz(OLD/'merged_continuous.npz')
    report['continuous_reprocessed']={
        'raw_rows':unchanged,
        'u0_independent':same(result['u0'],u0),
        'delta_u_unchanged':same(result['delta_u'],previous['delta_u']),
        'V_unchanged':same(result['V'],previous['V']),
        'U_reference':same(result['U'],u0+previous['delta_u']),
        'm_reference':same(result['m'],(u0+previous['delta_u'])@previous['V'].T),
        'U0VT_reference':same(result['m_U0VT'],u0@previous['V'].T),
        'max_U0_correction':float(np.max(np.abs(result['u0']-previous['u0']))),
        'max_m_partial_correction':float(np.max(np.abs(result['m']-previous['m']))),
        'partial_columns_not_physical_moments':True}
    report['protected_legacy_files_unchanged']=all(r._sha256_file(ROOT/key)==value for key,value in protected.items())
    assert report['protected_legacy_files_unchanged']
    assert r._source_code_hashes()==code
    report['legacy_file_hashes']=protected
    report['all_checks_pass']=True
    r._write_json_atomic(out/'repair_checks.json',report)
    print(json.dumps({'all_checks_pass':True,'report':str(out/'repair_checks.json')},indent=2))

if __name__=='__main__':
    main()
