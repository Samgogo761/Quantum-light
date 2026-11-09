"""Limited-column true time refinement; unchanged frozen input and detector.

Source files are read-only. Fresh electronic trajectories are computed at each
step size; no interpolation of the coarse vertices. Partial columns do not
provide a canonical output or a full quantum moment qualification.
"""
from pathlib import Path
import argparse
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json
import time
import numpy as np
import q8_m0_gamma12_local_prepare_20261007 as prep
import q8_m0_gamma12_local_runner_20261007 as local
import q8_m0_gamma12_runner_20261006 as r1

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/q_line/gamma12_output_diagnostic_20261007'
SRC=ROOT/'results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007'

def err(a,b):
    delta=np.asarray(a)-np.asarray(b)
    return dict(maxabs=float(np.max(abs(delta))),fro=float(np.linalg.norm(delta)),
                relative_fro=float(np.linalg.norm(delta)/max(np.linalg.norm(b),1e-30)))

def prepare_refine(ntpc):
    from q8_m0_model import build_geometry
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
    out=BASE/f'time_{ntpc}'
    if (out/'prepared.json').exists():
        return out
    if out.exists(): raise FileExistsError(out)
    (out/'arrays').mkdir(parents=True)
    source=prep.load_manifest(SRC)
    geo=build_geometry()
    sample,_=grid_sample(geo,12,0.,neel=1)
    checks={}
    for key,sourcekey in [('gap','gaps'),('u','u0'),('D','D0'),('kc','kc'),('weights','weights')]:
        old=np.load(SRC/source['files'][sourcekey]['path'])
        checks[key]=err(sample[key],old)
        if not np.allclose(sample[key],old,atol=1e-12,rtol=1e-11):
            raise ValueError('Refined trajectory changed initial basis: '+key)
    prot=p1_a2_protocol(source['physical_scope']['a_star'])
    h=prot.period/ntpc; nstep=9*ntpc
    started=time.perf_counter()
    prep._vertex_files(out,geo,sample,prot,'driven',nstep,h)
    prep._vertex_files(out,geo,sample,p1_a2_protocol(0.),'undriven',nstep,h)
    kernel,_=r1._backend_modules().backend.kernel_at_times(source['physical_scope']['gamma'],prot,(np.arange(nstep)+.5)*h)
    np.save(out/'arrays/k.npy',kernel)
    meta=dict(ntpc=ntpc,nstep=nstep,h=float(h),t0=0.,source_manifest=source['manifest_sha256'],
              initial_basis_checks=checks,execution_code_sha256=prep.execution_hashes(),
              prepare_wall_s=time.perf_counter()-started,files=prep._file_records(out))
    prep.write_json_atomic(out/'prepared.json',meta)
    return out

def run_probe(ntpc):
    out=BASE/f'time_{ntpc}'
    out.mkdir(exist_ok=True)
    target=out/'rows.npz'
    if target.exists(): raise FileExistsError(target)
    diag=json.loads((BASE/'saved_row_decomposition.json').read_text())[0]
    ids=np.asarray(diag['suggested_probe_ids'],int)
    manifest=prep.load_manifest(SRC)
    cache=prep.load_arrays(SRC,manifest)
    data=local.block_inputs(manifest,cache,ids)
    if ntpc != 1024:
        info=json.loads((out/'prepared.json').read_text())
        if info['execution_code_sha256']!=prep.execution_hashes(): raise ValueError('Code changed')
        for name,rec in info['files'].items():
            if prep.sha256_file(out/rec['path'])!=rec['sha256']: raise ValueError('Refined cache changed')
        data['vertices']={p:(np.load(out/f'arrays/{p}_u.npy',mmap_mode='r'),np.load(out/f'arrays/{p}_d.npy',mmap_mode='r')) for p in ('driven','undriven')}
        data['k']=np.load(out/'arrays/k.npy')
        data['h']=info['h'];data['nstep']=info['nstep']
    started=time.perf_counter()
    mods=r1._backend_modules()
    d=r1._run_one(data,'driven','cuda',mods)
    u=r1._run_one(data,'undriven','cuda',mods)
    du=d['delta_u']-u['delta_u'];V=d['v']-u['v'];U0=data['u0_canonical'];U=U0+du
    rows=dict(column_id=ids,U=U,V=V,u0=U0,delta_u=du,n=V.conj()@V.T,m=U@V.T,
              driven_raw_u=d['delta_u'],undriven_raw_u=u['delta_u'],
              driven_raw_v=d['v'],undriven_raw_v=u['v'],
              driven_final_b=d['state'].b,driven_final_bdag=d['state'].bdag,
              driven_final_A=d['state'].A,undriven_final_A=u['state'].A)
    r1._finite_arrays(rows);r1._write_npz_atomic(target,rows)
    with np.load(SRC/'merged.npz') as z:
        reference={k:z[k][:,ids] for k in ('U','V','delta_u','u0','driven_raw_u','undriven_raw_u','driven_raw_v','undriven_raw_v')}
    reference['n']=reference['V'].conj()@reference['V'].T
    reference['m']=reference['U']@reference['V'].T
    comparison={k:err(rows[k],v) for k,v in reference.items()}
    meta=dict(ntpc=ntpc,columns=len(ids),nstep=data['nstep'],wall_s=time.perf_counter()-started,
              source_manifest=manifest['manifest_sha256'],source_output_sha=prep.sha256_file(SRC/'merged.npz'),
              selection_captures_coarse_weight_fraction=diag['probe_source_weight_fraction'],
              comparison_vs_saved_1024=comparison,
              partial_n_trace=float(np.trace(rows['n']).real),
              partial_only_no_ccr_qualification=True,physical_conclusion=False,
              execution_code_sha256=prep.execution_hashes(),row_sha256=prep.sha256_file(target),
              gpu_memory={p:{k:r.get(k) for k in ('peak_allocated_bytes','peak_reserved_bytes','wall_s')}
                          for p,r in [('driven',d),('undriven',u)]})
    prep.write_json_atomic(out/'time_probe.json',meta)
    print(json.dumps(meta,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);ap.add_argument('--ntpc',type=int,required=True);args=ap.parse_args()
    if args.action=='prepare': prepare_refine(args.ntpc)
    else: run_probe(args.ntpc)

if __name__=='__main__':main()
