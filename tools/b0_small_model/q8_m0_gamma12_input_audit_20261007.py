"""Read-only audit of the actual full Gamma12 incoming-column cache.

No propagation or production-file modification.  Project actual cached
columns, not regenerated reduced-map columns, onto the frozen bright basis.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
ROOT = TOOLS.parents[1]
import q8_m0_p1_h2_finite_kcv_gamma12_ccr_fix_20261006 as fx
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_incoming_spectral_completeness_20261004 import dark_subspace

def ma(x): return float(np.max(np.abs(x))) if np.size(x) else 0.0
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=ROOT/'results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007')
    parser.add_argument('--output', type=Path, default=ROOT/'results/q_line/gamma12_output_diagnostic_20261007/input_audit.json')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    started=time.perf_counter()
    run=args.run_root; manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    arrays={p.stem:np.load(p,mmap_mode='r') for p in (run/'arrays').glob('*.npy')}
    gaps=np.asarray(arrays['gaps']); u=np.asarray(arrays['u0'])
    B=fx.bright_basis(gaps,u)
    dark,info,near=dark_subspace(gaps,u)
    nph=int(manifest['physical_scope']['n_photon_columns'])
    ncol=int(manifest['physical_scope']['n_columns'])
    nb=B.shape[1]; nd=dark.shape[1]
    print(f'actual cache audit {ncol=} {nb=} {nd=}',flush=True)
    device=torch.device('cuda')
    bt=torch.as_tensor(B,device=device,dtype=torch.complex128)
    acc=[torch.zeros((nb,nb),device=device,dtype=torch.complex128) for _ in range(4)]
    t0=time.perf_counter()
    for start in range(0,nph,1024):
        stop=min(start+1024,nph)
        bb=torch.as_tensor(np.array(arrays['b_cols'][start:stop].T),device=device)
        dd=torch.as_tensor(np.array(arrays['bdag_cols'][start:stop].T),device=device)
        b=bt.conj().T @ bb; d=bt.T @ dd
        acc[0]+=b@b.conj().T
        acc[1]+=d.conj()@d.T
        acc[2]+=b@d.conj().T
        acc[3]+=d.conj()@b.T
        if start%10240==0: print(f'projected {stop}/{nph}',flush=True)
    torch.cuda.synchronize()
    ab,ad,abd,adb=[a.cpu().numpy() for a in acc]
    normal=ab-ad-np.eye(nb); anomalous=abd-adb
    spec=np.linalg.eigvalsh((normal+normal.conj().T)*.5)
    cache_dark=np.asarray(arrays['b_cols'][nph:].T)
    dark_checks={
       'cache_vs_rebuilt_basis_maxabs':ma(cache_dark-dark),
       'bdag_maxabs':ma(arrays['bdag_cols'][nph:]),
       'A_maxabs':ma(arrays['A_cols'][nph:]),
       'free_A0_maxabs':ma(arrays['free_A0_cols'][nph:]),
       'U0_maxabs':ma(arrays['U0_canonical'][:,nph:]),
       'current_maxabs':ma(u.conj()@cache_dark),
       'bright_projection_maxabs':ma(B.conj().T@cache_dark),
       'mass_trace':float(np.sum(np.abs(cache_dark)**2)),
       'expected_mass':nd,
    }
    # All support is confined to exact gap clusters.  Compare the saved
    # basis to the fresh one above; independently check its cluster Gram.
    offset=0; orth=0.0; gaperr=0.0
    for entry in info:
        size=entry['nullity']; idx=entry['indices']
        block=cache_dark[np.ix_(idx,range(offset,offset+size))]
        orth=max(orth,ma(block.conj().T@block-np.eye(size)))
        gaperr=max(gaperr,ma(np.asarray(arrays['dark_gaps'][offset:offset+size])-entry['gap']))
        offset+=size
    dark_checks.update(cluster_orthogonality_maxabs=orth,gaps_maxabs=gaperr,
                       counted_once=(offset==nd and nph+nd==ncol))
    uu=np.asarray(arrays['U0_canonical']); u0ccr=uu@uu.conj().T-np.eye(2)
    records=manifest['columns']['records']; ids=[r['id'] for r in records]
    summary=json.loads((ROOT/'results/q_line/p1_h2_finite_kcv_gamma12_conj_prop_20261006/gamma12_static_accept_summary.json').read_text())
    old=summary['static_ccr']
    hashes=manifest['execution_code_sha256']
    changed=[p for p,h in hashes.items() if sha(ROOT/p)!=h]
    result={
      'status':'actual_cache_input_audit_no_propagation',
      'run_root':str(run),'manifest_sha256':manifest['manifest_sha256'],
      'n_el':len(gaps),'n_bright':nb,'n_dark':nd,'n_photon':nph,'ncol':ncol,
      'b_projection':'B.conj().T @ b','bdag_projection':'B.T @ bdag',
      'normal_maxabs':ma(normal),'anomalous_maxabs':ma(anomalous),
      'normal_min_eig':float(spec[0]),'normal_max_eig':float(spec[-1]),
      'normal_operator_norm':float(np.max(np.abs(spec))),
      'trace_bb_minus_dd':float(np.trace(ab-ad).real),
      'old_normal_maxabs':old['n96_normal'],'old_anomalous_maxabs':old['n96_anomalous'],
      'normal_value_delta':ma(normal)-old['n96_normal'],
      'anomalous_value_delta':ma(anomalous)-old['n96_anomalous'],
      'dark_checks':dark_checks,'U0_CCR_maxabs':ma(u0ccr),
      'labels_all_ids_unique_and_complete':len(set(ids))==ncol and len(ids)==ncol,
      'input_frequency_weights_positive':bool(np.all(arrays['wts']>0)),
      'cached_omega_matches_four_columns_per_frequency':ma(arrays['omega'][:nph]-np.repeat(arrays['oms'],4)),
      'executed_code_hash_mismatches':changed,
      'elapsed_s':time.perf_counter()-started,'project_s':time.perf_counter()-t0,
      'production_files_modified':False,'new_time_propagation':False,
      'interpretation':'Finite 0.05 static trial only. Does not certify driven output.',
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    arrays_path=args.output.with_name('input_bright_matrices.npz')
    if arrays_path.exists(): raise FileExistsError(arrays_path)
    np.savez(arrays_path,bright_basis=B,normal=normal,anomalous=anomalous,bb=ab,dd=ad,U0_ccr=u0ccr)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
