"""Read-only decomposition of saved local Gamma12 output rows.

Writes new diagnostics only. No propagation and no reinterpretation of raw
candidate moments as qualified photons, squeezing, or magnetic contrast.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/q_line/gamma12_output_diagnostic_20261007'

def mat(a):
    a=np.asarray(a)
    return dict(real=a.real.tolist(),imag=a.imag.tolist(),maxabs=float(np.max(np.abs(a))),fro=float(np.linalg.norm(a)))

def terms(U0,dU,V,subtract_identity=False):
    U=U0+dU
    base=U0@U0.conj().T
    cross=U0@dU.conj().T+dU@U0.conj().T
    quadratic=dU@dU.conj().T-V@V.conj().T
    normal=U@U.conj().T-V@V.conj().T
    a0=U0@V.T-V@U0.T
    ad=dU@V.T-V@dU.T
    return dict(n_trace=float(np.sum(np.abs(V)**2)),
                m=mat(U@V.T),u0_weight=mat(base),
                normal=mat(normal-np.eye(2) if subtract_identity else normal),
                normal_pump_increment=mat(cross+quadratic),
                normal_cross=mat(cross),normal_quadratic=mat(quadratic),
                anomalous=mat(a0+ad),anomalous_reference_cross=mat(a0),
                anomalous_source_source=mat(ad),
                uu=mat(U@U.conj().T),vv=mat(V@V.conj().T))

def analyze(domain):
    root=ROOT/f'results/q_line/p1_h2_finite_kcv_gamma12_local_{domain}_20261007'
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    meta=json.loads((root/'merged.json').read_text(encoding='utf-8'))
    with np.load(root/'merged.npz',allow_pickle=False) as z:
        U0,dU,V,U=[z[k] for k in ('u0','delta_u','V','U')]
        raw_und_u,raw_und_v=[z[k] for k in ('undriven_raw_u','undriven_raw_v')]
    records=manifest['columns']['records']
    omega=np.load(root/manifest['files']['omega']['path'])
    photon=np.array([r['kind']=='photon' for r in records])
    # Node-location groups, not complete quadrature panels.
    bands=[(0,.02),(.02,.07),(.07,.13),(.13,.3),(.3,1),(1,2.5),(2.5,6),(6,12),(12,np.inf)]
    groups={}
    for lo,hi in bands:
        mask=photon&(omega>=lo)&(omega<hi)
        if mask.any(): groups[f'frequency_{lo}_{hi}']=dict(columns=int(mask.sum()),**terms(U0[:,mask],dU[:,mask],V[:,mask]))
    mask=~photon
    groups['dark']=dict(columns=int(mask.sum()),**terms(U0[:,mask],dU[:,mask],V[:,mask]))
    weights=np.sum(abs(V)**2,axis=0)
    normal_cols=np.einsum('ic,jc->cij',U,U.conj())-np.einsum('ic,jc->cij',V,V.conj())
    normal_pump_cols=normal_cols-np.einsum('ic,jc->cij',U0,U0.conj())
    anti_cols=np.einsum('ic,jc->cij',U,V)-np.einsum('ic,jc->cij',V,U)
    normal_rank=np.linalg.norm(normal_pump_cols,axis=(1,2))
    ranks={}
    for name,score in [('source_weight',weights),('normal_increment',normal_rank),('anomalous',np.linalg.norm(anti_cols,axis=(1,2)))]:
        order=np.argsort(score)[::-1]
        ranks[name]=[dict(column_id=int(i),label=records[int(i)],score=float(score[i]),
                          V_norm=float(np.linalg.norm(V[:,i])),U_norm=float(np.linalg.norm(U[:,i]))) for i in order[:24]]
    # Long-double summation diagnoses reduction roundoff, not quadrature error.
    dtype=np.clongdouble
    ul,vl=U.astype(dtype),V.astype(dtype)
    normal_ld=np.asarray(ul@ul.conj().T-vl@vl.conj().T-np.eye(2),complex)
    anti_ld=np.asarray(ul@vl.T-vl@ul.T,complex)
    normal=U@U.conj().T-V@V.conj().T-np.eye(2)
    anti=U@V.T-V@U.T
    result=dict(domain=domain,source_manifest_sha256=manifest['manifest_sha256'],
                source_merged_sha256=meta['output_sha256'],
                total=terms(U0,dU,V,True),groups=groups,top_columns=ranks,
                stationary_u0_ccr=mat(U0@U0.conj().T-np.eye(2)),
                normal_summation_longdouble_difference=float(np.max(abs(normal-normal_ld))),
                anomalous_summation_longdouble_difference=float(np.max(abs(anti-anti_ld))),
                longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant),
                undriven_raw_u_norm=float(np.linalg.norm(raw_und_u)),
                undriven_raw_v_norm=float(np.linalg.norm(raw_und_v)),
                relative_to_large_gram=float(np.max(abs(normal))/max(np.max(abs(U@U.conj().T)),np.max(abs(V@V.conj().T)))),
                physical_qualification=False,
                note='Relative-to-large-Gram is cancellation context only; vacuum-scale absolute CCR is not relaxed.')
    # Complete four port/polarization columns for 8 top-contribution frequencies,
    # plus fixed coverage of low/H2/mid/high and dark directions.
    selected=set()
    nph=int(manifest['physical_scope']['n_photon_columns'])
    freq_scores=weights[:nph].reshape(-1,4).sum(axis=1)
    for fi in np.argsort(freq_scores)[-8:]: selected.update(range(4*int(fi),4*int(fi)+4))
    for lo,hi in bands:
        fis=np.flatnonzero(photon[::4][:nph//4] & (omega[:nph:4]>=lo) & (omega[:nph:4]<hi))
        if fis.size:
            for fi in [fis[0],fis[len(fis)//2],fis[-1]]: selected.update(range(4*int(fi),4*int(fi)+4))
    dark_ids=np.flatnonzero(~photon)
    for i in dark_ids[np.argsort(weights[dark_ids])[-16:]]: selected.add(int(i))
    result['suggested_probe_ids']=sorted(selected)
    result['probe_source_weight_fraction']=float(weights[list(selected)].sum()/weights.sum())
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'saved_row_decomposition.json'
    if target.exists(): raise FileExistsError(target)
    records=[analyze(d) for d in ('plusN','minusN')]
    target.write_text(json.dumps(records,indent=2),encoding='utf-8')
    for r in records:
        print(r['domain'],json.dumps({k:r[k] for k in ('stationary_u0_ccr','normal_summation_longdouble_difference','relative_to_large_gram','probe_source_weight_fraction')},ensure_ascii=False))
        print('bands',[(k,v['n_trace'],v['normal_pump_increment']['maxabs']) for k,v in r['groups'].items()])
        print('largest',r['top_columns']['source_weight'][:2])

if __name__=='__main__':main()
