#!/usr/bin/env python3
# Q-MATH-16.28. Classical susceptibility -> anomalous response PROXY only.
# Not a quantum-light HHG scattering amplitude or physically normalized emitted field.
from __future__ import annotations
import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[v]='1'
from pathlib import Path
import sys, time, json, math
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=(ROOT/'q16_27_reference') if (ROOT/'q16_27_reference').exists() else Path('/mnt/data/QMATH_16_27/src')
sys.path.insert(0,str(OLD))
from qmath_16_27_retarded_kernel import kernel
CACHE=ROOT/'kernel_cache.json'
RES=ROOT/'qmath_16_28_results.json'
CENTER=.18; OM=.36; SIG=.043

def kkey(L,D,eta,x,lam=.1): return f'L{L}_D{D:+.3f}_eta{eta:.4f}_x{x:.4f}_lam{lam:+.2f}'

def get_data():
    d=json.loads(CACHE.read_text()) if CACHE.exists() else {}
    # prefer reusing the original Q16.27 dataset as a cross-check for existing points
    original=json.loads((OLD/'qmath_16_27_shape_scan.json').read_text())['dataset']
    for q in original:
        if q['L'] in [161,201] and q['Delta'] in [.4,.8] and abs(q['eta']-.08)<1e-9:
            d.setdefault(kkey(q['L'],q['Delta'],q['eta'],q['w1']),q['chi12'])
    # 201 checks .08, .03; 161 checks .08, .03; 101 checks .03.
    cases=[(.08,161),(.08,201),(.03,101),(.03,161),(.03,201)]
    plan=[(eta,L,D,float(round(x,2))) for eta,L in cases for D in (.4,.8)
          for x in np.linspace(.10,.18,9)]
    changed=0; start=time.monotonic()
    for i,(eta,L,D,x) in enumerate(plan):
        key=kkey(L,D,eta,x)
        if key not in d:
            d[key]=kernel(L=L,Delta=D,eta=eta,w1=x,w2=OM-x)['chi12'];changed+=1
        if (i+1)%10==0:
            CACHE.write_text(json.dumps(d,indent=2),encoding='utf-8')
            print(f'{i+1}/{len(plan)} kernel points; new={changed}; {time.monotonic()-start:.1f}s',flush=True)
    CACHE.write_text(json.dumps(d,indent=2),encoding='utf-8')
    return d,cases

def projected(d,eta,L,D,h):
    ws=np.linspace(.10,.26,int(round(.16/h))+1)
    xs=np.linspace(.10,.18,int(round(.08/h))+1)
    values=[complex(*d[kkey(L,D,eta,float(round(x,2)))]) for x in xs]
    chi=np.array(values+list(reversed(values[:-1])),dtype=complex)
    assert len(ws)==len(chi)
    wt=np.full(len(ws),h); wt[[0,-1]]*=.5
    g=np.exp(-.5*((ws-CENTER)/SIG)**2)
    even=g/np.sqrt(np.sum(wt*g*g))
    odd=(ws-CENTER)*even
    odd=odd/np.sqrt(np.sum(wt*odd*odd))
    overlap=np.sum(wt*even*odd)
    Ke=np.sum(wt*chi*even*even[::-1]);Ko=np.sum(wt*chi*odd*odd[::-1])
    cross=np.sum(wt*chi*(even*odd[::-1]+odd*even[::-1])/2)
    C=values[-1]
    flat_e=np.sum(wt*C*even*even[::-1]);flat_o=np.sum(wt*C*odd*odd[::-1])
    return {'chi_center':C,'ke':Ke,'ko':Ko,'cross':cross,'flat_e':flat_e,'flat_o':flat_o,
            'u_overlap':overlap,'dark':Ke+Ko,'dark_rel':abs(Ke+Ko)/max(abs(Ke),1e-16),
            'ratio':Ko/Ke,'flat_dark':flat_e+flat_o}

def integrate_5grid_against_9(d,eta,L,D):
    # use the same 9 unique point table but sample every other frequency
    w=np.arange(.10,.2600001,.02);wt=np.full(len(w),.02);wt[[0,-1]]*=.5
    g=np.exp(-.5*((w-CENTER)/SIG)**2)
    e=g/np.sqrt(np.sum(wt*g*g));o=(w-CENTER)*e;o=o/np.sqrt(np.sum(wt*o*o))
    vals=[complex(*d[kkey(L,D,eta,float(round(x,2)))]) for x in np.arange(.10,.1800001,.02)]
    chi=np.array(vals+vals[-2::-1])
    ke=np.sum(wt*chi*e*e[::-1]);ko=np.sum(wt*chi*o*o[::-1])
    return {'ke':ke,'ko':ko,'dark_rel':abs(ke+ko)/abs(ke)}

def complexpair(z):return [float(np.real(z)),float(np.imag(z))]

def case_metrics(d,eta,L,D,h):
    r=projected(d,eta,L,D,h);s=integrate_5grid_against_9(d,eta,L,D)
    return dict(L=L,eta=eta,Delta=D,freq_step=h,
        ke=complexpair(r['ke']),ko=complexpair(r['ko']),cross=complexpair(r['cross']),
        flat_e=complexpair(r['flat_e']),flat_o=complexpair(r['flat_o']),
        flat_dark=abs(r['flat_dark']),dark=complexpair(r['dark']),dark_rel=r['dark_rel'],
        ratio=complexpair(r['ratio']),cross_abs=abs(r['cross']),u_overlap=abs(r['u_overlap']),
        freq_coarse_dark_rel=s['dark_rel'],freq_coarse_minus_fine=abs(s['dark_rel']-r['dark_rel']))

def jac_metrics(K):
    # two rows = distinct |Delta|; columns = two actual squeezed modes
    Kn=K/np.linalg.norm(K,axis=1)[:,None]
    s=np.linalg.svd(Kn,compute_uv=False)
    return dict(singular_values=[float(t) for t in s],
                condition=float(s[0]/s[1]),
                sin_column_angle=float(abs(np.linalg.det(Kn))/(np.linalg.norm(Kn[:,0])*np.linalg.norm(Kn[:,1]))),
                determinant_abs=float(abs(np.linalg.det(K))),determinant_normalized_abs=float(abs(np.linalg.det(Kn))))

def run():
    d,cases=get_data()
    out={'identity':'Q-MATH-16.28 two occupied even/odd squeezed wavepackets screened against the actual Q16.27 RETARDED CLASSICAL chi2 kernel; NOT yet a quantum HHG S-matrix',
         'band':[.10,.26],'Omega':.36,'center':CENTER,'sigma':SIG,
         'mode_definition':'u_even=N exp[-(omega-.18)^2/(2 sigma^2)], u_odd=N_odd*(omega-.18)u_even; normalize by relative-frequency integral',
         'energy_note':'Both modes have mean photon frequency .18 by parity; E=n1+n2 fixed number, physical energy .18 E in hbar=1 units.',
         'dataset':[],'readout':[],
         'classical_P_witness':{'definition':'Any positive GS P input with total modal mean number E has anomalous matrix M with nuclear norm <= E. For 2 independent pure squeezed modes with n1=n2=.1, ||M||_* = 2 sqrt(.1*1.1) > .2.',
            'per_mode_n':[.1,.1], 'm_BSV':math.sqrt(.11),
            'total_n':.2,'total_physical_energy':.036,
            'nuclear_norm':2*math.sqrt(.11),
            'excess_factor':math.sqrt(.11)/.1,
            'IMPORTANT':'A classical chi2 proxy G is insufficient to assert physical output nonclassicality; invertible G necessary for this simple witness, not sufficient for actual HHG certification.'}}
    for eta,L in cases:
        rows=[]
        for D in (.4,.8):
            z=case_metrics(d,eta,L,D,.01)
            out['dataset'].append(z); rows.append([complex(*z['ke']),complex(*z['ko'])])
        J=jac_metrics(np.array(rows))
        out['readout'].append(dict(L=L,eta=eta,**J))
    # mechanical numerical checks
    out['checks']={
        'max_orthogonality_err':max(v['u_overlap'] for v in out['dataset']),
        'max_cross_abs':max(v['cross_abs'] for v in out['dataset']),
        'max_flat_dark_abs':max(v['flat_dark'] for v in out['dataset']),
        'eta08_201_delta04_existing_q16_27_agrees': abs(complex(*d[kkey(201,.4,.08,.18)])-complex(-.6076349585117687,-.00485239189067291))<1e-9}
    RES.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('===FINISHED===',flush=True)
    for t in out['dataset']:print('L',t['L'],'eta',t['eta'],'Delta',t['Delta'],'dark',f"{t['dark_rel']:.6f}",'ratio',t['ratio'],'freq err',t['freq_coarse_minus_fine'],flush=True)
    print('RANK:',out['readout'],flush=True)
    print('CHECKS',out['checks'],flush=True)

if __name__=='__main__':run()
