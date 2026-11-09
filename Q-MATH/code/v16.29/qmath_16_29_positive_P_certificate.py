#!/usr/bin/env python3
"""Construct EXPLICIT finite nonnegative Glauber-Sudarshan P ensemble matching
BSV's two OUTPUT PROXY readouts at same M0 state. Not quantum HHG emission.
"""
from pathlib import Path
import json, numpy as np
ROOT=Path(__file__).resolve().parent
res=json.loads((ROOT/'qmath_16_29_results.json').read_text())
W=np.array(res['input_energy_W'],float)
lam,Ue=np.linalg.eigh(W)
Wh=(Ue*np.sqrt(lam))@Ue.T
Wi=(Ue*(1/np.sqrt(lam)))@Ue.T
U=float(res['U'])
def toz(z):return complex(*z)
def mmat(m):return np.array([[m[0],m[2]],[m[2],m[1]]],complex)
def explicit_ensemble(K, m_fit):
    M=mmat(np.array([toz(z) for z in m_fit]));Q=Wh@M@Wh
    Us,s,Vh=np.linalg.svd(Q)
    terms=[];recQ=np.zeros((2,2),complex)
    for j in range(2):
        if s[j]<1e-14:continue
        u=Us[:,j]
        phase=np.angle(np.vdot(u,Q@u.conj()))
        v=u*np.exp(0.5j*phase)
        recQ+=s[j]*np.outer(v,v)
        terms.append((float(s[j]),v))
    if np.linalg.norm(recQ-Q)>1e-9:
        raise ValueError(f'Takagi factorization failed {np.linalg.norm(recQ-Q)}')
    cost=sum(x[0] for x in terms)
    assert cost<U
    # P over +/- coherent amplitudes, with Takagi probabilities p_j=s_j/cost.
    P=[]
    for sj,v in terms:
        alpha=Wi@v*np.sqrt(cost)
        for sign in (+1,-1):P.append((sj/(2*cost),sign*alpha))
    # Add independent four-point circular random coherent phasor with M=0 and
    # mean extra energy U-cost (can be interpreted as classical mixed-source noise).
    evec=Ue[:,np.argmin(lam)]
    beta=evec*np.sqrt((U-cost)/lam.min())
    Z=[beta,-beta,1j*beta,-1j*beta]
    combined=[(p/4, alpha+z) for p,alpha in P for z in Z]
    probs=np.array([p for p,a in combined]);amps=np.array([a for p,a in combined])
    mout=np.einsum('a,ai,aj->ij',probs,amps,amps)
    Nout=np.einsum('a,ai,aj->ij',probs,amps,np.conj(amps))
    Eout=float(np.trace(W@Nout).real)
    mout_v=np.array([mout[0,0],mout[1,1],mout[0,1]])
    mom=np.einsum('a,ai->i',probs,amps)
    return {'cost_before_noise':cost,'resource_U':U,'classical_mix_points':len(combined),
          'probability_min':float(probs.min()),'probability_sum':float(probs.sum()),
          'mean_field_error':float(np.linalg.norm(mom)),
          'mean_energy_error':float(Eout-U),
          'anomalous_matrix_error':float(np.linalg.norm(mout-M)),
          'output_error':float(np.linalg.norm(K@mout_v-K@np.array([toz(z) for z in res['input_m']]))/np.linalg.norm(K@np.array([toz(z) for z in res['input_m']]))),
          'amplitudes':[{'p':float(p),'alpha':[[float(v.real),float(v.imag)] for v in a]} for p,a in combined]}

out={'scope':'Exact finite discrete GS-positive-P classical coherent-state ensemble, input canonical 2 modes, matching two M0 chi2 PROXY outputs, not physical emitted HHG. Equal true free photon energy and vanishing mean input field.','cases':[]}
for c in res['cases']:
    K=np.array([[toz(z) for z in row] for row in c['analysis']['K']])[[0,2]]
    x=explicit_ensemble(K,c['analysis']['pair_34_38']['m_fit'])
    out['cases'].append({'L':c['L'],'eta':c['eta'],'Delta':c['Delta'],**x})
    print(c['L'],c['eta'],c['Delta'],x['cost_before_noise']/U,'points',x['classical_mix_points'],'error',x['output_error'])
(ROOT/'qmath_16_29_explicit_classical_P.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
