#!/usr/bin/env python3
"""Fully adiabatically switched two-frequency retarded classical M0 H2 current kernel.
Uses all occupied and unoccupied M0 bands, exact mixed Peierls contact terms.
NOT a quantum squeezed-light computation, and not an LSZ/Maxwell photon output.
"""
import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[v]='1'
from pathlib import Path
import json,time, numpy as np
from qmath_16_26_cubic_audit import bond_data, derivatives, mixed_via_ordered_bonds
ROOT=Path(__file__).resolve().parent

def kernel(L=41,Delta=.8,N=1,lam=.1,w1=.16,w2=.20,eta=.02,offset=.5,split=False):
    # Take offset as uniform mesh shift in fractional reciprocal unit cell.
    fr,k,base,bonds=bond_data(L,N,Delta,lam)
    if offset != .5:
        fr=np.array([((i+offset)/L,(j+offset)/L) for i in range(L) for j in range(L)])
        from qmath_16_26_cubic_audit import q
        k=np.array([q.geo.k_cart(x) for x in fr]);
        # Existing base unaffected by fractional shift; rephase each bond relative to old k.
        oldk=np.array([q.geo.k_cart(x) for x in [((i+.5)/L,(j+.5)/L) for i in range(L) for j in range(L)]])
        for bi,(_,dot,Bs) in enumerate(bonds):
            delta=q.BONDS[bi][0]
            oldphase=bonds[bi][0]
            bonds[bi]=(oldphase*np.exp(1j*((k-oldk)@delta)),dot,Bs)
    H=derivatives(base,bonds,2)
    M=mixed_via_ordered_bonds(base,bonds,2)
    ep,U=np.linalg.eigh(H[0])
    # spin block shape (nk,2,4,4), first 2 states occupied in each spin block
    def rotated(A):
        return U.conj().swapaxes(-1,-2)@A@U
    P=np.zeros_like(H[0]);P[...,0,0]=1;P[...,1,1]=1
    V=rotated(H[1]);V2=rotated(H[2]);
    J0=rotated(M[0]);J1=rotated(M[1]);J2=rotated(M[2]);
    gaps=ep[..., :,None]-ep[...,None,:]
    def comm(A,B):return A@B-B@A
    z1=w1+1j*eta;z2=w2+1j*eta
    R1=comm(V,P)/(z1-gaps)
    R2=comm(V,P)/(z2-gaps)
    R12=(comm(V,R1)+comm(V,R2)+comm(V2,P))/(z1+z2-gaps)
    term0=np.einsum('...ij,...ji->...',R12,J0)
    term1=np.einsum('...ij,...ji->...',R1+R2,J1)
    term2=np.einsum('...ij,...ji->...',P,J2)
    el=term0+term1+term2
    # sum both spin blocks at each k, mean all k.
    per_k=el.sum(axis=-1)
    z=per_k.mean()
    data={'L':L,'Delta':Delta,'N':N,'lambda':lam,'w1':w1,'w2':w2,'eta':eta,'shift':offset,
          'chi12':[float(z.real),float(z.imag)],
          'retarded_output_terms':[[float(v.real),float(v.imag)] for v in (term0.sum(axis=-1).mean(),term1.sum(axis=-1).mean(),term2.sum(axis=-1).mean())],
          'min_gap':float(np.min(ep[...,2]-ep[...,1])),
          'meaning':'Retarded two-tone vector-potential transverse current kernel: each incoming tone exp(-i w t+eta*t); normalized coefficient for product of complex source amplitudes; not quantum light'}
    if split:return data,per_k,fr
    return data

def main():
    start=time.monotonic();rec=[]
    for eta in (.08,.03,.01):
        for L in (13,21,33,49,65,81,101):
            d=kernel(L=L,eta=eta);rec.append(d)
            print('retarded',L,eta,d['chi12'],round(time.monotonic()-start,1),flush=True)
    for w1,w2 in ((.20,.16),(.18,.18)):
        rec.append(kernel(L=49,eta=.03,w1=w1,w2=w2))
    for D,lam in ((-.8,.1),(0,.1),(.8,0),(.4,.1)):
        rec.append(kernel(L=49,eta=.03,Delta=D,lam=lam))
    for sh in (.0,.25,.5,.75):
        rec.append(kernel(L=49,eta=.03,offset=sh))
    # Static BZ derivative j_h(A) via band energies, should vanish as mesh -> large.
    # Also verify exact retarded permutation for equal eta sources.
    a=next(x for x in rec if x['eta']==.03 and x['L']==49 and x['w1']==.16 and x['w2']==.20 and x['Delta']==.8 and x['lambda']==.1 and x['shift']==.5)
    b=next(x for x in rec if x['eta']==.03 and x['L']==49 and x['w1']==.20)
    neg=next(x for x in rec if x['eta']==.03 and x['L']==49 and x['Delta']==-.8)
    zeros=[x for x in rec if x['eta']==.03 and x['L']==49 and (x['Delta']==0 or x['lambda']==0)]
    def cm(x):return complex(*x['chi12'])
    checks={'permutation_abs':float(abs(cm(a)-cm(b))),
            'neel_odd_abs':float(abs(cm(a)+cm(neg))),
            'delta0_soc0_abs':[float(abs(cm(x))) for x in zeros]}
    out={'scope':'Fully adiabatically switched causal M0 classical two-frequency transverse vector-potential current kernel, complete 8-band and Peierls contacts, NOT quantized BSV or radiated photon statistics',
         'checks':checks,'data':rec,'elapsed':round(time.monotonic()-start,2)}
    (ROOT/'qmath_16_27_retarded_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('CHECKS',json.dumps(checks,indent=2),'duration',out['elapsed'])
if __name__=='__main__':main()
