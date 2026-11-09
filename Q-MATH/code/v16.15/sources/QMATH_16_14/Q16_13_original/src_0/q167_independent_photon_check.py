#!/usr/bin/env python3
"""Independently contract 3-mode 0/1 photon Hilbert spaces with electronic Peierls derivatives.
Tests combinatorial factors in third-order Lippmann-Schwinger expansion.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json
from pathlib import Path
import q167_m0_three_photon_T as m
from itertools import product
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parent

def check(kf=(.4158,.1683),N=1,w1=.16,w2=.20,gamma=.10):
    k=m.geo.k_cart(kf)
    h_el,U=np.linalg.eigh(m.vertices(k,N)['0']);Eg=h_el[0]
    one=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex)
    modeX=[np.kron(np.kron(a,b),c) for a,b,c in [(X,one,one),(one,X,one),(one,one,X)]]
    V={}
    vecs=(m.p,m.p,m.h)
    for n in (1,2,3):
        mat=np.zeros((36*8,36*8),complex)
        for idx in product(range(3),repeat=n):
            el=m.dgamma(m.exact_vertex(k,N,[vecs[i] for i in idx]))
            el=U.conj().T@el@U
            photon=np.eye(8)
            for i in idx:photon=photon@modeX[i]
            mat+=np.kron(el,photon)/np.math.factorial(n) if hasattr(np,'math') else np.kron(el,photon)/__import__('math').factorial(n)
        V[n]=mat
    occs=list(product(range(2),repeat=3))
    w=np.array([w1,w2,w1+w2]);Es=np.array([e+np.dot(o,w) for e in h_el for o in occs]); E_i=Eg+w1+w2
    D=1/(E_i-Es+1j*gamma)
    ket=np.zeros(288,complex);ket[occs.index((1,1,0))]=1
    bra=np.zeros(288,complex);bra[occs.index((0,0,1))]=1
    direct=np.vdot(bra,V[3]@ket)
    mixed=np.vdot(bra,V[2]@(D*(V[1]@ket))+V[1]@(D*(V[2]@ket)))
    sequential=np.vdot(bra,V[1]@(D*(V[1]@(D*(V[1]@ket)))))
    ref=m.calc(kf,N,w1,w2,gamma)
    ans={'direct':[direct.real,direct.imag],'mixed':[mixed.real,mixed.imag], 'sequential':[sequential.real,sequential.imag], 'total':[(direct+mixed+sequential).real,(direct+mixed+sequential).imag],
         'different_basis_error':float(abs(direct-ref['direct'])+abs(mixed-ref['mixed'])+abs(sequential-ref['sequential'])),'hermiticity_V1':float(np.linalg.norm(V[1]-V[1].conj().T)),
         'hermiticity_V2':float(np.linalg.norm(V[2]-V[2].conj().T)), 'hermiticity_V3':float(np.linalg.norm(V[3]-V[3].conj().T))}
    assert ans['different_basis_error']<1e-8,ans
    (ROOT/'q167_photon_hilbert_validation.json').write_text(json.dumps(ans,indent=2))
    print(json.dumps(ans,indent=2));return ans
if __name__=='__main__':check()
