#!/usr/bin/env python3
"""Independent nonperturbative exp(i A.d) photon Hamiltonian;
extract weak-coupling three-photon T from the truncated LS series.
Two-level per mode is sufficient to test the cubic coefficient but NOT higher orders.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np, json
from scipy.linalg import expm
from pathlib import Path
import q167_m0_three_photon_T as m
ROOT=Path(__file__).resolve().parent

def run():
  kf=(.4158,.1683);N=1;gamm=.10;w1=.16;w2=.20;k=m.geo.k_cart(kf)
  es,U=np.linalg.eigh(m.vertices(k,N)['0']);Eg=es[0]
  I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex)
  occup=[(a,b,c) for a in range(2) for b in range(2) for c in range(2)]
  free_ph=np.array([a*w1+b*w2+c*(w1+w2) for a,b,c in occup]); all_energies=np.repeat(es,8)+np.tile(free_ph,36)
  resolvent=1/(Eg+w1+w2-all_energies+1j*gamm)
  ket=np.zeros(288,complex);ket[occup.index((1,1,0))]=1
  bra=np.zeros(288,complex);bra[occup.index((0,0,1))]=1
  H0=np.kron(np.diag(es),np.eye(8))
  def H(t):
    total=np.kron(U.conj().T@m.dgamma(np.diag([m.m.R0.delta*N,-m.m.R0.delta*N,m.m.R0.delta*N,-m.m.R0.delta*N,-m.m.R0.delta*N,m.m.R0.delta*N,-m.m.R0.delta*N,m.m.R0.delta*N]))@U,np.eye(8)).astype(complex)
    for d,B,kind in m.BONDS:
      pval=np.dot(d,m.p);hval=np.dot(d,m.h)
      photon=np.kron(np.kron(expm(1j*t*pval*X),expm(1j*t*pval*X)),expm(1j*t*hval*X))
      E=U.conj().T@m.dgamma(B*np.exp(1j*k@d))@U
      total+=np.kron(E,photon)+np.kron(E.conj().T,photon.conj().T)
    return total
  vals={};ref=m.calc(kf,N,w1,w2,gamm)
  for t in (.00025,.0005,.001,.002,.004):
    v=H(t)-H0
    a=v@ket
    term1=np.vdot(bra,a)
    term2=np.vdot(bra,v@(resolvent*a))
    term3=np.vdot(bra,v@(resolvent*(v@(resolvent*a))))
    result=(term1+term2+term3)/t**3
    vals[str(t)]={'extracted_total':[result.real,result.imag], 'diff':float(abs(result-ref['total'])),
                  'direct_diff':float(abs(term1/t**3-ref['direct'])), 'two_diff':float(abs(term2/t**3-ref['mixed'])),
                  'three_diff':float(abs(term3/t**3-ref['sequential']))}
  out={'reference':[ref['total'].real,ref['total'].imag],'full_peierls_LS_finite_t':vals,
       'note':'Truncated 2-level photonic oscillators recover ONLY t^3 coefficient. Higher-t terms not physical full Fock.'}
  (ROOT/'q167_full_peierls_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=='__main__':run()
