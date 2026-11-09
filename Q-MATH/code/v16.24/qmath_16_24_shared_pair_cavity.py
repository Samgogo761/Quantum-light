#!/usr/bin/env python3
"""EXPLORATORY: one shared quantized Fock mode + two M0-derived 2-state electrons.
Electronic 36d -> 2d per momentum projection is deliberately incomplete; the model
violates static Ward unless high bands generate an appropriate effective counterterm.
Not a converged QED material or a BSV-scattering simulation.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import sys,json,numpy as np
from scipy.linalg import eigh
from pathlib import Path
ROOT=Path(__file__).parent
sys.path[:0]=[str(ROOT/'source'/'deps'),str(ROOT/'source'/'deps'/'q19')]
from qmath_16_22_projected_pump import prepare_laguerre,projected_D
import qmath_16_19_quantum_pump as q19
q=q19.q

def projection(frac):
    k,bonds,onsite,*_=q19.prepared(frac,1,.1,.8,0)
    v=q.vertices(k,1)
    E,U=np.linalg.eigh(v['0']); g=U[:,0]
    J=U.conj().T@v['p']@U
    bright=int(1+np.argmax([2*abs(J[i,0])**2/max(E[i]-E[0],1e-12) for i in range(1,36)]))
    V=U[:,[0,bright]]
    ob=V.conj().T@onsite@V
    bs=[(phase,ep,V.conj().T@B@V) for phase,ep,hs,B in bonds]
    return {'onsite':ob,'bonds':bs,'Eg':float(E[0]),'gap':float(E[bright]-E[0]),
            'bright':bright,'onebody_projector_error':float(np.linalg.norm(V.conj().T@V-np.eye(2)))}

def make(nmax=12,gp=.015,omega=.18,frac=(.4158,.1683)):
    f1=projection(frac);f2=projection(tuple(-np.array(frac)))
    m=nmax+1;I2=np.eye(2);I4=np.eye(4)
    n=np.diag(np.arange(m,dtype=float))
    H=omega*np.kron(n,I4).astype(complex)
    preg=prepare_laguerre(nmax)
    for z,partner in ((f1,1),(f2,2)):
        elec=np.kron(z['onsite'],I2) if partner==1 else np.kron(I2,z['onsite'])
        H+=np.kron(np.eye(m),elec)
        for phase,ep,B in z['bonds']:
            D=projected_D(gp*ep,preg)*np.exp(1j*phase)
            O=np.kron(B,I2) if partner==1 else np.kron(I2,B)
            Z=np.kron(D,O)
            H+=Z+Z.conj().T
    H=(H+H.conj().T)/2
    E,V=eigh(H,overwrite_a=True,check_finite=False,subset_by_index=[0,1])
    psi=V[:,0].reshape(m,4)
    probs=np.sum(abs(psi)**2,axis=1)
    a=np.diag(np.sqrt(np.arange(1,m)),1)
    rho_p=psi@psi.conj().T
    return {'nmax':nmax,'gp':gp,'gap_bright':[f1['gap'],f2['gap']],
            'H_dimension':m*4,'E0':float(E[0]),'exc_gap':float(E[1]-E[0]),
            'mean_bare_photon_number':float(np.dot(np.arange(m),probs)),
            'X_mean':float(np.trace(rho_p@(a+a.T)).real),
            'top_fock_probability':float(probs[-1]),'photon_tail_last3':float(sum(probs[-3:]))}

def main():
    out={'scope':'Reduced 2x2 electronic states for each of two k sectors, sharing exactly ONE quantum photon mode; pre-asymptotic cavity toy truncated at fixed A=0 basis; NOT gauge-preserving downfolding, NOT full 36x36 dynamics, NOT BSV input/output.',
         'cases':[make(nmax=N,gp=g) for g in (0.,.005,.015,.025) for N in (8,12,16,24)]}
    path=ROOT/'qmath_16_24_shared_pair_toy_results.json';path.write_text(json.dumps(out,indent=2))
    for c in out['cases']:
        print('g=',c['gp'],'N=',c['nmax'],'E=',round(c['E0'],5),'ph=',round(c['mean_bare_photon_number'],5),'top=',format(c['top_fock_probability'],'.3e'),'gap=',round(c['exc_gap'],5))
if __name__=='__main__':main()
