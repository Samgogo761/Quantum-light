#!/usr/bin/env python3
"""Exact finite-dimensional symmetry counterexample: b coherence forbidden while n_b > 0.
This is a TWO-LEVEL TOY, not M0. Conserves exact total energy and combined electronic-photon parity.
"""
import json, math, numpy as np
from scipy.linalg import expm
from pathlib import Path
B=Path(__file__).resolve().parent

def arr(n):
 a=np.zeros((n,n),complex)
 for m in range(1,n):a[m-1,m]=np.sqrt(m)
 return a

def toy(E=.7,g=.04,t=.1,npump=34):
 w=.18;Omega=.27; gap=2*w-Omega
 a=arr(npump);b=arr(2)
 id2=np.eye(2)
 h0=np.diag([0,gap])
 ng=np.diag(np.arange(npump));nh=np.diag(np.arange(2))
 sigma_plus=np.array([[0,0],[1,0]],complex)
 sigma_minus=sigma_plus.conj().T
 H=(np.kron(np.kron(h0,np.eye(npump)),id2)
    +np.kron(np.kron(id2,w*ng),id2)
    +np.kron(np.kron(id2,np.eye(npump)),Omega*nh))
 H=H.astype(complex)
 H+=g*(np.kron(np.kron(sigma_plus,a@a),b.conj().T)
       +np.kron(np.kron(sigma_minus,a.conj().T@a.conj().T),b))
 sigmaz=np.diag([1,-1]);parpump=np.diag([(-1)**j for j in range(npump)]);parH=np.diag([1,-1])
 Q=np.kron(np.kron(sigmaz,parpump),parH)
 symmetry=float(np.linalg.norm(Q@H@Q-H))
 lam=math.sqrt(E/(E+1))
 psi_p=np.zeros(npump,complex)
 for j in range(npump//2):
  n=2*j
  psi_p[n]=((1-lam**2)**.25)*np.exp(.5*math.lgamma(n+1)-j*math.log(2)-math.lgamma(j+1))*(-lam)**j
 psi_p=psi_p/np.linalg.norm(psi_p)
 psi0=np.kron(np.kron(np.array([1,0]),psi_p),np.array([1,0]))
 psit=expm(-1j*H*t)@psi0
 opb=np.kron(np.kron(id2,np.eye(npump)),b)
 opn=opb.conj().T@opb
 beta=np.vdot(psit,opb@psit)
 nb=np.vdot(psit,opn@psit).real
 factor2=float(np.sum(np.arange(npump)*(np.arange(npump)-1)*abs(psi_p)**2))
 return dict(E=E,g=g,t=t,deltaE=gap,gap_kinematics=2*w-Omega,photon_F2=factor2,beta_abs=float(abs(beta)),n_b=float(nb),leading_n_b=float((g*t)**2*factor2),ratio=float(nb/((g*t)**2*factor2)),symmetry_norm=symmetry,psi_norm=float(np.linalg.norm(psit)),input_p_even_norm=float(np.linalg.norm(parpump@psi_p-psi_p)))
if __name__=='__main__':
 out={'scope':'toy electron qubit + pump BSV + one H output mode, energy conserving, NOT material M0','cases':[toy(e) for e in (.1,.7,1.5)]}
 (B/'qmath_16_15_parity_count_toy.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 for c in out['cases']:print(c)
