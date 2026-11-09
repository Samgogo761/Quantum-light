"""A posteriori Duhamel leakage probe on the single-k M0 two-column |0>,|2> trajectory.
This is a finite external Fock tail approximation; the omitted farther tail is NOT certified by this script.
"""
import os,sys,json
for e in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[e]='1'
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import expm_multiply
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'QMATH13_M0_H2H4'))
from q13_M0_BSV_weak_radiation import M0Source
sys.path.insert(0,str(P/'QMATH13_M0_H2H4'/'deps'))
import qmath07_projected_shared as q
from scipy import sparse as sp

def probe(npump=13,steps=48,extra=20):
 s=M0Source(npump,1,nk=1)
 psi=np.stack([np.kron(s.initial_e,np.eye(npump)[:,j]) for j in (0,2)],axis=1)
 T=4*np.pi/s.omega;dt=T/steps; residual_integrals=np.zeros(2);peaks=np.zeros(2)
 for it in range(steps):
  tm=(it+.5)*dt;gate=np.sin(np.pi*tm/T)**2
  H,_,_=s.assemble(gate)
  psi=expm_multiply(-.5j*dt*H,psi)
  arr=psi.reshape(s.nE,npump,2);leak=np.zeros((s.nE,extra,2),complex)
  for delta,O in s.terms:
   d=np.array(delta[:2]);eta=gate*s.azp*(s.ep@d)
   Db=q.exact_disp(npump+extra,eta)[npump:,:npump]
   Da=q.exact_disp(npump+extra,-eta)[npump:,:npump]
   leak+=np.einsum('ei,ipm,op->eom',O.toarray(),arr,Db,optimize=True)
   leak+=np.einsum('ei,ipm,op->eom',O.getH().toarray(),arr,Da,optimize=True)
  r=np.linalg.norm(leak.reshape(-1,2),axis=0)
  residual_integrals+=dt*r;peaks=np.maximum(peaks,r)
  psi=expm_multiply(-.5j*dt*H,psi)
 return {'npump':npump,'steps':steps,'extra_out_photons':extra,'residual_integral_truncated_out_tail':residual_integrals.tolist(),
         'max_instantaneous_out_leak':peaks.tolist(),'unitarity':float(np.max(abs(psi.conj().T@psi-np.eye(2)))),
         'warning':'This residual probe omits photon levels >=Np+extra; not a rigorous infinite-tail certificate yet.'}
if __name__=='__main__':
 o=probe()
 (P/'q15_duhamel_probe.json').write_text(json.dumps(o,indent=2))
 print(json.dumps(o,indent=2))
