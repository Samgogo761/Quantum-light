import os
for e in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[e]='1'
import numpy as np,sys,json
from scipy.sparse.linalg import expm_multiply
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'QMATH13_M0_H2H4'))
from q13_M0_BSV_weak_radiation import M0Source
s=M0Source(5,1,nk=2,pump_coupling=0)
psi=np.stack([np.kron(s.initial_e,np.eye(5)[:,j]) for j in (0,2)],axis=1)
T=4*np.pi/s.omega;dt=T/16;z=0j
for it in range(16):
 tm=(it+.5)*dt;H,Jp,Jt=s.assemble(np.sin(np.pi*tm/T)**2)
 psi=expm_multiply(-.5j*dt*H,psi)
 z+=-1j*dt*np.exp(2j*s.omega*tm)*(psi.conj().T@(Jt@psi))[0,1]
 psi=expm_multiply(-.5j*dt*H,psi)
print(json.dumps({'zero_coupling_K02':[z.real,z.imag],'abs':abs(z),'unitary_error':float(np.max(abs(psi.conj().T@psi-np.eye(2))))},indent=2))
