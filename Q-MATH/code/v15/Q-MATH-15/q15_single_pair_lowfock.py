import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[v]='1'
import numpy as np,time,sys,json
from scipy.sparse.linalg import expm_multiply
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'QMATH13_M0_H2H4'))
from q13_M0_BSV_weak_radiation import M0Source
from pathlib import Path
out=[]
for kfrac in ((.4158,.1683),(-.4158,-.1683)):
 for npump in (5,7):
    s=M0Source(npump,1,nk=1,kfrac=kfrac)
    psi=np.stack([np.kron(s.initial_e,np.eye(npump)[:,j]) for j in (0,2)],axis=1)
    T=4*np.pi/s.omega;dt=T/32;K=0j
    for it in range(32):
        tm=(it+.5)*dt;gate=np.sin(np.pi*tm/T)**2
        H,Jp,Jt=s.assemble(gate)
        psi=expm_multiply(-.5j*dt*H,psi)
        M=psi.conj().T@(Jt@psi)
        K+=-1j*dt*np.exp(2j*s.omega*tm)*M[0,1]
        psi=expm_multiply(-.5j*dt*H,psi)
    z={'kfrac':kfrac,'npump':npump,'K02_H2_perp':[K.real,K.imag],'norm':abs(K),'error':float(np.max(abs(psi.conj().T@psi-np.eye(2))))}
    print(z,flush=True);out.append(z)
Path(__file__).resolve().parent/'single_pair_32.json'.write_text(json.dumps(out,indent=2))
