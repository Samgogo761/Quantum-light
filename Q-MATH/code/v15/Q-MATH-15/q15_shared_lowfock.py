import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[v]='1'
import numpy as np,time,sys,json
from scipy.sparse.linalg import expm_multiply
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'QMATH13_M0_H2H4'))
from q13_M0_BSV_weak_radiation import M0Source
from pathlib import Path
start=time.time()
for npump,steps in [(5,32),(7,32)]:
    s=M0Source(npump,1,nk=2)
    print('START',npump,steps,'N_e',s.nE,'dims',s.nE*npump,flush=True)
    psi=np.stack([np.kron(s.initial_e,np.eye(npump)[:,j]) for j in (0,2)],axis=1)
    T=4*np.pi/s.omega;dt=T/steps
    K=np.zeros((2,2,2),complex)
    for it in range(steps):
        tm=(it+.5)*dt;gate=np.sin(np.pi*tm/T)**2
        H,Jp,Jt=s.assemble(gate)
        psi=expm_multiply(-.5j*dt*H,psi)
        for ip,J in enumerate((Jp,Jt)):
            M=psi.conj().T@(J@psi)
            for iq,q in enumerate((2,4)):
                K[ip,iq,:]+=-1j*dt*np.exp(1j*q*s.omega*tm)*np.array([M[0,1],M[0,0]])
        psi=expm_multiply(-.5j*dt*H,psi)
    z=dict(npump=npump,steps=steps,runtime=time.time()-start,unitary_error=float(np.max(abs(psi.conj().T@psi-np.eye(2)))),
     K02_H2_parallel=[K[0,0,0].real,K[0,0,0].imag],K02_H2_perp=[K[1,0,0].real,K[1,0,0].imag],
     K02_H4_perp=[K[1,1,0].real,K[1,1,0].imag],K00_H2_perp=[K[1,0,1].real,K[1,0,1].imag])
    print('DONE',z,flush=True)
    Path(__file__).resolve().parent/('shared_2k_%d.json'%npump).write_text(json.dumps(z,indent=2))
