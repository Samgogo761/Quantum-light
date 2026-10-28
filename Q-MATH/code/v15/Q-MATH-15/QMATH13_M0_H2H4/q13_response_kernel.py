#!/usr/bin/env python3
"""Source-response kernel for actual production M0: K_beta[mu,q] on pump Fock space.
No added classical drive. Weak-output first order, full electron-pump nonperturbative.
Runs Fock basis in parallel -> all BSV phase, dephasing, and coherent-mixture contrasts.
"""
import os
for x in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[x]='1'
import time,sys,json,argparse
from pathlib import Path
import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import expm_multiply
from scipy.special import gammaln
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from q13_M0_BSV_weak_radiation import M0Source,bsv_state

def coherent(n,alpha):
    nn=np.arange(n)
    coeff=np.exp(-abs(alpha)**2/2) * alpha**nn / np.exp(.5*gammaln(nn+1))
    tail=float(1-norm(coeff)**2)
    return coeff/norm(coeff),tail

def get_kernels(npump=17,nsteps=96,neel=1,ncycles=2):
    t=time.perf_counter();s=M0Source(npump,neel)
    S=np.stack([np.kron(s.initial_e,np.eye(npump,dtype=complex)[:,j]) for j in range(npump)],axis=1)
    T=ncycles*2*np.pi/s.omega;dt=T/nsteps
    kernels=np.zeros((2,2,npump,npump),complex) # pol,q,init_i,init_j
    maxunit=0
    for it in range(nsteps):
        tm=(it+.5)*dt;gate=np.sin(np.pi*tm/T)**2
        H,Jp,Jt=s.assemble(gate)
        S=expm_multiply((-0.5j*dt)*H,S)
        for pol,J in enumerate((Jp,Jt)):
            M=S.conj().T@(J@S)
            for qi,q in enumerate((2,4)):
                kernels[pol,qi]+=-1j*dt*np.exp(1j*q*s.omega*tm)*M
        S=expm_multiply((-0.5j*dt)*H,S)
        if (it+1)%max(1,nsteps//4)==0:
            maxunit=max(maxunit,float(np.max(np.abs(S.conj().T@S-np.eye(npump)))))
            print('n',npump,'steps',it+1,'/',nsteps,'N',neel,'sec',round(time.perf_counter()-t,1),'unit',maxunit,flush=True)
    return kernels,{'time':time.perf_counter()-t,'basis_unitarity_error':maxunit,'neel':neel,'npump':npump,'steps':nsteps}

def report(kern,meta,mean=.7):
    n=meta['npump'];theta=np.arange(16)*2*np.pi/16
    rows=[]
    for phase in theta:
        state,mean_actual,tail=bsv_state(n,mean,phase)
        vals=np.einsum('a,pqab,b->pq',state.conj(),kern,state,optimize=True)
        rows.append({'phase':float(phase),'mean_actual':mean_actual,'tail':tail,'beta':[[[float(z.real),float(z.imag)] for z in p] for p in vals]})
    state,_,_=bsv_state(n,mean,0)
    weight=np.abs(state)**2
    dephased=np.einsum('n,pqnn->pq',weight,kern)
    vac=kern[:,:,0,0]
    mixes=[]
    for phase in (0.,np.pi/2):
        c,_=coherent(n,np.sqrt(mean)*np.exp(1j*phase));d,_=coherent(n,-np.sqrt(mean)*np.exp(1j*phase))
        v=(np.einsum('a,pqab,b->pq',c.conj(),kern,c)+np.einsum('a,pqab,b->pq',d.conj(),kern,d))/2
        mixes.append({'phase':float(phase),'beta':[[[float(z.real),float(z.imag)] for z in row] for row in v]})
    FT=np.fft.fft(np.array([[[complex(*z) for z in p] for p in d['beta']] for d in rows]),axis=0)/len(rows)
    # FT[d] is exp(+id theta) coefficient.
    harmonics=[]
    for p in range(2):
        for qi in range(2):
            ar=FT[:,p,qi]
            harmonics.append({'pol':['parallel','perp'][p],'order':[2,4][qi],
              'coefficients':[[float(z.real),float(z.imag)] for z in ar],
              'dominant':[{'d':int(d if d<=8 else d-16),'absolute':float(abs(ar[d]))} for d in np.argsort(abs(ar))[::-1][:5]]})
    b=np.array([[[complex(*z) for z in p] for p in d['beta']] for d in rows]);phase_loop=np.angle(b[:,0,1]/b[:,0,0]**2)
    return {'meta':meta,'bsvoid_phase_scan':rows,'controls':{'vacuum':[[[float(z.real),float(z.imag)] for z in p] for p in vac],
        'dephased_BSV':[[[float(z.real),float(z.imag)] for z in p] for p in dephased],
        'zero_mean_classical_mixture':mixes},'harmonics':harmonics,'phase_loop_parallel':phase_loop.tolist(),
        'phase_loop_range':float(max(phase_loop)-min(phase_loop)),
        'phase_coherent_contrast_at_zero':{'beta2':float(abs(b[0,0,0]-dephased[0,0])), 'beta4':float(abs(b[0,0,1]-dephased[0,1]))}}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--npump',type=int,default=17);ap.add_argument('--steps',type=int,default=96);ap.add_argument('--neel',type=int,default=1);ap.add_argument('--out',default='kernel.json');a=ap.parse_args()
    k,meta=get_kernels(a.npump,a.steps,a.neel)
    r=report(k,meta)
    Path(a.out).write_text(json.dumps(r,indent=2,ensure_ascii=False))
    np.savez_compressed(str(Path(a.out).with_suffix('.npz')),kernel=k)
    print('SUCCESS','n',a.npump,'steps',a.steps,'N',a.neel,'zeroBSV_beta2_H2',r['bsvoid_phase_scan'][0]['beta'][0][0],
    'zeroBSV_beta4_H4',r['bsvoid_phase_scan'][0]['beta'][0][1],
    'dephasedH2',r['controls']['dephased_BSV'][0][0],
    'coherence contrast',r['phase_coherent_contrast_at_zero'],flush=True)
