#!/usr/bin/env python3
"""Q-MATH-07: shared two-k, 4 spin sectors (Lambda^2 C^4 each), common 3-mode photon Fock space.
True projected Peierls exp(i eta(a+a†)) via exact Laguerre matrix elements.
The finite, Hermitian Hamiltonian is evolved by matrix-free Krylov expm_multiply.
No cavity-output or continuum radiation claim.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from itertools import combinations
from functools import lru_cache
from pathlib import Path
import time, json, warnings, argparse
import numpy as np
from scipy import sparse as sp
from scipy import linalg as la
from scipy.special import eval_genlaguerre, gammaln
from scipy.sparse.linalg import LinearOperator, expm_multiply
import q8_m0_model as model
from spin_exterior_solver import spin_initial,SPINS,eval_from_states

BASE=Path(__file__).resolve().parent
PAIR=list(combinations(range(4),2)); NV=6; EI=sp.eye(6,dtype=np.complex128,format='csr')

def op_create_destroy(i,j):
    R=np.zeros((6,6),complex)
    for t,pair in enumerate(PAIR):
        occ=list(pair)
        if j not in occ:continue
        jpos=occ.index(j)
        after=occ[:];after.pop(jpos)
        if i in after:continue
        n_below=sum(x<i for x in after)
        val=(-1)**(jpos+n_below)
        after.append(i);after.sort();R[PAIR.index(tuple(after)),t]=val
    return sp.csr_matrix(R)
OPS={(i,j):op_create_destroy(i,j) for i in range(4) for j in range(4)}

def lift_factor(small,ind,nfactors=4):
    ret=sp.csr_matrix([[1.0+0j]])
    for i in range(nfactors):ret=sp.kron(ret,small if i==ind else EI,format='csr')
    return ret

def original_wedge(h):
    v=sum((h[i,j]*OPS[i,j] for i in range(4) for j in range(4)),sp.csr_matrix((6,6),dtype=complex))
    return v.toarray()

def exact_disp(n,eta):
    idx=np.arange(n);M,N=np.meshgrid(idx,idx,indexing='ij')
    low=np.minimum(M,N);di=np.abs(M-N)
    return np.exp(-.5*eta*eta+.5*(gammaln(low+1)-gammaln(low+di+1)))*(1j*eta)**di*eval_genlaguerre(low,di,eta*eta)

def disp_alpha(n,alpha):
    ii=np.arange(n); M,N=np.meshgrid(ii,ii,indexing='ij')
    low=np.minimum(M,N);di=np.abs(M-N)
    z=abs(alpha)**2
    powers=np.where(M>=N,alpha,-np.conj(alpha))**di
    return np.exp(-z/2+.5*(gammaln(low+1)-gammaln(low+di+1)))*powers*eval_genlaguerre(low,di,z)

def analytic_pump(npump,kind,phase=0.,mean=.7,nsq=.2,nref=92):
    full=np.zeros(nref,dtype=complex)
    if kind=='vacuum':full[0]=1
    elif kind=='coherent':
        alpha=np.sqrt(mean)*np.exp(1j*phase)
        full=np.exp(-.5*mean)*alpha**np.arange(nref)/np.exp(.5*gammaln(np.arange(nref)+1))
    elif kind in ('bsv','seeded'):
        sq=mean if kind=='bsv' else nsq
        r=np.arcsinh(np.sqrt(sq));t=np.tanh(r)
        j=np.arange((nref+1)//2);ev=2*j
        full[ev]=(-t)**j * np.exp(.5*gammaln(2*j+1)-gammaln(j+1)-j*np.log(2))/np.sqrt(np.cosh(r))
        if kind=='seeded':
            alpha=np.sqrt(mean-nsq)*np.exp(1j*phase)
            full=disp_alpha(nref,alpha)@full
    else:raise ValueError(kind)
    assert abs(la.norm(full)-1)<5e-9,(kind,la.norm(full))
    tail=float(1-np.sum(abs(full[:npump])**2))
    v=full[:npump]/la.norm(full[:npump]); mean_actual=float(np.dot(np.arange(npump),abs(v)**2))
    return v,mean_actual,tail,float(abs(v[-1])**2)

@lru_cache(None)
def electron_terms(kfract=(.4158,.1683), neel=1, nk=2):
    geo=model.build_geometry(BASE/'CrI3_scf_soc_a1h_clean.in');p=model.R0
    k=geo.k_cart(kfract)
    allk=(k,-k) if nk==2 else (k,)
    key=lambda delta:tuple(np.asarray(delta)[:2].round(12))
    dim=36**nk; nf=2*nk
    ops={}; offs=sp.csr_matrix((dim,dim),dtype=complex)
    for ik,kk in enumerate(allk):
        for spin in (0,1):
            fac=2*ik+spin;sg=1 if spin==0 else -1
            spin_idxs=SPINS[spin]
            oneh=model.h8(geo,p,kk,neel)[np.ix_(spin_idxs,spin_idxs)]
            # used for independent zero-field reconstruction
            local_onsite=np.zeros((6,6),complex)
            for layer in range(2):
                for sub in range(2):
                    idx=2*layer+sub
                    local_onsite+=neel*(1 if layer==0 else -1)*p.delta*sg*OPS[idx,idx].toarray()
            offs+=lift_factor(sp.csr_matrix(local_onsite),fac,nf)
            def collect(delta,loc_mat,amp):
                kk0=key(delta)
                ops[kk0]=ops.get(kk0,sp.csr_matrix((dim,dim),dtype=complex))+lift_factor(loc_mat,fac,nf)*amp*np.exp(1j*np.dot(kk,np.asarray(delta)[:2]))
            for d in geo.nn_deltas_cart:
                L=p.t_h*(OPS[0,1]+OPS[2,3])
                collect(d,L,1.0)
            for nu,b in geo.nnn_terms:
                L=sp.csr_matrix((6,6),dtype=complex)
                for layer in range(2):
                    L+=OPS[2*layer,2*layer]-OPS[2*layer+1,2*layer+1]
                collect(b,L,(-1j)*nu*p.lam_so*sg)
            inter={'A':0,'B':1}
            for bond in geo.interlayer_bonds:
                a=inter[bond['from'][1]];b=2+inter[bond['to'][1]]
                collect(bond['delta_cart'],OPS[a,b],getattr(p,bond['amplitude']))
            # Check zero-field local one-particle construction by full lifted coefficients later.
    return offs,ops,np.asarray(geo.a2_cart[:2],float)

@lru_cache(None)
def audit_zero_field(nk=2):
    geo=model.build_geometry(BASE/'CrI3_scf_soc_a1h_clean.in')
    offs,terms,_=electron_terms(nk=nk)
    allh=offs+sum((A+A.getH() for A in terms.values()),sp.csr_matrix((36**nk,36**nk),dtype=complex))
    ref=sp.csr_matrix((36**nk,36**nk),dtype=complex)
    k=geo.k_cart((.4158,.1683))
    gaps=[]
    for ik,kk in enumerate((k,-k) if nk==2 else (k,)):
        H=model.h8(geo,model.R0,kk,1)
        for spin in (0,1):
            fac=2*ik+spin
            b=original_wedge(H[np.ix_(SPINS[spin],SPINS[spin])]);ref+=lift_factor(sp.csr_matrix(b),fac,2*nk)
    return float(sp.linalg.norm(allh-ref)),float(sp.linalg.norm(allh-allh.getH()))

class MatrixFree:
    def __init__(self,npump,nharm,gate,omega=.18,azp=.05,azh=.005,kfract=(.4158,.1683),neel=1,nk=2):
        self.np=npump;self.nh=nharm;self.P=npump*nharm*nharm;self.nE=36**nk;self.dim=self.nE*self.P
        self.offs,self.terms,a2=electron_terms(kfract,neel,nk)
        ep=a2/la.norm(a2);et=np.array([ep[1],-ep[0]])
        self.D=[]
        for vec,op in self.terms.items():
            delta=np.array(vec)
            x=gate*azp*np.dot(ep,delta); y=gate*azh*np.dot(ep,delta);z=gate*azh*np.dot(et,delta)
            self.D.append((op,op.getH(), (exact_disp(npump,x),exact_disp(nharm,y),exact_disp(nharm,z))))
        pp=np.arange(npump)[:,None,None]; aa=np.arange(nharm)[None,:,None];bb=np.arange(nharm)[None,None,:]
        self.nfree=omega*(pp+2*aa+2*bb).reshape(self.P)
        self.trace=float(np.real(self.offs.diagonal().sum()))*self.P+self.nE*float(self.nfree.sum())
        self.calls=0
    def ph_trans(self,psi,mats,adjoint=False):
        Dp,Da,Db=mats
        if adjoint:Dp,Da,Db=Dp.conj().T,Da.conj().T,Db.conj().T
        y=psi.reshape(-1,self.nE,self.np,self.nh,self.nh)
        y=np.einsum('pi,beijk->bepjk',Dp,y,optimize=True)
        y=np.einsum('qj,bepjk->bepqk',Da,y,optimize=True)
        y=np.einsum('rk,bepqk->bepqr',Db,y,optimize=True)
        return y.reshape(-1,self.nE,self.P)
    def matmat(self,v):
        self.calls+=1
        orig=v.shape
        st=v.T.reshape(-1,self.nE,self.P)
        result= (st.astype(complex)*self.nfree[None,None,:]).copy()
        result+=np.stack([self.offs@v0 for v0 in st],axis=0)
        for O,OH,M in self.D:
            c=self.ph_trans(st,M)
            result+=np.stack([O@v0 for v0 in c],axis=0)
            c=self.ph_trans(st,M,True)
            result+=np.stack([OH@v0 for v0 in c],axis=0)
        return result.reshape(-1,self.dim).T
    def aslinop(self,dt):
        return LinearOperator((self.dim,self.dim),matvec=lambda x:-1j*dt*self.matmat(x[:,None])[:,0],
                             matmat=lambda x:-1j*dt*self.matmat(x),
                             rmatvec=lambda x:+1j*dt*self.matmat(x[:,None])[:,0],
                             rmatmat=lambda x:+1j*dt*self.matmat(x),dtype=complex)

def initial_state(npump,nharm,kind,phase,mean=.7,nsq=.2,nk=2):
    geo=model.build_geometry(BASE/'CrI3_scf_soc_a1h_clean.in');k=geo.k_cart((.4158,.1683))
    el=np.array([1.0+0j]);
    for kk in ((k,-k) if nk==2 else (k,)):
        H=model.h8(geo,model.R0,kk,1)
        for spin in (0,1):el=np.kron(el,spin_initial(H[np.ix_(SPINS[spin],SPINS[spin])]))
    pu,nbar,tail,top=analytic_pump(npump,kind,phase,mean,nsq)
    vac=np.eye(nharm,dtype=complex)[:,0]
    phi=np.kron(el,np.kron(pu,np.kron(vac,vac)))
    return phi,dict(actual_mean=nbar,initial_tail=tail,initial_top=top)

def observable(psi,npump,nharm,nk=2):
    s=psi.reshape(36**nk,npump,nharm,nharm)
    rho=np.einsum('epij,epkl->ijkl',s,s.conj(),optimize=True).reshape(nharm*nharm,nharm*nharm)
    a=np.diag(np.sqrt(np.arange(1,nharm)),1).astype(complex)
    ap=np.kron(a,np.eye(nharm));ax=np.kron(np.eye(nharm),a)
    ev=lambda op:np.trace(rho@op)
    bp=ev(ap); bx=ev(ax)
    nnpar=ev(ap.conj().T@ap).real;nnper=ev(ax.conj().T@ax).real
    mp=ev(ap@ap)-bp**2;mx=ev(ax@ax)-bx**2
    cr=ev(ap@ax)-bp*bx
    pump_prob=np.sum(abs(s)**2,axis=(0,2,3));harm_prob=np.sum(abs(s)**2,axis=(0,1))
    Ptop_par=float(harm_prob[-1,:].sum());Ptop_per=float(harm_prob[:,-1].sum())
    return dict(n_parallel=float(nnpar),n_perp=float(nnper),beta_parallel=[float(bp.real),float(bp.imag)],
                m_cross=[float(cr.real),float(cr.imag)],
                varmin_parallel=float(.5+nnpar-abs(bp)**2-abs(mp)),
                varmin_perp=float(.5+nnper-abs(bx)**2-abs(mx)),
                pump_edge=float(pump_prob[-1]),h2par_edge=Ptop_par,h2per_edge=Ptop_per,
                pump_mean=float(np.dot(np.arange(npump),pump_prob)),norm=float(la.norm(psi)))

def simulate(npump=7,nharm=3,nsteps=16,kind='seeded',phase=np.pi/2,mean=.7,nsq=.2,omega=.18,nk=2,verbose=True):
    t0=time.time();P=npump*nharm*nharm;D=(36**nk)*P
    ps=[];infos=[]
    for phi in ([phase,phase+np.pi] if kind in ('seeded','coherent') else [phase]):
        psi,inf=initial_state(npump,nharm,kind,phi,mean,nsq,nk);ps.append(psi);infos.append(inf)
    state=np.stack(ps,axis=-1)
    total_T=4*np.pi/omega;dt=total_T/nsteps;maxnorm=0;calls=0
    for it in range(nsteps):
        gate=np.sin(np.pi*(it+.5)/nsteps)**2
        ham=MatrixFree(npump,nharm,gate,omega,nk=nk)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Trace of LinearOperator not available*')
            state=expm_multiply(ham.aslinop(dt),state,traceA=-1j*dt*ham.trace)
        calls+=ham.calls;nm=np.linalg.norm(state,axis=0);maxnorm=max(maxnorm,float(np.max(abs(nm-1))))
        if verbose and (it==0 or (it+1)%max(1,nsteps//4)==0):print('slice',it+1,'/',nsteps,'norm_err',maxnorm,'calls',calls,'sec',round(time.time()-t0,1),flush=True)
    outs=[dict(observable(state[:,j],npump,nharm,nk),**infos[j]) for j in range(state.shape[1])]
    if len(outs)==2:
        cr=np.array(outs[0]['m_cross'])-np.array(outs[1]['m_cross']);m=complex(*(cr/2))
        odds={'m_odd':(cr/2).tolist(),'abs_m_odd':float(abs(m))}
    else:odds={}
    return dict(meta=dict(npump=npump,nharm=nharm,nsteps=nsteps,kind=kind,phase=phase,mean=mean,nsq=nsq,
                          mode='true Fock projected Peierls',end_switch='sin^2',single_kfrac=(.4158,.1683),
                          k_sector_count=nk,electron_dimension=36**nk, electronically_shared_k=(nk==2),not_far_field=True),outputs=outs,odd=odds,
                max_norm_error=maxnorm,matvec_calls=calls,wall_sec=time.time()-t0)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--npump',type=int,default=7);a.add_argument('--nharm',type=int,default=3);a.add_argument('--steps',type=int,default=16);a.add_argument('--nk',type=int,default=2);a.add_argument('--kind',default='seeded');a.add_argument('--phase',type=float,default=float(np.pi/2));a.add_argument('--out',default='result.json');z=a.parse_args()
    print('zero_field',audit_zero_field(nk=z.nk),flush=True)
    result=simulate(z.npump,z.nharm,z.steps,z.kind,z.phase,nk=z.nk)
    Path(z.out).write_text(json.dumps(result,ensure_ascii=False,indent=2));print('RESULT',json.dumps(result['odd']),result['outputs'],flush=True)
