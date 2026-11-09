#!/usr/bin/env python3
# Q-MATH-16.24 exact M0 Bloch occupied-band static sum-rule and truncation audit
# No input/output BSV or LSZ claims.
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np,sys, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'source'/'deps'/'q19'/'reference_q167'
sys.path[:0]=[str(SRC),str(SRC/'deps')]
import q167_m0_three_photon_T as q

# At half filling, there are 2 occupied and 2 unoccupied orbitals in each spin block.
def one(frac,N=1,lam=.1,Delta=.8):
    k=q.geo.k_cart(frac)
    # For generic lambda,Delta, reconstruct the same Peierls bond Hamiltonian.
    h0=np.diag(np.array([Delta*N,-Delta*N,Delta*N,-Delta*N,-Delta*N,Delta*N,-Delta*N,Delta*N],float)).astype(complex)
    hp=np.zeros((8,8),complex);hpp=np.zeros((8,8),complex)
    for dv,sp,kind in q.BONDS:
        A = sp*(lam/.1 if kind=='SOC' else 1.0)
        phase=np.exp(1j*np.dot(dv,k)); d=float(np.dot(dv,q.p))
        z=phase*A
        h0 += z+z.conj().T
        hp += 1j*d*(z-z.conj().T)
        hpp += -d*d*(z+z.conj().T)
    j=0.; D=0.; terms=[]; Esp=0.; Emin=1e100
    for spininds in [q.UP,q.DN]:
        h=h0[np.ix_(spininds,spininds)];p=hp[np.ix_(spininds,spininds)];pp=hpp[np.ix_(spininds,spininds)]
        e,U=np.linalg.eigh(h);v=U.conj().T@p@U;Dmat=U.conj().T@pp@U
        j+=float(np.sum(np.diag(v)[:2]).real);D+=float(np.sum(np.diag(Dmat)[:2]).real)
        Esp+=float(np.sum(e[:2]));Emin=min(Emin,float(e[2]-e[1]))
        for i in range(2):
            for a in range(2,4):
                gap=float(e[a]-e[i]); weight=2*float(abs(v[a,i])**2)/gap
                terms.append({'gap':gap, 'weight':weight})
    terms.sort(key=lambda x:x['gap'])
    sums=[D-sum(t['weight'] for t in terms[:n]) for n in [0,2,4,6,8]]
    # Hessian of occupied energy at fixed k (all interband virtual channels)
    return dict(j=j,D=D,paramagnetic=sum(t['weight'] for t in terms),K=sums[-1],
                K_ret=sums,weight_by_gap=[t['weight'] for t in terms],
                min_direct_gap=Emin,occupied_energy=Esp)

def pair(frac,N=1):
    a=one(frac,N);b=one(tuple(-np.array(frac)),N)
    return dict(frac=list(frac),N=N,
       Jpair=a['j']+b['j'],Kpair=a['K']+b['K'],
       Jindividual=[a['j'],b['j']],Kindividual=[a['K'],b['K']],
       Dpair=a['D']+b['D'],Ppair=a['paramagnetic']+b['paramagnetic'],
       Kpair_ret=[float(x+y) for x,y in zip(a['K_ret'],b['K_ret'])],
       gap_min=min(a['min_direct_gap'],b['min_direct_gap']))

def grid(L,N=1):
    js=[];ks=[];ds=[];ps=[];kr=[];ends=[]
    for i in range(L):
        for j in range(L):
            r=one(((i+.5)/L,(j+.5)/L),N)
            js.append(r['j']);ks.append(r['K']);ds.append(r['D']);ps.append(r['paramagnetic']);kr.append(r['K_ret']);ends.append(r['min_direct_gap'])
    ret=np.mean(np.array(kr),axis=0)
    return dict(L=L,N=N,meanJ=float(np.mean(js)),meanAbsJ=float(np.mean(np.abs(js))),
        meanD=float(np.mean(ds)),meanParam=float(np.mean(ps)),
        staticK_full=float(np.mean(ks)),
        staticK_keep_0_2_4_6_8=[float(x) for x in ret],
        minimum_direct_gap=float(np.min(ends)))

def fd_test(frac=(.4158,.1683),N=1,h=1e-4):
    def occ(k):
        kk=q.geo.k_cart(k)
        s=0.0
        for ix in [q.UP,q.DN]:
            vals=np.linalg.eigvalsh(q.exact_vertex(kk,N,[])[np.ix_(ix,ix)])
            s+=float(vals[:2].sum())
        return s
    # +/- a Cartesian momentum displacement is +/-h*p => fractional transformation
    k0=q.geo.k_cart(frac)
    x0=q.geo.k_frac(k0)
    xp=q.geo.k_frac(k0+h*q.p)
    xm=q.geo.k_frac(k0-h*q.p)
    fp,f0,fm=occ(xp),occ(x0),occ(xm)
    analytic=one(frac,N)
    return dict(frac=list(frac),N=N,h=h,
       E_prime_fd=float((fp-fm)/(2*h)),E_prime_analytic=analytic['j'],
       E_second_fd=float((fp-2*f0+fm)/h**2),E_second_analytic=analytic['K'])

def run():
    result={'scope':'Exact eight-band M0 occupied 4-band static Peierls/Ward audit; no HHG emission or BSV quantized field.',
            'definition':'D = <d^2 H/dA^2>, P = 2 sum_occ,unocc |<a|dH/dA|i>|^2/(ea-ei); K=D-P=d^2 Eocc/dA^2; K_BZ should go to zero.',
            'pair':[pair(frac,N) for frac in [(0.4158,.1683),(.37,.1913),(.21,.31)] for N in (1,-1)],
            'grid':[grid(L) for L in [3,5,9,15,25,41,61]],
            'fd':[fd_test(),fd_test((.37,.1913),h=2e-4)]}
    path=ROOT/'qmath_16_24_ward_sumrule_results.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for p in result['pair']:
        if p['N']==1: print('PAIR',p['frac'],'J_pair',p['Jpair'],'K_pair',p['Kpair'],'D_pair',p['Dpair'],'P_pair',p['Ppair'])
    for r in result['grid']:
        print('GRID L={L} J={meanJ:+.3e} D={meanD:+.3e} P={meanParam:+.3e} K={staticK_full:+.3e} K2={staticK_keep_0_2_4_6_8[1]:+.3e}'.format(**r))
    print('FD',result['fd']);print('saved',path)
if __name__=='__main__':run()
