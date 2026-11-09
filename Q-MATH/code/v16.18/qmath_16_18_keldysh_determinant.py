#!/usr/bin/env python3
"""Q-MATH-16.18: production M0 exact free-fermion Keldysh determinant vs 36d propagation.
Derived readouts are finite-time causal current susceptibilities, not photon S matrices.
"""
from pathlib import Path
import sys, json, time
import numpy as np
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'work'/'src_0'
sys.path.insert(0,str(SRC));sys.path.insert(0,str(SRC/'deps'))
import q167_m0_three_photon_T as q
m=q.m
UP=np.array([0,2,4,6]); DN=np.array([1,3,5,7]); spins=(UP,DN)


def spin_ground(k,N,params=m.R0):
    H=m.h8(q.geo,params,k,N)
    return tuple(np.linalg.eigh(H[np.ix_(ids,ids)])[1][:,:2] for ids in spins)


def one_body_paths(k,N,Aplus,Aminus,dt,params=m.R0):
    Uall=[]
    for As in (Aplus,Aminus):
        U=np.eye(8,dtype=complex)
        for A in As:
            H=m.h8(q.geo,params,k+A,N)
            # H is Hermitian and spin conserving.
            E,V=np.linalg.eigh(H)
            U=(V*np.exp(-1j*E*dt))@V.conj().T@U
        Uall.append(U)
    return Uall


def influence_det(k,N,Aplus,Aminus,dt,params=m.R0):
    B=spin_ground(k,N,params)
    Up,Um=one_body_paths(k,N,Aplus,Aminus,dt,params)
    vals=[]
    for ids,b in zip(spins,B):
        vals.append(np.linalg.det(b.conj().T@Um[np.ix_(ids,ids)].conj().T@Up[np.ix_(ids,ids)]@b))
    return vals[0]*vals[1]


def wedge2(A):
    pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
    cols=[(0,1)] if A.shape[1]==2 else pairs
    out=np.zeros((6,len(cols)),complex)
    for ii,(i,j) in enumerate(pairs):
        for jj,(a,b) in enumerate(cols):
            out[ii,jj]=A[i,a]*A[j,b]-A[i,b]*A[j,a]
    return out


def manybody_full_36(k,N,Aplus,Aminus,dt,params=m.R0):
    Us=[]
    for As in (Aplus,Aminus):
        U=np.eye(36,dtype=complex)
        for A in As:
            H=m.h8(q.geo,params,k+A,N)
            U=expm(-1j*q.dgamma(H)*dt)@U
        Us.append(U)
    B=spin_ground(k,N,params)
    ref=np.kron(*[wedge2(b)[:,0] for b in B])  # first pair (0,1)
    return np.vdot(ref,Us[1].conj().T@Us[0]@ref)


def det_logs(k,N,pump,probe,x,y,dt,params=m.R0):
    # A_cl=x*pump ; A_q=y*probe; A±=A_cl±A_q/2.
    A0=x*pump
    cp=A0+0.5*y*probe
    cm=A0-0.5*y*probe
    return np.log(influence_det(k,N,cp,cm,dt,params))


def integrate_current(k,N,pump,probe,x,dt,params=m.R0):
    B=spin_ground(k,N,params)
    C=[]
    for ids,b in zip(spins,B):
        # Full embedded 8x4 occupation matrix; equivalently use each spin separately.
        C.append(b.copy())
    integral=0.0
    for A,gg in zip(x*pump,probe):
        H=m.h8(q.geo,params,k+A,N)
        D=m.dh8(q.geo,params,k+A,N)
        # g(t) is a vector A_q; current = -partial H/partial A.
        J=-np.einsum('d,dij->ij',gg,D)
        for ids,b in zip(spins,C):
            integral+=dt*float(np.trace(b.conj().T@J[np.ix_(ids,ids)]@b).real)
        E,V=np.linalg.eigh(H)
        U=(V*np.exp(-1j*E*dt))@V.conj().T
        C=[U[np.ix_(ids,ids)]@b for ids,b in zip(spins,C)]
    return integral


def centered_second(fun,h):
    return (fun(h)-2*fun(0)+fun(-h))/h**2


def centered_fourth(fun,h):
    return (fun(-2*h)-4*fun(-h)+6*fun(0)-4*fun(h)+fun(2*h))/h**4


def test_exact():
    dt=0.086;steps=7
    ts=(np.arange(steps)+.5)*dt
    pump=np.sin(1.17*ts)[:,None]*q.p[None,:]*.08
    probe=np.cos(.79*ts)[:,None]*q.h[None,:]*.063
    checks=[]
    for frac,N in [((.173,.271),1),((.4158,.1683),-1)]:
        k=q.geo.k_cart(frac)
        vdet=influence_det(k,N,pump,probe,dt)
        v36=manybody_full_36(k,N,pump,probe,dt)
        same=influence_det(k,N,pump,pump,dt)
        checks.append({'frac':frac,'N':N,'det':[vdet.real,vdet.imag],
          '36':[v36.real,v36.imag], 'residual':float(abs(vdet-v36)),
          'same_branch_residual':float(abs(same-1))})
    return checks


def retarded_check():
    dt=.065;steps=35
    ts=(np.arange(steps)+.5)*dt
    env=np.sin(np.pi*(np.arange(steps)+.5)/steps)**2
    pump=(env*np.sin(.18*ts*6))[:,None]*q.p[None,:]
    probe=(env*np.cos(.18*ts*10))[:,None]*q.h[None,:]
    results=[]
    for frac,N in [((.4158,.1683),1),((.173,.271),-1)]:
        k=q.geo.k_cart(frac)
        for x in (0.0,.06,-.06):
            Jy=integrate_current(k,N,pump,probe,x,dt)
            for y in (.02,.01):
                Fp=det_logs(k,N,pump,probe,x,y,dt)
                Fm=det_logs(k,N,pump,probe,x,-y,dt)
                # At finite dt midpoint q-source is a finite-difference time integral.
                derivative=(Fp-Fm)/(2*y)
                results.append({'frac':frac,'N':N,'x':x,'y':y,
                    'integrated_current':Jy,'detivative_over_i': [float((derivative/1j).real),float((derivative/1j).imag)],
                    'real_residual':float(abs((derivative/1j).real-Jy))})
    return results


def low_order_response():
    dt=.072;steps=16
    ts=(np.arange(steps)+.5)*dt
    env=np.sin(np.pi*(np.arange(steps)+.5)/steps)**2
    pump=(env*np.sin(1.5*ts))[:,None]*q.p[None,:]
    probes=[(env*np.sin(2.8*ts))[:,None]*q.h[None,:],
            (env*np.cos(2.8*ts))[:,None]*q.h[None,:]]
    data=[]
    for L in (3,5,7):
        for lam in (.10,0.0):
            prm=m.M0Params(lam_so=lam)
            vals=[]
            for N in (1,-1):
                derivatives=[]
                for probe in probes:
                    for h in (.10,):
                        s=0.
                        for i in range(L):
                            for j in range(L):
                                kc=q.geo.k_cart(((i+.5)/L,(j+.5)/L))
                                jf=lambda x: integrate_current(kc,N,pump,probe,x,dt,prm)
                                s+=centered_second(jf,h)
                        derivatives.append(s/L**2)
                vals.append(derivatives)
            a,b=np.asarray(vals)
            data.append({'L':L,'lambda':lam,'response_plus':a.tolist(),
              'response_minus':b.tolist(),'neel_even':((a+b)/2).tolist(),
              'neel_odd':((a-b)/2).tolist(),'relative_even_leakage':float(np.linalg.norm(a+b)/max(np.linalg.norm(a-b),1e-20))})
    return data


def main():
    start=time.time()
    checks=test_exact()
    print('det-vs-36 maximal residual',max(x['residual'] for x in checks),flush=True)
    retarded=retarded_check()
    print('retarded J current / Keldysh det max midpoint error',max(x['real_residual'] for x in retarded),flush=True)
    scan=low_order_response()
    for row in scan:
        print('L',row['L'],'lambda',row['lambda'],'odd',row['neel_odd'],'even leak',row['relative_even_leakage'],flush=True)
    out={'type':'production M0 finite-time determinant vs 36d, no Maxwell output or LSZ connected photon S','exact_determinant_checks':checks,
       'retarded_current_checks':retarded,'neel_even_odd_current_second_order':scan,'elapsed_seconds':time.time()-start}
    (ROOT/'qmath_16_18_results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print('elapsed',time.time()-start)
if __name__=='__main__':main()
