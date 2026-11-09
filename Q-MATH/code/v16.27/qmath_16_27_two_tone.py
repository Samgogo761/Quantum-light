#!/usr/bin/env python3
"""Q-MATH-16.27 exact M0 finite-time, two independent pump-source mixed functional derivative.
NOT a quantum BSV or autonomous asymptotic susceptibility.  Uses original M0 bond table.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import json, math, time, sys
from pathlib import Path
import numpy as np
from qmath_16_26_cubic_audit import bond_data, derivatives, mixed_via_ordered_bonds, initial_occ, comm, q
OUT=Path(__file__).resolve().parent

def pulse(t,T,w,phase=0):
    return np.sin(np.pi*t/T)**2*np.cos(w*t+phase)

def propagate_two(L=9,Delta=.8,lam=.1,N=1,w1=.16,w2=.20,T=None,steps=360,phase2=0.,cross_direct=False,a=.0003, pulse_type='sin2'):
    if T is None: T=1.5*2*np.pi/.18
    fr, k, base, bonds=bond_data(L,N,Delta,lam)
    H=derivatives(base,bonds,2)
    M=mixed_via_ordered_bonds(base,bonds,2)
    P,eig=initial_occ(H[0]); tmp=np.zeros((3,)+P.shape,complex)
    R=np.zeros((4,)+P.shape,complex);R[0]=P
    # order: 00,10,01,11.  Coeff 11 = mixed partial derivative wrt amplitudes.
    ts=np.linspace(0,T,steps+1); dt=T/steps
    current=np.zeros(steps+1,float);parts=np.zeros((steps+1,4),float)
    if cross_direct:
        amps=np.array([(a,a),(a,-a),(-a,a),(-a,-a)])
        rho=np.broadcast_to(P[None],(4,)+P.shape).copy()
        direct=np.zeros(steps+1)
        def direct_H(t):
            f1=pulse(t,T,w1); f2=pulse(t,T,w2,phase2)
            A=amps[:,0]*f1+amps[:,1]*f2
            Hout=np.broadcast_to(base[None,:,:,:,:],rho.shape).copy()
            Mh=np.zeros_like(Hout)
            for ((phase,dot,Bs),(delta,B,kind)) in zip(bonds,q.BONDS):
                fac=np.exp(1j*A[:,None,None,None,None]*dot)*phase[None,:,None,None,None]*Bs[None,None,:,:,:]
                Hout+=fac+fac.conj().swapaxes(-1,-2)
                dh=float(np.dot(delta,q.h))
                Mfac=1j*dh*fac
                Mh+=Mfac+Mfac.conj().swapaxes(-1,-2)
            return Hout,Mh
        def d_rhs(t,rr):
            f1=pulse(t,T,w1); f2=pulse(t,T,w2,phase2)
            A=amps[:,0]*f1+amps[:,1]*f2
            hh=np.broadcast_to(base[None,:,:,:,:],rr.shape).copy()
            for phase,dot,Bs in bonds:
                fac=np.exp(1j*A[:,None,None,None,None]*dot)*phase[None,:,None,None,None]*Bs[None,None,:,:,:]
                hh+=fac+fac.conj().swapaxes(-1,-2)
            return -1j*comm(hh,rr)
    else:rho=None;direct=None
    def rhs(t,Y):
        f1=pulse(t,T,w1); f2=pulse(t,T,w2,phase2)
        dY=np.empty_like(Y)
        dY[0]=-1j*comm(H[0],Y[0])
        dY[1]=-1j*(comm(H[0],Y[1])+f1*comm(H[1],Y[0]))
        dY[2]=-1j*(comm(H[0],Y[2])+f2*comm(H[1],Y[0]))
        dY[3]=-1j*(comm(H[0],Y[3])+f1*comm(H[1],Y[2])+f2*comm(H[1],Y[1])+f1*f2*comm(H[2],Y[0]))
        return dY
    for j,t in enumerate(ts):
        f1=pulse(t,T,w1);f2=pulse(t,T,w2,phase2)
        parts[j,0]=np.einsum('...ij,...ji->...',R[3],M[0]).real.sum(axis=-1).mean()
        parts[j,1]=f1*np.einsum('...ij,...ji->...',R[2],M[1]).real.sum(axis=-1).mean()
        parts[j,2]=f2*np.einsum('...ij,...ji->...',R[1],M[1]).real.sum(axis=-1).mean()
        parts[j,3]=f1*f2*np.einsum('...ij,...ji->...',R[0],M[2]).real.sum(axis=-1).mean()
        current[j]=parts[j].sum()
        if cross_direct:
            _,jj=direct_H(t)
            vals=np.einsum('a...ij,a...ji->a...',rho,jj).real.sum(axis=-1).mean(axis=-1)
            direct[j]=(vals[0]-vals[1]-vals[2]+vals[3])/(4*a*a)
        if j==steps:break
        k1=rhs(t,R);k2=rhs(t+dt/2,R+dt/2*k1);k3=rhs(t+dt/2,R+dt/2*k2);k4=rhs(t+dt,R+dt*k3)
        R+=dt/6*(k1+2*k2+2*k3+k4)
        if cross_direct:
            k1=d_rhs(t,rho);k2=d_rhs(t+dt/2,rho+dt/2*k1);k3=d_rhs(t+dt/2,rho+dt/2*k2);k4=d_rhs(t+dt,rho+dt*k3)
            rho+=dt/6*(k1+2*k2+2*k3+k4)
    def fourier(x,Omega): return np.trapezoid(x*np.exp(1j*Omega*ts),ts)/T
    zsum=fourier(current,w1+w2); zdiff=fourier(current,abs(w1-w2))
    vals={'L':L,'Delta':Delta,'lambda':lam,'N':N,'w1':w1,'w2':w2,'T':T,'steps':steps,'dt':dt,'phase2':phase2,
          'sum_frequency': [float(zsum.real),float(zsum.imag)],
          'difference_frequency': [float(zdiff.real),float(zdiff.imag)],
          'contact_contributions': [[float(z.real),float(z.imag)] for z in [fourier(parts[:,i],w1+w2) for i in range(4)]],
          'min_direct_gap':float(np.min(eig[...,2]-eig[...,1])),
          'max_trace_error_11':float(np.max(abs(np.trace(R[3],axis1=-2,axis2=-1)))),
          'source':'Two-tone causal mixed derivative of prescribed classical Peierls source, not a quantum-light response; finite T window.'}
    if cross_direct:
        zz=fourier(direct,w1+w2)
        vals['direct_mixed']=[float(zz.real),float(zz.imag)]
        vals['relative_mixed_fd_error']=float(abs(zz-zsum)/max(abs(zsum),1e-14))
    return vals,ts,current,parts,(direct if cross_direct else None)

def run():
    tic=time.monotonic();records=[]
    common=dict(w1=.16,w2=.20,steps=420)
    for L in (5,9,13,17,25,33):
        v,ts,x,p,d=propagate_two(L=L,**common,cross_direct=(L==9))
        records.append(v);print('mesh',L,'sum',v['sum_frequency'],'gap',v['min_direct_gap'],'s',round(time.monotonic()-tic,1),flush=True)
        if L==9:np.savez_compressed(OUT/'qmath_16_27_mixed_timeseries.npz',t=ts,j11=x,parts=p,direct=d)
    for config in [dict(L=9,Delta=-.8,N=1),dict(L=9,Delta=0,N=1),dict(L=9,lam=0),
                   dict(L=9,w1=.2,w2=.16),dict(L=9,w1=.18,w2=.18),dict(L=9,phase2=np.pi/2),
                   dict(L=9,steps=840)]:
        v,*_=propagate_two(**dict(common,**config));records.append(v)
        print('check',[(k,config[k]) for k in config],'zsum',v['sum_frequency'], 's',round(time.monotonic()-tic,1),flush=True)
    # coalescence mixed derivative vs 2x single-envelope second derivative:
    from qmath_16_26_cubic_audit import propagate
    base, *_=propagate(L=9,Delta=.8,steps=420,omega=.18,cycles=1.5,mode='both',amplitude=.0003)
    pair=next(x for x in records if x['L']==9 and abs(x['w1']-.18)<1e-8 and abs(x['w2']-.18)<1e-8)
    a=complex(*pair['sum_frequency']);b=complex(*base['transverse_second_harmonic'])
    dd=lambda a,b:abs(a-b)/max(abs(b),1e-14)
    ref=next(x for x in records if x['L']==9 and x['Delta']==.8 and x['steps']==420 and x['phase2']==0 and x['w1']==.16)
    minus=next(x for x in records if x['L']==9 and x['Delta']==-.8)
    zeros=[x for x in records if x['L']==9 and (x['Delta']==0 or x['lambda']==0)]
    swap=next(x for x in records if x['L']==9 and abs(x['w1']-.2)<1e-8 and abs(x['w2']-.16)<1e-8)
    fine=next(x for x in records if x['L']==9 and x['steps']==840)
    checks={'coincident_freq_vs_2times_q16_26_rel':float(dd(a,2*b)),
            'neel_odd_relative':float(abs(complex(*ref['sum_frequency'])+complex(*minus['sum_frequency']))/max(abs(complex(*ref['sum_frequency'])),1e-14)),
            'pump_permutation_relative':float(dd(complex(*ref['sum_frequency']),complex(*swap['sum_frequency']))),
            'dt_refine_relative':float(dd(complex(*ref['sum_frequency']),complex(*fine['sum_frequency']))),
            'zero_N_or_SOC_magnitudes':[float(abs(complex(*x['sum_frequency']))) for x in zeros],
            'independent_full_peierls_fd_relative':float(ref['relative_mixed_fd_error'])}
    out={'checks':checks,'data':records,'scope':'Finite-time exact 8band M0 causal mixed derivative; not continuum stationary susceptibility; not quantized BSV.'}
    (OUT/'qmath_16_27_results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print('CHECKS',json.dumps(checks,indent=2), 'total_s',round(time.monotonic()-tic,1))
if __name__=='__main__':run()
