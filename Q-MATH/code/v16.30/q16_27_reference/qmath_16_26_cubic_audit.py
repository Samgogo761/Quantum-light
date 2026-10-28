#!/usr/bin/env python3
"""Q-MATH-16.26: exact M0 eight-band Peierls cubic current.
Reconstructs original q167 bond table, computes an independently expanded density
hierarchy and direct full-Peierls dynamics. No BSV/quantized output claim.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import sys, json, time, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'source'/'reference_q167'
if not SRC.exists(): SRC=ROOT/'work'/'Q-MATH-16.25'/'base'/'source'/'deps'/'q19'/'reference_q167'
sys.path[:0]=[str(SRC),str(SRC/'deps')]
import q167_m0_three_photon_T as q
SPINS=(q.UP,q.DN)

def bond_data(L,N,Delta=.8,lam=.1,shift=0.):
    # Construct entire BZ in single spin-decomposed batch.
    fr=np.array([((i+.5)/L,(j+.5)/L) for i in range(L) for j in range(L)])
    cart=np.array([q.geo.k_cart(x) for x in fr]); nk=len(cart)
    zmat=np.zeros((nk,2,4,4),complex)
    diag=np.array([Delta*N,-Delta*N,Delta*N,-Delta*N,-Delta*N,Delta*N,-Delta*N,Delta*N],float)
    for s,ix in enumerate(SPINS):
        zmat[:,s,:,:]=np.diag(diag[ix])
    bonds=[]
    for delta,B,kind in q.BONDS:
        B=B*(lam/.1 if kind=='SOC' else 1.)
        Bs=np.stack([B[np.ix_(ix,ix)] for ix in SPINS],axis=0)
        kp=cart@delta+shift*float(np.dot(delta,q.p))
        phase=np.exp(1j*kp)
        dot=float(np.dot(delta,q.p))
        bonds.append((phase,dot,Bs))
    return fr,cart,zmat,bonds

def derivatives(base,bonds,order=4):
    """Physical Hermitian ∂^r_A h(k+A p), r=0..order."""
    n=base.shape[0]
    out=np.zeros((order+1,n,2,4,4),complex)
    out[0]=base
    for phase,dot,Bs in bonds:
        Z=phase[:,None,None,None]*Bs[None,:,:,:]
        Zc=Z.conj().swapaxes(-1,-2)
        for r in range(order+1):
            val=(1j*dot)**r
            out[r]+=val*Z+np.conj(val)*Zc
    return out

def mixed_via_ordered_bonds(base,bonds,order=3):
    out=np.zeros((order+1,)+base.shape,complex)
    for ((phase,dot,Bs),(delta,B,kind)) in zip(bonds,q.BONDS):
        hd=float(np.dot(delta,q.h))
        Z=phase[:,None,None,None]*Bs[None,:,:,:]
        Zc=Z.conj().swapaxes(-1,-2)
        for m in range(order+1):
            val=(1j)**(m+1)*hd*dot**m
            out[m]+=val*Z+np.conj(val)*Zc
    return out

def initial_occ(h0):
    eig,U=np.linalg.eigh(h0)
    Uocc=U[...,:,:2]
    return Uocc @ Uocc.conj().swapaxes(-1,-2), eig

def comm(h,rr):
    return np.matmul(h,rr)-np.matmul(rr,h)

def wave(t,T,omega):
    return (np.sin(np.pi*t/T)**2)*np.cos(omega*t)

def hierarchy_rhs(t,R,H,T,omega):
    f=wave(t,T,omega)
    powers=[1.,f,f*f/2.,f*f*f/6.]
    derivative=np.zeros_like(R)
    for n in range(4):
        s=comm(H[0],R[n])
        for m in range(1,n+1):
            s+=powers[m]*comm(H[m],R[n-m])
        derivative[n]=-1j*s
    return derivative

def transverse_second_components(t,R,M,T,omega):
    f=wave(t,T,omega)
    terms=[np.einsum('...ij,...ji->...',R[2],M[0]).real,
           f*np.einsum('...ij,...ji->...',R[1],M[1]).real,
           f*f/2*np.einsum('...ij,...ji->...',R[0],M[2]).real]
    return np.array([a.sum(axis=1).mean() for a in terms])

def direct_transverse(t,rho,bonds,T,omega,alphas):
    f=wave(t,T,omega)
    j=np.zeros_like(rho)
    for ((phase,dot,Bs),(delta,B,kind)) in zip(bonds,q.BONDS):
        hdot=float(np.dot(delta,q.h))
        ZZ=phase[None,:,None,None,None]*Bs[None,None,:,:,:]
        M=1j*hdot*np.exp(1j*alphas[:,None,None,None,None]*dot*f)*ZZ
        j+=M+M.conj().swapaxes(-1,-2)
    return np.einsum('a...ij,a...ji->a...',rho,j).real.sum(axis=-1).mean(axis=-1)

def current_components(t,R,H,T,omega):
    f=wave(t,T,omega)
    terms=[np.einsum('...ij,...ji->...',R[3],H[1]).real,
           f*np.einsum('...ij,...ji->...',R[2],H[2]).real,
           f*f/2*np.einsum('...ij,...ji->...',R[1],H[3]).real,
           f*f*f/6*np.einsum('...ij,...ji->...',R[0],H[4]).real]
    return np.array([a.sum(axis=1).mean() for a in terms])

def direct_rhs(t,rho,base,bonds,T,omega,alphas):
    # rho shape (4 amplitudes, Nk,2,4,4)
    f=wave(t,T,omega)
    h=np.broadcast_to(base[None,:,:,:,:],rho.shape).copy()
    for phase,dot,Bs in bonds:
        ZZ=phase[None,:,None,None,None]*Bs[None,None,:,:,:]
        M=np.exp(1j*alphas[:,None,None,None,None]*dot*f)*ZZ
        h+=M+M.conj().swapaxes(-1,-2)
    return -1j*comm(h,rho)

def direct_current(t,rho,base,bonds,T,omega,alphas):
    f=wave(t,T,omega)
    j=np.zeros_like(rho)
    for phase,dot,Bs in bonds:
        ZZ=phase[None,:,None,None,None]*Bs[None,None,:,:,:]
        M=1j*dot*np.exp(1j*alphas[:,None,None,None,None]*dot*f)*ZZ
        j+=M+M.conj().swapaxes(-1,-2)
    val=np.einsum('a...ij,a...ji->a...',rho,j).real
    return val.sum(axis=-1).mean(axis=-1)

def propagate(L=5,Delta=.8,N=1,lam=.1,steps=480,omega=.18,cycles=1.5,mode='both',amplitude=.014):
    T=cycles*2*np.pi/omega;dt=T/steps
    fr,k,base,bonds=bond_data(L,N,Delta,lam)
    H=derivatives(base,bonds,4);Htrans=mixed_via_ordered_bonds(base,bonds,2)
    P,eig=initial_occ(H[0]);R=np.zeros((4,)+P.shape,complex);R[0]=P
    amps=np.array([-2*amplitude,-amplitude,amplitude,2*amplitude])
    rho=np.broadcast_to(P[None],(4,)+P.shape).copy()
    Jc=np.zeros((steps+1,4));Jf=np.zeros((steps+1,4));Jt=np.zeros((steps+1,3));Jtf=np.zeros((steps+1,4));ts=np.linspace(0,T,steps+1)
    max_drift=0
    for it,t in enumerate(ts):
        Jc[it]=current_components(t,R,H,T,omega)
        Jf[it]=direct_current(t,rho,base,bonds,T,omega,amps)
        Jt[it]=transverse_second_components(t,R,Htrans,T,omega)
        Jtf[it]=direct_transverse(t,rho,bonds,T,omega,amps)
        if it==steps:break
        # Exact evolution of finite polynomial response hierarchy up to third order.
        k1=hierarchy_rhs(t,R,H,T,omega)
        k2=hierarchy_rhs(t+dt/2,R+dt/2*k1,H,T,omega)
        k3=hierarchy_rhs(t+dt/2,R+dt/2*k2,H,T,omega)
        k4=hierarchy_rhs(t+dt,R+dt*k3,H,T,omega)
        R+=(dt/6)*(k1+2*k2+2*k3+k4)
        # Independent full-exponential-of-bond direct dynamics.
        k1=direct_rhs(t,rho,base,bonds,T,omega,amps)
        k2=direct_rhs(t+dt/2,rho+dt/2*k1,base,bonds,T,omega,amps)
        k3=direct_rhs(t+dt/2,rho+dt/2*k2,base,bonds,T,omega,amps)
        k4=direct_rhs(t+dt,rho+dt*k3,base,bonds,T,omega,amps)
        rho+=(dt/6)*(k1+2*k2+2*k3+k4)
    extract=(Jf[:,3]-Jf[:,0]-2*(Jf[:,2]-Jf[:,1]))/(12*amplitude**3)
    cubic=Jc.sum(axis=1)
    quad=Jt.sum(axis=1)
    # Zero-amplitude transverse background from time-independent filled-band projector.
    j0t=np.einsum('...ij,...ji->...',P,Htrans[0]).real.sum(axis=1).mean()
    quad_fd=(Jtf[:,2]+Jtf[:,1]-2*j0t)/(2*amplitude**2)
    # Smooth Fourier projection to 3omega of cubic in the vector potential amplitude.
    def transform(arr,harmonic=3):
        return np.trapezoid(arr*np.exp(1j*harmonic*omega*ts),ts)/T
    metrics={'L':L,'Delta':Delta,'N':N,'lambda':lam,'steps':steps,'dt':dt,'T':T,'omega':omega,
         'a_fd':amplitude,'fundamental_to_third_harmonic_coefficient': [float(x) for x in (transform(cubic).real,transform(cubic).imag)],
         'sum_cubic_components': [float(x) for x in (transform(cubic).real,transform(cubic).imag)],
         'direct_fd_cubic': [float(x) for x in (transform(extract).real,transform(extract).imag)],
         'fourth_vertex_term': [float(x) for x in (transform(Jc[:,3]).real,transform(Jc[:,3]).imag)],
         'without_fourth_vertex': [float(x) for x in (transform(cubic-Jc[:,3]).real,transform(cubic-Jc[:,3]).imag)],
         'component_fourier':[[float(transform(Jc[:,i]).real),float(transform(Jc[:,i]).imag)] for i in range(4)],
         'max_timeseries_fd_error_relative':float(np.max(abs(extract-cubic))/max(np.max(abs(cubic)),1e-12)),
         'fourier_fd_error_relative':float(abs(transform(extract)-transform(cubic))/max(abs(transform(cubic)),1e-12)),
         'fourth_vertex_fraction':float(abs(transform(Jc[:,3]))/max(abs(transform(cubic)),1e-12)),
         'transverse_second_harmonic':[float(x) for x in (transform(quad,2).real,transform(quad,2).imag)],
         'transverse_second_fd':[float(x) for x in (transform(quad_fd,2).real,transform(quad_fd,2).imag)],
         'transverse_second_fd_rel':float(abs(transform(quad_fd,2)-transform(quad,2))/max(abs(transform(quad,2)),1e-12)),
         'transverse_mixed_third_vertex_fraction':float(abs(transform(Jt[:,2],2))/max(abs(transform(quad,2)),1e-12)),
         'third_order_trace_max':float(np.max(abs(np.trace(R[3],axis1=-2,axis2=-1)))),
         'min_direct_gap':float(np.min(eig[...,2]-eig[...,1])),
         'pump_only_no_output_source':'classical finite pulse; NO BSV or physical H2 Maxwell output'}
    return metrics,ts,cubic,extract,Jc,quad,quad_fd,Jt

if __name__=="__main__":
    print("For validated benchmarks, run: python qmath_16_26_run_verified.py")
