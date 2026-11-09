#!/usr/bin/env python3
"""Selected continuous-photon columns with stationary dressing and feedback.
No photon oscillator box, no ground covariance, no production photon moments.
B and Bdag coefficients of ONE incoming annihilator are independent arrays.
Vertices u[a,p,t] already contain the propagated electronic interaction-picture
phases. Original D and gamma are retained. Raw electronic coordinates kept.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ColumnState:
    b: np.ndarray
    bdag: np.ndarray
    A: np.ndarray

    def copy(self):
        return ColumnState(self.b.copy(),self.bdag.copy(),self.A.copy())

    @classmethod
    def zeros(cls,n_el,n_col):
        return cls(np.zeros((n_el,n_col),complex),
                   np.zeros((n_el,n_col),complex),np.zeros((2,n_col),complex))

    def packed(self):
        return np.concatenate((self.b,self.bdag,self.A),axis=0)

    @classmethod
    def unpack(cls,packed,n_el):
        a=np.asarray(packed,complex)
        return cls(a[:n_el].copy(),a[n_el:2*n_el].copy(),a[2*n_el:].copy())


def _validate(state,u,D):
    u,D=np.asarray(u,complex),np.asarray(D)
    if u.ndim!=2 or u.shape[0]!=2 or D.shape!=(2,2):
        raise ValueError("Need u[a,p], D[a,b].")
    if state.b.ndim!=2 or state.bdag.shape!=state.b.shape or state.b.shape[0]!=u.shape[1] or state.A.shape!=(2,state.b.shape[1]):
        raise ValueError("Inconsistent original electron/source column dimensions.")
    if any(not np.isfinite(x).all() for x in (u,D,state.b,state.bdag,state.A)):
        raise ValueError("Nonfinite column input.")
    if np.max(abs(D.imag))>1e-11 or np.max(abs(D-D.T))>1e-11:
        raise ValueError("Original real symmetric contact required.")
    return u,D.real


def current(state,u,D):
    u,D=_validate(state,u,D)
    return u.conj()@state.b+u@state.bdag-D@state.A


def rhs(state,u,D,gamma,free_Adot,source=None):
    """Complex-linear selected-column equations, not c-number conjugation."""
    u,D=_validate(state,u,D)
    if not np.isfinite(gamma) or gamma<0:raise ValueError("Nonnegative gamma required.")
    if source is None:source=ColumnState.zeros(*state.b.shape)
    return ColumnState(1j*u.T@state.A+source.b,
                       -1j*u.conj().T@state.A+source.bdag,
                       np.asarray(free_Adot,complex)+gamma*current(state,u,D)+source.A)


def midpoint_step(state,u_mid,D_mid,dt,gamma,free_A_increment=None,source_mid=None):
    """Implicit midpoint via a 2x2 Schur solve; O(n_el*n_column).
    free_A_increment is the exact Af(t+h)-Af(t), equivalent to its step
    average in the forcing. source_mid holds extra bdot, bdagdot, Adot.
    All electronic midpoint rows are recovered, not discarded.
    """
    u,D=_validate(state,u_mid,D_mid)
    if not np.isfinite(dt) or dt<=0 or not np.isfinite(gamma) or gamma<0:
        raise ValueError("Positive step and nonnegative gamma required.")
    if free_A_increment is None:free_A_increment=np.zeros_like(state.A)
    if source_mid is None:source_mid=ColumnState.zeros(*state.b.shape)
    fa=np.asarray(free_A_increment,complex)
    if fa.shape!=state.A.shape or source_mid.b.shape!=state.b.shape or source_mid.bdag.shape!=state.bdag.shape or source_mid.A.shape!=state.A.shape:
        raise ValueError("Source dimensions disagree.")
    if any(not np.isfinite(x).all() for x in (fa,source_mid.b,source_mid.bdag,source_mid.A)):
        raise ValueError("Nonfinite forcing.")
    difference=u.conj()@u.T-u@u.conj().T
    lhs=np.eye(2)+(dt*gamma/2)*D-(1j*dt*dt*gamma/4)*difference
    bfree=state.b+dt*source_mid.b/2
    dfree=state.bdag+dt*source_mid.bdag/2
    target=(state.A+fa/2+dt*source_mid.A/2
            +(dt*gamma/2)*(u.conj()@bfree+u@dfree))
    amid=np.linalg.solve(lhs,target)
    return ColumnState(state.b+dt*(1j*u.T@amid+source_mid.b),
                       state.bdag+dt*(-1j*u.conj().T@amid+source_mid.bdag),
                       2*amid-state.A)


def real_frequency_kernel(gaps,u,D,omega):
    """Original raw rational electronic K at a NONPOLE real frequency.
    Not a continuous-BZ electronic spectral boundary prescription.
    """
    gaps=np.asarray(gaps,float);u=np.asarray(u,complex);D=np.asarray(D)
    if gaps.ndim!=1 or u.shape!=(2,len(gaps)) or D.shape!=(2,2):
        raise ValueError("Raw gap/vertex/contact shapes disagree.")
    if np.any(gaps<=0) or not np.isfinite(gaps).all() or not np.isfinite(u).all() or not np.isfinite(D).all():
        raise ValueError("Finite positive electronic gaps required.")
    if np.max(abs(D.imag))>1e-11 or np.max(abs(D-D.T))>1e-11:
        raise ValueError("Original real symmetric contact required.")
    if not np.isfinite(omega) or omega<=0:raise ValueError("Positive frequency required.")
    threshold=64*np.finfo(float).eps*max(1.,float(np.max(gaps)),float(omega))
    if np.min(abs(gaps-omega))<=threshold:
        raise ValueError("Electronic pole: requires complete spectral treatment, not clipping.")
    pi=(u.conj()/(gaps-omega))@u.T+(u/(gaps+omega))@u.conj().T
    return D.real-pi


@dataclass
class PhotonReference:
    gaps: np.ndarray
    u0: np.ndarray
    D0: np.ndarray
    gamma: float
    omega: np.ndarray
    free_A0: np.ndarray
    initial: ColumnState
    labels: tuple
    kernels: np.ndarray

    def vertices(self,t):
        return self.u0*np.exp(1j*self.gaps*t)[None,:]

    def state(self,t):
        phase=np.exp(-1j*self.omega*t)[None,:]
        return ColumnState(self.initial.b*np.exp(1j*self.gaps*t)[:,None]*phase,
                           self.initial.bdag*np.exp(-1j*self.gaps*t)[:,None]*phase,
                           self.initial.A*phase)

    def free_A(self,t):
        return self.free_A0*np.exp(-1j*self.omega*t)[None,:]

    def free_Adot(self,t):
        return -1j*self.omega[None,:]*self.free_A(t)


def stationary_photon_columns(gaps,u0,D0,gamma,frequencies,polarizations=None):
    """Two sides x two real polarizations at each selected positive omega.
    These are particular scattering photon columns, NOT a complete ground.
    Finite k M=K(0) is retained; no static correction, no hidden i*eta.
    """
    gaps=np.array(gaps,float,copy=True);u0=np.array(u0,complex,copy=True)
    D0=np.array(D0,copy=True)
    freq=np.atleast_1d(np.asarray(frequencies,float))
    pol=np.eye(2) if polarizations is None else np.asarray(polarizations,float)
    if pol.shape!=(2,2) or not np.isfinite(pol).all() or np.max(abs(pol.T@pol-np.eye(2)))>1e-11:
        raise ValueError("Two fixed real orthonormal polarization columns required.")
    if not np.isfinite(gamma) or gamma<=0 or freq.ndim!=1 or len(freq)==0 or not np.isfinite(freq).all() or np.any(freq<=0):
        raise ValueError("Positive gamma and a nonempty finite positive frequency vector required.")
    eta=np.sqrt(gamma/(2*np.pi))
    omega=np.repeat(freq,4)
    free=np.concatenate([eta/np.sqrt(w)*np.hstack((pol,pol)) for w in freq],axis=1)
    result=ColumnState.zeros(len(gaps),len(omega))
    kernels=[];labels=[]
    for i,w in enumerate(freq):
        k=real_frequency_kernel(gaps,u0,D0,float(w))
        start=4*i;sl=slice(start,start+4)
        result.A[:,sl]=np.linalg.solve(np.eye(2)+1j*gamma*k/w,free[:,sl])
        result.b[:,sl]=(u0.T@result.A[:,sl])/(gaps-w)[:,None]
        result.bdag[:,sl]=(u0.conj().T@result.A[:,sl])/(gaps+w)[:,None]
        kernels.append(k)
        labels.extend((float(w),side,pol_index) for side in ("L","R") for pol_index in (0,1))
    return PhotonReference(gaps,u0,D0.real,float(gamma),omega,free,result,tuple(labels),np.array(kernels))


def increment_source(t,uA,DA,reference):
    """Exact driven-minus-reference equation forcing in the SAME inputs.
    No vacuum or DC subtraction. Reference output still contributes to m.
    """
    ref=reference.state(t);ur=reference.vertices(t)
    du=np.asarray(uA,complex)-ur;dd=np.asarray(DA)-reference.D0
    si=du.conj()@ref.b+du@ref.bdag-dd@ref.A
    return ColumnState(1j*du.T@ref.A,-1j*du.conj().T@ref.A,reference.gamma*si)


def increment_current(t,delta,uA,DA,reference):
    src=increment_source(t,uA,DA,reference)
    return current(delta,uA,DA)+src.A/reference.gamma


def assemble_ground_or_photon_moments(*args,**kwargs):
    raise NotImplementedError(
        "Selected photon columns alone are not complete incoming ground channels. "
        "Electronic spectral completeness, dark/point weights and output tails "
        "remain required; no n,m,V or canonical-output claim from this prototype.")
