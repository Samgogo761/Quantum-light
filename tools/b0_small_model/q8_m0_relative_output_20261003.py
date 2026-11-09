#!/usr/bin/env python3
"""Output contractions in ONE complete canonical input basis.

U0 is the stationary physical output row. dU,V are the driven increment.
This does not construct the M0 continuum embedding or authorize P1 runs.
"""
from __future__ import annotations
import numpy as np


def maxabs(x):
    x=np.asarray(x)
    return float(np.max(np.abs(x))) if x.size else 0.


def compose_relative_output(u0,du,v,*,canonical_tolerance=1e-9):
    u0,du,v=(np.asarray(x,complex) for x in (u0,du,v))
    if u0.ndim!=2 or du.shape!=u0.shape or v.shape!=u0.shape:
        raise ValueError("All rows must use the same complete input channel basis.")
    if not all(np.isfinite(x).all() for x in (u0,du,v)):
        raise ValueError("Nonfinite output coefficient.")
    identity=np.eye(u0.shape[0])
    ref=maxabs(u0@u0.conj().T-identity)
    u=u0+du
    normal=maxabs(u@u.conj().T-v@v.conj().T-identity)
    anomalous=maxabs(u@v.T-v@u.T)
    if max(ref,normal,anomalous)>canonical_tolerance:
        raise ValueError(f"Noncanonical output: reference={ref}, normal={normal}, anomalous={anomalous}")
    # n_mu,nu=<b_mu^dagger b_nu>; this is NOT V V^dagger.
    n=v.conj()@v.T
    m=u@v.T
    return {"U":u,"V":v,"n":n,"m":m,
            "m_reference_source":u0@v.T,"m_source_source":du@v.T,
            "reference_ccr":ref,"normal_ccr":normal,"anomalous_ccr":anomalous,
            "scope":"complete-input vacuum contraction; not a computed M0 source"}


def full_ground_modes_finite(h):
    """Full finite positive h only; not an asymptotic wave operator.

    Xi=F c+F* c^dagger, [c,c^dagger]=I and c|ground>=0.
    F includes all finite normal modes, including source-dark directions.
    """
    h=np.asarray(h,float)
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or h.shape[0]%2:
        raise ValueError("Even square Hessian required.")
    if not np.isfinite(h).all() or maxabs(h-h.T)>1e-12:
        raise ValueError("Real symmetric finite Hessian required.")
    eigen,basis=np.linalg.eigh(h)
    if eigen.min()<=0:
        raise ValueError("Positive h required; no clipping or contact correction.")
    n=h.shape[0]//2
    j=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
    hs=(basis*np.sqrt(eigen))@basis.T
    hi=(basis/np.sqrt(eigen))@basis.T
    a=hs@j@hs
    nu,e=np.linalg.eigh(1j*a)
    keep=nu>0
    if np.count_nonzero(keep)!=n:
        raise ValueError("Incomplete positive spectrum.")
    nu=nu[keep]
    f=hi@(e[:,keep]*np.sqrt(nu))
    return nu,f,j


def resolvent_from_positive_modes(nu,f,z):
    """Q(z)=(h-i z J)^-1, both positive/negative frequency poles retained."""
    z=complex(z)
    nu=np.asarray(nu,float)
    f=np.asarray(f,complex)
    return (f/(nu-z))@f.conj().T+(f.conj()/(nu+z))@f.T


def physical_continuum_embedding(*args,**kwargs):
    raise NotImplementedError(
        "M0 needs scattering-resolved complete ground channels in the SAME "
        "basis as the direct photon row; rank-two current factors are insufficient.")


def dress_continuous_channels(q_free_retarded, interaction, free_incoming_columns):
    """Lippmann-Schwinger dressing with FIXED free incoming labels.

    Q=(I+Qf V)^-1 Qf; F_in=(I+Qf V)^-1 F_free,in.
    This returns the scattering spectral part only. Bound-pole residues and
    invariant dark sectors must not be silently omitted from a ground state.
    """
    q=np.asarray(q_free_retarded,complex)
    v=np.asarray(interaction,complex)
    f=np.asarray(free_incoming_columns,complex)
    if q.ndim!=2 or q.shape[0]!=q.shape[1] or v.shape!=q.shape or f.ndim!=2 or f.shape[0]!=q.shape[0]:
        raise ValueError("Inconsistent projected resolvent/source dimensions.")
    if not all(np.isfinite(x).all() for x in (q,v,f)) or maxabs(v-v.conj().T)>1e-12:
        raise ValueError("Finite Hermitian interaction required.")
    lhs=np.eye(q.shape[0])+q@v
    return np.linalg.solve(lhs,q),np.linalg.solve(lhs,f)
