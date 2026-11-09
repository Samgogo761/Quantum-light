#!/usr/bin/env python3
"""M0 raw electronic channels + analytic full photon half-line resolvent.

No photon oscillator discretization, no IR/UV cutoff and no broadening.
Off-axis resolvents are well-defined projections, NOT a certificate that
every unsmeared bare-coordinate ground covariance exists.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def photon_resolvent(z,gamma):
    """Two sides, one polarization: 4 eta^2 int_0^inf dw/(w^2-z^2).

    gamma=2*pi*eta^2. The lower-half-plane advanced branch has the other
    sign. A real-axis boundary/distribution is not silently inferred.
    """
    z=complex(z)
    if not np.isfinite(z) or z.imag==0 or not np.isfinite(gamma) or gamma<=0:
        raise ValueError("Nonreal finite z and positive gamma required.")
    return (1j if z.imag>0 else -1j)*float(gamma)/z


@dataclass
class ContinuumGroundBridge:
    gaps: np.ndarray
    vertices: np.ndarray
    contact: np.ndarray
    gamma: float

    def __post_init__(self):
        self.gaps=np.array(self.gaps,float,copy=True)
        self.vertices=np.array(self.vertices,complex,copy=True)
        raw=np.asarray(self.contact)
        if np.iscomplexobj(raw) and np.max(abs(raw.imag))>1e-12:
            raise ValueError("Original contact must be real.")
        self.contact=np.array(raw.real,float,copy=True)
        self.gamma=float(self.gamma)
        if self.gaps.ndim!=1 or self.vertices.shape!=(2,self.gaps.size) or self.contact.shape!=(2,2):
            raise ValueError("Require gaps[p], vertices[a,p], original contact[a,b].")
        if not all(np.isfinite(x).all() for x in (self.gaps,self.vertices,self.contact)):
            raise ValueError("Nonfinite input.")
        if np.any(self.gaps<=0) or not np.isfinite(self.gamma) or self.gamma<=0:
            raise ValueError("Positive gaps and gamma required.")
        if np.max(abs(self.contact-self.contact.T))>1e-12:
            raise ValueError("Contact must be symmetric; no physics repair.")
        self.ci=np.sqrt(2)*np.hstack((self.vertices.real,self.vertices.imag))
        self.interaction=np.block([[self.contact,-np.eye(2)],[-np.eye(2),np.zeros((2,2))]])
        for x in (self.gaps,self.vertices,self.contact,self.ci,self.interaction):
            x.setflags(write=False)

    def electron_action(self,z,rhs):
        z=complex(z)
        photon_resolvent(z,self.gamma) # same explicit branch/input gate
        rhs=np.asarray(rhs,complex)
        n=self.gaps.size
        if rhs.ndim not in (1,2) or rhs.shape[0]!=2*n:
            raise ValueError("Electronic real-coordinate RHS has 2*n_el rows.")
        d=self.gaps if rhs.ndim==1 else self.gaps[:,None]
        q,p=rhs[:n],rhs[n:]
        den=d*d-z*z
        return np.concatenate(((d*q+1j*z*p)/den,(-1j*z*q+d*p)/den),axis=0)

    def electron_current_resolvent(self,z):
        return self.ci@self.electron_action(z,self.ci.T)

    def electronic_kernel(self,z):
        return self.contact-self.electron_current_resolvent(z)

    def raw_static_residual(self):
        return self.contact-2*np.einsum("ap,bp,p->ab",
            self.vertices.conj(),self.vertices,1/self.gaps).real

    def vertex_free_resolvent(self,z):
        qa=photon_resolvent(z,self.gamma)*np.eye(2)
        qi=self.electron_current_resolvent(z)
        return np.block([[qa,np.zeros((2,2))],[np.zeros((2,2)),qi]])

    def vertex_resolvent(self,z):
        b=self.vertex_free_resolvent(z)
        return np.linalg.solve(np.eye(4)+b@self.interaction,b)

    def observable_resolvent(self,z):
        """Rows (A_x,A_y,I_total,x,I_total,y), I_total=I_para-D A."""
        transform=np.block([[np.eye(2),np.zeros((2,2))],[-self.contact,np.eye(2)]])
        q=self.vertex_resolvent(z)
        return transform@q@transform.T

    def solve_local_fields(self,z,a_free,i_free):
        a_free,i_free=np.asarray(a_free,complex),np.asarray(i_free,complex)
        if a_free.ndim not in (1,2) or a_free.shape!=i_free.shape or a_free.shape[0]!=2:
            raise ValueError("Two current components and common source-column shape required.")
        qa=photon_resolvent(z,self.gamma)
        qi=self.electron_current_resolvent(z)
        k=self.contact-qi
        a=np.linalg.solve(np.eye(2)+qa*k,a_free+qa*i_free)
        ip=i_free+qi@a
        return {"A":a,"I_para":ip,"I_total":ip-self.contact@a}

    def lift_electrons(self,z,A,free_electrons):
        """Recover ALL original p-h coordinates; retain free/dark contribution.

        Off-axis source response. This alone is not an on-shell incoming
        normalization or complete ground spectrum.
        """
        A,free=np.asarray(A,complex),np.asarray(free_electrons,complex)
        return free+self.electron_action(z,self.ci.T@A)

    def lift_photon_frequency(self,z,omega,polarization,I_total,free_pair):
        """Frequency-resolved photon (q,p) coefficient on ONE side.

        Applied after the analytic field solve; no frequency grid selected.
        """
        if not np.isfinite(omega) or omega<=0:
            raise ValueError("Positive photon frequency required.")
        e=np.asarray(polarization,float)
        cur=np.asarray(I_total,complex)
        free=np.asarray(free_pair,complex)
        if e.shape!=(2,) or abs(e@e-1)>1e-12:
            raise ValueError("Unit in-plane polarization required.")
        amp=np.sqrt(2*self.gamma/(2*np.pi*omega))
        drive=amp*(e@cur)
        correction=np.stack((omega*drive,-1j*complex(z)*drive))/(omega**2-complex(z)**2)
        if free.shape!=correction.shape:
            raise ValueError("Photon q,p source column shape mismatch.")
        return free+correction

    def bare_ground_covariance(self):
        raise NotImplementedError(
            "An off-axis resolvent does not imply a finite covariance for "
            "unsmeared A or all bare electronic coordinates. Use justified "
            "observable projections; no silent cutoff or contact replacement.")
