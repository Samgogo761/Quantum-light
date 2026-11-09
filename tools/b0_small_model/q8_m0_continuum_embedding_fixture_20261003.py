#!/usr/bin/env python3
"""Analytic continuous-band embedding fixture with an explicit bound mode.

This is a mathematical test, NOT the M0 photon cutoff or its physical
spectrum. A genuine continuous free spectrum is retained analytically.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.integrate import quad_vec
from q8_m0_relative_output_20261003 import dress_continuous_channels


def ma(x):return float(np.max(np.abs(x)))


def report():
    a,b,lam=.5,1.5,.8
    width=b-a
    p=.5*np.array([[1,1j],[-1j,1]])
    j=np.array([[0.,1.],[-1.,0.]])
    interaction=np.diag([lam,0.])
    source=np.array([[1],[-1j]])/np.sqrt(2*width)

    def free_q(omega,inside=False):
        plus=(np.log(abs((b-omega)/(a-omega)))+(1j*np.pi if inside else 0.))/width
        minus=np.log((b+omega)/(a+omega))/width
        return p*plus+p.conj()*minus

    def rho(omega):
        q=free_q(omega,True)
        _,f=dress_continuous_channels(q,interaction,source)
        return f@f.conj().T

    checks={}
    samples=[]
    for omega in (.55,.7,1.1,1.4):
        qf=free_q(omega,True)
        q,f=dress_continuous_channels(qf,interaction,source)
        free_jump=(qf-qf.conj().T)/(2j*np.pi)
        jump=(q-q.conj().T)/(2j*np.pi)
        samples.append(max(ma(free_jump-source@source.conj().T),ma(jump-f@f.conj().T)))
        direct=np.linalg.inv(np.linalg.inv(qf)+interaction)
        checks[f"resolvent_{omega}"]=ma(q-direct)
    checks["continuous_jump_fixed_incoming"]=max(samples)
    continuous,error=quad_vec(rho,a,b,epsabs=1e-10,epsrel=1e-10)

    # Positive q^2 perturbation creates a real pole above this artificial band.
    r=np.exp(-2*width/lam)
    bound=np.sqrt((b*b-r*a*a)/(1-r))
    qb=free_q(bound)
    derivative=bound/width*(1/(a*a-bound*bound)-1/(b*b-bound*bound))
    column=qb[:,0]
    residue=np.outer(column,column.conj())/derivative
    checks["bound_equation"]=abs(1+lam*qb[0,0])
    total=continuous+residue
    checks["full_CCR_with_bound"]=ma(total-total.conj()-1j*j)
    checks["ground_uncertainty"]=max(0.,-float(np.linalg.eigvalsh(total).min()))
    missing_bound=ma(continuous-continuous.conj()-1j*j)
    # Independent imaginary-frequency ground-covariance integral. No pole
    # residues are supplied to this path.
    def imaginary_covariance(nu):
        ap=np.log1p(width/(a-1j*nu))/width
        q=p*ap+p.conj()*ap.conjugate()
        dressed=np.linalg.solve(np.eye(2)+q@interaction,q)
        return .5*(dressed+dressed.T).real/np.pi
    gamma_imag,error_imag=quad_vec(imaginary_covariance,0.,np.inf,
                                   epsabs=1e-10,epsrel=1e-10)
    checks["spectral_ground_vs_imaginary_resolvent"]=ma(total.real-gamma_imag)

    # Spectral residue is checked separately from the sum rule.
    eps=1e-6
    qnear=free_q(bound+eps)
    dressed,_=dress_continuous_channels(qnear,interaction,source)
    checks["bound_residue_limit"]=ma(-eps*dressed-residue)
    # The dark second sector stays free; it is not erased by its zero coupling.
    qf=free_q(.9,True)
    q4=np.block([[qf,np.zeros((2,2))],[np.zeros((2,2)),qf]])
    v4=np.diag([lam,0.,0.,0.])
    f4=np.zeros((4,2),complex)
    f4[:2,0]=source[:,0]
    f4[2:,1]=source[:,0]
    qd,fd=dress_continuous_channels(q4,v4,f4)
    checks["dark_resolvent_retained"]=ma(qd[2:,2:]-qf)
    checks["dark_incoming_column_retained"]=ma(fd[2:,1]-source[:,0])
    limits={k:1e-9 for k in checks}
    limits["bound_residue_limit"]=2e-5
    flags={k:bool(v<limits[k]) for k,v in checks.items()}
    flags["bound_omission_detected"]=missing_bound>.1
    return {"all_checks_pass":bool(all(flags.values())),"checks":checks,"limits":limits,
            "pass_flags":flags,"missing_bound_CCR_error":missing_bound,
            "bound_frequency":float(bound),"continuous_integral_error_estimate":float(error),
            "imaginary_resolvent_integral_error_estimate":float(error_imag),
            "positive_spectral_projection_real":total.real.tolist(),
            "scope":"analytic continuous-band fixture only",
            "M0_continuum_complete":False,"M0_physical_cutoff_selected":False,
            "P1_run":False}


if __name__=="__main__":
    result=report()
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["all_checks_pass"] else 1)
