#!/usr/bin/env python3
"""Finite analytical fixtures for initial-state matching, not an M0 production run.

A positive stationary quadratic Hamiltonian has a well-defined Gaussian
ground covariance. Interaction-picture Hessian positivity is not the criterion.
No continuum scattering completeness or IR limit is inferred from these fixtures.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm, schur

def jmat(n):
    eye=np.eye(n)
    return np.block([[0*eye,eye],[-eye,0*eye]])

def maxabs(x):
    return float(np.max(np.abs(x)))

def positive_sqrt(h):
    h=np.asarray(h,float)
    if maxabs(h-h.T)>1e-12 or not np.isfinite(h).all():
        raise ValueError("Require a finite real symmetric stationary Hessian")
    d,u=np.linalg.eigh(h)
    if d.min()<=1e-12*max(1.,abs(d).max()):
        raise ValueError("No strict positive finite Hessian; do not clip zero/negative modes")
    return (u*np.sqrt(d))@u.T,(u/np.sqrt(d))@u.T

def stationary_matching(h):
    """Xi = W zeta_normal; zeta_normal vacuum has Gamma=I/2.

    W is a finite normal-mode map, NOT an asymptotic incoming scattering map.
    """
    root,invroot=positive_sqrt(h)
    n=len(h)//2
    jj=jmat(n)
    aa=root@jj@root
    tt,oo=schur(aa,output="real")
    qcols=[]; pcols=[]; nus=[]
    for i in range(n):
        k=2*i
        value=tt[k,k+1]
        if abs(value)<1e-12:
            raise ValueError("Zero normal frequency needs separate mathematical treatment")
        qcols.append(oo[:,k])
        pcols.append(np.sign(value)*oo[:,k+1])
        nus.append(abs(value))
    o=np.column_stack(qcols+pcols)
    nu=np.r_[nus,nus]
    w=invroot@o@np.diag(np.sqrt(nu))
    gamma=.5*w@w.T
    # Independent coordinate-invariant covariance formula.
    e,u=np.linalg.eigh(-(aa@aa))
    absa=(u*np.sqrt(e))@u.T
    independent=.5*invroot@absa@invroot
    return w,gamma,nu,independent

def finite_fixture():
    """Two photon oscillators, two bright matter oscillators, one dark matter oscillator."""
    frequencies=np.array([.65,1.35,.95,1.7,2.2])
    n=len(frequencies)
    hfree=np.diag(np.r_[frequencies,frequencies])
    ca=np.zeros((1,2*n)); ca[0,:2]=[.18,.11]
    ci=np.zeros((1,2*n))
    vertex=np.array([.32+.09j,.21-.07j,0.j])
    ci[0,2:5]=np.sqrt(2)*vertex.real
    ci[0,n+2:n+5]=np.sqrt(2)*vertex.imag
    invhe=np.zeros(2*n)
    invhe[2:5]=1/frequencies[2:5]
    invhe[n+2:n+5]=1/frequencies[2:5]
    dd=ci@np.diag(invhe)@ci.T
    h=hfree-ca.T@ci-ci.T@ca+ca.T@dd@ca
    # Contact chosen by construction for this manufactured fixture only.
    # This does NOT replace the actual M0 contact with its discrete paramagnetic sum.
    return h,hfree,ca,ci,dd

def packet_edge_fixture():
    """Independent exact sin2 Fourier formula, away from its removable poles."""
    omega0=.05
    tp=8*2*np.pi/omega0
    carrier=2*omega0
    envelope_frequency=2*np.pi/tp
    def raw(omega):
        nu=np.asarray(omega)-carrier
        return envelope_frequency**2*np.expm1(-1j*nu*tp)/(2j*nu*(nu**2-envelope_frequency**2))
    soft=omega0*np.array([1e-6,2e-6,4e-6])
    values=raw(soft)
    slopes=np.diff(np.log(abs(values)))/np.diff(np.log(soft))
    kernel_slopes=np.diff(np.log(abs(values/np.sqrt(soft))))/np.diff(np.log(soft))
    derivative=-envelope_frequency**2*tp/(2*(-carrier)*((-carrier)**2-envelope_frequency**2))
    derivative_rel=maxabs(values/soft-derivative)/abs(derivative)
    return {
        "raw_phi_IR_slopes":slopes.tolist(),
        "radiation_weighted_phi_IR_slopes":kernel_slopes.tolist(),
        "complex_IR_derivative_relative_error":derivative_rel,
        "analytical_IR":"phi=O(omega); phi/sqrt(omega)=O(sqrt(omega))",
        "analytical_UV":"phi=O(omega^-3); phi/sqrt(omega)=O(omega^-7/2)",
        "source_covariance_bound_is_conditional":True,
    }

def fixture_report():
    h,hfree,ca,ci,dd=finite_fixture()
    n=len(h)//2; jj=jmat(n)
    w,gamma,nu,independent=stationary_matching(h)
    winv=np.linalg.inv(w)
    kmat=jj@h
    u=expm(kmat*7.3)
    ph_idx=np.array([0,1,n,n+1])
    el_idx=np.array([2,3,4,n+2,n+3,n+4])
    schur_h=h[np.ix_(ph_idx,ph_idx)]-h[np.ix_(ph_idx,el_idx)]@np.linalg.solve(
        h[np.ix_(el_idx,el_idx)],h[np.ix_(el_idx,ph_idx)])
    bare=.5*np.eye(2*n)
    bare_change=maxabs(u@bare@u.T-bare)
    gdark=gamma.copy()
    gdark[4,4]+=.2; gdark[n+4,n+4]+=.2
    photon_dark_difference=maxabs((u@(gdark-gamma)@u.T)[np.ix_(ph_idx,ph_idx)])
    # A free rotation to the interaction picture is canonical but its generator
    # Hessian contains only the interaction, and need not be positive.
    free_rot=expm(jj@hfree*.37)
    hi=free_rot.T@(h-hfree)@free_rot
    bad_rejections={}
    for name,bad in (("negative",np.diag([1.,-1.])),("zero",np.diag([1.,0.]))):
        try:
            stationary_matching(bad)
            bad_rejections[name]=False
        except ValueError:
            bad_rejections[name]=True
    # Closed two-level counterexample: H after the pulse is constant, rho is not.
    he=np.diag([0.,1.])
    rho=.5*np.ones((2,2),complex)
    et=expm(-1j*he*1.2)
    rhot=et@rho@et.conj().T
    stationary_checks={
        "schur_complement_equals_free_photon":maxabs(schur_h-hfree[np.ix_(ph_idx,ph_idx)]),
        "W_preserves_CCR":maxabs(w@jj@w.T-jj),
        "W_diagonalizes_stationary_H":maxabs(w.T@h@w-np.diag(nu)),
        "covariance_two_constructions":maxabs(gamma-independent),
        "stationary_Lyapunov_equation":maxabs(kmat@gamma+gamma@kmat.T),
        "pure_covariance_identity":maxabs(gamma@jj@gamma-jj/4),
        "exact_no_drive_covariance_evolution":maxabs(u@gamma@u.T-gamma),
        "normal_coordinates_vacuum":maxabs(winv@gamma@winv.T-bare),
        "dark_state_change_in_photon_block":photon_dark_difference,
    }
    controls={
        "stationary_hessian_min_eigenvalue":float(np.linalg.eigvalsh(h).min()),
        "interaction_picture_hessian_min_eigenvalue":float(np.linalg.eigvalsh(hi).min()),
        "bare_vacuum_changes_without_drive":bare_change,
        "internal_dark_covariance_change":maxabs(gdark-gamma),
        "postpulse_rho_commutator":maxabs(he@rho-rho@he),
        "postpulse_rho_change":maxabs(rhot-rho),
        "negative_or_zero_h_rejected":bad_rejections,
    }
    edges=packet_edge_fixture()
    passed=(max(stationary_checks.values())<2e-11
            and controls["stationary_hessian_min_eigenvalue"]>0
            and controls["interaction_picture_hessian_min_eigenvalue"]<-.01
            and bare_change>1e-3
            and controls["postpulse_rho_change"]>.1
            and all(bad_rejections.values())
            and edges["complex_IR_derivative_relative_error"]<1e-3
            and maxabs(np.asarray(edges["radiation_weighted_phi_IR_slopes"])-.5)<1e-3)
    return {
        "status":"finite_boundary_mathematics_fixture_pass" if passed else "FAILED",
        "all_checks_pass":bool(passed),
        "residuals":stationary_checks,
        "counterexamples_and_controls":controls,
        "H2_detector_kernel_endpoint_analysis":edges,
        "M0_parameters_used":False,
        "M0_contact_modified":False,
        "M0_P1_propagation":False,
        "continuum_W_in_constructed":False,
        "continuum_W_out_constructed":False,
        "production_initial_state_selected":False,
        "note":"Finite positive-H covariance formula verified; no continuum/IR/scattering claim.",
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=None)
    args=ap.parse_args()
    rec=fixture_report()
    rec["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(rec,indent=2),encoding="utf-8")
    print(json.dumps(rec,indent=2))
    if not rec["all_checks_pass"]:
        raise SystemExit("Boundary mathematics fixture failed")

if __name__=="__main__":
    main()

