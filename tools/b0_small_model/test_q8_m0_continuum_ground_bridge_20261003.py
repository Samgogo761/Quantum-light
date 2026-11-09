#!/usr/bin/env python3
"""Independent off-axis M0/continuum bridge checks; no P1 or photon cutoff.

These are resolvent identities, NOT a claim that every bare coordinate has a
finite continuum ground covariance. The tests preserve the ORIGINAL contact.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import quad
from q8_m0_continuum_ground_bridge_20261003 import (
    ContinuumGroundBridge, photon_resolvent)
from q8_m0_classical import k_grid
from q8_m0_cjj import occup_empty_bases, x_from_bases
from q8_m0_gsheet import gamma_self
from q8_m0_model import R0, build_geometry, h8, current_ops
from q8_m0_response import d_ops, mean_d_from_bases


def maxabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.


def j_action(x):
    n=x.shape[0]//2
    return np.concatenate((x[n:],-x[:n]),axis=0)


def independent_free_electron_action(gaps,z,rhs):
    """Direct inverse of each 2x2 oscillator; all q then all p."""
    n=len(gaps)
    g=np.asarray(gaps)[:,None] if np.ndim(rhs)==2 else np.asarray(gaps)
    return np.concatenate(((g*rhs[:n]+1j*z*rhs[n:])/(g*g-z*z),
                            (-1j*z*rhs[:n]+g*rhs[n:])/(g*g-z*z)),axis=0)


def unflattened_m0(nk=8):
    """Retain native k,a,c,v axes for the independent response oracle."""
    geo=build_geometry()
    _,kc,w=k_grid(geo,nk,shift=0.)
    vv,vc,en=occup_empty_bases(h8(geo,R0,kc,1),-0.009010620372198276)
    x=x_from_bases(current_ops(geo,R0,kc,1),vv,vc)
    gaps=en[:,4:,None]-en[:,None,:4]
    original_d=np.einsum("k,kab->ab",w,
        mean_d_from_bases(d_ops(geo,R0,kc,1),vv))
    if maxabs(original_d.imag)>1e-11 or maxabs(original_d-original_d.T)>1e-11:
        raise AssertionError("Unexpected non-real/non-symmetric original contact.")
    d=(original_d.real+original_d.real.T)/2
    v=np.einsum("k,kacv->akcv",np.sqrt(w),x).reshape(2,-1)
    return geo,w,x,gaps,d,v


def independent_kernel(d,w,x,gap_kcv,z):
    # This contraction never calls the flattened spectral_data function.
    c=np.einsum("k,kacv,kbcv->abkcv",w,x.conj(),x)
    return (d+np.einsum("abkcv,kcv->ab",c,1/(z-gap_kcv))
             -np.einsum("abkcv,kcv->ab",c.conj(),1/(z+gap_kcv)))


def report():
    geo,w,x,gap_kcv,d,vertices=unflattened_m0()
    gaps=gap_kcv.reshape(-1)
    gamma=float(gamma_self(geo))
    bridge=ContinuumGroundBridge(gaps,vertices,d,gamma)
    n=len(gaps)
    ci=np.hstack((np.sqrt(2)*vertices.real,np.sqrt(2)*vertices.imag))
    if maxabs(bridge.ci-ci)>1e-12:
        raise AssertionError("Raw electronic coordinate convention mismatch.")
    rng=np.random.default_rng(2026100301)
    rhs=rng.normal(size=(2*n,3))+1j*rng.normal(size=(2*n,3))
    af=rng.normal(size=(2,3))+1j*rng.normal(size=(2,3))
    ipf=rng.normal(size=(2,3))+1j*rng.normal(size=(2,3))
    v4=np.block([[d,-np.eye(2)],[-np.eye(2),np.zeros((2,2))]])
    total_rows=np.block([[np.eye(2),np.zeros((2,2))],[-d,np.eye(2)]])
    checks={}
    samples=(.03+.02j,.1+.04j,1+.3j,10j,.13-.07j)
    for index,z in enumerate(samples):
        label=f"sample_{index}"
        native_k=independent_kernel(d,w,x,gap_kcv,z)
        re=independent_free_electron_action(gaps,z,rhs)
        qi=ci@independent_free_electron_action(gaps,z,ci.T)
        qa=(1j if z.imag>0 else -1j)*gamma/z*np.eye(2)
        bf=np.zeros((4,4),complex)
        bf[:2,:2]=qa; bf[2:,2:]=qi
        t=np.linalg.inv(np.eye(4)+bf@v4)
        q=t@bf
        local=bridge.solve_local_fields(z,af,ipf)
        f=np.eye(2)+qa@native_k
        expected_a=np.linalg.solve(f,af+qa@ipf)
        expected_total=ipf-native_k@expected_a
        lifted=bridge.lift_electrons(z,expected_a,re)
        diff=lifted-re
        acted=np.tile(gaps,2)[:,None]*diff-1j*z*j_action(diff)
        real_integral=quad(lambda om: (1/(om*om-z*z)).real,0,np.inf,
                           epsabs=1e-10,epsrel=1e-10,limit=200)[0]
        imag_integral=quad(lambda om: (1/(om*om-z*z)).imag,0,np.inf,
                           epsabs=1e-10,epsrel=1e-10,limit=200)[0]
        integrated_qa=2*gamma/np.pi*complex(real_integral,imag_integral)
        photon_lift_error=0.
        photon_lift_direct_error=0.
        polarization=np.array([.6,.8])
        for omega in (.005,.1,4.):
            free_pair=rng.normal(size=(2,3))+1j*rng.normal(size=(2,3))
            lifted_ph=bridge.lift_photon_frequency(
                z,omega,polarization,expected_total,free_pair)
            correction=lifted_ph-free_pair
            bare_matrix=np.array([[omega,-1j*z],[1j*z,omega]])
            eta=np.sqrt(gamma/(2*np.pi))
            source=np.vstack((np.sqrt(2)*eta/np.sqrt(omega)*
                              (polarization@expected_total),np.zeros(3)))
            photon_lift_error=max(photon_lift_error,
                                  maxabs(bare_matrix@correction-source))
            photon_lift_direct_error=max(photon_lift_direct_error,
                maxabs(lifted_ph-(free_pair+np.linalg.solve(bare_matrix,source))))
        checks[label]={
            "electron_action":maxabs(bridge.electron_action(z,rhs)-re),
            "raw_ci_resolvent":maxabs(bridge.electron_current_resolvent(z)-qi),
            "native_kacv_kernel":maxabs(bridge.electronic_kernel(z)-native_k),
            "D_minus_QI":maxabs(native_k-(d-qi)),
            "photon_integral_0_infinity":abs(integrated_qa-qa[0,0]),
            "photon_retarded_advanced_branch":maxabs(photon_resolvent(z,gamma)*np.eye(2)-qa),
            "free_vertex_block":maxabs(bridge.vertex_free_resolvent(z)-bf),
            "full_4x4":maxabs(bridge.vertex_resolvent(z)-q),
            "observable_total_current":maxabs(
                bridge.observable_resolvent(z)-total_rows@q@total_rows.T),
            "maxwell_A":maxabs(local["A"]-expected_a),
            "maxwell_I_total":maxabs(local["I_total"]-expected_total),
            "contact_relation":maxabs(local["I_total"]-(local["I_para"]-d@local["A"])),
            "raw_photon_lift_equation":photon_lift_error,
            "raw_photon_lift_independent_2x2":photon_lift_direct_error,
            "raw_electron_lift_equation":maxabs(acted-ci.T@expected_a),
            "raw_electron_lift_free_part":maxabs(lifted-(re+
                independent_free_electron_action(gaps,z,ci.T@expected_a))),
        }

    z=.13+.07j
    branch_error=maxabs(bridge.vertex_resolvent(z.conjugate())-
                       bridge.vertex_resolvent(z).conj().T)
    # A true invariant dark oscillator has zero vertex. Do not confuse an
    # arbitrary instantaneous kernel of the vertex with an invariant sector.
    dark_gaps=np.array([.7,1.1,1.6])
    dark_v=np.array([[.2+.04j,.03-.02j,0.],[.05j,.17,0.]])
    dark_d=np.array([[.6,.01],[.01,.5]])
    dark=ContinuumGroundBridge(dark_gaps,dark_v,dark_d,gamma)
    drhs=np.zeros((6,2),complex); drhs[2,0]=1;drhs[5,1]=1
    free=independent_free_electron_action(dark_gaps,z,drhs)
    forced=dark.solve_local_fields(z,np.zeros((2,2)),dark.ci@free)
    lift=dark.lift_electrons(z,forced["A"],free)
    dark_error=maxabs(lift-free)
    dark_feedback=maxabs(forced["A"])
    # Reject unresolved real-axis limiting prescriptions instead of quietly
    # giving a retarded value to an advanced or on-shell call.
    real_rejected=[]
    for zr in (0.,.1,-.1):
        try: photon_resolvent(zr,gamma)
        except (ValueError,NotImplementedError): real_rejected.append(zr)
    if len(real_rejected)!=3:
        raise AssertionError("photon_resolvent must reject all real-axis calls.")

    bare_covariance_rejected=False
    try: bridge.bare_ground_covariance()
    except NotImplementedError: bare_covariance_rejected=True
    if not bare_covariance_rejected:
        raise AssertionError("Bare continuum covariance must not be silently constructed.")

    maximum=max([branch_error,dark_error,dark_feedback]+
        [float(v) for row in checks.values() for v in row.values()])
    if maximum>2e-9:
        raise AssertionError(f"Independent continuum bridge checks failed: {maximum}")
    k0=independent_kernel(d,w,x,gap_kcv,0.)
    return {
        "status":"independent_off_axis_bridge_identities_passed",
        "date":"2026-10-03","M0_diagnostic_nk":8,
        "raw_electronic_channels":n,"gamma":gamma,
        "samples":[str(z) for z in samples],"checks":checks,
        "retarded_advanced_adjoint_residual":branch_error,
        "strict_dark_free_part_residual":dark_error,
        "strict_dark_feedback":dark_feedback,
        "real_axis_calls_rejected":real_rejected,
        "bare_ground_covariance_rejected":bare_covariance_rejected,
        "maximum_residual":maximum,
        "unmodified_static_kernel_eigenvalues":np.linalg.eigvalsh(k0).tolist(),
        "scope":["off-axis analytic 0<omega<infinity photon resolvent",
                 "raw k,c,v electronic channels and original contact retained",
                 "no photon cutoff, no new broadening, no P1",
                 "not a proof of finite raw continuum ground covariance",
                 "not an on-shell incoming basis or driven optical prediction"],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    result=report()
    source=Path(__file__)
    implementation=source.with_name("q8_m0_continuum_ground_bridge_20261003.py")
    result["source_sha256"]={
        p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source,implementation)}
    if args.out:
        if args.out.exists(): raise FileExistsError(args.out)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as handle:
            json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
