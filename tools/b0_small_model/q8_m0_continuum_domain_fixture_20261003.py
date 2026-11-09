#!/usr/bin/env python3
"""Analytic continuum definition-domain fixture; NOT M0 prediction.
The artificial two-oscillator contact is chosen to obey static Ward exactly.
It is never substituted into the real M0 contact. No physical photon cutoff.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from q8_m0_k1_packet import phi_unnormalized, omega_grid, normalize_phi

W0=.05
TP=8*2*np.pi/W0
WC=2*W0
A=2*np.pi/TP
GAPS=np.array([.7,1.1])
VERTICES=np.array([.2,.3])
D=2*VERTICES**2/GAPS
GAMMA=.15
_GRID=omega_grid(W0,TP)
_,NORM=normalize_phi(_GRID,phi_unnormalized(_GRID,WC,TP))


def phi_stable(omega):
    """Same frozen sin^2 packet; expm1 avoids loss at its IR zero."""
    original_shape=np.shape(omega)
    w=np.atleast_1d(np.asarray(omega,float))
    x=WC-w
    den=2j*x*(A*A-x*x)
    safe=(abs(x)>1e-9)&(abs(abs(x)-A)>1e-9)
    out=np.empty(w.shape,complex)
    # WC*TP is an integer multiple of 2*pi for the fixed H2 packet.
    out[safe]=A*A*np.expm1(-1j*w[safe]*TP)/den[safe]/NORM
    out[~safe]=phi_unnormalized(w[~safe],WC,TP)/NORM
    return out.reshape(original_shape)


def spectra(omega):
    """Exact boundary spectral diagonals, valid also at electronic poles.
    Positive-frequency spectral measure is a commutator/greater weight,
    not normally ordered photon occupation.
    """
    w=np.asarray(omega,float)[...,None]
    if np.any(w<=0): raise ValueError("Positive frequency required.")
    den=GAPS**2-w*w
    common=den*den+GAMMA**2*D**2*w*w
    rho_a=GAMMA*den*den/(np.pi*w*common)
    rho_i=GAMMA*D**2*w**3/(np.pi*common)
    return rho_a,rho_i


def upper_packet(omega):
    """Rigorous UV envelope for omega>WC+A; no averaging of oscillations."""
    w=np.asarray(omega,float)
    x=w-WC
    if np.any(x<=A): raise ValueError("UV envelope requires omega>WC+A.")
    return A*A/(x*(x*x-A*A)*NORM)


def slope(x,y):
    return float(np.polyfit(np.log(x),np.log(y),1)[0])


def report():
    # Matrix spectral jump versus physical left/right photon incoming columns.
    residuals=[]
    for w in (.013,.083,.23,1.6,7.):
        pi=np.diag(2*GAPS*VERTICES**2/(GAPS**2-w*w))
        k=np.diag(D)-pi
        q=1j*GAMMA/w
        r=np.linalg.inv(np.eye(2)/q+k)
        qr=np.block([[r,-r@k],[-k@r,pi+k@r@k]])
        rho=(qr-qr.conj().T)/(2j*np.pi)
        eta=np.sqrt(GAMMA/(2*np.pi))
        af=eta/np.sqrt(w)*np.hstack((np.eye(2),np.eye(2)))
        aa=np.linalg.solve(np.eye(2)+q*k,af)
        incoming=np.vstack((aa,-k@aa))
        residuals.append(float(np.max(abs(rho-incoming@incoming.conj().T))))
        a_spec,i_spec=spectra(w)
        residuals.append(float(np.max(abs(np.diag(rho).real-np.r_[a_spec,i_spec]))))
    check_w=np.array([1e-5,.002,.043,WC-A,WC,WC+A,.174,.387])
    frozen=phi_unnormalized(check_w,WC,TP)/NORM
    packet_error=float(np.max(abs(phi_stable(check_w)-frozen)))

    ir=np.geomspace(1e-8,1e-6,24)
    uv=np.geomspace(1e3,1e5,24)
    ra,ri=spectra(ir); _,ui=spectra(uv)
    slopes={
        "bare_A_IR":slope(ir,ra[:,0]),
        "complete_current_IR":slope(ir,ri[:,0]),
        "H2_source_spectral_integrand_IR":slope(ir,abs(phi_stable(ir))**2/ir*ri[:,0]),
        "complete_current_UV":slope(uv,ui[:,0]),
        "H2_source_spectral_integrand_UV_ENVELOPE":slope(uv,upper_packet(uv)**2/uv*ui[:,0]),
    }
    targets=[-1,3,4,-1,-8]
    if max(abs(v-t) for v,t in zip(slopes.values(),targets))>1e-3:
        raise AssertionError(slopes)
    if max(residuals)>1e-11 or packet_error>1e-10:
        raise AssertionError((residuals,packet_error))

    # Integrability demonstration ONLY. Split numerical intervals do not
    # remove IR/UV modes and never become a production cutoff.
    def integrand(w):
        if w==0: return 0.
        return float(abs(phi_stable(w))**2/w*spectra(w)[1][0])
    edges=[0.,.05,.075,.09,.11,.125,.15,.3,.6,.8,1.2,2.,4.]
    pieces=[quad(integrand,l,r,epsabs=1e-12,epsrel=1e-9,limit=300)
            for l,r in zip(edges[:-1],edges[1:])]
    end=edges[-1]
    tail_bound,error=quad(lambda w: float(upper_packet(w)**2/w*spectra(w)[1][0]),
                         end,np.inf,epsabs=1e-15,epsrel=1e-9,limit=200)
    # Deliberately do not interpret this weight as <b^dagger b>.
    return {
        "status":"analytic_fixture_domain_checks_passed",
        "scope":"artificial Ward-exact spectrum, endpoint and canonical-source identities only",
        "not_M0_contact_replacement":True,
        "not_photon_number":True,
        "not_P1_IR_proof":True,
        "gaps":GAPS.tolist(),"diagonal_vertices":VERTICES.tolist(),
        "artificial_contact":D.tolist(),"gamma":GAMMA,
        "original_H2_norm_retained":float(NORM),
        "matrix_jump_vs_four_photon_columns_max":max(residuals),
        "same_frozen_packet_max_abs_difference":packet_error,
        "asymptotic_slopes":slopes,
        "H2_positive_spectral_weight_integral_0_to_4":sum(v for v,e in pieces),
        "quad_reported_error_0_to_4":sum(e for v,e in pieces),
        "H2_tail_above_4_analytic_envelope_integrated_bound":tail_bound,
        "tail_bound_quadrature_error":error,
        "bare_A_IR_leading_coefficient_gamma_over_pi":GAMMA/np.pi,
        "bare_total_current_UV_leading_coefficients_gamma_D2_over_pi":(GAMMA*D**2/np.pi).tolist(),
        "interpretation":[
            "Positive spectral weight can be finite after H2 filtering although raw A variance diverges.",
            "The positive weight tests a commutator/greater or symmetrized domain; it is NOT stationary emitted photons.",
            "Exact static Ward contact is constructed only in this mathematical fixture.",
            "Finite BZ residual and continuum ground completeness remain separate for M0.",
            "Post-pulse infinite-time tails and driven Gaussian cancellation are not proved here."
        ]
    }


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path)
    args=p.parse_args();result=report()
    result["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
