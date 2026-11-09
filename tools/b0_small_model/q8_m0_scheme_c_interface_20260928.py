#!/usr/bin/env python3
"""Scheme C: pure mode/covariance interface fixtures. No physical propagation.

The production input state, finite-time/asymptotic matching and new detection
modes are NOT selected here. All nonzero output moments below are synthetic.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np

from q8_m0_classical import p1_a2_protocol, kappa_shift
from q8_m0_model import R0, build_geometry, h8, current_ops, op_P
from q8_m0_s1_kernel import parallel_perp_basis
from q8_m0_k1_packet import phi_unnormalized
from q8_m0_gsheet import two_port_from_current
from q8_m0_phgauss import io_from_phi

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/scheme_c_interface_20260928"
TOL = 2e-10

def maxabs(x):
    x = np.asarray(x)
    return float(np.max(np.abs(x))) if x.size else 0.

def jmat(n):
    eye = np.eye(n)
    return np.block([[0*eye, eye], [-eye, 0*eye]])

def transform_moments(d, beta, n, m, *, antiunitary=False):
    """Target b_B = D (U_g b_A U_g^-1), rho_B=U_g rho_A U_g^-1.
    This does not identify a time-reversed IN experiment with a same-pulse OUT.
    n_ij=<delta b_i^dag delta b_j>; no transposed alternative convention.
    """
    beta, n, m = map(np.asarray, (beta, n, m))
    if antiunitary:
        beta, n, m = beta.conj(), n.conj(), m.conj()
    d = np.asarray(d)
    return d @ beta, d.conj() @ n @ d.T, d @ m @ d.T

def canonical_residuals(f, g, gram=None):
    f, g = np.asarray(f), np.asarray(g)
    gram = np.eye(f.shape[0]) if gram is None else np.asarray(gram)
    return {
        "b_bdag": maxabs(f @ f.conj().T - g @ g.conj().T - gram),
        "b_b": maxabs(f @ g.T - g @ f.T),
    }

def real_map(f, g):
    """Full Bogoliubov map -> real quadrature map, all q followed by all p."""
    return np.block([[(f+g).real, -(f-g).imag],
                     [(f+g).imag, (f-g).real]])

def nm_from_symmetrized_covariance(v):
    n = len(v)//2
    qq, qp, pq, pp = v[:n,:n], v[:n,n:], v[n:,:n], v[n:,n:]
    return (qq+pp+1j*(qp-pq)-np.eye(n))/2, (qq-pp+1j*(qp+pq))/2

def input_moment_matrix(n, m):
    """Unsymmetrized <(c,c^dag)(c,c^dag)^T>, input is explicitly centered."""
    if n is None or m is None:
        raise ValueError("Input covariance is required; there is no vacuum default")
    n, m = np.asarray(n, complex), np.asarray(m, complex)
    if n.ndim != 2 or n.shape[0] != n.shape[1] or m.shape != n.shape:
        raise ValueError("Input covariance dimensions")
    if maxabs(n-n.conj().T)>TOL or maxabs(m-m.T)>TOL:
        raise ValueError("n must be Hermitian and m symmetric")
    eye = np.eye(len(n))
    q = np.block([[m, eye+n.T], [n, m.conj()]])
    l = np.block([[eye, eye], [-1j*eye, 1j*eye]])/np.sqrt(2)
    vv = l @ q @ l.T
    v = ((vv+vv.T)/2).real
    if np.linalg.eigvalsh(v+0.5j*jmat(len(n))).min() < -TOL:
        raise ValueError("Input covariance violates the uncertainty inequality")
    return q

def output_moments(f, g, mean_out, *, n_in, m_in, input_state_id):
    """Centered input moments are mandatory; mean_out comes from a separate mean calculation.
    No state is inferred from photon-row normalization or from existing Phi.
    """
    if not input_state_id:
        raise ValueError("An explicit input-state provenance label is required")
    f, g = np.asarray(f), np.asarray(g)
    beta = np.asarray(mean_out, complex)
    if f.shape != g.shape or beta.shape != (f.shape[0],):
        raise ValueError("Output dimensions")
    if max(canonical_residuals(f,g).values()) > TOL:
        raise ValueError("Noncanonical output rows")
    q = input_moment_matrix(n_in, m_in)
    if q.shape != (2*f.shape[1], 2*f.shape[1]):
        raise ValueError("Input channel count")
    row = np.concatenate([f,g], axis=1)
    dag = np.concatenate([g.conj(),f.conj()], axis=1)
    n, m = dag @ q @ row.T, row @ q @ row.T
    return beta, n, m

def rows_from_existing_phi(phi, n_ph, n_el, packet_rows):
    """Adapt the existing S,T,U,V extraction; no claim Phi already spans asymptotic OUT.
    packet_rows must be a coisometry on the designated photon output basis.
    """
    q = np.asarray(packet_rows, complex)
    if q.ndim != 2 or q.shape[1] != n_ph:
        raise ValueError("Packet row dimensions")
    if maxabs(q @ q.conj().T-np.eye(q.shape[0])) > TOL:
        raise ValueError("Overlapping packets: keep Gram or review new orthogonal modes")
    rec = io_from_phi(phi, n_ph, n_el)
    f = q @ np.concatenate([rec["S"], rec["U"]], axis=1)
    g = q @ np.concatenate([rec["T"], rec["V"]], axis=1)
    return f, g

def fixture_bogoliubov():
    """Four-mode physical algebra fixture, NOT M0 and NOT a physical input choice."""
    h = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],float)/2
    u = np.diag(np.exp(1j*np.array([.2,-.3,.5,.7]))) @ h
    r = np.array([.18,.12,.0,.07])
    theta = np.array([.3,-.8,.0,.5])
    return u @ np.diag(np.cosh(r)), u @ np.diag(np.exp(1j*theta)*np.sinh(r))

def fixture_input():
    """A correlated Gaussian input distinct from vacuum, for independent moment checks."""
    f,g = fixture_bogoliubov()
    p = real_map(f,g)
    thermal = np.array([.12,.05,.07,.02])+.5
    v = p @ np.diag(np.r_[thermal,thermal]) @ p.T
    n,m = nm_from_symmetrized_covariance(v)
    return v,n,m

def candidate_packets(geo, prot):
    """Numerical positive-frequency packet Gram; new candidates remain unapproved."""
    ep,et = parallel_perp_basis(geo)
    # No photon continuum propagation; this is only a detector-shape quadrature.
    om = np.linspace(0., 12*prot.omega0, 6145)
    w = np.full(len(om),om[1]-om[0]); w[[0,-1]] *= .5
    phis = {}
    for harmonic in (2,3):
        raw = phi_unnormalized(om,harmonic*prot.omega0,prot.t_pulse)
        phis[harmonic] = raw/np.sqrt(np.sum(w*abs(raw)**2))
    cards = []
    for name, ha, hb in (("A_H2parallel_H2perp",2,2),("B_H3parallel_H2perp",3,2)):
        q = np.stack([(np.sqrt(w)*phis[ha])[:,None]*ep,
                      (np.sqrt(w)*phis[hb])[:,None]*et]).reshape(2,-1)
        gram = q @ q.conj().T
        cards.append({"id":name,"status":"proposed_not_production_approved",
                      "side":"R (+z outgoing)","harmonics":[ha,hb],
                      "polarizations":["parallel_a2","perp_a2"],
                      "primary":"(m_plus[0,1]-m_minus[0,1])/2, complex",
                      "controls":"full beta,n,m; total |beta|^2+n_diag",
                      "gram_maxabs_minus_I":maxabs(gram-np.eye(2))})
    same_pol_overlap = np.sum(w*phis[2]*phis[3].conj())
    return {"cards":cards,"omega_max_over_omega0":12.,
            "quadrature_points":len(om),
            "same_polarization_H2_H3_overlap_abs":float(abs(same_pol_overlap)),
            "orthogonality_reason":"orthogonal real polarizations, not separated centers alone",
            "bandwidth":"existing positive-frequency pulse-matched sin2 Fourier family"}

def checks():
    geo=build_geometry(); prot=p1_a2_protocol(.8); ep,et=parallel_perp_basis(geo)
    p=op_P()
    times=np.array([0.,.17,.43,.81,1.])*prot.t_pulse
    aa=kappa_shift(times,geo,prot)
    p_h=p_j=0.
    for kf in ([0.,0.],[.23,-.17],[.31,.22]):
        k=geo.k_cart(kf)
        for neel in (1,-1):
            for a in aa:
                h=h8(geo,R0,k+a,neel); j=current_ops(geo,R0,k+a,neel)
                p_h=max(p_h,maxabs(p@h@p.conj().T-h8(geo,R0,-k-a,-neel)))
                p_j=max(p_j,maxabs(p@j@p.conj().T+current_ops(geo,R0,-k-a,-neel)))
    current=np.array([[.2,-.1],[0.,.3],[-.4,.2]])
    left=np.array([[.1,.2],[.4,.0],[-.1,.3]])
    right=np.array([[.3,0.],[-.2,.2],[.2,-.4]])
    rr=two_port_from_current(current,left,right,.01,.2)
    pr=two_port_from_current(-current,-right,-left,.01,.2)
    asym=maxabs((rr["A_R_out"]-rr["A_L_out"])-(left-right))
    p_port=max(maxabs(pr["A_R_out"]+rr["A_L_out"]),maxabs(pr["A_L_out"]+rr["A_R_out"]))
    f0,g0=fixture_bogoliubov(); phi=real_map(f0,g0)
    q=np.diag(np.exp(1j*np.array([.24,-.61]))) @ np.array([[1.,1.],[-1.,1.]])/np.sqrt(2)
    f,g=rows_from_existing_phi(phi,2,2,q)
    v0,n0,m0=fixture_input()
    beta=np.array([.2+.1j,-.3+.2j])
    _,n,m=output_moments(f,g,beta,n_in=n0,m_in=m0,input_state_id="synthetic_correlated_fixture_only")
    vt=phi@v0@phi.T; nr,mr=nm_from_symmetrized_covariance(vt)
    nref=q.conj()@nr[:2,:2]@q.T; mref=q@mr[:2,:2]@q.T
    d=np.diag(np.exp(1j*np.array([.3,-.7])))
    beta2,n2,m2=transform_moments(d,beta,n,m)
    b3,n3,m3=output_moments(d@f,d@g,d@beta,n_in=n0,m_in=m0,input_state_id="synthetic_fixture_only")
    allchecks={
        "P_H_approved_a2_protocol":p_h,"P_current_approved_a2_protocol":p_j,
        "drive_perp_component":maxabs(aa@et),
        "asymmetric_port_difference_identity":asym,"parity_full_ports":p_port,
        "existing_io_F_vs_known_fixture":maxabs(f-q@f0[:2]),
        "existing_io_G_vs_known_fixture":maxabs(g-q@g0[:2]),
        "canonical_b_bdag":canonical_residuals(f,g)["b_bdag"],
        "canonical_b_b":canonical_residuals(f,g)["b_b"],
        "n_vs_independent_quadrature":maxabs(n-nref),"m_vs_independent_quadrature":maxabs(m-mref),
        "mode_rephase_beta":maxabs(beta2-b3),"mode_rephase_n":maxabs(n2-n3),"mode_rephase_m":maxabs(m2-m3),
        "total_intensity_rephase":maxabs(abs(beta2)**2+np.diag(n2).real-abs(beta)**2-np.diag(n).real),
    }
    # This is ideal-mirror tensor algebra, not a simulation of approximate M0.
    mirror=np.diag([1.,-1.])
    _,nm,mm=transform_moments(mirror,beta,n,m)
    allchecks["ideal_mirror_m_diagonal_even"]=maxabs(np.diag(mm)-np.diag(m))
    allchecks["ideal_mirror_m_cross_odd"]=float(abs(mm[0,1]+m[0,1]))
    cards=candidate_packets(geo,prot)
    allchecks["candidate_pair_gram"]=max(x["gram_maxabs_minus_I"] for x in cards["cards"])
    return {
        "status":"scheme_C_definition_and_synthetic_interface_checks_complete",
        "protocol":asdict(prot),"protocol_id":prot.protocol_id,
        "old_test_only_protocol":"p1_protocol(0.1), x; not the research working point",
        "polarization_vectors":{"parallel":ep.tolist(),"perp":et.tolist()},
        "max_residuals":allchecks,"all_within_2e_minus10":bool(max(allchecks.values())<TOL),
        "candidate_modes":cards,
        "unresolved":["physical initial-state matching","asymptotic output analysis matching",
                      "driven quasi-boson validity","production candidate approval"],
        "P1_propagation_run":False,"physical_photon_moments_computed":False,
        "synthetic_moments_are_not_M0_results":True,
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=None)
    args=parser.parse_args()
    rec=checks()
    files=[Path(__file__),REPO/"tools/b0_small_model/q8_m0_phgauss.py",
           REPO/"tools/b0_small_model/q8_m0_model.py",
           REPO/"tools/b0_small_model/q8_m0_classical.py",
           REPO/"tools/b0_small_model/q8_m0_scheme_c_output_map_20260926.py",
           REPO/"tools/b0_small_model/q8_m0_s1_kernel.py",
           REPO/"tools/b0_small_model/q8_m0_k1_packet.py",
           REPO/"tools/b0_small_model/q8_m0_gsheet.py",
           REPO/"tools/b0_small_model/test_q8_m0_scheme_c_interface_20260928.py"]
    rec["source_sha256"]={str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(rec,indent=2),encoding="utf-8")
    print(json.dumps(rec,indent=2))
    if not rec["all_within_2e_minus10"]:
        raise SystemExit("Interface residual check failed")
if __name__=="__main__":
    main()

