#!/usr/bin/env python3
"""Stationary H2 OUTPUT packet in continuous incoming-channel representation.

The frequency integral is quadrature; these nodes are NOT finite bare photon
oscillators. Reuses frozen Stage1AO conventions, with eta=1e-4 only as a
retarded-spectrum diagnostic. It does not establish an eta/BZ continuum
limit, construct W_in, run P1, or predict driven photons/entanglement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from q8_m0_classical import k_grid, p1_a2_protocol
from q8_m0_cjj import occup_empty_bases, x_from_bases
from q8_m0_gsheet import gamma_self
from q8_m0_k1_packet import omega_grid, phi_unnormalized, normalize_phi
from q8_m0_model import R0, build_geometry, h8, current_ops
from q8_m0_response import d_ops, mean_d_from_bases
from q8_m0_s1_kernel import parallel_perp_basis
from q8_m0_stage1ai_lowfreq_audit import spectral_data
from q8_m0_stage1al_pole_audit import kernel_off_cuts
from q8_m0_stage1ao_canonical_channels import ETA, even_maps, factor_psd

REPO = Path(__file__).resolve().parents[2]
MU = -0.009010620372198276
DEFAULT_OUT = REPO/"results/q_line/stationary_h2_outpacket_20261003/packet.json"

def maxabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0

def pack(x):
    x=np.asarray(x)
    return {"real":np.real(x).tolist(),"imag":np.imag(x).tolist()}

def photon_output_rows(s, la):
    """Rows [R_out x/y,L_out x/y]; cols [R_in x/y,L_in x/y,el].

    R/L are fixed same-side labels. At zero coupling R_out=L_in.
    The electronic outputs are deliberately not invented.
    """
    s,la=np.asarray(s,complex),np.asarray(la,complex)
    if s.shape!=(2,2) or la.ndim!=2 or la.shape[0]!=2:
        raise ValueError("S is 2x2 and L_a is 2xr.")
    if not np.isfinite(s).all() or not np.isfinite(la).all():
        raise ValueError("Nonfinite channel coefficient.")
    eye=np.eye(2)
    right=np.hstack(((s-eye)/2,(s+eye)/2,la/np.sqrt(2)))
    left=np.hstack(((s+eye)/2,(s-eye)/2,la/np.sqrt(2)))
    return np.vstack((right,left))

def packet_quadrature(order=8, panels=48):
    """Original approved packet, without per-quadrature rescue normalization."""
    p=p1_a2_protocol(.8) # duration/frequency ONLY, no pump propagation
    tp,w0=float(p.t_pulse),float(p.omega0)
    grid=omega_grid(w0,tp)
    raw=phi_unnormalized(grid,2*w0,tp)
    _,normalizer=normalize_phi(grid,raw)
    lo,hi=float(grid[grid>0][0]),float(grid[-1])
    x,w=leggauss(order); edges=np.linspace(lo,hi,panels+1)
    centers=(edges[:-1]+edges[1:])/2; half=np.diff(edges)/2
    om=(centers[:,None]+half[:,None]*x).ravel()
    weights=(half[:,None]*w).ravel()
    phi=phi_unnormalized(om,2*w0,tp)/normalizer
    mass=float(np.dot(weights,abs(phi)**2))
    # |FT[sin^2]| <= (4/3) Omega^2 / |omega-omega_c|^3 beyond 2 Omega.
    envelope_omega=2*np.pi/tp
    if hi-2*w0<2*envelope_omega:
        raise ValueError("The analytic UV tail bound requires hi-omega_c>=2*Omega.")
    uv_bound=16*envelope_omega**4/(45*(hi-2*w0)**5*normalizer**2)
    ir_mass,ir_error=quad(lambda q:float(abs(phi_unnormalized(q,2*w0,tp)/normalizer)**2),
                           0,lo,epsabs=1e-18,epsrel=1e-10,limit=100)
    return {"omega":om,"weights":weights,"phi":phi,"raw_mass":mass,
            "normalizer":float(normalizer),"interval":[lo,hi],
            "omega0":w0,"center":2*w0,"tp":tp,
            "IR_omitted_mass_quadrature":float(ir_mass),
            "IR_omitted_mass_quad_error_estimate":float(ir_error),
            "UV_omitted_mass_analytic_upper_bound":float(uv_bound),
            "packet_renormalized_on_quadrature":False,
            "normalizer_reference":"existing positive-frequency omega_grid trapezoid; finite reference not exact continuum norm",
            "quadrature_meaning":"integral of continuum output rows, not discrete coupled photon modes"}

def spectral_input(nk=96):
    geo=build_geometry()
    _,kc,w=k_grid(geo,nk,shift=0.)
    vv,vc,en=occup_empty_bases(h8(geo,R0,kc,1),MU)
    x=x_from_bases(current_ops(geo,R0,kc,1),vv,vc)
    gaps=en[:,4:,None]-en[:,None,:4]
    gap,c=spectral_data(gaps,x,w)
    d=np.einsum("k,kab->ab",w,mean_d_from_bases(d_ops(geo,R0,kc,1),vv))
    return geo,d,gap,c

def maps_at_frequency(d,gap,c,omega,gamma,eta=ETA):
    subgap=omega<float(np.min(gap))
    if subgap:
        kr=kernel_off_cuts(d,gap,c,omega)[0]
        ci=np.zeros((2,2),complex)
    else:
        kr=kernel_off_cuts(d,gap,c,omega+1j*eta)[0]
        ka=kernel_off_cuts(d,gap,c,omega-1j*eta)[0]
        ci=1j*(kr-ka)
    factor=factor_psd(ci)
    if not factor["psd"]:
        raise ValueError("Electronic spectral factor is not PSD; no repair/subtraction permitted.")
    s,g,_=even_maps(kr,omega,gamma)
    la=np.sqrt(omega/gamma)*(g@factor["L"])
    return s,la,{"subgap":subgap,"rank":factor["rank"],
        "ci_factor_residual":factor["reconstruction_residual_after_tolerance_clip"],
        "ci_min_eigenvalue":float(min(factor["eigenvalues"])),
        "raw_even_ccr_residual":maxabs(s@s.conj().T+(omega/gamma)*g@ci@g.conj().T-np.eye(2)),
        "factor_even_ccr_residual":maxabs(s@s.conj().T+la@la.conj().T-np.eye(2))}

def report(nk=96,order=8,panels=48):
    start=time.perf_counter()
    q=packet_quadrature(order,panels)
    refined=packet_quadrature(2*order,panels) # packet mass only, no new electronic calculation
    geo,d,gap,c=spectral_input(nk)
    ep,et=parallel_perp_basis(geo)
    pol=np.asarray((ep,et))
    rotation=np.zeros((4,4)); rotation[:2,:2]=pol; rotation[2:,2:]=pol
    gram=np.zeros((4,4),complex)
    gram_noel=np.zeros_like(gram); gram_noodd=np.zeros_like(gram)
    wrong_number=np.zeros((2,2),complex)
    max_row=0.; max_raw=0.; max_factor=0.; min_ci=0.; ranks=set()
    gamma=float(gamma_self(geo))
    for i,(om,wt,ph) in enumerate(zip(q["omega"],q["weights"],q["phi"])):
        s,la,meta=maps_at_frequency(d,gap,c,float(om),gamma)
        row=rotation@photon_output_rows(s,la)
        a=wt*abs(ph)**2
        gram+=a*(row@row.conj().T)
        gram_noel+=a*(row[:,:4]@row[:,:4].conj().T)
        # Omit the odd incoming photon channel, which leaves the 1/2 vacuum
        # missing from each same-side output and introduces inter-side overlap.
        even_only=np.hstack((np.vstack((pol@s,pol@s))/np.sqrt(2),
                              np.vstack((pol@la,pol@la))/np.sqrt(2)))
        gram_noodd+=a*(even_only@even_only.conj().T)
        wrong_number+=a*(pol@la@la.conj().T@pol.T)/2
        max_row=max(max_row,maxabs(row@row.conj().T-np.eye(4)))
        max_raw=max(max_raw,meta["raw_even_ccr_residual"])
        max_factor=max(max_factor,meta["ci_factor_residual"])
        min_ci=min(min_ci,meta["ci_min_eigenvalue"]);ranks.add(meta["rank"])
        if i%96==0:
            print(f"stationary packet frequency {i+1}/{len(q['omega'])}",flush=True)
    packet_meta={k:v for k,v in q.items() if k not in ("omega","weights","phi")}
    packet_meta.update({"nodes":len(q["omega"]),"order":order,"panels":panels,
        "double_order_mass":refined["raw_mass"],
        "mass_refinement_absolute":abs(refined["raw_mass"]-q["raw_mass"])})
    required=[Path(__file__),Path(__file__).with_name("q8_m0_stage1ao_canonical_channels.py"),
        Path(__file__).with_name("q8_m0_stage1ai_lowfreq_audit.py"),
        Path(__file__).with_name("q8_m0_stage1al_pole_audit.py"),
        Path(__file__).with_name("q8_m0_k1_packet.py"),
        Path(__file__).with_name("q8_m0_model.py")]
    closed=max_row<1e-9 and abs(q["raw_mass"]-1)<1e-7
    return {
        "status":"stationary_output_packet_control_passed" if closed else "stationary_output_packet_control_not_qualified",
        "date":"2026-10-03","elapsed_seconds":time.perf_counter()-start,
        "electronic_input":{"domain":1,"Gamma_nk":nk,"n_k":nk*nk,
            "all_transitions":len(gap),"gap_min":float(gap.min()),
            "static_contact_retained":True,"eta_above_gap":ETA,
            "eta_meaning":"fixed retarded-spectrum diagnostic; no physical linewidth assigned",
            "eta_or_BZ_continuum_qualified":False},
        "gamma_self":gamma,"packet":packet_meta,
        "outputs":["R H2 parallel","R H2 perpendicular","L H2 parallel","L H2 perpendicular"],
        "source_channels":"at every frequency factor C_I=L Ldag; no global two-mode electron truncation",
        "positive_frequency_source":"annihilation B, not creation Bdag",
        "source_factor_ranks":sorted(ranks),"ci_min_eigenvalue":min_ci,
        "max_ci_factor_residual":max_factor,
        "max_pointwise_raw_even_ccr_residual":max_raw,
        "max_pointwise_all_photon_row_ccr_residual":max_row,
        "packet_gram":pack(gram),
        "packet_gram_minus_rawmass_identity_maxabs":maxabs(gram-q["raw_mass"]*np.eye(4)),
        "packet_gram_minus_identity_maxabs":maxabs(gram-np.eye(4)),
        "cross_side_packet_commutator_maxabs":maxabs(gram[:2,2:]),
        "stationary_in_vacuum_control":{"n":pack(np.zeros((2,2))),
            "m":pack(np.zeros((2,2))),"beta":pack(np.zeros(2)),
            "derivation":"linear combination of independent canonical incoming annihilators; all creation coefficients zero",
            "is_driven_output_prediction":False},
        "negative_controls":{
            "omit_electronic_channels_packet_gram_defect":maxabs(gram_noel-gram),
            "omit_odd_photon_channel_packet_gram_defect":maxabs(gram_noodd-gram),
            "wrong_positive_frequency_Bdag_spurious_right_n":pack(wrong_number),
            "wrong_positive_frequency_Bdag_spurious_right_n_trace":float(np.trace(wrong_number).real)},
        "limits":["fixed eta and BZ diagnostic, not continuous electronic spectral limit",
            "reference packet normalizer and finite quadrature interval reported without rescue rescaling",
            "no finite Xi to incoming-channel W_in constructed",
            "no electronic output rows or complete unitary scattering matrix",
            "no P1, quantum pump, driven optical n/m, squeezing, entanglement or domain contrast",
            "stationary output representation does not make finite internal photon coordinates asymptotic"],
        "source_sha256":{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in required}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--nk",type=int,default=96)
    args=ap.parse_args()
    if args.out.exists(): raise FileExistsError(f"Refuse overwrite: {args.out}")
    r=report(nk=args.nk)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as f:json.dump(r,f,indent=2)
    print(json.dumps({"status":r["status"],"out":str(args.out),
        "CCR":r["max_pointwise_all_photon_row_ccr_residual"],
        "packet_mass":r["packet"]["raw_mass"],"elapsed_seconds":r["elapsed_seconds"]},indent=2))

if __name__=="__main__":main()
