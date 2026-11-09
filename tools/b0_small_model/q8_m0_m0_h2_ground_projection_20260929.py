#!/usr/bin/env python3
"""Real M0 vertices and explicitly FINITE photon-coordinate diagnostics.

Repaired 2026-10-02. This does NOT attach an asymptotic H2 output mode.
The old 2026-09-29 outputs are preserved and superseded by the repair report.
No P1 propagation, new state, contact subtraction or production cutoff.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from q8_m0_classical import k_grid, p1_a2_protocol
from q8_m0_cjj import occup_empty_bases, x_from_bases, window_fourier
from q8_m0_gsheet import gamma_self
from q8_m0_ground_covariance_operator_20260929 import StationaryGroundOperator
from q8_m0_k1_packet import normalize_phi, omega_grid, phi_unnormalized
from q8_m0_model import R0, build_geometry, current_ops, h8
from q8_m0_mode_io import eta2_ncell
from q8_m0_response import d_ops, mean_d_from_bases
from q8_m0_s1_kernel import parallel_perp_basis
from q8_m0_finite_photon_rows_20261002 import (
    finite_time_photon_rows, photon_packet_rows, quadrature_rows, mode_commutators)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/h2_attachment_repair_20261002/attachment.json"
MU = -0.009010620372198276
NK = 8
QUAD_ORDERS = (64, 128, 256)


def maxabs(a):
    a = np.asarray(a)
    return float(np.max(np.abs(a))) if a.size else 0.


def pack_electronic_channels(gaps_kcv, x_kacv, d_kab, weights):
    gaps, x, dd, w = (np.asarray(gaps_kcv, float), np.asarray(x_kacv, complex),
                     np.asarray(d_kab), np.asarray(weights, float))
    if (gaps.ndim != 3 or x.shape != (gaps.shape[0], 2, *gaps.shape[1:])
            or dd.shape != (gaps.shape[0], 2, 2) or w.shape != (gaps.shape[0],)):
        raise ValueError("Expected gaps[k,c,v], X[k,a,c,v], D[k,a,b], weights[k].")
    if not all(np.isfinite(a).all() for a in (gaps,x,dd,w)):
        raise ValueError("Nonfinite electronic data.")
    if np.any(gaps <= 0) or np.any(w < 0) or not np.isclose(w.sum(), 1., rtol=0., atol=1e-12):
        raise ValueError("Positive gaps and nonnegative normalized BZ weights required.")
    dmean0 = np.einsum("k,kab->ab", w, dd)
    if maxabs(np.imag(dmean0)) > 1e-11 or maxabs(dmean0-dmean0.T) > 1e-11:
        raise ValueError("Original contact is not real symmetric within roundoff.")
    dmean = np.real(dmean0)
    dmean = (dmean+dmean.T)/2  # roundoff only, never a static subtraction
    return {
        "gaps": gaps.reshape(-1),
        "weighted_vertices": np.einsum("k,kacv->akcv", np.sqrt(w), x).reshape(2,-1),
        "contact": dmean,
        "channel_table": np.indices(gaps.shape).reshape(3,-1).T,
    }


def electronic_channels(geo, neel, nk=NK):
    kf,kc,w = k_grid(geo,nk,shift=0.)
    vv,vc,en = occup_empty_bases(h8(geo,R0,kc,neel),MU)
    x = x_from_bases(current_ops(geo,R0,kc,neel),vv,vc)
    gaps = en[:,4:,None]-en[:,None,:4]
    result = pack_electronic_channels(gaps,x,
        mean_d_from_bases(d_ops(geo,R0,kc,neel),vv),w)
    return dict(result,k_frac=kf,weights=w,min_gap=float(gaps.min()),
                n_k=int(nk),n_el=int(gaps.size))


def h2_packets(prot):
    omega0,tp = float(prot.omega0),float(prot.t_pulse)
    omega = omega_grid(omega0,tp)
    raw = phi_unnormalized(omega,2*omega0,tp)
    phi,nrm = normalize_phi(omega,raw)
    dens = np.zeros_like(phi)
    mask = omega>0
    dens[mask]=phi[mask]/np.sqrt(omega[mask])
    irw=omega0*np.array([1e-6,2e-6,4e-6])
    irk=phi_unnormalized(irw,2*omega0,tp)/np.sqrt(irw)
    return {"omega":omega,"phi":phi,"raw_norm":nrm,"kernel":dens,
            "l2":float(np.trapezoid(abs(phi)**2,omega)),
            "kernel_l2":float(np.trapezoid(abs(dens)**2,omega)),
            "ir_slopes":np.diff(np.log(abs(irk)))/np.diff(np.log(irw)),
            "tp":tp,"omega0":omega0,
            "window_fourier_at_zero":complex(window_fourier(-2*omega0,tp))}


def photon_channels(geo, prot, order=8, panels=48):
    """Explicit finite two-port fixture on the EXISTING packet grid interval.

    Composite Gauss nodes resolve the packet. They are not a new production
    IR/UV choice and not a continuum convergence claim.
    """
    packets=h2_packets(prot)
    grid=packets["omega"]
    lo,hi=float(grid[grid>0][0]),float(grid[-1])
    xx,ww=leggauss(order)
    edges=np.linspace(lo,hi,panels+1)
    center=(edges[:-1]+edges[1:])/2
    half=np.diff(edges)/2
    omega=(center[:,None]+half[:,None]*xx).reshape(-1)
    weights=(half[:,None]*ww).reshape(-1)
    # Same analytic packet and same continuum normalization; no interpolation.
    phi=phi_unnormalized(omega,2*float(prot.omega0),float(prot.t_pulse))/packets["raw_norm"]
    coeff=np.sqrt(weights)*phi
    mass=float(np.vdot(coeff,coeff).real)
    if abs(mass-1)>1e-3:
        raise ValueError("Frequency set misses the approved packet; do not normalize empty tails.")
    e_par,e_perp=parallel_perp_basis(geo)
    eta=float(np.sqrt(eta2_ncell(geo)))
    count=omega.size
    vertices=np.zeros((2,4*count),complex)
    modes=np.zeros((2,4*count),complex)
    channels=[]
    for pol,ehat in enumerate((e_par,e_perp)):
        for side in range(2):
            block=(2*pol+side)*count
            sl=slice(block,block+count)
            vertices[:,sl]=eta*np.sqrt(weights/omega)[None,:]*ehat[:,None]
            if side==0:  # right side only; left/dark coordinates remain in h0
                modes[pol,sl]=coeff/np.sqrt(mass)
            channels.extend((pol,side,i) for i in range(count))
    return {"frequencies":np.tile(omega,4),"vertices":vertices,
            "selected_omega":omega,"frequency_weights":weights,
            "packet_coefficients":modes,"packet_raw_mass":mass,
            "finite_packet_rescale":float(1/np.sqrt(mass)),
            "channel_table":np.asarray(channels,int),"e_par":e_par,"e_perp":e_perp,
            "eta":eta,"gamma_self":float(gamma_self(geo)),
            "existing_interval":[lo,hi],"nodes_per_port_polarization":count,
            "note":"finite two-port quadrature fixture; NOT asymptotic photons or a continuum qualification"}


def detection_rows(*args, **kwargs):
    raise RuntimeError("The old static detection_rows is invalid. Use finite_time_photon_rows for a finite diagnostic; asymptotic H2 output remains unconstructed.")


def _ccr_record(op,ell):
    cc=mode_commutators(op,ell)
    return {"normal_max_residual":maxabs(cc["normal"]-np.eye(ell.shape[0])),
            "anomalous_max_residual":maxabs(cc["anomalous"])}


def assemble(neel, nk=NK, diagnostic_time=1.):
    geo=build_geometry()
    prot=p1_a2_protocol(.8)  # defines the old packet only; the electronic H is P1=0
    el=electronic_channels(geo,neel,nk)
    ph=photon_channels(geo,prot)
    op=StationaryGroundOperator.from_vertices(ph["frequencies"],el["gaps"],
        ph["vertices"],el["weighted_vertices"],el["contact"],
        provenance=f"Corrected M0 N={neel} nk={nk}; explicit finite two-port fixture; original D; no P1")
    cert=op.positivity_certificate()
    ell0=photon_packet_rows(op,ph["packet_coefficients"])
    ell=finite_time_photon_rows(op,ph["packet_coefficients"],diagnostic_time)
    rows=quadrature_rows(ell)
    projections={}
    if cert["status"]=="finite_positive_certified":
        for order in QUAD_ORDERS:
            projections[str(order)]=op.projected_ground_covariance(rows,order=order)
    changes=[]
    for a,b in zip(QUAD_ORDERS[:-1],QUAD_ORDERS[1:]):
        if str(a) in projections and str(b) in projections:
            absolute=maxabs(projections[str(b)]-projections[str(a)])
            scale=maxabs(projections[str(b)])
            changes.append({"orders":[a,b],"absolute_max":absolute,
                            "relative_maxnorm":absolute/scale if scale else None})
    q0=quadrature_rows(ell0)
    # A stationary ground remains stationary in the original Schrödinger coordinates.
    # This is deliberately different from demanding zero internal photon occupation.
    end_undemod=finite_time_photon_rows(op,
        ph["packet_coefficients"]*np.exp(-1j*op.frequencies[:op.n_photon]*diagnostic_time),
        diagnostic_time)
    gg0=op.projected_ground_covariance(q0,order=QUAD_ORDERS[-1])
    ggT=op.projected_ground_covariance(quadrature_rows(end_undemod),order=QUAD_ORDERS[-1])
    record={
        "neel":int(neel),"n_k_per_axis":nk,"n_electronic_channels":el["n_el"],
        "dimension":op.dimension,"certificate":cert,
        "electronic_channel_order":"k,c,v",
        "photon":{"nodes_per_port_polarization":ph["nodes_per_port_polarization"],
                  "ports":["right","left"],"polarizations":["parallel","perp"],
                  "coupling":"eta sqrt(weight/omega) e_pol",
                  "eta":ph["eta"],"existing_interval":ph["existing_interval"],
                  "packet_raw_mass":ph["packet_raw_mass"],
                  "finite_packet_rescale":ph["finite_packet_rescale"],"note":ph["note"]},
        "finite_time":{"time":diagnostic_time,"time_units":"hbar/t_h",
                       "fraction_of_packet_pulse":diagnostic_time/prot.t_pulse,
                       "purpose":"implementation diagnostic, NOT an 8-cycle detection or far field"},
        "ccr_initial":_ccr_record(op,ell0),"ccr_finite_time":_ccr_record(op,ell),
        "finite_ground_projected_covariance":{k:v.tolist() for k,v in projections.items()},
        "virtual_frequency_quadrature_changes":changes,
        "stationary_ground_fixed_coordinate_residual":maxabs(ggT-gg0),
        "asymptotic_output":"not_constructed","P1_not_run":True,"continuum_claim":False,
        "magnetic_output_claim":False,"photon_yield_squeezing_entanglement":"not_evaluated",
    }
    return record,op,rows,projections,cert


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=OUT)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refuse to replace results: {args.out}")
    records=[assemble(n)[0] for n in (+1,-1)]
    payload={"status":"finite_implementation_repaired_asymptotic_output_not_constructed",
             "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "domains":records,"old_results_preserved":True,"P1_not_run":True,
             "state":"same finite interacting quadratic ground; no new preparation"}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"status":payload["status"],
        "domains":[{"neel":r["neel"],"dimension":r["dimension"],
                    "packet_mass":r["photon"]["packet_raw_mass"],
                    "ccr":r["ccr_finite_time"],
                    "quadrature":r["virtual_frequency_quadrature_changes"],
                    "stationarity":r["stationary_ground_fixed_coordinate_residual"]} for r in records]},indent=2))


if __name__=="__main__":
    main()
