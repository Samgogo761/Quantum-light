#!/usr/bin/env python3
"""Regression for frozen photon normalization and H2 packet resolution."""
import json
import numpy as np
from q8_m0_model import build_geometry
from q8_m0_classical import p1_a2_protocol
from q8_m0_m0_h2_ground_projection_20260929 import photon_channels, h2_packets
from q8_m0_k1_packet import phi_unnormalized


def maxabs(x):
    return float(np.max(np.abs(x)))


def report():
    geo=build_geometry()
    prot=p1_a2_protocol(.8)
    ph=photon_channels(geo,prot)
    pk=h2_packets(prot)
    n=ph["nodes_per_port_polarization"]
    ev=ph["vertices"]
    omega,w=ph["selected_omega"],ph["frequency_weights"]
    expected=ph["eta"]*np.sqrt(w/omega)
    epar,eperp=ph["e_par"],ph["e_perp"]
    coeff=ph["packet_coefficients"]
    r={}
    for pol,e in enumerate((epar,eperp)):
        a,b=2*pol*n,(2*pol+1)*n
        left,right=ev[:,a:a+n],ev[:,b:b+n]
        r[f"pol{pol}_coupling_normalization"]=maxabs(e@left-expected)
        r[f"pol{pol}_two_ports_identical"]=maxabs(left-right)
        r[f"pol{pol}_odd_dark"]=maxabs((left-right)/np.sqrt(2))
        r[f"pol{pol}_even_sqrt2"]=maxabs((left+right)/np.sqrt(2)-np.sqrt(2)*expected[None,:]*e[:,None])
        r[f"pol{pol}_left_packet_excluded"]=maxabs(coeff[pol,b:b+n])
        raw=np.sqrt(w)*phi_unnormalized(omega,2*prot.omega0,prot.t_pulse)/pk["raw_norm"]
        r[f"pol{pol}_packet_normalization_separate_from_coupling"]=maxabs(coeff[pol,a:a+n]-raw/np.sqrt(ph["packet_raw_mass"]))
    r["packet_gram"]=maxabs(coeff@coeff.conj().T-np.eye(2))
    r["polarization_orthogonality"]=abs(float(epar@eperp))
    r["two_modes_not_identical"]=float(np.linalg.norm(coeff[0]-coeff[1]))
    # Refine ONLY the finite frequency quadrature to verify packet resolution.
    fine=photon_channels(geo,prot,order=16)
    r["raw_packet_mass_change"]=abs(ph["packet_raw_mass"]-fine["packet_raw_mass"])
    r["raw_packet_missing_mass"]=abs(ph["packet_raw_mass"]-1)
    # The exact old four frequencies miss virtually the entire H2 packet.
    grid=pk["omega"][pk["omega"]>0]
    old=grid[np.linspace(0,grid.size-1,4).round().astype(int)]
    oldmass=float((grid[1]-grid[0])*np.sum(abs(phi_unnormalized(old,2*prot.omega0,prot.t_pulse)/pk["raw_norm"])**2))
    checks={k:(v<1e-8) for k,v in r.items() if k!="two_modes_not_identical"}
    checks["two_modes_not_identical"]=r["two_modes_not_identical"]>1
    checks["old_packet_negative_control_detected"]=oldmass<1e-8
    return {"all_checks_pass":bool(all(checks.values())),"checks":checks,"residuals":r,
            "old_four_node_packet_mass":oldmass,"new_packet_mass":ph["packet_raw_mass"],
            "finite_fixture_only":True,"production_cutoff_selected":False}


if __name__=="__main__":
    result=report()
    print(json.dumps(result,indent=2))
    if not result["all_checks_pass"]:
        raise SystemExit(1)
