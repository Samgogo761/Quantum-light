#!/usr/bin/env python3
"""Narrow real-M0 finite-time static-shift soft-column mapping check.

Two fixed k probes, both Neel domains, existing R0 and original P1_a2(.8).
No BZ integration, no full-pulse run, no photon/Gaussian joint propagation.
Local current derivatives are compared, NOT assumed zero. This check proves
no continuum ground state, infrared limit, scattering, or strong output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_m0_classical import p1_a2_protocol, kappa_shift
from q8_m0_model import MODEL_ID, N_OCC, R0, build_geometry, h8, current_ops
from q8_m0_cjj import occup_empty_bases, x_from_bases
from q8_m0_response import d_ops

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results/q_line/continuum_ground_bridge_20261003/m0_soft_column.json"
KF = np.array([[0., 0.], [.23, .12]])
MU = -0.009010620372198276
DELTA = 2e-5


def adj(a):
    return np.swapaxes(a.conj(), -1, -2)


def ma(a):
    return float(np.max(np.abs(a)))


def projector(v):
    return v@adj(v)


def reconstruct(c, b, v):
    off = np.einsum("kic,kbcv,kjv->kbij", c, b, v.conj())
    return off+adj(off)


def shifted_projector_fd(frames):
    p = projector(frames[..., :N_OCC])
    return np.stack(((p[1]-p[2])/(2*DELTA), (p[3]-p[4])/(2*DELTA)), axis=1)


def block_rotation():
    """Fixed nontrivial complex unitary mixes whole occupied/empty subspaces."""
    n=np.arange(4)
    return np.exp(2j*np.pi*n[:, None]*n[None, :]/4)/2


def run_case(geo, prot, neel, nsteps, duration):
    kc=geo.k_cart(KF)
    shifts=DELTA*np.array([[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    kappas=kc[None, :, :]+shifts[:, None, :]
    energies, frames=np.linalg.eigh(h8(geo,R0,kappas,neel))
    if not np.all(np.sum(energies<MU,axis=-1)==N_OCC):
        raise ValueError("The fixed chemical potential must occupy four states at every shifted probe")
    v,c,en=occup_empty_bases(h8(geo,R0,kc,neel),MU)
    # Use exactly the base frame of the simultaneous shifted propagation.
    v,c=frames[0,:,:,:4],frames[0,:,:,4:]
    gap=energies[0,:,4:,None]-energies[0,:,None,:4]
    x0=x_from_bases(current_ops(geo,R0,kc,neel),v,c)
    b0=x0/gap[:,None,:,:]
    b=b0.copy()
    b_without_initial=np.zeros_like(b)
    eta0=reconstruct(c,b0,v)
    initial_fd_error=ma(eta0-shifted_projector_fd(frames))
    dt=duration/nsteps
    saves={0,nsteps//4,nsteps//2,3*nsteps//4,nsteps}
    records=[]
    qc=block_rotation()
    qv=qc.conj().T
    max_norm=0.
    min_path_gap=float(np.min(gap))
    for step in range(nsteps+1):
        if step in saves:
            t=step*dt
            kap=kc+kappa_shift(t,geo,prot,+1)
            v,c=frames[0,:,:,:4],frames[0,:,:,4:]
            eta=reconstruct(c,b,v)
            eta_bad=reconstruct(c,b_without_initial,v)
            eta_fd=shifted_projector_fd(frames)
            j=current_ops(geo,R0,kap,neel)
            x=x_from_bases(j,v,c)
            rho=projector(v)
            # Current output index a, static-shift input index b.
            param=2*np.real(np.einsum("kbcv,kacv->kab",b,x.conj()))
            param_trace=np.real(np.einsum("kbij,kaji->kab",eta,j))
            contact=-np.real(np.einsum("kij,kabji->kab",rho,d_ops(geo,R0,kap,neel)))
            current=param+contact
            current_bad=2*np.real(np.einsum("kbcv,kacv->kab",b_without_initial,x.conj()))+contact
            js=current_ops(geo,R0,kappas+kappa_shift(t,geo,prot,+1),neel)
            ps=projector(frames[...,:4])
            currents=np.real(np.einsum("skij,skaji->ska",ps,js))
            current_fd=np.stack(((currents[1]-currents[2])/(2*DELTA),
                                 (currents[3]-currents[4])/(2*DELTA)),axis=-1)
            dj_fd=np.stack(((js[1]-js[2])/(2*DELTA),
                            (js[3]-js[4])/(2*DELTA)),axis=2)
            contact_fd=np.real(np.einsum("kij,kabji->kab",rho,dj_fd))
            b_rot=np.einsum("dc,kbcv,ve->kbde",qc.conj().T,b,qv)
            eta_rot=reconstruct(c@qc,b_rot,v@qv)
            max_norm=max(max_norm,ma(adj(frames)@frames-np.eye(8)))
            records.append({"t":float(t),"time_over_optical_period":float(t/prot.period),
                "A_cart":kappa_shift(t,geo,prot,+1).tolist(),
                "eta_matrix_maxabs_error":ma(eta-eta_fd),
                "eta_matrix_maxabs_scale":ma(eta_fd),
                "eta_per_k_maxabs_errors":[ma(eta[i]-eta_fd[i]) for i in range(len(kc))],
                "current_derivative_maxabs_error":ma(current-current_fd),
                "current_derivative_maxabs_scale":ma(current_fd),
                "current_derivative_per_k_errors":[ma(current[i]-current_fd[i]) for i in range(len(kc))],
                "current_derivative_full_local":current.tolist(),
                "current_derivative_independent_FD":current_fd.tolist(),
                "contact_analytic_vs_FD_error":ma(contact-contact_fd),
                "band_current_vs_matrix_trace_error":ma(param-param_trace),
                "occupied_empty_block_rotation_covariance_error":ma(eta_rot-eta),
                "omit_b0_eta_error":ma(eta_bad-eta_fd),
                "omit_b0_current_error":ma(current_bad-current_fd)})
        if step==nsteps:
            break
        tmid=(step+.5)*dt
        midmom=kappas+kappa_shift(tmid,geo,prot,+1)
        e,z=np.linalg.eigh(h8(geo,R0,midmom,neel))
        min_path_gap=min(min_path_gap,float(np.min(e[:,:,4]-e[:,:,3])))
        # One shared 8x8 unitary propagates every v/c column of each trial.
        u=(z*np.exp(-1j*e*dt)[...,None,:])@adj(z)
        half=(z[0]*np.exp(-1j*e[0]*dt/2)[...,None,:])@adj(z[0])
        midframe=half@frames[0]
        xm=x_from_bases(current_ops(geo,R0,midmom[0],neel),midframe[:,:,:4],midframe[:,:,4:])
        increment=1j*dt*xm
        b+=increment
        b_without_initial+=increment
        frames=u@frames
    return {"neel":neel,"steps":nsteps,"dt":dt,"initial_gap_per_k":(energies[0,:,4]-energies[0,:,3]).tolist(),
            "initial_projector_derivative_vs_x_over_gap_maxabs_error":initial_fd_error,
            "minimum_gap_on_propagated_probe_paths":min_path_gap,
            "frame_unitarity_maxabs_error":max_norm,"samples":records,
            "max_eta_error":max(s["eta_matrix_maxabs_error"] for s in records),
            "max_current_error":max(s["current_derivative_maxabs_error"] for s in records)}


def report():
    start=time.perf_counter()
    geo=build_geometry()
    prot=p1_a2_protocol(.8)
    duration=.5*prot.period
    cases=[]
    checks={}
    for neel in [1,-1]:
        coarse=run_case(geo,prot,neel,2048,duration)
        fine=run_case(geo,prot,neel,4096,duration)
        er=fine["max_eta_error"]/coarse["max_eta_error"]
        jr=fine["max_current_error"]/coarse["max_current_error"]
        records=fine["samples"]
        prefix=f"N{neel:+d}_"
        checks.update({
            prefix+"initial_projector_mapping":fine["initial_projector_derivative_vs_x_over_gap_maxabs_error"]<2e-6,
            prefix+"full_matrix_tangent_agrees":fine["max_eta_error"]<2e-3,
            prefix+"full_local_current_derivative_agrees":fine["max_current_error"]<2e-2,
            prefix+"eta_error_decreases_with_dt":er<.35,
            prefix+"current_error_decreases_with_dt":jr<.35,
            prefix+"contact_matches_independent_difference":max(s["contact_analytic_vs_FD_error"] for s in records)<2e-6,
            prefix+"band_current_equals_full_matrix_trace":max(s["band_current_vs_matrix_trace_error"] for s in records)<2e-11,
            prefix+"subspace_rotation_covariance":max(s["occupied_empty_block_rotation_covariance_error"] for s in records)<2e-11,
            prefix+"unitarity":fine["frame_unitarity_maxabs_error"]<2e-10,
            prefix+"omit_b0_is_detectable_in_eta":max(s["omit_b0_eta_error"] for s in records[1:])>.01,
            prefix+"omit_b0_is_detectable_in_current":max(s["omit_b0_current_error"] for s in records[1:])>.01,
            prefix+"local_current_derivative_is_not_zero":max(s["current_derivative_maxabs_scale"] for s in records[1:])>.1,
            prefix+"probe_path_gap_avoids_zero":fine["minimum_gap_on_propagated_probe_paths"]>.5,
        })
        cases.append({"neel":neel,"eta_fine_over_coarse_error":er,
                      "current_fine_over_coarse_error":jr,"coarse":coarse,"fine":fine})
    modules=[Path(__file__),Path(__file__).with_name("q8_m0_model.py"),Path(__file__).with_name("q8_m0_classical.py"),
             Path(__file__).with_name("q8_m0_response.py"),Path(__file__).with_name("q8_m0_cjj.py")]
    return {"schema":"real_M0_finite_time_local_soft_column_mapping_v1",
        "scope":"Two fixed momenta, +/-N, original P1_a2(.8) restricted to its first half optical period; electronic mapping only.",
        "not_proven":["BZ-integrated Ward identity by this check", "pointwise vanishing current derivative",
                       "continuum photon ground state", "finite-frequency infrared expansion", "infinite-time tail",
                       "Gaussian joint propagation or strong-output existence"],
        "model_id":MODEL_ID,"parameters":asdict(R0),"protocol":asdict(prot),
        "protocol_id":prot.protocol_id,"mu":MU,"k_fractional":KF.tolist(),"k_cartesian":geo.k_cart(KF).tolist(),
        "k_weights":None,"not_a_BZ_quadrature":True,
        "time_window":[0.,duration],"window_over_optical_period":.5,"window_over_original_pulse":duration/prot.t_pulse,
        "directions":"both Cartesian x,y static shifts and both current outputs", "delta_k":DELTA,
        "conventions":{"H":"Existing embedded-orbital H8(k+A)","j":"-partial_k H", "contact":"partial_b j_a=-D_ab",
                       "x":"C(t)^dagger j V(t)","b_initial":"x_b(0)/(epsilon_c-epsilon_v)","b_ode":"d_t b_b=i*x_b(t)",
                       "eta":"C b_b V^dagger+V b_b^dagger C^dagger", "current":"2 Re sum_cv[b_b conjugate(x_a)]-Tr[rho D_ab]"},
        "method":"Complete 8x8 midpoint unitaries; full 4x4 occupied/empty blocks, no eigenvector finite differences or phase matching; independent +/-delta propagations shift both waveform and initial projector. Scalar-matrix b integral uses midpoint quadrature.",
        "negative_control":"Set b(0)=0 while keeping identical dynamics and contact.",
        "degenerate_subspace_control":"Reconstruct after independent fixed complex unitary changes of full occupied and empty frames, with b transformed covariantly.",
        "geometry_sources":geo.source_hashes,
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in modules},
        "cases":cases,"checks":checks,"passed":all(checks.values()),"wall_seconds":time.perf_counter()-start}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    result=report()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print(json.dumps({"out":str(args.out),"passed":result["passed"],"checks":result["checks"],
                      "wall_seconds":result["wall_seconds"],
                      "errors":[{"neel":c["neel"],"eta":c["fine"]["max_eta_error"],"current":c["fine"]["max_current_error"],
                                 "eta_ratio":c["eta_fine_over_coarse_error"],"current_ratio":c["current_fine_over_coarse_error"]} for c in result["cases"]]},indent=2))
    return 0 if result["passed"] else 1

if __name__=="__main__":
    raise SystemExit(main())
