#!/usr/bin/env python3
"""Real M0 selected-column finite-time check; no optical moments or BZ claim.

Two fixed k points, +/-N, only the FIRST HALF CYCLE of the existing P1.
The photon frequencies label incoming scattering columns, not a photon box.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
from scipy.integrate import solve_ivp
from q8_m0_classical import kappa_shift,p1_a2_protocol
from q8_m0_cjj import occup_empty_bases,x_from_bases
from q8_m0_model import R0,build_geometry,h8,h8_and_current,current_ops
from q8_m0_response import d_ops,mean_d_from_bases
from q8_m0_gsheet import gamma_self
from q8_m0_continuum_columns_20261004 import (
    ColumnState,stationary_photon_columns,midpoint_step,increment_source)
REPO=Path(__file__).resolve().parents[2]
OUT=REPO/"results/q_line/continuum_columns_20261004/m0_checks.json"
KF=np.array([[0.,0.],[.23,.12]])
WEIGHTS=np.array([.5,.5])
FREQ=np.array([.025,.10])
MU=-.009010620372198276
NTPCS=(1024,2048)


def maxabs(a):
    return float(np.max(np.abs(a))) if np.size(a) else 0.


def errors(a,b):
    diff=maxabs(np.asarray(a)-np.asarray(b))
    scale=maxabs(b)
    return {"max_absolute":diff,"reference_max":scale,
            "max_relative":diff/max(scale,1e-30)}


def vertices_from_frame(geo,kc,prot,neel,t,frame):
    kap=kc+kappa_shift(t,geo,prot,+1)
    j=current_ops(geo,R0,kap,neel)
    vv,vc=frame[:,:,:4],frame[:,:,4:]
    x=x_from_bases(j,vv,vc)
    u=np.einsum("k,kacv->akcv",np.sqrt(WEIGHTS),x).reshape(2,-1)
    d=np.einsum("k,kab->ab",WEIGHTS,mean_d_from_bases(d_ops(geo,R0,kap,neel),vv))
    return u,d


def oracle_current(state,u,d):
    b,bd,a=state
    return u.conj()@b+u@bd-d@a


def unpack(packed,nel):
    return packed[:nel],packed[nel:2*nel],packed[2*nel:]


def stationary_checks(ref):
    g,u,d=ref.gaps,ref.u0,ref.D0
    b,bd,a=ref.initial.b,ref.initial.bdag,ref.initial.A
    omega=ref.omega[None,:]
    def defect(bb,dd,aa):
        ii=u.conj()@bb+u@dd-d@aa
        return max(maxabs(1j*(g[:,None]-omega)*bb-1j*u.T@aa),
            maxabs(-1j*(g[:,None]+omega)*dd+1j*u.conj().T@aa),
            maxabs(-1j*omega*aa-(-1j*omega*ref.free_A0+ref.gamma*ii)))
    static_kernel=d-2*((u.conj()/g)@u.T).real
    independent=[]
    for i,w in enumerate(FREQ):
        c=np.einsum("ap,bp->abp",u.conj(),u)
        k=d+np.einsum("abp,p->ab",c,1/(w-g))-np.einsum("abp,p->ab",c.conj(),1/(w+g))
        independent.append(maxabs(k-ref.kernels[i]))
    return {
        "stationary_equation_max_residual":defect(b,bd,a),
        "kernel_independent_spectral_formula_residual":max(independent),
        "self_field_initial_relation":maxabs(a-ref.free_A0-
            1j*ref.gamma/omega*oracle_current((b,bd,a),u,d)),
        "raw_static_kernel_eigenvalues":np.linalg.eigvalsh(static_kernel).tolist(),
        "gap_min":float(g.min()),"gap_max":float(g.max()),
        "selected_frequency_distance_from_discrete_poles":
            {str(w):float(np.min(abs(g-w))) for w in FREQ},
        "negative_controls":{
            "zero_initial_self_field_equation_residual":defect(b,bd,ref.free_A0),
            "zero_initial_electronic_dressing_equation_residual":defect(np.zeros_like(b),np.zeros_like(bd),a),
            "incorrect_bdag_equals_b_conjugate_residual":defect(b,b.conj(),a)}}


def midpoint_run(geo,kc,prot,neel,frame0,ref,ntpc):
    started=time.perf_counter()
    h=prot.period/ntpc
    nstep=ntpc//2
    indexes=np.linspace(0,nstep,5,dtype=int)
    frame=frame0.copy()
    direct=ref.initial.copy()
    delta=ColumnState.zeros(*direct.b.shape)
    snapframes=[];snapdirect=[];snapcombined=[];snapdelta=[]
    currents_direct=[];currents_combined=[]
    def record(step):
        t=step*h
        combined=ref.state(t).packed()+delta.packed()
        u,d=vertices_from_frame(geo,kc,prot,neel,t,frame)
        snapframes.append(frame.copy())
        snapdirect.append(direct.packed())
        snapcombined.append(combined)
        snapdelta.append(delta.packed())
        currents_direct.append(oracle_current(unpack(direct.packed(),len(ref.gaps)),u,d))
        currents_combined.append(oracle_current(unpack(combined,len(ref.gaps)),u,d))
    record(0)
    for step in range(nstep):
        t=step*h
        # True midpoint frame, no averaging of fast X endpoint phases.
        for which,when in enumerate((t+h/4,t+3*h/4)):
            ham=h8(geo,R0,kc+kappa_shift(when,geo,prot,+1),neel)
            en,vec=np.linalg.eigh(ham)
            evol=(vec*np.exp(-1j*en*h/2)[:,None,:])@vec.conj().swapaxes(1,2)
            frame=evol@frame
            if which==0:
                u,d=vertices_from_frame(geo,kc,prot,neel,t+h/2,frame)
                direct=midpoint_step(direct,u,d,h,ref.gamma,
                    free_A_increment=ref.free_A(t+h)-ref.free_A(t))
                src=increment_source(t+h/2,u,d,ref)
                delta=midpoint_step(delta,u,d,h,ref.gamma,source_mid=src)
        if step+1 in indexes:
            record(step+1)
    return {"ntpc":ntpc,"dt":h,"n_steps":nstep,
            "wall_seconds":time.perf_counter()-started,
            "times":indexes*h,"frames":np.asarray(snapframes),
            "direct":np.asarray(snapdirect),"combined":np.asarray(snapcombined),
            "delta":np.asarray(snapdelta),"current_direct":np.asarray(currents_direct),
            "current_combined":np.asarray(currents_combined)}


def independent_dop853(geo,kc,prot,neel,frame0,ref):
    """Integrate frames AND direct columns; never call library rhs/current."""
    started=time.perf_counter()
    nel=len(ref.gaps);ncol=len(ref.omega)
    nframe=frame0.size
    packed_shape=(2*nel+2,ncol)
    y0=np.r_[frame0.reshape(-1),ref.initial.packed().reshape(-1)]
    def ode(t,y):
        frame=y[:nframe].reshape(frame0.shape)
        state=y[nframe:].reshape(packed_shape)
        b,bd,a=state[:nel],state[nel:2*nel],state[2*nel:]
        kap=kc+kappa_shift(t,geo,prot,+1)
        ham,j=h8_and_current(geo,R0,kap,neel)
        vv,vc=frame[:,:,:4],frame[:,:,4:]
        x=np.einsum("kic,kaij,kjv->kacv",vc.conj(),j,vv)
        u=np.einsum("k,kacv->akcv",np.sqrt(WEIGHTS),x).reshape(2,-1)
        dop=d_ops(geo,R0,kap,neel)
        d=np.einsum("k,kiv,kabij,kjv->ab",WEIGHTS,vv.conj(),dop,vv).real
        ii=u.conj()@b+u@bd-d@a
        df=-1j*ham@frame
        db=1j*u.T@a
        dbdag=-1j*u.conj().T@a
        da=(-1j*ref.omega[None,:]*ref.free_A0*
            np.exp(-1j*ref.omega[None,:]*t)+ref.gamma*ii)
        return np.r_[df.reshape(-1),np.concatenate((db,dbdag,da),axis=0).reshape(-1)]
    times=np.linspace(0,prot.period/2,5)
    sol=solve_ivp(ode,(0,prot.period/2),y0,method="DOP853",t_eval=times,
                  rtol=2e-11,atol=2e-13,max_step=prot.period/128)
    if not sol.success:raise RuntimeError(sol.message)
    frames=sol.y[:nframe].T.reshape((5,*frame0.shape))
    states=sol.y[nframe:].T.reshape((5,*packed_shape))
    currents=[]
    for t,f,s in zip(times,frames,states):
        u,d=vertices_from_frame(geo,kc,prot,neel,t,f)
        currents.append(oracle_current(unpack(s,nel),u,d))
    return {"times":times,"frames":frames,"state":states,"current":np.asarray(currents),
            "nfev":sol.nfev,"wall_seconds":time.perf_counter()-started,
            "rtol":2e-11,"atol":2e-13,"independent_rhs":True}


def report():
    t0=time.perf_counter()
    geo=build_geometry();kc=geo.k_cart(KF);prot=p1_a2_protocol(.8)
    gamma=float(gamma_self(geo))
    results={};arrays={}
    for neel in (1,-1):
        vv,vc,en=occup_empty_bases(h8(geo,R0,kc,neel),MU)
        frame=np.concatenate((vv,vc),axis=2)
        gaps=(en[:,4:,None]-en[:,None,:4]).reshape(-1)
        u0,d0=vertices_from_frame(geo,kc,prot,neel,0.,frame)
        ref=stationary_photon_columns(gaps,u0,d0,gamma,FREQ)
        check=stationary_checks(ref)
        print(f"N={neel:+d}: starting independent DOP853",flush=True)
        oracle=independent_dop853(geo,kc,prot,neel,frame,ref)
        print(f"N={neel:+d}: DOP853 {oracle['wall_seconds']:.2f}s, nfev={oracle['nfev']}",flush=True)
        runs=[]
        prefix="plus" if neel==1 else "minus"
        for ntpc in NTPCS:
            run=midpoint_run(geo,kc,prot,neel,frame,ref,ntpc)
            comparisons={
                "direct_vs_DOP853":errors(run["direct"],oracle["state"]),
                "reference_plus_increment_vs_DOP853":errors(run["combined"],oracle["state"]),
                "reference_plus_increment_vs_direct":errors(run["combined"],run["direct"]),
                "current_direct_vs_DOP853":errors(run["current_direct"],oracle["current"]),
                "current_combined_vs_DOP853":errors(run["current_combined"],oracle["current"]),
                "electronic_frames_vs_DOP853":errors(run["frames"],oracle["frames"]),
            }
            runs.append({"ntpc":ntpc,"dt":run["dt"],"n_steps":run["n_steps"],
                         "wall_seconds":run["wall_seconds"],"comparisons":comparisons})
            for key in ("frames","direct","combined","delta","current_direct","current_combined"):
                arrays[f"{prefix}_{ntpc}_{key}"]=run[key]
            print(f"N={neel:+d}: ntpc={ntpc} {run['wall_seconds']:.2f}s, direct relative {comparisons['direct_vs_DOP853']['max_relative']:.3g}",flush=True)
        refinement={}
        for name in runs[0]["comparisons"]:
            coarse=runs[0]["comparisons"][name]["max_absolute"]
            fine=runs[1]["comparisons"][name]["max_absolute"]
            refinement[name]={"coarse_over_fine":coarse/max(fine,1e-30),
                             "decreased":bool(fine<coarse or coarse<1e-10)}
        passed=(check["stationary_equation_max_residual"]<1e-10 and
                all(v["decreased"] for v in refinement.values()) and
                runs[-1]["comparisons"]["direct_vs_DOP853"]["max_relative"]<.01 and
                runs[-1]["comparisons"]["reference_plus_increment_vs_DOP853"]["max_relative"]<.01)
        for key in ("frames","state","current"):
            arrays[f"{prefix}_DOP853_{key}"]=oracle[key]
        results[str(neel)]={
            "status":"finite_selected_column_check_passed" if passed else "finite_selected_column_check_not_qualified",
            "stationary":check,"incoming_columns":list(ref.labels),
            "oracle":{k:v for k,v in oracle.items() if k not in ("times","frames","state","current")},
            "levels":runs,"refinement":refinement}
        if time.perf_counter()-t0>120:
            raise RuntimeError("Two-minute scope exceeded; stop and report partial console evidence.")
    arrays["times"]=oracle["times"]
    passed=all(v["status"]=="finite_selected_column_check_passed" for v in results.values())
    payload={
        "status":"real_M0_finite_selected_columns_verified" if passed else "real_M0_finite_selected_columns_not_qualified",
        "date":"2026-10-04","wall_seconds":time.perf_counter()-t0,
        "k_points_fractional":KF.tolist(),"weights":WEIGHTS.tolist(),"not_BZ":True,
        "protocol":prot.__dict__,"time_interval":[0.,prot.period/2],
        "interval_meaning":"first half cycle of the UNCHANGED eight-cycle P1 envelope; not a shortened pulse",
        "gamma":gamma,"frequencies":FREQ.tolist(),
        "input_column_normalization":"each frequency, L/R x Cartesian x/y, eta/sqrt(omega); eta=sqrt(gamma/(2*pi))",
        "source_scope":"selected incoming photon particular columns, independent b and bdag; original raw electronic channels and D",
        "detector_scope":"no H2 packet integration; omega=.1 is one selected frequency, not an output mode",
        "ground_state_complete":False,"P1_full_pulse":False,
        "optical_moments_computed":False,"production":False,
        "domain_difference_claim":False,
        "domains":results,
        "limitations":["two k points are a finite diagnostic, not BZ or stability qualification",
                       "selected photon columns omit other incoming electronic/dark/point contributions",
                       "DOP853 is an independent numerical reference, not a proof of continuous-time error",
                       "no initial-state change, no static K(0) subtraction, no photon cutoff or extra damping",
                       "finite early interval does not establish post-pulse tails or driven optical output"]}
    return payload,arrays


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=OUT)
    args=ap.parse_args()
    data_path=args.out.with_name("m0_snapshots.npz")
    if args.out.exists() or data_path.exists():raise FileExistsError("Refuse overwrite")
    payload,arrays=report()
    source=Path(__file__)
    required=[source,source.with_name("q8_m0_continuum_columns_20261004.py"),
              source.with_name("q8_m0_model.py"),source.with_name("q8_m0_classical.py")]
    payload["source_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in required}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with data_path.open("xb") as f:np.savez_compressed(f,**arrays)
    payload["snapshots"]={"path":str(data_path.relative_to(REPO)),
                         "sha256":hashlib.sha256(data_path.read_bytes()).hexdigest()}
    with args.out.open("x",encoding="utf-8") as f:json.dump(payload,f,indent=2)
    print(json.dumps({"status":payload["status"],"wall_seconds":payload["wall_seconds"],
                      "out":str(args.out)},indent=2),flush=True)


if __name__=="__main__":
    main()
