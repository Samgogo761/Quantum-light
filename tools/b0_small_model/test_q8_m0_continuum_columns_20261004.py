#!/usr/bin/env python3
"""Independent small tests for continuous-photon selected-column equations.

There is no photon oscillator discretization here. Two original electronic
coordinates couple to a two-component Maxwell surface field. B and Bdag
coefficients belonging to one incoming annihilator are independent complex
arrays. These finite mathematical tests establish no complete incoming ground,
photon n/m/V, infrared limit, or asymptotic output. No M0 propagation is run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
sys.path.insert(0,str(Path(__file__).resolve().parent))
import q8_m0_continuum_columns_20261004 as lib

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/q_line/continuum_columns_20261004/math_checks.json"
GAPS=np.array([.9,1.7])
U0=np.array([[.22+.08j,-.13+.11j],[.07-.19j,.26+.03j]])
GAMMA=.24
FREQ=np.array([.43,1.21])
# Deliberately retain a nonzero real static residual, including off-diagonal D.
D0=((U0.conj()/GAPS)@U0.T+(U0/GAPS)@U0.conj().T).real+np.array([[.04,.025],[.025,.03]])
DURATION=3.7


def ma(x):
    return float(np.max(abs(x)))


def independent_generator(u,D):
    """Full original-electron generator, independent of the library Schur step."""
    n=u.shape[1]
    g=np.zeros((2*n+2,2*n+2),complex)
    g[:n,2*n:]=1j*u.T
    g[n:2*n,2*n:]=-1j*u.conj().T
    g[2*n:,:n]=GAMMA*u.conj()
    g[2*n:,n:2*n]=GAMMA*u
    g[2*n:,2*n:]=-GAMMA*D
    return g


def independent_kernel(w):
    # Sum each electron explicitly, avoiding the library's vectorized product.
    k=D0.astype(complex).copy()
    for p,gap in enumerate(GAPS):
        v=U0[:,p]
        k-=np.outer(v.conj(),v)/(gap-w)
        k-=np.outer(v,v.conj())/(gap+w)
    return k


def independent_initial():
    eta=np.sqrt(GAMMA/(2*np.pi))
    free=np.hstack([eta/np.sqrt(w)*np.hstack((np.eye(2),np.eye(2))) for w in FREQ])
    y=np.zeros((6,8),complex)
    for i,w in enumerate(FREQ):
        sl=slice(4*i,4*i+4)
        y[4:,sl]=np.linalg.solve(np.eye(2)+1j*GAMMA*independent_kernel(w)/w,free[:,sl])
        for p,gap in enumerate(GAPS):
            y[p,sl]=U0[:,p]@y[4:,sl]/(gap-w)
            y[p+2,sl]=U0[:,p].conj()@y[4:,sl]/(gap+w)
    return y,free


def independent_stationary(t,initial):
    w=np.repeat(FREQ,4)
    e=np.exp(-1j*w*t)[None,:]
    return np.vstack((initial[:2]*np.exp(1j*GAPS*t)[:,None]*e,
                      initial[2:4]*np.exp(-1j*GAPS*t)[:,None]*e,initial[4:]*e))


def stationary_matrixexp(t,initial,free):
    """Constant lab-frame 8x8 matrix plus incoming forcing oscillator phase.

    The last two rows only encode analytic monochromatic forcing; they are not
    retained photon oscillators, a bath box, or a continuum quadrature.
    """
    result=np.zeros_like(initial)
    for i,w in enumerate(FREQ):
        sl=slice(4*i,4*i+4)
        m=np.zeros((8,8),complex)
        m[:6,:6]=independent_generator(U0,D0)
        m[:2,:2]=-1j*np.diag(GAPS)
        m[2:4,2:4]=1j*np.diag(GAPS)
        m[4:6,6:]=-1j*w*np.eye(2)
        m[6:,6:]=-1j*w*np.eye(2)
        lab=expm(m*t)@np.vstack((initial[:,sl],free[:,sl]))
        result[:2,sl]=lab[:2]*np.exp(1j*GAPS*t)[:,None]
        result[2:4,sl]=lab[2:4]*np.exp(-1j*GAPS*t)[:,None]
        result[4:,sl]=lab[4:6]
    return result


def manufactured_coefficients(t):
    env=np.sin(np.pi*t/DURATION)**2
    change=np.array([[.08-.04j,.03+.05j],[-.06+.02j,.04-.03j]])
    u=(U0+env*change)*np.exp(1j*GAPS*t)[None,:]
    D=D0+.07*env*np.array([[.4,-.3],[-.3,.6]])
    return u,D


def independent_dynamic_reference(initial,free):
    omega=np.repeat(FREQ,4)
    def derivative(t,flat):
        y=flat.reshape(initial.shape)
        u,D=manufactured_coefficients(t)
        out=independent_generator(u,D)@y
        out[4:]+=-1j*omega[None,:]*free*np.exp(-1j*omega*t)[None,:]
        return out.ravel()
    sol=solve_ivp(derivative,(0.,DURATION),initial.ravel(),method="DOP853",
                  rtol=2e-12,atol=2e-14,dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def midpoint_dynamic(nsteps,reference,independent,initial):
    direct=lib.ColumnState.unpack(initial,2)
    delta=lib.ColumnState.zeros(2,initial.shape[1])
    dt=DURATION/nsteps
    samples=[]
    for i in range(nsteps+1):
        t=i*dt
        if i%(nsteps//4)==0:
            truth=independent.sol(t).reshape(initial.shape)
            separate=delta.packed()+reference.state(t).packed()
            u,D=manufactured_coefficients(t)
            direct_current=u.conj()@direct.b+u@direct.bdag-D@direct.A
            ref=reference.state(t)
            ur=U0*np.exp(1j*GAPS*t)[None,:]
            ref_current=ur.conj()@ref.b+ur@ref.bdag-D0@ref.A
            delta_current=lib.increment_current(t,delta,u,D,reference)
            truth_current=u.conj()@truth[:2]+u@truth[2:4]-D@truth[4:]
            samples.append({"t":t,"direct_vs_DOP853":ma(direct.packed()-truth),
                "reference_plus_delta_vs_DOP853":ma(separate-truth),
                "direct_vs_reference_plus_delta":ma(direct.packed()-separate),
                "increment_current_vs_DOP853":ma(ref_current+delta_current-truth_current),
                "direct_current_vs_DOP853":ma(direct_current-truth_current),
                "odd_surface_column_maxabs":ma(direct.packed()[:,[0,1,4,5]]-direct.packed()[:,[2,3,6,7]])})
        if i==nsteps:
            break
        mid=t+dt/2
        u,D=manufactured_coefficients(mid)
        direct=lib.midpoint_step(direct,u,D,dt,GAMMA,
                                 free_A_increment=reference.free_A(t+dt)-reference.free_A(t))
        src=lib.increment_source(mid,u,D,reference)
        delta=lib.midpoint_step(delta,u,D,dt,GAMMA,source_mid=src)
    return {"steps":nsteps,"dt":dt,"samples":samples,
            "max_direct_error":max(s["direct_vs_DOP853"] for s in samples),
            "max_delta_error":max(s["reference_plus_delta_vs_DOP853"] for s in samples)}


def run_checks():
    started=time.perf_counter()
    ref=lib.stationary_photon_columns(GAPS,U0,D0,GAMMA,FREQ)
    initial,free=independent_initial()
    metrics={"initial_columns_vs_independent":ma(ref.initial.packed()-initial),
             "free_normalization_vs_independent":ma(ref.free_A0-free),
             "kernel_vs_explicit_electron_sum":max(ma(lib.real_frequency_kernel(GAPS,U0,D0,w)-independent_kernel(w)) for w in FREQ),
             "D_offdiagonal_magnitude":abs(float(D0[0,1])),
             "static_contact_residual_retained":ma(D0-((U0.conj()/GAPS)@U0.T+(U0/GAPS)@U0.conj().T)),
             "annihilator_bdag_is_not_conjugate_b":ma(ref.initial.bdag-ref.initial.b.conj())}
    stationary=[]
    for t in [0.,.31,1.4,DURATION]:
        y=independent_stationary(t,initial)
        u=U0*np.exp(1j*GAPS*t)[None,:]
        omega=np.repeat(FREQ,4)
        expected=np.vstack((1j*(GAPS[:,None]-omega)*y[:2],
                           -1j*(GAPS[:,None]+omega)*y[2:4],-1j*omega*y[4:]))
        force=-1j*omega[None,:]*free*np.exp(-1j*omega*t)[None,:]
        actual=independent_generator(u,D0)@y
        actual[4:]+=force
        state=lib.ColumnState.unpack(y,2)
        stationary.append({"t":t,"library_vs_analytic":ma(ref.state(t).packed()-y),
                           "augmented_matrixexp_vs_analytic":ma(stationary_matrixexp(t,initial,free)-y),
                           "analytic_derivative_vs_independent_rhs":ma(actual-expected),
                           "library_rhs_vs_independent":ma(lib.rhs(state,u,D0,GAMMA,force).packed()-actual)})
    # Generic complex-linear state: no bdag=conj(b) assumption even transiently.
    rng=np.random.default_rng(20261004)
    y=rng.normal(size=(6,3))+1j*rng.normal(size=(6,3))
    f=rng.normal(size=(6,3))+1j*rng.normal(size=(6,3))
    dt=.137
    gen=independent_generator(U0,D0)
    dense=np.linalg.solve(np.eye(6)-dt*gen/2,(np.eye(6)+dt*gen/2)@y+dt*f)
    candidate=lib.midpoint_step(lib.ColumnState.unpack(y,2),U0,D0,dt,GAMMA,
        source_mid=lib.ColumnState.unpack(f,2))
    metrics["Schur_midpoint_vs_full_dense_Cayley"]=ma(candidate.packed()-dense)
    # Check delta forcing against independently formed full matrix difference.
    t=.93
    u,D=manufactured_coefficients(t)
    ur=U0*np.exp(1j*GAPS*t)[None,:]
    yr=independent_stationary(t,initial)
    exact_src=(independent_generator(u,D)-independent_generator(ur,D0))@yr
    metrics["increment_source_vs_independent_matrix_difference"]=ma(lib.increment_source(t,u,D,ref).packed()-exact_src)
    ode=independent_dynamic_reference(initial,free)
    coarse=midpoint_dynamic(64,ref,ode,initial)
    fine=midpoint_dynamic(128,ref,ode,initial)
    metrics["direct_refinement_ratio"]=fine["max_direct_error"]/coarse["max_direct_error"]
    metrics["delta_refinement_ratio"]=fine["max_delta_error"]/coarse["max_delta_error"]
    # Wrong current conjugation has a nonzero residual on the known exact wave.
    omega=np.repeat(FREQ,4)
    correct_A_derivative=-1j*omega[None,:]*initial[4:]
    wrong_current=U0@initial[:2]+U0.conj()@initial[2:4]-D0@initial[4:]
    wrong_A_derivative=-1j*omega[None,:]*free+GAMMA*wrong_current
    bare=np.vstack((np.zeros((4,8),complex),free))
    bare_rhs=independent_generator(U0,D0)@bare
    bare_rhs[4:]+=-1j*omega[None,:]*free
    bare_monochromatic=np.vstack((np.zeros((4,8),complex),-1j*omega[None,:]*free))
    negative={"wrong_current_conjugation_stationary_residual":ma(wrong_A_derivative-correct_A_derivative),
              "zero_dressing_stationary_residual":ma(bare_rhs-bare_monochromatic),
              "forced_bdag_conjugate_b_error":ma(initial[2:4]-initial[:2].conj())}
    # Add an explicit uncoupled original electronic coordinate; preserve it.
    udark=np.column_stack((U0,np.zeros(2,complex)))
    rd=lib.stationary_photon_columns(np.r_[GAPS,2.3],udark,D0,GAMMA,FREQ)
    ds=rd.initial.copy()
    ds.b[2]=.3+.2j
    ds.bdag[2]=-.1+.4j
    dn=lib.midpoint_step(ds,udark,D0,.07,GAMMA)
    metrics["dark_raw_electronic_dimension"]=int(dn.b.shape[0])
    metrics["dark_uncoupled_rows_unchanged"]=max(ma(dn.b[2]-ds.b[2]),ma(dn.bdag[2]-ds.bdag[2]))
    metrics["dark_zero_column_bright_solution_unchanged"]=ma(rd.initial.A-ref.initial.A)
    rejected={}
    for name,call in [
        ("exact_electronic_pole",lambda:lib.real_frequency_kernel(GAPS,U0,D0,GAPS[0])),
        ("complex_contact",lambda:lib.real_frequency_kernel(GAPS,U0,D0+1j*np.eye(2)*.01,FREQ[0])),
        ("negative_frequency",lambda:lib.stationary_photon_columns(GAPS,U0,D0,GAMMA,[-.1])),
        ("NaN_frequency",lambda:lib.stationary_photon_columns(GAPS,U0,D0,GAMMA,[np.nan])),
        ("NaN_polarization",lambda:lib.stationary_photon_columns(GAPS,U0,D0,GAMMA,FREQ,polarizations=np.full((2,2),np.nan))),
        ("ground_moments_unavailable",lambda:lib.assemble_ground_or_photon_moments())]:
        try:
            call()
            rejected[name]=False
        except (ValueError,NotImplementedError):
            rejected[name]=True
    checks={
        "stationary_initial_matches_independent":metrics["initial_columns_vs_independent"]<2e-13,
        "free_normalization_matches_independent":metrics["free_normalization_vs_independent"]<2e-14,
        "kernel_matches_explicit_sum":metrics["kernel_vs_explicit_electron_sum"]<2e-13,
        "contact_has_offdiagonal_and_nonzero_static_residual":metrics["D_offdiagonal_magnitude"]>.01 and metrics["static_contact_residual_retained"]>.01,
        "b_and_bdag_are_independent_complex_columns":metrics["annihilator_bdag_is_not_conjugate_b"]>.01,
        "stationary_analytic_and_matrixexp_agree":max(s["augmented_matrixexp_vs_analytic"] for s in stationary)<3e-12,
        "stationary_library_matches_analytic":max(s["library_vs_analytic"] for s in stationary)<3e-13,
        "stationary_exact_equations_hold":max(s["analytic_derivative_vs_independent_rhs"] for s in stationary)<3e-13,
        "library_rhs_matches_independent":max(s["library_rhs_vs_independent"] for s in stationary)<3e-13,
        "Schur_step_matches_full_dense_midpoint":metrics["Schur_midpoint_vs_full_dense_Cayley"]<3e-13,
        "increment_source_has_correct_independent_signs":metrics["increment_source_vs_independent_matrix_difference"]<3e-13,
        "dynamic_direct_matches_DOP853":fine["max_direct_error"]<5e-5,
        "dynamic_increment_matches_DOP853":fine["max_delta_error"]<5e-6,
        "dynamic_direct_second_order_refinement":.15<metrics["direct_refinement_ratio"]<.35,
        "dynamic_increment_second_order_refinement":.15<metrics["delta_refinement_ratio"]<.35,
        "increment_current_matches_DOP853":max(s["increment_current_vs_DOP853"] for s in fine["samples"])<4e-4,
        "odd_two_port_surface_columns_decouple":max(s["odd_surface_column_maxabs"] for s in fine["samples"])<2e-14,
        "wrong_conjugation_negative_control_detected":negative["wrong_current_conjugation_stationary_residual"]>.005,
        "zero_dressing_negative_control_detected":negative["zero_dressing_stationary_residual"]>.02,
        "dark_original_electronic_coordinate_retained":metrics["dark_raw_electronic_dimension"]==3 and metrics["dark_uncoupled_rows_unchanged"]<1e-14 and metrics["dark_zero_column_bright_solution_unchanged"]<1e-14,
        **{"rejects_"+k:v for k,v in rejected.items()},
    }
    return {"schema":"independent_continuum_columns_math_checks_v1",
        "scope":"Two original electronic modes and selected incident continuum columns; Maxwell field feedback, no finite photon oscillator bath; no M0 run.",
        "not_proven":["complete incoming ground channels","production photon moments n/m/V","infrared or infinite-time output limits"],
        "odd_control_scope":"Equal L/R surface couplings cancel for their odd difference; tests only surface field/current and electronic columns, not a separate full outgoing-field canonical normalization.",
        "manufactured_coefficients":"u=(u0+sin(pi*t/T)^2 V) exp(i Delta t); D=D0+.07 sin(pi*t/T)^2 H, with bounded explicit complex V and real symmetric H in source.",
        "parameters":{"gaps":GAPS.tolist(),"frequencies":FREQ.tolist(),"gamma":GAMMA,"D0":D0.tolist(),"duration":DURATION},
        "independent_references":["explicit electron-by-electron stationary kernel","constant lab-frame augmented matrix exponential",
                                  "full original-electron dense Cayley solve","DOP853 on independently assembled full equations"],
        "metrics":metrics,"stationary":stationary,"dynamic":{"DOP853_nfev":ode.nfev,"coarse":coarse,"fine":fine},
        "negative_controls":negative,"checks":checks,"passed":all(checks.values()),
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(lib.__file__)]},
        "wall_seconds":time.perf_counter()-started}


def test_independent_continuum_column_math():
    result=run_checks()
    assert result["passed"], {k:v for k,v in result["checks"].items() if not v}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    result=run_checks()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({"out":str(args.out),"passed":result["passed"],"checks":result["checks"],
                      "metrics":result["metrics"],"negative_controls":result["negative_controls"],
                      "wall_seconds":result["wall_seconds"]},indent=2))
    return 0 if result["passed"] else 1

if __name__=="__main__":
    raise SystemExit(main())
