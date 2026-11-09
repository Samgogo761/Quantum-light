#!/usr/bin/env python3
"""Independent finite mathematics checks; no M0 production or new light state."""
import json
import numpy as np
from scipy.linalg import expm
from q8_m0_relative_output_20261003 import (
    full_ground_modes_finite,resolvent_from_positive_modes,compose_relative_output)


def ma(a):return float(np.max(np.abs(a)))


def report():
    rng=np.random.default_rng(20261003)
    # Dense positive fixture with all six canonical modes retained.
    z=rng.normal(size=(12,12))
    h=z.T@z/20+np.diag(np.linspace(.4,1.5,12))
    nu,f,j=full_ground_modes_finite(h)
    k=j@h
    gamma=f@f.conj().T
    residuals={
        "positive_frequency_equation":ma(k@f+1j*f*nu),
        "full_CCR_completeness":ma(f@f.conj().T-f.conj()@f.T-1j*j),
        "ground_covariance_stationarity":ma(k@gamma.real+gamma.real@k.T),
        "unsymmetrized_ground":ma(gamma-(gamma.real+.5j*j)),
        "pure_ground":ma(gamma.real@j@gamma.real-j/4),
    }
    residuals["two_pole_resolvent"]=max(
        ma(np.linalg.inv(h-1j*w*j)-resolvent_from_positive_modes(nu,f,w))
        for w in (.17+.09j,.83+.07j,1.4+.21j))
    # Independent finite source increment: same ground coordinates, no new initial vacuum.
    times=np.linspace(0,1.7,4001)
    h1=h+.02*np.outer(z[0],z[0])
    k1=j@h1
    phi=expm(k1*times[-1])
    ref=expm(k*times[-1])
    from scipy.integrate import simpson
    samples=np.array([expm(k*(times[-1]-t))@(k1-k)@expm(k1*t)@f for t in times])
    duhamel=ref@f+simpson(samples,x=times,axis=0)
    residuals["common_ground_Duhamel"]=ma(duhamel-phi@f)

    # Independent known Bogoliubov transformation on two output modes.
    phase=np.diag(np.exp(1j*np.array([.19,-.27])))
    r=.23
    u0=phase
    u=phase*np.cosh(r)
    v=phase@np.array([[0,np.exp(.41j)*np.sinh(r)],[np.exp(.41j)*np.sinh(r),0]])
    m=compose_relative_output(u0,u-u0,v)
    residuals["vacuum_n_known"]=ma(m["n"]-np.eye(2)*np.sinh(r)**2)
    expected=np.cosh(r)*np.sinh(r)*np.exp(.41j)*(phase@np.array([[0,1],[1,0]])@phase.T)
    residuals["vacuum_m_known"]=ma(m["m"]-expected)
    residuals["m_cross_decomposition"]=ma(m["m"]-m["m_reference_source"]-m["m_source_source"])
    zero=compose_relative_output(u0,np.zeros_like(u0),np.zeros_like(u0))
    residuals["no_drive_photon_n"]=ma(zero["n"])
    residuals["no_drive_photon_m"]=ma(zero["m"])
    omitted_cross=ma(m["m"]-m["m_source_source"])
    # Complex rotated output tests n index orientation, not only diagonal populations.
    q,_=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
    rr=np.array([.12,.31])
    ur=q@np.diag(np.cosh(rr))
    vr=q@np.diag(np.sinh(rr)*np.exp(1j*np.array([.32,-.6])))
    rotated=compose_relative_output(q,ur-q,vr)
    # Explicit scalar sum is an independent index check.
    nexp=np.array([[sum(np.conj(vr[a,c])*vr[b,c] for c in range(2)) for b in range(2)] for a in range(2)])
    residuals["n_complex_index_order"]=ma(rotated["n"]-nexp)
    failed=False
    try:compose_relative_output(u0,np.zeros_like(u0),v)
    except ValueError:failed=True
    checks={name:value<1e-9 for name,value in residuals.items()}
    checks["missing_reference_source_cross_detected"]=omitted_cross>.1
    checks["noncanonical_rows_rejected"]=failed
    return {"all_checks_pass":bool(all(checks.values())),"checks":checks,
            "residuals":residuals,"missing_cross_error":omitted_cross,
            "scope":"finite mathematics only; no continuum completeness or P1 claim"}


if __name__=="__main__":
    result=report()
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["all_checks_pass"] else 1)
