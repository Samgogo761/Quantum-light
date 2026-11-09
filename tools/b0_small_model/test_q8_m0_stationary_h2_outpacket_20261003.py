#!/usr/bin/env python3
"""Independent algebra and normalization checks; no M0/P1 propagation."""
from __future__ import annotations
import json
import numpy as np
from q8_m0_stationary_h2_outpacket_20261003 import photon_output_rows,packet_quadrature,maxabs
from q8_m0_stage1ao_canonical_channels import even_maps,factor_psd

def report():
    rng=np.random.default_rng(391)
    # Non-normal passive response: noncommuting Hermitian and absorptive parts.
    h=np.array([[.41,.13+.07j],[.13-.07j,-.22]])
    t=np.array([[.32,.04+.11j],[.09-.02j,.27]])
    ci=t@t.conj().T
    kr=h-.5j*ci
    omega=.11;gamma=.021
    s,g,_=even_maps(kr,omega,gamma)
    la=np.sqrt(omega/gamma)*g@factor_psd(ci)["L"]
    rows=photon_output_rows(s,la)
    # Independent even/odd construction, then direct unitary side rotation.
    eye=np.eye(2);u=np.block([[eye,eye],[eye,-eye]])/np.sqrt(2)
    eo=np.zeros((4,4+la.shape[1]),complex)
    eo[:2,:2]=s;eo[:2,4:]=la;eo[2:4,2:4]=-eye
    inputrot=np.eye(4+la.shape[1]);inputrot[:4,:4]=u
    direct=u@eo@inputrot
    checks={"side_rotation":maxabs(rows-direct),
        "all_photon_CCR":maxabs(rows@rows.conj().T-np.eye(4))}
    noel=maxabs(rows[:,:4]@rows[:,:4].conj().T-np.eye(4))
    # Wrong creation flips electronic CCR sign and produces vacuum photons.
    wrong=maxabs(rows[:,:4]@rows[:,:4].conj().T-rows[:,4:]@rows[:,4:].conj().T-np.eye(4))
    free=photon_output_rows(eye,np.zeros((2,0)))
    checks["zero_coupling_transmission"]=maxabs(free-np.block([[np.zeros((2,2)),eye],[eye,np.zeros((2,2))]]))
    # Arbitrary electronic channel basis rotation must preserve observables.
    z=rng.normal(size=(la.shape[1],la.shape[1]))+1j*rng.normal(size=(la.shape[1],la.shape[1]))
    v=np.linalg.qr(z)[0]
    rotated=photon_output_rows(s,la@v)
    checks["electron_channel_basis_invariance"]=maxabs(rotated@rotated.conj().T-rows@rows.conj().T)
    q=packet_quadrature(8,48);fine=packet_quadrature(16,48)
    checks["packet_mass"]=abs(q["raw_mass"]-1)
    checks["packet_mass_refinement"]=abs(q["raw_mass"]-fine["raw_mass"])
    assert all(v<1e-8 for v in checks.values()),checks
    assert noel>1e-3 and wrong>2e-3
    assert q["UV_omitted_mass_analytic_upper_bound"]>0
    assert not q["packet_renormalized_on_quadrature"]
    return {"status":"PASS","checks":checks,
        "negative_controls":{"omit_electron_defect":noel,"wrong_creation_defect":wrong},
        "limits":"algebra and packet quadrature only; no driven result or electronic continuum qualification"}

if __name__=="__main__":print(json.dumps(report(),indent=2))
