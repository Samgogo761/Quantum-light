#!/usr/bin/env python3
"""Q-MATH-16.11 weighted classical P bound unit test. Toy kernels only; not M0."""
import math, json
import numpy as np

def pair_test(w1, w2, U, kappa=1.0):
    Om=w1+w2
    wbar=2*w1*w2/Om
    n=U/wbar
    classical=U*abs(kappa)/math.sqrt(w1*w2)
    u1,u2=math.sqrt(w2/Om),math.sqrt(w1/Om)
    bsv=math.sqrt(n*(n+1))*2*abs(kappa)*u1*u2
    return {"omega1":w1,"omega2":w2,"omega_out":Om,"U":U,
            "mean_frequency":wbar,"nbar":n,"classical_bound":classical,
            "bsv_response":bsv,"ratio_numeric":bsv/classical,
            "ratio_formula":math.sqrt(1+wbar/U)}

def main():
    out={"toy_A":pair_test(.16,.20,.10),"toy_B":pair_test(.08,.28,.10)}
    rng=np.random.default_rng(20261023)
    d=5; omega=np.array([.11,.16,.20,.28,.36])
    X=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
    K=(X+X.T)/2; Wmh=np.diag(1/np.sqrt(omega)); Kt=Wmh@K@Wmh
    norm=float(np.linalg.svd(Kt,compute_uv=False)[0]); U=.37; bound=U*norm
    responses=[]
    for _ in range(5000):
        a=rng.normal(size=d)+1j*rng.normal(size=d)
        a*=math.sqrt(U/float(np.vdot(a,np.diag(omega)@a).real))
        responses.append(float(abs(a.T@K@a)))
    out["random_test"]={"bound":float(bound),"largest_sample":float(max(responses)),
                        "passed":bool(max(responses)<=bound+1e-12)}
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
