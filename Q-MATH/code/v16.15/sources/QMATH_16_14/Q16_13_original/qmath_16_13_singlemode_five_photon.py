#!/usr/bin/env python3
"""M0 exact electronic (36d) / finite-pump-Fock formal Lippmann-Schwinger audit.
Computes coefficients of gp^2*gH and gp^4*gH in G,n,0 -> G,n-2,1.
These are regulated *bare* T coefficients; NOT connected/on-shell LSZ amplitudes.
Only single monochromatic pump mode; cannot infer frequency-spectrum D12.
"""
import sys, os, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src_0"
sys.path.insert(0,str(SRC))
import q167_m0_three_photon_T as q

def oscillator(dim):
    a=np.diag(np.sqrt(np.arange(1,dim)), 1)
    return a+a.T

def calc(kf=(.4158,.1683),neel=1,omega=.18,gamma=.25,nin=2, orders=(2,4),project_external=False):
    k=q.geo.k_cart(kf)
    Em,Ue=np.linalg.eigh(q.vertices(k,neel)["0"])
    Eg=Em[0]
    # choose Nmax= nin+4, for gp^4 harmonic oscillator operations.
    d=nin+5
    X=oscillator(d)
    Xp=[np.linalg.matrix_power(X,i) for i in range(5)]
    # Need exactly one H oscillator creation/annihilation.
    # Matrix-electron arrays arranged (electron,pump,H).
    vertices={}
    for r in range(5):
        for s in range(2):
            if (r+s)==0 or (r+s)>5:continue
            vv=q.dgamma(q.exact_vertex(k,neel,[q.p]*r+[q.h]*s))
            vv=Ue.conj().T@vv@Ue/math.factorial(r)
            vertices[r,s]=vv

    E_i=float(Eg+nin*omega)
    # omega_H = 2 omega fixed exact kinematics
    energy=Em[:,None,None]+omega*np.arange(d)[None,:,None]+(2*omega)*np.arange(2)[None,None,:]
    resol=1/(E_i-energy+1j*gamma)
    if project_external:
        resol[0,nin,0]=0.0
        resol[0,nin-2,1]=0.0

    def apply(r,s,vec):
        # vec electron,pump,H; photon power matrices in row-output col-input convention
        y=np.einsum("ef,fpb->epb",vertices[r,s],vec, optimize=True)
        y=np.einsum("ij,ejb->eib",Xp[r],y,optimize=True)
        if s: y=y[:,:,::-1]
        return y

    init=np.zeros((36,d,2),complex);init[0,nin,0]=1
    C={(0,0):init}
    for r in range(5):
        for s in range(2):
            if not r+s:continue
            cur=np.zeros_like(init)
            if (r,s) in vertices:cur+=apply(r,s,init)
            for (a,b) in vertices:
                if a>r or b>s or (a==r and b==s):continue
                lower=(r-a,s-b)
                if lower not in C:continue
                cur+=apply(a,b,resol*C[lower])
            C[r,s]=cur
    amp3=C[2,1][0,nin-2,1]
    amp5=C[4,1][0,nin-2,1]
    return {"kf":list(kf),"N":neel,"w":omega,"gamma":gamma,"n":nin,"project_external":project_external,
            "T3":[float(amp3.real),float(amp3.imag)],
            "T5":[float(amp5.real),float(amp5.imag)],
            "T3_scaled":[float((amp3/math.sqrt(nin*(nin-1))).real),float((amp3/math.sqrt(nin*(nin-1))).imag)],
            "T5_scaled":[float((amp5/math.sqrt(nin*(nin-1))).real),float((amp5/math.sqrt(nin*(nin-1))).imag)]}
if __name__=="__main__":
    import argparse
    pa=argparse.ArgumentParser()
    pa.add_argument("--out",default=str(ROOT/"five_photon_local.json"))
    args=pa.parse_args()
    cases=[]
    for g in (.25,.1):
        for n in (2,4):
            x=calc(nin=n,gamma=g)
            cases.append(x)
            print(x,flush=True)
    Path(args.out).write_text(json.dumps({"scope":"formal bare finite-gamma single-mode T, no LSZ dressing or broadband output","cases":cases},indent=2),encoding="utf-8")
