"""Small input-only checks. No electronic propagation or production imports."""
import json
from pathlib import Path
import numpy as np
from math import lgamma
h=6.62607015e-34
hbar=h/(2*np.pi)
c=299792458.0
e=1.602176634e-19
epsilon_0=8.8541878188e-12

def laguerre(n,a,x):
    prev,cur=1.0,1+a-x
    if n==0: return prev
    for k in range(1,n):
        prev,cur=cur,((2*k+1+a-x)*cur-(k+a)*prev)/(k+1)
    return cur

def sq(r, theta, nmax):
    v=np.zeros(nmax+1,complex); v[0]=1/np.sqrt(np.cosh(r))
    for n in range(2,nmax+1,2):
        v[n]=-np.exp(1j*theta)*np.tanh(r)*np.sqrt((n-1)/n)*v[n-2]
    return v

def displacement(alpha,nmax):
    out=np.zeros((nmax+1,nmax+1),complex)
    for m in range(nmax+1):
        for n in range(nmax+1):
            lo,hi=min(m,n),max(m,n)
            power=alpha if m>=n else -np.conj(alpha)
            out[m,n]=np.exp(-abs(alpha)**2/2+(lgamma(lo+1)-lgamma(hi+1))/2)*power**(hi-lo)*laguerre(lo,hi-lo,abs(alpha)**2)
    return out

def moments(v):
    v=v/np.linalg.norm(v)
    # Expectations are those of infinite operators in the finite-support state.
    n=np.dot(np.arange(len(v)),abs(v)**2).real
    a=np.vdot(v[:-1],np.sqrt(np.arange(1,len(v)))*v[1:])
    a2=np.vdot(v[:-2],np.sqrt(np.arange(2,len(v))*np.arange(1,len(v)-1))*v[2:])
    x=2*a.real; p=2*a.imag
    return {"n":float(n),"a":[float(a.real),float(a.imag)],"a2":[float(a2.real),float(a2.imag)],"var_X":float(2*n+1+2*a2.real-x*x),"var_P":float(2*n+1-2*a2.real-p*p),"cov_XP":float(2*a2.imag-x*p)}

alpha=.5; r=.3; nr=80
D=displacement(alpha,nr)
ann=np.diag(np.sqrt(np.arange(1,nr+1)),1)
eig,vec=np.linalg.eigh(-1j*(alpha*ann.T-alpha*ann))
Dex=(vec*np.exp(1j*eig))@vec.conj().T
records=[]
for theta in [0,np.pi/4,np.pi/2,np.pi,3*np.pi/2]:
    sv=sq(r,theta,nr); ideal=D@sv; ideal/=np.linalg.norm(ideal)
    alternate=Dex@sv; alternate/=np.linalg.norm(alternate)
    for cut in [5,6,8,10,12]:
        proj=ideal[:cut+1]; legacy=D[:cut+1,:cut+1]@sq(r,theta,cut)
        pv=proj/np.linalg.norm(proj); lv=legacy/np.linalg.norm(legacy)
        records.append({"theta_over_pi":float(theta/np.pi),"nmax":cut,
          "ideal_probability_above_cutoff":float(np.sum(abs(ideal[cut+1:])**2)),
          "ideal_projected_moments":moments(proj),"legacy_prepared_moments":moments(legacy),
          "prepared_vector_L2_difference":float(np.linalg.norm(pv-lv)),
          "independent_reference_max_difference":float(np.max(abs(ideal-alternate)))})

pure=[]
for nbar in [.343,1,5,10,100]:
    # Pair number is negative-binomial with shape 1/2 and p=sech^2(r).
    p=1/(1+nbar); mass=np.sqrt(p); cumulative=mass; pairs=0; previous=1.0
    while 1-cumulative>1e-4:
        previous=1-cumulative; pairs+=1
        mass*=((2*pairs-1)/(2*pairs))*(1-p)
        cumulative+=mass
    cut=2*pairs
    pure.append({"nbar":nbar,"r":float(np.arcsinh(np.sqrt(nbar))),"minimal_even_nmax_tail_1e4":cut,"tail":float(1-cumulative),"previous_even_tail":float(previous)})

energy=.05*e; omega=energy/hbar; g=.005*1e10
qpeak=2*g*alpha; Apeak=hbar*qpeak/e; Epeak=omega*Apeak
unit={"photon_eV":.05,"wavelength_um":h*c/energy*1e6,"period_fs":h/energy*1e15,"six_cycles_fs":6*h/energy*1e15,"mean_equivalent_vacuum_Epeak_Vm":Epeak,"mean_equivalent_vacuum_intensity_Wcm2":.5*epsilon_0*c*Epeak**2/1e4,
      "warning":"free plane-wave plateau and q=eA/hbar assumption; not calibrated incident intensity or total DS intensity"}
sizes=[{"Nk":nk,"dimension":36**nk*9,"one_complex128_vector_GiB":36**nk*9*16/1024**3} for nk in [2,4,6]]
result={"scope":"input-only independent checks; no dynamics", "DS":records,"pure_SV_cutoff":pure,"units":unit,"sizes":sizes}
out=Path(__file__).with_name("input_readonly_check.json")
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({"DS_pi_n8":next(x for x in records if x['theta_over_pi']==1 and x['nmax']==8),"pure_SV_cutoff":pure,"units":unit,"sizes":sizes},indent=2))
