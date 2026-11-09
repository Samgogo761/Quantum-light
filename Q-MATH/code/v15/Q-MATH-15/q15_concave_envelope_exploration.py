import numpy as np,json
from pathlib import Path
from scipy.special import gammaln
from scipy.optimize import linprog,minimize_scalar
D=Path(__file__).resolve().parent/'data'
a0=np.load(D/'q13_kernel_23_96_Nplus.npz')['kernel'][1,0]
a1=np.load(D/'q13_kernel_23_96_Nminus.npz')['kernel'][1,0]
K=(a0-a1)/2
N=len(K);j=np.arange(N-2)
coef=np.diag(K,2)*np.exp(-.5*(gammaln(j+1)+gammaln(j+3)))
def F(x):return np.exp(-x)*np.polynomial.polynomial.polyval(x,coef)
def bsv(n):
 t=np.sqrt(n/(1+n));v=np.zeros(N,complex);l=np.arange((N+1)//2);idx=2*l
 v[idx]=(-t)**l *np.exp(.5*gammaln(idx+1)-gammaln(l+1)-l*np.log(2))/(1+n)**.25
 v/=np.linalg.norm(v)
 return sum(np.conj(v[k])*v[k+2]*K[k,k+2] for k in range(0,N-2,2))
# precise test on 0..80; far tail |xF(x)| <= spectral norm K due coherent expectation
xs=np.r_[np.linspace(0,.1,1001),np.linspace(.101,4,1000),np.linspace(4.01,30,1500),np.linspace(30.1,100,400)]
h=np.array([x*abs(F(x)) for x in xs]);results={}
for mean in (.001,.01,.1,.2,.7):
  r=linprog(-h,A_eq=np.vstack((np.ones_like(xs),xs)),b_eq=np.array([1.,mean]),bounds=(0,None),method='highs')
  assert r.success,r.message
  optidx=np.where(r.x>1e-6)[0]
  theoretical=mean*abs(coef[0]);w=abs(bsv(mean))
  results[str(mean)]=dict(BSV=w,bound_linear=theoretical,bound_opt_grid=-r.fun,competitors=[dict(mean=float(xs[i]),prob=float(r.x[i])) for i in optidx],ratios=[w/theoretical,w/(-r.fun)])
print(json.dumps(results,indent=2))
Path(__file__).resolve().parent/'q15_concave_envelope.json'.write_text(json.dumps(results,indent=2))
# direct counterexample for shortened phase protocol
x=.01;theta=np.arange(16384)*2*np.pi/16384
v=np.exp(-x)*np.sqrt(x)*np.exp(0.5j*theta)
wrong=(np.exp(-1j*theta)*v).mean()
theta2=np.arange(16384)*4*np.pi/16384
right=(np.exp(-1j*theta2)*np.exp(-x)*np.sqrt(x)*np.exp(.5j*theta2)).mean()
print('counterexample: wrong',wrong,abs(wrong),'correct 4pi',right,abs(right),'exact',2*np.sqrt(x)*np.exp(-x)/np.pi)
print('coefficient ratio max',max(np.abs(coef[1:])*np.exp(gammaln(j[1:]+1))/abs(coef[0])),'finite tight C',abs(coef[0]))
