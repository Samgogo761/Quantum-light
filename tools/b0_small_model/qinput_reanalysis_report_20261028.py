"""Tables, mathematical identity checks and static SVGs for the read-only batch."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import time
import zipfile

import numpy as np

from qinput_spectrum_reanalysis_20261028 import OUT, ROOT, BASE, PERIOD, packed, unpack, dump, sha


def math_check():
    """Check the exact finite PDPS numerator/denominator angular expansion.

    K is a deterministic synthetic photon response matrix, NOT an M0 prediction.
    No electronic Hamiltonian or propagator is evaluated here.
    """
    nmax, alpha, r = 8, .5, .3
    d = nmax+1
    disp = np.zeros((d,d),complex)
    for m in range(d):
        for n in range(d):
            s=0j
            for k in range(min(m,n)+1):
                s += alpha**(m-k)*(-alpha)**(n-k)/(math.factorial(k)*math.factorial(m-k)*math.factorial(n-k))
            disp[m,n]=np.exp(-alpha**2/2)*np.sqrt(math.factorial(m)*math.factorial(n))*s
    pairs=np.arange(0,d,2)
    sv=np.array([(-np.tanh(r))**(m//2)*np.sqrt(math.factorial(m))/(2**(m//2)*math.factorial(m//2)*np.sqrt(np.cosh(r))) for m in pairs])
    rng=np.random.default_rng(281026)
    K=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
    KT=disp.conj().T@K@disp
    G=disp.conj().T@disp
    num,den={},{}
    for a,m in enumerate(pairs):
        for b,n in enumerate(pairs):
            ell=b-a
            num[ell]=num.get(ell,0j)+sv[a]*sv[b]*KT[m,n]
            den[ell]=den.get(ell,0j)+sv[a]*sv[b]*G[m,n]
    errs=[]; norms=[]
    for th in (0,.17,np.pi/4,np.pi/2,np.pi,3*np.pi/2):
        raw=disp[:,pairs]@(sv*np.exp(1j*np.arange(len(pairs))*th))
        direct=np.vdot(raw,K@raw)/np.vdot(raw,raw)
        angular=sum(v*np.exp(1j*k*th) for k,v in num.items())/sum(v*np.exp(1j*k*th) for k,v in den.items())
        errs.append(abs(direct-angular)); norms.append(float(np.vdot(raw,raw).real))
    assert max(errs)<1e-12
    return {"purpose":"synthetic algebra identity; no M0 response coefficients computed",
            "max_absolute_closure":max(errs),"finite_preparation_mass_range":[min(norms),max(norms)],
            "exact_finite_formula":"J(theta)=sum_l N_l exp(i l theta)/sum_l Z_l exp(i l theta)",
            "ideal_unitary_displacement":"Z_l=delta_l0; B=l0, A=l+1, C=l-1 in displaced basis"}


class SVG:
    def __init__(self,w,h,title):
        self.w,self.h=w,h
        self.s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<style>text{font-family:Arial,sans-serif;fill:#202b34;font-size:13px}</style>']
        self.text(w/2,27,title,19,"middle")
    def text(self,x,y,t,size=13,anchor="start",fill="#202b34"):
        from html import escape
        self.s.append(f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" style="font-size:{size}px;fill:{fill}">{escape(str(t))}</text>')
    def line(self,x1,y1,x2,y2,c="#b9c0c6",width=1,dash=""):
        self.s.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{c}" stroke-width="{width}" stroke-dasharray="{dash}"/>')
    def axis(self,x,y,w,h,xlim,ylim,title,xlabel,ylabel,xticks=None,xticklabels=None):
        self.text(x,y-18,title,15)
        fx=lambda z:x+(z-xlim[0])/(xlim[1]-xlim[0])*w
        fy=lambda z:y+h-(z-ylim[0])/(ylim[1]-ylim[0])*h
        for i,v in enumerate(np.linspace(*xlim,5) if xticks is None else xticks):
            px=fx(v);self.line(px,y+h,px,y+h+5);self.text(px,y+h+23,xticklabels[i] if xticklabels is not None else f'{v:.2g}',12,'middle')
        for v in np.linspace(*ylim,5):
            py=fy(v);self.line(x,py,x+w,py,"#edf0f2");self.text(x-9,py+4,f'{v:.2g}',12,'end')
        if xlim[0]<0<xlim[1]:self.line(fx(0),y,fx(0),y+h,"#d3d7da")
        if ylim[0]<0<ylim[1]:self.line(x,fy(0),x+w,fy(0),"#d3d7da")
        self.line(x,y+h,x+w,y+h,"#52616b");self.line(x,y,x,y+h,"#52616b")
        self.text(x+w/2,y+h+48,xlabel,13,"middle")
        self.text(x,y-38,ylabel,12)
        return fx,fy
    def curve(self,x,y,fx,fy,c,dash="",width=1.7):
        pts=" ".join(f'{fx(a):.2f},{fy(b):.2f}' for a,b in zip(x,y))
        self.s.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="{width}" stroke-dasharray="{dash}"/>')
    def point(self,x,y,c,kind="circle",fill=True,r=4):
        if kind=="cross":
            self.line(x-r,y-r,x+r,y+r,c,2);self.line(x-r,y+r,x+r,y-r,c,2)
        else:self.s.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{c if fill else "white"}" stroke="{c}" stroke-width="1.5"/>')
    def save(self,p):
        Path(p).write_text("\n".join(self.s+["</svg>"]),encoding="utf-8")


def main():
    start=time.perf_counter()
    r=json.loads((OUT/"reanalysis.json").read_text(encoding="utf-8"))
    cases=r["cases"]
    get=lambda label:next(c for c in cases if c["label"]==label)
    at=lambda c,wn="main",m="linear_exact",q=2:c["windows"][wn]["methods"][m][q-1]
    rows=[]
    for c in cases:
        for wn,w in c["windows"].items():
            if w["status"]!="available":continue
            v=at(c,wn);old=at(c,wn,"legacy")
            z=unpack(v["odd"][0]);ze=unpack(v["even"][0])
            alt={m:unpack(at(c,wn,m)["odd"][0]) for m in ("endpoint_trapezoid","quadratic_exact")}
            dec=[unpack(vv[1]["odd"][0]) for vv in w["sensitivity"].values()]
            rows.append({"case":c["label"],"window":wn,"odd_x":packed(z),"even_x":packed(ze),
                 "odd_2d":v["odd_norm2D"],"contrast_x_ppm":v["analyzers"]["x"]["contrast"]*1e6,
                 "legacy_contrast_x_ppm":old["analyzers"]["x"]["contrast"]*1e6,
                 "odd_relative_change_from_legacy":abs(z-unpack(old["odd"][0]))/max(abs(unpack(old["odd"][0])),1e-300),
                 "quad_vs_linear_relative_odd_x":abs(alt["quadratic_exact"]-z)/max(abs(z),1e-300),
                 "trap_vs_linear_relative_odd_x":abs(alt["endpoint_trapezoid"]-z)/max(abs(z),1e-300),
                 "decimation_relative_odd_x_max":max(abs(d-z) for d in dec)/max(abs(z),1e-300)})
    qnames=["QED_theta0","QED_theta_pi2","QED_theta_pi","QED_theta_3pi2"]
    qc=[get(x) for x in qnames]
    z=[unpack(at(c)["odd"][0]) for c in qc]
    phase=float(np.degrees(np.angle(z[3]/z[1])))
    ratio=at(qc[2])["odd_norm2D"]/at(qc[0])["odd_norm2D"]
    qfit=next(f for f in r["fits"] if f["solver"]=="QED" and f["window"]=="main" and f["estimator"]=="linear_exact" and f["channel"]=="odd" and f["axis"]=="x")
    ghfits=[f for f in r["fits"] if f["solver"]=="GH5" and f["window"]=="main" and f["estimator"]=="linear_exact" and f["channel"]=="odd" and f["axis"]=="x"]
    # Independent comparison of historical replay with the unmodified frozen Q6 predictions.
    frozen=json.loads((BASE/"qinput_qin_a2_141_gh5_q6_pi4_20261120/frozen_predictions.json").read_text(encoding="utf-8"))
    replay=[]
    for f in r["fits"]:
        if f["solver"]=="GH5" and f["window"]=="main" and f["estimator"]=="legacy" and f["axis"]=="x":
            ff=frozen["by_eta"][f'{f["eta"]:g}']["fit_"+f["channel"]]
            replay.append({"eta":f["eta"],"channel":f["channel"],"max_BAC_difference":max(abs(unpack(f[k])-unpack(ff[k])) for k in ("B","A","C"))})
    dump(OUT/"legacy_replay_checks.json",replay)
    assert max(x["max_BAC_difference"] for x in replay)<1e-12
    # Observed QED throughput is from archived two-process runs, not a new benchmark.
    costs=[]
    for c in r["archived_QED_costs"]:
        if c["case"] in ("QED_theta0","QED_theta_pi") and c["seconds_per_step"]:
            v=c["seconds_per_step"]
            costs.append({"case":c["case"],"domain":c["domain"],"archived_seconds_per_step":v,
                          "n8_to4T_s":v*7436,"n10_to4T_planning_low_s":v*7436*11/9,
                          "n10_to4T_planning_high_s":v*7436*2,
                          "dimension_n8":36*36*9,"dimension_n10":36*36*11,
                          "reason":"linear to quadratic photon contraction scaling plus overhead; not measured n10"})
    # Physical pulse normalization at ONE proposed reference point.
    h=6.62607015e-34;cc=299792458.;ee=1.602176634e-19;eps=8.8541878128e-12;hb=h/(2*np.pi)
    lam=1600e-9;tau=25e-15;waist=15e-6;I0=1e15
    Aeff=np.pi*waist**2/2;teff=tau*np.sqrt(np.pi)/(2*np.sqrt(np.log(2)))
    photon=h*cc/lam;energy=I0*Aeff*teff;nbar=energy/photon;fcoh=.8
    field=np.sqrt(2*I0/(eps*cc));omega=2*np.pi*cc/lam
    protocol={"label":"proposal only, no propagation; ideal pulse at sample in vacuum reference",
        "lambda_nm":1600,"photon_eV":photon/ee,"T_fs":lam/cc*1e15,"intensity_FWHM_fs":25,
        "Gaussian_1_over_e2_radius_um":15,"cycle_averaged_ensemble_peak_W_cm2":I0/1e4,
        "Aeff_m2":Aeff,"time_effective_fs":teff*1e15,"pulse_energy_J":energy,"mean_photons":nbar,
        "DS_coherent_energy_fraction":fcoh,"DS_alpha_real":np.sqrt(fcoh*nbar),
        "DS_r":np.arcsinh(np.sqrt((1-fcoh)*nbar)),"pure_BSV_r":np.arcsinh(np.sqrt(nbar)),
        "coherent_control_alpha":np.sqrt(nbar),"equivalent_classical_Epeak_V_m":field,
        "equivalent_classical_qpeak_Ainv":ee*field/(hb*omega)*1e-10,
        "DS_mean_Epeak_V_m":np.sqrt(fcoh)*field,"H2_eV":2*photon/ee,
        "assumptions":"Gaussian intensity envelope; paraxial transverse profile; narrowband pulse-energy normalization; undepleted external drive for first statistical BZ calculation; no CrI3 damage or local-field claim"}
    dump(OUT/"protocol_and_cost.json",{"protocol":protocol,"costs":costs,"proposed_QED_batch_budget_s":5400,"new_QED_started":False})
    # Summarize fit and GH5/QED compression-change agreement under the same estimator.
    g0=get("GH5_e10_theta0");gp=get("GH5_e10_theta_pi")
    qd=z[2]-z[0];gd=unpack(at(gp)["odd"][0])-unpack(at(g0)["odd"][0])
    # Boundary correction remains separable from final sampling accuracy.
    key={"QED_main_odd_2D_pi_over_0":ratio,"QED_covariance_flip_phase_x_deg":phase,
         "QED_covariance_flip_2D_ratio":at(qc[3])["odd_norm2D"]/at(qc[1])["odd_norm2D"],
         "QED_theta_pi_minus_0_x_phase_deg":float(np.degrees(np.angle(z[2]/z[0]))),
         "GH5_QED_pi_minus_0_difference_relative_x":float(abs(gd-qd)/abs(qd)),
         "main_QED_rows":[rr for rr in rows if rr["case"] in qnames and rr["window"]=="main"],
         "Q6_corrected":[{"eta":f["eta"],**f["pi4"]} for f in ghfits],
         "math_check":math_check(),"report_compute_s":time.perf_counter()-start}
    dump(OUT/"key_results.json",key)
    dump(OUT/"H2_sensitivity.json",rows)
    # Standalone scientific vector plots; no plotting-library install required.
    figs=OUT/"figures";figs.mkdir(exist_ok=True)
    fig=SVG(1280,900,"Q1-Q6 reanalysis: exact window endpoints, stored-current projections")
    fx,fy=fig.axis(90,108,490,290,(-9.5,2.5),(-5,7),"a  QED H2 odd current, four squeeze angles","Re Jx (10^-8 model units)","Im Jx (10^-8 model units)")
    colors=["#176b98","#b86b00","#6b4c9a","#a43d63"]
    for i,c in enumerate(qc):
        zz=unpack(at(c)["odd"][0])*1e8;old=unpack(at(c,m="legacy")["odd"][0])*1e8
        fig.line(fx(old.real),fy(old.imag),fx(zz.real),fy(zz.imag),colors[i],1)
        fig.point(fx(old.real),fy(old.imag),colors[i],fill=False)
        fig.point(fx(zz.real),fy(zz.imag),colors[i]);fig.text(fx(zz.real)+8,fy(zz.imag)-8,["0","pi/2","pi","3pi/2"][i],12)
    fig.text(90,465,"Open: archived estimator. Filled: exact endpoints + linear reconstruction.",12)
    fx,fy=fig.axis(810,108,290,290,(-1.3,1.3),(-1.3,1.3),"b  GH5 shape; held-out pi/4","Re [(J-B)/A]","Im [(J-B)/A]")
    th=np.linspace(0,2*np.pi,181)
    for i,f in enumerate(ghfits):
        A,B,C=[unpack(f[k]) for k in ("A","B","C")]
        zz=np.exp(1j*th)+(C/A)*np.exp(-1j*th)
        fig.curve(zz.real,zz.imag,fx,fy,colors[i],dash=["","5 3","2 3"][i])
        training=[c for c in cases if c['method']=='GH5' and c['eta']==f['eta'] and c['theta'] is not None and abs(c['theta']-np.pi/4)>1e-10]
        for c in training:
            tz=(unpack(at(c)['odd'][0])-B)/A
            fig.point(fx(tz.real),fy(tz.imag),colors[i],fill=False,r=3)
        zz=(unpack(f["pi4"]["actual"])-B)/A
        fig.point(fx(zz.real),fy(zz.imag),colors[i],"cross",r=6)
        fig.text(752,475+i*18,f'eta={f["eta"]:g}; pi/4 error/|A|={100*f["pi4"]["error_over_A"]:.2f}%',12,fill=colors[i])
    fig.text(752,457,"Lines: four-angle fits; open: training; crosses: pi/4.",11)
    fx,fy=fig.axis(90,570,490,235,(1,6),(-11,-6),"c  QED H1-H6 odd two-component norm","frequency / omega0","log10 |J odd| (model units)",xticks=list(range(1,7)))
    for c,col,lab in [(qc[0],colors[0],"theta=0"),(qc[2],colors[2],"theta=pi")]:
        yy=np.log10([v["odd_norm2D"] for v in c["windows"]["main"]["methods"]["linear_exact"]])
        fig.curve(range(1,7),yy,fx,fy,col)
        for x,y in zip(range(1,7),yy):fig.point(fx(x),fy(y),col)
        fig.text(370 if lab=="theta=0" else 470,548,lab,12,fill=col)
    fx,fy=fig.axis(735,570,440,235,(-.25,3.25),(-40,140),"d  H2 x-analyzer coherent-source contrast","squeeze angle","C = (P+ - P-)/(P+ + P-) [ppm]",xticks=[0,1,2,3],xticklabels=['0','pi/2','pi','3pi/2'])
    for i,c in enumerate(qc):
        old=at(c,m="legacy")["analyzers"]["x"]["contrast"]*1e6
        new=at(c)["analyzers"]["x"]["contrast"]*1e6
        fig.point(fx(i-.06),fy(old),"#677784",fill=False)
        fig.point(fx(i+.06),fy(new),colors[i]);fig.text(fx(i+.06)+7,fy(new)-7,f'{new:+.1f}',11)
    fig.text(640,884,"Main window [2T,4T]. Finite k/-k model; not calibrated photon yield. No new propagation.",12,"middle")
    fig.save(figs/"corrected_physics.svg")
    fig=SVG(1280,610,"Numerical sensitivity retained separately from the endpoint repair")
    fx,fy=fig.axis(90,120,500,340,(-.25,3.25),(0,6),"a  H2 odd Jx reconstruction sensitivity","squeeze angle","relative complex difference [%]",xticks=[0,1,2,3],xticklabels=['0','pi/2','pi','3pi/2'])
    for i,name in enumerate(qnames):
        rr=next(x for x in rows if x["case"]==name and x["window"]=="main")
        fig.point(fx(i-.06),fy(100*rr["quad_vs_linear_relative_odd_x"]),"#176b98")
        fig.point(fx(i+.06),fy(100*rr["decimation_relative_odd_x_max"]),"#b86b00",kind="cross",r=5)
    fig.text(90,520,"Blue: local quadratic vs linear. Orange: remove alternate samples (worst phase).",12)
    fx,fy=fig.axis(740,120,440,340,(-.25,3.25),(0,100),"b  Adjacent-window drift of H2 odd Jx","squeeze angle","relative complex difference to main [%]",xticks=[0,1,2,3],xticklabels=['0','pi/2','pi','3pi/2'])
    for i,c in enumerate(qc):
        mainz=unpack(at(c)["odd"][0])
        for wn,dx,col,kind in [("front",-.06,"#176b98","circle"),("back",.06,"#b86b00","cross")]:
            if c["windows"][wn]["status"]=="available":
                yy=100*abs(unpack(at(c,wn)["odd"][0])-mainz)/abs(mainz)
                fig.point(fx(i+dx),fy(yy),col,kind,r=5);fig.text(fx(i+dx)+8,fy(yy)-7,f'{yy:.1f}',11)
            else:fig.text(fx(i),fy(90),"back missing",11,"middle")
    fig.text(740,520,"Blue: [1T,3T]. Orange: [3T,5T]. These are physical window changes.",12)
    fig.text(640,578,"Sensitivity checks are not rigorous error bars and cannot reconstruct unsaved fast dynamics.",13,"middle")
    fig.save(figs/"sampling_and_windows.svg")
    # Reference evidence: read only, do not execute external Q-MATH programs.
    zp=Path("C:/Users/26507/Downloads/Q-MATH-16.27.zip")
    if zp.exists():
        with zipfile.ZipFile(zp) as zz:
            kb=zz.read("qmath_16_27_retarded_kernel.py")
        dump(OUT/"qmath_reference.json",{"zip_sha256":sha(zp),"kernel_member":"qmath_16_27_retarded_kernel.py",
             "kernel_sha256":hashlib.sha256(kb).hexdigest(),"executed":False,
             "mapping":"R12,J0; (R1+R2),J1; P,J2 all present; +dh convention vs line-Q -dh"})
    print(json.dumps({"ratio2D":ratio,"covsign_phase_deg":phase,"Q6_error_percent":[100*f["pi4"]["error_over_A"] for f in ghfits],
                      "protocol":protocol,"math_closure":key["math_check"]["max_absolute_closure"],
                      "old_replay_max":max(x["max_BAC_difference"] for x in replay),"runtime":time.perf_counter()-start},ensure_ascii=False))


if __name__=="__main__":
    main()
