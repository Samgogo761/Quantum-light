"""Read-only Q1--Q6 current reanalysis. No solver imports or propagation.

The primary estimator integrates the piecewise-linear stored current times the
Fourier exponential analytically, with the exact requested window boundaries.
Quadratic reconstruction, endpoint trapezoid and decimation are sensitivities,
not rigorous error bounds on the unknown current between stored samples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/q_line"
OUT = BASE / "qinput_spectrum_reanalysis_20261028"
OMEGA = 0.05
PERIOD = 2 * np.pi / OMEGA
WINDOWS = {"main": (2., 4.), "front": (1., 3.), "back": (3., 5.)}
METHODS = ("legacy", "endpoint_trapezoid", "linear_exact", "quadratic_exact")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packed(z):
    a = np.asarray(z)
    if a.ndim:
        return [packed(x) for x in a]
    z = complex(a)
    return {"re": z.real, "im": z.imag, "abs": abs(z),
            "phase_deg": float(np.degrees(np.angle(z)))}


def unpack(z):
    return complex(z["re"], z["im"])


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (complex, np.complexfloating)):
        return packed(x)
    if isinstance(x, np.generic):
        return x.item()
    return x


def dump(path, data):
    Path(path).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def validate(t, j):
    t, j = np.asarray(t, float), np.asarray(j, complex)
    if t.ndim != 1 or len(t) < 3 or j.shape[0] != len(t):
        raise ValueError("need >=3 time samples and matching current")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(j)):
        raise ValueError("nonfinite input")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing")
    return t, j[:, None] if j.ndim == 1 else j


def moments(z):
    """Integral_0^1 u^k exp(z*u) du, k=0,1,2; stable at z=0."""
    z = np.asarray(z, complex)
    ans = np.empty((3, z.size), complex)
    small = np.abs(z) < 0.15
    for k in range(3):
        v, power = np.zeros(np.count_nonzero(small), complex), np.ones(np.count_nonzero(small), complex)
        for n in range(18):
            v += power / (n + k + 1)
            power *= z[small] / (n + 1)
        ans[k, small] = v
    zz = z[~small]
    ans[0, ~small] = np.expm1(zz) / zz
    ans[1, ~small] = (np.exp(zz) - ans[0, ~small]) / zz
    ans[2, ~small] = (np.exp(zz) - 2 * ans[1, ~small]) / zz
    return ans


def spectrum(t, j, lo, hi, qs, method="linear_exact", omega=OMEGA):
    t, j = validate(t, j)
    if not hi > lo:
        raise ValueError("empty or reversed window")
    tol = 1e-10 * max(1, abs(lo), abs(hi))
    if lo < t[0] - tol or hi > t[-1] + tol:
        raise ValueError("window unavailable; extrapolation forbidden")
    # Only roundoff at the last sample is clamped, never a missing interval.
    lo, hi = max(lo, t[0]), min(hi, t[-1])
    if method == "legacy":
        keep = (t >= lo - 1e-12) & (t <= hi + 1e-12)
        x, y = t[keep], j[keep]
    else:
        x = np.r_[lo, t[(t > lo) & (t < hi)], hi]
        y = np.stack([np.interp(x, t, col.real) + 1j * np.interp(x, t, col.imag)
                      for col in j.T], axis=1)
    out = []
    if method in ("legacy", "endpoint_trapezoid"):
        for q in qs:
            out.append(np.trapezoid(np.exp(1j * q * omega * x)[:, None] * y, x, axis=0) / (hi-lo))
        return np.asarray(out)
    if method not in ("linear_exact", "quadratic_exact"):
        raise ValueError(method)
    h = np.diff(x)
    c0, c1 = y[:-1], np.diff(y, axis=0) / h[:, None]
    c2 = np.zeros_like(c1)
    if method == "quadratic_exact":
        # Three original samples per original interval, not synthetic refinement.
        ii = np.clip(np.searchsorted(t, x[:-1], side="right") - 1, 0, len(t)-2)
        i0 = np.maximum(ii-1, 0)
        i0 = np.minimum(i0, len(t)-3)
        ta, tb, tc = t[i0], t[i0+1], t[i0+2]
        ya, yb, yc = j[i0], j[i0+1], j[i0+2]
        b1 = (yb-ya)/(tb-ta)[:, None]
        b2 = ((yc-yb)/(tc-tb)[:, None] - b1)/(tc-ta)[:, None]
        xx = x[:-1]
        c0 = ya + b1*(xx-ta)[:, None] + b2*((xx-ta)*(xx-tb))[:, None]
        c1 = b1 + b2*(2*xx-ta-tb)[:, None]
        c2 = b2
    for q in qs:
        z = 1j * float(q) * omega * h
        a, b, c = moments(z)
        integ = h[:, None] * (a[:, None]*c0 + (h*b)[:, None]*c1 + (h*h*c)[:, None]*c2)
        out.append(np.sum(np.exp(1j * float(q)*omega*x[:-1])[:, None]*integ, axis=0)/(hi-lo))
    return np.asarray(out)


def cases():
    rows = []
    def add(label, method, eta, angle, folder, prefix):
        rows.append(dict(label=label, method=method, eta=eta, theta=angle, folder=folder, prefix=prefix))
    tag = "qinput_qin_a2_141_"
    for label, angle, folder in [("theta0", 0., "q1a"), ("theta_pi", np.pi, "qed_theta_pi"),
                                 ("theta_pi2", np.pi/2, "qed_q3a_pi2"), ("theta_3pi2", 3*np.pi/2, "qed_q3b_3pi2")]:
        add("QED_"+label, "QED", 1., angle, tag+folder+"_20261120", "ds_{n}")
    add("QED_coherent", "QED", 1., None, tag+"q1a_20261120", "coh_{n}")
    for label, angle, folder in [("theta0", 0., "gh5"), ("theta_pi", np.pi, "gh5_theta_pi"),
                                 ("theta_pi2", np.pi/2, "gh5_q2_covsign"), ("theta_3pi2", 3*np.pi/2, "gh5_q2_covsign")]:
        suffix = "_h" if angle == np.pi/2 else "_t" if angle == 3*np.pi/2 else ""
        add("GH5_e10_"+label, "GH5", 1., angle, tag+folder+"_20261120", "ds_{n}"+suffix)
    add("GH5_coherent", "GH5", 1., None, tag+"gh5_20261120", "coh_{n}")
    add("GH5_thermal", "GH5", 1., None, tag+"gh5_thermal_20261120", "th_{n}")
    for eta, pref in [(.5,"e05"), (1.5,"e15")]:
        for angle, suffix, folder in [(0.,"0","gh5_q5_fourangle"), (np.pi,"pi","gh5_q5_fourangle"),
                                       (np.pi/2,"h","gh5_q4_eta"), (3*np.pi/2,"t","gh5_q4_eta")]:
            add(f"GH5_{pref}_{suffix}", "GH5", eta, angle, tag+folder+"_20261120", pref+"_{n}_"+suffix)
    for eta, pref in [(.5,"e05"),(1.,"e10"),(1.5,"e15")]:
        add(f"GH5_{pref}_pi4", "GH5", eta, np.pi/4, tag+"gh5_q6_pi4_20261120", pref+"_{n}_q")
    return rows


def domain_stats(p, m):
    e, o = (p+m)/2, (p-m)/2
    vals = {"plus": packed(p), "minus": packed(m), "even": packed(e), "odd": packed(o)}
    stats = {}
    for name, vec in (("x",np.array([1.,0.])), ("y",np.array([0.,1.])),
                      ("45deg",np.ones(2)/np.sqrt(2)), ("135deg",np.array([1.,-1.])/np.sqrt(2))):
        pp, mm, ee, oo = vec@p, vec@m, vec@e, vec@o
        ip, im = abs(pp)**2, abs(mm)**2
        stats[name] = {"Pplus": ip, "Pminus": im, "difference": ip-im,
                       "interference_4Re": float(4*np.real(ee.conjugate()*oo)),
                       "contrast": float((ip-im)/(ip+im)) if ip+im > 1e-300 else None,
                       "phase_odd_minus_even_deg": float(np.degrees(np.angle(oo/ee))) if abs(ee)>1e-100 else None}
    vals["analyzers"] = stats
    vals["odd_norm2D"] = float(np.linalg.norm(o))
    vals["even_norm2D"] = float(np.linalg.norm(e))
    vals["stokes"] = {}
    for lab, z in [("plus",p),("minus",m)]:
        xy = z[0].conjugate()*z[1]
        vals["stokes"][lab] = {"S0": float(np.vdot(z,z).real), "S1": float(abs(z[0])**2-abs(z[1])**2),
                                    "S2": float(2*xy.real), "S3": float(2*xy.imag)}
    return vals


def fit_bac(z):
    th = np.arange(4)*np.pi/2
    X = np.column_stack([np.ones(4), np.exp(1j*th), np.exp(-1j*th)])
    coeff = np.linalg.lstsq(X, np.asarray(z), rcond=None)[0]
    residual = np.asarray(z)-X@coeff
    return coeff, {"B": packed(coeff[0]), "A": packed(coeff[1]), "C": packed(coeff[2]),
                   "relative_L2_residual": float(np.linalg.norm(residual)/max(np.linalg.norm(z),1e-300)),
                   "max_residual_over_A": float(max(abs(residual))/max(abs(coeff[1]),1e-300)),
                   "abs_C_over_A": float(abs(coeff[2])/max(abs(coeff[1]),1e-300))}


def run(out=OUT, budget_s=300.):
    start = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources, records, flat, costs = {}, [], [], []
    def clock():
        if time.perf_counter()-start > budget_s:
            raise TimeoutError("postprocessing budget exceeded; no propagation authorized")
    for spec in cases():
        clock()
        data = []
        row = dict(spec, windows={})
        for domain in ("p","m"):
            path = BASE/spec["folder"]/("series_"+spec["prefix"].format(n=domain)+".npz")
            sources[str(path.relative_to(ROOT))] = sha(path)
            with np.load(path, allow_pickle=False) as f:
                t, j = validate(f["t"], np.column_stack([f["Jx"], f["Jy"]]))
            data.append((t,j))
            row["coverage_"+domain] = {"n_samples":len(t),"end_T":float(t[-1]/PERIOD),
                "median_sample_dt":float(np.median(np.diff(t))), "max_sample_dt":float(max(np.diff(t))),
                "max_imag_current":float(np.max(abs(j.imag)))}
            jp = path.with_name(path.name.replace("series_", "job_").replace(".npz", ".json"))
            if jp.exists():
                sources[str(jp.relative_to(ROOT))] = sha(jp)
                job = json.loads(jp.read_text(encoding="utf-8"))
                if spec["method"] == "QED":
                    costs.append({"case":spec["label"],"domain":domain,"wall_seconds":job.get("wall_seconds"),
                                  "seconds_per_step":job.get("seconds_per_step"),"definition":job.get("definition"),
                                  "ledger":job.get("ledger"),"status":job.get("status")})
        for wn, (a,b) in WINDOWS.items():
            lo, hi = a*PERIOD, b*PERIOD
            if any(t[0] > lo+1e-9 or t[-1] < hi-1e-9 for t,j in data):
                row["windows"][wn] = {"status":"unavailable", "reason":"stored series do not cover both exact endpoints"}
                continue
            wrec = {"status":"available", "bounds_T":[a,b],"methods":{},"sensitivity":{}}
            for method in METHODS:
                vals = [spectrum(t,j,lo,hi,np.arange(1,7),method) for t,j in data]
                wrec["methods"][method] = [domain_stats(p,m) for p,m in zip(*vals)]
            # Coarsening tests interpolation sensitivity; it cannot certify unsaved fast motion.
            for phase in (0,1):
                dec = []
                for t,j in data:
                    idx = np.unique(np.r_[0,np.arange(phase,len(t),2),len(t)-1])
                    dec.append(spectrum(t[idx],j[idx],lo,hi,np.arange(1,7),"linear_exact"))
                wrec["sensitivity"]["decimate2_phase"+str(phase)] = [domain_stats(p,m) for p,m in zip(*dec)]
            for k,q in enumerate(range(1,7)):
                for method in METHODS:
                    val = wrec["methods"][method][k]
                    for ch in ("even","odd"):
                        for axis in range(2):
                            z = val[ch][axis]
                            flat.append(dict(case=spec["label"],eta=spec["eta"],theta=spec["theta"],window=wn,
                                estimator=method,q=q,channel=ch,axis="xy"[axis],**z))
            row["windows"][wn] = wrec
        records.append(row)
    # Fit ONLY the four original angles. pi/4 never enters a coefficient fit.
    fits = []
    for solver in ("QED","GH5"):
        for eta in ((1.,) if solver=="QED" else (.5,1.,1.5)):
            pool = [r for r in records if r["method"]==solver and r["eta"]==eta and r["theta"] is not None]
            training = [next(r for r in pool if abs(r["theta"]-th)<1e-10) for th in np.arange(4)*np.pi/2]
            test = next((r for r in pool if abs(r["theta"]-np.pi/4)<1e-10), None)
            for wn in WINDOWS:
                if not all(r["windows"][wn]["status"]=="available" for r in training):
                    continue
                for method in METHODS:
                    for ch in ("odd","even"):
                        for axis in range(2):
                            z = [unpack(r["windows"][wn]["methods"][method][1][ch][axis]) for r in training]
                            coeff, stats = fit_bac(z)
                            f = dict(solver=solver,eta=eta,window=wn,estimator=method,channel=ch,axis="xy"[axis],**stats)
                            if test and test["windows"][wn]["status"]=="available":
                                actual = unpack(test["windows"][wn]["methods"][method][1][ch][axis])
                                pred = coeff@np.array([1,np.exp(1j*np.pi/4),np.exp(-1j*np.pi/4)])
                                f["pi4"] = {"prediction":packed(pred),"actual":packed(actual),"difference":packed(actual-pred),
                                     "error_over_A":float(abs(actual-pred)/max(abs(coeff[1]),1e-300)),
                                     "phase_error_deg":float(np.degrees(np.angle(actual/pred))),
                                     "role":"held-out angle under revised estimator, retrospective reanalysis"}
                            fits.append(f)
    frozen = BASE/"qinput_qin_a2_141_gh5_q6_pi4_20261120/frozen_predictions.json"
    q5 = BASE/"qinput_qin_a2_141_gh5_q5_fourangle_20261120/gh5_q5_fourangle.json"
    for p in (frozen,q5):
        sources[str(p.relative_to(ROOT))] = sha(p)
    unchanged = all(sha(ROOT/p)==h for p,h in sources.items())
    dump(out/"input_manifest.json", {"before":sources,"after_equal":unchanged,
                                    "analyzer_sha256":sha(__file__),"python":platform.python_version(),"numpy":np.__version__})
    dump(out/"reanalysis.json", {"date":"2026-10-28", "legacy_task_tag":"20261120", "omega0":OMEGA,
         "estimator_primary":"linear_exact", "scope":"finite stored-current source projections, no new propagation",
         "fourier":"Jq=(hi-lo)^-1 integral_lo^hi exp(+i q omega0 t) J(t) dt",
         "coordinate":"Cartesian x,y; no polarization or phase adjustment", "cases":records,
         "fits":fits, "archived_QED_costs":costs,"old_inputs_unchanged":unchanged,
         "wall_seconds":time.perf_counter()-start,"new_propagation":False,
         "caveat":"Interpolation/decimation sensitivity is not a rigorous sampling error bound; Fock/BZ/output normalization not certified."})
    with (out/"H1_H6_complex.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    dump(out/"fits_BAC.json", fits)
    print(json.dumps({"out":str(out),"pairs":len(records),"rows":len(flat),"wall_seconds":time.perf_counter()-start,
                      "old_inputs_unchanged":unchanged},ensure_ascii=False))
    return records,fits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--budget-s", type=float, default=300.)
    args = ap.parse_args()
    run(args.out,args.budget_s)
