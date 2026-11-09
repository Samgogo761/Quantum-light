"""Read existing currents only; compare nominal vs sampled Fourier boundaries.

Does not import production modules, change archives, or propagate any state.
Linear interpolation is a diagnostic of stored data, not new dynamical evidence.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/q_line"
T = 2 * np.pi / 0.05
LO, HI = 2 * T, 4 * T

def pack(z):
    return {"re": float(np.real(z)), "im": float(np.imag(z))}

def integral(t, j, mode):
    if mode == "archived_mask_trapezoid":
        keep = (t >= LO - 1e-12) & (t <= HI + 1e-12)
        x, y = t[keep], j[keep]
    else:
        assert t[0] <= LO and t[-1] >= HI
        keep = (t > LO) & (t < HI)
        x = np.r_[LO, t[keep], HI]
        y = np.interp(x, t, j.real) + 1j*np.interp(x, t, j.imag)
    if mode != "endpoint_linear_exact_oscillatory":
        return np.trapezoid(y*np.exp(0.1j*x), x) / (HI-LO)
    h = np.diff(x)
    z = 0.1j*h
    ez = np.exp(z)
    a = np.expm1(z)/z
    b = (ez*(z-1)+1)/(z*z)
    return np.sum(np.exp(0.1j*x[:-1])*h*((a-b)*y[:-1]+b*y[1:]))/(HI-LO)

CASES = [
    ("QED_theta0", "qinput_qin_a2_141_q1a_20261120", "ds"),
    ("QED_theta_pi", "qinput_qin_a2_141_qed_theta_pi_20261120", "ds"),
    ("QED_theta_pi2", "qinput_qin_a2_141_qed_q3a_pi2_20261120", "ds"),
    ("QED_theta_3pi2", "qinput_qin_a2_141_qed_q3b_3pi2_20261120", "ds"),
    ("GH5_pi4_eta05", "qinput_qin_a2_141_gh5_q6_pi4_20261120", "e05"),
    ("GH5_pi4_eta10", "qinput_qin_a2_141_gh5_q6_pi4_20261120", "e10"),
    ("GH5_pi4_eta15", "qinput_qin_a2_141_gh5_q6_pi4_20261120", "e15"),
]
MODES = ["archived_mask_trapezoid", "endpoint_trapezoid", "endpoint_linear_exact_oscillatory"]
report = {"scope": "stored-series postprocessing diagnostic only; no propagation; not a new certified result", "cases": []}
for label, dirname, prefix in CASES:
    suffix = "_q" if prefix != "ds" else ""
    zp = np.load(BASE/dirname/f"series_{prefix}_p{suffix}.npz")
    zm = np.load(BASE/dirname/f"series_{prefix}_m{suffix}.npz")
    tp, tm = zp["t"], zm["t"]
    # Each domain is integrated on its own stored grid, with identical exact boundaries.
    vals = {}
    for mode in MODES:
        jp = integral(tp, zp["Jx"], mode)
        jm = integral(tm, zm["Jx"], mode)
        jo, je = (jp-jm)/2, (jp+jm)/2
        vals[mode] = {"odd": jo, "even": je}
    old, new = vals[MODES[0]]["odd"], vals[MODES[2]]["odd"]
    mask = (tp >= LO-1e-12) & (tp <= HI+1e-12)
    report["cases"].append({
        "case": label, "source_dir": str(BASE/dirname),
        "omitted_boundary_intervals": [float(tp[mask][0]-LO), float(HI-tp[mask][-1])],
        "boundary_fraction": float((tp[mask][0]-LO+HI-tp[mask][-1])/(HI-LO)),
        "constant_current_H2_leak_original": pack(integral(tp, np.ones_like(tp, dtype=complex), MODES[0])),
        "constant_current_H2_leak_endpoint_linear": pack(integral(tp, np.ones_like(tp, dtype=complex), MODES[2])),
        "values": {mode:{k:pack(v) for k,v in vv.items()} for mode,vv in vals.items()},
        "coherent_source_contrast_x": {mode: float(2*np.real(vv["even"].conjugate()*vv["odd"])/(abs(vv["even"])**2+abs(vv["odd"])**2)) for mode,vv in vals.items()},
        "odd_change_relative_to_original": float(abs(new-old)/abs(old)),
        "odd_phase_change_degrees": float(np.degrees(np.angle(new/old))),
    })
out = Path(__file__).with_name("window_readonly_check.json")
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
for c in report["cases"]:
    print(c["case"], "omitted=", c["boundary_fraction"],
          "odd_relative_change=", c["odd_change_relative_to_original"],
          "phase_change_deg=", c["odd_phase_change_degrees"])
    print("contrast_x", c["coherent_source_contrast_x"])
print(out)
