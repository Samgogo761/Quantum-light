#!/usr/bin/env python3
"""Read-only R1 audit: extract frozen QED/GH5 H2 numbers and recompute moments.

No propagation. Writes a JSON ledger next to the report directory.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light")
OUT = ROOT / "docs" / "04_BSV量子光" / "qmath_r1_recompute.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def walk_keys(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                yield from walk_keys(v, p)
            else:
                yield p, v
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:8]):
            p = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                yield from walk_keys(v, p)
            else:
                yield p, v


def interesting(k: str) -> bool:
    kl = k.lower()
    needles = (
        "h2", "odd", "jx", "j2", "lambda", "mix", "thermal", "theta",
        "eps", "r2", "m_", "var", "photon", "sha", "fourier", "window",
        "delta", "phase", "arg", "abs", "nmax", "alpha",
        "squeeze", "omega", "dt", "steps", "neel",
    )
    return any(n in kl for n in needles)


def main():
    paths = {
        "thermal": ROOT / "results/q_line/qinput_qin_a2_141_gh5_thermal_20261120/gh5_thermal.json",
        "qed_theta_pi": ROOT / "results/q_line/qinput_qin_a2_141_qed_theta_pi_20261120/qed_theta_pi.json",
        "mech": ROOT / "results/q_line/qinput_qin_a2_141_qed_theta_pi_mech_20261120/mech.json",
        "q1a": ROOT / "results/q_line/qinput_qin_a2_141_q1a_20261120/q1a.json",
        "gh5": ROOT / "results/q_line/qinput_qin_a2_141_gh5_20261120/gh5.json",
        "gh5_theta_pi": ROOT / "results/q_line/qinput_qin_a2_141_gh5_theta_pi_20261120/gh5_theta_pi.json",
        "model": ROOT / "tools/b0_small_model/q8_m0_model.py",
        "dyn": ROOT / "tools/b0_small_model/q8_m0_qinput_dynamics_20261026.py",
    }
    hashes = {k: sha256(p) for k, p in paths.items() if p.exists()}

    thermal = load(paths["thermal"])
    qed_pi = load(paths["qed_theta_pi"])
    mech = load(paths["mech"])
    q1a = load(paths["q1a"])
    gh5 = load(paths["gh5"])
    gh5_pi = load(paths["gh5_theta_pi"])

    key_maps = {}
    for name, obj in [
        ("thermal", thermal),
        ("qed_theta_pi", qed_pi),
        ("mech", mech),
        ("q1a", q1a),
        ("gh5", gh5),
        ("gh5_theta_pi", gh5_pi),
    ]:
        items = []
        for k, v in walk_keys(obj):
            if interesting(k):
                if isinstance(v, (int, float, str, bool)):
                    items.append((k, v))
        key_maps[name] = items[:400]

    alpha = 0.5
    r = 0.3
    ns = math.sinh(r) ** 2
    nbar = abs(alpha) ** 2 + ns
    M0 = -0.5 * math.sinh(2.0 * r)
    Mpi = +0.5 * math.sinh(2.0 * r)
    varx0 = math.exp(-2.0 * r)
    varxpi = math.exp(+2.0 * r)
    var_th = math.cosh(2.0 * r)
    x4_th = 3.0 * math.cosh(2.0 * r) ** 2
    x4_mix = 1.5 * (math.exp(-4.0 * r) + math.exp(+4.0 * r))
    m_classical_max = nbar - abs(alpha) ** 2

    out = {
        "hashes": hashes,
        "analytic": {
            "alpha": alpha,
            "r": r,
            "ns_sinh2r": ns,
            "nbar": nbar,
            "M0": M0,
            "Mpi": Mpi,
            "dM": Mpi - M0,
            "VarX_theta0": varx0,
            "VarX_theta_pi": varxpi,
            "VarX_thermal": var_th,
            "X4_thermal": x4_th,
            "X4_mix": x4_mix,
            "X4_rel_to_th": (x4_mix - x4_th) / x4_th,
            "classical_P_M_bound": m_classical_max,
            "squeezed_abs_M": abs(Mpi),
        },
        "key_maps": key_maps,
        "thermal_top_keys": sorted(list(thermal.keys())) if isinstance(thermal, dict) else None,
        "qed_top_keys": sorted(list(qed_pi.keys())) if isinstance(qed_pi, dict) else None,
        "mech_top_keys": sorted(list(mech.keys())) if isinstance(mech, dict) else None,
        "q1a_top_keys": sorted(list(q1a.keys())) if isinstance(q1a, dict) else None,
        "gh5_top_keys": sorted(list(gh5.keys())) if isinstance(gh5, dict) else None,
        "gh5_pi_top_keys": sorted(list(gh5_pi.keys())) if isinstance(gh5_pi, dict) else None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote", OUT)
    print("analytic", json.dumps(out["analytic"], indent=2))
    print("hashes", json.dumps(hashes, indent=2))
    for name, keys in [
        ("thermal", out["thermal_top_keys"]),
        ("qed", out["qed_top_keys"]),
        ("mech", out["mech_top_keys"]),
        ("q1a", out["q1a_top_keys"]),
        ("gh5", out["gh5_top_keys"]),
        ("gh5_pi", out["gh5_pi_top_keys"]),
    ]:
        print(name, keys)


if __name__ == "__main__":
    main()
