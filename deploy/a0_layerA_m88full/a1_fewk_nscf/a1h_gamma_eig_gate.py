#!/usr/bin/env python3
"""Second A1-H gate: new NSCF Gamma vs production CrI3.eig Gamma."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def parse_eig_gamma(path: Path) -> list[float]:
    bands = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) < 3:
            continue
        if int(p[1]) != 1:
            if bands:
                break
            continue
        bands.append(float(p[2]))
    return bands


def parse_qe_first_k(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"k\s*=\s*0\.0000\s+0\.0000\s+0\.0000[^\n]*\n\s*bands \(ev\):\s*\n\s*\n((?:[ \t-.\d]+\n)+)",
        text,
    )
    if not m:
        m = re.search(
            r"k\s*=\s*0\.0000\s+0\.0000\s+0\.0000[^\n]*bands \(ev\):\s*\n\s*\n((?:[ \t-.\d]+\n)+)",
            text,
        )
    if not m:
        raise RuntimeError("no QE Gamma band block found")
    return [float(x) for x in m.group(1).split()]


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: a1h_gamma_eig_gate.py nscf.out CrI3.eig targets.json out.json", file=sys.stderr)
        return 2
    nscf_out, eig, targets_path, out_path = map(Path, sys.argv[1:])
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    qe = parse_qe_first_k(nscf_out)
    ref = parse_eig_gamma(eig)
    n = min(len(qe), len(ref))
    qe, ref = qe[:n], ref[:n]
    shift = sum(a - b for a, b in zip(qe, ref)) / n
    aligned = [a - shift for a in qe]
    rms = (sum((a - b) ** 2 for a, b in zip(aligned, ref)) / n) ** 0.5
    maxabs = max(abs(a - b) for a, b in zip(aligned, ref))
    pass_rms = targets["gamma_rms_pass_eV"]
    fail_rms = targets["gamma_rms_fail_eV"]
    if rms <= pass_rms:
        status = "PASS"
    elif rms > fail_rms:
        status = "FAIL"
    else:
        status = "WARN"
    payload = {
        "A1H_GAMMA_EIG_GATE": status,
        "n_bands": n,
        "global_shift_eV": shift,
        "rms_eV": rms,
        "maxabs_eV": maxabs,
        "pass_rms_eV": pass_rms,
        "fail_rms_eV": fail_rms,
        "note": "FAIL means do not interpret off-mesh A1-H as interpolation error.",
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_path.with_name("A1H_GAMMA_EIG_GATE.txt").write_text(
        f"A1H_GAMMA_EIG_GATE={status}\n", encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in ("A1H_GAMMA_EIG_GATE", "rms_eV", "maxabs_eV", "global_shift_eV")}))
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
