#!/usr/bin/env python3
"""A1-H SCF provenance gate. Writes PASS/FAIL; exit 0 only on PASS."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def last_float(text: str, pattern: str):
    matches = list(re.finditer(pattern, text, re.M))
    if not matches:
        return None
    return float(matches[-1].group(1))


def last_mag_vec(text: str):
    matches = list(
        re.finditer(
            r"total magnetization\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+Bohr",
            text,
        )
    )
    if not matches:
        return None
    m = matches[-1]
    return [float(m.group(1)), float(m.group(2)), float(m.group(3))]


def parse_site_cr(text: str) -> list[dict]:
    """Best-effort; April low-verbosity out may have none."""
    rows = []
    for m in re.finditer(
        r"atom:\s+(\d+)\s+charge:\s+([\d.]+)\s+magn:\s+\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)",
        text,
    ):
        rows.append(
            {
                "atom": int(m.group(1)),
                "charge": float(m.group(2)),
                "magn": [float(m.group(3)), float(m.group(4)), float(m.group(5))],
            }
        )
    return rows[:4]  # Cr are atoms 1-2 and 9-10 in QE order, but keep whatever printed


def within(val, ref, tol) -> bool:
    return val is not None and abs(val - ref) <= tol


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: a1h_scf_provenance_gate.py scf.out targets.json out.json", file=sys.stderr)
        return 2
    scf_out = Path(sys.argv[1])
    targets = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[3])
    text = scf_out.read_text(encoding="utf-8", errors="replace")
    workdir = scf_out.resolve().parent.parent

    checks = {}
    job_done = "JOB DONE." in text
    checks["job_done"] = {"ok": job_done, "value": job_done}

    bang = list(re.finditer(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", text))
    etot = float(bang[-1].group(1)) if bang else None
    checks["etot_ry"] = {
        "ok": within(etot, targets["etot_ry"], targets["etot_tol_ry"]),
        "value": etot,
        "target": targets["etot_ry"],
        "tol": targets["etot_tol_ry"],
    }

    ef = last_float(text, r"the Fermi energy is\s+([-\d.]+)\s+ev")
    checks["efermi_ev"] = {
        "ok": within(ef, targets["efermi_ev"], targets["efermi_tol_ev"]),
        "value": ef,
        "target": targets["efermi_ev"],
        "tol": targets["efermi_tol_ev"],
    }

    mtot = last_mag_vec(text)
    if mtot is None:
        checks["m_tot"] = {"ok": False, "value": None}
    else:
        ok = all(abs(a - b) <= targets["m_tot_tol"] for a, b in zip(mtot, targets["m_tot_bohr"]))
        checks["m_tot"] = {"ok": ok, "value": mtot, "target": targets["m_tot_bohr"], "tol": targets["m_tot_tol"]}

    mabs = last_float(text, r"absolute magnetization\s+=\s+([\d.]+)\s+Bohr")
    checks["m_abs"] = {
        "ok": within(mabs, targets["m_abs_bohr"], targets["m_abs_tol"]),
        "value": mabs,
        "target": targets["m_abs_bohr"],
        "tol": targets["m_abs_tol"],
    }

    hub = last_float(text, r"Hubbard energy\s+=\s+([-\d.]+)\s+Ry")
    checks["hubbard_ry"] = {
        "ok": within(hub, targets["hubbard_ry"], targets["hubbard_tol_ry"]),
        "value": hub,
        "target": targets["hubbard_ry"],
        "tol": targets["hubbard_tol_ry"],
    }

    site = parse_site_cr(text)
    checks["m_cr_sites"] = {
        "ok": True,
        "required": False,
        "value": site,
        "note": targets.get("note_m_cr", ""),
    }

    schema = workdir / "temp_a1h_scf_clean/CrI3_scf_soc.save/data-file-schema.xml"
    schema_ok = schema.is_file() and "<calculation>scf</calculation>" in schema.read_text(
        encoding="utf-8", errors="replace"
    )
    checks["schema_scf"] = {"ok": schema_ok, "path": str(schema)}

    upf_ok = True
    upf_got = {}
    for name, expected in targets["upf_sha256"].items():
        p = workdir / "pseudo" / name
        got = sha256_file(p) if p.is_file() else None
        upf_got[name] = got
        if got != expected:
            upf_ok = False
    checks["upf_sha256"] = {"ok": upf_ok, "value": upf_got, "target": targets["upf_sha256"]}

    required = ["job_done", "etot_ry", "efermi_ev", "m_tot", "m_abs", "hubbard_ry", "schema_scf", "upf_sha256"]
    passed = all(checks[k]["ok"] for k in required)
    status = "PASS" if passed else "FAIL"
    payload = {
        "A1H_SCF_PROVENANCE_GATE": status,
        "checks": checks,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    flag = out_path.with_name("A1H_SCF_PROVENANCE_GATE.txt")
    flag.write_text(f"A1H_SCF_PROVENANCE_GATE={status}\n", encoding="utf-8")
    print(f"A1H_SCF_PROVENANCE_GATE={status}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
