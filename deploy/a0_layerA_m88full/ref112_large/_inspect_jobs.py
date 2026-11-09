#!/usr/bin/env python3
"""Inspect 112_large NSCF merge and wannier products. Login-node only."""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path("/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref112_large")
A1 = Path(
    "/public/home/wangjs/project/CrI3_TB/window144_hilbert/"
    "a1_fewk_nscf/temp_a1h_scf_clean/CrI3_scf_soc.save"
)
DST = ROOT / "temp_nscf576" / "CrI3_scf_soc.save"
LOGS = ROOT / "logs"
FORBIDDEN = ("temp_scf_soc", "temp_a1h_scf_clean")


def n_wfc(p: Path) -> int:
    if not p.is_dir():
        return -1
    return len(list(p.glob("wfc*.dat")))


def tail(path: Path, n: int = 40) -> str:
    if not path.is_file():
        return f"MISSING {path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def grep_counts(path: Path, pat: str) -> int:
    if not path.is_file():
        return -1
    text = path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(pat, text))


print("=== isolation ===")
print("A1O_wfc", n_wfc(A1))
print("dst_exists", DST.is_dir(), "dst_wfc", n_wfc(DST))
print("charge", (DST / "charge-density.dat").is_file(), "xml", (DST / "data-file-schema.xml").is_file())

xml = DST / "data-file-schema.xml"
if xml.is_file():
    t = xml.read_text(encoding="utf-8", errors="replace")
    print("xml_bytes", xml.stat().st_size)
    print("nks", re.findall(r"<nks>([^<]+)</nks>", t))
    print("nk", re.findall(r"<nk>([^<]+)</nk>", t)[:4], "nk_count", t.count("<nk>"))
    print("nbnd", set(re.findall(r"<nbnd>([^<]+)</nbnd>", t)))
    print("ks_energies", t.count("<ks_energies>"))
    print("wf_collected", re.findall(r"<wf_collected>([^<]+)</wf_collected>", t))
    print("calculation", re.findall(r"<calculation>([^<]+)</calculation>", t))
    print("outdir", re.findall(r"<outdir>([^<]+)</outdir>", t))
    print("forbidden_in_xml", [b for b in FORBIDDEN if b in t])

print("=== chunks ===")
n_done = 0
n_wfc8 = 0
missing = []
for i in range(72):
    tag = f"{i:02d}"
    log = LOGS / f"chunk_{tag}.out"
    save = ROOT / f"temp_chunk_{tag}" / "CrI3_scf_soc.save"
    ok_log = log.is_file() and "JOB DONE." in log.read_text(encoding="utf-8", errors="replace")
    nw = n_wfc(save)
    if ok_log:
        n_done += 1
    if nw == 8:
        n_wfc8 += 1
    if not ok_log or nw != 8:
        missing.append((tag, ok_log, nw))
print("chunk_JOB_DONE", n_done, "chunk_wfc8", n_wfc8, "bad", missing[:8], "n_bad", len(missing))

print("=== 32989 slurm log markers ===")
for p in sorted(LOGS.glob("r112L_nscf8.*")):
    print("--", p.name, p.stat().st_size)
    text = p.read_text(encoding="utf-8", errors="replace")
    for key in ("MERGE_OK", "NSCF576_CHUNKS_DONE", "A1O_WFC_STILL_52", "COUNT_576", "REFUSE", "Error", "OOM"):
        if key.lower() in text.lower() or key in text:
            print("  has", key)
    print("  tail:")
    print(tail(p, 25))

print("=== count json ===")
cj = LOGS / "outer_window_count_576.json"
print("exists", cj.is_file(), "size", cj.stat().st_size if cj.is_file() else None)
if cj.is_file():
    print(cj.read_text(encoding="utf-8")[:4000])

print("=== 32990 products ===")
for name in [
    "CrI3_112large.amn",
    "CrI3_112large.mmn",
    "CrI3_112large.eig",
    "CrI3_112large.nnkp",
    "CrI3_112large.win",
    "CrI3_112large.wout",
    "CrI3_112large_hr.dat",
    "CrI3_112large_tb.dat",
    "CrI3_112large.chk",
]:
    p = ROOT / name
    st = p.stat() if p.exists() else None
    print(f"{name:28s} exists={p.exists()} size={st.st_size if st else None}")

print("=== 32990 slurm logs ===")
for p in sorted(LOGS.glob("r112L_w90.*")) + sorted(LOGS.glob("*pw2wan*")) + sorted(LOGS.glob("*wannier*")):
    print("--", p.name, p.stat().st_size)
    text = p.read_text(encoding="utf-8", errors="replace")
    for key in (
        "JOB DONE",
        "PW2WAN_DONE",
        "W90_112LARGE_DONE",
        "Error",
        "error",
        "STOP",
        "Killed",
        "All done",
        "Exiting",
        "not implemented",
        "not found",
        "Fortran runtime",
        "SIG",
    ):
        if key in text:
            print("  has", repr(key), "count", text.count(key))
    print("  tail:")
    print(tail(p, 60))

print("=== wout key lines ===")
wout = ROOT / "CrI3_112large.wout"
if wout.is_file():
    text = wout.read_text(encoding="utf-8", errors="replace")
    print("lines", text.count("\n"), "bytes", wout.stat().st_size)
    for pat in (
        r"Disentanglement",
        r"Wannierise",
        r"All done",
        r"Error",
        r"WARNING",
        r"Omega",
        r"Spread",
        r"Convergence",
        r"not enough",
    ):
        print("pat", pat, text.count(pat))
    print("--- last 80 lines ---")
    print(tail(wout, 80))
