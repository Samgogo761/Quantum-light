#!/usr/bin/env python3
"""Tests for 576-k NSCF input builder."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_nscf576 import build_nscf, kpoints_from_win, write_chunks

TPL = """&CONTROL
    calculation = 'nscf'
    verbosity = 'high'
    prefix = 'CrI3_scf_soc'
    outdir = './temp_count/'
    disk_io = 'low'
/
&SYSTEM
    nbnd = 400
/
K_POINTS crystal
8
  0.0 0.0 0.0 1.0
HUBBARD (ortho-atomic)
   U Cr1-3d 2.8
"""


def require(cond: bool, msg: object) -> None:
    if not cond:
        raise RuntimeError(str(msg))


def test_kpoints() -> None:
    lines = ["begin kpoints"]
    n = 0
    for i in range(24):
        for j in range(24):
            lines.append(f"    {i/24:.10f}    {j/24:.10f}    0.0000000000")
            n += 1
    lines.append("end kpoints")
    ks = kpoints_from_win("\n".join(lines) + "\n")
    require(len(ks) == 576, len(ks))
    require("1.0" in ks[0], ks[0])


def test_build() -> None:
    ks = kpoints_from_win(
        "begin kpoints\n"
        + "\n".join(f"    {i/24:.10f}    {j/24:.10f}    0.0" for i in range(24) for j in range(24))
        + "\nend kpoints\n"
    )
    out = build_nscf(TPL, ks, "./temp_nscf576/")
    require("nbnd = 400" in out, "nbnd")
    require("disk_io = 'low'" in out, "disk_io")
    require("verbosity = 'high'" in out, "verb")
    require("outdir = './temp_nscf576/'" in out, "outdir")
    require("K_POINTS crystal\n576\n" in out, "nk")
    require("temp_scf_soc" not in out, "prod outdir leaked")
    require("temp_a1h_scf_clean" not in out, "a1o outdir leaked")
    try:
        build_nscf(TPL, ks, "./temp_scf_soc/")
        raise RuntimeError("should refuse production outdir")
    except RuntimeError as e:
        require("refuse" in str(e), e)


def test_chunks(tmp: Path) -> None:
    ks = kpoints_from_win(
        "begin kpoints\n"
        + "\n".join(f"    {i/24:.10f}    {j/24:.10f}    0.0" for i in range(24) for j in range(24))
        + "\nend kpoints\n"
    )
    man = write_chunks(TPL, ks, tmp, 8)
    require(len(man) == 72, len(man))
    require(man[0]["nk"] == 8, man[0])
    require(man[-1]["ik0"] == 568, man[-1])
    text0 = (tmp / "chunk_00.in").read_text(encoding="utf-8")
    require("K_POINTS crystal\n8\n" in text0, "chunk nk")
    require("outdir = './temp_chunk_00/'" in text0, "chunk outdir")
    require("disk_io = 'low'" in text0, "chunk disk_io")
    require("temp_scf_soc" not in text0, "prod leak")
    text_last = (tmp / "chunk_71.in").read_text(encoding="utf-8")
    require("outdir = './temp_chunk_71/'" in text_last, "last outdir")


def main() -> int:
    from tempfile import TemporaryDirectory

    test_kpoints()
    test_build()
    with TemporaryDirectory() as td:
        test_chunks(Path(td))
    print("test_make_nscf576: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
