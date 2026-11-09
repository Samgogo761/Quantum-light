#!/usr/bin/env python3
"""Minimal tests for I-d nnkp/radial layout parser."""
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from verify_232_id_radial import parse_spinor_projections, radial_R_tags


def require(cond: bool, msg: object) -> None:
    if not cond:
        raise RuntimeError(str(msg))


def test_parse_spinor(tmp: Path) -> None:
    lines = ["begin spinor_projections", "2"]
    for l, mr in ((2, 1), (1, 2)):
        lines.append(f"   0.10000    0.20000    0.30000     {l}  {mr}  1")
        lines.append("    0.0000000  0.0000000  1.0000000   1.0000000  0.0000000  0.0000000    1.00")
        lines.append("    1   0.0000000  0.0000000  1.0000000")
    lines.append("end spinor_projections")
    p = tmp / "x.nnkp"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    recs = parse_spinor_projections(p)
    require(len(recs) == 2, recs)
    require(recs[0]["l"] == 2, recs[0])
    require(recs[1]["l"] == 1, recs[1])


def test_radial_tags(tmp: Path) -> None:
    r = ["0.1", "0.2"]
    lines = ["3 2", *r, "Cr_d 2", *r, "I_p 1", *r, "I_d 2", *r]
    p = tmp / "x.radial_R"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    require(radial_R_tags(p) == [("Cr_d", 2), ("I_p", 1), ("I_d", 2)], radial_R_tags(p))


def main() -> int:
    tmp = Path(__file__).resolve().parent / "_tmp_verify_232"
    tmp.mkdir(exist_ok=True)
    try:
        test_parse_spinor(tmp)
        test_radial_tags(tmp)
    finally:
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    print("test_verify_232_id_radial: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
