#!/usr/bin/env python3
"""Tests for UPF-defined C projector radials. No zona scan."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from build_shell_trial_nnkp import expected_counts
from extract_upf_defined_radials import (
    EXPLODE_ABSMAX,
    SHELL_ORDER,
    build_all,
    chi_to_R,
    j_average_chi,
    normalize_R,
    parse_upf_radials,
    qe_radialpart_mesh,
    select_id_first_channel,
    write_fortran_tables,
)


def require(cond: bool, msg: object) -> None:
    if not cond:
        raise RuntimeError(str(msg))


def _chi_block(idx: int, label: str, l: int, r: np.ndarray, R: np.ndarray) -> str:
    chi = r * R
    body = " ".join(f"{x:.8e}" for x in chi)
    return (
        f'<PP_CHI.{idx} index="{idx}" label="{label}" l="{l}" n="1" occupation="1.0">\n'
        f"{body}\n</PP_CHI.{idx}>\n"
    )


def _ps_block(idx: int, label: str, l: int, r: np.ndarray, R: np.ndarray) -> str:
    chi = r * R
    body = " ".join(f"{x:.8e}" for x in chi)
    return (
        f'<PP_PSWFC.{idx} index="{idx}" label="{label}" l="{l}">\n'
        f"{body}\n</PP_PSWFC.{idx}>\n"
    )


def write_fake_upf(path: Path, element: str, chi: list[tuple], pswfc: list[tuple]) -> None:
    r = np.linspace(0.05, 4.0, 21)
    r_xml = " ".join(f"{x:.8e}" for x in r)
    chunks = [
        f'<UPF version="2.0.1"><PP_HEADER element="{element}" mesh_size="21"/>\n',
        f"<PP_MESH><PP_R size=\"21\">\n{r_xml}\n</PP_R></PP_MESH>\n<PP_PSWFC>\n",
    ]
    for idx, label, l, alpha in chi:
        R = 2.0 * (alpha**1.5) * np.exp(-alpha * r)
        chunks.append(_chi_block(idx, label, l, r, R))
    chunks.append("</PP_PSWFC>\n<PP_FULL_WFC>\n")
    for idx, label, l, alpha, explode in pswfc:
        if explode:
            R = np.full_like(r, 1.0e6)
        else:
            R = 2.0 * (alpha**1.5) * np.exp(-alpha * r)
        chunks.append(_ps_block(idx, label, l, r, R))
    chunks.append("</PP_FULL_WFC></UPF>\n")
    path.write_text("".join(chunks), encoding="utf-8")


def test_mesh() -> None:
    r = qe_radialpart_mesh()
    require(r.size == 333, r.size)
    require(abs(r[0] - math.exp(-6.0)) < 1e-12, r[0])
    require(abs(r[-1] - math.exp(-6.0 + 332 * 0.025)) < 1e-12, r[-1])


def test_normalize() -> None:
    r = qe_radialpart_mesh()
    R = 2.0 * np.exp(-r)
    Rn, n0 = normalize_R(r, R)
    w = 0.025 * r
    n1 = float(np.sum((Rn * r) ** 2 * w))
    require(abs(n1 - 1.0) < 1e-9, n1)
    require(n0 > 0, n0)


def test_explode_select(tmp: Path) -> None:
    upf = tmp / "I.rel-pbe.upf"
    write_fake_upf(
        upf,
        "I",
        chi=[(1, "5S", 0, 1.0), (2, "5P", 1, 1.2), (3, "5P", 1, 1.3)],
        pswfc=[
            (7, "5D", 2, 0.8, False),
            (8, "5D", 2, 0.85, False),
            (9, "5D", 2, 1.0, True),
            (10, "5D", 2, 1.0, True),
        ],
    )
    parsed = parse_upf_radials(upf)
    keep, rejected = select_id_first_channel(parsed, "5D")
    require([k["index"] for k in keep] == [7, 8], keep)
    require([k["index"] for k in rejected] == [9, 10], rejected)
    require(all(k["abs_max"] < EXPLODE_ABSMAX for k in keep), keep)


def test_build_tables(tmp: Path) -> None:
    cr = tmp / "Cr.rel-pbe.upf"
    iodine = tmp / "I.rel-pbe.upf"
    write_fake_upf(
        cr,
        "Cr",
        chi=[
            (1, "3S", 0, 2.0),
            (2, "4S", 0, 1.0),
            (3, "3P", 1, 1.1),
            (4, "3P", 1, 1.15),
            (5, "3D", 2, 1.4),
            (6, "3D", 2, 1.45),
        ],
        pswfc=[],
    )
    write_fake_upf(
        iodine,
        "I",
        chi=[(1, "5S", 0, 1.0), (2, "5P", 1, 1.2), (3, "5P", 1, 1.3)],
        pswfc=[
            (7, "5D", 2, 0.6, False),
            (8, "5D", 2, 0.62, False),
            (9, "5D", 2, 1.0, True),
            (10, "5D", 2, 1.0, True),
        ],
    )
    r = qe_radialpart_mesh()
    shells = build_all(parse_upf_radials(cr), parse_upf_radials(iodine), r)
    require(set(shells) == set(SHELL_ORDER), shells.keys())
    require(shells["I_d"]["channel_indices"] == [7, 8], shells["I_d"])
    require(shells["I_d"]["rejected_exploding_indices"] == [9, 10], shells["I_d"])
    require(shells["Cr_s"]["channel_indices"] == [2], shells["Cr_s"])
    require(shells["I_d"]["mean_r_bohr"] > shells["Cr_d"]["mean_r_bohr"], shells)
    counts = expected_counts()
    n_by = {k: counts[k] for k in SHELL_ORDER}
    rid = tmp / "seed.radial_id"
    rR = tmp / "seed.radial_R"
    write_fortran_tables(r, shells, n_by, rid, rR)
    ids = [int(x) for x in rid.read_text().split()[1:]]
    require(len(ids) == 288, len(ids))
    require(ids[0] == 1 and ids[-1] == 6, (ids[0], ids[-1]))
    header = rR.read_text().splitlines()[0].split()
    require(header == ["6", "333"], header)


def test_j_average() -> None:
    r = np.array([0.1, 0.2, 0.3])
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 2.0, 1.0])
    avg = j_average_chi(r, [a, b], r)
    require(np.allclose(avg, [2.0, 2.0, 2.0]), avg)
    R = chi_to_R(r, avg)
    require(np.allclose(R, avg / r), R)


def main() -> int:
    test_mesh()
    test_normalize()
    test_j_average()
    tmp = Path(__file__).resolve().parents[2] / ".codex_tmp" / "upf_c_radial_test"
    tmp.mkdir(parents=True, exist_ok=True)
    test_explode_select(tmp)
    test_build_tables(tmp)
    print("test_extract_upf_defined_radials: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
