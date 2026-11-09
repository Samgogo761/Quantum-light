#!/usr/bin/env python3
"""Tests for merging QE nscf chunk saves."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_nscf_chunks import merge_saves, merge_xml


def require(cond: bool, msg: object) -> None:
    if not cond:
        raise RuntimeError(str(msg))


def mini_xml(nk: int, outdir: str, coords: list[str]) -> str:
    ibz = "\n".join(
        f'      <k_point weight="1.00000000000000">{c}</k_point>' for c in coords
    )
    start = "\n".join(
        f'        <k_point weight="1.00000000000000">{c}</k_point>' for c in coords
    )
    w = f"{1.0 / nk:.15f}"
    ks = []
    for i, c in enumerate(coords):
        ks.append(
            "      <ks_energies>\n"
            f'        <k_point weight="{w}">{c}</k_point>\n'
            "        <npw>10</npw>\n"
            f'        <eigenvalues size="400">{float(i)}</eigenvalues>\n'
            '        <occupations size="400">1.0</occupations>\n'
            "      </ks_energies>"
        )
    return (
        '<?xml version="1.0"?>\n'
        "<qes:espresso>\n"
        "  <input>\n"
        "    <control_variables>\n"
        "      <calculation>nscf</calculation>\n"
        f"      <outdir>{outdir}</outdir>\n"
        "    </control_variables>\n"
        "    <bands><nbnd>400</nbnd></bands>\n"
        "  </input>\n"
        "  <output>\n"
        "    <band_structure>\n"
        "      <nbnd>400</nbnd>\n"
        "      <wf_collected>true</wf_collected>\n"
        "      <k_points_IBZ>\n"
        f"      <nk>{nk}</nk>\n"
        f"{ibz}\n"
        "    </k_points_IBZ>\n"
        "      <starting_k_points>\n"
        f"        <nk>{nk}</nk>\n"
        f"{start}\n"
        "      </starting_k_points>\n"
        f"      <nks>{nk}</nks>\n"
        + "\n".join(ks)
        + "\n    </band_structure>\n"
        "  </output>\n"
        "</qes:espresso>\n"
    )


def test_merge_xml() -> None:
    a = mini_xml(2, "./temp_chunk_00/", ["0.0 0.0 0.0", "0.1 0.0 0.0"])
    b = mini_xml(2, "./temp_chunk_01/", ["0.2 0.0 0.0", "0.3 0.0 0.0"])
    out = merge_xml([a, b], "./temp_nscf576/", 2)
    require("<nks>4</nks>" in out, "nks")
    require(out.count("<ks_energies>") == 4, "ks")
    require("<nk>4</nk>" in out, "nk")
    require("./temp_nscf576/" in out, "outdir")
    require("temp_chunk_" not in out.split("<outdir>")[1].split("</outdir>")[0], "chunk outdir left")
    require("0.001736111111111" not in out, "wrong 576 weight")
    require(f"{1.0 / 4:.15f}" in out, "weight 1/4")
    require("temp_scf_soc" not in out, "prod")
    try:
        merge_xml([a, b], "./temp_scf_soc/", 2)
        raise RuntimeError("should refuse production outdir")
    except RuntimeError as e:
        require("refuse" in str(e), e)


def test_merge_saves(tmp: Path) -> None:
    chunks = []
    coords = [
        (["0.0 0.0 0.0", "0.1 0.0 0.0"], "00"),
        (["0.2 0.0 0.0", "0.3 0.0 0.0"], "01"),
    ]
    for xy, tag in coords:
        save = tmp / f"temp_chunk_{tag}" / "CrI3_scf_soc.save"
        save.mkdir(parents=True)
        (save / "data-file-schema.xml").write_text(
            mini_xml(2, f"./temp_chunk_{tag}/", xy), encoding="utf-8"
        )
        for i in range(1, 3):
            (save / f"wfc{i}.dat").write_bytes(f"{tag}-{i}".encode())
        chunks.append(save)
    dest = tmp / "temp_nscf576" / "CrI3_scf_soc.save"
    dest.mkdir(parents=True)
    (dest / "charge-density.dat").write_bytes(b"charge")
    info = merge_saves(chunks, dest, "./temp_nscf576/", 2, 4)
    require(info["n_k"] == 4, info)
    xml = (dest / "data-file-schema.xml").read_text(encoding="utf-8")
    require("<nks>4</nks>" in xml, xml[:400])
    require((dest / "wfc1.dat").read_bytes() == b"00-1", "wfc1")
    require((dest / "wfc4.dat").read_bytes() == b"01-2", "wfc4")
    try:
        merge_saves(chunks, tmp / "temp_scf_soc" / "CrI3_scf_soc.save", "./x/", 2, 4)
        raise RuntimeError("should refuse dest")
    except RuntimeError as e:
        require("refuse" in str(e), e)


def main() -> int:
    from tempfile import TemporaryDirectory

    test_merge_xml()
    with TemporaryDirectory() as td:
        test_merge_saves(Path(td))
    print("test_merge_nscf_chunks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
