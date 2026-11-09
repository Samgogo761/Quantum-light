#!/usr/bin/env python3
"""Merge successful 8-k QE NSCF chunks into one 576-k save for pw2wannier90.

This concatenates QE-written nscf xml/wfc from isolated chunks. It does not
invent SCF eigenvalues, and it refuses production/A1-O paths.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

FORBIDDEN = ("temp_scf_soc", "temp_a1h_scf_clean")
KPOINT_RE = re.compile(r'<k_point weight="[^"]+">[^<]+</k_point>')
KS_RE = re.compile(r"<ks_energies>.*?</ks_energies>", re.S)
NKS_RE = re.compile(r"<nks>[^<]+</nks>")
NK_RE = re.compile(r"<nk>[^<]+</nk>")
OUTDIR_RE = re.compile(r"<outdir>[^<]+</outdir>")
W_IBZ = "1.00000000000000"


def refuse_path(path: Path) -> None:
    text = str(path).replace("\\", "/")
    for bad in FORBIDDEN:
        if bad in text:
            raise RuntimeError(f"refuse path containing {bad}: {path}")


def _one_block(xml: str, tag: str) -> str:
    matches = re.findall(rf"<{tag}>.*?</{tag}>", xml, flags=re.S)
    if len(matches) != 1:
        raise RuntimeError(f"expected 1 <{tag}> block, got {len(matches)}")
    return matches[0]


def parse_chunk_xml(xml: str, expect_nk: int) -> dict:
    if "temp_scf_soc" in xml or "temp_a1h_scf_clean" in xml:
        raise RuntimeError("chunk xml mentions a forbidden outdir")
    nks = NKS_RE.findall(xml)
    if nks != [f"<nks>{expect_nk}</nks>"]:
        raise RuntimeError(f"bad nks {nks} expected {expect_nk}")
    ibz = _one_block(xml, "k_points_IBZ")
    start = _one_block(xml, "starting_k_points")
    ibz_k = KPOINT_RE.findall(ibz)
    start_k = KPOINT_RE.findall(start)
    ks = KS_RE.findall(xml)
    if len(ibz_k) != expect_nk:
        raise RuntimeError(f"k_points_IBZ has {len(ibz_k)} k, expected {expect_nk}")
    if len(start_k) != expect_nk:
        raise RuntimeError(f"starting_k_points has {len(start_k)} k, expected {expect_nk}")
    if len(ks) != expect_nk:
        raise RuntimeError(f"ks_energies has {len(ks)} blocks, expected {expect_nk}")
    nbnd = set(re.findall(r"<nbnd>([^<]+)</nbnd>", xml))
    if nbnd != {"400"}:
        raise RuntimeError(f"nbnd not 400: {nbnd}")
    wf = re.findall(r"<wf_collected>([^<]+)</wf_collected>", xml)
    if wf != ["true"]:
        raise RuntimeError(f"wf_collected {wf}")
    calc = re.findall(r"<calculation>([^<]+)</calculation>", xml)
    if "nscf" not in calc:
        raise RuntimeError(f"calculation {calc}")
    return {"ibz": ibz_k, "start": start_k, "ks": ks}


def _fmt_ks_weight(n_k: int) -> str:
    return f"{1.0 / n_k:.15f}"


def rewrite_ks_weight(block: str, weight: str) -> str:
    out, n = re.subn(
        r'(<k_point weight=")[^"]+(">)',
        rf"\g<1>{weight}\g<2>",
        block,
        count=1,
    )
    if n != 1:
        raise RuntimeError("failed to rewrite ks_energies weight")
    return out


def rebuild_klist(tag: str, kpts: list[str], n: int, indent: str) -> str:
    body = [f"<{tag}>", f"{indent}<nk>{n}</nk>"]
    body.extend(f"{indent}{kp}" for kp in kpts)
    close = "    " if indent == "      " else "      "
    body.append(f"{close}</{tag}>")
    return "\n".join(body)


def merge_xml(chunk_xmls: list[str], dest_outdir: str, chunk_nk: int) -> str:
    if dest_outdir in (None, "") or any(bad in dest_outdir for bad in FORBIDDEN):
        raise RuntimeError(f"refuse dest outdir {dest_outdir!r}")
    parsed = [parse_chunk_xml(x, chunk_nk) for x in chunk_xmls]
    ibz = [kp for p in parsed for kp in p["ibz"]]
    start = [kp for p in parsed for kp in p["start"]]
    n_k = len(ibz)
    if n_k != len(chunk_xmls) * chunk_nk:
        raise RuntimeError("k-point count mismatch")
    w_ks = _fmt_ks_weight(n_k)
    ks = [rewrite_ks_weight(blk, w_ks) for p in parsed for blk in p["ks"]]
    xml = chunk_xmls[0]
    xml = re.sub(r"<k_points_IBZ>.*?</k_points_IBZ>", rebuild_klist("k_points_IBZ", ibz, n_k, "      "), xml, count=1, flags=re.S)
    xml = re.sub(r"<starting_k_points>.*?</starting_k_points>", rebuild_klist("starting_k_points", start, n_k, "        "), xml, count=1, flags=re.S)
    n_nks = len(NKS_RE.findall(xml))
    if n_nks != 1:
        raise RuntimeError(f"expected 1 nks, got {n_nks}")
    xml = NKS_RE.sub(f"<nks>{n_k}</nks>", xml, count=1)
    first = xml.find("<ks_energies>")
    last = xml.rfind("</ks_energies>")
    if first < 0 or last < 0:
        raise RuntimeError("no ks_energies in template xml")
    last += len("</ks_energies>")
    xml = xml[:first] + ("\n      ".join(ks)) + xml[last:]
    n_out = len(OUTDIR_RE.findall(xml))
    if n_out != 1:
        raise RuntimeError(f"expected 1 outdir, got {n_out}")
    xml = OUTDIR_RE.sub(f"<outdir>{dest_outdir}</outdir>", xml, count=1)
    check = parse_chunk_xml(xml, n_k)
    if len(check["ks"]) != n_k:
        raise RuntimeError("merged xml ks_energies count failed")
    nk_vals = NK_RE.findall(xml)
    if nk_vals != [f"<nk>{n_k}</nk>", f"<nk>{n_k}</nk>"]:
        raise RuntimeError(f"merged nk {nk_vals}")
    if any(bad in xml for bad in FORBIDDEN):
        raise RuntimeError("merged xml mentions a forbidden outdir")
    return xml


def list_wfc(save: Path, n: int) -> list[Path]:
    files = []
    for i in range(1, n + 1):
        p = save / f"wfc{i}.dat"
        if not p.is_file() or p.stat().st_size <= 0:
            raise RuntimeError(f"missing or empty {p}")
        files.append(p)
    extra = sorted(save.glob("wfc*.dat"))
    if len(extra) != n:
        raise RuntimeError(f"{save} has {len(extra)} wfc, expected {n}")
    return files


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists() or dst.is_symlink():
        if dst.exists() and src.resolve() == dst.resolve():
            return "exists"
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def merge_saves(
    chunk_saves: list[Path],
    dest_save: Path,
    dest_outdir: str,
    chunk_nk: int,
    expected_nk: int,
) -> dict:
    refuse_path(dest_save)
    for p in chunk_saves:
        refuse_path(p)
    if expected_nk != len(chunk_saves) * chunk_nk:
        raise RuntimeError("expected_nk != n_chunks * chunk_nk")
    dest_save.mkdir(parents=True, exist_ok=True)
    xmls = []
    for save in chunk_saves:
        xml_path = save / "data-file-schema.xml"
        if not xml_path.is_file():
            raise RuntimeError(f"no xml in {save}")
        xmls.append(xml_path.read_text(encoding="utf-8", errors="replace"))
        list_wfc(save, chunk_nk)
    merged = merge_xml(xmls, dest_outdir, chunk_nk)
    (dest_save / "data-file-schema.xml").write_text(merged, encoding="utf-8")
    copied = {"hardlink": 0, "copy": 0, "exists": 0}
    for ic, save in enumerate(chunk_saves):
        for j in range(1, chunk_nk + 1):
            src = save / f"wfc{j}.dat"
            dst = dest_save / f"wfc{ic * chunk_nk + j}.dat"
            copied[link_or_copy(src, dst)] += 1
    list_wfc(dest_save, expected_nk)
    charge = dest_save / "charge-density.dat"
    if not charge.is_file():
        raise RuntimeError(f"merged save missing charge-density.dat: {dest_save}")
    return {
        "n_k": expected_nk,
        "n_chunks": len(chunk_saves),
        "wfc": copied,
        "xml": str(dest_save / "data-file-schema.xml"),
    }


def default_chunk_saves(root: Path, n_chunks: int) -> list[Path]:
    return [root / f"temp_chunk_{i:02d}" / "CrI3_scf_soc.save" for i in range(n_chunks)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--dest-save", type=Path, default=Path("temp_nscf576/CrI3_scf_soc.save"))
    ap.add_argument("--dest-outdir", default="./temp_nscf576/")
    ap.add_argument("--chunk-nk", type=int, default=8)
    ap.add_argument("--n-chunks", type=int, default=72)
    ap.add_argument("--expected-nk", type=int, default=576)
    args = ap.parse_args()
    refuse_path(args.root)
    refuse_path(args.dest_save)
    info = merge_saves(
        default_chunk_saves(args.root, args.n_chunks),
        args.dest_save if args.dest_save.is_absolute() else args.root / args.dest_save,
        args.dest_outdir,
        args.chunk_nk,
        args.expected_nk,
    )
    print("MERGE_OK", info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
