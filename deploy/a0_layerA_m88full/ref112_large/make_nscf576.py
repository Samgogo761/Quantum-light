#!/usr/bin/env python3
"""Build isolated 576-k nbnd=400 NSCF input from the 8-k count template
and the production Wannier90 k-grid. Does not write temp_scf_soc."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORBIDDEN_OUTDIR = ("temp_scf_soc", "temp_a1h_scf_clean")


def kpoints_from_win(win_text: str) -> list[str]:
    lines = win_text.splitlines()
    on = False
    ks: list[str] = []
    for ln in lines:
        s = ln.strip().lower()
        if s.startswith("begin kpoints"):
            on = True
            continue
        if s.startswith("end kpoints"):
            break
        if on and ln.strip():
            parts = ln.split()
            if len(parts) < 3:
                raise RuntimeError(f"bad k-point line: {ln!r}")
            ks.append(f"  {float(parts[0]):.10f}  {float(parts[1]):.10f}  {float(parts[2]):.10f}  1.0")
    if len(ks) != 576:
        raise RuntimeError(f"expected 576 production k-points, got {len(ks)}")
    return ks


def build_nscf(
    template: str,
    klines: list[str],
    outdir: str,
    disk_io: str = "low",
) -> str:
    for bad in FORBIDDEN_OUTDIR:
        if bad in outdir:
            raise RuntimeError(f"refuse outdir containing {bad}")
    if disk_io not in ("low", "high"):
        raise RuntimeError(f"bad disk_io {disk_io}")
    text = template
    text = re.sub(r"outdir\s*=\s*'[^']*'", f"outdir = '{outdir}'", text, count=1, flags=re.I)
    text = re.sub(r"disk_io\s*=\s*'[^']*'", f"disk_io = '{disk_io}'", text, count=1, flags=re.I)
    if "nbnd = 400" not in text:
        raise RuntimeError("template must already be nbnd=400")
    m = re.search(r"(?m)^K_POINTS crystal\s*$", text)
    if not m:
        raise RuntimeError("no K_POINTS crystal")
    m_hub = re.search(r"(?m)^HUBBARD", text)
    if not m_hub:
        raise RuntimeError("no HUBBARD")
    head = text[: m.start()]
    tail = text[m_hub.start() :]
    body = f"K_POINTS crystal\n{len(klines)}\n" + "\n".join(klines) + "\n"
    out = head + body + tail
    for bad in FORBIDDEN_OUTDIR:
        if bad in out:
            raise RuntimeError(f"generated input still mentions {bad}")
    return out


def write_chunks(
    template: str,
    klines: list[str],
    chunk_dir: Path,
    chunk_size: int = 8,
) -> list[dict]:
    if len(klines) % chunk_size != 0:
        raise RuntimeError(f"{len(klines)} k-points not divisible by {chunk_size}")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    n_chunk = len(klines) // chunk_size
    for i in range(n_chunk):
        sl = klines[i * chunk_size : (i + 1) * chunk_size]
        outdir = f"./temp_chunk_{i:02d}/"
        text = build_nscf(template, sl, outdir, disk_io="low")
        path = chunk_dir / f"chunk_{i:02d}.in"
        path.write_text(text, encoding="utf-8")
        manifest.append(
            {
                "index": i,
                "nscf_in": str(path),
                "outdir": outdir,
                "save": f"{outdir}CrI3_scf_soc.save",
                "ik0": i * chunk_size,
                "nk": len(sl),
            }
        )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--win", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("CrI3_nscf_soc_576.in"))
    ap.add_argument("--outdir", default="./temp_nscf576/")
    ap.add_argument("--chunks", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--chunk-dir", type=Path, default=Path("chunks"))
    ap.add_argument("--manifest", type=Path, default=Path("chunks/manifest.json"))
    args = ap.parse_args()
    klines = kpoints_from_win(args.win.read_text(encoding="utf-8", errors="replace"))
    template = args.template.read_text(encoding="utf-8", errors="replace")
    nscf = build_nscf(template, klines, args.outdir, disk_io="low")
    args.out.write_text(nscf, encoding="utf-8")
    print(f"wrote {args.out} nk={len(klines)} outdir={args.outdir}")
    if args.chunks:
        man = write_chunks(template, klines, args.chunk_dir, args.chunk_size)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps({"n_k": len(klines), "chunk_size": args.chunk_size, "chunks": man}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(man)} chunks -> {args.chunk_dir} manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
