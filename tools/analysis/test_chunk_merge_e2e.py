#!/usr/bin/env python3
"""Lightweight merge-path unit test (NOT a substitute for real Fortran chunk smoke).

Synthesizes chunk_info + chunk_weighted_spectrum stubs so the hardened merge
gate can be exercised offline. Continuum fidelity requires
sbatch_full112_gh3_chunk_smoke_5plus4.sh + compare_chunked_vs_full.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

from hhg_fft_utils import parse_input_nml, pick_ics_cs_at_order
from merge_chunked_ensemble import merge_ensemble
from validate_qlight_nodes_production import load_manifest


def load_modes(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        rows.append(
            {
                "id": int(f[0]),
                "weight": float(f[1]),
                "intensity": float(f[2]),
                "phase": float(f[3]),
                "order": int(f[4]),
                "jx": complex(float(f[5]), float(f[6])),
                "jy": complex(float(f[7]), float(f[8])),
                "power": float(f[9]),
            }
        )
    return rows


def write_chunk(
    outdir: Path,
    rows: list[dict],
    meta_src: Path,
    prop_ids: list[int],
    n_manifest: int,
    harmonics: list[int],
    scale: float,
    continuum_h: list[float],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    header = [
        "# chunk subset",
        "# id weight I phi order ReJx ImJx ReJy ImJy |J|^2_scaled",
    ]
    lines = []
    for r in rows:
        lines.append(
            f"{r['id']:8d}"
            f"{r['weight']:25.17e}"
            f"{r['intensity']:25.17e}"
            f"{r['phase']:25.17e}"
            f"{r['order']:6d}"
            f"{r['jx'].real:25.17e}{r['jx'].imag:25.17e}"
            f"{r['jy'].real:25.17e}{r['jy'].imag:25.17e}"
            f"{r['power']:25.17e}"
        )
    (outdir / "HHG_nodes_modes.dat").write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    if not meta_src.is_file():
        raise FileNotFoundError(f"run_metadata.txt required for merge gate: {meta_src}")
    (outdir / "run_metadata.txt").write_text(meta_src.read_text(encoding="utf-8"), encoding="utf-8")
    (outdir / "SUCCESS").write_text("", encoding="utf-8")
    (outdir / "run_status.txt").write_text("status=PASS\nexit_code=0\n", encoding="utf-8")
    ids_csv = ",".join(str(i) for i in prop_ids)
    (outdir / "chunk_info.txt").write_text(
        "\n".join(
            [
                "# partial node-chunk propagation (ICS/CS not final)",
                f"n_manifest_nodes = {n_manifest}",
                f"n_propagate_nodes = {len(prop_ids)}",
                f"propagate_ids = {ids_csv}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # Shared continuum grid across chunks; put mode power only at exact harmonic bins.
    by_h = {h: [r for r in rows if r["order"] == h] for h in harmonics}
    spec = [
        "# stub continuum for offline merge-gate test (NOT Fortran continuum fidelity)",
        f"# spectrum_scale = {scale:.17e}",
        "# iw  harmonic_order  omega(a.u.)  weighted_ics  ReSumJx ImSumJx ReSumJy ImSumJy",
    ]
    for iw, h in enumerate(continuum_h, start=1):
        omega = float(h)  # stub ω∝h; shared across chunks
        group = by_h.get(int(round(h)), []) if abs(h - round(h)) < 1.0e-12 else []
        if abs(h - round(h)) < 1.0e-12 and int(round(h)) in by_h:
            group = by_h[int(round(h))]
        ics = sum(r["weight"] * r["power"] for r in group)
        jx = sum(r["weight"] * r["jx"] for r in group)
        jy = sum(r["weight"] * r["jy"] for r in group)
        spec.append(
            f"{iw:8d}{float(h):25.17e}{omega:25.17e}{ics:25.17e}"
            f"{jx.real:25.17e}{jx.imag:25.17e}{jy.real:25.17e}{jy.imag:25.17e}"
        )
    (outdir / "chunk_weighted_spectrum.dat").write_text("\n".join(spec) + "\n", encoding="utf-8")


def infer_scale(rows: list[dict]) -> float:
    scales = []
    for r in rows:
        raw = abs(r["jx"]) ** 2 + abs(r["jy"]) ** 2
        if raw > 1.0e-280:
            scales.append(r["power"] / raw)
    return sum(scales) / len(scales)


def chunk_ids(ids: list[int], chunk_size: int) -> list[list[int]]:
    return [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-run-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, default=5)
    ap.add_argument("--rtol", type=float, default=2.0e-7)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    modes = load_modes(args.full_run_dir / "HHG_nodes_modes.dat")
    manifest = load_manifest(args.manifest)
    node_ids = sorted({int(r["id"]) for r in manifest})
    orders = sorted({r["order"] for r in modes})
    chunks = chunk_ids(node_ids, args.chunk_size)
    scale = infer_scale(modes)

    # Patch metadata with nodes_sha256 of the provided manifest if absent.
    meta_text = (args.full_run_dir / "run_metadata.txt").read_text(encoding="utf-8")
    msha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    required = [
        "source_sha256",
        "binary_sha256",
        "tb_sha256",
        "nodes_sha256",
        "template_sha256",
        "nk",
        "model",
        "wvl_nm",
        "dt",
        "T2_cycles",
        "squeeze_r",
        "squeeze_theta_deg",
        "I_bar",
        "harmonics",
    ]
    present = {line.split("=", 1)[0] for line in meta_text.splitlines() if "=" in line}
    for key in required:
        if key not in present:
            meta_text += f"{key}=test_{key}\n"
    meta_text = re.sub(r"^nodes_sha256=.*$", f"nodes_sha256={msha}", meta_text, flags=re.M)
    if "nodes_sha256=" not in meta_text:
        meta_text += f"nodes_sha256={msha}\n"

    grid = parse_input_nml(args.full_run_dir / "input.nml")
    dt = float(grid["dt_au"])
    nt = int(grid["nt"])
    omega0 = float(grid["omega0"])
    ref: dict[int, dict[str, float]] = {}
    for order in orders:
        ref[order] = pick_ics_cs_at_order(
            args.full_run_dir / "HHG_ics_cs.dat", order, nt=nt, dt=dt, omega0=omega0
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        meta_path = tmp_path / "run_metadata.txt"
        meta_path.write_text(meta_text, encoding="utf-8")
        # Dense-enough stub continuum that includes all target harmonics.
        continuum_h = sorted({float(h) for h in orders} | {0.0, 1.0, 3.0, 4.0, 6.0, 8.0, 11.0})
        chunk_dirs: list[Path] = []
        for ci, ids in enumerate(chunks):
            cdir = tmp_path / f"chunk_{ci:02d}"
            chunk_dirs.append(cdir)
            subset = [r for r in modes if r["id"] in ids]
            write_chunk(
                cdir, subset, meta_path, ids, len(manifest), orders, scale, continuum_h
            )

        merged = tmp_path / "merged"
        merge_ensemble(args.manifest, chunk_dirs, merged, orders)

        diffs: dict[str, dict[str, float]] = {}
        ok = True
        for order in orders:
            got = pick_ics_cs_at_order(merged / "HHG_ics_cs.dat", order, nt=nt, dt=dt, omega0=omega0)
            for key in ("ICS", "CS"):
                a = ref[order][key]
                b = got[key]
                rel = abs(a - b) / max(abs(a), 1.0e-300)
                diffs[f"H{order}_{key}"] = {"ref": a, "merged": b, "rel_err": rel}
                if rel > args.rtol:
                    ok = False
        # Continuum file must contain more than just the five target harmonics.
        n_cont = sum(
            1
            for line in (merged / "HHG_ics_cs.dat").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
        if n_cont < len(continuum_h):
            ok = False
            diffs["continuum_rows"] = {"ref": float(len(continuum_h)), "merged": float(n_cont), "rel_err": 1.0}

        report = {
            "status": "PASS" if ok else "FAIL",
            "full_run_dir": str(args.full_run_dir.resolve()),
            "n_chunks": len(chunks),
            "chunk_size": args.chunk_size,
            "diffs": diffs,
            "warning": (
                "This is an offline merge-gate stub with synthetic spectra. "
                "Real Fortran chunk smoke is required before GH5."
            ),
        }
        text = json.dumps(report, indent=2) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text, encoding="utf-8")
        print(text)
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
