#!/usr/bin/env python3
"""Merge partial node-chunk HHG outputs into a full ensemble result.

Requirements:
  - Full manifest with sum(weights)=1.
  - Each chunk: SUCCESS, run_status=PASS, run_metadata (14 keys),
    chunk_info.txt, chunk_weighted_spectrum.dat, HHG_nodes_modes.dat.
  - spectrum_scale finite and > 0; full FFT continuum written with real h(ω),ω.
  - Does NOT compare input_sha256 across chunks (propagate_ids differ by design).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from hhg_fft_utils import harmonic_fft_index, parse_input_nml
from validate_qlight_nodes_production import load_manifest

PROVENANCE_KEYS = (
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
)
WEIGHT_TOL = 1.0e-12
# Match Fortran nodes weight-sum gate (do NOT regenerate 27992 manifests).
WEIGHT_SUM_TOL = 5.0e-8
INTENSITY_RTOL = 1.0e-8
PHASE_TOL = 1.0e-6
OMEGA_RTOL = 1.0e-12
HORDER_ATOL = 1.0e-10
NEG_VAR_RTOL = 1.0e-10
NODE_SPECTRUM_RTOL = 2.0e-7
# Hard physics keys written into merge provenance (template SHA is build-level).
PHYSICS_KEYS = (
    "tb_sha256",
    "nodes_sha256",
    "nk",
    "model",
    "wvl_nm",
    "dt",
    "T2_cycles",
    "squeeze_r",
    "squeeze_theta_deg",
    "I_bar",
    "harmonics",
)
BUILD_KEYS = ("source_sha256", "binary_sha256", "template_sha256")
RUNTIME_KEYS = ("omp_num_threads", "mkl_num_threads")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_modes(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 10:
            raise ValueError(f"{path}: expected 10 columns")
        vals = [float(x) for x in f[1:4] + f[5:]]
        if not all(math.isfinite(x) for x in vals):
            raise ValueError(f"{path}: non-finite values")
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


def read_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required run_metadata.txt: {path}")
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def require_provenance(meta: dict[str, str], cdir: Path) -> None:
    missing = [k for k in PROVENANCE_KEYS if not meta.get(k)]
    if missing:
        raise ValueError(f"{cdir}: empty/missing provenance keys: {missing}")
    runtime_missing = [k for k in RUNTIME_KEYS if not meta.get(k)]
    if runtime_missing:
        raise ValueError(f"{cdir}: empty/missing runtime keys: {runtime_missing}")


def verify_output_sha256(cdir: Path) -> dict[str, str]:
    manifest = cdir / "output_sha256.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"{cdir}: missing output_sha256.txt")
    checked: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"{cdir}: malformed output_sha256.txt line: {line!r}")
        digest, name = parts[0], parts[-1]
        path = cdir / name
        if not path.is_file():
            raise FileNotFoundError(f"{cdir}: output_sha256 lists missing file {name}")
        got = sha256(path)
        if got != digest:
            raise ValueError(f"{cdir}: output_sha256 mismatch {name}: {got} != {digest}")
        checked[name] = digest
    if not checked:
        raise ValueError(f"{cdir}: output_sha256.txt has no entries")
    return checked


def require_chunk_success(cdir: Path) -> None:
    if not (cdir / "SUCCESS").is_file():
        raise FileNotFoundError(f"{cdir}: missing SUCCESS")
    status_path = cdir / "run_status.txt"
    if not status_path.is_file():
        raise FileNotFoundError(f"{cdir}: missing run_status.txt")
    status = None
    for raw in status_path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("status="):
            status = raw.split("=", 1)[1].strip()
            break
    if status != "PASS":
        raise ValueError(f"{cdir}: run_status status={status!r}, expected PASS")


def parse_chunk_info(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing required chunk_info.txt: {path}")
    info: dict = {"propagate_ids": []}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "propagate_ids":
            tokens = [x for x in re.split(r"[,\s]+", v) if x]
            if not tokens:
                raise ValueError(f"{path}: propagate_ids empty")
            # Reject ambiguous separators like trailing empties already stripped by split;
            # still forbid raw empties by comparing token count to commas.
            if re.search(r",\s*,", v) or v.strip().endswith(",") or v.strip().startswith(","):
                raise ValueError(f"{path}: invalid empty token in propagate_ids: {v!r}")
            info["propagate_ids"] = [int(x) for x in tokens]
        elif k.startswith("n_"):
            info[k] = int(v)
        else:
            info[k] = v
    if not info["propagate_ids"]:
        raise ValueError(f"{path}: propagate_ids empty")
    return info


def load_chunk_spectrum(path: Path) -> tuple[float, list[dict]]:
    """Format: iw  harmonic_order  omega  ics  ReJx ImJx ReJy ImJy (ES25.17)."""
    scale: float | None = None
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# spectrum_scale"):
            scale = float(line.split("=", 1)[1].strip())
            continue
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 8:
            raise ValueError(f"{path}: spectrum row needs 8 fields, got {len(f)}: {line}")
        iw = int(float(f[0]))
        h, omega, ics = float(f[1]), float(f[2]), float(f[3])
        jx = complex(float(f[4]), float(f[5]))
        jy = complex(float(f[6]), float(f[7]))
        if not all(math.isfinite(x) for x in (h, omega, ics, jx.real, jx.imag, jy.real, jy.imag)):
            raise ValueError(f"{path}: non-finite spectrum row iw={iw}")
        rows.append({"iw": iw, "h_order": h, "omega": omega, "ics": ics, "jx": jx, "jy": jy})
    if scale is None:
        raise ValueError(f"{path}: missing # spectrum_scale header")
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(f"{path}: spectrum_scale must be finite and > 0, got {scale!r}")
    if not rows:
        raise ValueError(f"{path}: no spectrum rows")
    for i, row in enumerate(rows):
        if row["iw"] != i + 1:
            raise ValueError(f"{path}: FFT bin not contiguous at row {i}: iw={row['iw']}")
    return scale, rows


def validate_spectrum_grid(ref: list[dict], other: list[dict], cdir: Path) -> None:
    if len(ref) != len(other):
        raise ValueError(f"{cdir}: spectrum length {len(other)} != {len(ref)}")
    for i, (a, b) in enumerate(zip(ref, other)):
        if a["iw"] != b["iw"]:
            raise ValueError(f"{cdir}: FFT bin mismatch at i={i}: {a['iw']} vs {b['iw']}")
        if abs(a["omega"] - b["omega"]) > OMEGA_RTOL * max(abs(a["omega"]), 1.0):
            raise ValueError(f"{cdir}: omega mismatch at iw={a['iw']}")
        if abs(a["h_order"] - b["h_order"]) > HORDER_ATOL:
            raise ValueError(f"{cdir}: harmonic_order mismatch at iw={a['iw']}")


def validate_provenance(meta_ref: dict[str, str], meta: dict[str, str], cdir: Path) -> None:
    for key in PROVENANCE_KEYS:
        if meta_ref.get(key) != meta.get(key):
            raise ValueError(
                f"provenance mismatch {key} in {cdir}: {meta_ref.get(key)!r} vs {meta.get(key)!r}"
            )


def validate_node_against_manifest(row: dict, ref: dict) -> None:
    if abs(row["weight"] - float(ref["weight"])) > WEIGHT_TOL:
        raise ValueError(f"node {row['id']}: weight differs from manifest")
    i_row, i_ref = row["intensity"], float(ref["intensity"])
    if abs(i_row - i_ref) > INTENSITY_RTOL * max(abs(i_row), abs(i_ref), 1.0):
        raise ValueError(f"node {row['id']}: intensity differs from manifest")
    phase_row = row["phase"] % (2.0 * math.pi)
    phase_ref = float(ref["phase"]) % (2.0 * math.pi)
    dphi = abs(phase_row - phase_ref)
    dphi = min(dphi, 2.0 * math.pi - dphi)
    if dphi > PHASE_TOL:
        raise ValueError(f"node {row['id']}: phase differs from manifest")


def merge_ensemble(
    manifest_path: Path,
    chunk_dirs: list[Path],
    outdir: Path,
    harmonics: list[int],
) -> dict:
    manifest_rows = load_manifest(manifest_path)
    manifest = {int(r["id"]): r for r in manifest_rows}
    wsum = sum(float(r["weight"]) for r in manifest_rows)
    if abs(wsum - 1.0) > WEIGHT_SUM_TOL:
        raise ValueError(f"manifest weight sum = {wsum:.16e}, expected 1")
    manifest_sha = sha256(manifest_path)

    merged: dict[tuple[int, int], dict] = {}
    meta_ref: dict[str, str] | None = None
    spectrum_scale: float | None = None
    merged_spec: list[dict] | None = None
    seen_ids: set[int] = set()
    chunk_output_sha256: dict[str, dict[str, str]] = {}

    for cdir in chunk_dirs:
        require_chunk_success(cdir)
        meta = read_metadata(cdir / "run_metadata.txt")
        require_provenance(meta, cdir)
        if meta_ref is None:
            meta_ref = meta
        else:
            validate_provenance(meta_ref, meta, cdir)
            for key in RUNTIME_KEYS:
                if meta_ref.get(key) != meta.get(key):
                    raise ValueError(
                        f"runtime mismatch {key} in {cdir}: "
                        f"{meta_ref.get(key)!r} vs {meta.get(key)!r}"
                    )
        chunk_output_sha256[str(cdir.resolve())] = verify_output_sha256(cdir)
        if meta["nodes_sha256"] != manifest_sha:
            raise ValueError(f"{cdir}: nodes_sha256 != manifest file hash")

        info = parse_chunk_info(cdir / "chunk_info.txt")
        prop_ids = set(info["propagate_ids"])
        if "n_propagate_nodes" in info and int(info["n_propagate_nodes"]) != len(prop_ids):
            raise ValueError(
                f"{cdir}: n_propagate_nodes={info['n_propagate_nodes']} != "
                f"len(propagate_ids)={len(prop_ids)}"
            )
        if "n_manifest_nodes" in info and int(info["n_manifest_nodes"]) != len(manifest):
            raise ValueError(
                f"{cdir}: n_manifest_nodes={info['n_manifest_nodes']} != "
                f"manifest size={len(manifest)}"
            )

        spec_path = cdir / "chunk_weighted_spectrum.dat"
        if not spec_path.is_file():
            raise FileNotFoundError(f"missing required {spec_path}")
        scale, spec_rows = load_chunk_spectrum(spec_path)
        if spectrum_scale is None:
            spectrum_scale = scale
            merged_spec = [
                {
                    "iw": r["iw"],
                    "h_order": r["h_order"],
                    "omega": r["omega"],
                    "ics": r["ics"],
                    "jx": r["jx"],
                    "jy": r["jy"],
                }
                for r in spec_rows
            ]
        else:
            if abs(spectrum_scale - scale) > 1.0e-12 * max(abs(spectrum_scale), 1.0):
                raise ValueError(f"spectrum_scale mismatch in {cdir}")
            validate_spectrum_grid(merged_spec, spec_rows, cdir)
            for i, row in enumerate(spec_rows):
                merged_spec[i]["ics"] += row["ics"]
                merged_spec[i]["jx"] += row["jx"]
                merged_spec[i]["jy"] += row["jy"]

        modes_path = cdir / "HHG_nodes_modes.dat"
        if not modes_path.is_file():
            raise FileNotFoundError(modes_path)
        mode_ids: set[int] = set()
        for row in load_modes(modes_path):
            key = (row["id"], row["order"])
            if key in merged:
                raise ValueError(f"duplicate node/order {key} across chunks")
            if row["id"] not in prop_ids:
                raise ValueError(f"{cdir}: modes node {row['id']} not in chunk_info propagate_ids")
            if row["id"] not in manifest:
                raise ValueError(f"{cdir}: modes node {row['id']} not in manifest")
            validate_node_against_manifest(row, manifest[row["id"]])
            merged[key] = row
            mode_ids.add(row["id"])
        if mode_ids != prop_ids:
            raise ValueError(
                f"{cdir}: chunk_info IDs {sorted(prop_ids)} != modes IDs {sorted(mode_ids)}"
            )
        overlap = seen_ids & mode_ids
        if overlap:
            raise ValueError(f"duplicate node ids across chunks: {sorted(overlap)}")
        seen_ids |= mode_ids

    if seen_ids != set(manifest.keys()):
        missing = sorted(set(manifest.keys()) - seen_ids)[:10]
        extra = sorted(seen_ids - set(manifest.keys()))[:10]
        raise ValueError(f"node coverage mismatch missing={missing} extra={extra}")

    expected = {(nid, h) for nid in manifest for h in harmonics}
    if set(merged.keys()) != expected:
        missing = sorted(expected - set(merged.keys()))[:10]
        extra = sorted(set(merged.keys()) - expected)[:10]
        raise ValueError(f"coverage mismatch missing={missing} extra={extra}")

    assert spectrum_scale is not None and merged_spec is not None
    scale = spectrum_scale
    outdir.mkdir(parents=True, exist_ok=True)

    header = [
        "# Per-node complex amplitudes at target harmonics (merged chunk)",
        "# id weight I phi order ReJx ImJx ReJy ImJy |J|^2_scaled",
    ]
    lines = []
    for key in sorted(merged.keys()):
        r = merged[key]
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

    # Full continuum ICS/CS with real h(ω), ω (not just five harmonics / omega=0).
    ics_lines = [
        "# merged ICS/CS continuum from full manifest + chunked propagation",
        f"# n_nodes = {len(manifest)}",
        f"# n_omega = {len(merged_spec)}",
        f"# manifest_sha256 = {manifest_sha}",
        f"# spectrum_scale = {scale:.17e}",
        "# harmonic_order  omega(a.u.)  ICS  CS  classical_trajectory_variance",
    ]
    per_h: dict[int, dict[str, float]] = {}
    for row in merged_spec:
        ics = row["ics"]
        cs = scale * (abs(row["jx"]) ** 2 + abs(row["jy"]) ** 2)
        var = ics - cs
        floor = NEG_VAR_RTOL * max(abs(ics), 1.0e-300)
        if var < -floor:
            raise ValueError(
                f"iw={row['iw']}: ICS-CS = {var:.6e} largely negative "
                f"(ICS={ics:.6e}, CS={cs:.6e})"
            )
        if var < 0.0:
            var = 0.0
        ics_lines.append(
            f"{row['h_order']:25.17e}  {row['omega']:25.17e}  "
            f"{ics:25.17e}  {cs:25.17e}  {var:25.17e}"
        )

    # Cross-check: reconstruct target-order ICS/CS from merged nodes, compare
    # to continuum at the Fortran harmonic_fft_index bin.
    input_nml = chunk_dirs[0] / "input.nml"
    if not input_nml.is_file():
        raise FileNotFoundError(f"missing {input_nml} for FFT-bin cross-check")
    grid = parse_input_nml(input_nml)
    nt = int(grid["nt"])
    dt = float(grid["dt_au"])
    omega0 = float(grid["omega0"])

    for h in harmonics:
        group = [merged[(nid, h)] for nid in manifest]
        ics_nodes = sum(r["weight"] * r["power"] for r in group)
        jx_nodes = sum(r["weight"] * r["jx"] for r in group)
        jy_nodes = sum(r["weight"] * r["jy"] for r in group)
        cs_nodes = scale * (abs(jx_nodes) ** 2 + abs(jy_nodes) ** 2)

        iw = harmonic_fft_index(nt, dt, omega0, h)
        if iw < 1 or iw > len(merged_spec):
            raise ValueError(f"H{h}: FFT index {iw} out of continuum range")
        row = merged_spec[iw - 1]
        if row["iw"] != iw:
            raise ValueError(f"H{h}: continuum iw mismatch at index {iw}")
        ics_c = row["ics"]
        cs_c = scale * (abs(row["jx"]) ** 2 + abs(row["jy"]) ** 2)
        for label, a, b in (("ICS", ics_nodes, ics_c), ("CS", cs_nodes, cs_c)):
            rel = abs(a - b) / max(abs(a), abs(b), 1.0e-300)
            if rel > NODE_SPECTRUM_RTOL:
                raise ValueError(
                    f"H{h}: node vs continuum {label} mismatch rel={rel:.3e} "
                    f"(nodes={a:.16e}, continuum_iw{iw}={b:.16e})"
                )
        per_h[h] = {
            "ICS": ics_nodes,
            "CS": cs_nodes,
            "ICS_continuum": ics_c,
            "CS_continuum": cs_c,
            "variance": max(ics_nodes - cs_nodes, 0.0),
            "from": "nodes_crosschecked_continuum",
            "iw": iw,
            "h_bin": row["h_order"],
            "omega": row["omega"],
        }

    (outdir / "HHG_ics_cs.dat").write_text("\n".join(ics_lines) + "\n", encoding="utf-8")

    spec_lines = [
        "# merged weighted spectrum (sum of chunk partial spectra)",
        f"# spectrum_scale = {scale:.17e}",
        "# iw  harmonic_order  omega(a.u.)  weighted_ics  ReSumJx ImSumJx ReSumJy ImSumJy",
    ]
    for row in merged_spec:
        spec_lines.append(
            f"{row['iw']:8d}"
            f"{row['h_order']:25.17e}"
            f"{row['omega']:25.17e}"
            f"{row['ics']:25.17e}"
            f"{row['jx'].real:25.17e}{row['jx'].imag:25.17e}"
            f"{row['jy'].real:25.17e}{row['jy'].imag:25.17e}"
        )
    (outdir / "merged_weighted_spectrum.dat").write_text("\n".join(spec_lines) + "\n", encoding="utf-8")

    bsv_lines = [
        f"# n_nodes = {len(manifest)}",
        "# NOTE: HHG_bsv.dat is the weighted ICS average (merged chunks).",
        "# harmonic_order  omega(a.u.)  HHG_ics_avg  HHG_ics_stderr",
    ]
    for row in merged_spec:
        bsv_lines.append(
            f"{row['h_order']:10.4f} {row['omega']:16.8e} {row['ics']:16.8e} {0.0:16.8e}"
        )
    (outdir / "HHG_bsv.dat").write_text("\n".join(bsv_lines) + "\n", encoding="utf-8")

    # Required artifacts for compare: manifest snapshot + merged provenance.
    assert meta_ref is not None
    man_out = outdir / "nodes_manifest.input.dat"
    man_out.write_bytes(manifest_path.read_bytes())
    if sha256(man_out) != manifest_sha:
        raise ValueError("failed to snapshot manifest into outdir")

    meta_lines = [
        "campaign=merged_chunked_ensemble",
        f"manifest_sha256={manifest_sha}",
        f"n_chunks={len(chunk_dirs)}",
        f"n_nodes={len(manifest)}",
        f"weight_sum={wsum:.17e}",
        f"spectrum_scale={scale:.17e}",
        f"n_omega={len(merged_spec)}",
        "status=PASS",
    ]
    for key in PROVENANCE_KEYS:
        meta_lines.append(f"{key}={meta_ref[key]}")
    for key in RUNTIME_KEYS:
        meta_lines.append(f"{key}={meta_ref[key]}")
    # Prefer first chunk source/binary as informational; compare vs 27992
    # must NOT require these to match (new code vs old job).
    (outdir / "run_metadata.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    (outdir / "merge_provenance.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    return {
        "status": "PASS",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "weight_sum": wsum,
        "weight_sum_tol": WEIGHT_SUM_TOL,
        "n_chunks": len(chunk_dirs),
        "n_nodes": len(manifest),
        "harmonics": harmonics,
        "spectrum_scale": scale,
        "spectrum_source": "chunk_spectra",
        "n_omega": len(merged_spec),
        "per_harmonic": {f"H{h}": per_h[h] for h in harmonics},
        "chunk_dirs": [str(p.resolve()) for p in chunk_dirs],
        "provenance_keys_checked": list(PROVENANCE_KEYS),
        "runtime_keys_checked": list(RUNTIME_KEYS),
        "physics_keys": list(PHYSICS_KEYS),
        "build_keys": list(BUILD_KEYS),
        "chunk_output_sha256": chunk_output_sha256,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--chunk-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--harmonics", default="2,5,7,9,10")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    harmonics = [int(x.strip()) for x in args.harmonics.split(",") if x.strip()]
    report = merge_ensemble(args.manifest, args.chunk_dirs, args.outdir, harmonics)
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    (args.outdir / "merge_report.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
