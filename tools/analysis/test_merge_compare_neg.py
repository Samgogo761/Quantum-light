#!/usr/bin/env python3
"""Negative tests for merge/compare gates (offline, no Fortran)."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

from merge_chunked_ensemble import WEIGHT_SUM_TOL, merge_ensemble, sha256


def _write_minimal_chunk(
    cdir: Path,
    *,
    meta: str,
    prop_ids: str,
    modes: str,
    spectrum: str,
    success: bool = True,
    status: str = "PASS",
) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "run_metadata.txt").write_text(meta, encoding="utf-8")
    (cdir / "chunk_info.txt").write_text(
        f"n_manifest_nodes = 2\nn_propagate_nodes = 1\npropagate_ids = {prop_ids}\n",
        encoding="utf-8",
    )
    (cdir / "HHG_nodes_modes.dat").write_text(modes, encoding="utf-8")
    (cdir / "chunk_weighted_spectrum.dat").write_text(spectrum, encoding="utf-8")
    (cdir / "input.nml").write_text(
        "&laser\n  wvl_nm = 3200.0\n  dt = 0.35\n  ncyc = 4.0\n/\n",
        encoding="utf-8",
    )
    if success:
        (cdir / "SUCCESS").write_text("", encoding="utf-8")
    (cdir / "run_status.txt").write_text(f"status={status}\n", encoding="utf-8")
    hashed = [
        "run_metadata.txt",
        "chunk_info.txt",
        "HHG_nodes_modes.dat",
        "chunk_weighted_spectrum.dat",
        "input.nml",
        "run_status.txt",
    ]
    (cdir / "output_sha256.txt").write_text(
        "\n".join(f"{sha256(cdir / name)}  {name}" for name in hashed if (cdir / name).is_file()) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    fails: list[str] = []
    # Weight-sum tolerance must accept 27992-like 0.9999999992 and reject worse.
    if abs(0.9999999992 - 1.0) > WEIGHT_SUM_TOL:
        fails.append("WEIGHT_SUM_TOL too tight for 27992 manifests")
    if abs(0.999999 - 1.0) <= WEIGHT_SUM_TOL:
        fails.append("WEIGHT_SUM_TOL too loose (0.999999 should fail)")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        man = root / "manifest.dat"
        # two equal-weight nodes; sum exactly 1
        man.write_text(
            "1 0.5 0.0 0.0 1.0 0.0\n2 0.5 1.0 0.0 1.0 0.0\n",
            encoding="utf-8",
        )
        msha = sha256(man)
        base_meta = "\n".join(
            [
                f"source_sha256=src",
                f"binary_sha256=bin",
                f"tb_sha256=tb",
                f"nodes_sha256={msha}",
                f"template_sha256=tmpl",
                "nk=20",
                "model=full112",
                "wvl_nm=3200.0",
                "dt=0.35",
                "T2_cycles=0.5",
                "squeeze_r=2.5",
                "squeeze_theta_deg=180",
                "I_bar=1.0e11",
                "harmonics=2",
                "omp_num_threads=36",
                "mkl_num_threads=1",
            ]
        )
        # NaN in modes must fail
        c0 = root / "c0"
        modes_nan = (
            "# id weight I phi order ReJx ImJx ReJy ImJy power\n"
            "1 5.00000000000000000e-01 1.0 0.0 2 NaN 0.0 0.0 0.0 1.0\n"
        )
        spec = (
            "# spectrum_scale = 1.0\n"
            "# iw h omega ics ...\n"
            "1 2.0 2.0 0.5 0.0 0.0 0.0 0.0\n"
        )
        _write_minimal_chunk(c0, meta=base_meta, prop_ids="1", modes=modes_nan, spectrum=spec)
        try:
            merge_ensemble(man, [c0], root / "out_nan", [2])
            fails.append("NaN modes were accepted")
        except Exception:
            pass

        # Wrong provenance must fail
        c1 = root / "c1"
        bad_meta = base_meta.replace("tb_sha256=tb", "tb_sha256=OTHER")
        modes_ok = (
            "# id weight I phi order ReJx ImJx ReJy ImJy power\n"
            "1 5.00000000000000000e-01 1.0 0.0 2 1.0 0.0 0.0 0.0 1.0\n"
        )
        _write_minimal_chunk(c1, meta=bad_meta, prop_ids="1", modes=modes_ok, spectrum=spec)
        c2 = root / "c2"
        modes_ok2 = (
            "# id weight I phi order ReJx ImJx ReJy ImJy power\n"
            "2 5.00000000000000000e-01 1.0 0.0 2 1.0 0.0 0.0 0.0 1.0\n"
        )
        _write_minimal_chunk(c2, meta=base_meta, prop_ids="2", modes=modes_ok2, spectrum=spec)
        try:
            merge_ensemble(man, [c1, c2], root / "out_prov", [2])
            fails.append("provenance mismatch was accepted")
        except Exception:
            pass

        # compare must require merged manifest/metadata (no silent fallback)
        from compare_chunked_vs_full import main as compare_main
        import sys

        empty = root / "empty_merged"
        empty.mkdir()
        (empty / "HHG_nodes_modes.dat").write_text(modes_ok, encoding="utf-8")
        (empty / "HHG_ics_cs.dat").write_text("2.0 2.0 1.0 1.0 0.0\n", encoding="utf-8")
        full = root / "full"
        full.mkdir()
        (full / "nodes_manifest.input.dat").write_bytes(man.read_bytes())
        (full / "run_metadata.txt").write_text(base_meta + "\n", encoding="utf-8")
        (full / "HHG_nodes_modes.dat").write_text(modes_ok + modes_ok2.replace("\n2 ", "\n2 "), encoding="utf-8")
        # simplify: just check SystemExit for missing merged manifest
        argv = sys.argv
        try:
            sys.argv = [
                "compare_chunked_vs_full.py",
                "--merged",
                str(empty),
                "--full-run-dir",
                str(full),
            ]
            rc = compare_main()
            if rc == 0:
                fails.append("compare accepted merged without manifest/metadata")
        except SystemExit as exc:
            if exc.code in (0, None):
                fails.append("compare SystemExit 0 without merged provenance")
        finally:
            sys.argv = argv

    if fails:
        print("FAIL:")
        for f in fails:
            print(" ", f)
        return 1
    print(
        f"RESULT[merge_compare_neg]: PASS "
        f"(WEIGHT_SUM_TOL={WEIGHT_SUM_TOL:g}; NaN/provenance/compare gates)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
