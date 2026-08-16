#!/usr/bin/env python3
"""Positive offline merge+compare: must succeed for the right reasons.

Synthesizes two chunks whose spectra sum to a known continuum, merges them,
then compares against a "full" run whose continuum uses historical F10.4/ES16.8
rounding and a different template_sha256. PASS means:
  - merge completed
  - compare returns 0
  - physics keys match
  - build keys (template/source/binary) may differ without failing
  - h/omega roundoff within historical atol
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

from compare_chunked_vs_full import H_ORDER_ATOL, OMEGA_ATOL, main as compare_main
from hhg_fft_utils import (
    TWOPI,
    harmonic_fft_index,
    harmonic_order_at_index,
    omega0_from_wvl_nm,
    parse_input_nml,
)
from merge_chunked_ensemble import merge_ensemble, sha256


def _write_chunk(
    cdir: Path,
    *,
    meta: str,
    prop_id: int,
    n_manifest: int,
    modes_line: str,
    spectrum_lines: list[str],
    nml: str,
) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "run_metadata.txt").write_text(meta + "\n", encoding="utf-8")
    (cdir / "chunk_info.txt").write_text(
        f"n_manifest_nodes = {n_manifest}\n"
        f"n_propagate_nodes = 1\n"
        f"propagate_ids = {prop_id}\n",
        encoding="utf-8",
    )
    (cdir / "HHG_nodes_modes.dat").write_text(
        "# id weight I phi order ReJx ImJx ReJy ImJy power\n" + modes_line + "\n",
        encoding="utf-8",
    )
    (cdir / "chunk_weighted_spectrum.dat").write_text(
        "\n".join(spectrum_lines) + "\n", encoding="utf-8"
    )
    (cdir / "input.nml").write_text(nml, encoding="utf-8")
    (cdir / "SUCCESS").write_text("", encoding="utf-8")
    (cdir / "run_status.txt").write_text("status=PASS\n", encoding="utf-8")
    hashed = [
        "run_metadata.txt",
        "chunk_info.txt",
        "HHG_nodes_modes.dat",
        "chunk_weighted_spectrum.dat",
        "input.nml",
        "run_status.txt",
    ]
    (cdir / "output_sha256.txt").write_text(
        "\n".join(f"{sha256(cdir / name)}  {name}" for name in hashed) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    wvl = 3200.0
    dt = 0.35
    ncyc = 4.0
    omega0 = omega0_from_wvl_nm(wvl)
    t_total = ncyc * (TWOPI / omega0)
    nt = int(math.ceil(t_total / dt)) + 1
    n_omega = nt // 2 + 1
    domega = TWOPI / (float(nt) * dt)
    order = 2
    iw_h = harmonic_fft_index(nt, dt, omega0, order)

    nml = (
        "&laser\n"
        f"  wvl_nm = {wvl}\n"
        f"  dt = {dt}\n"
        f"  ncyc = {ncyc}\n"
        "/\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        man = root / "manifest.dat"
        man.write_text(
            "1 0.5 1.0 0.0 1.0 0.0\n"
            "2 0.5 1.0 0.0 1.0 0.0\n",
            encoding="utf-8",
        )
        msha = sha256(man)
        meta_chunk = "\n".join(
            [
                "source_sha256=src_new",
                "binary_sha256=bin_new",
                "tb_sha256=tb_phys",
                f"nodes_sha256={msha}",
                "template_sha256=tmpl_new",
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

        # Each chunk carries half the weight at H2; continuum bins match FFT layout.
        def spectrum_for(weight: float, jx: float) -> list[str]:
            lines = [
                "# spectrum_scale = 1.0",
                "# iw h omega ics ReJx ImJx ReJy ImJy",
            ]
            for iw in range(1, n_omega + 1):
                omega = float(iw - 1) * domega
                h = omega / omega0
                if iw == iw_h:
                    ics = weight * 1.0  # weight * power
                    lines.append(
                        f"{iw} {h:.17e} {omega:.17e} {ics:.17e} {jx:.17e} 0.0 0.0 0.0"
                    )
                else:
                    lines.append(f"{iw} {h:.17e} {omega:.17e} 0.0 0.0 0.0 0.0 0.0")
            return lines

        c1 = root / "c1"
        c2 = root / "c2"
        _write_chunk(
            c1,
            meta=meta_chunk,
            prop_id=1,
            n_manifest=2,
            modes_line=(
                "1 5.00000000000000000e-01 1.0 0.0 2 "
                "1.0 0.0 0.0 0.0 1.0"
            ),
            spectrum_lines=spectrum_for(0.5, 0.5),
            nml=nml,
        )
        _write_chunk(
            c2,
            meta=meta_chunk,
            prop_id=2,
            n_manifest=2,
            modes_line=(
                "2 5.00000000000000000e-01 1.0 0.0 2 "
                "1.0 0.0 0.0 0.0 1.0"
            ),
            spectrum_lines=spectrum_for(0.5, 0.5),
            nml=nml,
        )

        merged = root / "merged"
        try:
            merge_ensemble(man, [c1, c2], merged, [order])
        except Exception as exc:
            print(f"FAIL: merge raised {exc!r}", file=sys.stderr)
            return 1

        # Historical full run: same physics, F10.4/ES16.8 continuum, old template SHA.
        full = root / "full"
        full.mkdir()
        (full / "nodes_manifest.input.dat").write_bytes(man.read_bytes())
        meta_full = meta_chunk.replace("template_sha256=tmpl_new", "template_sha256=tmpl_27992")
        meta_full = meta_full.replace("source_sha256=src_new", "source_sha256=src_27992")
        meta_full = meta_full.replace("binary_sha256=bin_new", "binary_sha256=bin_27992")
        (full / "run_metadata.txt").write_text(meta_full + "\n", encoding="utf-8")
        (full / "HHG_nodes_modes.dat").write_text(
            (merged / "HHG_nodes_modes.dat").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        # Rewrite continuum with historical Fortran formats (round h/omega).
        hist_lines = [
            "# Layer-A spectra: ICS / CS / classical_trajectory_variance",
            "# harmonic_order  omega(a.u.)  ICS  CS  classical_trajectory_variance",
        ]
        for raw in (merged / "HHG_ics_cs.dat").read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split()
            h, omega, ics, cs, var = (float(x) for x in f[:5])
            # Mimic write(u,'(F10.4, 4ES16.8)')
            hist_lines.append(
                f"{h:10.4f}{omega:16.8e}{ics:16.8e}{cs:16.8e}{var:16.8e}"
            )
        (full / "HHG_ics_cs.dat").write_text("\n".join(hist_lines) + "\n", encoding="utf-8")

        # Sanity: roundoff should be within the atol budget we claim.
        grid = parse_input_nml(c1 / "input.nml")
        assert int(grid["nt"]) == nt
        h_hi = harmonic_order_at_index(nt, dt, omega0, iw_h)
        h_lo = float(f"{h_hi:10.4f}")
        if abs(h_hi - h_lo) > H_ORDER_ATOL:
            print("FAIL: F10.4 roundoff exceeds H_ORDER_ATOL budget", file=sys.stderr)
            return 1
        omega_hi = float(iw_h - 1) * domega
        omega_lo = float(f"{omega_hi:16.8e}")
        if abs(omega_hi - omega_lo) > OMEGA_ATOL:
            print("FAIL: ES16.8 roundoff exceeds OMEGA_ATOL budget", file=sys.stderr)
            return 1

        argv = sys.argv
        from io import StringIO
        from contextlib import redirect_stdout

        buf = StringIO()
        try:
            sys.argv = [
                "compare_chunked_vs_full.py",
                "--merged",
                str(merged),
                "--full-run-dir",
                str(full),
            ]
            # Keep make test output focused; compare_main prints a full JSON report.
            with redirect_stdout(buf):
                rc = compare_main()
        finally:
            sys.argv = argv

        if rc != 0:
            print("FAIL: compare returned non-zero on positive fixture", file=sys.stderr)
            print(buf.getvalue(), file=sys.stderr)
            return 1

    print(
        "RESULT[merge_compare_pos]: PASS "
        "(merge+compare; F10.4/ES16.8 atol; template SHA informational)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
