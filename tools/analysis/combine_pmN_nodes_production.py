#!/usr/bin/env python3
"""Production entry point for strict shared-node +N/-N pairing.

Hard requirements relative to ``combine_pmN_nodes_strict.py``:

* ``--plus-nodes`` and ``--minus-nodes`` are mandatory;
* the two manifests must be byte-identical (SHA-256);
* every ``HHG_nodes_modes.dat`` row must match manifest ``(id,w,I,phi)``,
  with phase compared on the circle (0 ≡ 2π);
* ``--plus-meta`` / ``--minus-meta`` (or sibling ``run_metadata.txt``) must
  match on nk, dt, T2, wavelength, r, θ, harmonics, source/binary/node SHA;
  TB hash and domain labels may differ;
* ICS−CS negativity tolerance is tightened for weak channels.

These are layer-A classical mixture diagnostics only — not quantum g^(2),
squeezing, or entanglement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import combine_pmN_nodes_strict as strict


def _nonnegative_variance(ics: float, cs: float, label: str) -> float:
    variance = float(ics - cs)
    tolerance = 2.0e-12 * max(abs(ics), abs(cs), 1.0e-300)
    if variance < -tolerance:
        raise ValueError(
            f"{label}: ICS-CS={variance:.17e} is negative beyond roundoff; "
            "check weights and amplitudes"
        )
    return max(variance, 0.0)


strict._nonnegative_variance = _nonnegative_variance


def _resolve_metadata(explicit: Path | None, modes_path: Path, label: str) -> Path:
    if explicit is not None:
        return explicit
    sibling = modes_path.parent / "run_metadata.txt"
    if sibling.is_file():
        return sibling
    raise SystemExit(
        f"ERROR: {label} run_metadata not provided and not found next to "
        f"{modes_path} (expected {sibling})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Production ±N pairing: requires shared node manifests, "
            "modes↔manifest cross-check, and ±N run_metadata consistency."
        )
    )
    parser.add_argument("--plus", type=Path, required=True)
    parser.add_argument("--minus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--plus-nodes", type=Path, required=True)
    parser.add_argument("--minus-nodes", type=Path, required=True)
    parser.add_argument(
        "--plus-meta", type=Path, default=None,
        help="plus-domain run_metadata.txt (default: sibling of --plus)",
    )
    parser.add_argument(
        "--minus-meta", type=Path, default=None,
        help="minus-domain run_metadata.txt (default: sibling of --minus)",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--weight-sum-tol", type=float, default=5.0e-8)
    parser.add_argument(
        "--phase-atol",
        type=float,
        default=1.0e-6,
        help="absolute radian tolerance for circular phase compare (0 vs 2π)",
    )
    args = parser.parse_args()

    if min(args.rtol, args.atol, args.weight_sum_tol, args.phase_atol) < 0.0:
        raise SystemExit("tolerances must be non-negative")

    plus_meta = _resolve_metadata(args.plus_meta, args.plus, "plus")
    minus_meta = _resolve_metadata(args.minus_meta, args.minus, "minus")

    try:
        matched_meta = strict.compare_run_metadata_pair(plus_meta, minus_meta)
        print(
            "run_metadata_match "
            + " ".join(f"{k}={v}" for k, v in matched_meta.items())
        )

        nodes_sha = strict.require_identical_node_manifests(
            args.plus_nodes, args.minus_nodes
        )
        print(f"nodes_manifest_sha256={nodes_sha}")
        if matched_meta.get("nodes_sha256") not in (None, nodes_sha):
            raise ValueError(
                "run_metadata nodes_sha256 disagrees with manifest SHA: "
                f"meta={matched_meta['nodes_sha256']} file={nodes_sha}"
            )
        manifest = strict.load_node_manifest(args.plus_nodes)

        plus = strict.load_modes(
            args.plus, rtol=args.rtol, atol=args.atol,
            weight_sum_tol=args.weight_sum_tol,
        )
        minus = strict.load_modes(
            args.minus, rtol=args.rtol, atol=args.atol,
            weight_sum_tol=args.weight_sum_tol,
        )
        strict.crosscheck_modes_against_manifest(
            plus, args.plus, manifest,
            rtol=args.rtol, atol=args.atol, phase_atol=args.phase_atol,
        )
        strict.crosscheck_modes_against_manifest(
            minus, args.minus, manifest,
            rtol=args.rtol, atol=args.atol, phase_atol=args.phase_atol,
        )

        minus_aligned = strict.align_and_validate_pair(
            plus, minus, args.plus, args.minus,
            rtol=args.rtol, atol=args.atol,
        )

        jx_plus, jy_plus = strict._complex_components(plus)
        jx_minus, jy_minus = strict._complex_components(minus_aligned)
        jx_even = 0.5 * (jx_plus + jx_minus)
        jy_even = 0.5 * (jy_plus + jy_minus)
        jx_odd = 0.5 * (jx_plus - jx_minus)
        jy_odd = 0.5 * (jy_plus - jy_minus)

        strict.write_node_output(args.out, plus, jx_even, jy_even, jx_odd, jy_odd)
        print(f"wrote per-node Je/Jo: {args.out}")
        strict.write_summary(
            args.summary, plus,
            jx_plus, jy_plus, jx_minus, jy_minus,
            jx_even, jy_even, jx_odd, jy_odd,
        )
        with args.summary.open("a", encoding="utf-8") as stream:
            stream.write(f"# nodes_manifest_sha256 = {nodes_sha}\n")
            stream.write(
                "# metric note: |Jo_bar|^2 == trajectory_odd_CS; "
                "domain VAR = ICS-CS on one Neel domain (classical mixture only)\n"
            )
            for key, value in matched_meta.items():
                stream.write(f"# meta_{key} = {value}\n")
        print(f"wrote weighted summary: {args.summary}")
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
