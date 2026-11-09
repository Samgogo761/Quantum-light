#!/usr/bin/env python3
"""Frozen-spec gate: Q_THEORY_FROZEN_SPEC.md == pt_afm_toy.FROZEN_SPEC == result metadata."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pt_afm_toy import (  # noqa: E402
    FORBIDDEN_BACKEND_KNOBS,
    FROZEN_SPEC,
    ToyParams,
    assert_frozen_spec,
    model_version,
    run_toy,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_spec_block(text: str) -> dict[str, object]:
    m = re.search(r"```text\n(.*?)```", text, flags=re.S)
    require(m is not None, "no ```text block in Q_THEORY_FROZEN_SPEC.md")
    out: dict[str, object] = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value in ("True", "False"):
            out[key] = value == "True"
        else:
            out[key] = value
    return out


def test_md_matches_code() -> None:
    text = Path(__file__).with_name("Q_THEORY_FROZEN_SPEC.md").read_text(encoding="utf-8")
    md = parse_spec_block(text)
    require(set(md) == set(FROZEN_SPEC), f"key mismatch: {set(md) ^ set(FROZEN_SPEC)}")
    for key, value in FROZEN_SPEC.items():
        require(md[key] == value, f"{key}: md={md[key]!r} code={value!r}")


def test_result_metadata_obeys_spec() -> None:
    p = ToyParams(nk1d=4, ncyc=1.0, nt_per_cycle=8, e0=0.05, omega=0.25)
    r = run_toy(p)
    assert_frozen_spec(r)
    require(r["model_version"] == model_version(), "model_version missing")
    require(r["model_version"].startswith("sha256:"), r["model_version"])


def test_violation_is_detected() -> None:
    bad = dict(FROZEN_SPEC)
    bad["dephasing"] = "T2_phenomenological"
    try:
        assert_frozen_spec(bad)
    except ValueError as exc:
        require("dephasing" in str(exc), str(exc))
    else:
        raise RuntimeError("spec violation not detected")


def test_forbidden_knobs_listed() -> None:
    for knob in ("T2", "gauge", "integrator", "lg_cov", "dtype"):
        require(knob in FORBIDDEN_BACKEND_KNOBS, f"{knob} missing from FORBIDDEN_BACKEND_KNOBS")


def main() -> None:
    test_md_matches_code()
    test_result_metadata_obeys_spec()
    test_violation_is_detected()
    test_forbidden_knobs_listed()
    print("test_q_frozen_spec: PASS")


if __name__ == "__main__":
    main()
