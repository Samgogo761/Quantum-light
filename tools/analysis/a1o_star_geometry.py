#!/usr/bin/env python3
"""A1-O k-star: centers plus ±δ crystal neighbors. No G wrapping.

Satellites keep unwrapped crystal coordinates so nnkpts G = (0,0,0).
"""
from __future__ import annotations

from typing import Any

CENTERS: list[dict[str, Any]] = [
    {
        "label": "k_ctrl",
        "crystal": (0.00, 0.00, 0.00),
        "role": "gamma_method_control_on_mesh",
    },
    {
        "label": "k_eta",
        "crystal": (0.10, 0.55, 0.00),
        "role": "eta_winner_off_mesh",
    },
    {
        "label": "k_H5",
        "crystal": (0.45, 0.55, 0.00),
        "role": "H5_candidate_off_mesh",
    },
    {
        "label": "k_H7",
        "crystal": (0.45, 0.05, 0.00),
        "role": "H7_candidate_off_mesh",
    },
]

DELTAS = (0.002, 0.005, 0.010)
DIR_NAMES = ("+x", "-x", "+y", "-y")
NNB_CENTER = len(DELTAS) * len(DIR_NAMES)


def dir_shift(delta: float, name: str) -> tuple[float, float, float]:
    if name == "+x":
        return (delta, 0.0, 0.0)
    if name == "-x":
        return (-delta, 0.0, 0.0)
    if name == "+y":
        return (0.0, delta, 0.0)
    if name == "-y":
        return (0.0, -delta, 0.0)
    raise ValueError(name)


def build_star() -> dict[str, Any]:
    kpoints: list[tuple[float, float, float]] = []
    centers_out: list[dict[str, Any]] = []
    for c in CENTERS:
        kpoints.append(tuple(float(x) for x in c["crystal"]))
        centers_out.append(
            {
                "label": c["label"],
                "role": c["role"],
                "crystal": [float(x) for x in c["crystal"]],
                "k_index_1based": len(kpoints),
                "neighbors": [],
            }
        )
    n_centers = len(centers_out)
    for ic, rec in enumerate(centers_out):
        cx, cy, cz = rec["crystal"]
        slot = 0
        for delta in DELTAS:
            for name in DIR_NAMES:
                dx, dy, dz = dir_shift(delta, name)
                sat = (cx + dx, cy + dy, cz + dz)
                kpoints.append(sat)
                rec["neighbors"].append(
                    {
                        "delta": float(delta),
                        "dir": name,
                        "shift": [dx, dy, dz],
                        "crystal": [sat[0], sat[1], sat[2]],
                        "k_index_1based": len(kpoints),
                        "g": [0, 0, 0],
                        "nn_slot": slot,
                    }
                )
                slot += 1
        if slot != NNB_CENTER:
            raise RuntimeError(f"nnb mismatch at {rec['label']}")

    nnkpts: list[tuple[int, int, int, int, int]] = []
    for rec in centers_out:
        ik = rec["k_index_1based"]
        for nb in rec["neighbors"]:
            nnkpts.append((ik, nb["k_index_1based"], 0, 0, 0))
    for ik in range(n_centers + 1, len(kpoints) + 1):
        for _ in range(NNB_CENTER):
            nnkpts.append((ik, ik, 0, 0, 0))

    labels = [c["label"] for c in centers_out]
    if len(set(labels)) != len(labels):
        raise RuntimeError("duplicate center labels")
    rounded = [(round(k[0], 12), round(k[1], 12), round(k[2], 12)) for k in kpoints]
    if len(set(rounded)) != len(rounded):
        raise RuntimeError("duplicate k-points in A1-O star")
    if len(nnkpts) != len(kpoints) * NNB_CENTER:
        raise RuntimeError("nnkpts length mismatch")

    return {
        "schema": "a1o_mmn_star_v1",
        "n_k": len(kpoints),
        "n_centers": n_centers,
        "nnb": NNB_CENTER,
        "deltas_crystal": list(DELTAS),
        "dir_names": list(DIR_NAMES),
        "kpoints": [[float(x) for x in k] for k in kpoints],
        "centers": centers_out,
        "nnkpts": [list(row) for row in nnkpts],
        "note": (
            "Centers eta/H5/H7 are the optical tests; Gamma is a method control. "
            "Satellite nnkpts are dummy self-overlaps so nnb is uniform."
        ),
    }
