#!/usr/bin/env python3
"""Dump frozen H2/window/resource blocks from Q1a JSON files."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light")
OUT = ROOT / "docs" / "04_BSV量子光" / "qmath_r1_h2_extract.json"


def load(p):
    with Path(p).open("r", encoding="utf-8") as f:
        return json.load(f)


def keep(obj, keys):
    if not isinstance(obj, dict):
        return obj
    return {k: obj.get(k) for k in keys if k in obj}


def main():
    thermal = load(ROOT / "results/q_line/qinput_qin_a2_141_gh5_thermal_20261120/gh5_thermal.json")
    qed_pi = load(ROOT / "results/q_line/qinput_qin_a2_141_qed_theta_pi_20261120/qed_theta_pi.json")
    mech = load(ROOT / "results/q_line/qinput_qin_a2_141_qed_theta_pi_mech_20261120/mech.json")
    q1a = load(ROOT / "results/q_line/qinput_qin_a2_141_q1a_20261120/q1a.json")
    gh5 = load(ROOT / "results/q_line/qinput_qin_a2_141_gh5_20261120/gh5.json")
    gh5_pi = load(ROOT / "results/q_line/qinput_qin_a2_141_gh5_theta_pi_20261120/gh5_theta_pi.json")

    out = {
        "thermal": {
            "h2_main": thermal.get("h2_main"),
            "windows": thermal.get("windows"),
            "resources": thermal.get("resources"),
            "verdict": thermal.get("verdict"),
            "answer": thermal.get("answer"),
            "source_hashes": thermal.get("source_hashes"),
            "protocol": thermal.get("protocol"),
            "n_nodes": thermal.get("n_nodes"),
            "theta0_archive": thermal.get("theta0_archive"),
            "thetapi_archive": thermal.get("thetapi_archive"),
        },
        "qed_theta_pi": {
            "h2_main": qed_pi.get("h2_main"),
            "windows": qed_pi.get("windows"),
            "coverage": qed_pi.get("coverage"),
            "preflight": qed_pi.get("preflight"),
            "qed_vs_gh5": qed_pi.get("qed_vs_gh5"),
            "verdict": qed_pi.get("verdict"),
            "answer": qed_pi.get("answer"),
            "source_hashes": qed_pi.get("source_hashes"),
            "protocol": qed_pi.get("protocol"),
            "theta0_qed_archive": qed_pi.get("theta0_qed_archive"),
        },
        "mech": {
            "answers": mech.get("answers"),
            "delta_complex": mech.get("delta_complex"),
            "window_table": mech.get("window_table"),
            "fit_table": mech.get("fit_table"),
            "gh5_vs_qed": mech.get("gh5_vs_qed"),
            "theory": mech.get("theory"),
            "scales": mech.get("scales"),
            "coverage": mech.get("coverage"),
        },
        "q1a": {
            "h2_from_windows": None,
            "windows": q1a.get("windows"),
            "verdict": q1a.get("verdict"),
            "answer": q1a.get("answer"),
            "source_hashes": q1a.get("source_hashes"),
            "protocol": q1a.get("protocol"),
            "channels_H1_H6": q1a.get("channels_H1_H6"),
            "time_series_summary": q1a.get("time_series_summary"),
        },
        "gh5": {
            "h2_main": gh5.get("h2_main"),
            "windows": gh5.get("windows"),
            "compare_main": gh5.get("compare_main"),
            "compare_front": gh5.get("compare_front"),
            "compare_back": gh5.get("compare_back"),
            "qed_readonly_delta_complex": gh5.get("qed_readonly_delta_complex"),
            "mtef_delta_complex": gh5.get("mtef_delta_complex"),
            "verdict": gh5.get("verdict"),
            "answer": gh5.get("answer"),
            "source_hashes": gh5.get("source_hashes"),
            "protocol": gh5.get("protocol"),
            "qed_archive": gh5.get("qed_archive"),
        },
        "gh5_theta_pi": {
            "h2_main": gh5_pi.get("h2_main"),
            "windows": gh5_pi.get("windows"),
            "resources": gh5_pi.get("resources"),
            "verdict": gh5_pi.get("verdict"),
            "answer": gh5_pi.get("answer"),
            "source_hashes": gh5_pi.get("source_hashes"),
            "protocol": gh5_pi.get("protocol"),
            "theta0_archive": gh5_pi.get("theta0_archive"),
        },
    }
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
