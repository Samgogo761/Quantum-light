#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light")
OUT = ROOT / "docs" / "04_BSV量子光" / "qmath_r1_compact.txt"


def load(rel):
    with (ROOT / rel).open("r", encoding="utf-8") as f:
        return json.load(f)


def brief(obj, depth=0, max_depth=4, max_list=3):
    pad = "  " * depth
    if isinstance(obj, dict):
        if depth >= max_depth:
            return pad + "{... keys=" + str(list(obj.keys())[:12]) + "}\n"
        s = ""
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                s += pad + f"{k}:\n" + brief(v, depth + 1, max_depth, max_list)
            else:
                s += pad + f"{k}: {v!r}\n"
        return s
    if isinstance(obj, list):
        if depth >= max_depth:
            return pad + f"[list len={len(obj)}]\n"
        s = pad + f"[list len={len(obj)}]\n"
        for i, v in enumerate(obj[:max_list]):
            s += pad + f"[{i}]\n" + brief(v, depth + 1, max_depth, max_list)
        if len(obj) > max_list:
            s += pad + "...\n"
        return s
    return pad + repr(obj) + "\n"


def main():
    thermal = load("results/q_line/qinput_qin_a2_141_gh5_thermal_20261120/gh5_thermal.json")
    qed_pi = load("results/q_line/qinput_qin_a2_141_qed_theta_pi_20261120/qed_theta_pi.json")
    mech = load("results/q_line/qinput_qin_a2_141_qed_theta_pi_mech_20261120/mech.json")
    q1a = load("results/q_line/qinput_qin_a2_141_q1a_20261120/q1a.json")
    gh5 = load("results/q_line/qinput_qin_a2_141_gh5_20261120/gh5.json")
    gh5_pi = load("results/q_line/qinput_qin_a2_141_gh5_theta_pi_20261120/gh5_theta_pi.json")

    parts = []
    parts.append("===== THERMAL h2_main/resources/verdict =====\n")
    parts.append(brief({k: thermal.get(k) for k in ["h2_main", "resources", "verdict", "answer", "protocol", "n_nodes", "source_hashes"]}))
    parts.append("\n===== THERMAL windows keys =====\n")
    w = thermal.get("windows")
    parts.append(brief(w, max_depth=3))
    parts.append("\n===== QED_PI h2_main/coverage/qed_vs_gh5/preflight =====\n")
    parts.append(brief({k: qed_pi.get(k) for k in ["h2_main", "coverage", "qed_vs_gh5", "preflight", "verdict", "answer", "protocol", "source_hashes", "theta0_qed_archive"]}))
    parts.append("\n===== QED_PI windows =====\n")
    parts.append(brief(qed_pi.get("windows"), max_depth=3))
    parts.append("\n===== MECH =====\n")
    parts.append(brief({k: mech.get(k) for k in ["answers", "delta_complex", "window_table", "fit_table", "gh5_vs_qed", "theory", "scales", "coverage"]}, max_depth=4))
    parts.append("\n===== Q1A verdict/protocol/windows =====\n")
    parts.append(brief({k: q1a.get(k) for k in ["verdict", "answer", "protocol", "source_hashes", "windows", "channels_H1_H6", "time_series_summary"]}, max_depth=3))
    parts.append("\n===== GH5 h2_main/compare_main =====\n")
    parts.append(brief({k: gh5.get(k) for k in ["h2_main", "compare_main", "qed_readonly_delta_complex", "mtef_delta_complex", "verdict", "answer", "protocol", "source_hashes", "qed_archive"]}, max_depth=4))
    parts.append("\n===== GH5 windows =====\n")
    parts.append(brief(gh5.get("windows"), max_depth=3))
    parts.append("\n===== GH5_PI h2_main/resources =====\n")
    parts.append(brief({k: gh5_pi.get(k) for k in ["h2_main", "resources", "verdict", "answer", "protocol", "source_hashes", "theta0_archive"]}, max_depth=4))
    parts.append("\n===== GH5_PI windows =====\n")
    parts.append(brief(gh5_pi.get("windows"), max_depth=3))

    text = "".join(parts)
    OUT.write_text(text, encoding="utf-8")
    print("wrote", OUT, "chars", len(text))


if __name__ == "__main__":
    main()
