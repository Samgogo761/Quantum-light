"""Read-only inventory: finite window vs asymptotic CCR (theory slice).

Does not assemble U,V. Does not call rows_with_kernels. Does not start K'.
Does not overwrite archived JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_window_vs_asymptotic_20261019"
OUTJSON = OUTDIR / "window_vs_asymptotic.json"
PROGRESS = OUTDIR / "progress.txt"

TIME = ROOT / "results/q_line/g1_gamma12_time_order_20261015"
K9DIR = ROOT / "results/q_line/g1_kernel_correction_20261017"
K11DIR = ROOT / "results/q_line/g1_kernel_correction_11cycles_20261017"
EXPECTED = {
    1024: "a21e59b48574a5e0beaf599c747f392ce791dd55f6119a17348fbcd674b40736",
    2048: "c04da52436198758155b3f2351911afcc513ed3cf46a3cef96f97baa747b9bde",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stat_path(path):
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path)}
    info = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
    }


def adjoint_inventory():
    out = {}
    for macro, digest in EXPECTED.items():
        path = TIME / f"adjoint_rows_macro{macro}.npz"
        rec = stat_path(path)
        rec["expected_sha256"] = digest
        if rec["exists"]:
            got = sha256(path)
            rec["sha256"] = got
            rec["sha256_matches_identity"] = got == digest
        else:
            rec["sha256_matches_identity"] = False
        out[str(macro)] = rec
    windows = {
        "9.0": TIME / "adjoint_rows_macro1024.npz",
        "9.5": TIME / "adjoint_rows_macro1024_cycles9p5.npz",
        "10.0": TIME / "adjoint_rows_macro1024_cycles10p0.npz",
        "11.0": TIME / "adjoint_rows_macro1024_cycles11p0.npz",
    }
    out["window_rows"] = {label: stat_path(path) for label, path in windows.items()}
    return out


def kernel_cost_from_disk():
    k9 = K9DIR / "kernel_correction_9cycles.json"
    rec = {
        "nine_cycle_elapsed_s_from_json": 423.9480345249176,
        "nine_cycle_json": stat_path(k9),
        "nine_cycle_rows_cache": stat_path(K9DIR / "rows_kernel_correction_9cycles.npz"),
        "eleven_cycle_progress": stat_path(K11DIR / "progress.txt"),
        "eleven_cycle_rows_cache": stat_path(K11DIR / "rows_kernel_correction_11cycles.npz"),
        "vertices_on_disk": False,
        "vertex_search": "no *vertex* under kernel_correction_20261017; vertices produced then released",
        "eleven_cycle_from_progress": {
            "kernels_ready_s": 48.5,
            "vertices_driven_done_s": 514.9,
            "vertices_undriven_done_s": 1870.5,
            "rows_cached_s": 2622.7,
            "assembled_json_s": 3102.7,
        },
        "source": "kernel_correction_9cycles.json elapsed_s; 11-cycle progress.txt. Not a new run.",
    }
    return rec


def theory_bout_T():
    return {
        "production_map": "U = U0 + raw_driven_u - raw_undriven_u; V = raw_driven_v - raw_undriven_v",
        "production_map_code": "q8_m0_g1_boundary_sensitivity_20261015.py production_map (not written back)",
        "U0": {
            "kind": "frequency_domain_closed_form_not_window_integral",
            "code": "q8_m0_p1_h2_finite_kcv_input_close_20261005.u0_from_bundle",
            "photon_column": "u0_j = phi(omega) * (pol @ ((sqrt(omega)/eta) * a_r))",
            "dark_column": "U0 = 0  (gram_grouping assemble dark block)",
            "static_minus_I_2048_00": 2.13e-12,
            "static_minus_I_source": "docs/04_BSV量子光/线Q_G1动态映射分解_20261015.md",
        },
        "DeltaU_T": "raw_driven_u(T) - raw_undriven_u(T); adjoint rows integrate detector K on [0,T]",
        "detector_kernel": {
            "K_shape": "int_0^inf d omega phi(omega)/sqrt(omega) exp(+i omega t)",
            "code": "q8_m0_k1_kernel.k_shape_on_times; rows_with_kernels uses production K",
            "eta": "eta_of(gamma)=sqrt(gamma/2pi); do not multiply resolvent by extra 2pi",
        },
        "mixed_operator": True,
        "mixed_operator_meaning": (
            "U0 is the full-phi frequency map; DeltaU and V only accumulate through "
            "the T-window adjoint. t<T uses driven response; t>T still carries U0."
        ),
        "R_N_T": "R_N = (U0 U0^dag - I) + (U0 DeltaU^dag + DeltaU U0^dag) + (DeltaU DeltaU^dag - V V^dag)",
        "R_A_T": "R_A = U V^T - V U^T = U0 V^T + DeltaU V^T - V U0^T - V DeltaU^T",
        "dark_quadratic_only": "dark U0=0 so R_N_dark = DeltaU DeltaU^dag - V V^dag; no U0-cross",
        "do_not_invent_third_split": True,
        "answers_external_AI_q1": "U0 is frequency/stationary closed form; dynamics integrate only to T.",
    }


def theory_tail():
    return {
        "b_out_inf_formal": "b_out(inf) = (U0 + DeltaU_inf) a + V_inf a^dag",
        "delta_b_tail": "delta b = b_out(inf) - b_out(T) = deltaU a + deltaV a^dag",
        "deltaU": "DeltaU_inf - DeltaU_T",
        "deltaV": "V_inf - V_T",
        "tail_keeps_time_varying": ["u(t)", "D(t)", "Maxwell thin-sheet feedback"],
        "not_free_electron_frequency_sum": True,
        "R_N_inf_minus_R_N_T": (
            "U0 deltaU^dag + deltaU U0^dag "
            "+ DeltaU_T deltaU^dag + deltaU DeltaU_T^dag "
            "+ deltaU deltaU^dag "
            "- V_T deltaV^dag - deltaV V_T^dag - deltaV deltaV^dag"
        ),
        "R_A_inf_minus_R_A_T": (
            "U(T) deltaV^T + deltaU V(T)^T + deltaU deltaV^T "
            "- V(T) deltaU^T - deltaV U(T)^T - deltaV deltaU^T"
        ),
        "R_N_inf_equals_zero": "unproven_premise_this_round",
        "do_not_assume_all_finite_window_residual_is_tail": True,
        "coefficients_are_input_columns": True,
    }


def theory_must_ccr():
    return {
        "finite_window_b_out_T_must_be_canonical": False,
        "reason": [
            "mixed operator: full-phi U0 plus dynamics truncated at T",
            "detector K support is not proven compact in [0,T]",
            "dark columns have U0=0, so truncated dynamics are uncompensated by static map",
            "even if the asymptotic map were canonical, b_out(T) is a different operator",
        ],
        "what_would_be_needed_to_close": (
            "boundary/tail identity that reconstructs the missing [T,inf) contraction "
            "with time-varying u, D, Maxwell; or change the observable so its kernel "
            "lives inside the computed window and U0 uses that same kernel"
        ),
        "counterexample_condition_for_Kprime_identity": {
            "observable_support_inside_window": True,
            "U0_and_dynamics_use_same_Kprime": True,
            "phi_prime_0_equals_0": True,
            "must_first_prove": "U0(phi') equals the undriven time-contraction of that K'",
            "then_possibly": "R_N = ||K'||^2 - 1, R_A = 0, same norm convention as int |phi|^2 = 1",
            "not_a_prior": True,
        },
        "do_not_prelabel_1p7e-5_as_explained_window_effect": True,
        "current_RN_order": "O(1e-5) at 9-cycle; four-window RN(0,0) non-monotonic",
    }


def feasibility(adj, cost):
    return {
        "this_round_does_not_run_Kprime": True,
        "this_round_does_not_assemble_UV": True,
        "path_existing_adjoint_rows": {
            "can_do": "repeat current-window R_N(T) with production kernel",
            "cannot_do": "swap K'",
            "why": "force/kernel already baked into saved adjoint rows",
            "caches": adj,
            "cost": "already paid; no new job",
        },
        "path_saved_vertices_plus_rows_with_kernels": {
            "vertices_on_disk": False,
            "would_need": "regenerate vertices for the same window, then rows_with_kernels(K')",
            "not_new_forward_UV_assembly": True,
            "kernel_correction_already_demonstrated_the_row_rebuild": True,
            "quoted_wall_s": {
                "nine_cycle_kernel_correction_elapsed_s": cost["nine_cycle_elapsed_s_from_json"],
                "eleven_cycle_vertices_plus_rows_s": 2622.7,
                "eleven_cycle_including_assemble_s": 3102.7,
            },
            "quote_class": (
                "K' same-window rebuild is the kernel-correction class "
                "(~7 min at 9-cycle elapsed; ~44 min vertices+rows at 11-cycle), "
                "not a new long-time U,V production job"
            ),
            "status": "quoted_not_authorized",
        },
        "path_missing_coverage_or_2048_new_rows": {
            "vertices_2048_on_disk": False,
            "adjoint_rows_macro2048_exists": adj["2048"]["exists"],
            "full_4x4_UV_assembly_2048": False,
            "nm_2x2_timestep_9cycle_2048": "exists in nm_tables_with_timestep.json; not production 4x4 U,V",
            "dynamic_map_split_2048": "dynamic_map_decomposition_macro2048.json exists; R_N split, not 4x4 U,V",
            "if_Kprime_support_outside_existing_window": "missing; estimate not invented; STOP for human review",
            "timestep_1024_may_alias_1e-5": True,
            "do_not_invent_C_equals_I": True,
            "status": "missing_stop_for_human_review",
        },
        "Kprime_diagnostic_not_this_round": {
            "example_support": "smooth drop 8.5-9 to 0, keep phi'(0)=0",
            "U0_and_dynamic_same_Kprime": True,
            "does_not_replace_frozen_phi": True,
            "identity_must_be_derived_first": True,
            "even_if_identity_holds_at_one_grade": [
                "do not declare G1 pass",
                "do not declare W_in audited",
                "do not declare quadrature or timestep passed",
            ],
        },
        "human_review_gates_after_identity_not_chosen_this_round": [
            "keep frozen phi; write R_N(T) into the error statement",
            "or change the in-window detector kernel (changes W_out)",
        ],
    }


def win_vs_win():
    return {
        "window_problem": "finite-T mixed operator vs asymptotic CCR; dark/kept cancellation remainder",
        "W_in_problem": "discrete incoming optical CCR vs port-normalized continuum kernel",
        "separated": True,
        "W_in_is_not_D": True,
        "D_is_not_radiative_damping": True,
        "gamma_is_radiative_feedback_not_D": True,
        "B3_status": "code_formula_mapping_only; numerical_integral_audit not_started",
        "comparison_object": (
            "continuum incoming optical CCR / port-normalized free-field kernel; "
            "pair eta^2 = gamma/2pi; not D"
        ),
        "unverified": [
            "nodes/weights counted once",
            "L/R x 2 polarizations",
            "IR/UV panels",
            "discrete commutator kernel vs continuum kernel",
            "error propagation from node/weight into R_N",
        ],
        "do_not_submit_integral_job": True,
        "1e-8_is_heuristic_only": True,
        "source": "ab12_residual_groups.json slice B3; 线Q_G1_AB12残差列组_复核_20261019.md",
    }


def main():
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("start window-vs-asymptotic inventory\n", encoding="utf-8")
    adj = adjoint_inventory()
    PROGRESS.write_text(PROGRESS.read_text(encoding="utf-8") + "hashed adjoint caches\n", encoding="utf-8")
    cost = kernel_cost_from_disk()
    report = {
        "task": "G1 finite window vs asymptotic CCR; theory + read-only inventory; do not run K'",
        "production_G1_pass": False,
        "quasi_boson": "pending",
        "did_not_assemble_UV": True,
        "did_not_call_rows_with_kernels": True,
        "did_not_run_Kprime": True,
        "did_not_start_long_propagation": True,
        "b_out_T": theory_bout_T(),
        "b_out_inf_and_tail": theory_tail(),
        "finite_window_must_ccr": theory_must_ccr(),
        "feasibility": feasibility(adj, cost),
        "W_in_separated_from_window": win_vs_win(),
        "kernel_cost_quoted_from_archive": cost,
        "do_not_overwrite": [
            "ab12_residual_groups.json",
            "nm_tables.json",
            "kernel_correction_9cycles.*",
            "kernel_correction_11cycles.json",
            "q_slices_bcde.json",
            "q_cov4_fix.json",
        ],
        "provenance_sha256": provenance([__file__]),
    }
    write_new_json(OUTJSON, report)
    PROGRESS.write_text(PROGRESS.read_text(encoding="utf-8") + f"wrote {OUTJSON}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
