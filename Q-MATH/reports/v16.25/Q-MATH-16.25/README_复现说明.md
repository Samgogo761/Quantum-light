# Q-MATH-16.25 reproducibility package

Contents:
- Report: Markdown and Word
- qmath_16_25_dynamic_ward.py: direct retarded kernel / static contact audit
- qmath_16_25_retarded_downfolding.py: dynamic Feshbach/downfolding implementation
- qmath_16_25_static_subtracted_kernel.py: Ward-subtracted kernel with BZ convergence
- qmath_16_25_static_subtracted_results.json: final compact results
- qmath_16_25_retarded_downfolding_results.json: raw retarded-kernel outputs
- figures: omitted-channel fraction, dynamic kernel comparison, BZ convergence
- base/: extracted Q-MATH-16.24 original source and dependencies for direct reproducibility

Scope warning: These are linear-response calculations on the original eight-band M0. They are not nonlinear HHG, BSV output, or a quantum-advantage certification. The SOC=0 control is gapless and excluded from the insulating Ward-subtracted expansion.

Run (Python with numpy, scipy, matplotlib):
`python qmath_16_25_dynamic_ward.py`
`python qmath_16_25_retarded_downfolding.py`
`python qmath_16_25_static_subtracted_kernel.py`
