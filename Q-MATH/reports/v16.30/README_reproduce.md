# Q-MATH-16.30 minimal reproducibility pack
Python dependencies: numpy, scipy, matplotlib; MS Word optional for the docx report.

From unpacked package root, run:
  OPENBLAS_NUM_THREADS=1 python qmath_16_30_bridge_audit.py
  python qmath_16_30_output_mode_gram.py

The bridge script runs L=9,17,25; the clean results JSON also includes L=41,61,81 and gamma=.03 checkpoint runs from the same standalone implementation. The original reference Q16.7 complete 36D matrix-element implementation is inside q16_27_reference/source/reference_q167. The Q16.29 classical-chi proxy cached matrices and input energy data are included.

Evidence warning: all new T outputs are regulated *bare* elastic 3-photon T coefficients. They are not asymptotic LSZ-normalized, continuum-propagated H2 output nor actual emission photon count. The classical-positive-P dual bound is exact only in the explicitly specified linear anomalous-moment proxy. Do not quote it as experimental quantum advantage.
