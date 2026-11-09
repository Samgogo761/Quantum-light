# hBN A0 end-to-end report

- solver: `C:\Users\26507\Documents\New_SBEs\Quantum-light\hhg_sbe.exe`

## 1. Classical complex FFT regression
- max relative error HHG.dat vs HHG_complex total: `0.000e+00`
- gate `<1e-10`: **PASS**

## 2. Coherent Gauss-Hermite (α0=30, θ=90°, ⟨I⟩ matched to classical)
- ICS≥CS (num): **PASS**
- ICS≈classical @H1/H3/H5 ratios: `[1.0014, 1.0112, 0.9865]` (gate |r-1|<5%: PASS)
- ICS vs CS max rel (informational, not hard gate): `7.471e-01`
- nodes manifest exists: **True**
- overall: **PASS**

## 3. Squeezed-vacuum direction smoke (r=0.8, θs=0° vs 180°)
- ICS≥CS @θ=0: **PASS**
- ICS≥CS @θ=180: **PASS**
- node (I,φ) distribution L2 distance: `1.394e+13` (gate >0: PASS)
- HHG_nodes_modes.dat present: **True**
- overall: **PASS**

## Notes
- Equal-mean-intensity mapping: not experimental BSV energy calibration.
- Do NOT interpret ICS-CS as quantum squeezing.

## OVERALL: **PASS**
