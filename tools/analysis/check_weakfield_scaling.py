#!/usr/bin/env python3
"""
Weak-field scaling check for HHG solver.

Validates perturbative regime:
  H1 (fundamental):  J(w)/E0     should be constant
  H2 (second harm.): J(2w)/E0^2  should be constant
  H3 (third harm.):  J(3w)/E0^3  should be constant

Usage:
  python check_weakfield_scaling.py output_weakfield/

Expects subdirectories E0/, E0_2/, E0_4/ each containing HHG.dat.
"""
import sys
import os
import numpy as np

def load_hhg(path):
    data = np.loadtxt(path, comments='#')
    return data[:, 0], data[:, -1]  # harmonic_order, HHG_total

def find_peak(h_order, hhg, target_h, window=0.3):
    mask = np.abs(h_order - target_h) < window
    if not mask.any():
        return 0.0
    return np.max(hhg[mask])

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else 'output_weakfield'

    I_full = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0e11  # W/cm^2
    configs = [
        ('E0',   I_full),
        ('E0_2', I_full / 4.0),
        ('E0_4', I_full / 16.0),
    ]

    results = {}
    for label, intensity in configs:
        path = os.path.join(base, label, 'HHG.dat')
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
        h_order, hhg = load_hhg(path)
        E0 = np.sqrt(intensity)  # relative scale
        results[label] = {
            'E0': E0,
            'H1': find_peak(h_order, hhg, 1.0),
            'H2': find_peak(h_order, hhg, 2.0),
            'H3': find_peak(h_order, hhg, 3.0),
        }

    if len(results) < 2:
        print("ERROR: Need at least 2 data sets for scaling comparison.")
        sys.exit(1)

    print("=" * 70)
    print("  Weak-field scaling check")
    print("=" * 70)
    print(f"  {'Config':<8} {'E0_rel':<12} {'H1':<14} {'H2':<14} {'H3':<14}")
    print("-" * 70)
    for label in ['E0', 'E0_2', 'E0_4']:
        if label not in results:
            continue
        r = results[label]
        print(f"  {label:<8} {r['E0']:<12.4e} {r['H1']:<14.6e} {r['H2']:<14.6e} {r['H3']:<14.6e}")

    print()
    print("  Scaling ratios (should be ~1 if perturbative):")
    print(f"  {'Pair':<16} {'H1/E0^2':<14} {'H2/E0^4':<14} {'H3/E0^6':<14}")
    print("-" * 70)

    ref_label = 'E0'
    if ref_label not in results:
        ref_label = list(results.keys())[0]
    ref = results[ref_label]

    for label in ['E0_2', 'E0_4']:
        if label not in results:
            continue
        r = results[label]
        e_ratio = r['E0'] / ref['E0']

        # HHG ~ |J(w)|^2 ~ E0^(2n), so ratio should be e_ratio^(2n)
        h1_ratio = r['H1'] / ref['H1'] / e_ratio**2 if ref['H1'] > 0 else float('nan')
        h2_ratio = r['H2'] / ref['H2'] / e_ratio**4 if ref['H2'] > 0 else float('nan')
        h3_ratio = r['H3'] / ref['H3'] / e_ratio**6 if ref['H3'] > 0 else float('nan')
        print(f"  {ref_label}→{label:<10} {h1_ratio:<14.4f} {h2_ratio:<14.4f} {h3_ratio:<14.4f}")

    print()
    print("  Empirical exponents p (HHG ~ E0^p, ideal: H1=2, H2=4, H3=6):")
    print(f"  {'Pair':<16} {'H1 p':<10} {'H2 p':<10} {'H3 p':<10}")
    print("-" * 70)

    labels_sorted = [l for l in ['E0', 'E0_2', 'E0_4'] if l in results]
    for i in range(len(labels_sorted) - 1):
        l1, l2 = labels_sorted[i], labels_sorted[i+1]
        r1, r2 = results[l1], results[l2]
        log_e = np.log(r2['E0'] / r1['E0'])
        if abs(log_e) < 1e-30:
            continue
        exps = []
        for h_key in ['H1', 'H2', 'H3']:
            if r1[h_key] > 0 and r2[h_key] > 0:
                exps.append(np.log(r2[h_key] / r1[h_key]) / (2 * log_e))
            else:
                exps.append(float('nan'))
        print(f"  {l1}→{l2:<10} {exps[0]:<10.4f} {exps[1]:<10.4f} {exps[2]:<10.4f}")

    print()
    print("  Ratios near 1.0 = perturbative regime confirmed.")
    print("  Empirical p near ideal = perturbative scaling holds.")
    print("  Large deviations = non-perturbative effects (expected for strong fields).")
    print("=" * 70)

if __name__ == '__main__':
    main()
