#!/usr/bin/env python3
"""Analyze new solver HHG results against expectations."""

import numpy as np
import os

NEW_HHG = "C:/Users/26507/Documents/量子光研究/新SBEs/Quantum-light-claude-wannier90-fortran-integration-dydlH/input/HHG.dat"
OLD_HHG = "C:/Users/26507/Documents/量子光研究/HHG/hhg_tot.txt"

def read_new_hhg(filepath):
    """harmonic_order, omega, HHG_x, HHG_y, HHG_total"""
    data = np.loadtxt(filepath, skiprows=1)
    return data[:, 0], data[:, 4]

def read_old_hhg(filepath):
    """harmonic, para, perp, yield_R, yield_L, yield_tot"""
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 5]

print("=== 步骤3: HHG产额对比 ===\n")

# Read new solver data
h_new, y_new = read_new_hhg(NEW_HHG)
print(f"新求解器: {len(h_new)} 点, 谐波范围 [{h_new.min():.1f}, {h_new.max():.1f}]")

# Find key harmonics
print("\n新求解器 HHG_total (修复后):")
for order in [1, 3, 5, 7, 9]:
    idx = np.argmin(np.abs(h_new - order))
    if abs(h_new[idx] - order) < 0.5:
        print(f"  H{order}: {y_new[idx]:.6e}")

# Expected values after fix
print("\n用户预期的修复后值:")
expected = {1: 290, 3: 206, 5: 8.7, 7: 0.025, 9: 1e-4}
for order, val in expected.items():
    print(f"  H{order}: ~{val}")

# Ratio vs expected
print("\n实际值 vs 预期值:")
for order in [1, 3, 5, 7, 9]:
    idx = np.argmin(np.abs(h_new - order))
    if abs(h_new[idx] - order) < 0.5:
        ratio = y_new[idx] / expected[order]
        print(f"  H{order}: 实际={y_new[idx]:.2e}, 预期={expected[order]:.2e}, 比值={ratio:.1e}x")

# Read old solver data
h_old, y_old = read_old_hhg(OLD_HHG)
print(f"\n旧求解器 (112带): {len(h_old)} 点")

print("\n=== 步骤4: 与旧求解器对比 ===\n")
print(f"{'阶次':>6} {'旧(112带)':>15} {'新(20带)':>15} {'比值(新/旧)':>15}")
for order in [1, 3, 5, 7, 9]:
    i_old = np.argmin(np.abs(h_old - order))
    i_new = np.argmin(np.abs(h_new - order))
    if abs(h_old[i_old] - order) < 0.5 and abs(h_new[i_new] - order) < 0.5:
        ratio = y_new[i_new] / y_old[i_old] if y_old[i_old] > 0 else float('nan')
        print(f"H{order:>2} {y_old[i_old]:>15.2e} {y_new[i_new]:>15.2e} {ratio:>15.2f}")

print("\n=== 关键发现 ===")
for order in [1, 3, 5, 7, 9]:
    i_old = np.argmin(np.abs(h_old - order))
    i_new = np.argmin(np.abs(h_new - order))
    if abs(h_old[i_old] - order) < 0.5 and abs(h_new[i_new] - order) < 0.5:
        ratio = y_new[i_new] / y_old[i_old]
        # User expected ~0.5x for low harmonics
        if ratio > 10:
            status = "❌ 差异过大"
        elif ratio > 2:
            status = "⚠️ 偏高"
        elif ratio < 0.2:
            status = "⚠️ 偏低"
        else:
            status = "✅ 合理"
        print(f"  H{order}: {status} (比值={ratio:.2f})")
