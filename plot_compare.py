#!/usr/bin/env python3
"""Compare old and new HHG solver outputs."""

import numpy as np
import matplotlib.pyplot as plt
import os

OLD_HHG_DIR = "C:/Users/26507/Documents/量子光研究/HHG"
NEW_SOLVER_DIR = "C:/Users/26507/Documents/New_SBEs/Quantum-light"
COMPARE_OLD_DIR = os.path.join(NEW_SOLVER_DIR, "compare_old")

def read_old_hhg(filepath):
    """Read old solver HHG: col0=harmonic, col1=para, col2=perp, col5=yeild_tot"""
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 5]

def read_new_hhg(filepath):
    """Read new solver HHG: col0=harmonic_order, col1=omega, col4=HHG_total"""
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 4]

def read_compare_old_hhg(filepath):
    """Read compare_old HHG: col0=harmonic, col1=omega, col4=HHG_total"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    harmonic = []
    hhg = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                harmonic.append(float(parts[0]))
                hhg.append(float(parts[4]))
            except ValueError:
                continue
    return np.array(harmonic), np.array(hhg)

def read_Jt(filepath):
    """Read Jt: col1=time, col2=Jx, col3=Jy"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    time_vals = []
    jx_vals = []
    jy_vals = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                time_vals.append(float(parts[1]))
                jx_vals.append(float(parts[2]))
                jy_vals.append(float(parts[3]))
            except ValueError:
                continue
    return np.array(time_vals), np.array(jx_vals), np.array(jy_vals)

def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Top left: Old solver HHG spectra ---
    ax = axes[0, 0]
    old_files = {
        'total': os.path.join(OLD_HHG_DIR, 'hhg_tot.txt'),
        'transverse': os.path.join(OLD_HHG_DIR, 'hhg_tra.txt'),
        'terrace': os.path.join(OLD_HHG_DIR, 'hhg_ter.txt'),
    }
    for name, path in old_files.items():
        if os.path.exists(path):
            h, y = read_old_hhg(path)
            ax.plot(h, y, label=name, linewidth=1.2)
    ax.set_xlabel('Harmonic Order (w/w0)')
    ax.set_ylabel('HHG Yield')
    ax.set_title('Old Solver: All Components')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)

    # --- Top right: Old total vs compare_old (new solver, 20 bands, ncyc=4) ---
    ax = axes[0, 1]
    h_old, y_old = read_old_hhg(old_files['total'])
    ax.plot(h_old, y_old, 'k-', label='Old Solver (total)', linewidth=1.5)

    compare_hhg = os.path.join(COMPARE_OLD_DIR, 'HHG.dat')
    if os.path.exists(compare_hhg):
        h_new_20, y_new_20 = read_compare_old_hhg(compare_hhg)
        ax.plot(h_new_20, y_new_20, 'r--', label='New Solver (20 bands, ncyc=4)', linewidth=1.2)
        print(f"compare_old HHG: {len(h_new_20)} points")

    ax.set_xlabel('Harmonic Order (w/w0)')
    ax.set_ylabel('HHG Yield')
    ax.set_title('Old vs New (20 bands, ncyc=4)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)

    # --- Bottom left: Current new solver output (ncyc=4, 20 bands from input/) ---
    ax = axes[1, 0]
    input_hhg = os.path.join(NEW_SOLVER_DIR, 'input/HHG.dat')
    if os.path.exists(input_hhg):
        h_in, y_in = read_new_hhg(input_hhg)
        ax.plot(h_in, y_in, 'b-', label='New Solver (input/HHG.dat)', linewidth=1.2)
        ax.plot(h_old, y_old, 'k--', label='Old Solver (total)', linewidth=1.2, alpha=0.7)
        print(f"input/HHG.dat: {len(h_in)} points, harmonic range [{h_in.min():.2f}, {h_in.max():.2f}]")

    ax.set_xlabel('Harmonic Order (w/w0)')
    ax.set_ylabel('HHG Yield')
    ax.set_title('New Solver Output vs Old (input/HHG.dat)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)

    # --- Bottom right: J(t) comparison ---
    ax = axes[1, 1]

    # Read old Jt (binary, skip for now - just compare new solver outputs)
    # Read compare_old Jt
    compare_jt = os.path.join(COMPARE_OLD_DIR, 'Jt.dat')
    if os.path.exists(compare_jt):
        t_old, jx_old, jy_old = read_Jt(compare_jt)
        ax.plot(t_old, jx_old, 'r-', label='Old solver Jx', linewidth=1.0, alpha=0.8)

    # Read input Jt
    input_jt = os.path.join(NEW_SOLVER_DIR, 'input/Jt.dat')
    if os.path.exists(input_jt):
        t_new, jx_new, jy_new = read_Jt(input_jt)
        ax.plot(t_new, jx_new, 'b--', label='New solver Jx', linewidth=1.0, alpha=0.8)

    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Jx (a.u.)')
    ax.set_title('Current Jx(t) Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(NEW_SOLVER_DIR, 'hhg_comparison.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

    # Print key harmonic yields for comparison
    print("\n=== Old Solver HHG Yields (total) ===")
    key_orders = [1, 3, 5, 7, 9, 11, 13, 15]
    for order in key_orders:
        idx = np.argmin(np.abs(h_old - order))
        if abs(h_old[idx] - order) < 0.5:
            print(f"  Order {order:2d}: {y_old[idx]:.6e}")

    if os.path.exists(compare_hhg):
        print("\n=== New Solver HHG Yields (20 bands, ncyc=4) ===")
        for order in key_orders:
            idx = np.argmin(np.abs(h_new_20 - order))
            if abs(h_new_20[idx] - order) < 0.5:
                print(f"  Order {order:2d}: {y_new_20[idx]:.6e}")

    print("\n=== compare_metrics (fixedE0) ===")
    metrics = {}
    with open(os.path.join(COMPARE_OLD_DIR, 'compare_metrics_fixedE0.txt')) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=')
                metrics[k] = v
    for k, v in metrics.items():
        print(f"  {k} = {v}")

if __name__ == '__main__':
    main()
