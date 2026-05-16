#!/usr/bin/env python3
"""Compare old and NEWLY RUN new solver HHG results."""

import numpy as np
import matplotlib.pyplot as plt
import os

OLD_DIR = "C:/Users/26507/Documents/量子光研究"
NEW_DIR = "C:/Users/26507/Documents/量子光研究/新SBEs/Quantum-light-claude-wannier90-fortran-integration-dydlH"

def read_old_hhg(filepath):
    """Read: harmonic, para, perp, yield_R, yield_L, yield_tot"""
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 5]

def read_new_hhg(filepath):
    """Read: harmonic_order, omega, HHG_x, HHG_y, HHG_total"""
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 4]

def read_new_jt(filepath):
    """Read: it, time, Jx, Jy"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    t, jx, jy = [], [], []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                t.append(float(parts[1]))
                jx.append(float(parts[2]))
                jy.append(float(parts[3]))
            except ValueError:
                continue
    return np.array(t), np.array(jx), np.array(jy)

def read_old_jt(filepath):
    """Read old solver Jt binary/formatted - check format first"""
    try:
        data = np.loadtxt(filepath)
        return data[:, 0], data[:, 1], data[:, 2]
    except:
        with open(filepath, 'r') as f:
            content = f.read()
        # Check if binary
        if content[:4].encode('hex') != '':
            print(f"  {filepath} appears to be binary, skipping")
            return None, None, None
        return None, None, None

def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # === Top-left: Old solver HHG (total) ===
    ax = axes[0, 0]
    old_hhg = os.path.join(OLD_DIR, "HHG/hhg_tot.txt")
    if os.path.exists(old_hhg):
        h_old, y_old = read_old_hhg(old_hhg)
        ax.plot(h_old, y_old, 'k-', label='Old Solver', linewidth=1.5)
        print(f"Old HHG: {len(h_old)} points, range [{h_old.min():.1f}, {h_old.max():.1f}]")

    ax.set_xlabel('Harmonic Order (w/w0)')
    ax.set_ylabel('HHG Yield')
    ax.set_title('Old Solver HHG (90 bands, ncyc=2→4, cos²)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)

    # === Top-right: New solver HHG (NEWLY RUN) ===
    ax = axes[0, 1]
    new_hhg = os.path.join(NEW_DIR, "HHG.dat")
    if os.path.exists(new_hhg):
        h_new, y_new = read_new_hhg(new_hhg)
        ax.plot(h_new, y_new, 'r-', label='New Solver (90 bands, ncyc=4)', linewidth=1.5)
        print(f"New HHG: {len(h_new)} points, range [{h_new.min():.1f}, {h_new.max():.1f}]")
    else:
        ax.text(0.5, 0.5, 'New solver output not yet available', ha='center', va='center', transform=ax.transAxes)
        print("New solver output NOT FOUND")

    ax.set_xlabel('Harmonic Order (w/w0)')
    ax.set_ylabel('HHG Yield')
    ax.set_title('New Solver HHG (90 bands, ncyc=4)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)

    # === Bottom-left: Overlay comparison ===
    ax = axes[1, 0]
    if os.path.exists(old_hhg) and os.path.exists(new_hhg):
        h_old, y_old = read_old_hhg(old_hhg)
        h_new, y_new = read_new_hhg(new_hhg)
        ax.plot(h_old, y_old, 'k-', label='Old Solver', linewidth=1.5)
        ax.plot(h_new, y_new, 'r--', label='New Solver', linewidth=1.5)
        ax.set_xlabel('Harmonic Order (w/w0)')
        ax.set_ylabel('HHG Yield')
        ax.set_title('Old vs New: Direct Comparison')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 30)

        # Print key harmonic yields
        print("\n=== HHG Yield Comparison ===")
        print(f"{'Order':>6} {'Old':>15} {'New':>15} {'Ratio(New/Old)':>15}")
        for order in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
            i_old = np.argmin(np.abs(h_old - order))
            i_new = np.argmin(np.abs(h_new - order))
            if abs(h_old[i_old] - order) < 0.5 and abs(h_new[i_new] - order) < 0.5:
                ratio = y_new[i_new] / y_old[i_old] if y_old[i_old] > 0 else float('nan')
                print(f"{order:>6} {y_old[i_old]:>15.6e} {y_new[i_new]:>15.6e} {ratio:>15.2f}")
    else:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)

    # === Bottom-right: J(t) comparison ===
    ax = axes[1, 1]

    # Read new Jt
    new_jt = os.path.join(NEW_DIR, "Jt.dat")
    if os.path.exists(new_jt):
        t_new, jx_new, jy_new = read_new_jt(new_jt)
        ax.plot(t_new, jx_new, 'r-', label='New Solver Jx', linewidth=1.0, alpha=0.8)
        print(f"\nNew Jt: {len(t_new)} points, t range [{t_new.min():.2f}, {t_new.max():.2f}] fs")

    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Jx (a.u.)')
    ax.set_title('Current Jx(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(NEW_DIR, 'hhg_comparison_new.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

if __name__ == '__main__':
    main()
