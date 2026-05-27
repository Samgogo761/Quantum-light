#!/bin/bash
#=============================================================================
# Combine BSV array job results
# Averages HHG_bsv.dat from all task directories
#
# Usage: bash deploy/combine_bsv.sh [N_JOBS]
#   Default N_JOBS=10. Automatically handles missing tasks.
#=============================================================================

WORKDIR="${BSV_WORKDIR:-/public/home/wangjs/project/New_SBEs/Quantum-light}"
BSV_DIR="${WORKDIR}/output_bsv"
N_JOBS=${1:-10}

echo "Scanning for BSV results (up to ${N_JOBS} tasks)..."

FOUND=0
FILES=""
for i in $(seq 0 $((N_JOBS - 1))); do
    f="${BSV_DIR}/task_${i}/HHG_bsv.dat"
    if [ -f "$f" ]; then
        FILES="${FILES} ${f}"
        FOUND=$((FOUND + 1))
        echo "  Found: task_${i}/HHG_bsv.dat"
    else
        echo "  Missing: task_${i}/HHG_bsv.dat"
    fi
done

if [ "${FOUND}" -eq 0 ]; then
    echo "ERROR: No completed tasks found."
    exit 1
fi

echo ""
echo "Combining ${FOUND} task files..."

# Each file has header lines (# ...) then data lines.
# Computes mean and standard error across task files.
awk '
/^#/ { next }
{
    idx = FNR - 1
    if (idx < 1) idx = 1
    col1[idx] = $1
    col2[idx] = $2
    sum3[idx] += $3 + 0
    sum3sq[idx] += ($3 + 0)^2
    cnt[idx]++
    if (idx > maxidx) maxidx = idx
}
END {
    print "# harmonic_order  omega(a.u.)  HHG_bsv_avg  HHG_bsv_stderr"
    for (i = 1; i <= maxidx; i++) {
        if (cnt[i] > 0) {
            avg = sum3[i] / cnt[i]
            var = sum3sq[i] / cnt[i] - avg^2
            if (var < 0) var = 0
            se = (cnt[i] > 1) ? sqrt(var / (cnt[i] - 1)) : 0
            printf "%10s %16s %16.8e %16.8e\n", col1[i], col2[i], avg, se
        }
    }
}
' ${FILES} > "${BSV_DIR}/HHG_bsv_combined.dat"

NLINES=$(grep -c '^[^#]' "${BSV_DIR}/HHG_bsv_combined.dat")
echo "Combined spectrum: ${BSV_DIR}/HHG_bsv_combined.dat (${NLINES} points, ${FOUND} tasks)"

echo ""
echo "--- Combined BSV HHG (first 10 harmonics) ---"
awk '$1 ~ /^[0-9]/ && $1+0 >= 0.8 && $1+0 <= 10.2 && ($1+0)%1 < 0.15 {
    printf "  H%-3d  HHG_bsv = %s\n", int($1+0.5), $3
}' "${BSV_DIR}/HHG_bsv_combined.dat"

echo ""
echo "Done."
