#!/bin/bash
#=============================================================================
# Combine BSV array job results
# Averages HHG_bsv.dat from all task directories
#
# Usage: bash deploy/combine_bsv.sh
#=============================================================================

WORKDIR="/public/home/wangjs/project/Quantum-light"
BSV_DIR="${WORKDIR}/output_bsv"
N_JOBS=10

echo "Combining BSV results from ${N_JOBS} tasks..."

# Check all tasks completed
MISSING=0
for i in $(seq 0 $((N_JOBS - 1))); do
    if [ ! -f "${BSV_DIR}/task_${i}/HHG_bsv.dat" ]; then
        echo "  MISSING: task_${i}/HHG_bsv.dat"
        MISSING=$((MISSING + 1))
    fi
done

if [ "${MISSING}" -gt 0 ]; then
    echo "ERROR: ${MISSING} tasks incomplete. Wait for all jobs to finish."
    exit 1
fi

echo "All ${N_JOBS} task files found."

# Average: read all files, sum column 3, divide by N_JOBS
# Each task already averaged its own samples, so final = mean of means
awk -v nj="${N_JOBS}" '
BEGIN { n = 0 }
FNR == 1 && NR > 1 { next_file = 1 }
/^#/ { if (file_count == 0) header = $0; next }
{
    if (FNR == 1 || next_file) { file_count++; next_file = 0 }
    idx = FNR
    col1[idx] = $1
    col2[idx] = $2
    sum3[idx] += $3
    if (idx > n) n = idx
}
END {
    print "# harmonic_order  omega(a.u.)  HHG_bsv_avg"
    for (i = 1; i <= n; i++) {
        if (col1[i] != "") {
            printf "%10s %16s %16.8e\n", col1[i], col2[i], sum3[i] / nj
        }
    }
}
' "${BSV_DIR}"/task_*/HHG_bsv.dat > "${BSV_DIR}/HHG_bsv_combined.dat"

NLINES=$(grep -c '^[^#]' "${BSV_DIR}/HHG_bsv_combined.dat")
echo "Combined spectrum written to: ${BSV_DIR}/HHG_bsv_combined.dat"
echo "  ${NLINES} frequency points"

echo ""
echo "--- Combined BSV HHG (first 10 harmonics) ---"
awk '$1 ~ /^[0-9]/ && $1+0 >= 0.8 && $1+0 <= 10.2 && ($1+0)%1 < 0.15 {
    printf "  H%-3d  HHG_bsv = %s\n", int($1+0.5), $3
}' "${BSV_DIR}/HHG_bsv_combined.dat"

echo ""
echo "Done."
