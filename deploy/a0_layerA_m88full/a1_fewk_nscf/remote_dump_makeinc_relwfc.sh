#!/bin/bash
set -eo pipefail
export PATH=/usr/bin:/bin:$PATH
QE=/public/home/wangjs/software/q-e-qe-7.2
OUT=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf/logs
mkdir -p "$OUT"
sed -n '1,80p' "$QE/make.inc" > "$OUT/qe72_make.inc.head.txt"
sed -n '150,170p' "$QE/PP/src/Makefile" > "$OUT/qe72_pp_makefile_link.txt"
PY=/public/home/wangjs/software/miniconda3/bin/python
$PY - <<'PY'
import re
from pathlib import Path
p = Path("/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf/temp_a1h_scf_clean/CrI3_scf_soc.save/I.rel-pbe.upf")
t = p.read_text(errors="replace")
out = Path("/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf/logs/I_upf_spin_orb.txt")
m = re.search(r"<PP_SPIN_ORB>(.*?)</PP_SPIN_ORB>", t, re.S)
chunks = []
chunks.append("SPIN_ORB found %s len %s\n" % (bool(m), 0 if not m else len(m.group(1))))
if m:
    chunks.append(m.group(1)[:4000])
    chunks.append("\n")
for tag in ("PP_RELWFC", "PP_RELBETA"):
    n = t.count("<" + tag)
    chunks.append("%s count_open=%s\n" % (tag, n))
    for mm in re.finditer(r"<%s\.(\d+)([^>]*)>" % tag, t):
        chunks.append("%s.%s %s\n" % (tag, mm.group(1), mm.group(2)[:240]))
out.write_text("".join(chunks), encoding="utf-8")
print("wrote", out)
PY
echo DUMP_OK
