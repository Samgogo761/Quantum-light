# TB-v2 freeze (full112 −N)

Policy (2026-08-13):

- `+N`: production `/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat`
- `−N`: `/public/home/wangjs/project/CrI3_TB_mN/wannier/CrI3_tb_mN_Pconj_v2.dat`
  - `H` identical to production E18.8 `CrI3_tb_mN_Pconj.dat`
  - `r` rewritten ES25.17 from the same +N P-conjugate construction
- Production `CrI3_tb_mN_Pconj.dat` is **not** overwritten.

GH3-v2 / GH5-v2 use this pair. Job 27992 remains historical selection + E18.8/ES25 serialization control.

SHA and k20 ΔD/Δv audit: `AUDIT_TB_V2.json` / `FREEZE.json` (filled on the server after write).
