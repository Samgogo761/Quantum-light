from pathlib import Path

lines = Path(
    "/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref112_large/CrI3_112large.wout"
).read_text(errors="replace").splitlines()
for a, b in [(120, 180), (280, 430), (880, 930)]:
    print(f"===== lines {a+1}-{b} =====")
    print("\n".join(f"{i+1:5d}|{lines[i]}" for i in range(a, min(b, len(lines)))))

pl = Path("/public/home/wangjs/project/CrI3_TB/wannier/CrI3.wout").read_text(errors="replace").splitlines()
print("===== prod selected =====")
for i, ln in enumerate(pl):
    if any(
        s in ln
        for s in (
            "Omega I",
            "Omega Total",
            "Number of Wannier",
            "Number of input",
            "inner window",
            "outer window",
            "Disentanglement convergence",
        )
    ):
        if i < 500 or "Omega" in ln or "Disentang" in ln:
            print(f"{i+1:5d}|{ln}")
