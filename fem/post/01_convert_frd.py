#!/usr/bin/env python3
"""
Step 1 — Convert CCX .frd files to ParaView .vtu/.pvd for all four compositions.

Run with system python3 (NOT pvpython) — only needs ccx2paraview + vtk.
This is the slow step; run once and keep the output.

Usage:
    python3 01_convert_frd.py
"""

from pathlib import Path
from ccx2paraview import Converter
import time

# ── composition registry ──────────────────────────────────────────────────
PROJECT_ROOT = Path("~/dft_projects/fe-cr_cu-ni_DFT").expanduser()
FEM_CONICAL  = PROJECT_ROOT / "fem" / "conical"

COMPOSITIONS = {
    "fe16cr00": FEM_CONICAL / "fe16cr00" / "indentation_fe16cr00_v19.frd",
    "fe08cr08": FEM_CONICAL / "fe08cr08" / "indentation_fe08cr08_v19.frd",
    "fe04cr12": FEM_CONICAL / "fe04cr12" / "indentation_fe04cr12_v19.frd",
    "fe00cr16": FEM_CONICAL / "fe00cr16" / "indentation_fe00cr16_v19.frd",
}

for label, frd_path in COMPOSITIONS.items():
    if not frd_path.exists():
        print(f"[{label}] SKIP — .frd not found at {frd_path}")
        continue

    pvd_path = frd_path.with_suffix(".pvd")
    if pvd_path.exists():
        print(f"[{label}] SKIP — .pvd already exists at {pvd_path}")
        continue

    print(f"[{label}] Converting {frd_path.name} ...")
    t0 = time.time()
    try:
        c = Converter(str(frd_path), ["vtu"])
        c.run()
        dt = time.time() - t0
        print(f"[{label}] DONE in {dt:.1f}s -> {pvd_path}")
    except Exception as exc:
        print(f"[{label}] ERROR: {exc}")

print("\nAll conversions attempted. Check each composition directory for .pvd + .vtu files.")
