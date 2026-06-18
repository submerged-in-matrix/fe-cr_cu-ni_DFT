#!/usr/bin/env python3
"""
Step 2 — Scan all four .vtkhdf time series to compute GLOBAL min/max for each
field (U2, Mises, SYY, EYY) across ALL compositions and ALL increments.

This must run BEFORE step 3, since the rendering script needs fixed
colorbar ranges to make a fair visual comparison across compositions.

Run with pvpython (needs the ParaView Python API):
    pvpython 02_scan_global_ranges.py

REQUIRES: 01b_consolidate_vtkhdf.py to have been run first (ParaView 6.0+;
the apt-packaged 5.11.2 cannot write .vtkhdf — see session notes), producing
one .vtkhdf file per composition instead of 119+ loose .vtu files.

Writes: global_ranges.json
"""

from paraview.simple import *
import json
from pathlib import Path

PROJECT_ROOT = Path("~/dft_projects/fe-cr_cu-ni_DFT").expanduser()
FEM_CONICAL  = PROJECT_ROOT / "fem" / "conical"

# Reads the consolidated .vtkhdf files (produced by 01b_consolidate_vtkhdf.py).
# Array names CONFIRMED directly against indentation_fe16cr00_v19.vtkhdf:
#   Point arrays: E, ERROR, E_Mises, E_Principal, S, S_Mises, S_Principal, U
#   Cell arrays:  (none — all data is nodal)
COMPOSITIONS = {
    "fe16cr00": FEM_CONICAL / "fe16cr00" / "indentation_fe16cr00_v19.vtkhdf",
    "fe08cr08": FEM_CONICAL / "fe08cr08" / "indentation_fe08cr08_v19.vtkhdf",
    "fe04cr12": FEM_CONICAL / "fe04cr12" / "indentation_fe04cr12_v19.vtkhdf",
    "fe00cr16": FEM_CONICAL / "fe00cr16" / "indentation_fe00cr16_v19.vtkhdf",
}

# Field name : (array name as written by ccx2paraview, component index or 'magnitude')
# NOTE: ccx2paraview writes vector/tensor fields with their CCX names (U, S, E).
# Mises and principal components are exported as separate derived scalar arrays.
# VERIFY these array names against one .vtu file before trusting this script —
# see the verification block at the bottom of this file.
FIELDS = {
    "U2":    {"array": "U",          "component": 1},   # D2 = axial displacement
    "Mises": {"array": "S_Mises",    "component": -1},  # scalar, no component index
    "SYY":   {"array": "S",          "component": 1},   # SXX=0, SYY=1, SZZ=2, SXY=3, SYZ=4, SZX=5
    "EYY":   {"array": "E",          "component": 1},
}

global_ranges = {k: [float("inf"), float("-inf")] for k in FIELDS}

for label, vtkhdf_path in COMPOSITIONS.items():
    if not vtkhdf_path.exists():
        print(f"[{label}] SKIP — .vtkhdf not found at {vtkhdf_path}")
        continue

    print(f"[{label}] Scanning {vtkhdf_path.name} ...")
    reader = OpenDataFile(str(vtkhdf_path))
    reader.UpdatePipeline()

    tk = GetTimeKeeper()
    timesteps = list(reader.TimestepValues) if reader.TimestepValues else [0.0]
    print(f"[{label}]   {len(timesteps)} increments found")

    for t in timesteps:
        reader.UpdatePipeline(time=t)
        di = reader.GetDataInformation()

        for fname, cfg in FIELDS.items():
            arr_name = cfg["array"]
            comp = cfg["component"]
            point_info = di.GetPointDataInformation().GetArrayInformation(arr_name)
            cell_info  = di.GetCellDataInformation().GetArrayInformation(arr_name)
            arr_info = point_info if point_info is not None else cell_info
            if arr_info is None:
                continue  # array not found on this dataset, skip silently this step

            if comp == -1:
                rng = arr_info.GetComponentRange(0)
            else:
                rng = arr_info.GetComponentRange(comp)

            if rng is None:
                continue
            lo, hi = rng
            if lo < global_ranges[fname][0]:
                global_ranges[fname][0] = lo
            if hi > global_ranges[fname][1]:
                global_ranges[fname][1] = hi

    Delete(reader)
    del reader

print("\n=== Global ranges across all compositions and increments ===")
for fname, (lo, hi) in global_ranges.items():
    print(f"  {fname:8s}  min={lo:.6g}  max={hi:.6g}")

out_path = Path(__file__).parent / "global_ranges.json"
with open(out_path, "w") as f:
    json.dump(global_ranges, f, indent=2)
print(f"\nSaved: {out_path}")

print("""
─────────────────────────────────────────────────────────────────────────
Array names CONFIRMED 2026-06 against indentation_fe16cr00_v19.vtkhdf:
  Point arrays: E, ERROR, E_Mises, E_Principal, S, S_Mises, S_Principal, U
  Cell arrays:  (none)
All data is nodal (point data) — assoc='POINTS' in step 3 is correct.
─────────────────────────────────────────────────────────────────────────
""")