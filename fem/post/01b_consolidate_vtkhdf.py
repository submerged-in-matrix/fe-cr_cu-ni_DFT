#!/usr/bin/env python3
"""
Step 1b — Consolidate the per-increment .vtu files (one per timestep, as
produced by ccx2paraview) into a SINGLE .vtkhdf file per composition.

This directly addresses folder clutter: ccx2paraview creates one .vtu file
per output interval (119 files for a 119-increment run). The official
ccx2paraview README documents bundling these into one .vtkhdf file via
ParaView's Python API. After writing the .vtkhdf and verifying it, the
loose .vtu files + .pvd are removed.

Run with pvpython (needs the ParaView Python API):
    pvpython 01b_consolidate_vtkhdf.py

Run AFTER 01_convert_frd.py and BEFORE 02_scan_global_ranges.py.
Steps 2 and 3 should then read the .vtkhdf files instead of .pvd —
see the path change noted at the bottom of this file.
"""

from paraview.simple import SaveData, PVDReader, Delete
from pathlib import Path
import os

PROJECT_ROOT = Path("~/dft_projects/fe-cr_cu-ni_DFT").expanduser()
FEM_CONICAL  = PROJECT_ROOT / "fem" / "conical"

COMPOSITIONS = {
    "fe16cr00": FEM_CONICAL / "fe16cr00" / "indentation_fe16cr00_v19.pvd",
    "fe08cr08": FEM_CONICAL / "fe08cr08" / "indentation_fe08cr08_v19.pvd",
    "fe04cr12": FEM_CONICAL / "fe04cr12" / "indentation_fe04cr12_v19.pvd",
    "fe00cr16": FEM_CONICAL / "fe00cr16" / "indentation_fe00cr16_v19.pvd",
}

COMPRESSION_LEVEL = 4   # per ccx2paraview docs: keeps size roughly comparable
                        # to the sum of the individual .vtu files

# Safety switch: set True only after you've confirmed the .vtkhdf files open
# correctly in ParaView. Defaults to False so nothing is deleted on first run.
DELETE_VTU_AFTER_CONSOLIDATION = False


def find_vtu_files(pvd_path: Path):
    """The .vtu files live alongside the .pvd, typically named
    <jobname>.<index>.vtu — discover them by globbing rather than assuming
    an exact naming pattern."""
    stem = pvd_path.stem  # e.g. indentation_fe16cr00_v19
    return sorted(pvd_path.parent.glob(f"{stem}*.vtu"))


for label, pvd_path in COMPOSITIONS.items():
    if not pvd_path.exists():
        print(f"[{label}] SKIP — .pvd not found at {pvd_path}")
        continue

    vtkhdf_path = pvd_path.with_suffix(".vtkhdf")
    if vtkhdf_path.exists():
        print(f"[{label}] SKIP — .vtkhdf already exists at {vtkhdf_path}")
        continue

    print(f"[{label}] Reading {pvd_path.name} ...")
    reader = PVDReader(registrationName=label, FileName=str(pvd_path))
    reader.UpdatePipeline()

    n_timesteps = len(list(reader.TimestepValues)) if reader.TimestepValues else 1
    print(f"[{label}]   {n_timesteps} timesteps found, writing single .vtkhdf ...")

    try:
        SaveData(
            str(vtkhdf_path),
            proxy=reader,
            WriteAllTimeSteps=1,
            CompressionLevel=COMPRESSION_LEVEL,
        )
    except Exception as exc:
        print(f"[{label}] ERROR writing .vtkhdf: {exc}")
        Delete(reader)
        del reader
        continue

    if not vtkhdf_path.exists():
        print(f"[{label}] WARNING — SaveData ran without error but "
              f"{vtkhdf_path.name} was not created. Do not delete .vtu files.")
        Delete(reader)
        del reader
        continue

    size_mb = vtkhdf_path.stat().st_size / 1e6
    print(f"[{label}]   DONE -> {vtkhdf_path.name} ({size_mb:.1f} MB)")

    Delete(reader)
    del reader

    # ── cleanup: remove the loose per-increment .vtu files + .pvd ────────
    if DELETE_VTU_AFTER_CONSOLIDATION:
        vtu_files = find_vtu_files(pvd_path)
        for f in vtu_files:
            os.remove(f)
        os.remove(pvd_path)
        print(f"[{label}]   Cleaned up {len(vtu_files)} .vtu files + .pvd")
    else:
        vtu_files = find_vtu_files(pvd_path)
        print(f"[{label}]   {len(vtu_files)} .vtu files + .pvd left in place "
              f"(DELETE_VTU_AFTER_CONSOLIDATION=False — verify the .vtkhdf "
              f"opens correctly in ParaView, then re-run with that flag set "
              f"to True to clean up)")

print("\nDone. Once .vtkhdf files are verified, steps 2 and 3 should load")
print("them via OpenDataFile(str(vtkhdf_path)) instead of the .pvd path.")
