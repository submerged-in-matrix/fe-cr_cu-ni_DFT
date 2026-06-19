#!/usr/bin/env python3
"""
Step 3 — Two-panel field animations and static snapshots for all four
Fe-Cr nanoindentation compositions.

Layout per animation:
  Left panel  — full experiment view (U2 displacement), overview camera.
  Right panel — zoomed contact zone (~1.2 µm window), specific field.

Colorbar caps (overrides automatic scan for Mises and SYY):
  Mises: 0 → 15,000 MPa  shared across all compositions
         (~1-2× mean contact pressure; reveals physical stress bulb,
          clips singularity nodes at contact tip)
  SYY:   −15,000 → +15,000 MPa  symmetric, shared
  EYY:   automatic per composition (no colorbar burn; mesh-structural artifacts
          cannot be fixed by clipping — would need mesh refinement)
  U2:    automatic per composition (clean, no artifacts)

Key frame times derived from .dat RF2 threshold (|RF2| > 0.5 µN):
  fe16cr00: first_contact=0.06958  peak=0.40000  last_contact=0.46070
  fe08cr08: first_contact=0.06958  peak=0.40000  last_contact=0.45729
  fe04cr12: first_contact=0.05958  peak=0.40000  last_contact=0.44470
  fe00cr16: first_contact=0.05958  peak=0.40000  last_contact=0.45670

Contact geometry (verified):
  half-angle from axis = 70.3°  →  a = h × tan(70.3°) = 0.404 × 2.793 = 1.128 µm
  Mean contact pressure ~7,700–13,000 MPa across compositions (elastic-only FEM)

Run:  pvpython 03_render_outputs.py
Requires: ffmpeg on PATH
"""

import subprocess
from pathlib import Path
from paraview.simple import *

PROJECT_ROOT = Path("~/dft_projects/fe-cr_cu-ni_DFT").expanduser()
FEM_CONICAL  = PROJECT_ROOT / "fem" / "conical"
OUT_DIR = PROJECT_ROOT / "fem" / "post" / "paraview_figures_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPOSITIONS = {
    "fe16cr00": FEM_CONICAL / "fe16cr00" / "indentation_fe16cr00_v19.vtkhdf",
    "fe08cr08": FEM_CONICAL / "fe08cr08" / "indentation_fe08cr08_v19.vtkhdf",
    "fe04cr12": FEM_CONICAL / "fe04cr12" / "indentation_fe04cr12_v19.vtkhdf",
    "fe00cr16": FEM_CONICAL / "fe00cr16" / "indentation_fe00cr16_v19.vtkhdf",
}

KEY_FRAME_TIMES = {
    "fe16cr00": {"first_contact": 0.06958, "peak": 0.40000, "last_contact": 0.46070},
    "fe08cr08": {"first_contact": 0.06958, "peak": 0.40000, "last_contact": 0.45729},
    "fe04cr12": {"first_contact": 0.05958, "peak": 0.40000, "last_contact": 0.44470},
    "fe00cr16": {"first_contact": 0.05958, "peak": 0.40000, "last_contact": 0.45670},
}

FIELDS = {
    "U2":    {"array": "U",       "component": 1,  "label": "Axial displacement U2 (µm)"},
    "Mises": {"array": "S_Mises", "component": -1, "label": "von Mises stress (MPa)"},
    "SYY":   {"array": "S",       "component": 1,  "label": "Axial stress SYY (MPa)"},
    "EYY":   {"array": "E",       "component": 1,  "label": "Axial strain EYY (-)"},
}

# Manual colorbar caps for fields where automatic scan produces singularity-dominated ranges.
# Mises and SYY: shared range across all compositions for direct comparability.
# U2 and EYY: None → automatic scan from vtkhdf at peak load.
FIELD_RANGE_OVERRIDES = {
    "Mises": (0.0,      15000.0),
    "SYY":   (-15000.0, 15000.0),
    "U2":    None,
    "EYY":   None,
}

ANIM_RES   = [900, 900]
STATIC_RES = [900, 900]
FRAME_RATE = 10

# Overview camera: full indenter (r=0–14, y=0–20 µm) + near-surface substrate
OV_CX    = 10.0
OV_CY    =  8.5
OV_SCALE = 15.0

# Zoom camera: contact zone, ~1.2 µm window centred on tip/surface interface
# Contact radius at peak a = 1.128 µm → scale=0.6 gives ±0.6 µm around centre
ZM_CX    = 0.3
ZM_CY    = -0.2
ZM_SCALE =  0.6


def find_ts_index(t_target, timesteps, tol=5e-4):
    diffs = [abs(t - t_target) for t in timesteps]
    idx = diffs.index(min(diffs))
    if min(diffs) > tol:
        print(f"  WARNING: no timestep within tol of t={t_target:.5f}, "
              f"closest diff={min(diffs):.5f}")
    return idx


def get_range(reader, t_peak, fname, cfg):
    override = FIELD_RANGE_OVERRIDES.get(fname)
    if override is not None:
        return override
    reader.UpdatePipeline(time=t_peak)
    arr  = cfg["array"]
    comp = cfg["component"]
    info = (reader.GetDataInformation()
                  .GetPointDataInformation()
                  .GetArrayInformation(arr))
    if info is None:
        return (0.0, 1.0)
    return tuple(info.GetComponentRange(0 if comp == -1 else comp))


def make_view(size):
    v = CreateRenderView()
    v.ViewSize                  = size
    v.OrientationAxesVisibility = 0
    v.Background                = [0.0, 0.0, 0.0]
    return v


def set_camera(view, cx, cy, scale):
    view.CameraPosition           = [cx, cy, 1500.0]
    view.CameraFocalPoint         = [cx, cy,    0.0]
    view.CameraViewUp             = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = 1
    view.CameraParallelScale      = scale


def apply_field(reader, cfg, view, lo, hi):
    arr  = cfg["array"]
    comp = cfg["component"]
    disp = Show(reader, view)
    disp.Representation = "Surface With Edges"
    disp.EdgeColor      = [0.15, 0.15, 0.15]
    if comp == -1:
        ColorBy(disp, ("POINTS", arr))
    else:
        ColorBy(disp, ("POINTS", arr, comp))
    lut = GetColorTransferFunction(arr)
    lut.RescaleTransferFunction(lo, hi)
    sb = GetScalarBar(lut, view)
    sb.Visibility      = 1
    sb.Title           = cfg["label"]
    sb.ComponentTitle  = ""
    sb.TitleFontSize   = 12
    sb.LabelFontSize   = 10
    sb.Position        = [0.85, 0.05]
    sb.ScalarBarLength = 0.80
    return disp


def add_title(text_str, view):
    t      = Text()
    t.Text = text_str
    td = Show(t, view)
    td.FontSize       = 11
    td.Bold           = 1
    td.WindowLocation = "Upper Center"
    td.Color          = [1.0, 1.0, 1.0]
    return t


def save_animation_single_view(reader, cfg, lo, hi, cx, cy, scale,
                                timesteps, out_path, label_text):
    view = make_view(ANIM_RES)
    apply_field(reader, cfg, view, lo, hi)
    title_obj = add_title(label_text, view)
    reader.UpdatePipeline(time=timesteps[0])
    view.ViewTime = timesteps[0]
    Render(view)
    set_camera(view, cx, cy, scale)
    Render(view)
    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    scene.PlayMode  = "Snap To TimeSteps"
    scene.StartTime = timesteps[0]
    scene.EndTime   = timesteps[-1]
    n = len(timesteps)
    SaveAnimation(str(out_path), view, FrameRate=FRAME_RATE, FrameWindow=[0, n - 1])
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"    saved {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"    WARNING — {out_path.name} empty or missing")
    Delete(title_obj)
    Delete(view)


def save_screenshot_single_view(reader, cfg, lo, hi, cx, cy, scale,
                                 t_snap, out_path, label_text):
    view = make_view(STATIC_RES)
    apply_field(reader, cfg, view, lo, hi)
    title_obj = add_title(label_text, view)
    reader.UpdatePipeline(time=t_snap)
    view.ViewTime = t_snap
    Render(view)
    set_camera(view, cx, cy, scale)
    Render(view)
    SaveScreenshot(str(out_path), view, ImageResolution=STATIC_RES)
    print(f"    saved {out_path.name}")
    Delete(title_obj)
    Delete(view)


def hstack(left_path, right_path, out_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(left_path),
        "-i", str(right_path),
        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
        "-map", "[v]",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ffmpeg ERROR:\n{result.stderr[-400:]}")
    elif out_path.exists():
        print(f"    -> {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)")


for comp_label, vtkhdf_path in COMPOSITIONS.items():
    if not vtkhdf_path.exists():
        print(f"[{comp_label}] SKIP — not found: {vtkhdf_path}")
        continue

    comp_out = OUT_DIR / comp_label
    comp_out.mkdir(exist_ok=True)
    tmp_dir = comp_out / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"\n[{comp_label}] Loading {vtkhdf_path.name} ...")
    reader = OpenDataFile(str(vtkhdf_path))
    reader.UpdatePipeline()
    timesteps = list(reader.TimestepValues)
    n = len(timesteps)
    print(f"[{comp_label}]   {n} timesteps")

    kft    = KEY_FRAME_TIMES[comp_label]
    t_fc   = timesteps[find_ts_index(kft["first_contact"], timesteps)]
    t_peak = timesteps[find_ts_index(kft["peak"],          timesteps)]
    t_lc   = timesteps[find_ts_index(kft["last_contact"],  timesteps)]
    print(f"[{comp_label}]   fc={t_fc:.5f}  peak={t_peak:.5f}  lc={t_lc:.5f}")

    ranges = {}
    for fname, cfg in FIELDS.items():
        lo, hi = get_range(reader, t_peak, fname, cfg)
        ranges[fname] = (lo, hi)
        src = "OVERRIDE" if FIELD_RANGE_OVERRIDES.get(fname) else "auto"
        print(f"[{comp_label}]   {fname:8s} [{lo:.4g}, {hi:.4g}]  ({src})")

    # ── ANIMATIONS ──────────────────────────────────────────────────────────
    ov_tmp = tmp_dir / f"{comp_label}_ov_tmp.ogv"
    print(f"\n[{comp_label}] Overview animation (U2) ...")
    save_animation_single_view(
        reader, FIELDS["U2"], *ranges["U2"],
        OV_CX, OV_CY, OV_SCALE, timesteps, ov_tmp,
        f"{comp_label} — U2 overview",
    )

    for fname, cfg in FIELDS.items():
        zm_tmp  = tmp_dir / f"{comp_label}_zm_{fname}_tmp.ogv"
        out_ogv = comp_out / f"{comp_label}_{fname}.ogv"
        print(f"[{comp_label}] {fname} zoom animation ...")
        save_animation_single_view(
            reader, cfg, *ranges[fname],
            ZM_CX, ZM_CY, ZM_SCALE, timesteps, zm_tmp,
            f"{comp_label} — {cfg['label']} — contact zone",
        )
        print(f"[{comp_label}] Combining {fname} ...")
        hstack(ov_tmp, zm_tmp, out_ogv)
        zm_tmp.unlink(missing_ok=True)

    ov_tmp.unlink(missing_ok=True)

    # ── STATIC SNAPSHOTS ────────────────────────────────────────────────────
    print(f"\n[{comp_label}] Static snapshots ...")
    for snap_label, t_snap in [("first_contact", t_fc),
                                ("peak",          t_peak),
                                ("last_contact",  t_lc)]:
        for fname, cfg in FIELDS.items():
            ov_png  = tmp_dir / f"ov_{fname}_{snap_label}.png"
            zm_png  = tmp_dir / f"zm_{fname}_{snap_label}.png"
            out_png = comp_out / f"{comp_label}_{fname}_{snap_label}.png"
            save_screenshot_single_view(
                reader, FIELDS["U2"], *ranges["U2"],
                OV_CX, OV_CY, OV_SCALE,
                t_snap, ov_png,
                f"{comp_label} — U2 overview — {snap_label}",
            )
            save_screenshot_single_view(
                reader, cfg, *ranges[fname],
                ZM_CX, ZM_CY, ZM_SCALE,
                t_snap, zm_png,
                f"{comp_label} — {cfg['label']} — {snap_label}",
            )
            hstack(ov_png, zm_png, out_png)
            ov_png.unlink(missing_ok=True)
            zm_png.unlink(missing_ok=True)

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    Delete(reader)
    del reader
    print(f"[{comp_label}] Complete.")

print("\nAll compositions done. Outputs in:", OUT_DIR)
