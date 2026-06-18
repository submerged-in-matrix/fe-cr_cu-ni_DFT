#!/usr/bin/env python3
"""
Step 3 v2 — Two-panel field animations and static snapshots for all four
Fe-Cr nanoindentation compositions.

Layout per animation:
  Left panel  — full experiment view (U2 displacement), overview camera.
                Shows the indenter moving down and back up across the full
                loading/unloading cycle.
  Right panel — zoomed contact zone, colored by the specific field for that
                animation. Camera fixed on the contact zone (~8×8 µm window).

Key frame indices are derived directly from the .dat files (RF2 threshold
to detect contact spring activation and deactivation):
  fe16cr00: first_contact=7  peak=69  last_contact=100  total=121
  fe08cr08: first_contact=7  peak=69  last_contact=100  total=123
  fe04cr12: first_contact=6  peak=99  last_contact=122  total=151
  fe00cr16: first_contact=6  peak=67  last_contact=96   total=119

Time-based index matching is used to map .dat times to vtkhdf TimestepValues,
guarding against any off-by-one from ccx2paraview initial-state frames.

Approach:
  1. SaveAnimation (single-view) for overview panel  → _ov_tmp.ogv
  2. SaveAnimation (single-view) for each zoom field → _zm_FIELD_tmp.ogv
  3. ffmpeg hstack the two per-field videos          → FIELD.ogv  (final)
  4. Static two-panel PNGs at 3 key frames via SaveScreenshot + ffmpeg hstack
  5. Remove temp files

Run: pvpython 03_render_outputs_v2.py
Requires: ffmpeg on PATH
"""

import subprocess
from pathlib import Path
from paraview.simple import *

PROJECT_ROOT = Path("~/dft_projects/fe-cr_cu-ni_DFT").expanduser()
FEM_CONICAL  = PROJECT_ROOT / "fem" / "conical"
OUT_DIR      = PROJECT_ROOT / "fem" / "post" / "paraview_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPOSITIONS = {
    # "fe16cr00": FEM_CONICAL / "fe16cr00" / "indentation_fe16cr00_v19.vtkhdf",
    "fe08cr08": FEM_CONICAL / "fe08cr08" / "indentation_fe08cr08_v19.vtkhdf",
    "fe04cr12": FEM_CONICAL / "fe04cr12" / "indentation_fe04cr12_v19.vtkhdf",
    "fe00cr16": FEM_CONICAL / "fe00cr16" / "indentation_fe00cr16_v19.vtkhdf",
}

# .dat-derived key frame times (time value, not index — matched to vtkhdf by proximity)
KEY_FRAME_TIMES = {
    # "fe16cr00": {"first_contact": 0.06958, "peak": 0.40000, "last_contact": 0.46070},
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

ANIM_RES   = [900, 900]
STATIC_RES = [900, 900]
FRAME_RATE = 10

# Overview camera: shows full indenter (r=0-14, y=0-20 µm) + near-surface substrate
OV_CX    = 10.0
OV_CY    =  8.5
OV_SCALE = 15.0

# Zoom camera: contact zone ( ~1.2 µm window centered just below the tip apex)
ZM_CX    = 0.3
ZM_CY    = -0.2
ZM_SCALE = 0.6

def find_ts_index(t_target, timesteps, tol=5e-4):
    diffs = [abs(t - t_target) for t in timesteps]
    idx = diffs.index(min(diffs))
    if min(diffs) > tol:
        print(f"  WARNING: closest timestep to t={t_target:.5f} is "
              f"t={timesteps[idx]:.5f} (diff={min(diffs):.5f}) — exceeds tol")
    return idx


def get_peak_range(reader, t_peak, arr_name, component):
    reader.UpdatePipeline(time=t_peak)
    info = reader.GetDataInformation().GetPointDataInformation().GetArrayInformation(arr_name)
    if info is None:
        return (0.0, 1.0)
    return tuple(info.GetComponentRange(0 if component == -1 else component))


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


def save_animation_single_view(reader, cfg, lo, hi, cx, cy, scale, timesteps,
                                out_path, label_text):
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


def save_screenshot_single_view(reader, cfg, lo, hi, cx, cy, scale, t_snap,
                                 out_path, label_text):
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


def hstack_videos(left_path, right_path, out_path):
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
        print(f"    ffmpeg ERROR:\n{result.stderr[-500:]}")
    elif out_path.exists():
        print(f"    hstacked -> {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def hstack_images(left_path, right_path, out_path):
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
        print(f"    ffmpeg ERROR:\n{result.stderr[-500:]}")
    else:
        print(f"    hstacked -> {out_path.name}")


for comp_label, vtkhdf_path in COMPOSITIONS.items():
    if not vtkhdf_path.exists():
        print(f"[{comp_label}] SKIP — .vtkhdf not found: {vtkhdf_path}")
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
    print(f"[{comp_label}]   {n} timesteps  "
          f"t=[{timesteps[0]:.5f}, {timesteps[-1]:.5f}]")

    # Resolve key frame indices by time proximity
    kft = KEY_FRAME_TIMES[comp_label]
    t_fc   = timesteps[find_ts_index(kft["first_contact"], timesteps)]
    t_peak = timesteps[find_ts_index(kft["peak"],          timesteps)]
    t_lc   = timesteps[find_ts_index(kft["last_contact"],  timesteps)]
    print(f"[{comp_label}]   first_contact t={t_fc:.5f}  "
          f"peak t={t_peak:.5f}  last_contact t={t_lc:.5f}")

    # Pre-compute colorbar ranges at peak load for all fields
    ranges = {}
    for fname, cfg in FIELDS.items():
        lo, hi = get_peak_range(reader, t_peak, cfg["array"], cfg["component"])
        ranges[fname] = (lo, hi)
        print(f"[{comp_label}]   {fname:8s} peak range [{lo:.4g}, {hi:.4g}]")

    # ── ANIMATIONS ──────────────────────────────────────────────────────────

    # Overview animation (U2, overview camera) — rendered once, reused for all 4 fields
    ov_tmp = tmp_dir / f"{comp_label}_ov_tmp.ogv"
    print(f"\n[{comp_label}] Rendering overview animation (U2, overview cam) ...")
    save_animation_single_view(
        reader, FIELDS["U2"], *ranges["U2"],
        OV_CX, OV_CY, OV_SCALE,
        timesteps, ov_tmp,
        f"{comp_label} — Axial displacement U2 (µm) — overview",
    )

    # Per-field zoom animations, then hstack with overview
    for fname, cfg in FIELDS.items():
        zm_tmp = tmp_dir / f"{comp_label}_zm_{fname}_tmp.ogv"
        out_ogv = comp_out / f"{comp_label}_{fname}.ogv"

        print(f"[{comp_label}] Rendering {fname} zoom animation ...")
        save_animation_single_view(
            reader, cfg, *ranges[fname],
            ZM_CX, ZM_CY, ZM_SCALE,
            timesteps, zm_tmp,
            f"{comp_label} — {cfg['label']} — contact zone",
        )

        print(f"[{comp_label}] Combining {fname} panels ...")
        hstack_videos(ov_tmp, zm_tmp, out_ogv)
        zm_tmp.unlink(missing_ok=True)

    ov_tmp.unlink(missing_ok=True)

    # ── STATIC SNAPSHOTS (two-panel PNGs at 3 key frames) ──────────────────

    print(f"\n[{comp_label}] Rendering static snapshots ...")
    for snap_label, t_snap in [("first_contact", t_fc),
                                ("peak",          t_peak),
                                ("last_contact",  t_lc)]:
        for fname, cfg in FIELDS.items():
            ov_png = tmp_dir / f"{comp_label}_ov_{fname}_{snap_label}.png"
            zm_png = tmp_dir / f"{comp_label}_zm_{fname}_{snap_label}.png"
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
            hstack_images(ov_png, zm_png, out_png)
            ov_png.unlink(missing_ok=True)
            zm_png.unlink(missing_ok=True)

    # Clean up tmp dir if empty
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    Delete(reader)
    del reader
    print(f"\n[{comp_label}] Complete. Outputs in {comp_out}")

print("\nAll compositions processed. Outputs in:", OUT_DIR)