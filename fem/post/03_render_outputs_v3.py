#!/usr/bin/env python3
"""
Step 3 v3 — Three-panel field animations and static snapshots.
Output path: fem/post/paraview_figures/v3/<comp>/

Panel layout (left → right):
  1. Overview   — full indenter + substrate, U2 field (OV camera)
  2. Near-field — full contact patch + decay into far field (NF camera, scale=1.5)
  3. Close-up   — inside contact zone, fine detail (ZM camera, scale=0.6)

Panels 2 and 3 are colored by the per-animation field; panel 1 is always U2.

Camera rationale:
  Contact radius a = h × tan(70.3°) = 0.404 × 2.793 = 1.128 µm.
  Close-up (scale=0.6) sits INSIDE the contact patch — resolves the 0.4 µm
  displacement detail that is lost when zoomed out.
  Near-field (scale=1.5) captures the full 1.128 µm contact patch plus the
  surrounding stress/strain decay into the substrate.

Colorbar caps derived from material cards (calculix_material_cards_best.inp)
and mean contact pressure p_mean = F×180 / (π×a²):

  Composition  C11(MPa)  K=5×C11    p_mean(MPa)  Mises_hi  SYY_lo/hi
  fe16cr00     307430    1537150     10666        20000     ±15000
  fe08cr08     300500    1502500      7697        15000     ±12000
  fe04cr12     396510    1982550      9688        25000     ±15000
  fe00cr16     438290    2191450     12978        25000     ±20000

  EYY / U2: automatic per composition.
  SYY tensile islands = penalty contact edge singularities (documented in README).

Key frame times from .dat RF2 threshold (|RF2| > 0.5 µN):
  fe16cr00: fc=0.06958  peak=0.40000  lc=0.46070
  fe08cr08: fc=0.06958  peak=0.40000  lc=0.45729
  fe04cr12: fc=0.05958  peak=0.40000  lc=0.44470
  fe00cr16: fc=0.05958  peak=0.40000  lc=0.45670

Run:  pvpython 03_render_outputs.py
Requires: ffmpeg on PATH
"""

import subprocess
from pathlib import Path
from paraview.simple import *

PROJECT_ROOT = Path("~/dft_projects/fe-cr_cu-ni_DFT").expanduser()
FEM_CONICAL  = PROJECT_ROOT / "fem" / "conical"
OUT_DIR      = PROJECT_ROOT / "fem" / "post" / "paraview_figures" / "v3"
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

FIELD_RANGE_OVERRIDES = {
    "fe16cr00": {"Mises": (0.0, 20000.0), "SYY": (-15000.0, 15000.0), "EYY": None, "U2": None},
    "fe08cr08": {"Mises": (0.0, 15000.0), "SYY": (-12000.0, 12000.0), "EYY": None, "U2": None},
    "fe04cr12": {"Mises": (0.0, 25000.0), "SYY": (-15000.0, 15000.0), "EYY": None, "U2": None},
    "fe00cr16": {"Mises": (0.0, 25000.0), "SYY": (-20000.0, 20000.0), "EYY": None, "U2": None},
}

ANIM_RES   = [700, 900]
STATIC_RES = [700, 900]
FRAME_RATE = 10

# Overview camera: full indenter + near-surface substrate
OV_CX, OV_CY, OV_SCALE = 10.0, 8.5, 15.0

# Near-field camera: full contact patch (a=1.128 µm) + decay into far field
NF_CX, NF_CY, NF_SCALE = 1.2, -0.5, 1.5

# Close-up camera: inside the contact zone, fine displacement detail
ZM_CX, ZM_CY, ZM_SCALE = 0.3, -0.2, 0.6


def find_ts_index(t_target, timesteps, tol=5e-4):
    diffs = [abs(t - t_target) for t in timesteps]
    idx = diffs.index(min(diffs))
    if min(diffs) > tol:
        print(f"  WARNING: no timestep within tol of t={t_target:.5f}, "
              f"closest diff={min(diffs):.5f}")
    return idx


def get_range(reader, t_peak, comp_label, fname, cfg):
    override = FIELD_RANGE_OVERRIDES.get(comp_label, {}).get(fname)
    if override is not None:
        return override
    reader.UpdatePipeline(time=t_peak)
    info = (reader.GetDataInformation()
                  .GetPointDataInformation()
                  .GetArrayInformation(cfg["array"]))
    if info is None:
        return (0.0, 1.0)
    comp = cfg["component"]
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
    sb.TitleFontSize   = 11
    sb.LabelFontSize   = 9
    sb.Position        = [0.82, 0.05]
    sb.ScalarBarLength = 0.80
    return disp


def add_title(text_str, view):
    t      = Text()
    t.Text = text_str
    td = Show(t, view)
    td.FontSize       = 10
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


def hstack3(p1, p2, p3, out_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(p1),
        "-i", str(p2),
        "-i", str(p3),
        "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]",
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
        lo, hi = get_range(reader, t_peak, comp_label, fname, cfg)
        ranges[fname] = (lo, hi)
        src = "OVERRIDE" if FIELD_RANGE_OVERRIDES[comp_label].get(fname) else "auto"
        print(f"[{comp_label}]   {fname:8s} [{lo:.4g}, {hi:.4g}]  ({src})")

    # ── ANIMATIONS ──────────────────────────────────────────────────────────
    # Overview (U2) rendered once, reused as panel 1 for every field.
    ov_tmp = tmp_dir / f"{comp_label}_ov_tmp.ogv"
    print(f"\n[{comp_label}] Overview animation (U2) ...")
    save_animation_single_view(
        reader, FIELDS["U2"], *ranges["U2"],
        OV_CX, OV_CY, OV_SCALE, timesteps, ov_tmp,
        f"{comp_label} — U2 overview",
    )

    for fname, cfg in FIELDS.items():
        nf_tmp  = tmp_dir / f"{comp_label}_nf_{fname}_tmp.ogv"
        zm_tmp  = tmp_dir / f"{comp_label}_zm_{fname}_tmp.ogv"
        out_ogv = comp_out / f"{comp_label}_{fname}.ogv"

        print(f"[{comp_label}] {fname} near-field animation ...")
        save_animation_single_view(
            reader, cfg, *ranges[fname],
            NF_CX, NF_CY, NF_SCALE, timesteps, nf_tmp,
            f"{comp_label} — {cfg['label']} — near-field",
        )
        print(f"[{comp_label}] {fname} close-up animation ...")
        save_animation_single_view(
            reader, cfg, *ranges[fname],
            ZM_CX, ZM_CY, ZM_SCALE, timesteps, zm_tmp,
            f"{comp_label} — {cfg['label']} — close-up",
        )
        print(f"[{comp_label}] Combining {fname} (3 panels) ...")
        hstack3(ov_tmp, nf_tmp, zm_tmp, out_ogv)
        nf_tmp.unlink(missing_ok=True)
        zm_tmp.unlink(missing_ok=True)

    ov_tmp.unlink(missing_ok=True)

    # ── STATIC SNAPSHOTS ────────────────────────────────────────────────────
    print(f"\n[{comp_label}] Static snapshots ...")
    for snap_label, t_snap in [("first_contact", t_fc),
                                ("peak",          t_peak),
                                ("last_contact",  t_lc)]:
        for fname, cfg in FIELDS.items():
            ov_png  = tmp_dir / f"ov_{fname}_{snap_label}.png"
            nf_png  = tmp_dir / f"nf_{fname}_{snap_label}.png"
            zm_png  = tmp_dir / f"zm_{fname}_{snap_label}.png"
            out_png = comp_out / f"{comp_label}_{fname}_{snap_label}.png"

            save_screenshot_single_view(
                reader, FIELDS["U2"], *ranges["U2"],
                OV_CX, OV_CY, OV_SCALE, t_snap, ov_png,
                f"{comp_label} — U2 overview — {snap_label}",
            )
            save_screenshot_single_view(
                reader, cfg, *ranges[fname],
                NF_CX, NF_CY, NF_SCALE, t_snap, nf_png,
                f"{comp_label} — {cfg['label']} — near-field — {snap_label}",
            )
            save_screenshot_single_view(
                reader, cfg, *ranges[fname],
                ZM_CX, ZM_CY, ZM_SCALE, t_snap, zm_png,
                f"{comp_label} — {cfg['label']} — close-up — {snap_label}",
            )
            hstack3(ov_png, nf_png, zm_png, out_png)
            ov_png.unlink(missing_ok=True)
            nf_png.unlink(missing_ok=True)
            zm_png.unlink(missing_ok=True)

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    Delete(reader)
    del reader
    print(f"[{comp_label}] Complete.")

print("\nAll compositions done. Outputs in:", OUT_DIR)