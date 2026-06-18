#!/usr/bin/env python3
"""
Contact pressure vs time/depth — derived from existing .dat files and the
Oliver-Pharr fit already performed in fecr_nanoindentation_analysis_v19_all.ipynb.

No CGX or ParaView needed: *CONTACT FILE was not requested in the v19 .inp
decks, so there is no spatial contact-pressure field in the .frd files.
This computes the SCALAR mean contact pressure p(t) = P(t) / A_c(t), using
the contact area formula already validated in the O-P notebook:

    A_c(h) = pi * (h_c)^2 * tan^2(theta)   where h_c is back-solved per
    increment from the same Oliver-Pharr geometry (NOT the single h_c at
    peak — here we approximate A_c(t) using the loading-curve relation
    A_c(t) ~ pi * (h(t) - h_f)^2 * tan^2(theta) for points after first
    contact, which is consistent with Sneddon conical contact theory).

Run with regular python3 (no ParaView/CGX dependency):
    python3 contact_pressure_plot.py

Requires the .dat files already used in the nanoindentation notebook, plus
fecr_all_oliver_pharr_summary.csv for h_f and theta per composition.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

WEDGE_SCALE = 180.0
THETA_DEG = 70.3
THETA_RAD = np.radians(THETA_DEG)

PROJECT_ROOT = Path("~/dft_projects/fe-cr_cu-ni_DFT").expanduser()
FEM_CONICAL  = PROJECT_ROOT / "fem" / "conical"
POST_DIR     = PROJECT_ROOT / "fem" / "post"

SUMMARY_CSV = POST_DIR / "fecr_all_oliver_pharr_summary.csv"

COMPOSITIONS = [
    dict(tag="fe16cr00", label="Fe16Cr0", color="#1f5fa8"),
    dict(tag="fe08cr08", label="Fe8Cr8",  color="#2ecc71"),
    dict(tag="fe04cr12", label="Fe4Cr12", color="#e07b39"),
    dict(tag="fe00cr16", label="Fe0Cr16", color="#c0392b"),
]


def parse_ccx_dat(fpath):
    content = Path(fpath).read_text()
    step_blocks = re.split(r"S T E P\s+(\d+)", content)
    records = []
    for s in range(1, len(step_blocks), 2):
        step_nr = int(step_blocks[s])
        inc_blocks = re.split(r"INCREMENT\s+(\d+)", step_blocks[s + 1])
        for i in range(1, len(inc_blocks), 2):
            blk = inc_blocks[i + 1]
            tm = re.search(r"displacements.*?\n\s+\d+\s+[\d.E+\-]+\s+([\-\d.E+]+)", blk)
            fm = re.search(r"total force.*?\n\s+([\-\d.E+]+)\s+([\-\d.E+]+)\s+([\-\d.E+]+)", blk)
            if tm and fm:
                records.append((step_nr, abs(float(tm.group(1))), abs(float(fm.group(2)))))
    return records


summary_df = pd.read_csv(SUMMARY_CSV, index_col="Composition")

fig, ax = plt.subplots(figsize=(8, 5))

for comp in COMPOSITIONS:
    tag, label, color = comp["tag"], comp["label"], comp["color"]
    dat_path = FEM_CONICAL / tag / f"indentation_{tag}_v19.dat"

    if not dat_path.exists():
        print(f"[{label}] SKIP — .dat not found")
        continue
    if label not in summary_df.index:
        print(f"[{label}] SKIP — not in O-P summary CSV")
        continue

    row = summary_df.loc[label]
    if row.get("status", "ok") == "missing" or pd.isna(row.get("h_f (µm)", np.nan)):
        print(f"[{label}] SKIP — no valid O-P fit in summary")
        continue

    h_f = float(str(row["h_f (µm)"]).replace(",", ""))

    records = parse_ccx_dat(dat_path)
    n_steps = max(r[0] for r in records)
    loading = [(h, F * WEDGE_SCALE) for (st, h, F) in records if st < n_steps]

    h_arr = np.array([x[0] for x in loading])
    P_arr = np.array([x[1] for x in loading])
    order = np.argsort(h_arr)
    h_arr, P_arr = h_arr[order], P_arr[order]

    # Contact area via Sneddon conical relation, using h_f from the O-P fit
    valid = h_arr > h_f
    h_c_approx = h_arr[valid] - h_f
    A_c = np.pi * (h_c_approx * np.tan(THETA_RAD)) ** 2
    p_mean = P_arr[valid] / A_c  # MPa (uN / um^2)

    ax.plot(h_arr[valid], p_mean / 1000, "-", color=color, lw=1.8, label=label)

ax.set_xlabel("Indentation depth h (µm)")
ax.set_ylabel("Mean contact pressure  P / A$_c$  (GPa)")
ax.set_title(
    "Approximate mean contact pressure vs depth (loading)\n"
    "A$_c$ from Sneddon conical relation using O-P h$_f$ — NOT a spatial field "
    "(no *CONTACT FILE in .inp)"
)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

out_path = POST_DIR / "fecr_contact_pressure_vs_depth.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")
