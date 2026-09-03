import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "dft_magnetization.csv"
RESULTS_PNG = Path(__file__).resolve().parent.parent / "results" / "dft_magnetization.png"

# fe02cr14 diagnostic runs — separate folder, separate naming convention,
# from fe02cr14_convergence_study.ipynb. Not part of the standard per-tag
# vcr.out pipeline; kept as an explicit special case since it directly
# demonstrates magnetic frustration at this composition.
FE02CR14_DIAG_DIR = Path(__file__).resolve().parent.parent.parent / "fe02cr14"
FE02CR14_RUNS = {
    "h+0.01 FM-plain":       ("fecr_fe02cr14_h0.01.out", True),
    "h-0.01 FM-plain":       ("fecr_fe02cr14_h-0.01.out", True),
    "t+0.01 FM-plain":       ("fecr_fe02cr14_t0.01.out", False),
    "h0.00 pinned-localTF":  ("fecr_fe02cr14_h0.00.out", False),
    "t0.00 pinned-localTF":  ("fecr_fe02cr14_t0.00.out", False),
    "t-0.01 pinned-localTF": ("fecr_fe02cr14_t-0.01.out", False),
}

N_ATOMS = 16

# From prior session notes — verify against the repo README before trusting blindly.
KNOWN_UNCONVERGED_SANITY_SCF = {"fe09cr07", "fe02cr14"}

TOTAL_MAG_RE = re.compile(r"total magnetization\s*=\s*(-?\d+\.?\d*)")
ABS_MAG_RE = re.compile(r"absolute magnetization\s*=\s*(-?\d+\.?\d*)")

TAG_RE = re.compile(r"fe(\d{2})cr(\d{2})")


def parse_tag(tag: str):
    m = TAG_RE.match(tag)
    if not m:
        raise ValueError(f"Could not parse tag: {tag}")
    n_fe, n_cr = int(m.group(1)), int(m.group(2))
    return n_cr, n_cr / N_ATOMS


def extract_magnetization(vcr_path: Path):
    text = vcr_path.read_text(errors="ignore")
    total_matches = TOTAL_MAG_RE.findall(text)
    abs_matches = ABS_MAG_RE.findall(text)

    if not total_matches or not abs_matches:
        return None, None, 0

    return float(total_matches[-1]), float(abs_matches[-1]), len(total_matches)


def extract_fe02cr14_diagnostics():
    """
    Parse the six one-off fe02cr14 diagnostic runs (from
    fe02cr14_convergence_study.ipynb) to show the full spread of absolute
    magnetization each run explored, not just a single final value —
    this is what actually demonstrates magnetic frustration at this
    composition, since the failed runs never settled to one value.
    """
    rows = []
    for label, (fname, expected_converged) in FE02CR14_RUNS.items():
        fpath = FE02CR14_DIAG_DIR / fname
        if not fpath.exists():
            print(f"  fe02cr14 diagnostic missing: {fpath}")
            continue
        text = fpath.read_text(errors="ignore")
        abs_vals = [float(x) for x in ABS_MAG_RE.findall(text)]
        conv_achieved = bool(re.search(r"convergence has been achieved", text))
        if not abs_vals:
            continue
        rows.append({
            "run_label": label,
            "converged": conv_achieved,
            "abs_mag_min": min(abs_vals) / N_ATOMS,
            "abs_mag_max": max(abs_vals) / N_ATOMS,
            "abs_mag_final": abs_vals[-1] / N_ATOMS,
            "n_iterations": len(abs_vals),
        })
    return pd.DataFrame(rows)


def plot_fe02cr14_detail(ax2, fe02cr14_diag):
    """Dedicated panel for fe02cr14's six diagnostic runs — kept separate from
    the main composition plot so its range doesn't distort that plot's scale."""
    order = list(FE02CR14_RUNS.keys())
    ordered = fe02cr14_diag.set_index("run_label").reindex(
        [r for r in order if r in fe02cr14_diag["run_label"].values]
    ).reset_index()

    for i, (_, row) in enumerate(ordered.iterrows()):
        color = "darkgreen" if row["converged"] else "darkorange"
        if row["converged"]:
            ax2.plot(row["abs_mag_final"], i, "D", color=color, markersize=9, zorder=6)
        else:
            ax2.plot([row["abs_mag_min"], row["abs_mag_max"]], [i, i],
                      "-", color=color, linewidth=3, alpha=0.8, zorder=5)
            ax2.plot(row["abs_mag_final"], i, "x", color=color, markersize=8,
                      markeredgewidth=2, zorder=6)

    ax2.set_yticks(range(len(ordered)))
    ax2.set_yticklabels(ordered["run_label"], fontsize=8)
    ax2.set_xlabel("Absolute magnetization (Bohr mag / atom)")
    ax2.set_title("fe02cr14 diagnostic runs\n(87.5% Cr)", fontsize=10)
    ax2.grid(alpha=0.3, axis="x")

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="D", color="darkgreen", linestyle="", markersize=8,
               label="Converged (final value)"),
        Line2D([0], [0], color="darkorange", linewidth=3, label="Failed (oscillation range)"),
    ]
    ax2.legend(handles=legend_elems, fontsize=7.5, loc="lower right")


def run():
    rows = []
    vcr_files = sorted(OUTPUTS_DIR.glob("fecr_*_vcr.out"))

    if not vcr_files:
        print(f"No fecr_*_vcr.out files found in {OUTPUTS_DIR} — check the path.")
        return

    for vcr_path in vcr_files:
        m = re.search(r"fecr_(fe\d{2}cr\d{2})_vcr\.out", vcr_path.name)
        if not m:
            print(f"Skipping unrecognised filename: {vcr_path.name}")
            continue
        tag = m.group(1)
        n_cr, x_cr = parse_tag(tag)

        total_mag, abs_mag, n_scf_steps = extract_magnetization(vcr_path)
        if total_mag is None:
            print(f"WARNING: no magnetization lines found in {vcr_path.name}")
            continue

        rows.append({
            "tag": tag,
            "n_cr": n_cr,
            "x_cr": x_cr,
            "total_mag_per_cell": total_mag,
            "abs_mag_per_cell": abs_mag,
            "total_mag_per_atom": total_mag / N_ATOMS,
            "abs_mag_per_atom": abs_mag / N_ATOMS,
            "n_scf_blocks_found": n_scf_steps,
            "sanity_scf_flagged_unconverged": tag in KNOWN_UNCONVERGED_SANITY_SCF,
        })

    df = pd.DataFrame(rows).sort_values("x_cr").reset_index(drop=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Wrote {len(df)} rows to {RESULTS_CSV}")
    print(df[["tag", "x_cr", "total_mag_per_atom", "abs_mag_per_atom", "sanity_scf_flagged_unconverged"]]
          .to_string(index=False))

    print("\nfe02cr14 diagnostic runs:")
    fe02cr14_diag = extract_fe02cr14_diagnostics()
    if not fe02cr14_diag.empty:
        print(fe02cr14_diag.to_string(index=False))
        fe02cr14_diag.to_csv(
            RESULTS_CSV.parent / "fe02cr14_diagnostic_magnetization.csv", index=False
        )
    else:
        print(f"  None found in {FE02CR14_DIAG_DIR} — skipping that overlay.")

    plot_magnetization(df, fe02cr14_diag)


def plot_magnetization(df, fe02cr14_diag=None):
    has_diag = fe02cr14_diag is not None and not fe02cr14_diag.empty

    if has_diag:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    else:
        fig, ax = plt.subplots(figsize=(9, 6))

    converged = df[~df["sanity_scf_flagged_unconverged"]]
    unconverged = df[df["sanity_scf_flagged_unconverged"]]

    ax.plot(converged["x_cr"] * 100, converged["total_mag_per_atom"],
            "o-", color="tab:red", label="Total magnetization (converged)")
    ax.plot(converged["x_cr"] * 100, converged["abs_mag_per_atom"],
            "s-", color="tab:blue", label="Absolute magnetization (converged)")

    if not unconverged.empty:
        ax.plot(unconverged["x_cr"] * 100, unconverged["total_mag_per_atom"],
                "o", color="tab:red", markerfacecolor="none", markeredgewidth=2,
                markersize=10, label="Total mag. (sanity SCF unconverged)")
        ax.plot(unconverged["x_cr"] * 100, unconverged["abs_mag_per_atom"],
                "s", color="tab:blue", markerfacecolor="none", markeredgewidth=2,
                markersize=10, label="Absolute mag. (sanity SCF unconverged)")

    ax.axvline(90.0, color="gray", linestyle="--", linewidth=1,
               label="CALPHAD FM/AFM boundary (89.99%)")
    ax.axvspan(85, 90, color="gold", alpha=0.15, label="Literature spin-glass window (85-90%)")

    ax.set_xlabel("at.% Cr")
    ax.set_ylabel("Magnetization (Bohr mag / atom)")
    ax.set_title("DFT magnetization vs composition\n(final vc-relax sanity SCF)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    if has_diag:
        plot_fe02cr14_detail(ax2, fe02cr14_diag)

    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved {RESULTS_PNG}")


if __name__ == "__main__":
    run()