from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV = Path(__file__).resolve().parent.parent / "results" / "equilibrium_sweep.csv"
OUTPUT_PNG = Path(__file__).resolve().parent.parent / "results" / "sigma_fraction_map.png"


def build_sigma_grid():
    df = pd.read_csv(INPUT_CSV)
    sigma = df[df["phase"] == "SIGMA"][["tag", "x_cr_nominal", "T_K", "phase_fraction"]].copy()

    x_cr_values = sorted(df["x_cr_nominal"].unique())
    T_values = sorted(df["T_K"].unique())

    grid = np.zeros((len(T_values), len(x_cr_values)))
    x_idx = {x: i for i, x in enumerate(x_cr_values)}
    t_idx = {t: i for i, t in enumerate(T_values)}

    for _, row in sigma.iterrows():
        grid[t_idx[row["T_K"]], x_idx[row["x_cr_nominal"]]] = row["phase_fraction"]

    return np.array(x_cr_values), np.array(T_values), grid


def plot_sigma_map():
    x_cr, T, grid = build_sigma_grid()

    fig, ax = plt.subplots(figsize=(9, 7))

    levels = np.linspace(0, grid.max(), 21)
    cf = ax.contourf(x_cr * 100, T, grid, levels=levels, cmap="Reds")
    ax.contour(x_cr * 100, T, grid, levels=[0.01], colors="black", linewidths=0.8)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Sigma phase fraction (mole fraction of alloy)")

    max_frac_idx = np.unravel_index(np.argmax(grid), grid.shape)
    max_x = x_cr[max_frac_idx[1]] * 100
    max_T = T[max_frac_idx[0]]
    max_val = grid[max_frac_idx]
    ax.plot(max_x, max_T, "k*", markersize=15)
    ax.annotate(f"Max: {max_val:.3f} at {max_x:.1f}% Cr, {max_T:.0f} K",
                (max_x, max_T), textcoords="offset points", xytext=(10, 10), fontsize=9)

    dft_grid_x = [100 * i / 16 for i in range(17)]
    for x in dft_grid_x:
        ax.axvline(x, color="black", linewidth=0.3, alpha=0.25)

    ax.set_xlabel("at.% Cr (nominal alloy composition)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Sigma phase fraction across composition and temperature\n"
                 "(embrittlement risk map — 0 = no sigma present)")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")

    nonzero = grid[grid > 0]
    print(f"\nSigma present in {len(nonzero)} of {grid.size} (composition, T) grid points")
    print(f"Max sigma fraction: {max_val:.3f} at {max_x:.1f}% Cr, {max_T:.0f} K")


if __name__ == "__main__":
    plot_sigma_map()