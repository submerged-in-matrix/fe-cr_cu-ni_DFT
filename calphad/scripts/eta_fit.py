from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

INPUT_CSV = Path(__file__).resolve().parent.parent / "results" / "lattice_parameter.csv"
OUTPUT_PNG = Path(__file__).resolve().parent.parent / "results" / "eta_fit.png"
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "results" / "eta_x.csv"

N_FINE = 200
SPLINE_SMOOTHING_FACTOR = 5e-6  # tuned to follow the real trend without fitting point-to-point noise


def fit_lattice_parameter():
    df = pd.read_csv(INPUT_CSV)
    x = df["x_cr"].values
    a = df["a_angstrom"].values

    spline = UnivariateSpline(x, a, k=4, s=SPLINE_SMOOTHING_FACTOR * len(x))

    x_fine = np.linspace(x.min(), x.max(), N_FINE)
    a_fine = spline(x_fine)
    da_dx_fine = spline.derivative()(x_fine)
    eta_fine = da_dx_fine / a_fine

    return df, x_fine, a_fine, da_dx_fine, eta_fine


def plot_and_save():
    df, x_fine, a_fine, da_dx_fine, eta_fine = fit_lattice_parameter()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(df["x_cr"] * 100, df["a_angstrom"], "o", color="black", label="DFT (vc-relax)")
    axes[0].plot(x_fine * 100, a_fine, "-", color="tab:blue", label="Spline fit")
    axes[0].set_xlabel("at.% Cr")
    axes[0].set_ylabel(r"Lattice parameter $a$ ($\mathrm{\AA}$)")
    axes[0].set_title("a(x): DFT vs smooth fit")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_fine * 100, da_dx_fine, "-", color="tab:green")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("at.% Cr")
    axes[1].set_ylabel(r"$da/dx$ ($\mathrm{\AA}$)")
    axes[1].set_title("da/dx")
    axes[1].grid(alpha=0.3)

    axes[2].plot(x_fine * 100, eta_fine, "-", color="tab:red")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_xlabel("at.% Cr")
    axes[2].set_ylabel(r"$\eta = (1/a)(da/dx)$")
    axes[2].set_title(r"Compositional expansion coefficient $\eta(x)$")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")

    out_df = pd.DataFrame({"x_cr": x_fine, "a_angstrom": a_fine, "da_dx": da_dx_fine, "eta": eta_fine})
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")

    print(f"\neta range: {eta_fine.min():.4f} to {eta_fine.max():.4f}")
    print(f"eta at x=0.5: {eta_fine[np.argmin(np.abs(x_fine - 0.5))]:.4f}")


if __name__ == "__main__":
    plot_and_save()