from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy
from pycalphad import Database, Model, variables as v

TDB_PATH = Path(__file__).resolve().parent.parent / "tdb" / "CrFeNb_Jacob2016.tdb"
OUTPUT_PNG = Path(__file__).resolve().parent.parent / "results" / "curie_neel_temperature.png"
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "results" / "curie_neel_temperature.csv"

N_POINTS = 201


def compute_tc_curve():
    dbf = Database(str(TDB_PATH))
    mod = Model(dbf, ["CR", "FE", "VA"], "BCC_A2")

    tc_expr = mod.curie_temperature
    # The raw (pre-sign-correction) weighted sum determines the FM/AFM boundary:
    # tc_expr = raw_expr / Piecewise((-1, raw_expr <= 0), (1, True))
    # raw_expr > 0 -> ferromagnetic; raw_expr <= 0 -> antiferromagnetic.
    raw_expr = tc_expr.args[0]

    y_cr = v.Y("BCC_A2", 0, "CR")
    y_fe = v.Y("BCC_A2", 0, "FE")
    y_va = v.Y("BCC_A2", 1, "VA")

    x_cr_values = np.linspace(1e-4, 1 - 1e-4, N_POINTS)
    tc_values = np.zeros(N_POINTS)
    raw_values = np.zeros(N_POINTS)

    for i, x_cr in enumerate(x_cr_values):
        subs = {y_cr: x_cr, y_fe: 1.0 - x_cr, y_va: 1.0}
        tc_values[i] = float(tc_expr.subs(subs))
        raw_values[i] = float(raw_expr.subs(subs))

    return x_cr_values, tc_values, raw_values


def plot_tc_curve(x_cr, tc, raw):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x_cr * 100, tc, color="tab:purple", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5)

    zero_crossings = np.where(np.diff(np.sign(raw)))[0]
    fm_afm_boundary_idx = zero_crossings[0] if len(zero_crossings) > 0 else len(x_cr)

    ax.fill_between(x_cr[:fm_afm_boundary_idx] * 100, 0, tc[:fm_afm_boundary_idx],
                     color="tab:red", alpha=0.15, label="Ferromagnetic")
    ax.fill_between(x_cr[fm_afm_boundary_idx:] * 100, 0, tc[fm_afm_boundary_idx:],
                     color="tab:blue", alpha=0.15, label="Antiferromagnetic")

    for idx in zero_crossings:
        x_cross = x_cr[idx] * 100
        ax.axvline(x_cross, color="gray", linestyle="--", linewidth=0.8)
        ax.annotate(f"{x_cross:.1f}%", (x_cross, 0), textcoords="offset points",
                    xytext=(5, 10), fontsize=9, color="gray")

    ax.scatter([0, 100], [abs(tc[0]), tc[-1]], color="black", zorder=5)
    ax.annotate(f"Pure Fe: T_C = {tc[0]:.0f} K (ferro)", (0, tc[0]),
                textcoords="offset points", xytext=(10, -15), fontsize=9)
    ax.annotate(f"Pure Cr: T_N = {tc[-1]:.0f} K (antiferro)", (100, tc[-1]),
                textcoords="offset points", xytext=(-120, 10), fontsize=9)

    ax.set_xlabel("at.% Cr")
    ax.set_ylabel("Magnetic ordering temperature (K)")
    ax.set_title("BCC Fe-Cr: composition-dependent Curie/Neel temperature\n(CrFeNb_Jacob2016.tdb)")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")


def run():
    x_cr, tc, raw = compute_tc_curve()

    import pandas as pd
    df = pd.DataFrame({"x_cr": x_cr, "TC_or_TN_K": tc, "raw_weighted_sum": raw})
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")

    plot_tc_curve(x_cr, tc, raw)

    zero_idx = np.where(np.diff(np.sign(raw)))[0]
    for idx in zero_idx:
        print(f"FM/AFM boundary near x_Cr = {x_cr[idx]*100:.2f}%")


if __name__ == "__main__":
    run()