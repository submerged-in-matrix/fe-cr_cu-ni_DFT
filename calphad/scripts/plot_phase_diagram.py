from pathlib import Path

import matplotlib.pyplot as plt
from pycalphad import Database, variables as v
from pycalphad.plot.binary import binplot

TDB_PATH = Path(__file__).resolve().parent.parent / "tdb" / "CrFeNb_Jacob2016.tdb"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "phase_diagram.png"

COMPS = ["CR", "FE", "VA"]
PHASES = ["BCC_A2", "FCC_A1", "LIQUID", "SIGMA"]

T_MIN = 298.15
T_MAX = 1200
T_STEP = 10
X_CR_STEP = 0.01


def plot_phase_diagram():
    dbf = Database(str(TDB_PATH))

    conditions = {
        v.X("CR"): (0, 1, X_CR_STEP),
        v.T: (T_MIN, T_MAX, T_STEP),
        v.P: 101325,
        v.N: 1,
    }

    ax = binplot(dbf, COMPS, PHASES, conditions)
    ax.set_title("Cr-Fe phase diagram (CrFeNb_Jacob2016.tdb, Nb excluded)")
    ax.set_xlabel("X(Cr)")
    ax.set_ylabel("Temperature (K)")

    fig = ax.figure
    fig.set_size_inches(9, 7)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_phase_diagram()