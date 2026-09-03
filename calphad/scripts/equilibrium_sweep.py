import csv
from pathlib import Path

from pycalphad import Database, equilibrium, variables as v

TDB_PATH = Path(__file__).resolve().parent.parent / "tdb" / "CrFeNb_Jacob2016.tdb"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "equilibrium_sweep.csv"

COMPS = ["CR", "FE", "VA"]
PHASES = ["BCC_A2", "FCC_A1", "LIQUID", "SIGMA"]

N_CR_ATOMS = 16
T_MIN = 298.15
T_MAX = 1200
T_STEP = 20
KEY_TRANSITION_TEMPS = [298.15, 311.5, 772, 1093, 1095.5, 1111, 1126, 1185]


def temperature_grid():
    grid = set(KEY_TRANSITION_TEMPS)
    t = T_MIN
    while t <= T_MAX:
        grid.add(round(t, 2))
        t += T_STEP
    return sorted(grid)


TEMPERATURES = temperature_grid()


def cr_fractions():
    return [i / N_CR_ATOMS for i in range(N_CR_ATOMS + 1)]


def run_sweep():
    dbf = Database(str(TDB_PATH))
    x_cr_values = cr_fractions()

    eq = equilibrium(
        dbf,
        COMPS,
        PHASES,
        {v.X("CR"): x_cr_values, v.T: TEMPERATURES, v.P: 101325, v.N: 1},
    )

    rows = []
    for i, x_cr in enumerate(x_cr_values):
        n_cr_atoms = round(x_cr * N_CR_ATOMS)
        for j, temp in enumerate(TEMPERATURES):
            point = eq.isel(X_CR=i, T=j, P=0, N=0)
            phase_names = point.Phase.values.squeeze()
            phase_fracs = point.NP.values.squeeze()
            phase_comps = point.X.values.squeeze()

            phase_names = phase_names.reshape(-1)
            phase_fracs = phase_fracs.reshape(-1)
            phase_comps = phase_comps.reshape(len(phase_names), -1)

            for name, frac, comp in zip(phase_names, phase_fracs, phase_comps):
                if name == "" or name is None:
                    continue
                cr_idx = COMPS.index("CR")
                rows.append(
                    {
                        "tag": f"fe{16 - n_cr_atoms:02d}cr{n_cr_atoms:02d}",
                        "x_cr_nominal": round(x_cr, 6),
                        "T_K": temp,
                        "phase": str(name),
                        "phase_fraction": float(frac),
                        "phase_x_cr": float(comp[cr_idx]),
                    }
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tag", "x_cr_nominal", "T_K", "phase", "phase_fraction", "phase_x_cr"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} phase-equilibrium rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_sweep()