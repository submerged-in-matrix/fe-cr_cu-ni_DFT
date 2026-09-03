from pathlib import Path

import pandas as pd

INPUT_CSV = Path(__file__).resolve().parent.parent.parent / "analysis" / "calphad_homogenized.csv"
OUTPUT_INP = Path(__file__).resolve().parent.parent.parent / "analysis" / "calphad_material_cards.inp"

# GPa -> MPa, matching the unit convention already used in fem/conical/*
GPA_TO_MPA = 1000.0


def material_name(tag: str, T_K: float) -> str:
    t_int = int(round(T_K))
    return f"{tag}_{t_int}K"


def write_material_block(f, name: str, E_MPa: float, nu: float):
    f.write(f"*MATERIAL, NAME={name}\n")
    f.write("*ELASTIC, TYPE=ISO\n")
    f.write(f"{E_MPa:.1f}, {nu:.5f}\n")


def generate_cards():
    df = pd.read_csv(INPUT_CSV)

    with open(OUTPUT_INP, "w") as f:
        f.write("** CalculiX isotropic material cards\n")
        f.write("** Generated from CALPHAD equilibrium + ML surrogate + VRH/HS homogenisation\n")
        f.write("** E in MPa, nu dimensionless. VRH used as primary (consistent with\n")
        f.write("** previously validated single-phase FEM cards); HS-average provided\n")
        f.write("** as an alternate block, commented out, for comparison runs.\n\n")

        for _, r in df.iterrows():
            name = material_name(r["tag"], r["T_K"])
            E_vrh_MPa = r["E_VRH"] * GPA_TO_MPA
            E_hs_MPa = r["E_HS_avg"] * GPA_TO_MPA

            f.write(f"** {r['tag']} at {r['T_K']:.2f} K — phases: {r['phases']} (n={r['n_phases']})\n")
            write_material_block(f, name, E_vrh_MPa, r["nu_VRH"])
            f.write(f"** Alternate (Hashin-Shtrikman average): E={E_hs_MPa:.1f} MPa, nu={r['nu_HS_avg']:.5f}\n")
            f.write("\n")

    print(f"Wrote {len(df)} material cards to {OUTPUT_INP}")


if __name__ == "__main__":
    generate_cards()