from pathlib import Path

import numpy as np
import pandas as pd

INPUT_CSV = Path(__file__).resolve().parent.parent.parent / "analysis" / "calphad_equilibrium_with_Cij.csv"
OUTPUT_CSV = Path(__file__).resolve().parent.parent.parent / "analysis" / "calphad_homogenized.csv"


def cubic_to_isotropic_hill(C11, C12, C44):
    """
    Voigt-Reuss-Hill averaging of single-crystal cubic elastic constants
    to isotropic polycrystalline bulk modulus K and shear modulus G.
    K is identical for Voigt and Reuss in cubic symmetry.
    """
    K = (C11 + 2 * C12) / 3.0
    G_voigt = (C11 - C12 + 3 * C44) / 5.0
    G_reuss = 5.0 * (C11 - C12) * C44 / (4.0 * C44 + 3.0 * (C11 - C12))
    G_hill = 0.5 * (G_voigt + G_reuss)
    return K, G_hill


def KG_to_E_nu(K, G):
    E = 9.0 * K * G / (3.0 * K + G)
    nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    return E, nu


def vrh_mix(f1, K1, G1, f2, K2, G2):
    """Voigt-Reuss-Hill mixing of two isotropic phases by volume fraction."""
    K_voigt = f1 * K1 + f2 * K2
    G_voigt = f1 * G1 + f2 * G2
    K_reuss = 1.0 / (f1 / K1 + f2 / K2)
    G_reuss = 1.0 / (f1 / G1 + f2 / G2)
    K_hill = 0.5 * (K_voigt + K_reuss)
    G_hill = 0.5 * (G_voigt + G_reuss)
    return dict(
        K_voigt=K_voigt, G_voigt=G_voigt,
        K_reuss=K_reuss, G_reuss=G_reuss,
        K_hill=K_hill, G_hill=G_hill,
    )


def hs_bound(f_ref, K_ref, G_ref, f_other, K_other, G_other):
    """
    One Hashin-Shtrikman bound, using phase 'ref' as the reference/matrix.
    Passing the stiffer phase as ref gives the upper bound; the softer
    phase as ref gives the lower bound.
    """
    K_hs = K_ref + f_other / (
        1.0 / (K_other - K_ref) + f_ref / (K_ref + 4.0 * G_ref / 3.0)
    )
    G_hs = G_ref + f_other / (
        1.0 / (G_other - G_ref)
        + 2.0 * f_ref * (K_ref + 2.0 * G_ref) / (5.0 * G_ref * (K_ref + 4.0 * G_ref / 3.0))
    )
    return K_hs, G_hs


def hs_mix(f1, K1, G1, f2, K2, G2):
    """Hashin-Shtrikman upper and lower bounds for a two-phase isotropic composite."""
    if K1 >= K2:
        stiff = (f1, K1, G1)
        soft = (f2, K2, G2)
    else:
        stiff = (f2, K2, G2)
        soft = (f1, K1, G1)

    K_upper, G_upper = hs_bound(*stiff, *soft)
    K_lower, G_lower = hs_bound(*soft, *stiff)

    return dict(
        K_hs_upper=K_upper, G_hs_upper=G_upper,
        K_hs_lower=K_lower, G_hs_lower=G_lower,
        K_hs_avg=0.5 * (K_upper + K_lower),
        G_hs_avg=0.5 * (G_upper + G_lower),
    )


def homogenize():
    df = pd.read_csv(INPUT_CSV)

    K, G = cubic_to_isotropic_hill(df["C11"].values, df["C12"].values, df["C44"].values)
    df["K_phase"] = K
    df["G_phase"] = G
    E, nu = KG_to_E_nu(K, G)
    df["E_phase"] = E
    df["nu_phase"] = nu

    rows = []
    for (tag, T_K), group in df.groupby(["tag", "T_K"]):
        n_phases = len(group)

        if n_phases == 1:
            r = group.iloc[0]
            rows.append({
                "tag": tag, "T_K": T_K, "n_phases": 1,
                "phases": r["phase"],
                "K_VRH": r["K_phase"], "G_VRH": r["G_phase"],
                "K_HS_upper": r["K_phase"], "G_HS_upper": r["G_phase"],
                "K_HS_lower": r["K_phase"], "G_HS_lower": r["G_phase"],
                "K_HS_avg": r["K_phase"], "G_HS_avg": r["G_phase"],
            })
            continue

        if n_phases > 2:
            # Not expected for this binary TDB (max 2 coexisting phases at
            # equilibrium away from invariant points), but guard rather than
            # silently mis-mix if it happens.
            print(f"WARNING: {tag} at {T_K} K has {n_phases} phases, skipping (only 2-phase mixing implemented)")
            continue

        r1, r2 = group.iloc[0], group.iloc[1]
        f1, f2 = r1["phase_fraction"], r2["phase_fraction"]

        vrh = vrh_mix(f1, r1["K_phase"], r1["G_phase"], f2, r2["K_phase"], r2["G_phase"])
        hs = hs_mix(f1, r1["K_phase"], r1["G_phase"], f2, r2["K_phase"], r2["G_phase"])

        rows.append({
            "tag": tag, "T_K": T_K, "n_phases": 2,
            "phases": f"{r1['phase']}+{r2['phase']}",
            "K_VRH": vrh["K_hill"], "G_VRH": vrh["G_hill"],
            "K_HS_upper": hs["K_hs_upper"], "G_HS_upper": hs["G_hs_upper"],
            "K_HS_lower": hs["K_hs_lower"], "G_HS_lower": hs["G_hs_lower"],
            "K_HS_avg": hs["K_hs_avg"], "G_HS_avg": hs["G_hs_avg"],
        })

    out = pd.DataFrame(rows)
    E_vrh, nu_vrh = KG_to_E_nu(out["K_VRH"].values, out["G_VRH"].values)
    E_hs, nu_hs = KG_to_E_nu(out["K_HS_avg"].values, out["G_HS_avg"].values)
    out["E_VRH"] = E_vrh
    out["nu_VRH"] = nu_vrh
    out["E_HS_avg"] = E_hs
    out["nu_HS_avg"] = nu_hs

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    homogenize()