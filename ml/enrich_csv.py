"""
enrich_csv.py
=============
Adds accuracy metadata columns to elastic_constants_fecr.csv:

    conv_thr   : float  — SCF convergence threshold used (Ry)
    afm_init   : bool   — True if AFM sublattice initialisation was used
    noise_gpa  : float  — recommended noise σ (GPa) for ML weighting

Noise assignment rationale
--------------------------
Three independent sources of uncertainty affect each tag:

  1. conv_thr tier
       1e-8 → tightest, lowest stress noise
       1e-7 → one decade looser, still standard
       1e-5 → two decades looser than standard, significant numerical noise

  2. Magnetic initialisation
       FM  → appropriate for Fe-rich and mid-range compositions
       AFM → required for high-Cr (≥14 Cr atoms); spin state is physically
             different from the FM-initialised rest of the dataset

  3. Compound effect for fe02cr14/fe01cr15/fe00cr16
       Both looser conv_thr AND different magnetic init → highest uncertainty

Noise σ values (GPa):
  conv_thr=1e-8, FM  → σ = 1.0  GPa  (tier A — reference quality)
  conv_thr=1e-7, FM  → σ = 2.0  GPa  (tier B — one decade looser)
  conv_thr=1e-5, AFM → σ = 10.0 GPa  (tier C — two decades looser + AFM)

These are physically motivated estimates, NOT empirically measured.
The only empirical data point available is ΔP = 2.29 kbar between
conv_thr=1e-5 and 1e-7 at h0.00 for fe02cr14 (session_summary_may26).
That corresponds to a stress error of ~0.23 GPa in the hydrostatic
channel alone. The σ=10 GPa assignment is deliberately conservative
to down-weight these points without excluding them entirely.

Usage
-----
Run once after re-running elastic_extraction.ipynb:
    python enrich_csv.py --csv ../analysis/elastic_constants_fecr.csv

The script is idempotent — safe to re-run, overwrites existing columns.
"""

import argparse
import pandas as pd
import numpy as np

# ── Accuracy tier definitions ─────────────────────────────────────────────────
# Source: session_summary_may25_2026.md, run_elastic_grid_latest.sh
#
# Tag naming: fe{n_fe}cr{n_cr}  e.g. fe14cr02 = 2 Cr atoms out of 16
#
# Tier A: conv_thr=1e-8, FM init
#   fe16cr00 through fe05cr11  (n_cr = 0..11)
#
# Tier B: conv_thr=1e-7, FM init
#   fe04cr12, fe03cr13          (n_cr = 12, 13)
#   Note: the session summary says "~fe04cr12 → fe03cr13" — the boundary
#   between 1e-8 and 1e-7 was not precisely logged during the run.
#   n_cr=12 and n_cr=13 are assigned 1e-7 as the conservative choice.
#
# Tier C: conv_thr=1e-5, AFM sublattice init
#   fe02cr14, fe01cr15, fe00cr16  (n_cr = 14, 15, 16)

TIER_A = set(range(0,  12))   # n_cr 0..11
TIER_B = set(range(12, 14))   # n_cr 12..13
TIER_C = set(range(14, 17))   # n_cr 14..16

CONV_THR = {
    'A': 1e-8,
    'B': 1e-7,
    'C': 1e-5,
}
AFM_INIT = {
    'A': False,
    'B': False,
    'C': True,
}
NOISE_GPA = {
    'A': 1.0,
    'B': 2.0,
    'C': 10.0,
}


def get_tier(n_cr: int) -> str:
    if n_cr in TIER_A: return 'A'
    if n_cr in TIER_B: return 'B'
    if n_cr in TIER_C: return 'C'
    raise ValueError(f"n_cr={n_cr} out of expected range 0-16")


def enrich(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalise column names
    if 'cr_count' in df.columns and 'n_cr' not in df.columns:
        df['n_cr'] = df['cr_count']
    if 'cr_frac' in df.columns and 'x_cr' not in df.columns:
        df['x_cr'] = df['cr_frac']

    assert 'n_cr' in df.columns, "CSV must have 'n_cr' or 'cr_count' column"

    tiers = df['n_cr'].apply(get_tier)
    df['conv_thr']  = tiers.map(CONV_THR)
    df['afm_init']  = tiers.map(AFM_INIT)
    df['noise_gpa'] = tiers.map(NOISE_GPA)

    # Keep loose_thr for backwards compatibility (True = Tier C)
    df['loose_thr'] = tiers.map(lambda t: t == 'C')

    # Reorder columns cleanly
    meta_cols = ['tag', 'n_cr', 'cr_count', 'cr_frac', 'x_cr',
                 'C11', 'C12', 'C44', 'B', 'A',
                 'conv_thr', 'afm_init', 'noise_gpa', 'loose_thr', 'status']
    present = [c for c in meta_cols if c in df.columns]
    rest    = [c for c in df.columns if c not in present]
    df = df[present + rest]

    df.to_csv(csv_path, index=False)
    return df


def print_summary(df: pd.DataFrame):
    print("\n── Accuracy tier assignment ──────────────────────────────")
    print(f"{'tag':<14} {'n_cr':>5} {'conv_thr':>10} {'afm_init':>9} "
          f"{'noise_gpa':>10} {'tier':>5}")
    print("─" * 60)
    for _, r in df.iterrows():
        tier = get_tier(int(r['n_cr']))
        print(f"{r['tag']:<14} {int(r['n_cr']):>5} {r['conv_thr']:>10.0e} "
              f"{str(r['afm_init']):>9} {r['noise_gpa']:>10.1f} {tier:>5}")
    print()
    for t, label in [('A','1e-8 FM'), ('B','1e-7 FM'), ('C','1e-5 AFM')]:
        mask = df['n_cr'].apply(get_tier) == t
        print(f"Tier {t} ({label}): {mask.sum()} tags — "
              f"σ = {NOISE_GPA[t]} GPa")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enrich elastic constants CSV with accuracy metadata')
    parser.add_argument('--csv', default='../analysis/elastic_constants_fecr.csv',
                        help='Path to elastic_constants_fecr.csv')
    args = parser.parse_args()

    print(f"Enriching: {args.csv}")
    df = enrich(args.csv)
    print_summary(df)
    print(f"\nSaved: {args.csv}")
    print("Columns added: conv_thr, afm_init, noise_gpa")
    print("Column updated: loose_thr (now derived from tier, same values)")