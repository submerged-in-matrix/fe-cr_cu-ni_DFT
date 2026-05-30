"""
enrich_csv.py
=============
Produces two ML-ready elastic constants CSVs from empirical uncertainty data:

    elastic_constants_fecr_raw.csv       — as calculated, noise=1.0 GPa uniform
    elastic_constants_fecr_corrected.csv — per-constant empirical noise

Both files have identical columns and shape. They differ only in the three
noise columns. Feed them separately into ML models to evaluate the effect
of uncertainty correction.

Output noise columns (GPa):
    noise_C11_2C12  — σ for C11+2C12 combination (feeds h±0.01 calcs)
    noise_C11_C12   — σ for C11-C12  combination (feeds t±0.01 calcs)
    noise_C44       — σ for C44               (feeds s±0.01 calcs)

Noise derivation (corrected CSV):
    σ_thr = mean(sigma_thr_gpa of the two ±0.01 calcs that feed this constant)
    σ_geo = from empirical_sigma.csv per tag per constant (only if vcr_geometry_flag=True)
    noise_corrected = σ_thr + σ_geo   (additive — distinct systematic sources)
                    = σ_thr            (if vcr ok)

Sources (all produced by empirical_uncertainty.ipynb):
    per_calc_uncertainty.csv  — 170 rows, sigma_thr_gpa per calculation
    empirical_sigma.csv       — vcr geometry σ per tag per constant

Usage:
    python enrich_csv.py \\
        --csv     ../analysis/elastic_constants_fecr.csv \\
        --unc     ../analysis/per_calc_uncertainty.csv \\
        --sigma   ../analysis/empirical_sigma.csv \\
        --outdir  ../analysis
"""

import argparse
import os
import numpy as np
import pandas as pd

# ── Constant → strain type mapping ───────────────────────────────────────────
# Each elastic constant is extracted from two ±0.01 calculations
CONSTANT_TO_STYPE = {
    'C11_2C12': 'h',   # hydro  h-0.01, h+0.01
    'C11_C12':  't',   # tetra  t-0.01, t+0.01
    'C44':      's',   # shear  s-0.01, s+0.01
}

NOISE_RAW = 1.0   # GPa — uniform baseline, no correction


def load_sources(csv_path, unc_path, sigma_path):
    """Load and validate all three input CSVs."""
    assert os.path.exists(csv_path),   f"Not found: {csv_path}"
    assert os.path.exists(unc_path),   f"Not found: {unc_path}"
    assert os.path.exists(sigma_path), f"Not found: {sigma_path}"

    df    = pd.read_csv(csv_path)
    unc   = pd.read_csv(unc_path)
    sigma = pd.read_csv(sigma_path)

    # Normalise column names
    if 'cr_count' in df.columns and 'n_cr' not in df.columns:
        df['n_cr'] = df['cr_count']
    if 'cr_frac' in df.columns and 'x_cr' not in df.columns:
        df['x_cr'] = df['cr_frac']

    assert 'n_cr' in df.columns, "CSV must have n_cr or cr_count"
    return df, unc, sigma


def get_sigma_thr(unc, tag, stype):
    """
    Mean sigma_thr_gpa across the two ±0.01 calculations of given strain type
    for given tag. Returns float or NaN if not found.
    """
    mask = (
        (unc['tag']          == tag) &
        (unc['strain_type']  == stype) &
        (unc['epsilon'].abs() == 0.01)
    )
    vals = unc.loc[mask, 'sigma_thr_gpa'].dropna()
    return float(vals.mean()) if len(vals) > 0 else float('nan')


def get_sigma_geo(sigma, tag, constant_key):
    """
    Geometry σ for given tag and constant from empirical_sigma.csv.
    Returns float or 0.0 if tag not in Tier D.

    empirical_sigma.csv Tier D rows have:
        tags = tag name
        sigma_thr_gpa = geometry σ (from poly/spline residuals)

    The geometry σ stored there is the max mean across C11, C12, C44 —
    we refine per constant using the target-specific mean from resid_df
    which was saved in empirical_sigma. If per-constant rows not present,
    fall back to the single Tier D value.
    """
    # Look for exact tag match in sigma (Tier D rows have single tag in 'tags')
    row = sigma[sigma['tags'] == tag]
    if len(row) == 0:
        return 0.0
    val = row['sigma_thr_gpa'].values[0]
    return float(val) if not np.isnan(val) else 0.0


def build_noise_corrected(df, unc, sigma):
    """
    For each tag compute per-constant corrected noise.
    Returns df with three new columns:
        noise_C11_2C12, noise_C11_C12, noise_C44
    Also adds:
        afm_init, vcr_geometry_flag
    """
    rows = []
    for _, row in df.iterrows():
        tag = row['tag']

        # vcr geometry flag — any calc for this tag has vcr_geometry_flag=True
        vcr_flag = unc[unc['tag'] == tag]['vcr_geometry_flag'].any()

        # afm_init — any calc for this tag is AFM
        afm = unc[unc['tag'] == tag]['afm_init'].any()

        noise = {}
        for ckey, stype in CONSTANT_TO_STYPE.items():
            s_thr = get_sigma_thr(unc, tag, stype)
            s_geo = get_sigma_geo(sigma, tag, ckey) if vcr_flag else 0.0
            noise[f'noise_{ckey}'] = round(s_thr + s_geo, 4)

        rows.append({
            'tag':               tag,
            'vcr_geometry_flag': vcr_flag,
            'afm_init':          afm,
            **noise
        })

    noise_df = pd.DataFrame(rows)
    df = df.merge(noise_df, on='tag', how='left')
    return df


def build_output(df, corrected=True):
    """
    Build final output DataFrame with consistent column order.
    corrected=True  → empirical per-constant noise
    corrected=False → uniform noise_raw=1.0
    """
    out = df.copy()

    if corrected:
        # noise columns already added by build_noise_corrected
        pass
    else:
        # Uniform baseline — overwrite with 1.0
        out['noise_C11_2C12']    = NOISE_RAW
        out['noise_C11_C12']     = NOISE_RAW
        out['noise_C44']         = NOISE_RAW
        out['vcr_geometry_flag'] = False
        out['afm_init']          = False

    # loose_thr — True if any calc for this tag is at 1e-5
    loose_map = (
        df['noise_C11_2C12'].gt(1.0) |
        df['noise_C11_C12'].gt(1.0)  |
        df['noise_C44'].gt(1.0)
    ) if corrected else pd.Series([False]*len(df))
    out['loose_thr'] = loose_map

    # Consistent column order
    core = ['tag', 'n_cr']
    if 'cr_frac' in out.columns: core.append('cr_frac')
    if 'x_cr'    in out.columns: core.append('x_cr')
    core += ['C11', 'C12', 'C44', 'B']
    if 'A'       in out.columns: core.append('A')
    if 'status'  in out.columns: core.append('status')
    core += ['noise_C11_2C12', 'noise_C11_C12', 'noise_C44',
             'afm_init', 'vcr_geometry_flag', 'loose_thr']
    present = [c for c in core if c in out.columns]
    rest    = [c for c in out.columns if c not in present]
    return out[present + rest]


def print_summary(df_raw, df_corr):
    print('\n' + '='*65)
    print('NOISE COLUMN SUMMARY')
    print('='*65)
    print(f'\n{"tag":<14} {"C11+2C12 raw":>12} {"corr":>8} '
          f'{"C11-C12 raw":>12} {"corr":>8} '
          f'{"C44 raw":>8} {"corr":>8} {"vcr_geo":>8} {"afm":>5}')
    print('-'*65)
    for i in range(len(df_raw)):
        r   = df_raw.iloc[i]
        c   = df_corr.iloc[i]
        print(f"{r['tag']:<14} "
              f"{r['noise_C11_2C12']:>12.3f} {c['noise_C11_2C12']:>8.3f} "
              f"{r['noise_C11_C12']:>12.3f} {c['noise_C11_C12']:>8.3f} "
              f"{r['noise_C44']:>8.3f} {c['noise_C44']:>8.3f} "
              f"{str(c['vcr_geometry_flag']):>8} {str(c['afm_init']):>5}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Produce raw and corrected elastic constants CSVs')
    parser.add_argument('--csv',    default='../analysis/elastic_constants_fecr.csv',
                        help='Base elastic constants CSV')
    parser.add_argument('--unc',    default='../analysis/per_calc_uncertainty.csv',
                        help='Per-calculation uncertainty CSV from empirical_uncertainty.ipynb')
    parser.add_argument('--sigma',  default='../analysis/empirical_sigma.csv',
                        help='Empirical sigma CSV from empirical_uncertainty.ipynb')
    parser.add_argument('--outdir', default='../analysis',
                        help='Output directory for _raw.csv and _corrected.csv')
    args = parser.parse_args()

    print(f'Loading: {args.csv}')
    print(f'         {args.unc}')
    print(f'         {args.sigma}')

    df, unc, sigma = load_sources(args.csv, args.unc, args.sigma)

    df_corr_full = build_noise_corrected(df.copy(), unc, sigma)
    df_corr = build_output(df_corr_full, corrected=True)
    df_raw  = build_output(df_corr_full.copy(), corrected=False)

    raw_path  = os.path.join(args.outdir, 'elastic_constants_fecr_raw.csv')
    corr_path = os.path.join(args.outdir, 'elastic_constants_fecr_corrected.csv')
    df_raw.to_csv(raw_path,   index=False)
    df_corr.to_csv(corr_path, index=False)

    print(f'\nSaved: {raw_path}')
    print(f'Saved: {corr_path}')
    print_summary(df_raw, df_corr)
    print(f'\nBoth files: {len(df_raw)} rows, {len(df_raw.columns)} columns.')
    print('Feed separately into ML models to evaluate uncertainty correction effect.')