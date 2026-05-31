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

Noise derivation (corrected CSV) — additive, distinct systematic sources:
    σ_thr  = mean(sigma_thr_gpa of the two ±0.01 calcs that feed this constant)
    σ_geo  = from empirical_sigma.csv per tag (only if vcr_geometry_flag=True)
    σ_cub  = second-order leak from cubic enforcement (only if cubic_enforcement_residual=True)
             derived from (C11, C12) × eps_enforced × eps_strain
    noise_corrected = σ_thr + σ_geo + σ_cub

Sources (all produced by empirical_uncertainty.ipynb):
    per_calc_uncertainty.csv  — 170 rows, sigma_thr_gpa per calculation
    empirical_sigma.csv       — vcr geometry σ per tag per constant
    vcr_cell_distortion.csv   — relaxed cell geometry, cubic enforcement metrics

Usage:
    python enrich_csv.py \\
        --csv     ../analysis/elastic_constants_fecr.csv \\
        --unc     ../analysis/per_calc_uncertainty.csv \\
        --sigma   ../analysis/empirical_sigma.csv \\
        --dist    ../analysis/vcr_cell_distortion.csv \\
        --outdir  ../analysis
"""

import argparse
import os
import numpy as np
import pandas as pd

# ── Constant → strain type mapping ───────────────────────────────────────────
CONSTANT_TO_STYPE = {
    'C11_2C12': 'h',
    'C11_C12':  't',
    'C44':      's',
}

NOISE_RAW  = 1.0    # GPa — uniform baseline
EPS_STRAIN = 0.01   # finite-difference strain step


def load_sources(csv_path, unc_path, sigma_path):
    """Load and validate the three core input CSVs."""
    assert os.path.exists(csv_path),   f"Not found: {csv_path}"
    assert os.path.exists(unc_path),   f"Not found: {unc_path}"
    assert os.path.exists(sigma_path), f"Not found: {sigma_path}"

    df    = pd.read_csv(csv_path)
    unc   = pd.read_csv(unc_path)
    sigma = pd.read_csv(sigma_path)

    if 'cr_count' in df.columns and 'n_cr' not in df.columns:
        df['n_cr'] = df['cr_count']
    if 'cr_frac' in df.columns and 'x_cr' not in df.columns:
        df['x_cr'] = df['cr_frac']

    assert 'n_cr' in df.columns, "CSV must have n_cr or cr_count"
    return df, unc, sigma


def get_sigma_thr(unc, tag, stype):
    """
    Mean sigma_thr_gpa across the two ±0.01 calculations of given strain type.
    """
    mask = (
        (unc['tag']          == tag)   &
        (unc['strain_type']  == stype) &
        (unc['epsilon'].abs() == 0.01)
    )
    vals = unc.loc[mask, 'sigma_thr_gpa'].dropna()
    return float(vals.mean()) if len(vals) > 0 else float('nan')


def get_sigma_geo(sigma, tag, constant_key):
    """
    Geometry σ for given tag from empirical_sigma.csv Tier D rows.
    Returns float or 0.0 if tag not in Tier D.
    """
    row = sigma[sigma['tags'] == tag]
    if len(row) == 0:
        return 0.0
    val = row['sigma_thr_gpa'].values[0]
    return float(val) if not np.isnan(val) else 0.0


def compute_cubic_enforcement_sigma(ec_df, dist_path, unc):
    """
    Compute second-order leak noise from cubic symmetry enforcement.

    Physical basis
    --------------
    The pipeline enforces b = c = SA_EQ = a[0,0] on the relaxed cell.
    For tags where b_true != a, the enforced strain is:
        eps_enforced = (a - b_true) / b_true

    This introduces a residual stress at the reference geometry:
        sigma_xx = 2*C12*eps_enforced        (a-direction)
        sigma_yy = (C11+C12)*eps_enforced    (b-direction)
        sigma_zz = (C11+C12)*eps_enforced    (c-direction)

    Since all three extraction formulas are pure central differences
    (stress(+eps) - stress(-eps)), this constant offset cancels to first
    order. The second-order leak scales as eps_strain * residual_scale:

        sigma_C11_2C12 = eps_strain * |P_residual|
        sigma_C11_C12  = eps_strain * |(C11-C12)*eps_enforced|
        sigma_C44      = 0  (normal stress, no shear projection)

    Parameters
    ----------
    ec_df     : DataFrame with columns tag, C11, C12
    dist_path : path to vcr_cell_distortion.csv
    unc       : per_calc_uncertainty DataFrame (to check cubic_enforcement_residual)

    Returns
    -------
    dict {tag: {'sigma_C11_2C12': float,
                'sigma_C11_C12':  float,
                'sigma_C44':      float}}
    """
    if dist_path is None or not os.path.exists(dist_path):
        print('  WARNING: vcr_cell_distortion.csv not found — '
              'cubic enforcement sigma skipped (add --dist path)')
        return {}

    vc = pd.read_csv(dist_path)

    # Check cubic_enforcement_residual column exists
    if 'cubic_enforcement_residual' not in unc.columns:
        print('  WARNING: cubic_enforcement_residual column not in '
              'per_calc_uncertainty.csv — skipping')
        return {}

    # Tags flagged for cubic enforcement
    flagged_tags = set(
        unc[unc['cubic_enforcement_residual'] == True]['tag'].unique()
    )

    out = {}
    for _, r in vc[vc['tag'].isin(flagged_tags)].iterrows():
        tag = r['tag']
        ec_row = ec_df[ec_df['tag'] == tag]
        if ec_row.empty:
            print(f'  WARNING: {tag} not in elastic constants CSV — skipping')
            continue

        c11 = float(ec_row['C11'].values[0])
        c12 = float(ec_row['C12'].values[0])
        a   = float(r['a'])
        b   = float(r['b'])

        eps_enf = (a - b) / b                         # engineering strain

        # Residual stress components at enforced cubic reference
        sig_xx = 2 * c12 * eps_enf
        sig_yy = (c11 + c12) * eps_enf
        sig_zz = (c11 + c12) * eps_enf
        P_res  = (sig_xx + sig_yy + sig_zz) / 3.0
        DS_res = sig_zz - sig_xx                       # = (C11-C12)*eps_enf

        # Second-order leak after central-difference cancellation
        out[tag] = {
            'sigma_C11_2C12': round(abs(EPS_STRAIN * P_res),  4),
            'sigma_C11_C12':  round(abs(EPS_STRAIN * DS_res), 4),
            'sigma_C44':      0.0,
        }

    if out:
        print(f'  Cubic enforcement sigma computed for {len(out)} tags:')
        print(f'  {"tag":<12} {"s_C11+2C12":>11} {"s_C11-C12":>10} {"s_C44":>6}')
        for tag, v in out.items():
            print(f'  {tag:<12} {v["sigma_C11_2C12"]:>11.4f} '
                  f'{v["sigma_C11_C12"]:>10.4f} {v["sigma_C44"]:>6.1f}')

    return out


def build_noise_corrected(df, unc, sigma, cubic_sigma):
    """
    For each tag compute per-constant corrected noise.
    noise_corrected = sigma_thr + sigma_geo + sigma_cubic  (all additive)

    Returns df with noise columns and flag columns added.
    """
    rows = []
    for _, row in df.iterrows():
        tag = row['tag']

        vcr_flag = unc[unc['tag'] == tag]['vcr_geometry_flag'].any()
        afm      = unc[unc['tag'] == tag]['afm_init'].any()

        # cubic_enforcement_residual flag
        if 'cubic_enforcement_residual' in unc.columns:
            cub_flag = unc[unc['tag'] == tag]['cubic_enforcement_residual'].any()
        else:
            cub_flag = False

        noise   = {}
        sources = {}
        for ckey, stype in CONSTANT_TO_STYPE.items():
            s_thr = get_sigma_thr(unc, tag, stype)
            s_geo = get_sigma_geo(sigma, tag, ckey) if vcr_flag else 0.0
            s_cub = cubic_sigma.get(tag, {}).get(f'sigma_{ckey}', 0.0)

            noise[f'noise_{ckey}'] = round(s_thr + s_geo + s_cub, 4)

            # Build uncertainty source string
            src_parts = []
            if not np.isnan(s_thr):
                src_parts.append(f'sigma_thr={s_thr:.3f}')
            if s_geo > 0:
                src_parts.append(f'sigma_geo={s_geo:.3f}')
            if s_cub > 0:
                src_parts.append(f'sigma_cub={s_cub:.4f}')
            sources[ckey] = ' | '.join(src_parts) if src_parts else 'baseline'

        rows.append({
            'tag':                        tag,
            'vcr_geometry_flag':          vcr_flag,
            'afm_init':                   afm,
            'cubic_enforcement_residual': cub_flag,
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

    if not corrected:
        out['noise_C11_2C12']             = NOISE_RAW
        out['noise_C11_C12']              = NOISE_RAW
        out['noise_C44']                  = NOISE_RAW
        out['vcr_geometry_flag']          = False
        out['afm_init']                   = False
        out['cubic_enforcement_residual'] = False

    # loose_thr — any noise above raw baseline
    if corrected:
        loose_map = (
            df['noise_C11_2C12'].gt(NOISE_RAW) |
            df['noise_C11_C12'].gt(NOISE_RAW)  |
            df['noise_C44'].gt(NOISE_RAW)
        )
    else:
        loose_map = pd.Series([False] * len(df))
    out['loose_thr'] = loose_map

    # Consistent column order
    core = ['tag', 'n_cr']
    for c in ['cr_frac','x_cr']:
        if c in out.columns: core.append(c)
    core += ['C11', 'C12', 'C44', 'B']
    for c in ['A','status']:
        if c in out.columns: core.append(c)
    core += ['noise_C11_2C12', 'noise_C11_C12', 'noise_C44',
             'afm_init', 'vcr_geometry_flag',
             'cubic_enforcement_residual', 'loose_thr']
    present = [c for c in core if c in out.columns]
    rest    = [c for c in out.columns if c not in present]
    return out[present + rest]


def print_summary(df_raw, df_corr):
    print('\n' + '='*75)
    print('NOISE COLUMN SUMMARY')
    print('='*75)
    print(f'\n{"tag":<14} {"C11+2C12":>8} {"":>8} '
          f'{"C11-C12":>8} {"":>8} '
          f'{"C44":>6} {"":>6} '
          f'{"vcr_geo":>8} {"cub_enf":>8} {"afm":>5}')
    print(f'{"":14} {"raw":>8} {"corr":>8} '
          f'{"raw":>8} {"corr":>8} '
          f'{"raw":>6} {"corr":>6}')
    print('-'*75)
    for i in range(len(df_raw)):
        r = df_raw.iloc[i]
        c = df_corr.iloc[i]
        print(f"{r['tag']:<14} "
              f"{r['noise_C11_2C12']:>8.3f} {c['noise_C11_2C12']:>8.3f} "
              f"{r['noise_C11_C12']:>8.3f} {c['noise_C11_C12']:>8.3f} "
              f"{r['noise_C44']:>6.3f} {c['noise_C44']:>6.3f} "
              f"{str(c['vcr_geometry_flag']):>8} "
              f"{str(c['cubic_enforcement_residual']):>8} "
              f"{str(c['afm_init']):>5}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Produce raw and corrected elastic constants CSVs')
    parser.add_argument('--csv',    default='../analysis/elastic_constants_fecr.csv')
    parser.add_argument('--unc',    default='../analysis/per_calc_uncertainty.csv')
    parser.add_argument('--sigma',  default='../analysis/empirical_sigma.csv')
    parser.add_argument('--dist',   default='../analysis/vcr_cell_distortion.csv',
                        help='vcr_cell_distortion.csv from distortion audit cell')
    parser.add_argument('--outdir', default='../analysis')
    args = parser.parse_args()

    print(f'Loading: {args.csv}')
    print(f'         {args.unc}')
    print(f'         {args.sigma}')
    print(f'         {args.dist}')

    df, unc, sigma = load_sources(args.csv, args.unc, args.sigma)

    # Compute cubic enforcement sigma using exact C11, C12 from elastic CSV
    cubic_sigma = compute_cubic_enforcement_sigma(df, args.dist, unc)

    df_corr_full = build_noise_corrected(df.copy(), unc, sigma, cubic_sigma)
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