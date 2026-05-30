"""
data_utils.py
=============
Shared data loading and preprocessing for all ML surrogate notebooks.

Two data modes — pass the appropriate CSV:
    elastic_constants_fecr_raw.csv       → load_data(...) → uniform noise
    elastic_constants_fecr_corrected.csv → load_data(...) → empirical per-constant noise

Noise columns in both CSVs:
    noise_C11_2C12  σ (GPa) for C11+2C12 combination
    noise_C11_C12   σ (GPa) for C11-C12  combination
    noise_C44       σ (GPa) for C44

Pipeline:
    DFT extraction
        → empirical_uncertainty.ipynb  (produces per_calc_uncertainty.csv, empirical_sigma.csv)
        → enrich_csv.py                (produces _raw.csv and _corrected.csv)
        → [this module]                (loads either CSV)
        → GP / XGBoost / MLP
        → FEM
"""

import numpy as np
import pandas as pd
import os


# ── Pure numpy replacements — zero sklearn dependency ─────────────────────────

class LeaveOneOut:
    """Minimal LOO splitter — same interface as sklearn.model_selection.LeaveOneOut."""
    def split(self, X):
        n = len(X)
        idx = np.arange(n)
        for i in range(n):
            yield np.concatenate([idx[:i], idx[i+1:]]), np.array([i])


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ── Shared constants ──────────────────────────────────────────────────────────
TARGETS    = ['C11', 'C12', 'C44']
PALETTE    = {'C11': 'steelblue', 'C12': 'darkorange', 'C44': 'seagreen'}
N_ATOMS    = 16
EPS_STRAIN = 0.01

# Constant → noise column mapping
NOISE_COL = {
    'C11': 'noise_C11_2C12',   # C11 is recovered from C11+2C12 and C11-C12
    'C12': 'noise_C11_C12',    # C12 same
    'C44': 'noise_C44',
}

# Backwards-compatible scalar — use noise columns from CSV instead
NOISE_NORMAL  = 1.0
NOISE_FLAGGED = 10.0


def load_data(csv_path: str) -> dict:
    """
    Load elastic constants CSV — works with both _raw and _corrected variants.

    Noise columns noise_C11_2C12, noise_C11_C12, noise_C44 must be present.
    If missing (old CSV), falls back to uniform noise=1.0 with a warning.

    Returns
    -------
    dict with keys:
        df              full DataFrame
        X               (N,1) Cr fraction
        X_pred          (200,1) dense prediction grid
        x_pred_atoms    (200,) atom count axis
        targets         {C11/C12/C44: np.array}
        alpha           dict {C11/C12/C44: (N,) noise VARIANCE array}
                        — use as GP diagonal noise per target
        noise_gpa       dict {C11/C12/C44: (N,) noise σ array}
                        — use as sample weights in XGB/MLP
        alpha_mean      (N,) mean noise variance across C11,C12,C44
                        — convenience for single-noise models
        flagged         (N,) bool — True where any noise > 1.0
        afm_init        (N,) bool
        vcr_geometry    (N,) bool
        mode            str — 'raw' or 'corrected'
    """
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    df = pd.read_csv(csv_path)

    # Normalise column names
    if 'cr_count' in df.columns and 'n_cr' not in df.columns:
        df['n_cr'] = df['cr_count']
    if 'cr_frac' in df.columns and 'x_cr' not in df.columns:
        df['x_cr'] = df['cr_frac']

    required = ['tag', 'n_cr', 'C11', 'C12', 'C44']
    missing  = [c for c in required if c not in df.columns]
    assert not missing, f"CSV missing columns: {missing}"

    # Derived
    if 'x_cr' not in df.columns:
        df['x_cr'] = df['n_cr'] / float(N_ATOMS)
    if 'B' not in df.columns:
        df['B'] = (df['C11'] + 2*df['C12']) / 3.0

    # Detect mode from filename
    mode = 'corrected' if 'corrected' in os.path.basename(csv_path) else 'raw'

    # ── Noise columns ─────────────────────────────────────────────────────────
    noise_cols_present = all(c in df.columns for c in
                             ['noise_C11_2C12','noise_C11_C12','noise_C44'])
    if not noise_cols_present:
        print("WARNING: noise columns not found — falling back to uniform σ=1.0")
        print("         Run enrich_csv.py to produce _raw.csv and _corrected.csv")
        df['noise_C11_2C12'] = 1.0
        df['noise_C11_C12']  = 1.0
        df['noise_C44']      = 1.0
        mode = 'raw_fallback'

    # afm_init / vcr_geometry_flag
    if 'afm_init' not in df.columns:
        df['afm_init'] = False
    if 'vcr_geometry_flag' not in df.columns:
        df['vcr_geometry_flag'] = False
    if 'loose_thr' not in df.columns:
        df['loose_thr'] = False

    # ── Build arrays ──────────────────────────────────────────────────────────
    X            = df['x_cr'].values.reshape(-1, 1)
    X_pred       = np.linspace(0, 1, 200).reshape(-1, 1)
    x_pred_atoms = (X_pred * N_ATOMS).flatten()
    tgt          = {t: df[t].values.astype(float) for t in TARGETS}

    # Per-constant noise — C11 and C12 both come from C11+2C12 and C11-C12
    # C11 = (C11+2C12 + 2*C11-C12)/3 → propagate both noise sources
    # C12 = (C11+2C12 -   C11-C12)/3 → same
    # For ML purposes assign each constant its primary noise column
    noise_gpa = {
        'C11': df['noise_C11_2C12'].values.astype(float),
        'C12': df['noise_C11_C12'].values.astype(float),
        'C44': df['noise_C44'].values.astype(float),
    }
    alpha = {t: noise_gpa[t]**2 for t in TARGETS}

    # Convenience: mean alpha across targets for single-noise models
    alpha_mean = np.mean([alpha[t] for t in TARGETS], axis=0)

    flagged      = df['loose_thr'].values.astype(bool)
    afm_init     = df['afm_init'].values.astype(bool)
    vcr_geometry = df['vcr_geometry_flag'].values.astype(bool)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\nLoaded {len(df)} tags — mode: {mode.upper()}")
    print(f"{'tag':<14} {'n_cr':>4} "
          f"{'σ C11+2C12':>11} {'σ C11-C12':>10} {'σ C44':>7} "
          f"{'afm':>5} {'vcr_geo':>8}")
    print('-' * 60)
    for _, r in df.iterrows():
        print(f"{r['tag']:<14} {int(r['n_cr']):>4} "
              f"{r['noise_C11_2C12']:>11.3f} {r['noise_C11_C12']:>10.3f} "
              f"{r['noise_C44']:>7.3f} "
              f"{str(r['afm_init']):>5} {str(r['vcr_geometry_flag']):>8}")

    return dict(
        df=df,
        X=X, X_pred=X_pred, x_pred_atoms=x_pred_atoms,
        targets=tgt,
        alpha=alpha,           # {target: (N,) variance} — per-constant GP noise
        noise_gpa=noise_gpa,   # {target: (N,) σ}        — per-constant sample weight
        alpha_mean=alpha_mean, # (N,) mean variance       — convenience
        flagged=flagged,
        afm_init=afm_init,
        vcr_geometry=vcr_geometry,
        mode=mode,
    )


def LOO_evaluate(model_fn, X, y, extra_kw=None):
    """
    Generic LOO cross-validation.
    model_fn(X_train, y_train, **extra_kw) → fitted model with .predict(X)
    Per-point arrays in extra_kw are auto-sliced to train indices.
    """
    if extra_kw is None:
        extra_kw = {}
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr, te in loo.split(X):
        kw = {k: v[tr] if (hasattr(v,'__len__') and len(v)==len(X)) else v
              for k, v in extra_kw.items()}
        model = model_fn(X[tr], y[tr], **kw)
        preds.append(float(model.predict(X[te])))
        trues.append(float(y[te[0]]))
    preds = np.array(preds)
    trues = np.array(trues)
    return dict(loo_true=trues, loo_preds=preds,
                mae=mean_absolute_error(trues, preds),
                r2=r2_score(trues, preds),
                residuals=trues - preds)


def plot_loo_residuals(ax, residuals, n_cr, tier_or_flagged, title):
    """
    Residual bar chart. tier_or_flagged accepts:
        - str array ('A','B','C') → coloured by tier
        - bool array             → red if True, blue if False
    """
    tier_colours = {'A':'steelblue','B':'goldenrod','C':'tomato'}
    if hasattr(tier_or_flagged[0], 'item'):
        arr = tier_or_flagged
    else:
        arr = tier_or_flagged
    if arr.dtype == bool or arr.dtype == np.bool_:
        cols = ['tomato' if f else 'steelblue' for f in arr]
    else:
        cols = [tier_colours.get(t,'steelblue') for t in arr]
    ax.bar(n_cr, residuals, color=cols, edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='k', linewidth=1)
    ax.set_xlabel('Cr atoms (out of 16)')
    ax.set_ylabel('Residual (GPa)')
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=f'Tier {t}')
               for t, c in tier_colours.items()]
    ax.legend(handles=handles, fontsize=8, loc='upper left')


# ── .dat file readers ─────────────────────────────────────────────────────────

def read_hydro(path):
    if not os.path.exists(path): return None
    d = np.loadtxt(path); d = d[d[:,0].argsort()]
    return {'P_minus':d[0,1],'P_zero':d[1,1],'P_plus':d[2,1]}


def read_shear(path):
    if not os.path.exists(path): return None
    d = np.loadtxt(path); d = d[d[:,0].argsort()]
    return {'S12_minus':d[0,1],'S12_zero':d[1,1],'S12_plus':d[2,1]}


def read_tetra(path):
    if not os.path.exists(path): return None
    d = np.loadtxt(path); d = d[d[:,0].argsort()]
    return {'S11_minus':d[0,1],'S11_zero':d[1,1],'S11_plus':d[2,1],
            'S33_minus':d[0,2],'S33_zero':d[1,2],'S33_plus':d[2,2],
            'DS_minus': d[0,3],'DS_zero': d[1,3],'DS_plus': d[2,3]}


def load_asymmetry(df, dat_root):
    """Tension-compression asymmetry from hydro .dat files."""
    records = []
    for row in df.itertuples():
        h = read_hydro(os.path.join(dat_root, f'fecr_{row.tag}_hydro.dat'))
        if h is None:
            print(f'  SKIP {row.tag}: hydro.dat not found')
            continue
        records.append({'tag':row.tag,'n_cr':row.n_cr,'x_cr':row.x_cr,
                        'P_minus':h['P_minus'],'P_zero':h['P_zero'],
                        'P_plus':h['P_plus'],
                        'asymmetry_kbar':h['P_plus']+h['P_minus']-2*h['P_zero']})
    return pd.DataFrame(records)


def load_all_dat(df, dat_root):
    """All three strain types for all tags into one wide DataFrame (kbar)."""
    records = []
    for row in df.itertuples():
        tag = row.tag
        h = read_hydro(os.path.join(dat_root, f'fecr_{tag}_hydro.dat'))
        s = read_shear(os.path.join(dat_root, f'fecr_{tag}_shear.dat'))
        t = read_tetra(os.path.join(dat_root, f'fecr_{tag}_tetra.dat'))
        rec = {'tag':tag,'n_cr':row.n_cr,'x_cr':row.x_cr}
        if h: rec.update({f'hydro_{k}':v for k,v in h.items()})
        if s: rec.update({f'shear_{k}':v for k,v in s.items()})
        if t: rec.update({f'tetra_{k}':v for k,v in t.items()})
        records.append(rec)
    return pd.DataFrame(records)