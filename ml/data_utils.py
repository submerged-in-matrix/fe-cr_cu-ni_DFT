"""
data_utils.py
=============
Shared data loading and preprocessing for all ML surrogate notebooks.

Import at the top of each notebook:
    from data_utils import load_data, LOO_evaluate, plot_loo_residuals,
                          load_asymmetry, load_all_dat, TARGETS, PALETTE

Pipeline: DFT extraction → enrich_csv.py → [this module] → GP / XGBoost / MLP → FEM

Accuracy tiers (from session_summary_may25_2026.md + run_elastic_grid.sh):
─────────────────────────────────────────────────────────────────────────────
  Tier A │ n_cr 0–11  │ conv_thr=1e-8 │ FM init   │ noise_gpa = 1.0
  Tier B │ n_cr 12–13 │ conv_thr=1e-7 │ FM init   │ noise_gpa = 2.0
  Tier C │ n_cr 14–16 │ conv_thr=1e-5 │ AFM init  │ noise_gpa = 10.0
─────────────────────────────────────────────────────────────────────────────
noise_gpa is the recommended σ (GPa) per point for ML weighting.
It encodes BOTH convergence quality AND magnetic initialisation uncertainty.
"""

import numpy as np
import pandas as pd
import os


# ── Pure numpy replacements for sklearn metrics and CV ────────────────────────
# data_utils.py has zero sklearn dependency by design.

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
N_ATOMS    = 16      # BCC supercell size
EPS_STRAIN = 0.01    # strain magnitude used in DFT

# Tier noise values (GPa) — kept here for notebooks that need them as scalars
NOISE_TIER = {'A': 1.0, 'B': 2.0, 'C': 10.0}

# ── Backwards-compatible scalar noise (deprecated — use noise_gpa column) ─────
# These remain so old code doesn't break, but load_data() now prefers noise_gpa
NOISE_NORMAL  = 1.0   # Tier A
NOISE_FLAGGED = 10.0  # Tier C


def _assign_tier(n_cr: int) -> str:
    """Map n_cr → accuracy tier label A/B/C."""
    if   n_cr <= 11: return 'A'
    elif n_cr <= 13: return 'B'
    else:            return 'C'


def load_data(csv_path: str) -> dict:
    """
    Load elastic constants CSV. Handles both old CSVs (no noise_gpa column)
    and new enriched CSVs (with conv_thr, afm_init, noise_gpa).

    If noise_gpa is missing from CSV, assigns it from the tier table above
    and prints a warning — run enrich_csv.py to fix permanently.

    Returns
    -------
    dict with keys:
        df           - full DataFrame (with all metadata columns)
        X            - (N,1) Cr fraction — training feature
        X_pred       - (200,1) dense prediction grid
        x_pred_atoms - (200,)  same in atom units for x-axis
        targets      - dict {C11/C12/C44: np.array}
        alpha        - (N,) noise VARIANCE array  = noise_gpa²
                       used directly in GP kernel as K_noise = diag(alpha)
        noise_gpa    - (N,) noise σ array (std dev, not variance)
        flagged      - (N,) bool — True for Tier C (conv_thr=1e-5, AFM)
        tier         - (N,) str array — 'A', 'B', or 'C' per point
    """
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    df = pd.read_csv(csv_path)

    # ── Normalise column names to internal standard ───────────────────────────
    # CSV uses 'cr_count' and 'cr_frac'; internally we use 'n_cr' and 'x_cr'
    if 'cr_count' in df.columns and 'n_cr' not in df.columns:
        df['n_cr'] = df['cr_count']
    if 'cr_frac' in df.columns and 'x_cr' not in df.columns:
        df['x_cr'] = df['cr_frac']

    required = ['tag', 'n_cr', 'C11', 'C12', 'C44']
    missing  = [c for c in required if c not in df.columns]
    assert not missing, f"CSV missing columns: {missing}"

    # ── Derived columns ───────────────────────────────────────────────────────
    if 'x_cr' not in df.columns:
        df['x_cr'] = df['n_cr'] / float(N_ATOMS)

    if 'B' not in df.columns:
        df['B'] = (df['C11'] + 2*df['C12']) / 3.0

    # ── Accuracy tier ─────────────────────────────────────────────────────────
    df['tier'] = df['n_cr'].apply(_assign_tier)

    # ── conv_thr ──────────────────────────────────────────────────────────────
    if 'conv_thr' not in df.columns:
        tier_to_thr = {'A': 1e-8, 'B': 1e-7, 'C': 1e-5}
        df['conv_thr'] = df['tier'].map(tier_to_thr)
        print("INFO: 'conv_thr' column not in CSV — assigned from tier table.")
        print("      Run enrich_csv.py to add it permanently.")

    # ── afm_init ──────────────────────────────────────────────────────────────
    if 'afm_init' not in df.columns:
        df['afm_init'] = df['tier'].map({'A': False, 'B': False, 'C': True})
        print("INFO: 'afm_init' column not in CSV — assigned from tier table.")

    # ── noise_gpa — THE KEY COLUMN ────────────────────────────────────────────
    if 'noise_gpa' not in df.columns:
        df['noise_gpa'] = df['tier'].map(NOISE_TIER)
        print("WARNING: 'noise_gpa' column not in CSV — assigned from tier table.")
        print("         Run enrich_csv.py to add it permanently to the CSV.")
        print("         Assigned values:")
        for tier, grp in df.groupby('tier'):
            print(f"           Tier {tier}: {grp['tag'].tolist()} → σ={NOISE_TIER[tier]} GPa")
    else:
        # Verify CSV values are consistent with tier table (warn if not)
        expected = df['tier'].map(NOISE_TIER)
        mismatch = df[df['noise_gpa'] != expected]
        if len(mismatch) > 0:
            print(f"WARNING: {len(mismatch)} rows have noise_gpa inconsistent with tier table:")
            for _, r in mismatch.iterrows():
                print(f"  {r['tag']}: CSV noise_gpa={r['noise_gpa']}, "
                      f"tier {r['tier']} expects {NOISE_TIER[r['tier']]}")
            print("  Using CSV values. Edit noise_gpa in CSV if you want to override.")

    # ── loose_thr — backwards compatibility ──────────────────────────────────
    if 'loose_thr' not in df.columns:
        df['loose_thr'] = df['tier'] == 'C'
    df['loose_thr'] = df['loose_thr'].astype(bool)

    # ── Build output arrays ───────────────────────────────────────────────────
    X            = df['x_cr'].values.reshape(-1, 1)
    X_pred       = np.linspace(0, 1, 200).reshape(-1, 1)
    x_pred_atoms = (X_pred * N_ATOMS).flatten()
    noise_gpa    = df['noise_gpa'].values.astype(float)
    alpha        = noise_gpa ** 2          # variance = σ²
    flagged      = df['loose_thr'].values  # Tier C only
    tier_arr     = df['tier'].values
    tgt          = {t: df[t].values.astype(float) for t in TARGETS}

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\nLoaded {len(df)} points across {len(set(tier_arr))} accuracy tiers:")
    for t in ['A', 'B', 'C']:
        mask = tier_arr == t
        tags = df.loc[mask, 'tag'].tolist()
        thr  = df.loc[mask, 'conv_thr'].iloc[0] if mask.any() else '—'
        afm  = df.loc[mask, 'afm_init'].iloc[0] if mask.any() else '—'
        σ    = noise_gpa[mask][0] if mask.any() else '—'
        print(f"  Tier {t}: {sum(mask):2d} tags | conv_thr={thr:.0e} | "
              f"afm_init={afm} | noise_gpa={σ} GPa")
        print(f"           {tags}")

    return dict(
        df=df,
        X=X, X_pred=X_pred, x_pred_atoms=x_pred_atoms,
        targets=tgt,
        alpha=alpha,          # (N,) noise VARIANCE — use in GP
        noise_gpa=noise_gpa,  # (N,) noise σ       — use in XGB/MLP as sample weight
        flagged=flagged,       # (N,) bool           — Tier C mask for plots
        tier=tier_arr,         # (N,) str            — full tier labels
    )


def LOO_evaluate(model_fn, X: np.ndarray, y: np.ndarray,
                 extra_kw: dict = None) -> dict:
    """
    Generic Leave-One-Out cross-validation.

    Parameters
    ----------
    model_fn : callable — (X_train, y_train, **extra_kw) → fitted model
    X        : (N,1) feature array
    y        : (N,)  target values
    extra_kw : dict  — passed to model_fn each fold
                       per-point arrays are automatically sliced to train indices

    Returns
    -------
    dict: loo_true, loo_preds, mae, r2, residuals
    """
    if extra_kw is None:
        extra_kw = {}
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr, te in loo.split(X):
        kw = {k: v[tr] if (hasattr(v, '__len__') and len(v) == len(X)) else v
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


def plot_loo_residuals(ax, residuals: np.ndarray, n_cr: np.ndarray,
                       tier: np.ndarray, title: str):
    """
    Reusable residual bar chart coloured by accuracy tier.

    Tier A: steelblue  (reference quality)
    Tier B: goldenrod  (slightly looser)
    Tier C: tomato     (loosest + AFM init)
    """
    tier_colours = {'A': 'steelblue', 'B': 'goldenrod', 'C': 'tomato'}
    # Accept either tier labels ('A','B','C') or legacy bool flagged array
    if tier.dtype == bool:
        cols = ['tomato' if f else 'steelblue' for f in tier]
    else:
        cols = [tier_colours.get(t, 'steelblue') for t in tier]
    ax.bar(n_cr, residuals, color=cols, edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='k', linewidth=1)
    ax.set_xlabel('Cr atoms (out of 16)')
    ax.set_ylabel('Residual (GPa)')
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=f'Tier {t}') for t, c in tier_colours.items()]
    ax.legend(handles=handles, fontsize=8, loc='upper left')


def read_hydro(path: str):
    """Read hydro .dat → dict with P_minus, P_zero, P_plus (kbar). None if missing."""
    if not os.path.exists(path): return None
    d = np.loadtxt(path); d = d[d[:,0].argsort()]
    return {'P_minus': d[0,1], 'P_zero': d[1,1], 'P_plus': d[2,1]}


def read_shear(path: str):
    """Read shear .dat → dict with S12_minus, S12_zero, S12_plus (kbar). None if missing."""
    if not os.path.exists(path): return None
    d = np.loadtxt(path); d = d[d[:,0].argsort()]
    return {'S12_minus': d[0,1], 'S12_zero': d[1,1], 'S12_plus': d[2,1]}


def read_tetra(path: str):
    """Read tetra .dat → dict with S11/S33/DS _minus/zero/plus (kbar). None if missing."""
    if not os.path.exists(path): return None
    d = np.loadtxt(path); d = d[d[:,0].argsort()]
    return {'S11_minus': d[0,1], 'S11_zero': d[1,1], 'S11_plus': d[2,1],
            'S33_minus': d[0,2], 'S33_zero': d[1,2], 'S33_plus': d[2,2],
            'DS_minus':  d[0,3], 'DS_zero':  d[1,3], 'DS_plus':  d[2,3]}


def load_asymmetry(df: pd.DataFrame, dat_root: str) -> pd.DataFrame:
    """
    Build tension-compression asymmetry DataFrame from hydro .dat files.
    Files: dat_root/fecr_{tag}_hydro.dat  (flat directory, no subdirs)
    asymmetry_kbar = P(+ε) + P(−ε) − 2·P(0)   (0 = perfectly harmonic)
    """
    records = []
    for row in df.itertuples():
        tag  = row.tag
        h    = read_hydro(os.path.join(dat_root, f'fecr_{tag}_hydro.dat'))
        if h is None:
            print(f'  SKIP {tag}: hydro.dat not found in {dat_root}')
            continue
        records.append({'tag': tag, 'n_cr': row.n_cr, 'x_cr': row.x_cr,
                        'P_minus': h['P_minus'], 'P_zero': h['P_zero'],
                        'P_plus': h['P_plus'],
                        'asymmetry_kbar': h['P_plus'] + h['P_minus'] - 2*h['P_zero'],
                        'tier': _assign_tier(row.n_cr),
                        'loose_thr': row.loose_thr})
    return pd.DataFrame(records)


def load_all_dat(df: pd.DataFrame, dat_root: str) -> pd.DataFrame:
    """Load all three strain types for all tags into one wide DataFrame (kbar)."""
    records = []
    for row in df.itertuples():
        tag = row.tag
        h = read_hydro(os.path.join(dat_root, f'fecr_{tag}_hydro.dat'))
        s = read_shear(os.path.join(dat_root, f'fecr_{tag}_shear.dat'))
        t = read_tetra(os.path.join(dat_root, f'fecr_{tag}_tetra.dat'))
        rec = {'tag': tag, 'n_cr': row.n_cr, 'x_cr': row.x_cr,
               'tier': _assign_tier(row.n_cr), 'loose_thr': row.loose_thr}
        if h: rec.update({f'hydro_{k}': v for k, v in h.items()})
        if s: rec.update({f'shear_{k}': v for k, v in s.items()})
        if t: rec.update({f'tetra_{k}': v for k, v in t.items()})
        records.append(rec)
    return pd.DataFrame(records)