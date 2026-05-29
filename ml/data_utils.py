"""
data_utils.py
=============
Shared data loading and preprocessing for all ML surrogate notebooks.
Import at the top of each notebook:
    from data_utils import load_data, LOO_evaluate, TARGETS, PALETTE

Pipeline: DFT extraction → [this module] → GP / XGBoost / MLP → FEM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
import os

# ── Constants shared across all notebooks ─────────────────────────────────────
TARGETS   = ['C11', 'C12', 'C44']
PALETTE   = {'C11': 'steelblue', 'C12': 'darkorange', 'C44': 'seagreen'}
N_ATOMS   = 16          # atoms in BCC supercell
EPS_STRAIN = 0.01       # strain magnitude used in DFT

# Noise levels for GP (not used by XGB/MLP, but stored here for reference)
NOISE_NORMAL  = 1.0     # GPa — well-converged points
NOISE_FLAGGED = 10.0    # GPa — loose_thr points (fe02cr14/15/16)


def load_data(csv_path: str) -> dict:
    """
    Load elastic constants CSV and return a dict with all arrays
    needed by every model notebook.

    Expected CSV columns: tag, n_cr, C11, C12, C44, B, loose_thr

    Returns
    -------
    d : dict with keys:
        df          - full DataFrame
        X           - (17,1) Cr fraction, training feature
        X_pred      - (200,1) dense grid for plotting
        x_pred_atoms- (200,) same in atom units (for x-axis)
        targets     - dict {name: np.array} for C11, C12, C44
        alpha       - (17,) GP noise variance per point
        flagged     - bool array, True for loose_thr points
    """
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required = ['tag', 'n_cr', 'C11', 'C12', 'C44']
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"CSV missing columns: {missing}"

    df['x_cr'] = df['n_cr'] / float(N_ATOMS)

    if 'loose_thr' not in df.columns:
        print("WARNING: 'loose_thr' column not found — assuming all points well-converged")
        df['loose_thr'] = False
    df['loose_thr'] = df['loose_thr'].astype(bool)

    if 'B' not in df.columns:
        df['B'] = (df['C11'] + 2*df['C12']) / 3.0

    X          = df['x_cr'].values.reshape(-1, 1)
    X_pred     = np.linspace(0, 1, 200).reshape(-1, 1)
    x_pred_atoms = (X_pred * N_ATOMS).flatten()
    flagged    = df['loose_thr'].values
    alpha      = np.where(flagged, NOISE_FLAGGED**2, NOISE_NORMAL**2)
    tgt        = {t: df[t].values.astype(float) for t in TARGETS}

    print(f"Loaded {len(df)} points | flagged (loose_thr): {flagged.sum()}")
    print(df[['tag','n_cr','C11','C12','C44','B','loose_thr']].to_string(index=False))

    return dict(df=df, X=X, X_pred=X_pred, x_pred_atoms=x_pred_atoms,
                targets=tgt, alpha=alpha, flagged=flagged)


def LOO_evaluate(model_fn, X: np.ndarray, y: np.ndarray,
                 extra_kw: dict = None) -> dict:
    """
    Generic Leave-One-Out cross-validation.

    Parameters
    ----------
    model_fn : callable
        Takes (X_train, y_train, **extra_kw) → fitted model object
        The model must have a .predict(X) method.
    X        : (n,1) feature array
    y        : (n,)  target array
    extra_kw : dict passed to model_fn each fold (e.g. alpha for GP)

    Returns
    -------
    dict with keys: loo_true, loo_preds, mae, r2, residuals
    """
    if extra_kw is None:
        extra_kw = {}
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr, te in loo.split(X):
        kw = {k: v[tr] if hasattr(v, '__len__') and len(v) == len(X) else v
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
                       flagged: np.ndarray, title: str):
    """Reusable residual bar chart for any model."""
    cols = ['tomato' if f else 'steelblue' for f in flagged]
    ax.bar(n_cr, residuals, color=cols, edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='k', linewidth=1)
    ax.set_xlabel('Cr atoms (out of 16)')
    ax.set_ylabel('Residual (GPa)')
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')


def read_hydro(path: str) -> dict:
    """
    Read a hydro .dat file.

    Format (3 rows, 2 columns):
        eps   P_kbar
        -0.01  65.57
         0.00   2.84
         0.01 -51.43

    Returns dict with keys: P_minus, P_zero, P_plus  (kbar)
    or None if file missing.
    """
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path)          # shape (3, 2)
    # Sort by eps column so row order doesn't matter
    data = data[data[:, 0].argsort()]
    # rows: ε=-0.01 (compression), ε=0.00 (ref), ε=+0.01 (tension)
    return {'P_minus': data[0, 1],
            'P_zero':  data[1, 1],
            'P_plus':  data[2, 1]}


def read_shear(path: str) -> dict:
    """
    Read a shear .dat file.

    Format (3 rows, 2 columns):
        eps   S12_kbar
        -0.01   20.08
         0.00    0.00
         0.01  -20.13

    Returns dict with keys: S12_minus, S12_zero, S12_plus  (kbar)
    or None if file missing.
    """
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path)
    data = data[data[:, 0].argsort()]
    return {'S12_minus': data[0, 1],
            'S12_zero':  data[1, 1],
            'S12_plus':  data[2, 1]}


def read_tetra(path: str) -> dict:
    """
    Read a tetra .dat file.

    Format (3 rows, 4 columns):
        eps   S11_kbar   S33_kbar   DS_kbar   (DS = S33 - S11)
        -0.01  -21.69    35.91      57.60
         0.00    1.12     1.12       0.00
         0.01   18.33   -37.86     -56.19

    Returns dict with keys: S11_minus/zero/plus, S33_minus/zero/plus,
                            DS_minus/zero/plus  (all kbar)
    or None if file missing.
    """
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path)
    data = data[data[:, 0].argsort()]
    return {'S11_minus': data[0, 1], 'S11_zero': data[1, 1], 'S11_plus': data[2, 1],
            'S33_minus': data[0, 2], 'S33_zero': data[1, 2], 'S33_plus': data[2, 2],
            'DS_minus':  data[0, 3], 'DS_zero':  data[1, 3], 'DS_plus':  data[2, 3]}


def load_asymmetry(df: pd.DataFrame, dat_root: str) -> pd.DataFrame:
    """
    Build tension-compression asymmetry DataFrame from hydro .dat files.

    Actual file layout (flat directory, no subdirs):
        dat_root/fecr_{tag}_hydro.dat
        dat_root/fecr_{tag}_shear.dat
        dat_root/fecr_{tag}_tetra.dat

    e.g.  dft_data/fecr_fe16cr00_hydro.dat

    Asymmetry index (hydrostatic):
        asym = P(+ε) + P(−ε) − 2·P(0)
        = 0 for a perfectly harmonic solid
        ≠ 0 indicates anharmonicity

    Parameters
    ----------
    df       : DataFrame from load_data (needs 'tag', 'n_cr', 'x_cr', 'loose_thr')
    dat_root : path to the flat dft_data/ directory

    Returns
    -------
    asym_df : DataFrame with columns:
              tag, n_cr, x_cr, P_minus, P_zero, P_plus,
              asymmetry_kbar, loose_thr
    """
    records = []
    for row in df.itertuples():
        tag  = row.tag
        path = os.path.join(dat_root, f'fecr_{tag}_hydro.dat')
        h    = read_hydro(path)
        if h is None:
            print(f'  SKIP {tag}: file not found — {path}')
            continue
        asym = h['P_plus'] + h['P_minus'] - 2.0 * h['P_zero']
        records.append({
            'tag':             tag,
            'n_cr':            row.n_cr,
            'x_cr':            row.x_cr,
            'P_minus':         h['P_minus'],
            'P_zero':          h['P_zero'],
            'P_plus':          h['P_plus'],
            'asymmetry_kbar':  asym,
            'loose_thr':       row.loose_thr
        })
    return pd.DataFrame(records)


def load_all_dat(df: pd.DataFrame, dat_root: str) -> pd.DataFrame:
    """
    Load all three strain types for all tags into one wide DataFrame.
    Useful for independent elastic constant verification or further analysis.

    Returns one row per tag with all raw stress values (kbar).
    """
    records = []
    for row in df.itertuples():
        tag = row.tag
        h = read_hydro(os.path.join(dat_root, f'fecr_{tag}_hydro.dat'))
        s = read_shear(os.path.join(dat_root, f'fecr_{tag}_shear.dat'))
        t = read_tetra(os.path.join(dat_root, f'fecr_{tag}_tetra.dat'))
        rec = {'tag': row.tag, 'n_cr': row.n_cr, 'x_cr': row.x_cr,
               'loose_thr': row.loose_thr}
        if h: rec.update({f'hydro_{k}': v for k, v in h.items()})
        if s: rec.update({f'shear_{k}': v for k, v in s.items()})
        if t: rec.update({f'tetra_{k}': v for k, v in t.items()})
        records.append(rec)
    return pd.DataFrame(records)
