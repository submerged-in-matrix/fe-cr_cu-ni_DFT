"""
strain_distribution_check.py
=============================
Diagnostic plots for all 17 Fe-Cr tags across all strain types.

Figure structure
----------------
Figure 1 — Hydrostatic zero-strain:
    Panel 1: C11+2C12 elastic modulus vs n_cr  (from CSV)
    Panel 2: P(ε=0.00) equilibrium pressure vs n_cr  (from hydro.dat)

Figure 2 — Hydrostatic symmetric strains:
    Panel 1: P at ε=−0.01, 0.00, +0.01 for all 17 tags
    Panel 2: P(+ε)−P(−ε) bar — linearity check

Figure 3 — Tetragonal:
    Row 1: S11 symmetric strains | S11(ε=0) residual
    Row 2: S33 symmetric strains | S33(ε=0) residual

Figure 4 — Shear:
    Panel 1: S12 symmetric strains
    Panel 2: S12(ε=0) — should be ≈0 for cubic symmetry

Run from ml/ directory:
    python3 strain_distribution_check.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DAT_ROOT  = '../dft_data'
CSV_PATH  = '../analysis/elastic_constants_fecr_corrected.csv'
OUT_DIR   = '../analysis'
HIGHLIGHT = {'fe10cr06', 'fe09cr07', 'fe02cr14'}
TIER_COL  = {'A': 'steelblue', 'B': 'goldenrod', 'C': 'tomato'}
TIER_MARK = {'A': 'o',         'B': 's',          'C': 'D'}

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['tier'] = np.where(df['n_cr'] <= 9, 'A',
             np.where(df['n_cr'] <= 13, 'B', 'C'))

def read_dat(path):
    if not os.path.exists(path):
        return None
    try:
        d = np.loadtxt(path)
        return d[d[:, 0].argsort()]
    except Exception:
        return None

def build_table(stype, col_indices, col_names):
    rows = []
    for r in df.itertuples():
        path = os.path.join(DAT_ROOT, f'fecr_{r.tag}_{stype}.dat')
        d = read_dat(path)
        if d is None:
            print(f'  MISSING: fecr_{r.tag}_{stype}.dat')
            continue
        for row in d:
            rec = {'tag': r.tag, 'n_cr': r.n_cr, 'tier': r.tier,
                   'eps': row[0]}
            for ci, cn in zip(col_indices, col_names):
                rec[cn] = row[ci]
            rows.append(rec)
    return pd.DataFrame(rows)

print('Reading .dat files...')
hyd = build_table('hydro', [1],       ['P'])
tet = build_table('tetra', [1, 2, 3], ['S11', 'S33', 'DS'])
shr = build_table('shear', [1],       ['S12'])
print(f'  hydro: {hyd["tag"].nunique()} tags | tetra: {tet["tag"].nunique()} | shear: {shr["tag"].nunique()}\n')

# ── Helpers ───────────────────────────────────────────────────────────────────

def tier_legend(ax):
    handles = [Line2D([0],[0], color=TIER_COL[t], marker=TIER_MARK[t],
                      lw=0, markersize=7, label=f'Tier {t}')
               for t in ['A','B','C']]
    ax.legend(handles=handles, fontsize=8, loc='best')


def plot_zero_strain(ax, table, col, ylabel, title):
    """Single curve: stress at ε=0 for all 17 tags."""
    sub = table[np.isclose(table['eps'], 0.00, atol=1e-4)].sort_values('n_cr')
    ax.plot(sub['n_cr'], sub[col], '-', color='gray', lw=0.8, alpha=0.5, zorder=1)
    for _, row in sub.iterrows():
        col_c = TIER_COL[row['tier']]
        ax.scatter(row['n_cr'], row[col], color=col_c,
                   marker=TIER_MARK[row['tier']], s=55, zorder=3)
        if row['tag'] in HIGHLIGHT:
            ax.annotate(row['tag'], xy=(row['n_cr'], row[col]),
                        xytext=(4,4), textcoords='offset points',
                        fontsize=7, color=col_c)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('Cr atoms (out of 16)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.5, 16.5)
    tier_legend(ax)


def plot_three_strains(ax, table, col, ylabel, title):
    """Three curves: ε=−0.01, 0.00, +0.01 overlaid, coloured by tier."""
    styles = {-0.01: ('-',  0.90, 'ε=−0.01'),
               0.00: ('--', 0.50, 'ε=0.00 ref'),
               0.01: ('-',  0.90, 'ε=+0.01')}
    for eps, (ls, alpha, label) in styles.items():
        sub = table[np.isclose(table['eps'], eps, atol=1e-4)].sort_values('n_cr')
        ax.plot(sub['n_cr'], sub[col], ls=ls, color='gray',
                lw=0.8, alpha=0.35, zorder=1)
        for _, row in sub.iterrows():
            col_c = TIER_COL[row['tier']]
            ax.scatter(row['n_cr'], row[col], color=col_c,
                       marker=TIER_MARK[row['tier']], s=35,
                       alpha=alpha, zorder=3)
            if row['tag'] in HIGHLIGHT and eps != 0.00:
                ax.annotate(row['tag'], xy=(row['n_cr'], row[col]),
                            xytext=(4,3), textcoords='offset points',
                            fontsize=6.5, color=col_c)
        ax.plot([], [], ls=ls, color='gray', lw=1.5, label=label)
    ax.axhline(0, color='k', lw=0.5, ls=':', alpha=0.3)
    ax.set_xlabel('Cr atoms (out of 16)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.5, 16.5)
    # Combined legend: strain lines + tier markers
    strain_h = [Line2D([0],[0], ls=ls, color='gray', lw=1.5, label=lbl)
                for _, (ls, _, lbl) in styles.items()]
    tier_h   = [Line2D([0],[0], color=TIER_COL[t], marker=TIER_MARK[t],
                       lw=0, markersize=7, label=f'Tier {t}')
                for t in ['A','B','C']]
    ax.legend(handles=strain_h + tier_h, fontsize=7, loc='best')


def plot_range_bar(ax, table, col, ylabel, title, op='diff'):
    """Bar: P(+ε)−P(−ε) or S(+ε)+S(−ε) per tag."""
    tags   = df['tag'].values
    ncrs   = df['n_cr'].values
    tiers  = df['tier'].values
    vals   = []
    for tag in tags:
        sub = table[table['tag'] == tag]
        pm  = sub[np.isclose(sub['eps'], -0.01, atol=1e-4)][col].values
        pp  = sub[np.isclose(sub['eps'],  0.01, atol=1e-4)][col].values
        if len(pm) and len(pp):
            v = float(pp[0]) - float(pm[0]) if op == 'diff' else float(pp[0]) + float(pm[0])
        else:
            v = np.nan
        vals.append(v)
    vals = np.array(vals)
    cols_b = [TIER_COL[t] for t in tiers]
    bars = ax.bar(ncrs, vals, color=cols_b, edgecolor='k', lw=0.5, width=0.7)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    for i, (tag, v) in enumerate(zip(tags, vals)):
        if tag in HIGHLIGHT and not np.isnan(v):
            ax.annotate(tag, xy=(ncrs[i], v),
                        xytext=(0, 4), textcoords='offset points',
                        fontsize=7, ha='center', color=TIER_COL[tiers[i]])
    ax.set_xlabel('Cr atoms (out of 16)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, axis='y')
    tier_h = [Patch(color=TIER_COL[t], label=f'Tier {t}') for t in ['A','B','C']]
    ax.legend(handles=tier_h, fontsize=8)


# ── Figure 1: Hydrostatic zero-strain ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: C11+2C12 from CSV
sub0 = df.sort_values('n_cr')
axes[0].plot(sub0['n_cr'], sub0['C11']+2*sub0['C12'],
             '-', color='gray', lw=0.8, alpha=0.5, zorder=1)
for _, row in sub0.iterrows():
    col_c = TIER_COL[row['tier']]
    axes[0].scatter(row['n_cr'], row['C11']+2*row['C12'],
                    color=col_c, marker=TIER_MARK[row['tier']], s=55, zorder=3)
    if row['tag'] in HIGHLIGHT:
        axes[0].annotate(row['tag'],
                         xy=(row['n_cr'], row['C11']+2*row['C12']),
                         xytext=(4,4), textcoords='offset points',
                         fontsize=7, color=col_c)
axes[0].set_xlabel('Cr atoms (out of 16)')
axes[0].set_ylabel('C11+2C12 (GPa)')
axes[0].set_title('Elastic modulus C11+2·C12\n(extracted from DFT stress)', fontsize=10)
axes[0].grid(alpha=0.25); axes[0].set_xlim(-0.5, 16.5)
tier_legend(axes[0])

# Panel 2: P(ε=0) from hydro.dat
plot_zero_strain(axes[1], hyd, 'P',
                 'P (kbar)',
                 'Equilibrium pressure P(ε=0.00)\nshould be ≈0 after vc-relax')

plt.suptitle('Figure 1 — Hydrostatic: Elastic Modulus vs Equilibrium Pressure',
             fontweight='bold')
plt.tight_layout()
out = os.path.join(OUT_DIR, 'strain_diag_fig1_hydro_zero.png')
plt.savefig(out, bbox_inches='tight', dpi=130); plt.show()
print(f'Saved: {out}')


# ── Figure 2: Hydrostatic symmetric strains ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

plot_three_strains(axes[0], hyd, 'P', 'P (kbar)',
                   'Hydrostatic P: all three strain levels\n(ε=−0.01, 0.00, +0.01)')

plot_range_bar(axes[1], hyd, 'P',
               'P(+ε) − P(−ε)  (kbar)',
               'Hydrostatic: stress range P(+ε)−P(−ε)\nuniform → good linearity',
               op='diff')

plt.suptitle('Figure 2 — Hydrostatic: Symmetric Strains', fontweight='bold')
plt.tight_layout()
out = os.path.join(OUT_DIR, 'strain_diag_fig2_hydro_strains.png')
plt.savefig(out, bbox_inches='tight', dpi=130); plt.show()
print(f'Saved: {out}')


# ── Figure 3: Tetragonal ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

plot_three_strains(axes[0,0], tet, 'S11', 'S11 (kbar)',
                   'Tetragonal S11: all three strain levels')
plot_zero_strain(axes[0,1], tet, 'S11', 'S11 (kbar)',
                 'Tetragonal S11(ε=0.00)\nresidual stress — should be ≈0')

plot_three_strains(axes[1,0], tet, 'S33', 'S33 (kbar)',
                   'Tetragonal S33: all three strain levels')
plot_zero_strain(axes[1,1], tet, 'S33', 'S33 (kbar)',
                 'Tetragonal S33(ε=0.00)\nresidual stress — should be ≈0')

plt.suptitle('Figure 3 — Tetragonal: S11 and S33', fontweight='bold')
plt.tight_layout()
out = os.path.join(OUT_DIR, 'strain_diag_fig3_tetra.png')
plt.savefig(out, bbox_inches='tight', dpi=130); plt.show()
print(f'Saved: {out}')


# ── Figure 4: Shear ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

plot_three_strains(axes[0], shr, 'S12', 'S12 (kbar)',
                   'Shear S12: all three strain levels\nantisymmetric → S12(+ε)≈−S12(−ε)')

plot_zero_strain(axes[1], shr, 'S12', 'S12 (kbar)',
                 'Shear S12(ε=0.00)\nshould be ≈0 — non-zero = cubic symmetry broken')

plt.suptitle('Figure 4 — Shear: S12', fontweight='bold')
plt.tight_layout()
out = os.path.join(OUT_DIR, 'strain_diag_fig4_shear.png')
plt.savefig(out, bbox_inches='tight', dpi=130); plt.show()
print(f'Saved: {out}')

print('\nAll figures saved. Key checks:')
print('  Fig 1 right : P(ε=0) near zero → vc-relax geometry consistent with strain SCFs')
print('  Fig 2 right : uniform bar heights → linear elastic response')
print('  Fig 3 right : S11/S33 at ε=0 near zero → no residual stress from symmetrisation')
print('  Fig 4 right : S12(ε=0) near zero → cubic symmetry intact after enforcement')