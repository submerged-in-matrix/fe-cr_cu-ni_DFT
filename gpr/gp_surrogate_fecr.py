"""
GP Surrogate Model — Fe-Cr Elastic Constants
=============================================
Trains 3 independent Gaussian Process Regressors:
    C11(x_Cr), C12(x_Cr), C44(x_Cr)

Features  : x_Cr (Cr fraction, 0.0 – 1.0)
Targets   : C11, C12, C44 in GPa
Weighting : 3 flagged tags (fe02cr14, fe01cr15, fe00cr16) get higher noise
Validation: Leave-One-Out Cross-Validation (LOO-CV)
Kernels   : RBF vs Matern-5/2 comparison

Output files:
    gp_predictions.csv       — GP mean + std at fine grid
    gp_loo_residuals.csv     — LOO-CV residuals per tag
    fig_gp_fit.png           — GP fit with uncertainty bands
    fig_loo_residuals.png    — LOO residual plots
    fig_zener_anisotropy.png — Zener A vs x_Cr
    fig_directional_modulus.png — E[100], E[110], E[111] vs x_Cr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── 0. Data ───────────────────────────────────────────────────────────────────
# Paste your elastic_constants_fecr.csv path here when running on your machine
# For now: hardcoded from session_summary_may26_2026.md

data = {
    'tag':  ['fe16cr00','fe15cr01','fe14cr02','fe13cr03','fe12cr04',
             'fe11cr05','fe10cr06','fe09cr07','fe08cr08','fe07cr09',
             'fe06cr10','fe05cr11','fe04cr12','fe03cr13',
             'fe02cr14','fe01cr15','fe00cr16'],
    'n_Cr': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
    'C11':  [286.68,255.48,237.92,180.60,170.43,190.92,266.25,231.10,
             265.07,294.18,342.18,353.27,358.17,362.80,405.17,368.02,384.65],
    'C12':  [172.28,156.03,147.22,112.82,106.68,112.24,168.25,139.43,
             171.92,185.96,213.16,204.69,196.89,166.45,178.34,108.97,100.18],
    'C44':  [119.95,119.55,126.65,110.95,107.20,111.20,119.98,119.18,
             121.45,123.40,127.28,120.20,112.65,110.10,106.75,100.68,100.53],
    'flag': [False]*14 + [True, True, True]   # last 3 are uncertain
}

df = pd.DataFrame(data)
df['x_Cr'] = df['n_Cr'] / 16.0   # Cr fraction 0–1

# ── 1. Setup ──────────────────────────────────────────────────────────────────
X = df['x_Cr'].values.reshape(-1, 1)
targets = ['C11', 'C12', 'C44']
colors  = {'C11': '#2196F3', 'C12': '#E91E63', 'C44': '#4CAF50'}

# Noise per point: flagged points get 10x higher noise (uncertain convergence)
# Units: GPa. Unflagged ~2 GPa noise, flagged ~20 GPa noise
# NOTE: these noise values are physically motivated guesses, not rigorous.
#       Adjust based on your convergence quality assessment.
noise_base    = 2.0   # GPa — unflagged
noise_flagged = 20.0  # GPa — flagged (conv_thr=1e-5, mixed magnetic)
noise_vec = np.where(df['flag'], noise_flagged, noise_base)

# ── 2. Kernel comparison ──────────────────────────────────────────────────────
def make_kernel(kind='matern'):
    """Build kernel: amplitude * shape + white noise (fit per point via alpha)"""
    amplitude = C(100.0, (1.0, 1e5))
    if kind == 'rbf':
        shape = RBF(length_scale=0.3, length_scale_bounds=(0.05, 2.0))
    else:  # matern-5/2
        shape = Matern(length_scale=0.3, length_scale_bounds=(0.05, 2.0), nu=2.5)
    return amplitude * shape

def fit_gp(X, y, noise_vec, kernel):
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise_vec**2,   # per-point noise variance
        n_restarts_optimizer=10,
        normalize_y=True
    )
    gp.fit(X, y)
    return gp

# ── 3. LOO-CV ─────────────────────────────────────────────────────────────────
def loo_cv(X, y, noise_vec, kernel_fn):
    loo = LeaveOneOut()
    y_pred_loo = np.zeros(len(y))
    y_std_loo  = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        gp = fit_gp(X[train_idx], y[train_idx],
                    noise_vec[train_idx], kernel_fn())
        mu, sigma = gp.predict(X[test_idx], return_std=True)
        y_pred_loo[test_idx] = mu
        y_std_loo[test_idx]  = sigma
    residuals = y - y_pred_loo
    mae  = mean_absolute_error(y, y_pred_loo)
    rmse = np.sqrt(mean_squared_error(y, y_pred_loo))
    return y_pred_loo, y_std_loo, residuals, mae, rmse

# ── 4. Train final models + run LOO ───────────────────────────────────────────
X_fine = np.linspace(0, 1, 200).reshape(-1, 1)
results = {}
loo_results = {}

print("=" * 60)
print("GP Surrogate — Fe-Cr Elastic Constants")
print("=" * 60)

for target in targets:
    y = df[target].values

    # Compare kernels via LOO MAE
    for kname in ['matern', 'rbf']:
        _, _, _, mae, rmse = loo_cv(X, y, noise_vec, lambda k=kname: make_kernel(k))
        print(f"{target} | kernel={kname:6s} | LOO MAE={mae:.2f} GPa | LOO RMSE={rmse:.2f} GPa")

    # Use Matern-5/2 as final (smoother, physically reasonable for composition dependence)
    # UNCERTAINTY NOTE: kernel choice affects prediction bands. Matern-5/2 is
    # a common default for physical property interpolation but is not uniquely correct.
    gp = fit_gp(X, y, noise_vec, make_kernel('matern'))
    mu_fine, std_fine = gp.predict(X_fine, return_std=True)

    y_loo, std_loo, resid, mae, rmse = loo_cv(X, y, noise_vec,
                                               lambda: make_kernel('matern'))

    results[target] = {'gp': gp, 'mu': mu_fine, 'std': std_fine}
    loo_results[target] = {'y_loo': y_loo, 'std_loo': std_loo,
                           'residuals': resid, 'mae': mae, 'rmse': rmse}
    print(f"  → Final Matern: LOO MAE={mae:.2f} GPa, RMSE={rmse:.2f} GPa")
    print(f"     Optimised kernel: {gp.kernel_}")
    print()

# ── 5. Derived quantities ─────────────────────────────────────────────────────
C11_f = results['C11']['mu']
C12_f = results['C12']['mu']
C44_f = results['C44']['mu']

# Zener anisotropy ratio A = 2*C44 / (C11 - C12)
# A=1 → isotropic; A>1 → stiffer along <111>
zener_A = 2 * C44_f / (C11_f - C12_f)

# Directional Young's modulus for cubic crystal (closed-form)
# 1/E[hkl] = S11 - 2*(S11-S12-S44/2) * (l1²l2² + l2²l3² + l3²l1²)
# where l1,l2,l3 are direction cosines, Sij = compliance tensor
# [100]: li² products = 0  → 1/E = S11
# [110]: li² products = 1/4 → 1/E = S11 - (S11-S12-S44/2)/2
# [111]: li² products = 1/3 → 1/E = S11 - 2/3*(S11-S12-S44/2)
# NOTE: This derivation is standard for cubic symmetry.
#       Verify against Nye "Physical Properties of Crystals" if citing.

def cubic_compliance(C11, C12, C44):
    """Returns S11, S12, S44 from Cij via matrix inversion (Voigt notation)"""
    # 3x3 sub-block for cubic: [[C11,C12,C12],[C12,C11,C12],[C12,C12,C11]]
    Cm = np.array([[C11,C12,C12],[C12,C11,C12],[C12,C12,C11]])
    Sm = np.linalg.inv(Cm)
    S11 = Sm[0,0]
    S12 = Sm[0,1]
    S44 = 1.0/C44   # shear compliance
    return S11, S12, S44

E100 = np.zeros(len(X_fine))
E110 = np.zeros(len(X_fine))
E111 = np.zeros(len(X_fine))

for i, (c11, c12, c44) in enumerate(zip(C11_f, C12_f, C44_f)):
    S11, S12, S44 = cubic_compliance(c11, c12, c44)
    aniso = S11 - S12 - S44/2
    E100[i] = 1.0 / S11
    E110[i] = 1.0 / (S11 - aniso/2)
    E111[i] = 1.0 / (S11 - 2*aniso/3)

# ── 6. Save CSVs ──────────────────────────────────────────────────────────────
x_cr_pct = X_fine.flatten() * 100

pred_df = pd.DataFrame({
    'x_Cr_fraction': X_fine.flatten(),
    'x_Cr_percent':  x_cr_pct,
    'C11_mean_GPa':  results['C11']['mu'],
    'C11_std_GPa':   results['C11']['std'],
    'C12_mean_GPa':  results['C12']['mu'],
    'C12_std_GPa':   results['C12']['std'],
    'C44_mean_GPa':  results['C44']['mu'],
    'C44_std_GPa':   results['C44']['std'],
    'Zener_A':       zener_A,
    'E100_GPa':      E100,
    'E110_GPa':      E110,
    'E111_GPa':      E111,
})
pred_df.to_csv('/home/claude/gp_predictions.csv', index=False)

loo_df = pd.DataFrame({
    'tag':     df['tag'],
    'x_Cr':    df['x_Cr'],
    'flag':    df['flag'],
    'C11_true': df['C11'], 'C11_loo': loo_results['C11']['y_loo'],
    'C11_res':  loo_results['C11']['residuals'],
    'C12_true': df['C12'], 'C12_loo': loo_results['C12']['y_loo'],
    'C12_res':  loo_results['C12']['residuals'],
    'C44_true': df['C44'], 'C44_loo': loo_results['C44']['y_loo'],
    'C44_res':  loo_results['C44']['residuals'],
})
loo_df.to_csv('/home/claude/gp_loo_residuals.csv', index=False)

print("CSVs saved.")

# ── 7. Figure 1 — GP fits with uncertainty bands ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('GP Surrogate — Fe-Cr Elastic Constants (BCC, 16-atom)', fontsize=13)

x_plot = X_fine.flatten() * 100  # percent for x-axis

for ax, target in zip(axes, targets):
    mu  = results[target]['mu']
    std = results[target]['std']
    col = colors[target]

    ax.fill_between(x_plot, mu-2*std, mu+2*std, alpha=0.2, color=col, label='±2σ')
    ax.fill_between(x_plot, mu-std,   mu+std,   alpha=0.35, color=col, label='±1σ')
    ax.plot(x_plot, mu, color=col, lw=2, label='GP mean')

    # Data points — distinguish flagged
    mask_ok   = ~df['flag'].values
    mask_flag =  df['flag'].values
    ax.scatter(df['x_Cr'][mask_ok]*100,   df[target][mask_ok],
               color=col, s=60, zorder=5, label='DFT (good)')
    ax.scatter(df['x_Cr'][mask_flag]*100, df[target][mask_flag],
               color='gray', s=60, marker='x', zorder=5, linewidths=2,
               label='DFT (flagged)')

    mae  = loo_results[target]['mae']
    rmse = loo_results[target]['rmse']
    ax.set_title(f'{target}   LOO MAE={mae:.1f} GPa', fontsize=11)
    ax.set_xlabel('Cr content (at.%)')
    ax.set_ylabel('Elastic constant (GPa)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig('/home/claude/fig_gp_fit.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig_gp_fit.png saved.")

# ── 8. Figure 2 — LOO residuals ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('LOO-CV Residuals (true − predicted)', fontsize=13)

for ax, target in zip(axes, targets):
    resid = loo_results[target]['residuals']
    col   = colors[target]
    x_pct = df['x_Cr'].values * 100

    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.bar(x_pct[~df['flag'].values], resid[~df['flag'].values],
           width=3, color=col, alpha=0.7, label='good')
    ax.bar(x_pct[df['flag'].values],  resid[df['flag'].values],
           width=3, color='gray', alpha=0.7, label='flagged')

    mae = loo_results[target]['mae']
    ax.set_title(f'{target}  MAE={mae:.1f} GPa', fontsize=11)
    ax.set_xlabel('Cr content (at.%)')
    ax.set_ylabel('Residual (GPa)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fig_loo_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig_loo_residuals.png saved.")

# ── 9. Figure 3 — Zener anisotropy ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x_plot, zener_A, color='#FF9800', lw=2)
ax.axhline(1.0, color='k', lw=0.8, ls='--', label='Isotropic (A=1)')

# Propagate uncertainty: A = 2*C44 / (C11-C12)
# Rough uncertainty via GP std (first-order, ignores covariance between targets)
# UNCERTAINTY NOTE: proper uncertainty propagation here requires joint GP covariance
# across C11, C12, C44 — not implemented. These bands are indicative only.
dA_C44 = 2 / (C11_f - C12_f) * results['C44']['std']
dA_C11 =  2*C44_f / (C11_f-C12_f)**2 * results['C11']['std']
dA_C12 = -2*C44_f / (C11_f-C12_f)**2 * results['C12']['std']
std_A  = np.sqrt(dA_C44**2 + dA_C11**2 + dA_C12**2)

ax.fill_between(x_plot, zener_A-std_A, zener_A+std_A, alpha=0.2, color='#FF9800')

# Scatter data points
A_data = 2*df['C44'].values / (df['C11'].values - df['C12'].values)
ax.scatter(df['x_Cr']*100, A_data, color='#FF9800', s=60, zorder=5)
ax.scatter(df['x_Cr'][df['flag']]*100, A_data[df['flag']],
           color='gray', s=60, marker='x', zorder=6, linewidths=2)

ax.set_xlabel('Cr content (at.%)')
ax.set_ylabel('Zener anisotropy A = 2C₄₄/(C₁₁−C₁₂)')
ax.set_title('Elastic Anisotropy vs Cr Composition')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/fig_zener_anisotropy.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig_zener_anisotropy.png saved.")

# ── 10. Figure 4 — Directional Young's moduli ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x_plot, E100, lw=2, label='E[100]', color='#2196F3')
ax.plot(x_plot, E110, lw=2, label='E[110]', color='#9C27B0')
ax.plot(x_plot, E111, lw=2, label='E[111]', color='#F44336')

ax.set_xlabel('Cr content (at.%)')
ax.set_ylabel("Young's modulus (GPa)")
ax.set_title('Directional Young\'s Modulus vs Cr Composition')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/fig_directional_modulus.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig_directional_modulus.png saved.")

# ── 11. Summary print ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for target in targets:
    mae  = loo_results[target]['mae']
    rmse = loo_results[target]['rmse']
    print(f"{target}: LOO MAE = {mae:.2f} GPa | RMSE = {rmse:.2f} GPa")

print("\nZener A range (GP mean):")
print(f"  x_Cr=0%  → A = {zener_A[0]:.3f}")
print(f"  x_Cr=50% → A = {zener_A[100]:.3f}")
print(f"  x_Cr=100%→ A = {zener_A[-1]:.3f}")

print("\nE[100] range:", f"{E100.min():.1f} – {E100.max():.1f} GPa")
print("E[110] range:", f"{E110.min():.1f} – {E110.max():.1f} GPa")
print("E[111] range:", f"{E111.min():.1f} – {E111.max():.1f} GPa")
print("\nDone. Check /home/claude/ for output files.")
