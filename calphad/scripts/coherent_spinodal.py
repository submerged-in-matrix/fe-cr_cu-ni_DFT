import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from pycalphad import Database, Model, variables as v

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
TDB_PATH = Path(__file__).resolve().parent.parent / "tdb" / "CrFeNb_Jacob2016.tdb"
ETA_CSV = Path(__file__).resolve().parent.parent / "results" / "eta_x.csv"
OUTPUT_PNG = Path(__file__).resolve().parent.parent / "results" / "coherent_spinodal.png"
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "results" / "coherent_spinodal.csv"

MODEL_FILE_FOR_TARGET = {
    "C11": ANALYSIS_DIR / "mlp_models_raw_no14.pkl",
    "C12": ANALYSIS_DIR / "mlp_models_raw_no14.pkl",
    "C44": ANALYSIS_DIR / "mlp_models_raw_full.pkl",
}

T_MIN, T_MAX, T_STEP = 298.15, 1100, 10
X_MIN, X_MAX, N_X = 0.02, 0.98, 400

GPA_TO_EV_A3 = 1.0  # placeholder; unit reconciliation handled explicitly below


class MLP:
    """Identical to ml/03_mlp.ipynb's MLP class — required so pickle.load()
    can reconstruct the trained model instances."""
    def __init__(self, hidden=(16, 16), lr=1e-2, n_epochs=3000, lam=1e-3, seed=42):
        self.hidden = hidden
        self.lr = lr
        self.n_epochs = n_epochs
        self.lam = lam
        self.seed = seed
        self.loss_history = []

    def _init_params(self, layer_sizes):
        rng = np.random.default_rng(self.seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2. / n_in), (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def _forward(self, X):
        A = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A[-1] @ W + b
            A.append(np.tanh(Z) if i < len(self.W) - 1 else Z)
        return A

    def predict(self, X):
        Xn = (X - self.X_mean) / self.X_std
        A = self._forward(Xn)
        return A[-1].flatten() * self.y_std + self.y_mean


def load_mlp_model(pkl_path: Path, target: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and target in data:
        models = data
    elif isinstance(data, dict) and "mlp_models" in data:
        models = data["mlp_models"]
    else:
        raise KeyError(f"Unexpected structure in {pkl_path.name}: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    return models[target]


def get_Cij(x_grid):
    X = x_grid.reshape(-1, 1)
    Cij = {}
    for target, pkl_path in MODEL_FILE_FOR_TARGET.items():
        model = load_mlp_model(pkl_path, target)
        Cij[target] = model.predict(X)
    return Cij


def get_d2G_dx2_function():
    dbf = Database(str(TDB_PATH))
    mod = Model(dbf, ["CR", "FE", "VA"], "BCC_A2")
    GM = mod.GM

    x = sympy.Symbol("x_cr", positive=True)
    T_plain = sympy.Symbol("T_plain", positive=True)

    y_cr = v.Y("BCC_A2", 0, "CR")
    y_fe = v.Y("BCC_A2", 0, "FE")
    y_va = v.Y("BCC_A2", 1, "VA")

    G_x = GM.subs({y_cr: x, y_fe: 1 - x, y_va: 1})
    d2G_dx2 = sympy.diff(G_x, x, 2)
    d2G_dx2_plain = d2G_dx2.subs({v.T: T_plain})

    return sympy.lambdify((x, T_plain), d2G_dx2_plain, "numpy")


def compute_Y(C11, C12, C44):
    """Cahn's elastic energy coefficient for a cubic crystal.
    Returns Y for whichever of <100>/<111> is elastically softer,
    determined per-point from the actual C11, C12, C44 values."""
    anisotropy_lhs = C11 - C12
    anisotropy_rhs = 2 * C44
    soft_is_100 = anisotropy_lhs < anisotropy_rhs

    Y_100 = C11 + C12 - 2 * C12**2 / C11
    Y_111 = 6 * C44 * (C11 + 2 * C12) / (C11 + 2 * C12 + 4 * C44)

    Y = np.where(soft_is_100, Y_100, Y_111)
    direction = np.where(soft_is_100, "100", "111")
    return Y, direction


def find_zero_crossings(x, y):
    """Return x-values where y crosses zero, via linear interpolation between grid points."""
    signs = np.sign(y)
    crossing_idx = np.where(np.diff(signs) != 0)[0]
    roots = []
    for i in crossing_idx:
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        root = x0 - y0 * (x1 - x0) / (y1 - y0)
        roots.append(root)
    return roots


def run():
    x_grid = np.linspace(X_MIN, X_MAX, N_X)
    T_grid = np.arange(T_MIN, T_MAX + T_STEP, T_STEP)

    print("Loading elastic constants from ML surrogate...")
    Cij = get_Cij(x_grid)
    C11, C12, C44 = Cij["C11"], Cij["C12"], Cij["C44"]

    Y, direction = compute_Y(C11, C12, C44)
    n_100 = np.sum(direction == "100")
    print(f"Soft direction: <100> at {n_100}/{len(direction)} points, "
          f"<111> at {len(direction) - n_100}/{len(direction)} points")

    print("Loading eta(x) and a(x)...")
    eta_df = pd.read_csv(ETA_CSV)
    eta_interp = np.interp(x_grid, eta_df["x_cr"], eta_df["eta"])
    a_interp_angstrom = np.interp(x_grid, eta_df["x_cr"], eta_df["a_angstrom"])

    # Cahn coherency term: 2*eta^2*Y, converted from GPa to J/mol via the
    # actual composition-dependent molar volume (not a fixed constant).
    a_interp_m = a_interp_angstrom * 1e-10
    n_atoms_per_cell = 16
    N_A = 6.02214076e23
    V_m = N_A * (a_interp_m**3) / n_atoms_per_cell  # m^3/mol, per composition

    coherency_term_Pa = 2 * eta_interp**2 * (Y * 1e9)  # GPa -> Pa
    coherency_term_J_per_mol = coherency_term_Pa * V_m  # Pa * m^3/mol = J/mol

    print("Computing d2G/dx2 from CALPHAD...")
    d2G_func = get_d2G_dx2_function()

    print("Finding spinodal boundaries...")
    chemical_spinodal_pts = []
    coherent_spinodal_pts = []

    for T in T_grid:
        d2G = d2G_func(x_grid, T)
        chem_roots = find_zero_crossings(x_grid, d2G)
        for r in chem_roots:
            chemical_spinodal_pts.append((r, T))

        d2G_coherent = d2G + coherency_term_J_per_mol
        coh_roots = find_zero_crossings(x_grid, d2G_coherent)
        for r in coh_roots:
            coherent_spinodal_pts.append((r, T))

    chem_df = pd.DataFrame(chemical_spinodal_pts, columns=["x_cr", "T_K"])
    coh_df = pd.DataFrame(coherent_spinodal_pts, columns=["x_cr", "T_K"])
    chem_df["type"] = "chemical_spinodal"
    coh_df["type"] = "coherent_spinodal"
    out_df = pd.concat([chem_df, coh_df], ignore_index=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")

    plot_results(chem_df, coh_df, x_grid, Y, direction, sigma_mask_range=(0.38, 0.55))


def plot_results(chem_df, coh_df, x_grid, Y, direction, sigma_mask_range=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    for label, df, color in [("Chemical spinodal", chem_df, "tab:blue"),
                              ("Coherent spinodal (Cahn)", coh_df, "tab:red")]:
        if sigma_mask_range is not None:
            x_lo, x_hi = sigma_mask_range
            df = df[~((df["x_cr"] > x_lo) & (df["x_cr"] < x_hi))]
        low = df[df["x_cr"] < 0.5].sort_values("T_K")
        high = df[df["x_cr"] >= 0.5].sort_values("T_K")
        for i, branch in enumerate([low, high]):
            if branch.empty:
                continue
            ax.plot(branch["x_cr"] * 100, branch["T_K"], "-", color=color,
                    linewidth=2, label=label if i == 0 else None)

    if sigma_mask_range is not None:
        ax.axvspan(sigma_mask_range[0] * 100, sigma_mask_range[1] * 100,
                   color="gray", alpha=0.15,
                   label="Sigma-dominated (BCC metastable, excluded)")

    ax.set_xlabel("at.% Cr")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Chemical vs coherent (Cahn) spinodal\nBCC Fe-Cr")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)

    ax2 = axes[1]
    colors = np.where(direction == "100", "tab:green", "tab:purple")
    ax2.scatter(x_grid * 100, Y, c=colors, s=8)
    ax2.set_xlabel("at.% Cr")
    ax2.set_ylabel("Y (GPa)")
    ax2.set_title("Cahn elastic coefficient Y(x)\n(green=<100> soft, purple=<111> soft)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    run()