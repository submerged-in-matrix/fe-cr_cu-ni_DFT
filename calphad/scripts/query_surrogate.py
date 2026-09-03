import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class MLP:
    """
    Two-hidden-layer MLP for scalar regression.
    Identical to the class defined in ml/03_mlp.ipynb — required here so
    pickle.load() can reconstruct the trained model instances.
    """
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
            if i < len(self.W) - 1:
                A.append(np.tanh(Z))
            else:
                A.append(Z)
        return A

    def _backward(self, A, y, sw):
        n = len(y)
        dWs, dbs = [], []
        pred = A[-1].flatten()
        delta = -2 * sw * (y - pred) / n
        delta = delta.reshape(-1, 1)
        for i in reversed(range(len(self.W))):
            dW = A[i].T @ delta + self.lam * self.W[i]
            db = delta.sum(axis=0)
            dWs.insert(0, dW)
            dbs.insert(0, db)
            if i > 0:
                delta = (delta @ self.W[i].T) * (1 - A[i] ** 2)
        return dWs, dbs

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        sw = sample_weight / sample_weight.sum() * len(y)
        self.X_mean, self.X_std = X.mean(), X.std() + 1e-8
        self.y_mean, self.y_std = y.mean(), y.std() + 1e-8
        Xn = (X - self.X_mean) / self.X_std
        yn = (y - self.y_mean) / self.y_std
        sizes = [1] + list(self.hidden) + [1]
        self._init_params(sizes)
        m = [np.zeros_like(W) for W in self.W]
        v = [np.zeros_like(W) for W in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        self.loss_history = []
        for epoch in range(1, self.n_epochs + 1):
            A = self._forward(Xn)
            pred = A[-1].flatten()
            loss = np.sum(sw * (yn - pred) ** 2) / len(yn) + \
                self.lam * sum(np.sum(W ** 2) for W in self.W)
            self.loss_history.append(loss)
            dWs, dbs = self._backward(A, yn, sw)
            for i in range(len(self.W)):
                m[i] = beta1 * m[i] + (1 - beta1) * dWs[i]
                v[i] = beta2 * v[i] + (1 - beta2) * dWs[i] ** 2
                mb[i] = beta1 * mb[i] + (1 - beta1) * dbs[i]
                vb[i] = beta2 * vb[i] + (1 - beta2) * dbs[i] ** 2
                mc = m[i] / (1 - beta1 ** epoch)
                vc = v[i] / (1 - beta2 ** epoch)
                mbc = mb[i] / (1 - beta1 ** epoch)
                vbc = vb[i] / (1 - beta2 ** epoch)
                self.W[i] -= self.lr * mc / (np.sqrt(vc) + eps_adam)
                self.b[i] -= self.lr * mbc / (np.sqrt(vbc) + eps_adam)
        return self

    def predict(self, X):
        Xn = (X - self.X_mean) / self.X_std
        A = self._forward(Xn)
        return A[-1].flatten() * self.y_std + self.y_mean

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
EQUILIBRIUM_CSV = Path(__file__).resolve().parent.parent / "results" / "equilibrium_sweep.csv"
OUTPUT_CSV = ANALYSIS_DIR / "calphad_equilibrium_with_Cij.csv"

# Best model per constant
#   C11, C12 -> MLP, raw dataset, fe02cr14 ablated ("no14")
#   C44      -> MLP, raw dataset, full (fe02cr14 included)
MODEL_FILE_FOR_TARGET = {
    "C11": ANALYSIS_DIR / "mlp_models_raw_no14.pkl",
    "C12": ANALYSIS_DIR / "mlp_models_raw_no14.pkl",
    "C44": ANALYSIS_DIR / "mlp_models_raw_full.pkl",
}


def load_mlp_model(pkl_path: Path, target: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "mlp_models" in data:
        models = data["mlp_models"]
    elif isinstance(data, dict) and target in data:
        models = data
    elif isinstance(data, dict):
        mode_key = next(iter(data.keys()))
        inner = data[mode_key]
        if "mlp_models" not in inner:
            raise KeyError(
                f"Could not find 'mlp_models' in {pkl_path.name}. "
                f"Top-level keys: {list(data.keys())}"
            )
        models = inner["mlp_models"]
    else:
        raise TypeError(f"Unexpected pickle structure in {pkl_path.name}: {type(data)}")

    if target not in models:
        raise KeyError(f"Target '{target}' not found in {pkl_path.name}. Available: {list(models.keys())}")

    return models[target]


def predict_Cij(x_cr_values: np.ndarray) -> dict:
    X = np.asarray(x_cr_values, dtype=float).reshape(-1, 1)
    predictions = {}
    for target, pkl_path in MODEL_FILE_FOR_TARGET.items():
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"{pkl_path} not found. Check the actual filename in {ANALYSIS_DIR} "
                f"and update MODEL_FILE_FOR_TARGET if it differs."
            )
        model = load_mlp_model(pkl_path, target)
        predictions[target] = model.predict(X)
    return predictions


def run():
    df = pd.read_csv(EQUILIBRIUM_CSV)

    preds = predict_Cij(df["phase_x_cr"].values)
    for target, values in preds.items():
        df[target] = values

    df["B"] = (df["C11"] + 2 * df["C12"]) / 3.0

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows with C11, C12, C44, B to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()