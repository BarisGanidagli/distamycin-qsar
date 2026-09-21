"""Free-Wilson and Tobit ridge models for R-group SAR."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100]


def build_X(frame, include_interactions=False):
    """Build a one-hot design matrix from TYPE_fw labels.

    include_interactions=True adds AB, BC, AC interaction columns.
    """
    pos = frame["TYPE_fw"].str.split(" + ", regex=False, expand=True).iloc[:, :3]
    pos.columns = ["A", "B", "C"]
    if include_interactions:
        for p1, p2 in [("A", "B"), ("B", "C"), ("A", "C")]:
            pos[p1 + p2] = pos[p1] + "|" + pos[p2]
    return pd.get_dummies(pos, prefix=pos.columns.tolist()).astype(float)


def scaffold_smiles(smiles):
    """Bemis-Murcko scaffold SMILES, or the input string if it cannot be parsed."""
    m = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else smiles


class TobitRidge:
    """Ridge regression with left-censored targets, where the observed y is an upper bound on the true value."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, left_cens):
        X1 = np.column_stack([np.ones(len(X)), X])
        p = X1.shape[1]
        unc = ~left_cens

        def nll(par):
            b = par[:-1]
            s = np.exp(par[-1])
            mu = X1 @ b
            ll = norm.logpdf(y[unc], mu[unc], s).sum()
            if left_cens.any():
                ll += norm.logcdf((y[left_cens] - mu[left_cens]) / s).sum()
            return -ll + self.alpha * np.sum(b[1:] ** 2)

        self.coef_ = minimize(nll, np.concatenate([np.zeros(p), [0.0]]),
                              method="L-BFGS-B").x[:-1]
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.coef_


def ridge_cv(frame, tag, include_interactions=False, random_state=42):
    """Random-split and scaffold-split RidgeCV for the additive model.

    Scores on all points and returns the scaffold-split R² and RMSE.
    """
    X = build_X(frame, include_interactions).values
    y = frame["pIC50"].values
    groups = frame["SMILES"].map(scaffold_smiles).values

    def _cv(splitter, g=None):
        yhat = np.full_like(y, np.nan)
        it = splitter.split(X, y, g) if g is not None else splitter.split(X)
        for tr, te in it:
            yhat[te] = RidgeCV(alphas=RIDGE_ALPHAS).fit(X[tr], y[tr]).predict(X[te])
        return r2_score(y, yhat), np.sqrt(mean_squared_error(y, yhat))

    rr = _cv(KFold(5, shuffle=True, random_state=random_state))
    ng = len(np.unique(groups))
    rs = _cv(GroupKFold(min(5, ng)), groups)
    print(f"{tag:26s}| random R2={rr[0]:.3f} RMSE={rr[1]:.3f} | "
          f"scaffold R2={rs[0]:.3f} RMSE={rs[1]:.3f}")
    return rs


def tobit_cv(frame, tag="Tobit (all data)", alpha=1.0, random_state=42):
    """Random-split and scaffold-split Tobit-ridge CV.

    Fits on all points including the left-censored values, but scores only on uncensored
    points. Returns the scaffold-split R² and RMSE.
    """
    X = build_X(frame).values
    y = frame["pIC50"].values
    cens = frame["censored"].values
    groups = frame["SMILES"].map(scaffold_smiles).values
    ev = ~cens

    def _cv(splitter, g=None):
        yhat = np.full_like(y, np.nan)
        it = splitter.split(X, y, g) if g is not None else splitter.split(X)
        for tr, te in it:
            yhat[te] = TobitRidge(alpha).fit(X[tr], y[tr], cens[tr]).predict(X[te])
        return r2_score(y[ev], yhat[ev]), np.sqrt(mean_squared_error(y[ev], yhat[ev]))

    rr = _cv(KFold(5, shuffle=True, random_state=random_state))
    ng = len(np.unique(groups))
    rs = _cv(GroupKFold(min(5, ng)), groups)
    print(f"{tag:26s}| random R2={rr[0]:.3f} RMSE={rr[1]:.3f} | "
          f"scaffold R2={rs[0]:.3f} RMSE={rs[1]:.3f}")
    return rs