"""Scaffold-based cross-validation and grid search for model hyperparameters."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from copy import deepcopy
from sklearn.model_selection import ParameterGrid


def foldwise_cv(model, X, y, folds, return_oof=True):
    """Run scaffold-based CV across the folds."""

    oof_preds = np.zeros(len(y)) if return_oof else None
    fold_rmses, fold_r2s, fold_pearsons = [], [], []

    for i, test_idx in enumerate(folds):
        train_idx = [idx for j, fold in enumerate(folds) if j != i for idx in fold]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model_fold = clone(model)

        model_fold.fit(X_train, y_train)
        y_pred = model_fold.predict(X_test)

        if return_oof:
            oof_preds[test_idx] = y_pred

        fold_rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        fold_r2s.append(r2_score(y_test, y_pred))
        fold_pearsons.append(pearsonr(y_test, y_pred)[0])

    metrics = {
        'RMSE_mean': np.mean(fold_rmses),
        'RMSE_std': np.std(fold_rmses, ddof=1),
        'R2_mean': np.mean(fold_r2s),
        'Pearson_mean': np.mean(fold_pearsons),
    }

    return metrics, oof_preds

def run_grid_search(model_builder, param_grid, folds, X, y):
    """Grid search using scaffold-based CV."""

    best_metrics = {'RMSE_mean': float('inf')}
    best_params = None
    results = []

    for params in ParameterGrid(param_grid):
        model = model_builder(**params)
        metrics, _ = foldwise_cv(model, X, y, folds, return_oof=False)
        metrics_rounded = {k: round(float(v), 3) for k, v in metrics.items()}
        results.append({**params, **metrics_rounded})
        if metrics['RMSE_mean'] < best_metrics['RMSE_mean']:
            best_metrics = deepcopy(metrics_rounded)
            best_params = params

    return best_params, best_metrics, pd.DataFrame(results)