"""Model pipelines, ensembling, IC50 conversion, and compound ranking."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from xgboost import XGBRegressor
from rdkit import DataStructs


def pIC50_to_IC50(pIC50, uncertainty=None):
    """Convert pIC₅₀ to IC₅₀ in micromolar and calculate error bars when needed."""

    IC50_uM = 10 ** (6 - pIC50)
    if uncertainty is not None:
        IC50_upper = 10 ** (6 - (pIC50 - uncertainty))
        IC50_lower = 10 ** (6 - (pIC50 + uncertainty))
        return IC50_uM, (IC50_upper - IC50_lower) / 2
    return IC50_uM


def pIC50_to_IC50_array(pIC50_array):
    """Vectorised conversion of pIC₅₀ values to IC₅₀ in micromolar."""

    return 10 ** (-np.asarray(pIC50_array)) * 1e6


def load_and_clean_dataset(csv_path, verbose=True):
    """Load the CSV, calculate pIC₅₀, and split into labelled and unlabelled subsets."""

    df = pd.read_csv(csv_path)
    df['IC50'] = pd.to_numeric(df['IC50'].replace(['', 'null'], np.nan), errors='coerce')

    if verbose:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            print("Missing values:\n", missing.to_frame('count').assign(
                pct=lambda x: (x['count'] / len(df) * 100).round(1)
            ))

    df['pIC50'] = -np.log10(df['IC50'].replace(0, 1e-12) * 1e-6)

    labelled_df   = df[df['pIC50'].notnull()].reset_index(drop=True)
    unlabelled_df = df[df['pIC50'].isnull()].reset_index(drop=True)

    return labelled_df, unlabelled_df


def build_pipeline(model_type, random_state=None, n_jobs=None, kernel=None,
                   n_restarts_optimizer=None, **kwargs):
    """Wrap StandardScaler and the selected model in a Pipeline."""

    if model_type == 'bayesian_ridge':
        model = BayesianRidge(**kwargs)

    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)

    elif model_type == 'xgboost':
        model = XGBRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)

    elif model_type == 'gp':
        if kernel is None:
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=random_state,
            n_restarts_optimizer=n_restarts_optimizer or 0,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    return Pipeline([('scaler', StandardScaler()), ('model', model)])


def gp_predict_with_std(pipeline, X):
    """Return GP mean predictions and posterior standard deviations."""

    if hasattr(pipeline, 'named_steps'):
        X_scaled = pipeline.named_steps['scaler'].transform(X)
        return pipeline.named_steps['model'].predict(X_scaled, return_std=True)
    return pipeline.predict(X, return_std=True)

def ensemble_predictions(rf_model, gp_model, w_rf, w_gp, X, y_actual):
    """Return weighted RF and GP ensemble predictions in IC₅₀ units."""

    rf_pred          = rf_model.predict(X)
    gpr_pred, gpr_std = gp_predict_with_std(gp_model, X)
    ensemble_pred    = w_rf * rf_pred + w_gp * gpr_pred

    IC50_actual          = pIC50_to_IC50_array(y_actual)
    IC50_pred, IC50_uncert = zip(*[pIC50_to_IC50(p, u) for p, u in zip(ensemble_pred, gpr_std)])

    return IC50_actual, np.array(IC50_pred), np.array(IC50_uncert)

def search_ensemble_weights(rf_oof, gpr_oof, y, min_gp_fraction=0.25):
    """Search RF and GP weight combinations and return the best weights with a results table."""

    best_rmse = float('inf')
    best_weights = None
    results = []

    for w_rf in np.linspace(0.0, 1.0, 21):
        w_gp = 1 - w_rf
        gp_frac = w_gp / (w_rf + w_gp)
        if gp_frac < min_gp_fraction:
            continue
        oof_pred = w_rf * rf_oof + w_gp * gpr_oof
        rmse_cand = np.sqrt(mean_squared_error(y, oof_pred))
        r2_cand = r2_score(y, oof_pred)
        if rmse_cand < best_rmse:
            best_rmse = rmse_cand
            best_weights = (w_rf, w_gp)
        results.append({
            'w_RF': round(w_rf, 3), 'w_GP': round(w_gp, 3),
            'GP_frac': round(gp_frac, 3),
            'OOF_RMSE': round(rmse_cand, 6),
            'OOF_R2': round(r2_cand, 4),
        })

    df = pd.DataFrame(results)
    df['Selected'] = ''
    df.loc[df['OOF_RMSE'].idxmin(), 'Selected'] = 'Yes'
    return best_weights, df


def compute_applicability_domain(unlabelled_fps, train_fps, threshold=0.4):
    """Return nearest-neighbour Tanimoto similarity and applicability-domain flags."""

    nn_tanimoto = np.array([
        max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        for fp in unlabelled_fps
    ])
    return nn_tanimoto, nn_tanimoto >= threshold