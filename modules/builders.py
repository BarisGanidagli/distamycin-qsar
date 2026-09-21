"""Helper functions for building the scikit-learn model pipelines."""

from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from modules.prediction import build_pipeline


def build_rf(seed, n_jobs=-1, **params):
    return build_pipeline("random_forest", random_state=seed, n_jobs=n_jobs, **params)


def build_gp(seed, **params):
    kernel = RBF(length_scale=params.pop('length_scale')) + \
             WhiteKernel(noise_level=params.pop('noise_level'))
    return build_pipeline("gp", kernel=kernel, n_restarts_optimizer=2,
                          random_state=seed, **params)


def build_xgb(seed, n_jobs=-1, **params):
    return Pipeline([('model', XGBRegressor(
        objective='reg:squarederror', tree_method="hist",
        random_state=seed, n_jobs=n_jobs, **params))])


def build_br(**params):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', BayesianRidge(**params))
    ])