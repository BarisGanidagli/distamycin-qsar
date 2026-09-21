"""Tanimoto fingerprint kernel for scikit-learn Gaussian processes."""

import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter


class TanimotoKernel(Kernel):
    """Tanimoto kernel for binary vectors.
     
    k(x, y) = variance * <x,y> / (<x,x> + <y,y> - <x,y>)

    Implemented as a scikit-learn kernel with one amplitude hyperparameter.
    """

    def __init__(self, variance=1.0, variance_bounds=(1e-4, 1e4)):
        self.variance = variance
        self.variance_bounds = variance_bounds

    @property
    def hyperparameter_variance(self):
        return Hyperparameter("variance", "numeric", self.variance_bounds)

    @staticmethod
    def _tanimoto(X, Y):
        Xs = np.einsum("ij,ij->i", X, X)
        Ys = np.einsum("ij,ij->i", Y, Y)
        XY = X @ Y.T
        denom = Xs[:, None] + Ys[None, :] - XY
        return np.where(denom > 0, XY / np.where(denom > 0, denom, 1.0), 0.0)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            K = self.variance * self._tanimoto(X, X)
            if eval_gradient:
                if self.hyperparameter_variance.fixed:
                    return K, np.empty((X.shape[0], X.shape[0], 0))
                return K, K[:, :, np.newaxis]
            return K
        if eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")
        return self.variance * self._tanimoto(X, np.atleast_2d(Y))

    def diag(self, X):
        X = np.atleast_2d(X)
        Xs = np.einsum("ij,ij->i", X, X)
        return self.variance * np.where(Xs > 0, 1.0, 0.0)

    def is_stationary(self):
        return False

    def __repr__(self):
        return f"Tanimoto(variance={self.variance:.3g})"
