"""从零实现：PCA（SVD 路径），接口对齐 sklearn.decomposition.PCA。

fit(X) -> transform(X) -> inverse_transform(Z)；提供 explained_variance_ratio_。
"""
import numpy as np

from .svd import svd


class PCA:
    def __init__(self, n_components=2, whiten=False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, s, Vt = svd(Xc)
        self.components_ = Vt[: self.n_components]      # 主成分（行向量）
        self.singular_values_ = s
        var = (s ** 2) / (len(X) - 1)
        self.explained_variance_ = var[: self.n_components]
        self.explained_variance_ratio_ = var / var.sum()
        return self

    def transform(self, X):
        Xc = np.asarray(X, dtype=np.float64) - self.mean_
        Z = Xc @ self.components_.T
        if self.whiten:
            Z = Z / np.sqrt(self.explained_variance_ + 1e-12)
        return Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_
