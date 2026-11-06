"""从零实现：K-Means（含 k-means++ 与随机初始化），接口对齐 sklearn。"""
import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, init="k-means++", n_init=10,
                 max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _kmeanspp(self, X, rng):
        n = len(X)
        centers = [X[int(rng.integers(n))]]
        for _ in range(1, self.n_clusters):
            d = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2)
                       .sum(-1), axis=1)
            probs = d / (d.sum() + 1e-12)
            centers.append(X[int(rng.choice(n, p=probs))])
        return np.array(centers)

    def _single(self, X):
        rng = np.random.default_rng(self.random_state)
        if self.init == "k-means++":
            centers = self._kmeanspp(X, rng)
        else:
            centers = X[rng.choice(len(X), self.n_clusters, replace=False)]
        prev = None
        for _ in range(self.max_iter):
            d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            labels = d.argmin(1)
            new = np.array([X[labels == k].mean(0) if np.any(labels == k)
                            else centers[k] for k in range(self.n_clusters)])
            shift = np.linalg.norm(new - centers)
            centers = new
            if shift < self.tol:
                break
        inertia = float(((X - centers[labels]) ** 2).sum())
        return inertia, labels, centers

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        best = None
        for i in range(self.n_init):
            self.random_state = self.random_state + i if i else self.random_state
            out = self._single(X)
            if best is None or out[0] < best[0]:
                best = out
        self.inertia_, self.labels_, self.cluster_centers_ = best
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
