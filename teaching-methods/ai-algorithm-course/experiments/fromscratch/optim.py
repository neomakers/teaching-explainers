"""从零实现：一阶优化器族（SGD / Momentum / RMSProp / Adam）+ 线性模型。

用于 L1「梯度下降族」实验：在同一凸目标（岭回归）上比较收敛速度与稳定性。
"""
import numpy as np


def minimize(grad_fn, x0, method="sgd", lr=1e-2, steps=500,
             momentum=0.9, beta2=0.999, eps=1e-8):
    """grad_fn(x) -> (loss, grad)。返回 (x, loss_history)。"""
    x = np.array(x0, dtype=np.float64).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    hist = []
    for t in range(1, steps + 1):
        loss, g = grad_fn(x)
        hist.append(float(loss))
        if method == "sgd":
            x -= lr * g
        elif method == "momentum":
            m = momentum * m + g
            x -= lr * m
        elif method == "rmsprop":
            v = beta2 * v + (1 - beta2) * g * g
            x -= lr * g / (np.sqrt(v) + eps)
        elif method == "adam":
            m = momentum * m + (1 - momentum) * g
            v = beta2 * v + (1 - beta2) * g * g
            mh = m / (1 - momentum ** t)
            vh = v / (1 - beta2 ** t)
            x -= lr * mh / (np.sqrt(vh) + eps)
        else:
            raise ValueError(method)
    loss, _ = grad_fn(x)
    hist.append(float(loss))
    return x, hist


class RidgeGD:
    """岭回归目标：0.5*||Xw-y||^2/n + 0.5*lambda*||w||^2，用于优化器对比。"""

    def __init__(self, X, y, l2=1e-2):
        self.X, self.y, self.l2 = X, y, l2

    def __call__(self, w):
        r = self.X @ w - self.y
        loss = 0.5 * np.mean(r ** 2) + 0.5 * self.l2 * np.sum(w ** 2)
        grad = self.X.T @ r / len(self.y) + self.l2 * w
        return loss, grad


class LogisticGD:
    """逻辑回归（含 L2），用于验证自实现与 sklearn 的一致性。接口 fit/predict。"""

    def __init__(self, lr=0.1, steps=800, l2=1e-3, method="adam"):
        self.lr, self.steps, self.l2, self.method = lr, steps, l2, method

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        Xb = np.hstack([X, np.ones((len(X), 1))])

        def grad(w):
            p = self._sigmoid(Xb @ w)
            loss = -np.mean(y * np.log(p + 1e-12) +
                            (1 - y) * np.log(1 - p + 1e-12)) \
                + 0.5 * self.l2 * np.sum(w ** 2)
            g = Xb.T @ (p - y) / len(y) + self.l2 * w
            return loss, g

        w0 = np.zeros(Xb.shape[1])
        self.w_, self.history_ = minimize(grad, w0, method=self.method,
                                           lr=self.lr, steps=self.steps)
        return self

    def predict_proba(self, X):
        Xb = np.hstack([np.asarray(X, dtype=np.float64),
                        np.ones((len(X), 1))])
        return self._sigmoid(Xb @ self.w_)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
