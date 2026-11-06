"""从零实现：2D 卷积 / 池化 / 自注意力 / Elman RNN（numpy，纯 CPU）。

用于为 CNN、Transformer、RNN 提供「可跑的替代验证」：本机无 PyTorch 且显卡损坏，
完整训练不可行，因此改为验证算子的数学性质 + 小规模可训练模型。
"""
import numpy as np


# ---------------- CNN 算子 ----------------
def conv2d(X, kernel, stride=1, pad=0):
    """单通道 2D 互相关（与 PyTorch conv2d 的 cross-correlation 一致）。"""
    X = np.asarray(X, dtype=np.float64)
    K = np.asarray(kernel, dtype=np.float64)
    if pad:
        X = np.pad(X, pad)
    kh, kw = K.shape
    oh = (X.shape[0] - kh) // stride + 1
    ow = (X.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(X[i * stride:i * stride + kh,
                                 j * stride:j * stride + kw] * K)
    return out


def maxpool(X, size=2):
    X = np.asarray(X, dtype=np.float64)
    h, w = X.shape[0] // size * size, X.shape[1] // size * size
    X = X[:h, :w]
    return X.reshape(h // size, size, w // size, size).max(axis=(1, 3))


def relu(X):
    return np.maximum(0, X)


# ---------------- 自注意力 ----------------
def softmax(A, axis=-1):
    E = np.exp(A - A.max(axis=axis, keepdims=True))
    return E / (E.sum(axis=axis, keepdims=True) + 1e-12)


def self_attention(X, Wq, Wk, Wv):
    """缩放点积自注意力，返回 (输出, 注意力矩阵)。"""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    dk = K.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    A = softmax(scores, axis=-1)
    return A @ V, A


def multihead_attention(X, Wq, Wk, Wv, n_heads=2):
    """多头注意力的等价实现（按列切分后拼接）。"""
    d = X.shape[1]
    outs = []
    for h in range(n_heads):
        sl = slice(h * (d // n_heads), (h + 1) * (d // n_heads))
        o, _ = self_attention(X, Wq[:, sl], Wk[:, sl], Wv[:, sl])
        outs.append(o)
    return np.hstack(outs)


# ---------------- Elman RNN ----------------
class ElmanRNN:
    """简单循环网络：h_t = tanh(Wxh x_t + Whh h_{t-1} + b)，输出线性头。

    全展开 BPTT，梯度裁剪，SGD 更新。用于时序预测（正弦/振动衰减序列）。
    """

    def __init__(self, n_input=1, n_hidden=16, n_output=1, lr=0.02,
                 seed=42, clip=5.0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(n_hidden)
        self.Wxh = rng.uniform(-s, s, (n_hidden, n_input))
        self.Whh = rng.uniform(-s, s, (n_hidden, n_hidden)) * 0.5
        self.bh = np.zeros(n_hidden)
        self.Why = rng.uniform(-s, s, (n_output, n_hidden))
        self.by = np.zeros(n_output)
        self.lr, self.clip, self.n_hidden = lr, clip, n_hidden
        self.n_input, self.n_output = n_input, n_output

    def forward(self, X):
        X = np.asarray(X, dtype=np.float64)
        T = len(X)
        H = np.zeros((T + 1, self.n_hidden))
        Y = np.zeros((T, self.n_output))
        for t in range(T):
            H[t + 1] = np.tanh(self.Wxh @ X[t] + self.Whh @ H[t] + self.bh)
            Y[t] = self.Why @ H[t + 1] + self.by
        return Y, H

    def fit(self, X, y, epochs=120, log_every=0):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        T = len(X)
        hist = []
        for ep in range(epochs):
            Y, H = self.forward(X)
            err = Y[:, 0] - y
            loss = float(np.mean(err ** 2))
            hist.append(loss)
            dWhy = np.zeros_like(self.Why)
            dby = np.zeros_like(self.by)
            dWxh = np.zeros_like(self.Wxh)
            dWhh = np.zeros_like(self.Whh)
            dbh = np.zeros_like(self.bh)
            dh_next = np.zeros(self.n_hidden)
            for t in range(T - 1, -1, -1):
                dy = (2.0 / T) * np.full(self.n_output, err[t])  # dL/dy_t
                dWhy += np.outer(dy, H[t + 1])
                dby += dy
                dh = self.Why.T @ dy + dh_next                # dL/dh_t
                draw = dh * (1 - H[t + 1] ** 2)               # tanh 导数
                dWxh += np.outer(draw, X[t])
                dWhh += np.outer(draw, H[t])
                dbh += draw
                dh_next = self.Whh.T @ draw
            for g in (dWxh, dWhh, dWhy, dbh, dby):
                np.clip(g, -self.clip, self.clip, out=g)
            self.Wxh -= self.lr * dWxh
            self.Whh -= self.lr * dWhh
            self.Why -= self.lr * dWhy
            self.bh -= self.lr * dbh
            self.by -= self.lr * dby
            if log_every and ep % log_every == 0:
                print("  epoch %4d  mse=%.6f" % (ep, loss))
        return hist

    def predict(self, X):
        return self.forward(X)[0][:, 0]
