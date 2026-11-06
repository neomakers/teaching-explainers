"""从零实现：对称矩阵 Jacobi 特征分解 + 截断 SVD（教学参考实现）。

接口风格与 numpy.linalg.svd 保持一致：返回 (U, s, Vt)。
仅用于教学对照与原理验证，不修改任何第三方库实现。
"""
import numpy as np


def jacobi_eigen(A, max_sweeps=100, tol=1e-12):
    """对称矩阵 Jacobi 特征分解，返回 (特征值, 特征向量矩阵 V 的列)。"""
    A = np.array(A, dtype=np.float64)
    n = A.shape[0]
    a = A.copy()
    V = np.eye(n)
    for _ in range(max_sweeps):
        off = np.sum(np.triu(a, 1) ** 2) * 2
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p, q]) < 1e-15:
                    continue
                theta = (a[q, q] - a[p, p]) / (2 * a[p, q])
                t = np.sign(theta if theta != 0 else 1) / \
                    (abs(theta) + np.sqrt(theta * theta + 1))
                c = 1.0 / np.sqrt(t * t + 1)
                s = t * c
                ap, aq = a[:, p].copy(), a[:, q].copy()
                a[:, p], a[:, q] = c * ap - s * aq, s * ap + c * aq
                ap, aq = a[p, :].copy(), a[q, :].copy()
                a[p, :], a[q, :] = c * ap - s * aq, s * ap + c * aq
                vp, vq = V[:, p].copy(), V[:, q].copy()
                V[:, p], V[:, q] = c * vp - s * vq, s * vp + c * vq
    return np.diag(a), V


def svd(A):
    """通过 A^T A 的特征分解得到 SVD，返回 (U, s, Vt)，与 np.linalg.svd 一致。"""
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    k = min(m, n)
    ata = A.T @ A
    w, V = jacobi_eigen(ata)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    s = np.sqrt(np.clip(w[:k], 0, None))
    Vs = V[:, :k]
    U = np.zeros((m, k))
    for j in range(k):
        if s[j] > 1e-12:
            U[:, j] = A @ Vs[:, j] / s[j]
    return U, s, Vs.T


def truncate(U, s, Vt, k):
    """秩 k 截断重建 A_k = sum_{i<k} s_i u_i v_i^T。"""
    k = int(min(max(k, 0), len(s)))
    return (U[:, :k] * s[:k]) @ Vt[:k, :]
