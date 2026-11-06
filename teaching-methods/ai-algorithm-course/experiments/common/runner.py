"""实验公共工具：固定种子、计时辅助、指标计算、环境信息。"""
import platform
import sys
import time

import numpy as np


def set_seed(seed=42):
    np.random.seed(seed)
    return np.random.default_rng(seed)


def env_info():
    import sklearn
    return {"python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__}


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


# ---------- 指标 ----------
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def rel_fro(A, B):
    """相对 Frobenius 误差 ||A-B||_F / ||A||_F"""
    A, B = np.asarray(A), np.asarray(B)
    return float(np.linalg.norm(A - B) / (np.linalg.norm(A) + 1e-12))


def classification_report_short(y, p):
    """不依赖 sklearn.metrics 的 P/R/F1（宏平均）。"""
    y, p = np.asarray(y), np.asarray(p)
    labs = np.unique(np.r_[y, p])
    ps, rs = [], []
    for c in labs:
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
    P, R = float(np.mean(ps)), float(np.mean(rs))
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return float(P), float(R), float(F1)


def roc_auc(y, score):
    """二分类 AUC（秩和法，零依赖）。"""
    y, s = np.asarray(y), np.asarray(score)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    sp = ranks[y == 1].sum()
    return float((sp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def adjusted_rand(y, p):
    """零依赖 ARI（math.comb 实现，不需要 scipy）。"""
    from math import comb
    from collections import Counter
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    cy, cp = Counter(y), Counter(p)
    both = Counter(zip(y, p))
    sum_c = lambda x: sum(comb(v, 2) for v in x.values())
    a, b, d = sum_c(cy), sum_c(cp), sum_c(both)
    tot = comb(n, 2)
    exp = a * b / tot
    mx = (a + b) / 2
    return float((d - exp) / (mx - exp)) if mx != exp else 0.0
