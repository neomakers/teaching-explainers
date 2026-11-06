"""实验数据集与合成用例。

原则：全部离线、可复现。真实数据用 sklearn 内置数据集（随包分发，无需联网），
合成数据用固定随机种子生成，并尽量贴合智能制造场景（工艺参数、传感器时序等）。
"""
import numpy as np
from sklearn.datasets import (load_digits, load_breast_cancer, load_wine,
                              make_blobs, make_moons, make_circles,
                              make_classification)

SEED = 42


def _desc(name, X, y=None, extra=""):
    s = "%s: X%s" % (name, X.shape)
    if y is not None:
        s += ", y%s, 类别%s" % (y.shape, np.bincount(y).tolist())
    return s + ("  " + extra if extra else "")


def digits(seed=SEED):
    """手写数字 8x8 灰度图（1797x64），当作"工件表面图像"使用。"""
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    return X, d.target, _desc("digits(工件图像类比)", X, d.target, "像素已缩放到 [0,1]")


def breast():
    """乳腺癌诊断（569x30，二分类），当作"合格/不合格判定"。"""
    d = load_breast_cancer()
    return d.data, d.target, _desc("breast_cancer(合格判定类比)", d.data, d.target)


def wine():
    """红酒数据（178x13，三分类），当作"工况三分类"。"""
    d = load_wine()
    return d.data, d.target, _desc("wine(工况分类类比)", d.data, d.target)


def synth_process(n=1200, seed=SEED):
    """合成工艺数据集：切削参数 -> 表面粗糙度 Ra。

    刻意包含非线性与交互项，用于检验"线性模型 vs 树模型"的表达力差异。
    特征: [切削速度 v, 进给量 f, 切深 ap, 冷却液温度 T, 主轴振动 vib]
    """
    rng = np.random.default_rng(seed)
    v = rng.uniform(80, 220, n)          # m/min
    f = rng.uniform(0.05, 0.35, n)      # mm/r
    ap = rng.uniform(0.2, 2.0, n)        # mm
    T = 25 + 0.05 * v + rng.normal(0, 2, n)
    vib = 0.4 * f * ap + rng.normal(0, 0.05, n)
    X = np.column_stack([v, f, ap, T, vib])
    Ra = (0.9 + 62.0 * f ** 2 + 0.35 * ap + 0.004 * v
          + 1.6 * f * vib + rng.normal(0, 0.18, n))
    return X, Ra, _desc("synth_process(工艺->粗糙度)", X, None, "含非线性项 62*f^2 与交互项")


def blobs(n=900, centers=4, seed=SEED):
    rng = np.random.default_rng(seed)
    X, y = make_blobs(n_samples=n, centers=centers, cluster_std=0.9,
                      random_state=seed)
    return X, y, _desc("blobs(工况分型)", X, y, "cluster_std=0.9")


def moons(n=500, noise=0.10, seed=SEED):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X, y, _desc("moons(非凸簇)", X, y, "noise=%.2f" % noise)


def circles(n=500, noise=0.06, factor=0.5, seed=SEED):
    X, y = make_circles(n_samples=n, noise=noise, factor=factor,
                        random_state=seed)
    return X, y, _desc("circles(环形分布)", X, y)


def noisy_dims(base_X, n_extra=200, seed=SEED):
    """在特征后追加随机噪声维度，用于演示维度灾难。"""
    rng = np.random.default_rng(seed)
    return np.hstack([base_X, rng.normal(0, 1, (base_X.shape[0], n_extra))])


def timeseries(n=800, seed=SEED):
    """设备振动仿真序列：衰减正弦 + 谐波 + 噪声（预测性维护场景）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.05
    x = (np.exp(-0.004 * np.arange(n)) * np.sin(2 * np.pi * 1.2 * t)
         + 0.35 * np.sin(2 * np.pi * 3.7 * t) + rng.normal(0, 0.05, n))
    return x.astype(np.float64), ("设备振动仿真序列", "n=%d, 采样间隔 0.05" % n)


def sensor_anomaly(n_normal=900, n_anom=60, seed=SEED):
    """多传感器正常工况 + 少量异常点（异常检测场景）。"""
    rng = np.random.default_rng(seed)
    mu = np.array([0.0, 0.0, 0.0])
    C = np.array([[1.0, 0.85, 0.70],
                  [0.85, 1.0, 0.60],
                  [0.70, 0.60, 1.0]])
    L = np.linalg.cholesky(C)
    Xn = rng.normal(size=(n_normal, 3)) @ L.T + mu
    Xa = rng.normal(size=(n_anom, 3)) @ L.T + np.array([4.2, -3.4, 3.0])
    X = np.vstack([Xn, Xa])
    y = np.r_[np.zeros(n_normal), np.ones(n_anom)].astype(int)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], _desc("sensor_anomaly(传感器异常)", X, y,
                                 "异常占比 %.1f%%" % (100 * n_anom / len(X)))


def mixed_signals(n=1000, seed=SEED):
    """盲源分离用例：3 个独立源线性混合（ICA 场景）。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8, n)
    s1 = np.sin(2 * np.pi * 1.0 * t)
    s2 = np.sign(np.sin(2 * np.pi * 0.7 * t))       # 方波（非高斯）
    s3 = rng.uniform(-1, 1, n)                      # 均匀噪声（非高斯）
    S = np.column_stack([s1, s2, s3])
    A = np.array([[0.6, 0.5, 0.3],
                  [0.4, 0.7, 0.5],
                  [0.5, 0.3, 0.8]])
    X = S @ A.T
    return X, S, _desc("mixed_signals(盲源分离)", X, None, "3 源线性混合，混合矩阵已知")


def corr_data(n=600, rho=0.9, seed=SEED):
    """二维相关高斯，用于 PCA 演示。"""
    rng = np.random.default_rng(seed)
    c = np.sqrt(max(0.0, 1 - rho ** 2))
    z = rng.normal(size=(n, 2))
    X = np.column_stack([z[:, 0], rho * z[:, 0] + c * z[:, 1]])
    return X, ("二维相关高斯", "n=%d, rho=%.2f" % (n, rho))


def classification(n=2000, n_features=20, seed=SEED):
    X, y = make_classification(n_samples=n, n_features=n_features,
                               n_informative=6, n_redundant=4,
                               random_state=seed)
    return X, y, _desc("make_classification(通用二分类)", X, y)


class GridWorld:
    """5x5 车间 AGV 网格环境（自建，零依赖，用于表格 Q-Learning）。

    状态 = 格子编号 r*5+c；动作 = 0上 1下 2左 3右。
    到达终点 +10，踩到陷阱 -5 并结束，其余每步 -0.1；slippery>0 时以该概率走偏。
    """

    def __init__(self, size=5, slippery=0.0, seed=SEED):
        self.size = size
        self.slippery = slippery
        self.rng = np.random.default_rng(seed)
        self.start, self.goal = 0, size * size - 1
        self.traps = {7, 11, 17}
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, a):
        if self.slippery > 0 and self.rng.random() < self.slippery:
            a = int(self.rng.integers(0, 4))
        r, c = divmod(self.state, self.size)
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][a]
        r = min(max(r + dr, 0), self.size - 1)
        c = min(max(c + dc, 0), self.size - 1)
        self.state = r * self.size + c
        if self.state == self.goal:
            return self.state, 10.0, True
        if self.state in self.traps:
            return self.state, -5.0, True
        return self.state, -0.1, False
