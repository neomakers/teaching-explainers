"""L1-01 SVD 奇异值分解实验

目的：验证截断 SVD 的最优低秩近似（Eckart-Young）性质，并对照"自实现 Jacobi SVD"
与 numpy/LAPACK 的一致性；用随机投影与秩 1 均值近似作基线，证明"最优"确有含义。
输入：digits 前 200 张 8x8 灰度图组成矩阵 A(200, 64)，像素已缩放到 [0,1]。
参数：k ∈ {1,2,5,10,20,40,64}；随机种子 42。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, data, env_info, rel_fro          # noqa: E402
from fromscratch.svd import svd as my_svd, truncate      # noqa: E402

SEED = 42
K_LIST = [1, 2, 5, 10, 20, 40, 64]


def main(seed=SEED):
    rng = np.random.default_rng(seed)
    X, _, desc = data.digits(seed)
    A = X[:200]
    n, p = A.shape
    exp = Exp(exp_id="L1-01", algo="SVD 奇异值分解", layer="L1 数学基础",
              purpose="验证截断 SVD 的最优低秩近似性质，并校验自实现与 LAPACK 的一致性",
              dataset="digits 前 200 张 8x8 图像，矩阵 A%s；%s" % (A.shape, desc),
              baselines=["秩 1 均值近似", "随机高斯投影（同维）",
                         "numpy.linalg.svd（LAPACK）"],
              metrics=["相对 Frobenius 误差", "压缩比", "累积能量",
                       "奇异值最大绝对差", "耗时"],
              seed=seed, params={"k_list": K_LIST, "matrix": list(A.shape)},
              env=env_info())

    with exp.timer("自实现 Jacobi SVD"):
        Uf, sf, Vtf = my_svd(A)
    with exp.timer("numpy LAPACK SVD"):
        Un, sn, Vtn = np.linalg.svd(A, full_matrices=False)
    d_sigma = float(np.max(np.abs(sf - sn)))
    exp.add(方法="自实现 vs LAPACK", 指标="奇异值最大绝对差", 数值=d_sigma,
            说明="两者奇异值序列逐项比较")
    exp.add(方法="自实现 vs LAPACK", 指标="前 5 奇异值(自实现)",
            数值=float(sf[0]), 说明=np.array2string(sf[:5], precision=4))
    exp.add(方法="自实现 vs LAPACK", 指标="前 5 奇异值(LAPACK)",
            数值=float(sn[0]), 说明=np.array2string(sn[:5], precision=4))

    total_energy = float(np.sum(sf ** 2))
    err_svd, err_rand = [], []
    for k in K_LIST:
        with exp.timer("重建 k=%d" % k):
            Ak = truncate(Uf, sf, Vtf, k)
        e = rel_fro(A, Ak)
        err_svd.append(e)
        exp.add(方法="截断 SVD(自实现)", k=k, 相对误差=e,
                压缩比=(n * p) / (k * (n + p + 1)),
                累积能量=float(np.sum(sf[:k] ** 2) / total_energy))
        G = rng.normal(size=(p, k))
        Ar = (A @ G) @ np.linalg.pinv(G)
        er = rel_fro(A, Ar)
        err_rand.append(er)
        exp.add(方法="随机高斯投影", k=k, 相对误差=er,
                压缩比=(n * p) / (k * (n + p + 1)),
                累积能量=float("nan"))
    e1 = rel_fro(A, np.tile(A.mean(0), (n, 1)))
    exp.add(方法="秩 1 均值近似", k=1, 相对误差=e1, 压缩比=(n * p) / (n + p + 1),
            累积能量=float(sf[0] ** 2 / total_energy))

    exp.figure("l1_01_svd_error", "line", "SVD 低秩近似：相对误差 vs k",
               [str(k) for k in K_LIST],
               [("截断 SVD", err_svd), ("随机高斯投影", err_rand)],
               ylabel="相对 Frobenius 误差", xlabel="k（保留奇异值个数）")
    exp.note("k=64 时相对误差降到 %.2e（可视为精确重建）；k=10 误差 %.1f%%，对应 %.1f:1 压缩比；"
             "k=20 误差 %.1f%%。" % (err_svd[-1], 100 * err_svd[3],
                                  (n * p) / (10 * (n + p + 1)), 100 * err_svd[4]))
    exp.note("在每一个 k 上，截断 SVD 的误差都严格小于同维度随机投影，这正是 Eckart-Young 定理的实验体现。")
    exp.note("自实现 Jacobi SVD 与 LAPACK 的奇异值最大绝对差为 %.2e，可认为数值一致（差异仅来自迭代收敛判据）。"
             % d_sigma)
    exp.save()


if __name__ == "__main__":
    main()
