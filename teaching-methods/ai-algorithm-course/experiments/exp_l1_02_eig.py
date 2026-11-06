"""L1-02 对称矩阵特征值分解实验

目的：验证 Jacobi 特征分解的正确性，并建立"协方差特征值 λ"与"SVD 奇异值 σ"的定量关系 λ = σ²/(n-1)。
输入：digits 前 400 张图像的协方差矩阵 C(64, 64)。
参数：种子 42；Jacobi 收敛判据 off-diagonal < 1e-12，最多 100 sweep。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, data, env_info                   # noqa: E402
from fromscratch.svd import jacobi_eigen, svd as my_svd   # noqa: E402

SEED = 42


def main(seed=SEED):
    X, _, desc = data.digits(seed)
    A = X[:400]
    n, p = A.shape
    C = np.cov(A, rowvar=False)
    exp = Exp(exp_id="L1-02", algo="特征值分解", layer="L1 数学基础",
              purpose="校验 Jacobi 特征分解的精度与正交性，并验证 σ²=(n-1)λ 的等价关系",
              dataset="digits 前 400 张图像的协方差矩阵 C%s；%s" % (C.shape, desc),
              baselines=["numpy.linalg.eigh（LAPACK）", "SVD 关系 σ²/(n-1)"],
              metrics=["特征值最大绝对差", "特征向量正交误差",
                       "σ²/(n-1) 与 λ 最大差", "耗时"],
              seed=seed, params={"n_samples": n, "tol": 1e-12},
              env=env_info())

    with exp.timer("自实现 Jacobi 特征分解"):
        w_mine, V_mine = jacobi_eigen(C)
    with exp.timer("numpy.linalg.eigh"):
        w_np, V_np = np.linalg.eigh(C)
    order = np.argsort(w_mine)[::-1]
    w_mine, V_mine = w_mine[order], V_mine[:, order]
    w_np, V_np = w_np[::-1], V_np[:, ::-1]

    d_lam = float(np.max(np.abs(w_mine - w_np)))
    orth_mine = float(np.max(np.abs(V_mine.T @ V_mine - np.eye(p))))
    orth_np = float(np.max(np.abs(V_np.T @ V_np - np.eye(p))))

    _, s, _ = my_svd(A - A.mean(0))
    lam_from_svd = s ** 2 / (n - 1)
    d_rel = float(np.max(np.abs(lam_from_svd[:p] - w_mine[:p])))

    exp.add(方法="自实现 Jacobi", 指标="特征值最大绝对差", 数值=d_lam, 说明="与 LAPACK eigh 对比")
    exp.add(方法="自实现 Jacobi", 指标="正交误差 max|V^T V - I|", 数值=orth_mine, 说明="越小越好")
    exp.add(方法="LAPACK eigh", 指标="正交误差 max|V^T V - I|", 数值=orth_np, 说明="参考值")
    exp.add(方法="SVD vs 特征分解", 指标="max|σ²/(n-1) - λ|", 数值=d_rel,
            说明="验证协方差特征值 = 奇异值平方/(n-1)")
    exp.add(方法="前 5 特征值", 指标="自实现", 数值=w_mine[0],
            说明=np.array2string(w_mine[:5], precision=5))
    exp.add(方法="前 5 特征值", 指标="LAPACK", 数值=w_np[0],
            说明=np.array2string(w_np[:5], precision=5))

    exp.figure("l1_02_eig", "line", "协方差特征值谱（前 30 个，对数尺度已省略）",
               [str(i + 1) for i in range(30)],
               [("自实现 Jacobi", w_mine[:30].tolist()),
                ("LAPACK eigh", w_np[:30].tolist())],
               ylabel="特征值 λ", xlabel="主成分序号")
    exp.note("特征值最大绝对差 %.2e、正交误差 %.2e，说明自实现 Jacobi 与 LAPACK 数值一致。" % (d_lam, orth_mine))
    exp.note("σ²/(n-1) 与 λ 的最大差为 %.2e，实验确认 PCA 的两条推导路径（协方差特征分解 / 数据中心化后 SVD）完全等价。" % d_rel)
    exp.note("特征值谱衰减很快：前 5 个特征值已占总和的 %.1f%%，这是 PCA 能用少数主成分保留大部分方差的原因。"
             % (100 * w_mine[:5].sum() / w_mine.sum()))
    exp.save()


if __name__ == "__main__":
    main()
