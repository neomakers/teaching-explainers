"""L3-03 NMF 非负矩阵分解实验

目的：对比 NMF 与 PCA 在重构误差、非负性与可解释性（稀疏部件）上的差异。
输入：digits(1797x64)，像素非负且已缩放至 [0,1]。
参数：n_components=20；NMF init=nndsvd，max_iter=500；PCA 随机种子 42。
"""
import os
import sys

import numpy as np
from sklearn.decomposition import NMF, PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, data, env_info, rel_fro      # noqa: E402

SEED = 42
K = 20


def sparsity(M):
    """Hinton 式稀疏度：越接近 1 表示越稀疏。"""
    a = np.abs(M)
    return float((np.sqrt(M.size) - a.sum() / (np.sqrt((a ** 2).sum()) + 1e-12))
                 / (np.sqrt(M.size) - 1))


def main(seed=SEED):
    X, _, desc = data.digits(seed)
    Xtr = X[:1200]
    exp = Exp(exp_id="L3-03", algo="NMF 非负矩阵分解", layer="L3 降维与矩阵分解",
              purpose="对比 NMF 与 PCA 的重构误差、非负性与部件稀疏度，说明可解释性差异",
              dataset="digits 前 1200 张（非负，数值 ∈ [0,1]）；%s" % desc,
              baselines=["PCA（允许负系数）"],
              metrics=["重建相对误差", "负值像素占比", "部件稀疏度", "耗时"],
              seed=seed, params={"n_components": K, "nmf_init": "nndsvd",
                                 "max_iter": 500},
              env=env_info())

    with exp.timer("NMF fit"):
        nmf = NMF(n_components=K, init="nndsvd", max_iter=500,
                  random_state=seed).fit(Xtr)
    W = nmf.transform(Xtr)
    Xn = nmf.inverse_transform(W)
    exp.add(方法="NMF", 重建相对误差=rel_fro(Xtr, Xn),
            负值像素占比=float(np.mean(Xn < -1e-9)),
            部件稀疏度=sparsity(nmf.components_),
            重构非负性="满足（约束保证）")

    with exp.timer("PCA fit"):
        pca = PCA(n_components=K, random_state=seed).fit(Xtr)
    Z = pca.transform(Xtr)
    Xp = pca.inverse_transform(Z)
    exp.add(方法="PCA", 重建相对误差=rel_fro(Xtr, Xp),
            负值像素占比=float(np.mean(Xp < -1e-9)),
            部件稀疏度=sparsity(pca.components_),
            重构非负性="不满足（存在负值像素）")

    exp.figure("l3_03_nmf_comp", "bar", "NMF 前 6 个部件的稀疏度",
               ["部件%d" % (i + 1) for i in range(6)],
               [("NMF", [sparsity(nmf.components_[i:i + 1]) for i in range(6)]),
                ("PCA", [sparsity(pca.components_[i:i + 1]) for i in range(6)])],
               ylabel="稀疏度", xlabel="部件序号")
    r0, r1 = exp.rows[0], exp.rows[1]
    exp.note("重构误差：NMF %.4f vs PCA %.4f。PCA 误差更低是意料之中——它的解空间不受非负约束，"
             "是 Frobenius 意义下的最优低秩近似；NMF 用少量精度换取可解释性。"
             % (r0["重建相对误差"], r1["重建相对误差"]))
    exp.note("PCA 重构中有 %.1f%% 的像素为负（物理上不成立的「负亮度」），NMF 恒为 0%%；"
             "NMF 的部件稀疏度 %.3f 高于 PCA 的 %.3f，因此更像「零件的局部部件叠加」。"
             % (100 * r1["负值像素占比"], r0["部件稀疏度"], r1["部件稀疏度"]))
    exp.note("应用提示：光谱分解、故障模式分解这类要求「成分可加、非负」的场景选 NMF；"
             "纯压缩与去噪选 PCA/SVD。")
    exp.save()


if __name__ == "__main__":
    main()
