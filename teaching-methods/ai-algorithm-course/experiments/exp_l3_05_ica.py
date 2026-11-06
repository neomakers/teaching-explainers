"""L3-05 ICA 独立成分分析实验

目的：验证 ICA 能恢复统计独立的源信号，而 PCA 只能做到"不相关"（对非高斯源失效）。
输入：data.mixed_signals —— 3 个非高斯源（正弦、方波、均匀噪声）经已知矩阵线性混合。
参数：n_components=3；FastICA(whiten='unit-variance', max_iter=500)；PCA 作对照；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.decomposition import FastICA, PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, data, env_info                 # noqa: E402

SEED = 42


def match_quality(S_true, S_est):
    """分离信号与真实源的最大 |相关系数| 匹配（顺序与符号不敏感）。"""
    S_true = (S_true - S_true.mean(0)) / (S_true.std(0) + 1e-12)
    S_est = (S_est - S_est.mean(0)) / (S_est.std(0) + 1e-12)
    n = S_true.shape[1]
    C = np.abs(S_true.T @ S_est / len(S_true))
    best = [float(C[i].max()) for i in range(n)]        # 每个源被恢复的最好程度
    return float(np.mean(best)), C


def main(seed=SEED):
    X, S, desc = data.mixed_signals(seed=seed)
    exp = Exp(exp_id="L3-05", algo="ICA 独立成分分析", layer="L3 降维与矩阵分解",
              purpose="验证 ICA 对非高斯源的盲源分离能力，并与 PCA 对照说明「不相关≠独立」",
              dataset="%s；源含正弦/方波/均匀噪声，均非高斯" % desc,
              baselines=["PCA（仅保证不相关）"],
              metrics=["平均最佳 |相关系数|", "最差源恢复 |相关系数|", "耗时"],
              seed=seed, params={"n_components": 3, "max_iter": 500,
                                "whiten": "unit-variance"},
              env=env_info())

    with exp.timer("FastICA fit"):
        ica = FastICA(n_components=3, whiten="unit-variance", max_iter=500,
                      random_state=seed).fit(X)
    S_ica = ica.transform(X)
    m_ica, C_ica = match_quality(S, S_ica)
    exp.add(方法="FastICA", 平均相关系数=m_ica,
            最差源相关系数=float(np.abs(C_ica).max(1).min()),
            说明=np.array2string(np.abs(C_ica).max(1), precision=3))

    with exp.timer("PCA fit"):
        pca = PCA(n_components=3, random_state=seed).fit(X)
    S_pca = pca.transform(X)
    m_pca, C_pca = match_quality(S, S_pca)
    exp.add(方法="PCA", 平均相关系数=m_pca,
            最差源相关系数=float(np.abs(C_pca).max(1).min()),
            说明=np.array2string(np.abs(C_pca).max(1), precision=3))

    exp.figure("l3_05_ica", "bar", "各源信号的最佳恢复 |相关系数|",
               ["源1(正弦)", "源2(方波)", "源3(均匀)"],
               [("FastICA", np.abs(C_ica).max(1).tolist()),
                ("PCA", np.abs(C_pca).max(1).tolist())],
               ylabel="|相关系数|", xlabel="源信号")
    exp.note("FastICA 平均相关系数 %.4f，PCA 仅 %.4f。差距来自目标函数：PCA 只要求二阶不相关，"
             "ICA 最大化非高斯性（独立性），因此能分离 PCA 无法处理的非高斯源。"
             % (m_ica, m_pca))
    exp.note("应用：设备多源振动分离（如主轴转频、齿轮啮合、背景噪声叠加）属于典型的 ICA 场景；"
             "若源近似高斯，ICA 失效，应退回 PCA。")
    exp.note("注意 ICA 输出的分量顺序与符号不固定，实验中用「最佳匹配相关系数」评价而非逐列比对。")
    exp.save()


if __name__ == "__main__":
    main()
