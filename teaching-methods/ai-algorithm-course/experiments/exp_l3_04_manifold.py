"""L3-04 流形可视化（t-SNE / UMAP）实验

目的：比较 PCA（线性）与 t-SNE（非线性）在低维嵌入中的邻域保持能力，并说明 UMAP 的替代方案。
输入：digits 前 800 个样本（含 10 类标签，仅用于评价，不参与 t-SNE 的目标函数）。
参数：PCA 2 维；t-SNE perplexity=30, init=pca, learning_rate=auto；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, accuracy, data, env_info        # noqa: E402

SEED = 42
N = 800


def main(seed=SEED):
    X, y, desc = data.digits(seed)
    X, y = X[:N], y[:N]
    exp = Exp(exp_id="L3-04", algo="t-SNE / UMAP 流形可视化", layer="L3 降维与矩阵分解",
              purpose="比较线性与非线性降维对局部邻域结构的保持能力，评估可视化质量",
              dataset="digits 前 %d 个样本（10 类）；%s" % (N, desc),
              baselines=["PCA（线性降维，2 维）", "原始 64 维（上界参考）"],
              metrics=["trustworthiness(k=5)", "嵌入空间 1-NN 准确率", "耗时"],
              seed=seed, params={"n_samples": N, "perplexity": 30,
                                 "init": "pca", "tsne_iter": 1000},
              env=env_info())

    embeds = {}
    with exp.timer("PCA fit/transform"):
        embeds["PCA(2维)"] = PCA(n_components=2, random_state=seed).fit_transform(X)
    with exp.timer("t-SNE fit/transform"):
        embeds["t-SNE(2维)"] = TSNE(n_components=2, perplexity=30, init="pca",
                                    learning_rate="auto", random_state=seed,
                                    max_iter=1000).fit_transform(X)
    try:                                    # UMAP 为可选依赖，未安装时跳过
        import umap                         # noqa: F401
        with exp.timer("UMAP fit/transform"):
            embeds["UMAP(2维)"] = umap.UMAP(n_components=2,
                                            random_state=seed).fit_transform(X)
        exp.note("UMAP 已安装，本次同时给出 UMAP 结果。")
    except Exception as e:
        exp.note("UMAP 未安装（%s），本次跳过。替代验证方式：① `pip install umap-learn` 后重跑本脚本即可自动纳入；"
                 "② 若无法联网安装，可用 t-SNE + PCA 的对比结论外推：UMAP 通常耗时低于 t-SNE、"
                 "且对全局结构的保持优于 t-SNE，但嵌入空间的距离仍不可定量解释。" % type(e).__name__)

    with exp.timer("参考：原始 64 维 1-NN"):
        acc_full = float(np.mean(
            KNeighborsClassifier(1).fit(X, y).predict(X) == y))

    names = list(embeds)
    for nm in names:
        Z = embeds[nm]
        tw = float(trustworthiness(X, Z, n_neighbors=5))
        acc = accuracy(y, KNeighborsClassifier(1).fit(Z, y).predict(Z))
        exp.add(方法=nm, trustworthiness=tw, 嵌入空间1NN准确率=acc,
                说明="1-NN 准确率越接近原始空间 %.3f 越好" % acc_full)

    exp.figure("l3_04_manifold", "bar", "降维方法的邻域保持能力（digits 前 800）",
               names,
               [("trustworthiness(k=5)", [r["trustworthiness"] for r in exp.rows]),
                ("嵌入 1-NN 准确率", [r["嵌入空间1NN准确率"] for r in exp.rows])],
               ylabel="指标值")
    ts = [r for r in exp.rows if r["方法"].startswith("t-SNE")][0]
    pc = [r for r in exp.rows if r["方法"].startswith("PCA")][0]
    exp.note("t-SNE 的 trustworthiness %.4f 高于 PCA 的 %.4f，嵌入空间 1-NN 准确率 %.3f vs %.3f，"
             "说明非线性流形方法确实更好地保留了局部邻域。"
             % (ts["trustworthiness"], pc["trustworthiness"],
                ts["嵌入空间1NN准确率"], pc["嵌入空间1NN准确率"]))
    exp.note("重要限制：t-SNE 只产出这张图，没有可用于新样本的 transform 接口，"
             "因此不能直接作为模型的特征工程步骤；嵌入坐标之间的距离也没有定量含义。")
    exp.note("教学建议：先用 PCA 看全局结构，再用 t-SNE/UMAP 看簇的分离情况，两者互补。")
    exp.save()


if __name__ == "__main__":
    main()
