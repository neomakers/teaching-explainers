"""L6-02 DBSCAN 实验（任意形状簇与噪声识别）

目的：在非凸簇（月牙、同心圆）上对比密度聚类与划分聚类，并观察 ε 参数的敏感性。
输入：make_moons(n=500, noise=0.10) 与 make_circles(n=500, noise=0.06)。
参数：eps ∈ {0.1, 0.2, 0.3, 0.5}；min_samples=5；KMeans(k=2) 与 Ward 作对照；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, adjusted_rand, data, env_info   # noqa: E402

SEED = 42
EPS = [0.1, 0.2, 0.3, 0.5]


def main(seed=SEED):
    exp = Exp(exp_id="L6-02", algo="DBSCAN", layer="L6 无监督学习",
              purpose="验证密度聚类对任意形状簇与噪声点的处理能力，并量化 eps 的敏感性",
              dataset="make_moons(n=500, noise=0.10) 与 make_circles(n=500, noise=0.06)",
              baselines=["KMeans(k=2)", "层次聚类 Ward(k=2)"],
              metrics=["ARI", "发现簇数(不含噪声)", "噪声点数", "轮廓系数", "耗时"],
              seed=seed, params={"eps_list": EPS, "min_samples": 5},
              env=env_info())

    for name, loader in [("moons", data.moons), ("circles", data.circles)]:
        X, y, desc = loader()
        for eps in EPS:
            m = DBSCAN(eps=eps, min_samples=5)
            with exp.timer("%s/DBSCAN eps=%g" % (name, eps)):
                lab = m.fit_predict(X)
            n_cl = len(set(lab)) - (1 if -1 in lab else 0)
            n_noise = int(np.sum(lab == -1))
            sil = silhouette_score(X, lab) if n_cl > 1 else float("nan")
            exp.add(数据集=name, 方法="DBSCAN(eps=%g)" % eps, ARI=adjusted_rand(y, lab),
                    发现簇数=n_cl, 噪声点数=n_noise, 轮廓系数=sil)
        for tag, m in [("KMeans(k=2)", KMeans(n_clusters=2, n_init=10,
                                               random_state=seed)),
                       ("Ward(k=2)", AgglomerativeClustering(n_clusters=2))]:
            with exp.timer("%s/%s" % (name, tag)):
                lab = m.fit_predict(X)
            exp.add(数据集=name, 方法=tag, ARI=adjusted_rand(y, lab),
                    发现簇数=len(set(lab)), 噪声点数=0,
                    轮廓系数=silhouette_score(X, lab))

    mo = [r for r in exp.rows if r["数据集"] == "moons"]
    exp.figure("l6_02_dbscan", "bar", "moons 数据集上的聚类质量",
               [r["方法"] for r in mo],
               [("ARI", [r["ARI"] for r in mo]),
                ("轮廓系数", [r["轮廓系数"] for r in mo])], ylabel="指标值")
    best_db = max([r for r in exp.rows if r["方法"].startswith("DBSCAN")],
                  key=lambda r: r["ARI"])
    km = [r for r in exp.rows if r["方法"] == "KMeans(k=2)"][0]
    exp.note("DBSCAN 最佳配置 %s（%s）ARI %.4f，而 KMeans(k=2) ARI 仅 %.4f："
             "基于质心的划分只能切出凸形区域，无法表达月牙/同心圆结构。"
             % (best_db["方法"], best_db["数据集"], best_db["ARI"], km["ARI"]))
    exp.note("eps 敏感性：过小则多数点被判为噪声，过大则所有点并成一簇。"
             "工程上用 k 距离图（k=min_samples）的拐点来选 eps，而非网格盲搜。")
    exp.note("DBSCAN 不需要预设簇数且能显式输出噪声点，非常适合工况异常初筛；"
             "但密度不均匀的数据集（疏密差异大）会失效，此时考虑 OPTICS 或 HDBSCAN。")
    exp.save()


if __name__ == "__main__":
    main()
