"""L3-02 LDA 线性判别分析实验

目的：对比有监督降维（LDA）与无监督降维（PCA）在分类任务上的差异，并验证 LDA 的维度上限 C-1。
输入：wine(178x13, 3 类)、breast_cancer(569x30, 2 类)，分层抽样 30% 测试集。
参数：wine 降到 2 维，breast 降到 1 维；下游 1-NN 与 LogisticRegression；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, accuracy, data, env_info        # noqa: E402

SEED = 42


def main(seed=SEED):
    exp = Exp(exp_id="L3-02", algo="LDA 线性判别分析", layer="L3 降维与矩阵分解",
              purpose="对比有监督降维与无监督降维的分类收益，并验证 LDA 维度上限 = 类别数-1",
              dataset="wine(178x13, 3 类) 与 breast_cancer(569x30, 2 类)，分层抽样 30%",
              baselines=["PCA 同维降维", "原始特征全维"],
              metrics=["降维维数", "1-NN 准确率", "逻辑回归准确率", "耗时"],
              seed=seed, params={"wine_k": 2, "breast_k": 1,
                                 "下游": ["1-NN", "LogisticRegression"]},
              env=env_info())

    for name, loader, k in [("wine", data.wine, 2), ("breast_cancer", data.breast, 1)]:
        X, y, desc = loader()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                              random_state=seed, stratify=y)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        n_cls = len(np.unique(y))
        variants = {"原始特征": (Xtr, Xte)}
        with exp.timer("%s/LDA fit" % name):
            lda = LinearDiscriminantAnalysis(n_components=k).fit(Xtr, ytr)
        variants["LDA(%d维)" % k] = (lda.transform(Xtr), lda.transform(Xte))
        with exp.timer("%s/PCA fit" % name):
            pca = PCA(n_components=k, random_state=seed).fit(Xtr)
        variants["PCA(%d维)" % k] = (pca.transform(Xtr), pca.transform(Xte))
        for tag, (A, B) in variants.items():
            for clf_name, clf in [("1-NN", KNeighborsClassifier(1)),
                                  ("LogisticRegression",
                                   LogisticRegression(max_iter=5000))]:
                with exp.timer("%s/%s/%s" % (name, tag, clf_name)):
                    clf.fit(A, ytr)
                    p = clf.predict(B)
                acc = accuracy(yte, p)
                exp.add(数据集=name, 类别数=n_cls, 方法=tag, 降维维数=B.shape[1],
                        下游模型=clf_name, 测试准确率=acc)
        exp.note("%s：类别数 %d，LDA 最多只能降到 %d 维（受类间散度矩阵秩的限制）；"
                 "wine 上 LDA 2 维的 1-NN 准确率 %.3f 对比 PCA 2 维 %.3f。"
                 % (name, n_cls, n_cls - 1,
                    [r["测试准确率"] for r in exp.rows
                     if r["数据集"] == name and r["方法"] == "LDA(%d维)" % k
                     and r["下游模型"] == "1-NN"][0],
                    [r["测试准确率"] for r in exp.rows
                     if r["数据集"] == name and r["方法"] == "PCA(%d维)" % k
                     and r["下游模型"] == "1-NN"][0]))

    exp.figure("l3_02_lda", "bar", "LDA 与 PCA 降维后的下游准确率",
               ["%s/%s/%s" % (r["数据集"], r["方法"], r["下游模型"])
                for r in exp.rows],
               [("测试准确率", [r["测试准确率"] for r in exp.rows])],
               ylabel="测试准确率")
    exp.note("在 breast_cancer（2 类）上 LDA 只能降到 1 维，却仍保留了几乎全部判别信息："
             "它最大化类间/类内散度比，而 PCA 只追求方差最大，与标签无关。")
    exp.note("工程结论：分类任务做降维时优先 LDA；若需可视化或无标签，则用 PCA / t-SNE / UMAP。")
    exp.save()


if __name__ == "__main__":
    main()
