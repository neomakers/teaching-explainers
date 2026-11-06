"""L2-01 标准化影响实验

目的：量化"是否标准化"对距离/梯度类模型的影响，并确认树模型不受影响。
输入：wine(178x13, 三分类) 与 breast_cancer(569x30, 二分类)，分层抽样 30% 作测试集。
参数：seed=42；KNN(k=5)、SVC(kernel=rbf, C=1, gamma=scale)、
      LogisticRegression(max_iter=5000)、RandomForest(200, 对照)。
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, accuracy, data, env_info        # noqa: E402

SEED = 42


def main(seed=SEED):
    exp = Exp(exp_id="L2-01", algo="标准化 / 归一化", layer="L2 数据表示",
              purpose="量化标准化对距离类与梯度类模型的影响，验证树模型不敏感",
              dataset="wine(178x13, 三分类) 与 breast_cancer(569x30, 二分类)，"
                      "分层抽样 test_size=0.3",
              baselines=["原始数据（不做标准化）"],
              metrics=["测试集准确率", "训练+预测耗时"],
              seed=seed, params={"test_size": 0.3, "knn_k": 5,
                                 "svc": "rbf, C=1, gamma=scale"},
              env=env_info())

    for name, loader in [("wine", data.wine), ("breast_cancer", data.breast)]:
        X, y, desc = loader()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y)
        sc = StandardScaler().fit(Xtr)
        for prep in ["raw", "standardized"]:
            Xtr2 = Xtr if prep == "raw" else sc.transform(Xtr)
            Xte2 = Xte if prep == "raw" else sc.transform(Xte)
            for model in ["KNN(k=5)", "SVC(rbf)", "LogisticRegression",
                          "RandomForest(对照)"]:
                m = {"KNN(k=5)": lambda: KNeighborsClassifier(5),
                     "SVC(rbf)": lambda: SVC(C=1, gamma="scale",
                                             random_state=seed),
                     "LogisticRegression": lambda: LogisticRegression(
                         max_iter=5000),
                     "RandomForest(对照)": lambda: RandomForestClassifier(
                         n_estimators=200, random_state=seed)}[model]()
                with exp.timer("%s/%s/%s" % (name, prep, model)):
                    m.fit(Xtr2, ytr)
                    p = m.predict(Xte2)
                exp.add(数据集=name, 预处理=prep, 模型=model,
                        测试准确率=accuracy(yte, p), 特征量纲差异=float(
                            np.max(Xtr.std(0) / (np.min(Xtr.std(0)) + 1e-9))))

    agg = {}
    for r in exp.rows:
        agg.setdefault((r["数据集"], r["模型"]), {})[r["预处理"]] = r["测试准确率"]
    labels = ["%s/%s" % k for k in agg]
    exp.figure("l2_01_scaling", "bar", "标准化对各模型测试准确率的影响",
               labels,
               [("原始", [v.get("raw", 0) for v in agg.values()]),
                ("标准化", [v.get("standardized", 0) for v in agg.values()])],
               ylabel="测试准确率")
    knn_wine = agg.get(("wine", "KNN(k=5)"), {})
    exp.note("wine 上 KNN 标准化后准确率由 %.3f 提升到 %.3f：KNN 直接用距离投票，量纲大的特征会主导距离，"
             "因此必须标准化。" % (knn_wine.get("raw", 0), knn_wine.get("standardized", 0)))
    exp.note("SVC(rbf) 同理对尺度敏感（gamma 作用于欧氏距离）；LogisticRegression 虽不依赖距离，"
             "但标准化能显著改善梯度下降的条件数、加快收敛。")
    exp.note("RandomForest 作为对照：基于阈值的划分对单调缩放不变，两种预处理结果完全一致——"
             "这正是「树模型不需要标准化」的实验依据。")
    exp.save()


if __name__ == "__main__":
    main()
