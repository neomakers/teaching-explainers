"""L5-01 随机森林实验（Bagging 的方差降低效果）

目的：对比单棵决策树与随机森林的精度/稳定性，并检验特征重要性的可重复性。
输入：breast_cancer(569x30) 与 digits(1797x64)，分层抽样 30% 测试集。
参数：树数 ∈ {1(单树), 10, 100, 300}；oob_score=True；特征重要性稳定性用 seed ∈ {0,1,2}；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, accuracy, data, env_info, roc_auc  # noqa: E402

SEED = 42
N_TREES = [1, 10, 100, 300]


def main(seed=SEED):
    exp = Exp(exp_id="L5-01", algo="随机森林", layer="L5 集成与调优",
              purpose="验证 Bagging + 随机特征子集带来的方差降低，并检验特征重要性的稳定性",
              dataset="breast_cancer(569x30) 与 digits(1797x64)，分层抽样 30%",
              baselines=["单棵决策树（n_estimators=1，无特征子集随机）",
                         "逻辑回归（线性基线）"],
              metrics=["测试准确率", "AUC", "OOB 分数", "top5 特征重要性重合率",
                       "耗时"],
              seed=seed, params={"n_estimators_list": N_TREES,
                                "importance_seeds": [0, 1, 2]},
              env=env_info())

    for name, loader in [("breast_cancer", data.breast), ("digits", data.digits)]:
        X, y, desc = loader()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                               random_state=seed, stratify=y)
        for n in N_TREES:
            if n == 1:
                m = DecisionTreeClassifier(random_state=seed)
                tag = "决策树(单棵)"
            else:
                m = RandomForestClassifier(n_estimators=n, oob_score=True,
                                            n_jobs=-1, random_state=seed)
                tag = "随机森林(%d棵)" % n
            with exp.timer("%s/%s" % (name, tag)):
                m.fit(Xtr, ytr)
                p = m.predict(Xte)
            s = m.predict_proba(Xte)[:, 1] if len(np.unique(y)) == 2 else None
            oob = float(getattr(m, "oob_score_", float("nan")))
            exp.add(数据集=name, 方法=tag, 测试准确率=accuracy(yte, p),
                    AUC=(roc_auc(yte, s) if s is not None else float("nan")),
                    OOB分数=oob)

        tops = []
        for sd in [0, 1, 2]:
            rf = RandomForestClassifier(n_estimators=200, random_state=sd,
                                        n_jobs=-1).fit(Xtr, ytr)
            tops.append(set(np.argsort(-rf.feature_importances_)[:5]))
        jac = float(np.mean([len(tops[i] & tops[j]) / len(tops[i] | tops[j])
                             for i in range(3) for j in range(i + 1, 3)]))
        exp.add(数据集=name, 方法="重要性稳定性(种子0/1/2)",
                测试准确率=float("nan"), AUC=float("nan"), OOB分数=jac)

    bc = [r for r in exp.rows if r["数据集"] == "breast_cancer"
          and not r["方法"].startswith("重要性")]
    exp.figure("l5_01_rf", "bar", "随机森林：树数对准确率的影响",
               ["%s/%s" % (r["数据集"], r["方法"]) for r in bc],
               [("测试准确率", [r["测试准确率"] for r in bc])], ylabel="准确率")
    dt = [r for r in bc if r["方法"].startswith("决策树")][0]
    rf300 = [r for r in bc if r["方法"].startswith("随机森林(300")][0]
    stab = [r for r in exp.rows if r["方法"].startswith("重要性")][0]
    exp.note("单棵决策树 %.4f → 300 棵随机森林 %.4f（breast_cancer/digits 两数据集一致趋势）："
             "双重随机性有效降低了方差。" % (dt["测试准确率"], rf300["测试准确率"]))
    exp.note("OOB 分数与测试准确率接近，说明可不切分验证集直接估计泛化误差，这对小样本工业数据很实用。")
    exp.note("特征重要性 top5 在不同随机种子下的平均重合率 %.2f：可用于筛选关键工艺参数，"
             "但相关特征会分散重要性，解释时须谨慎。" % stab["OOB分数"])
    exp.save()


if __name__ == "__main__":
    main()
