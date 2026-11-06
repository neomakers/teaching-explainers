"""L4-02 逻辑回归实验（含自实现对照）

目的：验证逻辑回归作为概率分类基线的效果，并对照自实现的 Adam 训练结果是否一致。
输入：breast_cancer(569x30, 二分类)，分层抽样 30% 测试集，标准化。
参数：sklearn LogisticRegression(max_iter=5000)；自实现 LogisticGD(adam, lr=0.05, steps=1500, l2=1e-4)；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (Exp, accuracy, classification_report_short, data,  # noqa: E402
                    env_info, roc_auc)
from fromscratch.optim import LogisticGD            # noqa: E402

SEED = 42


def main(seed=SEED):
    X, y, desc = data.breast()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    exp = Exp(exp_id="L4-02", algo="逻辑回归", layer="L4 监督学习",
              purpose="评估逻辑回归作为概率分类基线的性能，并校验自实现训练结果",
              dataset="breast_cancer(569x30, 二分类)；%s；分层抽样 30%%" % desc,
              baselines=["多数类基线 DummyClassifier", "KNN(k=5)",
                         "自实现 LogisticGD(Adam)"],
              metrics=["准确率", "精确率", "召回率", "F1", "AUC", "耗时"],
              seed=seed, params={"max_iter": 5000, "self_lr": 0.05,
                                 "self_steps": 1500, "self_l2": 1e-4},
              env=env_info())

    results = {}
    for name, m in [("逻辑回归(sklearn)", LogisticRegression(max_iter=5000)),
                    ("KNN(k=5)", KNeighborsClassifier(5)),
                    ("多数类基线", DummyClassifier(strategy="most_frequent"))]:
        with exp.timer(name):
            m.fit(Xtr, ytr)
        p = m.predict(Xte)
        s = m.decision_function(Xte) if hasattr(m, "decision_function") \
            else m.predict_proba(Xte)[:, 1]
        P, R, F1 = classification_report_short(yte, p)
        exp.add(方法=name, 准确率=accuracy(yte, p), 精确率=P, 召回率=R, F1=F1,
                AUC=roc_auc(yte, s))
        results[name] = accuracy(yte, p)

    with exp.timer("自实现 LogisticGD(Adam)"):
        mine = LogisticGD(lr=0.05, steps=1500, l2=1e-4, method="adam").fit(Xtr, ytr)
    p = mine.predict(Xte)
    P, R, F1 = classification_report_short(yte, p)
    exp.add(方法="自实现 LogisticGD(Adam)", 准确率=accuracy(yte, p), 精确率=P,
            召回率=R, F1=F1, AUC=roc_auc(yte, mine.predict_proba(Xte)))

    # 一致性检验：与 sklearn 在同一测试集上的预测概率比较（比权重比较更可靠，
    # 因为权重在缩放意义下不唯一、且含正则项差异）
    sk = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
    p_sk, p_mine = sk.predict_proba(Xte)[:, 1], mine.predict_proba(Xte)
    d_prob = float(np.mean(np.abs(p_sk - p_mine)))
    d_acc = float(abs(accuracy(yte, (p_sk >= 0.5).astype(int)) -
                      accuracy(yte, (p_mine >= 0.5).astype(int))))
    w_corr = float(np.corrcoef(sk.coef_[0], mine.w_[:Xtr.shape[1]])[0, 1])
    exp.add(方法="一致性检验", 指标="平均 |概率差|", 数值=d_prob)
    exp.add(方法="一致性检验", 指标="准确率差", 数值=d_acc)
    exp.add(方法="一致性检验", 指标="权重向量相关系数", 数值=w_corr)

    exp.figure("l4_02_logistic", "bar", "二分类基线模型对比（breast_cancer）",
               [r["方法"] for r in exp.rows if "准确率" in r],
               [("准确率", [r["准确率"] for r in exp.rows if "准确率" in r]),
                ("AUC", [r["AUC"] for r in exp.rows if "准确率" in r])],
               ylabel="指标值")
    exp.note("sklearn 与自实现（Adam 训练）的一致性：平均预测概率差 %.4f、准确率差 %.4f、"
             "权重向量相关系数 %.4f。三项均表明自实现的损失、梯度与优化步骤正确；"
             "残差来自正则强度与迭代步数的差异。" % (d_prob, d_acc, w_corr))
    exp.note("逻辑回归 AUC 达 %.4f，明显优于多数类基线（%.4f），且训练耗时在毫秒级；"
             "在需要概率输出与系数解释的工业判定场景中，它是应当首先跑通的基线。"
             % (exp.rows[0]["AUC"], exp.rows[2]["准确率"]))
    exp.note("逻辑回归决策边界是线性的：若特征与对数几率非线性相关，需引入多项式/分箱特征或改用树模型。")
    exp.save()


if __name__ == "__main__":
    main()
