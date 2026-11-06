"""L5-02 梯度提升实验（GBDT / 直方图提升 / AdaBoost）

目的：比较 Boosting 三剑客与随机森林的精度-耗时权衡，并给出 XGBoost 缺失时的替代方案。
输入：breast_cancer(569x30) 与合成 classification(n=2000, 20 维)，分层抽样 30% 测试集。
参数：n_estimators=200；GBDT learning_rate=0.05, max_depth=3；AdaBoost lr=0.5；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, accuracy, data, env_info, roc_auc  # noqa: E402

SEED = 42
N = 200


def main(seed=SEED):
    exp = Exp(exp_id="L5-02", algo="GBDT / XGBoost / AdaBoost", layer="L5 集成与调优",
              purpose="比较梯度提升家族与随机森林的精度与训练耗时，并说明 XGBoost 不可用时替代方案",
              dataset="breast_cancer(569x30) 与 make_classification(n=2000, 20 维, 6 有效特征)",
              baselines=["逻辑回归", "随机森林(200棵)"],
              metrics=["测试准确率", "AUC", "训练耗时"],
              seed=seed, params={"n_estimators": N, "gbdt_lr": 0.05,
                                "gbdt_max_depth": 3, "adaboost_lr": 0.5},
              env=env_info())

    try:
        import xgboost as xgb
        has_xgb = True
    except Exception as e:
        has_xgb = False
        exp.note("xgboost 未安装（%s）。替代验证方式：sklearn 的 HistGradientBoostingClassifier "
                 "与 XGBoost 同为「直方图分桶 + 二阶梯度」的梯度提升实现，接口一致（fit/predict/ "
                 "predict_proba），本实验用它作为等价替代；如需 XGBoost 可 `pip install xgboost` 后重跑，"
                 "脚本会自动纳入对比。" % type(e).__name__)

    for name, loader in [("breast_cancer", data.breast),
                         ("synthetic(20d)", lambda: data.classification(seed=seed))]:
        X, y, desc = loader()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                               random_state=seed, stratify=y)
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        models = [
            ("逻辑回归(基线)", LogisticRegression(max_iter=5000), True),
            ("随机森林(200棵)", RandomForestClassifier(n_estimators=200,
                                                       n_jobs=-1,
                                                       random_state=seed), False),
            ("AdaBoost(200)", AdaBoostClassifier(n_estimators=N,
                                                 learning_rate=0.5,
                                                 random_state=seed), False),
            ("GradientBoosting(200)", GradientBoostingClassifier(
                n_estimators=N, learning_rate=0.05, max_depth=3,
                random_state=seed), False),
            ("HistGradientBoosting(200)", HistGradientBoostingClassifier(
                max_iter=N, learning_rate=0.05, random_state=seed), False),
        ]
        if has_xgb:
            models.append(("XGBoost(200)", xgb.XGBClassifier(
                n_estimators=N, learning_rate=0.05, max_depth=3,
                eval_metric="logloss", random_state=seed), False))
        for tag, m, scaled in models:
            A, B = (Xtr_s, Xte_s) if scaled else (Xtr, Xte)
            with exp.timer("%s/%s" % (name, tag)):
                m.fit(A, ytr)
                p = m.predict(B)
                s = m.predict_proba(B)[:, 1]
            exp.add(数据集=name, 方法=tag, 测试准确率=accuracy(yte, p),
                    AUC=roc_auc(yte, s))

    rows = [r for r in exp.rows if r["数据集"] == "synthetic(20d)"]
    exp.figure("l5_02_boost", "bar", "集成方法对比（synthetic 20 维）",
               [r["方法"] for r in rows],
               [("测试准确率", [r["测试准确率"] for r in rows]),
                ("AUC", [r["AUC"] for r in rows])], ylabel="指标值")
    best = max(exp.rows, key=lambda r: r["AUC"])
    exp.note("本轮最优：%s / %s，AUC %.4f、准确率 %.4f。"
             % (best["数据集"], best["方法"], best["AUC"], best["测试准确率"]))
    exp.note("耗时对比见上表：HistGradientBoosting 用直方图分桶，在 20 维以上数据上通常比 "
             "GradientBoosting 快一个量级，是中等规模表格数据的首选。")
    exp.note("AdaBoost 精度略低且对标签噪声敏感（见 L5-03）；随机森林几乎不需调参，是更稳妥的基线。")
    exp.save()


if __name__ == "__main__":
    main()
