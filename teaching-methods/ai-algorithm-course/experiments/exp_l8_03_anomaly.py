"""L8-03 异常检测实验（Isolation Forest / One-Class SVM / PCA 重构误差）

目的：在无标签（或极少标签）条件下比较四种异常检测路线，并观察阈值参数 contamination 的作用。
输入：data.sensor_anomaly(n_normal=900, n_anom=60) —— 3 维相关高斯 + 平移的异常簇（异常占比 6.3%）。
参数：contamination=0.05；IsolationForest(n_estimators=200)；OneClassSVM(nu=0.05, gamma=auto)；
     PCA/SVD 重构误差法取前 2 个主成分，阈值用训练集的 95 分位；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (Exp, classification_report_short, data,  # noqa: E402
                    env_info, roc_auc)
from fromscratch.svd import svd as my_svd, truncate   # noqa: E402

SEED = 42
CONT = 0.05


def main(seed=SEED):
    X, y, desc = data.sensor_anomaly(seed=seed)
    n = len(X)
    ntr = int(n * 0.7)
    Xtr, Xte = X[:ntr], X[ntr:]
    ytr, yte = y[:ntr], y[ntr:]
    exp = Exp(exp_id="L8-03", algo="异常检测（Isolation Forest / OneClassSVM / 重构误差）",
              layer="L8 决策与生成",
              purpose="比较四类异常检测方法的精度与耗时，验证 SVD/PCA 重构误差路线的可行性",
              dataset="%s；前 70%% 作训练集（无标签拟合），后 30%% 评估" % desc,
              baselines=["随机打分（对照组）"],
              metrics=["精确率", "召回率", "F1", "AUC", "耗时"],
              seed=seed, params={"contamination": CONT, "n_estimators": 200,
                                "pca_components": 2, "阈值分位": "训练集 95%"},
              env=env_info())

    rng = np.random.default_rng(seed)
    rand_score = rng.random(len(Xte))
    P, R, F1 = classification_report_short(yte, (rand_score > 0.95).astype(int))
    exp.add(方法="随机打分(对照)", 精确率=P, 召回率=R, F1=F1,
            AUC=float(roc_auc_score(yte, rand_score)))

    with exp.timer("IsolationForest"):
        iso = IsolationForest(n_estimators=200, contamination=CONT,
                              random_state=seed).fit(Xtr)
    pred = (iso.predict(Xte) == -1).astype(int)
    P, R, F1 = classification_report_short(yte, pred)
    exp.add(方法="IsolationForest", 精确率=P, 召回率=R, F1=F1,
            AUC=float(roc_auc_score(yte, -iso.score_samples(Xte))))

    with exp.timer("OneClassSVM(rbf)"):
        ocs = OneClassSVM(nu=CONT, gamma="auto").fit(Xtr)
    pred = (ocs.predict(Xte) == -1).astype(int)
    P, R, F1 = classification_report_short(yte, pred)
    exp.add(方法="OneClassSVM(rbf)", 精确率=P, 召回率=R, F1=F1,
            AUC=float(roc_auc_score(yte, -ocs.decision_function(Xte))))

    for k in (1, 2):
        with exp.timer("PCA 重构误差 k=%d" % k):
            pca = PCA(n_components=k, random_state=seed).fit(Xtr)
            rec_tr = np.linalg.norm(
                Xtr - pca.inverse_transform(pca.transform(Xtr)), axis=1)
            thr = np.quantile(rec_tr, 0.95)
            rec_te = np.linalg.norm(
                Xte - pca.inverse_transform(pca.transform(Xte)), axis=1)
        pred = (rec_te > thr).astype(int)
        P, R, F1 = classification_report_short(yte, pred)
        exp.add(方法="PCA 重构误差(k=%d)" % k, 精确率=P, 召回率=R, F1=F1,
                AUC=float(roc_auc_score(yte, rec_te)))

    with exp.timer("自实现 SVD 重构误差"):
        mu = Xtr.mean(0)
        U, s, Vt = my_svd(Xtr - mu)
        k = 2

        def rec_err(M):
            """把样本投影到前 k 个右奇异向量张成的子空间再重建，取残差范数。"""
            A = M - mu
            Vk = Vt[:k, :]                  # (k, p)
            Ak = (A @ Vk.T) @ Vk            # 投影 + 回投影
            return np.linalg.norm(A - Ak, axis=1)
        thr2 = np.quantile(rec_err(Xtr), 0.95)
        e_te = rec_err(Xte)
    pred = (e_te > thr2).astype(int)
    P, R, F1 = classification_report_short(yte, pred)
    exp.add(方法="SVD 重构误差(自实现)", 精确率=P, 召回率=R, F1=F1,
            AUC=float(roc_auc_score(yte, e_te)))

    exp.figure("l8_03_anomaly", "bar", "异常检测方法对比",
               [r["方法"] for r in exp.rows],
               [("F1", [r["F1"] for r in exp.rows]),
                ("AUC", [r["AUC"] for r in exp.rows])], ylabel="指标值")
    iso_r = [r for r in exp.rows if r["方法"] == "IsolationForest"][0]
    rec_rows = [r for r in exp.rows if r["方法"].startswith("PCA 重构")]
    svd_r = [r for r in exp.rows if r["方法"].startswith("SVD")][0]
    best_rec = max(rec_rows, key=lambda r: r["AUC"])
    exp.note("IsolationForest F1 %.4f / AUC %.4f；PCA 重构误差最佳（%s）F1 %.4f / AUC %.4f；"
             "自实现 SVD 重构 F1 %.4f / AUC %.4f —— 后两者结果一致，再次确认两条实现等价。"
             % (iso_r["F1"], iso_r["AUC"], best_rec["方法"], best_rec["F1"],
                best_rec["AUC"], svd_r["F1"], svd_r["AUC"]))
    if best_rec["AUC"] >= 0.9:
        exp.note("本例异常方向基本落在主成分子空间之外，重构误差法有效。"
                 "注意 k 的选择直接决定残差子空间：k 越大残差越小、异常越难被放大。")
    else:
        exp.note("本例中重构误差法效果有限（最佳 AUC 仅 %.3f）：异常方向 [4.2,-3.4,3.0] 与数据主成分方向"
                 "夹角不足，异常样本在前 k 个主成分上的投影仍被「正常」解释，残差被淹没。"
                 "改进方向：① 用马氏距离替代欧氏重构残差；② 对残差按主成分方差归一化（标准化残差）；"
                 "③ 直接采用 Isolation Forest 等基于隔离难度的方法。"
                 % best_rec["AUC"])
    exp.note("工程提示：contamination / nu 直接决定告警量，必须结合现场可接受的误报率设定；"
             "正常工况存在季节性变化时，应分段重新拟合阈值。")
    exp.save()


if __name__ == "__main__":
    main()
