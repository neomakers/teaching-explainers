"""L4-05 支持向量机实验（核函数与软间隔）

目的：比较线性/多项式/RBF 核的表达力与代价，并观察软间隔参数 C 的作用。
输入：breast_cancer(569x30)（线性可分倾向）与合成 moons（非线性），标准化后 30% 测试集。
参数：C ∈ {0.1, 1, 10}；kernel ∈ {linear, poly(3), rbf}；gamma=scale；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, accuracy, data, env_info, roc_auc  # noqa: E402

SEED = 42
C_LIST = [0.1, 1.0, 10.0]


def main(seed=SEED):
    exp = Exp(exp_id="L4-05", algo="支持向量机 SVM", layer="L4 监督学习",
              purpose="比较不同核函数的表达力与计算代价，并观察软间隔参数 C 的作用",
              dataset="breast_cancer(569x30) 与 moons(n=500, noise=0.10)，标准化后 30% 测试集",
              baselines=["线性核（无核技巧）", "RBF 核默认参数"],
              metrics=["测试准确率", "AUC", "支持向量个数", "训练耗时"],
              seed=seed, params={"C_list": C_LIST, "kernels":
                                 ["linear", "poly(degree=3)", "rbf"],
                                 "gamma": "scale"},
              env=env_info())

    for name, loader in [("breast_cancer", data.breast), ("moons", data.moons)]:
        X, y, desc = loader()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                               random_state=seed, stratify=y)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        for kernel, kw in [("linear", {}), ("poly(degree=3)", {"degree": 3}),
                           ("rbf", {})]:
            for C in C_LIST:
                m = SVC(kernel=kernel.split("(")[0], C=C, gamma="scale",
                        random_state=seed, **kw)
                with exp.timer("%s/%s/C=%g" % (name, kernel, C)):
                    m.fit(Xtr, ytr)
                    p = m.predict(Xte)
                exp.add(数据集=name, 核函数=kernel, C=C, 测试准确率=accuracy(yte, p),
                        AUC=roc_auc(yte, m.decision_function(Xte)),
                        支持向量个数=int(np.sum(m.n_support_)))

    bc = [r for r in exp.rows if r["数据集"] == "breast_cancer"]
    mo = [r for r in exp.rows if r["数据集"] == "moons"]
    exp.figure("l4_05_svm_C", "line", "breast_cancer：C 对准确率的影响",
               ["C=0.1", "C=1(linear)", "C=10", "C=0.1", "C=1(rbf)", "C=10"],
               [("linear", [r["测试准确率"] for r in bc if r["核函数"] == "linear"]),
                ("rbf", [r["测试准确率"] for r in bc if r["核函数"] == "rbf"])],
               ylabel="测试准确率", xlabel="C（按核分组）")
    lin_m = max(r["测试准确率"] for r in mo if r["核函数"] == "linear")
    rbf_m = max(r["测试准确率"] for r in mo if r["核函数"] == "rbf")
    exp.note("moons（非线性）上线性核最高 %.4f，RBF 核最高 %.4f：核技巧把线性方法扩展到非线性边界，"
             "这是 SVM 处理「环形/月牙」分布的关键。" % (lin_m, rbf_m))
    exp.note("支持向量个数随 C 增大而减少（硬间隔倾向），同时训练时间上升；"
             "C 过大易过拟合噪声，C 过小则欠拟合，需交叉验证选择。")
    exp.note("规模提示：SVC 的训练复杂度约为 O(n²)~O(n³)，本实验样本量在千级以内尚可接受，"
             "万级以上建议用 LinearSVC 或核近似（Nystroem / RBFSampler）。")
    exp.save()


if __name__ == "__main__":
    main()
