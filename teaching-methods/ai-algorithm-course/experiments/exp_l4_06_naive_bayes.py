"""L4-06 朴素贝叶斯实验（独立性假设的边界）

目的：比较三种朴素贝叶斯变体，并用"人为制造强相关特征"验证条件独立性假设失效时的性能退化。
输入：① digits(1797x64)（像素非负，适合 Multinomial/Bernoulli）；② 强相关构造数据：
      X2 = [x, x+噪声, x+噪声, ...] 使特征高度相关，标签由第一个特征决定。
参数：GaussianNB / MultinomialNB / BernoulliNB(binarize=0.5)；对照 LogisticRegression；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, accuracy, data, env_info        # noqa: E402

SEED = 42


def make_correlated(n=1200, seed=SEED):
    """构造特征强相关数据：后 5 个特征都是第 1 个特征的复制 + 噪声。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    X = np.column_stack([x] + [x + rng.normal(0, 0.05, n) for _ in range(5)]
                        + [rng.normal(0, 1, n) for _ in range(4)])
    y = (x + rng.normal(0, 0.3, n) > 0).astype(int)
    return X, y


def main(seed=SEED):
    exp = Exp(exp_id="L4-06", algo="朴素贝叶斯", layer="L4 监督学习",
              purpose="比较 NB 三种变体，并验证条件独立性假设失效时的性能退化",
              dataset="digits(1797x64) 与 强相关构造数据(n=1200, 后 5 维为第 1 维的复制)",
              baselines=["逻辑回归（不做独立性假设）"],
              metrics=["测试准确率", "训练耗时"],
              seed=seed, params={"bernoulli_binarize": 0.5,
                                "logistic_max_iter": 2000},
              env=env_info())

    errs = []
    cases = [("digits", lambda: data.digits(seed)[:2]),
             ("强相关数据", lambda: make_correlated(seed=seed))]
    for name, fn in cases:
        X, y = fn()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                               random_state=seed, stratify=y)
        for tag, m in [("GaussianNB", GaussianNB()),
                       ("MultinomialNB", MultinomialNB()),
                       ("BernoulliNB", BernoulliNB(binarize=0.5)),
                       ("LogisticRegression(对照)",
                        LogisticRegression(max_iter=2000))]:
            try:
                with exp.timer("%s/%s" % (name, tag)):
                    m.fit(Xtr, ytr)
                    p = m.predict(Xte)
                exp.add(数据集=name, 方法=tag, 测试准确率=accuracy(yte, p))
            except ValueError as e:          # MultinomialNB 不接受负值
                exp.add(数据集=name, 方法=tag, 测试准确率=float("nan"))
                errs.append("%s 在 %s 上不可用：%s" % (tag, name, e))

    exp.figure("l4_06_nb", "bar", "朴素贝叶斯变体与对照（按数据集）",
               ["%s/%s" % (r["数据集"], r["方法"]) for r in exp.rows],
               [("测试准确率", [r["测试准确率"] for r in exp.rows])],
               ylabel="准确率")
    dg = {r["方法"]: r["测试准确率"] for r in exp.rows if r["数据集"] == "digits"}
    cr = {r["方法"]: r["测试准确率"] for r in exp.rows if r["数据集"] == "强相关数据"}
    exp.note("digits 上 MultinomialNB %.4f / BernoulliNB %.4f 明显优于 GaussianNB %.4f："
             "像素计数型特征更贴近多项式/伯努利分布假设。" %
             (dg.get("MultinomialNB", float("nan")),
              dg.get("BernoulliNB", float("nan")),
              dg.get("GaussianNB", float("nan"))))
    exp.note("强相关数据上 GaussianNB %.4f 相对逻辑回归 %.4f：重复特征被重复计入似然，"
             "独立性假设被严重违背时 NB 会出现系统性偏差（概率过于自信）。"
             % (cr.get("GaussianNB", float("nan")),
                cr.get("LogisticRegression(对照)", float("nan"))))
    exp.note("优点仍成立：NB 训练复杂度 O(np)，无需调参，小数据与文本分类上是极佳的第一基线。")
    for e in errs:
        exp.note("异常提示：%s" % e)
    exp.save()


if __name__ == "__main__":
    main()
