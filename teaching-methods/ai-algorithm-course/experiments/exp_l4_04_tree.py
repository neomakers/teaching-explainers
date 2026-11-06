"""L4-04 决策树实验（深度与可解释性的权衡）

目的：观察树深度对过拟合的影响，并量化"可解释性代价"（规则数 vs 精度）。
输入：wine(178x13, 3 类) 与 breast_cancer(569x30, 2 类)，分层抽样 30% 测试集。
参数：max_depth ∈ {1,2,3,5,None}；criterion=gini；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, accuracy, data, env_info        # noqa: E402

SEED = 42
DEPTHS = [1, 2, 3, 5, None]


def main(seed=SEED):
    exp = Exp(exp_id="L4-04", algo="决策树", layer="L4 监督学习",
              purpose="观察树深度对过拟合的影响，量化精度与可解释性的权衡",
              dataset="wine(178x13, 3 类) 与 breast_cancer(569x30, 2 类)，分层抽样 30%",
              baselines=["决策树桩(max_depth=1)", "无深度限制(完全生长)"],
              metrics=["训练准确率", "测试准确率", "叶子数", "规则条数", "耗时"],
              seed=seed, params={"max_depth_list": DEPTHS, "criterion": "gini"},
              env=env_info())

    for name, loader in [("wine", data.wine), ("breast_cancer", data.breast)]:
        X, y, desc = loader()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                              random_state=seed, stratify=y)
        cols = None
        for d in DEPTHS:
            m = DecisionTreeClassifier(max_depth=d, criterion="gini",
                                       random_state=seed)
            with exp.timer("%s depth=%s" % (name, d)):
                m.fit(Xtr, ytr)
            tr = accuracy(ytr, m.predict(Xtr))
            te = accuracy(yte, m.predict(Xte))
            rules = export_text(m).count("\n|---") + export_text(m).count("\n|-")
            exp.add(数据集=name, max_depth=("不限" if d is None else d),
                    训练准确率=tr, 测试准确率=te, 叶子数=int(m.get_n_leaves()),
                    规则条数=int(rules))
            if name == "wine" and d == 3:
                cols = export_text(m, max_depth=3)
        if cols:
            exp.note("wine 上 max_depth=3 的树只有 %d 条规则，可直接读成工艺判定逻辑（节选）：\n%s"
                     % (int(export_text(
                         DecisionTreeClassifier(max_depth=3, random_state=seed)
                         .fit(Xtr, ytr)).count("|---")), cols[:400]))

    wine_rows = [r for r in exp.rows if r["数据集"] == "wine"]
    exp.figure("l4_04_tree", "line", "决策树训练/测试准确率 vs 深度（wine）",
               [str(r["max_depth"]) for r in wine_rows],
               [("训练准确率", [r["训练准确率"] for r in wine_rows]),
                ("测试准确率", [r["测试准确率"] for r in wine_rows])],
               ylabel="准确率", xlabel="max_depth")
    w0, wlast = wine_rows[0], wine_rows[-1]
    exp.note("wine 上深度 1（树桩）测试准确率 %.4f，深度不限时 %.4f，但训练准确率升到 %.4f——"
             "训练与测试的差距就是过拟合的直接证据。"
             % (w0["测试准确率"], wlast["测试准确率"], wlast["训练准确率"]))
    exp.note("规则数随深度指数增长：可解释性是有代价的，工程上常把 max_depth 限制在 3~5，"
             "用少量精度损失换取可审核的判定逻辑。")
    exp.note("单棵树对样本扰动敏感（方差大），这正是需要随机森林/梯度提升的原因（见 L5 实验）。")
    exp.save()


if __name__ == "__main__":
    main()
