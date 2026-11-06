"""L7-01 多层感知机实验（网络结构与学习率）

目的：观察隐层规模、学习率与早停对 MLP 的影响，并与线性/树模型对比精度-耗时。
输入：digits(1797x64)，分层抽样 30% 测试集，像素已缩放至 [0,1]。
参数：hidden_layer_sizes ∈ {(32,), (64,32), (128,64)}；lr ∈ {0.001, 0.01}；
     max_iter=300；early_stopping=True；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, accuracy, data, env_info        # noqa: E402

SEED = 42
STRUCTS = [(32,), (64, 32), (128, 64)]
LRS = [0.001, 0.01]


def main(seed=SEED):
    X, y, desc = data.digits(seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                           random_state=seed, stratify=y)
    exp = Exp(exp_id="L7-01", algo="多层感知机 MLP", layer="L7 深度学习",
              purpose="观察网络结构、学习率与早停对 MLP 精度与耗时的影响",
              dataset="digits(1797x64)；%s；分层抽样 30%%" % desc,
              baselines=["逻辑回归（线性）", "随机森林(200棵)"],
              metrics=["测试准确率", "实际迭代数", "训练耗时"],
              seed=seed, params={"hidden_layer_sizes": [list(s) for s in STRUCTS],
                                "lr_list": LRS, "max_iter": 300,
                                "early_stopping": True},
              env=env_info())

    for tag, m in [("逻辑回归(基线)", LogisticRegression(max_iter=2000)),
                   ("随机森林(200棵)", RandomForestClassifier(n_estimators=200,
                                                              n_jobs=-1,
                                                              random_state=seed))]:
        with exp.timer(tag):
            m.fit(Xtr, ytr)
        exp.add(方法=tag, 结构="-", 学习率="-", 测试准确率=accuracy(yte, m.predict(Xte)),
                实际迭代数=0)

    for hs in STRUCTS:
        for lr in LRS:
            m = MLPClassifier(hidden_layer_sizes=hs, learning_rate_init=lr,
                              max_iter=300, early_stopping=True,
                              n_iter_no_change=15, random_state=seed)
            with exp.timer("MLP%s lr=%g" % (hs, lr)):
                m.fit(Xtr, ytr)
            exp.add(方法="MLP", 结构=str(hs), 学习率=lr,
                    测试准确率=accuracy(yte, m.predict(Xte)),
                    实际迭代数=int(m.n_iter_))

    rows = [r for r in exp.rows if r["方法"] == "MLP"]
    exp.figure("l7_01_mlp", "bar", "MLP：结构与学习率对准确率的影响",
               ["%s/lr=%g" % (r["结构"], r["学习率"]) for r in rows],
               [("测试准确率", [r["测试准确率"] for r in rows])], ylabel="准确率")
    best = max(rows, key=lambda r: r["测试准确率"])
    lin = [r for r in exp.rows if r["方法"].startswith("逻辑回归")][0]
    exp.note("最优配置 %s / lr=%g：测试准确率 %.4f，超过逻辑回归基线 %.4f，说明该数据存在非线性可分结构。"
             % (best["结构"], best["学习率"], best["测试准确率"], lin["测试准确率"]))
    exp.note("学习率 0.01 时迭代数明显少于 0.001，但过大易在早期触发早停、损失曲线震荡；"
             "early_stopping 能在验证集不再改善时自动停止，是防过拟合的低成本手段。")
    exp.note("耗时随隐层规模上升，但相比随机森林仍在同一量级；真正的差距要到图像/序列等需要用"
             "卷积或循环结构的任务上才体现（见 L7-02 / L7-03 对 torch 依赖的说明）。")
    exp.save()


if __name__ == "__main__":
    main()
