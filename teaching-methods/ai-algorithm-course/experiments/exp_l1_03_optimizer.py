"""L1-03 梯度下降族（SGD / Momentum / RMSProp / Adam）实验

目的：在同一凸目标（岭回归）上比较四种一阶优化器的收敛速度与学习率鲁棒性。
输入：合成工艺数据 synth_process(n=1200, seed=42)，特征标准化后加偏置列。
参数：steps=500；学习率 0.05 与 0.01 两档；l2=1e-2；种子 42。
基线：岭回归闭式解（正规方程）。
"""
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, data, env_info              # noqa: E402
from fromscratch.optim import minimize, RidgeGD      # noqa: E402

SEED = 42
STEPS = 500
LRS = [0.05, 0.01]


def main(seed=SEED):
    X, y, desc = data.synth_process(seed=seed)
    Xs = StandardScaler().fit_transform(X)
    Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
    obj = RidgeGD(Xb, y, l2=1e-2)
    n, d = Xb.shape
    with_exp = Exp(exp_id="L1-03", algo="梯度下降族 (SGD/Momentum/RMSProp/Adam)",
                   layer="L1 数学基础",
                   purpose="比较一阶优化器的收敛速度、最终精度与学习率鲁棒性",
                   dataset="synth_process(n=1200)；%s；标准化后维度 %d" % (desc, d),
                   baselines=["岭回归闭式解（正规方程）"],
                   metrics=["最终损失", "达到最优损失 1% 邻域所需步数", "耗时"],
                   seed=seed, params={"steps": STEPS, "lr_list": LRS, "l2": 1e-2},
                   env=env_info())

    A = Xb.T @ Xb + n * 1e-2 * np.eye(d)
    w_star = np.linalg.solve(A, Xb.T @ y)
    loss_star = float(obj(w_star)[0])
    with_exp.add(方法="闭式解（基线）", 学习率="-", 最终损失=loss_star,
                 收敛步数=0, 说明="正规方程，作为最优参考")

    curves = {}
    for lr in LRS:
        for m in ["sgd", "momentum", "rmsprop", "adam"]:
            with with_exp.timer("%s lr=%g" % (m, lr)):
                w, hist = minimize(obj, np.zeros(d), method=m, lr=lr,
                                   steps=STEPS)
            hit = next((i for i, h in enumerate(hist)
                        if h <= loss_star * 1.01 + 1e-12), STEPS + 1)
            with_exp.add(方法=m, 学习率=lr, 最终损失=float(hist[-1]),
                         收敛步数=int(hit),
                         说明="步数越小越快收敛（>%d 表示未收敛）" % STEPS)
            if lr == LRS[0]:
                curves[m] = [hist[i] for i in range(0, len(hist), 10)]

    with_exp.figure("l1_03_optimizer", "line",
                    "优化器收敛曲线（lr=0.05，每 10 步采样）",
                    [str(i * 10) for i in range(STEPS // 10 + 1)],
                    [(k, v) for k, v in curves.items()],
                    ylabel="损失", xlabel="迭代步数")
    rows = [r for r in with_exp.rows if r.get("方法") not in ("闭式解（基线）",)]
    for lr in LRS:
        sub = [r for r in rows if r["学习率"] == lr]
        conv = sorted([r for r in sub if r["收敛步数"] <= STEPS],
                      key=lambda r: r["收敛步数"])
        slow = [r for r in sub if r["收敛步数"] > STEPS]
        if conv:
            with_exp.note("lr=%g：%d 种优化器全部收敛（最慢 %s 用 %d 步），收敛步数排序：%s；"
                          "最终损失均达到闭式解水平 %.6f。"
                          % (lr, len(conv), conv[-1]["方法"],
                             conv[-1]["收敛步数"],
                             " < ".join("%s(%d)" % (r["方法"], r["收敛步数"])
                                        for r in conv), loss_star))
        if slow:
            with_exp.note("lr=%g：%s 在 %d 步内未收敛（最终损失 %s，闭式解 %.6f）——"
                          "步长缩小后收敛变慢，Adam 的二阶矩估计还会进一步缩小有效步长，"
                          "因此自适应优化器同样需要合适的学习率与足够的迭代预算。"
                          % (lr, "、".join(r["方法"] for r in slow), STEPS,
                             "、".join("%.4f" % r["最终损失"] for r in slow),
                             loss_star))
    with_exp.note("教学要点：本目标是二次函数且已标准化（条件数较小），各优化器差距有限；"
                  "目标函数条件数越差、越非凸，自适应优化器（RMSProp/Adam）的优势才越明显。"
                  "不要把它讲成「免调参」——学习率仍是第一超参。")
    with_exp.save()


if __name__ == "__main__":
    main()
