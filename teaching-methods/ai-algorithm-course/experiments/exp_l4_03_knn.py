"""L4-03 KNN 实验（k 值选择与维度灾难）

目的：①观察 k 值对偏差-方差的影响；②在高维噪声特征下验证距离度量失效（维度灾难）。
输入：digits(1797x64)；另构造 +200 / +800 个标准正态噪声维度的版本。
参数：k ∈ {1,3,5,9,15,25}；维度实验固定 k=5；分层抽样 30% 测试集；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, accuracy, data, env_info        # noqa: E402

SEED = 42
K_LIST = [1, 3, 5, 9, 15, 25]


def main(seed=SEED):
    X, y, desc = data.digits(seed)
    rng = np.random.default_rng(seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=seed, stratify=y)
    exp = Exp(exp_id="L4-03", algo="K 近邻 KNN", layer="L4 监督学习",
              purpose="分析 k 值的偏差-方差权衡，并验证维度灾难对距离度量类算法的影响",
              dataset="digits(1797x64)；%s；分层抽样 30%%；噪声维度实验 n_extra∈{0,200,800}"
                      % desc,
              baselines=["k=1（方差最大）", "k=25（偏差最大）"],
              metrics=["测试准确率", "训练耗时", "预测耗时"],
              seed=seed, params={"k_list": K_LIST, "dim_k": 5,
                                "noise_dims": [0, 200, 800]},
              env=env_info())

    accs = []
    for k in K_LIST:
        m = KNeighborsClassifier(n_neighbors=k)
        with exp.timer("fit k=%d" % k):
            m.fit(Xtr, ytr)
        with exp.timer("predict k=%d" % k):
            p = m.predict(Xte)
        accs.append(accuracy(yte, p))
        exp.add(实验="k 值扫描", 设置="k=%d" % k, 维度=64, 测试准确率=accs[-1])

    dims, dim_acc = [], []
    for extra in [0, 200, 800]:
        if extra:
            Xtr2 = np.hstack([Xtr, rng.normal(0, 1, (len(Xtr), extra))])
            Xte2 = np.hstack([Xte, rng.normal(0, 1, (len(Xte), extra))])
        else:
            Xtr2, Xte2 = Xtr, Xte
        m = KNeighborsClassifier(n_neighbors=5)
        with exp.timer("fit dim=%d" % Xtr2.shape[1]):
            m.fit(Xtr2, ytr)
        with exp.timer("predict dim=%d" % Xtr2.shape[1]):
            p = m.predict(Xte2)
        a = accuracy(yte, p)
        dims.append(Xtr2.shape[1])
        dim_acc.append(a)
        exp.add(实验="维度灾难", 设置="k=5, 加 %d 噪声维" % extra,
                维度=Xtr2.shape[1], 测试准确率=a)

    exp.figure("l4_03_knn_k", "line", "KNN 准确率 vs k（digits, 64 维）",
               [str(k) for k in K_LIST], [("测试准确率", accs)],
               ylabel="准确率", xlabel="k")
    exp.figure("l4_03_knn_dim", "bar", "维度灾难：准确率随噪声维度下降",
               ["%d 维" % d for d in dims], [("k=5 准确率", dim_acc)],
               ylabel="准确率", xlabel="特征维度")
    best_k = K_LIST[int(np.argmax(accs))]
    exp.note("k 值扫描：k=1 准确率 %.4f，最优 k=%d 时 %.4f，k=25 时 %.4f。"
             % (accs[0], best_k, max(accs), accs[-1]))
    exp.note("本数据集（digits）类间分离度高，因此小 k 反而最优——这提示「k 越大越稳健」并非普适结论；"
             "在噪声大、类别重叠的数据上，k=1 会被孤立错标样本带偏，k 稍大更稳。"
             "工程上 k 必须用交叉验证选，且通常取奇数以避免平票。")
    exp.note("k 增大到 25 时准确率下降 %.4f：邻域半径扩大后跨类别样本被卷入投票，偏差上升。"
             % (max(accs) - accs[-1]))
    exp.note("维度灾难：特征从 64 维增至 864 维（新增的全是纯噪声）时，准确率由 %.4f 掉到 %.4f。"
             "高维空间中最近邻与最远邻的距离趋于相同，欧氏距离失去区分度。"
             % (dim_acc[0], dim_acc[-1]))
    exp.note("工程提示：KNN 必须标准化；高维场景应先 PCA 降维或改用树模型/线性模型；"
             "预测阶段需要遍历训练集，大数据量下应换用近似最近邻（如 faiss）。")
    exp.save()


if __name__ == "__main__":
    main()
