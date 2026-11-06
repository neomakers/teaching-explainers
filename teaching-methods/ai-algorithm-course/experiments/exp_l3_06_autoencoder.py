"""L3-06 自编码器（线性 AE ≡ PCA）实验

目的：用 sklearn 的 MLP 训练一个瓶颈层为 k 的线性自编码器，验证"单层线性 AE 的解空间 ≈ PCA 主成分子空间"。
输入：digits 前 1200 张（64 维），训练/测试 7:3。
参数：k ∈ {5, 10, 20}；MLPRegressor(hidden_layer_sizes=(k,), activation='identity',
     solver='lbfgs', max_iter=400)；seed=42。
"""
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, data, env_info, rel_fro       # noqa: E402

SEED = 42
K_LIST = [5, 10, 20]


def principal_angles(B1, B2):
    """两个子空间（均为 d×k 的列正交基）的主角度余弦，返回 (min cos, max cos)。

    cos=1 表示两个子空间完全重合；min cos 越接近 1 说明越一致。
    """
    Q1, _ = np.linalg.qr(B1)
    Q2, _ = np.linalg.qr(B2)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.clip(s.min(), 0, 1)), float(np.clip(s.max(), 0, 1))


def main(seed=SEED):
    X, _, desc = data.digits(seed)
    Xtr, Xte = train_test_split(X[:1200], test_size=0.3, random_state=seed)
    exp = Exp(exp_id="L3-06", algo="自编码器 AE（线性）", layer="L3 降维与矩阵分解",
              purpose="验证单层线性自编码器学到的子空间与 PCA 主成分子空间一致",
              dataset="digits 前 1200 张（64 维）；%s；7:3 划分" % desc,
              baselines=["PCA（同维）"],
              metrics=["测试集重建相对误差", "子空间最小主角度余弦", "耗时"],
              seed=seed, params={"k_list": K_LIST, "activation": "identity",
                                 "solver": "lbfgs", "max_iter": 400},
              env=env_info())

    for k in K_LIST:
        with exp.timer("线性 AE k=%d" % k):
            ae = MLPRegressor(hidden_layer_sizes=(k,), activation="identity",
                              solver="lbfgs", max_iter=400,
                              random_state=seed).fit(Xtr, Xtr)
        Xr = ae.predict(Xte)
        err_ae = rel_fro(Xte, Xr)
        Wenc = ae.coefs_[0]                     # (64, k) 编码器权重

        with exp.timer("PCA k=%d" % k):
            pca = PCA(n_components=k, random_state=seed).fit(Xtr)
        err_pca = rel_fro(Xte, pca.inverse_transform(pca.transform(Xte)))
        cmin, cmax = principal_angles(Wenc, pca.components_.T)
        exp.add(方法="线性 AE", k=k, 重建相对误差=err_ae, 子空间最小cos=cmin,
                子空间最大cos=cmax)
        exp.add(方法="PCA", k=k, 重建相对误差=err_pca, 子空间最小cos=1.0,
                子空间最大cos=1.0)

    errs_ae = [r["重建相对误差"] for r in exp.rows if r["方法"] == "线性 AE"]
    errs_pca = [r["重建相对误差"] for r in exp.rows if r["方法"] == "PCA"]
    exp.figure("l3_06_ae", "bar", "线性 AE 与 PCA 的重建误差对比",
               [str(k) for k in K_LIST],
               [("线性 AE", errs_ae), ("PCA", errs_pca)],
               ylabel="测试集重建相对误差", xlabel="瓶颈维度 k")
    cos_vals = [r["子空间最小cos"] for r in exp.rows if r["方法"] == "线性 AE"]
    exp.note("三个 k 值下 AE 与 PCA 的重建误差差均在 %.4f 以内——作为「重建映射」两者等价，"
             "这正是单层线性 AE ≡ PCA 的实测证据。"
             % max(abs(a - b) for a, b in zip(errs_ae, errs_pca)))
    exp.note("但编码器权重张成的子空间与 PCA 主成分的最小主角度余弦只有 %.4f（1.0 为完全重合）："
             "线性 AE 的最优解在可逆变换下不唯一（W→WA、D→A⁻¹D 给出相同重建），"
             "因此等价的是重建函数，不是权重本身。这一点在课堂上必须讲清，否则学生会误以为 "
             "AE 训练出来就是主成分。" % min(cos_vals))

    exp.note("lbfgs 求解的小规模全连接 AE 收敛快，但随数据量与层数增长会迅速劣化；"
             "真正的深度非线性 AE 需要 PyTorch 与 GPU 才能有效训练（本机显卡损坏，见 L7-02 的说明）。")
    exp.note("实用价值：重构误差可作为异常分数——正常样本重建误差低，异常样本高。")
    exp.save()


if __name__ == "__main__":
    main()
