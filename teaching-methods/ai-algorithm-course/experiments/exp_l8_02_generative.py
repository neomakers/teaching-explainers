"""L8-02 生成模型实验（GAN / Diffusion）—— 不可行说明 + 密度估计替代验证

难以直接实验的原因：
  1. GAN / 扩散模型的训练需要 PyTorch 与显存；本机显卡损坏，只能纯 CPU，
     连 MNIST 级别的 GAN 也要数十分钟到数小时，且结果不稳定、难以复现；
  2. 生成模型的评价本身依赖 FID/IS 等指标，需要预训练的 Inception 网络（需下载权重）。
替代验证方案：把「生成」还原成它的数学本质——**从估计出的数据分布中采样**，
  用两组可控的密度模型做定量对照：
    M1 高斯混合模型 GMM（参数化，可解析采样）；
    M2 核密度估计 KDE（非参数，bootstrap 采样）；
  并以 1-Wasserstein 距离（一维排序法）、MMD（RBF 核）与负对数似然作为分布相似度指标。
输入：二维混合高斯（3 个分量），n=1200，seed=42。
参数：GMM n_components ∈ {1,2,3,5}；KDE bandwidth ∈ {0.1, 0.3, 0.8}；采样 1200 个。
"""
import os
import sys

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, env_info                        # noqa: E402

SEED = 42
N = 1200


def wasserstein_1d(a, b):
    """一维 1-Wasserstein 距离（排序法），对每维计算后取平均。"""
    a = np.sort(a, axis=0)
    b = np.sort(b, axis=0)
    n = min(len(a), len(b))
    return float(np.mean(np.abs(a[:n] - b[:n])))


def mmd_rbf(X, Y, gamma=0.5):
    """RBF 核最大均值差异（无偏估计，抽样子集加速）。"""
    rng = np.random.default_rng(0)
    X = X[rng.choice(len(X), min(300, len(X)), replace=False)]
    Y = Y[rng.choice(len(Y), min(300, len(Y)), replace=False)]
    def k(A, B):
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d)
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def main(seed=SEED):
    rng = np.random.default_rng(seed)
    means = np.array([[-2.0, -2.0], [2.0, 2.0], [0.0, 3.0]])
    covs = np.array([[[0.6, 0.2], [0.2, 0.6]],
                     [[0.8, -0.3], [-0.3, 0.5]],
                     [[0.4, 0.0], [0.0, 0.9]]])
    comp = rng.integers(0, 3, N)
    X = np.vstack([rng.multivariate_normal(means[c], covs[c]) for c in comp])
    exp = Exp(exp_id="L8-02", algo="GAN / Diffusion（替代验证：GMM / KDE 采样）",
              layer="L8 决策与生成",
              purpose="把生成问题还原为密度估计与采样，定量比较参数化与非参数化生成模型",
              dataset="二维三分量高斯混合，n=%d；%s" % (N, "真实分量数=3，用于检验模型选择"),
              baselines=["GMM(k=1，欠拟合对照)", "KDE（非参数密度估计）"],
              metrics=["1-Wasserstein 距离", "MMD(RBF)", "平均负对数似然", "耗时"],
              seed=seed, params={"gmm_k": [1, 2, 3, 5],
                                "kde_bandwidth": [0.1, 0.3, 0.8],
                                "n_samples": N},
              env=env_info())

    for k in [1, 2, 3, 5]:
        with exp.timer("GMM k=%d" % k):
            g = GaussianMixture(n_components=k, random_state=seed,
                                n_init=3).fit(X)
            S = g.sample(N)[0]
        nll = -g.score_samples(X).mean()
        exp.add(方法="GMM(k=%d)" % k, Wasserstein=wasserstein_1d(X, S),
                MMD=mmd_rbf(X, S), 负对数似然=float(nll), BIC=float(g.bic(X)))

    for bw in [0.1, 0.3, 0.8]:
        with exp.timer("KDE bw=%g" % bw):
            kde = KernelDensity(bandwidth=bw).fit(X)
            S = kde.sample(N, random_state=seed)
        nll = -kde.score_samples(X).mean()
        exp.add(方法="KDE(bw=%g)" % bw, Wasserstein=wasserstein_1d(X, S),
                MMD=mmd_rbf(X, S), 负对数似然=float(nll), BIC=float("nan"))

    exp.figure("l8_02_gen", "bar", "生成样本与真实分布的差异（越小越像）",
               [r["方法"] for r in exp.rows],
               [("1-Wasserstein", [r["Wasserstein"] for r in exp.rows]),
                ("MMD(RBF)", [r["MMD"] for r in exp.rows])], ylabel="分布差异")
    best = min(exp.rows, key=lambda r: r["Wasserstein"])
    g3 = [r for r in exp.rows if r["方法"] == "GMM(k=3)"][0]
    g1 = [r for r in exp.rows if r["方法"] == "GMM(k=1)"][0]
    exp.note("最优：%s，1-Wasserstein %.4f、MMD %.4f。真实分量数为 3，GMM(k=3) 的 BIC %.1f "
             "低于 k=1 的 %.1f，说明模型选择准则能自动识别正确复杂度。"
             % (best["方法"], best["Wasserstein"], best["MMD"], g3["BIC"], g1["BIC"]))
    exp.note("带宽是 KDE 的关键：过小→过拟合（样本几乎复制训练点，MMD 反而变大），"
             "过大→过度平滑丢失多峰结构；这与 GAN/Diffusion 中「模式崩塌 vs 过度平滑」的权衡同源。")
    exp.note("局限与后续：GMM/KDE 只能建模低维、结构简单的分布；高维图像生成必须依赖"
             "深度生成模型。若需完整实验，建议在有 GPU 的环境安装 torch，"
             "从 DCGAN（MNIST）或 DDPM（CIFAR-10 子集）入手，并用 FID 评价。")
    exp.save()


if __name__ == "__main__":
    main()
