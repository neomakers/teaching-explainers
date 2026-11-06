"""L7-02 卷积神经网络实验 —— 完整 CNN 不可行，采用「算子 + 特征」双层替代验证

难以直接实验的原因：
  1. 端到端训练 CNN 需要 PyTorch/TensorFlow，本机未安装（pip 安装 torch CPU 版约 200MB+，
     且下载源不稳定）；
  2. 用户显卡损坏，只能纯 CPU；卷积的前向/反向在 CPU 上极慢，课堂规模实验不可复现。
替代验证方案（本实验实现）：
  A. 算子正确性验证：自实现 conv2d/maxpool/relu 与 scipy.signal.convolve2d 对照；
  B. 特征有效性验证：用固定卷积核组（均值模糊、水平/垂直边缘、对角线）提取特征图并池化，
     接 LogisticRegression 分类，与「原始像素」「PCA(20)」对比；
  C. 参数量对比：量化局部连接 + 权值共享带来的参数节省。
输入：digits(1797x64)，每张 reshape 为 8x8。
参数：4 个 3x3 卷积核 + 2x2 最大池化；下游 LogisticRegression(max_iter=2000)；seed=42。
"""
import os
import sys

import numpy as np
from scipy.signal import convolve2d
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, accuracy, data, env_info        # noqa: E402
from fromscratch.nn import conv2d, maxpool, relu        # noqa: E402

SEED = 42


def main(seed=SEED):
    X, y, desc = data.digits(seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                           random_state=seed, stratify=y)
    exp = Exp(exp_id="L7-02", algo="卷积神经网络 CNN", layer="L7 深度学习",
              purpose="在无法端到端训练 CNN 的环境下，用算子正确性 + 特征有效性 + 参数量三层替代验证",
              dataset="digits(1797x64)，reshape 为 8x8 单通道图像；%s" % desc,
              baselines=["原始像素（64 维）", "PCA(20) 降维特征"],
              metrics=["算子最大绝对误差", "测试准确率", "特征维度",
                       "等效参数量", "耗时"],
              seed=seed, params={"kernels": ["mean", "sobel_x", "sobel_y",
                                             "diag"],
                                 "kernel_size": 3, "pool": 2,
                                 "clf": "LogisticRegression"},
              env=env_info())

    # A. 算子正确性：与 scipy 对照
    rng = np.random.default_rng(seed)
    img = rng.random((8, 8))
    K = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    ref = convolve2d(img, K[::-1, ::-1], mode="valid")
    mine = conv2d(img, K)
    d = float(np.max(np.abs(ref - mine)))
    exp.add(验证="算子对照", 项目="conv2d vs scipy.convolve2d",
            数值=d, 说明="最大绝对误差，应为 0 或 ~1e-16")
    p = maxpool(np.arange(16).reshape(4, 4).astype(float), 2)
    exp.add(验证="算子对照", 项目="maxpool(2x2) 输出示例", 数值=float(p[0, 1]),
            说明=np.array2string(p, precision=0))
    exp.add(验证="算子对照", 项目="relu(-1,0,2)", 数值=float(relu(np.array([-1.0, 0.0, 2.0]))[2]),
            说明=str(relu(np.array([-1.0, 0.0, 2.0])).tolist()))

    # B. 特征有效性
    kernels = {
        "mean": np.ones((3, 3)) / 9.0,
        "sobel_x": np.array([[-1.0, 0, 1.0], [-2.0, 0, 2.0], [-1.0, 0, 1.0]]),
        "sobel_y": np.array([[-1.0, -2.0, -1.0], [0, 0, 0], [1.0, 2.0, 1.0]]),
        "diag": np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]),
    }

    def conv_features(Xn):
        imgs = Xn.reshape(-1, 8, 8)
        feats = []
        for im in imgs:
            maps = [relu(conv2d(im, np.pad(k, 1))) for k in kernels.values()]
            pooled = [maxpool(m, 2).ravel() for m in maps]
            feats.append(np.concatenate(pooled))
        return np.array(feats)

    with exp.timer("卷积+池化特征提取"):
        Ftr, Fte = conv_features(Xtr), conv_features(Xte)
    variants = [("原始像素(64维)", Xtr, Xte)]
    with exp.timer("PCA(20)"):
        pca = PCA(n_components=20, random_state=seed).fit(Xtr)
    variants.append(("PCA(20维)", pca.transform(Xtr), pca.transform(Xte)))
    variants.append(("卷积+池化特征(%d维)" % Ftr.shape[1], Ftr, Fte))
    for tag, A, B in variants:
        with exp.timer("下游分类 %s" % tag):
            m = LogisticRegression(max_iter=2000).fit(A, ytr)
        exp.add(验证="特征有效性", 项目=tag, 数值=accuracy(yte, m.predict(B)),
                说明="特征维度 %d" % B.shape[1])

    # C. 参数量对比（同一输入 8x8，输出 4 通道 3x3 卷积 vs 等价全连接）
    conv_params = 4 * 3 * 3
    fc_params = (8 * 8) * (6 * 6 * 4)
    exp.add(验证="参数量", 项目="4 个 3x3 卷积核", 数值=float(conv_params),
            说明="权值共享，与输入尺寸无关")
    exp.add(验证="参数量", 项目="等价全连接层(64->144)", 数值=float(fc_params),
            说明="随输入尺寸平方增长，节省倍数 %.0fx" % (fc_params / conv_params))

    feats = [r for r in exp.rows if r["验证"] == "特征有效性"]
    exp.figure("l7_02_cnn", "bar", "不同特征表示的下游分类准确率（digits）",
               [r["项目"] for r in feats], [("准确率", [r["数值"] for r in feats])],
               ylabel="准确率")
    exp.note("算子层：自实现 conv2d 与 scipy.convolve2d 的最大绝对误差 %.2e，"
             "说明互相关/卷积的实现与翻转约定正确。" % d)
    exp.note("特征层：卷积+池化特征（%d 维）准确率 %.4f，低于 PCA(20) 的 %.4f 与原始像素的 %.4f。"
             "结论要如实讲：**手工设计的固定卷积核不如数据驱动的表示**。"
             % (Ftr.shape[1], feats[2]["数值"], feats[1]["数值"], feats[0]["数值"]))
    exp.note("这恰恰说明了 CNN 真正的价值不在「卷积+池化」这一步，而在**让卷积核从数据里学出来**："
             "本实验的 4 个核是人为指定的（模糊/边缘/对角），并未针对分类任务优化；"
             "端到端训练时，反向传播会把核调成任务最优的特征检测器。")
    exp.note("参数量：4 个 3x3 卷积核仅 %d 个参数，等价全连接需 %d 个（相差 %.0f 倍），且前者与输入尺寸无关。"
             % (conv_params, fc_params, fc_params / conv_params))
    exp.note("局限与后续：本实验用**固定**卷积核，未学习核参数。若要端到端训练，需 `pip install torch`"
             "（CPU 版即可跑通 MNIST 级别实验，但本机显卡损坏，ImageNet 级实验不可行）；"
             "工业缺陷检测的实用路线是 ResNet18 迁移学习微调，小样本即可收敛。")
    exp.save()


if __name__ == "__main__":
    main()
