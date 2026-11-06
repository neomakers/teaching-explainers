"""教学用从零参考实现（numpy，纯 CPU）。

接口刻意与 scikit-learn 保持一致（fit / transform / predict / fit_predict），
便于实验中与库实现直接对照。这些文件**不修改**任何第三方库的核心实现，
只是独立实现，用于原理验证与课堂演示。
"""
from .svd import svd, jacobi_eigen, truncate
from .pca import PCA
from .kmeans import KMeans
from .optim import minimize, RidgeGD, LogisticGD
from .nn import conv2d, maxpool, relu, self_attention, multihead_attention, ElmanRNN
from .qlearning import QLearning

__all__ = ["svd", "jacobi_eigen", "truncate", "PCA", "KMeans", "minimize",
           "RidgeGD", "LogisticGD", "conv2d", "maxpool", "relu",
           "self_attention", "multihead_attention", "ElmanRNN", "QLearning"]
