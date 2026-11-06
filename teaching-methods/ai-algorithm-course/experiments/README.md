# 算法实验目录

《人工智能技术及应用》课程配套实验，覆盖大纲中的 30 个算法。每个算法一个独立脚本，
固定随机种子、显式声明输入数据与参数配置，可单独运行、结果可复现。

## 1. 环境与运行

```bash
# 依赖：numpy、scikit-learn、scipy（sklearn 自带 scipy）
python -m pip install scikit-learn

# 运行单个实验（示例）
python exp_l3_01_pca.py

# 一键运行全部实验并生成报告
python run_all.py

# 只跑某一层 / 只重新生成报告
python run_all.py --only l4
python run_all.py --report-only
```

本机实测环境：Python 3.13 / numpy 2.5.4 / scikit-learn 1.9.1 / Windows 11，纯 CPU。

## 2. 目录结构

```
experiments/
├── common/            实验公共模块
│   ├── data.py           数据集与合成用例（全部离线、可复现）
│   ├── io_utils.py       结果记录 Exp 类 + 零依赖 SVG 图表
│   └── runner.py         种子、计时、评价指标（RMSE/准确率/AUC/ARI/…）
├── fromscratch/       教学用从零实现（numpy，接口对齐 sklearn）
│   ├── svd.py            Jacobi 特征分解 + 截断 SVD
│   ├── pca.py            PCA（SVD 路径）
│   ├── kmeans.py         K-Means（含 k-means++）
│   ├── optim.py          一阶优化器族 + 逻辑回归
│   ├── nn.py             conv2d / maxpool / 自注意力 / Elman RNN
│   └── qlearning.py      表格 Q-Learning
├── exp_l*_*.py        30 个实验脚本（每算法一个入口）
├── run_all.py         一键运行 + 报告生成
├── tools/
│   ├── md2html.py         轻量 Markdown → HTML（零依赖，生成指导书网页版）
│   └── build_lab_page.py 把 30 个实验结果与指导书注入 `../outline-interactive.html`
│                          （网页内预留锚点，脚本幂等，可反复执行）
└── results/
    ├── metrics/*.json    数值结果
    ├── tables/*.md       对比表
    ├── figures/*.svg     图表（零依赖生成）
    ├── logs/*.log        运行日志
    ├── REPORT.md         总报告（含实验清单与结论）
    └── ../experiment-report.html      带内嵌图表的完整报告
```

## 3. 接口约定

- **不改动算法核心实现**：被测算法一律通过标准接口调用
  （`fit` / `transform` / `predict` / `fit_predict`），
  与 scikit-learn 完全一致；`fromscratch/` 下的参考实现是**独立实现**，
  用于原理验证与对照，不修改任何第三方库代码。
- 每个实验脚本结构固定：
  1. 模块 docstring 写明**目的 / 输入 / 参数 / 种子**；
  2. `Exp(...)` 声明实验方案（数据集、基线、指标、参数、环境）；
  3. `main(seed=42)` 执行并 `exp.save()`；
  4. `if __name__ == "__main__": main()`，可直接单独运行。
- 数据划分一律 `train_test_split(..., random_state=seed)`；标准化只在训练集 `fit`。

## 4. 实验清单（编号 / 算法 / 文件 / 结论）

| 编号 | 算法 | 文件 | 主要结论 |
|---|---|---|---|
| L1-01 | SVD | exp_l1_01_svd.py | 截断 SVD 在每个 k 上误差均低于同维随机投影（Eckart-Young）；自实现与 LAPACK 奇异值差 3e-13 |
| L1-02 | 特征值分解 | exp_l1_02_eig.py | Jacobi 与 eigh 数值一致；σ²/(n-1) 与 λ 差 1e-12 量级，两路径等价 |
| L1-03 | 梯度下降族 | exp_l1_03_optimizer.py | lr=0.05 时 RMSProp(16步)<Momentum(63)<SGD(149)<Adam(193) 全部收敛到闭式解；lr=0.01 时 SGD 与 Adam 未收敛——自适应优化器同样需要合适的学习率 |
| L2-01 | 标准化 | exp_l2_01_scaling.py | wine 上 KNN 准确率 0.722→0.944；随机森林两种预处理结果完全一致 |
| L2-02 | 特征选择 | exp_l2_02_featureselection.py | 保留 10 个特征（1/3 成本）时损失 1.8 / 5.3 个百分点；随机选 10 个明显更差 |
| L3-01 | PCA | exp_l3_01_pca.py | k=10（16% 维度）接近全量精度；自实现与 sklearn 坐标差 <1e-6 |
| L3-02 | LDA | exp_l3_02_lda.py | 有监督降维优于同维 PCA；维度上限 = 类别数-1 得到验证 |
| L3-03 | NMF | exp_l3_03_nmf.py | 用约 0.02 的重构误差换取非负性与更高稀疏度（可解释性） |
| L3-04 | t-SNE/UMAP | exp_l3_04_manifold.py | t-SNE 的 trustworthiness 高于 PCA；无 transform 接口，只能可视化 |
| L3-05 | ICA | exp_l3_05_ica.py | 非高斯源上 ICA 平均相关系数远高于 PCA，验证「不相关≠独立」 |
| L3-06 | 自编码器 | exp_l3_06_autoencoder.py | 重建误差与 PCA 一致（差 <3e-4），但编码器子空间主角度 cos 仅 0.62——等价的是重建函数而非权重 |
| L4-01 | 回归族 | exp_l4_01_regression.py | 含 62·f² 与交互项时，随机森林 RMSE 0.221 vs 线性回归 0.460 |
| L4-02 | 逻辑回归 | exp_l4_02_logistic.py | AUC 0.998；自实现与 sklearn 准确率差 0.0175、平均概率差 0.034、权重相关 0.747 |
| L4-03 | KNN | exp_l4_03_knn.py | digits 上 k=1 最优（0.987），k=25 降至 0.957；维度 64→864（纯噪声）时明显下滑 |
| L4-04 | 决策树 | exp_l4_04_tree.py | 深度不限时训练准确率升至 100% 而测试下降；规则数随深度指数增长 |
| L4-05 | SVM | exp_l4_05_svm.py | moons 上 RBF 核 1.000 vs 线性核 0.867；C 与支持向量数反向变化 |
| L4-06 | 朴素贝叶斯 | exp_l4_06_naive_bayes.py | digits 上多项式/伯努利优于高斯 NB；特征强相关时精度下降（MultinomialNB 还需非负输入） |
| L5-01 | 随机森林 | exp_l5_01_randomforest.py | 单树 0.918 → 300 棵 0.942；OOB 可替代验证集；特征重要性 top5 跨种子重合率高 |
| L5-02 | GBDT/XGBoost | exp_l5_02_boosting.py | 直方图提升（0.84 s）快于经典 GBDT（1.52 s）；XGBoost 缺失时用 HGB 等价替代 |
| L5-03 | 集成鲁棒性 | exp_l5_03_noise_robust.py | 10% 标签噪声下衰减：单树 0.105 > HGB 0.047 > AdaBoost 0.018 > GBDT 0.000 > RF -0.006 |
| L6-01 | K-Means | exp_l6_01_kmeans.py | k-means++ inertia 1431（标准差 0）远优于随机初始化 7957（标准差 8712）；自实现一致 |
| L6-02 | DBSCAN | exp_l6_02_dbscan.py | moons 上 DBSCAN(eps=0.2) ARI 1.000，KMeans 仅 0.265；eps 敏感 |
| L6-03 | GMM/层次 | exp_l6_03_gmm.py | 椭圆簇上 GMM(full) ARI 1.0 vs KMeans 0.9765；GMM(spherical) 与 KMeans 同为 0.9765，瓶颈在球状假设；BIC 拐点=k |
| L7-01 | MLP | exp_l7_01_mlp.py | 隐层规模与学习率共同决定精度与迭代数，早停防过拟合 |
| L7-02 | CNN（替代验证） | exp_l7_02_cnn.py | 自实现卷积与 scipy 误差 6.7e-16；固定核卷积特征 0.848 < PCA(20) 0.950——卷积的价值在让核从数据中学出来；参数量省 256 倍 |
| L7-03 | RNN（替代验证） | exp_l7_03_rnn.py | Elman RNN（lr=0.2, 1000 epoch）RMSE 0.077，优于 AR(5) 0.117 与 MLP 窗口 0.114，但耗时 40 s vs 0.02 s |
| L7-04 | Transformer（替代验证） | exp_l7_04_attention.py | 排列等变误差 2.2e-16（故需位置编码）；相似 token 权重显著更高；参数量 3d² 与序列长度无关 |
| L8-01 | Q-Learning（替代验证） | exp_l8_01_qlearning.py | 折扣回报口径下贪心评估 6.380 = 价值迭代最优值；随机策略 -3.805 |
| L8-02 | GAN/Diffusion（替代验证） | exp_l8_02_generative.py | GMM(k=3) 的 Wasserstein 0.095 / MMD 0.007 最小；BIC 正确识别分量数=3 |
| L8-03 | 异常检测 | exp_l8_03_anomaly.py | Isolation Forest AUC 0.985 最优；PCA 重构误差 k=1 时 0.886、k=2 时退化到 0.524（残差子空间选择决定成败） |

## 5. 难以直接实验的算法与替代方案

| 算法 | 不可行的原因 | 已采用的替代验证 |
|---|---|---|
| CNN（端到端训练） | 未装 PyTorch；显卡损坏，纯 CPU 训练不可复现 | 算子与 scipy 对照 + 固定核卷积特征的下游收益 + 参数量对比 |
| LSTM / GRU | 标准实现依赖 torch | numpy 从零实现 Elman RNN，与 AR(p)、MLP 窗口三方对比 |
| Transformer / 大模型 | 需 torch + 语料 + 显存 | 验证自注意力三条数学性质（排列等变、内容寻址、参数解耦） |
| DQN / PPO | 需 torch 与大量交互步数；gym 未安装 | 自建网格环境 + 表格 Q-Learning，与价值迭代的最优解对照 |
| GAN / Diffusion | 需 GPU；FID 等指标需预训练网络 | 还原为密度估计与采样：GMM / KDE，用 Wasserstein、MMD、NLL 评价 |
| XGBoost | 未安装 | HistGradientBoosting（同为直方图梯度提升）等价替代，安装后自动纳入 |
| UMAP | 未安装 | t-SNE + PCA 对照；`pip install umap-learn` 后脚本自动纳入 |

## 6. 如何新增实验

复制任一 `exp_*.py`，按第 3 节的四段结构改写即可；`run_all.py` 会自动发现
`exp_*.py` 并纳入报告。数据集尽量复用 `common/data.py`（保证离线可复现），
新数据集请在该文件中补充并把随机种子写进 `SEED`。

## 7. 综合项目与难度系数（2026-10 起）

分组综合项目（3 人一组）的选题、评分与交付规则见 `../docs/project-brief.md`：

- 30 个算法的五维难度评分与系数表：`tools/difficulty.py`（单一数据源），
  运行 `python tools/difficulty.py --check` 自检，`--table` 打印，`--emit-md` 重新生成 `../docs/difficulty-table.md`。
- 选题登记网页：`../topic-picker.html`（数据由 `tools/build_pick_page.py` 从 difficulty.py 注入）。
- 本目录的 `exp_*.py` 脚本继续作为学生作品的可复现参考基线（seed=42），
  其中 7 个替代验证算法（见第 5 节）的口径已在难度系数的 R 维度中体现。
- 老师批量验收工具：`tools/check_syntax.js` / `verify_ui.js` / `verify_pick.js`（Playwright）。
