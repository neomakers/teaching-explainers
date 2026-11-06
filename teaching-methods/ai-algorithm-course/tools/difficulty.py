# -*- coding: utf-8 -*-
"""《人工智能技术及应用》综合项目 —— 难度系数数据源。

单一事实源：既给 outputs/难度系数表.md 供公示，也用 --emit-js 给 outputs/选题系统.html 注入。
零第三方依赖，Python 3.8+ 可直接跑。

用法：
    python tools/difficulty.py --check      # 自检（公式、范围、档位、与主页面 DATA 对齐）
    python tools/difficulty.py --table      # 打印 30 行表格
    python tools/difficulty.py --emit-md    # 输出 outputs/难度系数表.md
    python tools/difficulty.py --emit-js    # 打印可直接粘贴的 JS 数组
"""

import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HINT_HTML = os.path.join(ROOT, "outputs", "AI算法大纲-交互版.html")
METRICS = os.path.join(ROOT, "experiments", "results", "metrics")

# 五维权重，和为 1.0
W = {"M": 0.25, "C": 0.20, "V": 0.25, "A": 0.15, "R": 0.15}
DIMS = ["M", "C", "V", "A", "R"]

DIM_RUBRIC = [
    ("M", "数学深度", 0.25,
     "1=四则运算有闭式解；2=一维导数、≤3 行推导；3=矩阵运算+迭代式≤1 屏；"
     "4=需对偶/EM/变分下界；5=需 SDE/博弈/贝尔曼方程"),
    ("C", "编程复杂度", 0.20,
     "1=sklearn 一行 fit；2=加参数扫描与对照组；3=手写 numpy 迭代；"
     "4=手写前向+反向传播；5=必须依赖训练框架"),
    ("V", "可视化难度", 0.25,
     "1=静态图+1 滑块；2=2D 曲线/散点+1~2 滑块；3=需迭代动画/状态机/递归绘制；"
     "4=rAF 内跑 epoch 或多图联动；5=真算法级动画（Jacobi/ t-SNE 播放/去噪采样）"),
    ("A", "概念抽象度", 0.15,
     "1=可比作日常经验；2=有几何图像；3=需概率直觉；4=抽象机制（EM/注意力/门控）；"
     "5=反直觉（梯度消失/KL 非对称/t-SNE 距离不可解释）"),
    ("R", "资源支撑度", 0.15,
     "逆向计分，越高越难：1=有 exp 脚本+现成 demo+离线数据集；"
     "2=依赖有替代实现；3=仅替代验证口径；4=无对照基线；5=需自建全部验证"),
]


def _t(i, n, l, star, dims, hours, exp, demo, datasets, scene, pitfalls, must, **kw):
    return dict(
        id=i, n=n, l=l, lt=LAYER_T.get(l, ""), star=star,
        d=dict(zip(DIMS, dims)), hours=hours, exp=exp, demo=demo,
        datasets=datasets, scene=scene, pitfalls=pitfalls, mustHave=must,
        open=kw.get("open", 1), cap=kw.get("cap", 3), note=kw.get("note", ""),
    )


LAYER_T = {
    "L1": "数学基础", "L2": "数据表示", "L3": "降维与矩阵分解",
    "L4": "监督学习", "L5": "集成与调优", "L6": "无监督学习",
    "L7": "深度学习", "L8": "决策与生成（选讲）",
}

# demo 行号记的是 outputs/AI算法大纲-交互版.html 里 regDemo(...) 的起始行
TOPICS = [
    _t("L1-01", "奇异值分解 SVD", "L1", 1, (3, 3, 2, 2, 1), 16,
       "exp_l1_01_svd.py", 1328,
       ["common/data.py::digits()"],
       "设备多通道振动信号/图像的特征压缩与去噪，工件图像的低秩重建",
       ["直接对原始像素做特征分解会因大矩阵极慢，应走截断 SVD",
        "协方差矩阵 A^T A 会丢失数值精度，优先 numpy.linalg.svd",
        "重建误差随 k 下降但边际收益递减，必须画碎石图说明拐点"],
       ["奇异值谱（碎石图）与累积能量曲线",
        "拖动 k 实时对比原图/重建图/残差",
        "与随机高斯投影同维度的误差对照（Eckart-Young 举证）"]),

    _t("L1-02", "特征值分解", "L1", 0, (2, 3, 1, 2, 1), 12,
       "exp_l1_02_eig.py", 2280,
       ["common/data.py::corr_data()"],
       "主成分方差解释、结构模态分析中的特征模态",
       ["非对称矩阵的特征值可能为复数，教学务必限定对称阵",
        "Jacobi 旋转要设收敛阈值，否则演示会卡住",
        "务必与 numpy.linalg.eigh 对照校验"],
       ["2x2 对称矩阵元素可调输入框/滑块",
        "单位圆在该矩阵作用下变形为椭圆",
        "实时显示两个特征向量方向与特征值"]),

    _t("L1-03", "梯度下降族", "L1", 0, (3, 3, 2, 3, 1), 18,
       "exp_l1_03_optimizer.py", 1615,
       ["合成二次型（可控条件数）"],
       "神经网络训练与损失曲面优化原理",
       ["条件数大时 SGD 会震荡，演示必须可调条件数",
        "学习率超过 2/L 必然发散，别让页面显示 Infinity，要做发散检测",
        "Momentum/Adam 的偏置修正不能漏，否则前几步失真"],
       ["同一损失曲面上三条优化轨迹同屏对比",
        "学习率与条件数两个滑块",
        "显示各优化器最终损失与是否发散"]),

    _t("L2-01", "标准化与归一化", "L2", 0, (1, 1, 2, 1, 1), 8,
       "exp_l2_01_scaling.py", 2378,
       ["合成两维数据（量纲相差 100 倍）"],
       "多源传感器数据融合前的量纲统一",
       ["必须在训练集 fit、测试集只 transform，否则数据泄露",
        "MinMax 对离群点极敏感，要演示被单个异常值压扁的情形",
        "树模型不需要标准化，要说清适用范围"],
       ["四种变换方式可切换（Standard/MinMax/Robust/不处理）",
        "显示标准化前后最近邻距离占比变化",
        "含一个离群点开关，演示 MinMax 的脆弱性"]),

    _t("L2-02", "特征选择", "L2", 0, (2, 2, 3, 2, 1), 16,
       "exp_l2_02_featureselection.py", 3057,
       ["common/data.py::classification()", "common/data.py::noisy_dims()"],
       "工艺关键参数识别、传感器布点优化",
       ["过滤法打分要用训练集算，全量数据打分会虚高",
        "准确率要留出验证集，用全量训练准确率等于造假",
        "Lasso 的 alpha 与最终保留特征数要一并展示"],
       ["特征得分排序条形图（含噪声特征）",
        "随保留特征数 k 变化的验证准确率曲线（要出现先升后降）",
        "过滤法与嵌入法（Lasso）结果对比"]),

    _t("L3-01", "主成分分析 PCA", "L3", 1, (3, 2, 2, 2, 1), 14,
       "exp_l3_01_pca.py", 1359,
       ["common/data.py::wine()", "common/data.py::synth_process()"],
       "多传感器信号融合降维、工艺参数压缩、工业数据去噪",
       ["PCA 前必须标准化，否则大方差特征会主导",
        "只能捕捉线性相关；对环形/月牙数据失效",
        "主成分的物理含义需人工解释，不能自动赋予"],
       ["方差解释率柱状图+累积曲线",
        "散点图上画出 PC1/PC2 主方向箭头",
        "拖动 k 看降维后重建误差与下游任务精度"]),

    _t("L3-02", "线性判别分析 LDA", "L3", 0, (4, 2, 2, 3, 1), 18,
       "exp_l3_02_lda.py", 1359,
       ["common/data.py::wine()", "common/data.py::classification()"],
       "故障类别可分性增强、小样本高维分类前置降维",
       ["降维上限是类别数-1，二分类只能降到 1 维，别让学生困惑",
        "必须类内散度矩阵可逆，小样本要先做正则",
        "核心看点：方差最大方向未必是最可分方向"],
       ["同一数据上 PCA 方向与 LDA 方向并列画出",
        "1-NN 在 PCA(1维)/LDA(1维)/原始三维三种表示下的精度对比",
        "可调类内方差与类间分离，展示何时两者差距最大"]),

    _t("L3-03", "非负矩阵分解 NMF", "L3", 0, (4, 3, 3, 3, 1), 24,
       "exp_l3_03_nmf.py", 3218,
       ["common/data.py::blobs()", "合成非负矩阵"],
       "光谱/能谱分解、图像部件提取、故障模式分解",
       ["乘法更新里分子分母的矩阵容易转置错，务必单独写 mm/mtm 两个函数",
        "必须加 1e-9 防除零，否则整片 NaN",
        "输入必须非负，含负值的数据要先平移"],
       ["原始矩阵 / W / H / 重建矩阵四格热力图",
        "随迭代次数下降的重构误差曲线",
        "与 PCA 的「可加部件 vs 正交方向」对比"]),

    _t("L3-04", "t-SNE / UMAP", "L3", 0, (4, 3, 5, 5, 2), 32,
       "exp_l3_04_manifold.py", 3614,
       ["common/data.py::blobs()", "common/data.py::digits()"],
       "高维工艺数据聚类可视化、故障模式探索性分析",
       ["簇间距离不可解释，图上看着远不代表真的远",
        "perplexity 敏感性必须演示，否则学生以为默认是权威",
        "KL 散度非对称，KL(P||Q) 与 KL(Q||P) 行为不同"],
       ["真实 KLD 优化的迭代动画（不是调用一次 fit_transform）",
        "perplexity 滑块并实时重plot",
        "与 PCA 结果并排对比，说明非线性结构"]),

    _t("L3-05", "独立成分分析 ICA", "L3", 0, (4, 3, 4, 4, 1), 28,
       "exp_l3_05_ica.py", 3376,
       ["common/data.py::mixed_signals()"],
       "设备多源振动信号分离、声学故障诊断",
       ["白化矩阵必须是 D^(-1/2) V^T，分母下标写错会导致 ICA 反而弱于 PCA",
        "FastICA 用 tanh 时务必说明非高斯性假设",
        "分离结果有顺序与符号的不确定性，要显式说明"],
       ["源信号 → 混合 → 分离 三列波形图",
        "散布图 + 直方图显示非高斯性变化",
        "与 PCA 的恢复相似度定量对照（应显著占优）"]),

    _t("L3-06", "自编码器 AE", "L3", 0, (4, 4, 4, 3, 1), 30,
       "exp_l3_06_autoencoder.py", 3859,
       ["common/data.py::blobs()", "合成环形/月牙数据"],
       "工业设备异常检测、非线性降维",
       ["手写 BP 在隐层大时会梯度爆炸，必须加梯度范数裁剪",
        "线性 AE 的解空间是 PCA 子空间的旋转，要说清不等价",
        "学习率过大直接 NaN，页面上要能看出来并提示"],
       ["训练损失随 epoch 下降的实时曲线",
        "原空间 → 隐空间 → 重建的三视图",
        "瓶颈维度 H 滑块，展示欠拟合到过拟合"]),

    _t("L4-01", "线性回归 / Ridge / Lasso", "L4", 1, (2, 2, 2, 2, 1), 12,
       "exp_l4_01_regression.py", 1522,
       ["common/data.py::synth_process()"],
       "工艺参数 → 零件性能预测、表面粗糙度建模",
       ["多项式阶数过高必然过拟合，要同时显示训练/测试 RMSE",
        "Lasso 的稀疏性：alpha 增大时系数逐步归零的过程必须可见",
        "Ridge 只会压缩不会置零，别和 Lasso 混为一谈"],
       ["拟合曲线与真实曲线叠加，随阶数 d 变化",
        "训练/测试 RMSE 双曲线，出现明显的泛化 gap",
        "Lasso 各特征系数随 alpha 的收缩路径"]),

    _t("L4-02", "逻辑回归", "L4", 0, (3, 2, 2, 2, 1), 14,
       "exp_l4_02_logistic.py", 2485,
       ["common/data.py::classification()", "common/data.py::breast()"],
       "合格/不合格判定、故障概率估计",
       ["数据太容易分时看不出正则作用，应加噪声或降分离度",
        "输出是概率不是硬标签，要画概率等高线",
        "务必与线性 SVM 决策边界做对照"],
       ["散点 + 概率等高线渐变 + 决策边界",
        "交叉熵损失随迭代下降曲线",
        "正则强度滑块，展示权重收缩"]),

    _t("L4-03", "K 近邻 KNN", "L4", 0, (1, 1, 2, 1, 1), 8,
       "exp_l4_03_knn.py", 1766,
       ["common/data.py::moons()", "common/data.py::circles()"],
       "小样本模式识别、推荐系统召回",
       ["k=1 必然过拟合边界锯齿，要动画演示 k 增大后边界平滑",
        "距离度量选择比 k 更关键，但常被忽略",
        "维度灾难：加噪声维度看精度崩塌"],
       ["决策区域填色随 k 实时变化",
        "单个查询点的 k 个近邻连线",
        "维度灾难曲线（精度 vs 噪声维数）"]),

    _t("L4-04", "决策树", "L4", 1, (3, 2, 3, 2, 1), 18,
       "exp_l4_04_tree.py", 1816,
       ["common/data.py::moons()", "common/data.py::classification()"],
       "工艺规则提取、故障原因树分析",
       ["样本极少的一侧会被优先分出，要用信息增益而非单纯的深度控制",
        "完全生长的树会切出碎片区域，必须对照剪枝效果",
        "旋转敏感性：数据旋转后树结构全变"],
       ["递归二分过程的分步动画",
        "一边生长的树结构与特征区域划分同步显示",
        "深度滑块 + 剪枝前后对比"]),

    _t("L4-05", "支持向量机 SVM", "L4", 1, (4, 2, 3, 3, 1), 22,
       "exp_l4_05_svm.py", 1886,
       ["common/data.py::circles()", "common/data.py::moons()"],
       "小样本缺陷分类、故障模式识别",
       ["C 的含义必须与软间隔讲清楚，否则调参没依据",
        "gamma 过大会导致过拟合边界包住每个点",
        "核技巧是难点，务必说明「升维后线性可分」"],
       ["线性/多项式/RBF 核可切换",
        "支持向量高亮 + 间隔带宽可视化",
        "C 与 gamma 两个滑块下的边界形变"]),

    _t("L4-06", "朴素贝叶斯", "L4", 0, (2, 1, 2, 2, 1), 10,
       "exp_l4_06_naive_bayes.py", 2613,
       ["common/data.py::blobs()", "合成文本词频"],
       "故障工单文本分类、初步筛选模型",
       ["条件独立假设在相关特征上失效，要专门演示失败案例",
        "连续特征用高斯假设时方差为 0 要平滑",
        "先验改变会整体平移决策边界"],
       ["两类的高斯等高线与决策边界",
        "先验概率滑块带动边界整体平移",
        "相关 vs 独立两种数据的假设失效对照"]),

    _t("L5-01", "随机森林", "L5", 1, (3, 2, 3, 2, 1), 18,
       "exp_l5_01_randomforest.py", 2181,
       ["common/data.py::classification()", "common/data.py::synth_process()"],
       "通用表格数据建模、关键工艺参数识别",
       ["单棵树方差大、森林方差小，要用多棵树投票动画说明",
        "特征重要性不等于因果重要性",
        "OOB 误差可作为内置交叉验证，值得展示"],
       ["不同 Bootstrap 子集上的多棵树单棵边界并列显示",
        "随树数 B 增加的准确率收敛曲线",
        "特征重要性排序条形图"]),

    _t("L5-02", "GBDT / XGBoost / LightGBM", "L5", 1, (4, 3, 3, 4, 2), 28,
       "exp_l5_02_boosting.py", 4086,
       ["common/data.py::synth_process()"],
       "工艺参数寻优建模、质量预测与良率分析",
       ["本机无 GPU 且未装 xgboost，统一用 HistGradientBoosting 替代，须注明",
        "逐棵加树拟合残差的过程必须可见，否则退化成调库",
        "学习率与树数存在权衡，不是越大越好"],
       ["残差逐步被磨平的动画（每加一棵树）",
        "训练/测试 RMSE 随迭代轮数变化",
        "整体/局部的预测曲面随树数演化"]),

    _t("L5-03", "AdaBoost", "L5", 0, (3, 2, 3, 3, 1), 20,
       "exp_l5_03_noise_robust.py", 4242,
       ["common/data.py::classification()"],
       "Boosting 思想原型、集成鲁棒性分析",
       ["难样本权重放大是核心看点，必须可视化权重变化",
        "标签噪声下 AdaBoost 会崩，这是与 RF 对比的关键场景",
        "弱分类器必须是决策树桩"],
       ["样本点大小=权重，随迭代明显变化",
        "每轮的弱分类器边界叠加显示",
        "噪声率滑块下的泛化退化对照"]),

    _t("L6-01", "K-Means", "L6", 1, (2, 3, 3, 1, 1), 16,
       "exp_l6_01_kmeans.py", 1414,
       ["common/data.py::blobs()", "common/data.py::circles()"],
       "工况分型、生产批次聚类、零件分组",
       ["初始化不同结果可能天差地别，必须做多次初始化",
        "对非凸形状（月牙/环）失效，要和 DBSCAN 对照",
        "K 需要手选，手肘法要给曲线"],
       ["迭代过程动画：质心移动 + 颜色重分配",
        "K 滑块 + inertia 手肘曲线",
        "random 与 k-means++ 初始化的结果对照"]),

    _t("L6-02", "DBSCAN", "L6", 0, (2, 3, 3, 2, 1), 18,
       "exp_l6_02_dbscan.py", 1978,
       ["common/data.py::moons()", "common/data.py::circles()"],
       "异常工况检测、轨迹/空间点聚类",
       ["eps 与 minPts 有耦合，只调一个很难出效果，应给出推荐默认值",
        "密度差异大的数据集很难用一组参数通吃",
        "噪声点标记为单独颜色，别忽略"],
       ["核心点/边界点/噪声点三类可视化",
        "eps 与 minPts 双滑块",
        "与 K-Means 在同一非凸数据上的对比"]),

    _t("L6-03", "层次聚类 / GMM", "L6", 0, (4, 3, 4, 3, 1), 28,
       "exp_l6_03_gmm.py", 2739,
       ["common/data.py::blobs()"], "零件族谱系划分、工况软分类",
       ["簇标签编号是任意的，跨算法比较要按最优排列匹配",
        "树状图（谱系树）的绘制工作量常被低估，建议二选一",
        "GMM 的 EM 可能落入局部最优，需多次初始化"],
       ["自底向上合并的动画 + 同步生成的树状图",
        "GMM 的高斯椭圆逐步收敛过程",
        "硬聚类 vs 软聚类（归属概率）对比"]),

    _t("L7-01", "多层感知机 MLP", "L7", 0, (3, 4, 4, 3, 1), 26,
       "exp_l7_01_mlp.py", 2083,
       ["common/data.py::moons()", "common/data.py::circles()"],
       "非线性质量预测、结构化数据建模",
       ["手写反向传播极易写错链式法则，建议逐层数值梯度校验",
        "螺旋数据分离常常失败，需要足够大的隐层和合适的学习率",
        "隐藏层过大会记住训练集中的每个点"],
       ["隐藏层神经元 H 与训练轮数滑块",
        "决策边界随训练轮数演化的动画（或逐步重绘）",
        "损失曲线 + 训练/测试准确率"]),

    _t("L7-02", "卷积神经网络 CNN", "L7", 1, (3, 4, 4, 2, 3), 30,
       "exp_l7_02_cnn.py", 2935,
       ["合成纹理/缺陷图", "common/data.py::digits()"],
       "增材制造缺陷检测、工件视觉分拣、表面质量判定",
       ["本机无 torch，CNN 只能做「卷积/池化算子层面的替代验证」，须在报告中注明",
        "权值共享必须用参数量对比说明，不能只讲概念",
        "padding/stride 的对齐容易被忽略，卷积后尺寸算错"],
       ["卷积核可视化 + 特征图实时更新",
        "六个以上预设卷积核可切换，看到边缘/纹理不同响应",
        "ReLU 稀疏化效果对比图"]),

    _t("L7-03", "循环网络 RNN / LSTM", "L7", 1, (4, 4, 4, 4, 3), 34,
       "exp_l7_03_rnn.py", 4379,
       ["common/data.py::timeseries()"],
       "设备振动信号预测性维护、剩余寿命 RUL 预测",
       ["本机无 torch，只能用手写 Elman RNN 做替代验证",
        "简单的加和记忆任务会被 tanh 饱和锁存绕过，证明不了梯度消失；"
        "建议改为直接可视化梯度流 PNG 本身",
        "LSTM 的遗忘门初始化对收敛影响巨大"],
       ["梯度回传的保留率对数曲线（RNN vs LSTM）",
        "隐藏状态随时间步的热力图",
        "门控值的可视化（哪些门在什么时候关闭）"]),

    _t("L7-04", "Transformer / 注意力", "L7", 1, (5, 4, 5, 4, 3), 40,
       "exp_l7_04_attention.py", 4516,
       ["合成序列"],
       "多模态工业检测、时序大模型、工艺知识问答底座",
       ["自注意力只用到矩阵乘法和 softmax，可用 numpy 复现并验证排列等变性",
        "缩放点积里的 sqrt(d_k) 不能漏，否则 softmax 饱和",
        "位置编码缺失时为序列不变，这是最直观的反例"],
       ["注意力权重热力图（Q-K 关系）",
        "多头注意力各头模式切换",
        "因果掩码与温度 τ 的开关，并能验证排列等变性"]),

    _t("L8-01", "强化学习 DQN / PPO", "L8", 0, (4, 4, 4, 4, 3), 34,
       "exp_l8_01_qlearning.py", 4651,
       ["common/data.py::GridWorld"],
       "机器人路径与抓取规划、工艺参数在线寻优、生产排程",
       ["argmax 平局时若总选第一个动作会卡在起点撞墙，平局必须随机",
        "终止状态的目标应为 0，误用 max Q(s') 会导致学不会",
        "贪心率衰减策略要显式给出"],
       ["GridWorld 上的 V 值热图与策略箭头",
        "每回合步数的收敛曲线",
        "「走一遍」按钮演示学习后的路径"]),

    _t("L8-02", "生成模型 GAN / Diffusion", "L8", 0, (5, 5, 5, 5, 4), 48,
       "exp_l8_02_generative.py", 4882,
       ["common/data.py::moons()", "common/data.py::blobs()"],
       "缺陷样本数据增强、仿真场景生成",
       ["本机无任何对照基线，FID/IS 等需预训练网络的指标一律不可做",
        "建议只做低维 toy 版本：用真实后验做反向去噪过程",
        "GAN 的判别器/生成器对抗容易不收敛，慎选"],
       ["前向加噪 → 反向去噪的完整过程动画",
        "噪声规λ(t) 随时间步的变化曲线",
        "二维点云从纯噪声逐步恢复为目标的中间帧"],
       note="", open=0),

    _t("L8-03", "异常检测", "L8", 1, (3, 3, 3, 2, 1), 20,
       "exp_l8_03_anomaly.py", 5053,
       ["common/data.py::sensor_anomaly()", "common/data.py::blobs()"],
       "设备异常早期预警、产品离群检测、传感器故障识别",
       ["阈值与 contamination 的设定直接决定精确率/召回率，必须可调",
        "重构误差法与密度法的失效场景不同，值得对比",
        "极度不平衡时准确率是无效指标，必须用 F1"],
       ["密度场/重构误差场的可视化",
        "阈值滑块联动精确率/召回率/F1",
        "Isolation Forest 与重构误差法的并排对比"]),
]

# 开放限制之外的特殊约束
TOPIC_NOTE = {
    "L8-02": "默认关闭。若有组强烈要求须签《超额选题承诺书》，且限定做 1D/2D 玩具版 "
             "Diffusion 的加噪/去噪过程可视化，禁止使用 FID/IS 等需预训练网络的指标。",
    "L7-04": "全班最多 1 组。交付物限定为注意力机制可视化，禁止训练语言模型。",
    "L3-04": "易被做成「调一次 sklearn 然后画图」，必须提供 perplexity 滑块并在视频中"
             "解释「簇间距离不可解释」。",
    "L3-05": "数学门槛高，选题绑定多源振动信号分离的应用场景叙事。",
    "L6-03": "谱系树绘制工作量常被低估，建议层次聚类与 GMM 二选一深做。",
}


# ---------------- 公式 ----------------

def score(t):
    return sum(W[k] * t["d"][k] for k in DIMS)


def coef(s):
    """S∈[1,5] → K∈[1.00,1.50]，步长 0.05。
    注意：Python 的 round() 是银行家舍入，必须用 floor(x+0.5)。"""
    import math
    return round(1.0 + 0.05 * math.floor(2.5 * (s - 1.0) + 0.5), 2)


def tier(s):
    """档位直接由 K 决定，避免出现「同为 K=1.35 却分属 C/D 两档」的困惑。
    （若用未取整的 S 判档，t-SNE 的 3.90 会落进 D 档，但它的 K 与 RNN 相同。）"""
    k = coef(s)
    if k <= 1.10:
        return "A"
    if k <= 1.20:
        return "B"
    if k <= 1.35:
        return "C"
    return "D"


TIER_DESC = {"A": "基础", "B": "标准", "C": "进阶", "D": "挑战"}


def decorate(topics=None):
    """给每条补上 S / K / tier / note。"""
    out = []
    for t in (topics or TOPICS):
        t = dict(t)
        t["S"] = round(score(t), 2)
        t["K"] = coef(t["S"])
        t["tier"] = tier(t["S"])
        if not t.get("note"):
            t["note"] = TOPIC_NOTE.get(t["id"], "")
        out.append(t)
    return out


def _notes_of(tid):
    """从 experiments/results/metrics/<id>.json 抓第一条结论作为参考基准。"""
    p = os.path.join(METRICS, tid + ".json")
    if not os.path.isfile(p):
        return ""
    try:
        with io.open(p, encoding="utf-8") as f:
            j = json.load(f)
        ns = j.get("notes") or []
        return str(ns[0]).strip() if ns else ""
    except Exception:
        return ""


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


# ---------------- 输出 ----------------

def emit_md():
    ts = decorate()
    L = []
    A = L.append
    A("# 《人工智能技术及应用》综合项目 · 难度系数表\n")
    A("> 分组：3 人/组，每组 1 题，同一算法最多 3 组。　评分：**最终 = min(100, 原始分 × K)**。\n")
    A("> 选题时即锁定难度系数 K，详见《项目任务书》§2。\n")

    A("\n## 一、难度怎么算出来的\n")
    A("**难度分 S** = 五个维度的加权和（每维 1–5 分，S ∈ [1, 5]）：\n")
    A("```")
    A("S = 0.25·M + 0.20·C + 0.25·V + 0.15·A + 0.15·R")
    A("```\n")
    A("| 维度 | 含义 | 权重 | 1 分 ↔ 5 分 |")
    A("|---|---|---|---|")
    for k, name, w, rub in DIM_RUBRIC:
        A("| **%s** %s | %.2f | %s |" % (k, name, w, rub))
    A("")
    A("其中 **R（资源支撑度）是逆向计分**：越缺乏可对照的基线，做起来越难。"
      "依据是 `experiments/README.md` §5 列出的 7 个因本机无 torch、无 GPU 而只能做替代验证的算法。\n")
    A("**难度系数 K** 由 S 归一化而来，步长固定 0.05（共 11 档），避免成绩册出现 1.1372 这类数字：\n")
    A("```")
    A("K = 1.00 + 0.05 × floor( 2.5 × (S − 1) + 0.5 )")
    A("```\n")
    A("| 档位 | K 区间 | 含义 | 题目数 |")
    A("|---|---|---|---|")
    cnt = {}
    for t in ts:
        cnt[t["tier"]] = cnt.get(t["tier"], 0) + 1
    for k, lo, hi in (("A", "1.05–1.10", "基础"), ("B", "1.15–1.20", "标准"),
                      ("C", "1.25–1.35", "进阶"), ("D", "1.40–1.50", "挑战")):
        A("| %s | %s | %s | %d |" % (k, lo, hi, cnt.get(k, 0)))

    A("\n## 二、30 个算法的难度系数\n")
    A("| 编号 | 算法 | 层 | ★ | M | C | V | A | R | S | **K** | 档 | 人时 | 参考脚本 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in ts:
        A("| %s | %s%s | %s | %s | %d | %d | %d | %d | %d | %.2f | **%.2f** | %s | %d | `%s` |" % (
            t["id"], t["n"], "", t["l"],
            "★" if t["star"] else "",
            t["d"]["M"], t["d"]["C"], t["d"]["V"], t["d"]["A"], t["d"]["R"],
            t["S"], t["K"], t["tier"], t["hours"], t["exp"]))
    A("")
    A("> ★ 为本课重点算法，优先推荐选择。\n")

    A("\n## 三、分数怎么换算\n")
    A("```")
    A("最终 = min( 100 , 原始分 × K )")
    A("```\n")
    A("| 原始分 | K=1.05（最易） | K=1.15 | K=1.30 | K=1.50（最难） |")
    A("|---|---|---|---|---|")
    for raw in (45, 60, 75):
        row = "| %d%s |" % (raw, "（保底）" if raw == 45 else ("（及格）" if raw == 60 else "（中等）"))
        for k in (1.05, 1.15, 1.30, 1.50):
            row += " %.1f |" % min(100, round(raw * k, 1))
        A(row)
    A("")
    A("| | 最易题 (K=1.05) | 最难题 (K=1.50) |")
    A("|---|---|---|")
    A("| 拿到 100 分需要 | 原始分 95.2 | 原始分 66.7 |")
    A("| 原始分 60 的换算 | 63.0 | 90.0 |")
    A("")
    A("两条补充规则：\n")
    A("- **封顶**：所有题目上限 100 分，简单题做不出超额分。\n"
      "- **保底**：四项交付物齐全且自动初筛全绿，原始分不低于 45 分"
      "（最难题 K=1.50 时换算为 67.5 分）。初筛没过则无保底 —— 避免无脑选难题。\n")
    A("系数上限取 1.50 而不是 2.00，是因为 2.00 会让简单题组即使满分也只有 100，"
      "而难题组 50 分就够了，公平感会崩塌。1.50 使「最难题满绩门槛 66.7」与"
      "「最易题良好线 85」大致对齐。\n")

    A("\n## 四、选题建议（面向智能制造专业）\n")
    reco = ["L1-01", "L3-01", "L4-01", "L4-04", "L5-01", "L2-02",
            "L6-01", "L6-02", "L8-03", "L4-05", "L7-02", "L5-02", "L1-03"]
    A("**强烈推荐**（场景贴合且资源齐全）：")
    A("、".join("`%s` %s" % (t["id"], t["n"]) for t in ts if t["id"] in reco) + "。\n")
    care = {
        "L3-04": "易做成调库画图，须强制 perplexity 滑块并解释「距离不可解释」",
        "L3-05": "数学门槛高，须绑定多源振动信号分离的应用叙事",
        "L6-03": "谱系树绘制工作量大，建议层次聚类与 GMM 二选一",
        "L7-04": "全班最多 1 组，交付物限定为注意力可视化",
        "L8-02": "默认关闭，需要申请并签署承诺书",
    }
    A("**需谨慎或有限制**：\n")
    for tid, why in care.items():
        t = [x for x in ts if x["id"] == tid][0]
        A("- `%s` %s（K=%.2f）：%s" % (tid, t["n"], t["K"], why))
    A("")
    A("\n## 五、题意、参考资源与踩坑提示\n")
    A("> 选题登记与详细信息见同目录 `选题系统.html`。\n")
    for t in ts:
        flag = "" if t["open"] else "　🚫 **默认关闭**"
        A("### %s　%s（K=%.2f · %s 档 · 约 %d 人时）%s\n"
          % (t["id"], t["n"], t["K"], t["tier"], t["hours"], flag))
        A("- **典型场景**：%s" % t["scene"])
        A("- **参考脚本**：`experiments/%s`（seed=42）" % t["exp"])
        A("- **现成演示**：`outputs/AI算法大纲-交互版.html` 第 %d 行起的 `regDemo`" % t["demo"])
        A("- **离线数据集**：%s" % "、".join("`%s`" % d for d in t["datasets"]))
        note = _notes_of(t["id"])
        if note:
            A("- **参考基准**：%s" % note)
        A("- **交互作品必须包含**：")
        for m in t["mustHave"]:
            A("  1. %s" % m)
        A("- **常见踩坑**：")
        for p in t["pitfalls"]:
            A("  - %s" % p)
        if t["note"]:
            A("- **限制说明**：%s" % t["note"])
        A("")
    return "\n".join(L) + "\n"


def emit_js():
    ts = decorate()
    for t in ts:
        t["baseline"] = _notes_of(t["id"])
    return json.dumps(ts, ensure_ascii=False, indent=1)


def check():
    ts = decorate()
    err = []

    if len(ts) != 30:
        err.append("题目数应为 30，实际 %d" % len(ts))

    ids = [t["id"] for t in ts]
    if len(set(ids)) != len(ids):
        err.append("编号有重复")

    for t in ts:
        if abs(sum(W[k] * t["d"][k] for k in DIMS) - t["S"]) > 1e-9:
            err.append("%s 的 S 与五维不符" % t["id"])
        for k in DIMS:
            if not (1 <= t["d"][k] <= 5):
                err.append("%s 的 %s=%s 超出 1–5" % (t["id"], k, t["d"][k]))
        if not (1.0 <= t["K"] <= 1.5):
            err.append("%s 的 K=%.2f 越界" % (t["id"], t["K"]))
        if abs(t["K"] * 100 - round(t["K"] * 100)) > 1e-9 or \
           abs((t["K"] - 1.0) * 20 - round((t["K"] - 1.0) * 20)) > 1e-6:
            err.append("%s 的 K=%.2f 不是 0.05 的整数倍" % (t["id"], t["K"]))
        exp = t["exp"]
        if not os.path.isfile(os.path.join(ROOT, "experiments", exp)):
            err.append("%s 的脚本 experiments/%s 不存在" % (t["id"], exp))
        if not os.path.isfile(os.path.join(METRICS, t["id"] + ".json")):
            err.append("%s 缺少 results/metrics/%s.json" % (t["id"], t["id"]))

    # 与主页面 DATA 对齐
    if os.path.isfile(HINT_HTML):
        with io.open(HINT_HTML, encoding="utf-8") as f:
            html = f.read()
        names = set(re.findall(r'\bn:\s*"([^"]+)"', html))
        for t in ts:
            if t["n"] not in names:
                err.append("%s 的算法名「%s」在主页面 DATA 中找不到" % (t["id"], t["n"]))
        # demo 行号必须真的落在某个 regDemo 上
        lines = html.split("\n")
        for t in ts:
            ln = t["demo"]
            if ln > len(lines) or "regDemo(" not in lines[ln - 1]:
                err.append("%s 的 demo 行号 %d 不是 regDemo 起始行" % (t["id"], ln))
    else:
        err.append("找不到主页面 HTML，跳过对齐检查")

    rho = spearman([t["hours"] for t in ts], [t["K"] for t in ts])
    if rho < 0.8:
        err.append("人时与难度的 Spearman 相关仅 %.3f（应 > 0.8），两者口径不一致" % rho)

    # 档位分布
    cnt = {}
    for t in ts:
        cnt[t["tier"]] = cnt.get(t["tier"], 0) + 1
    print("难度档位分布： " + "  ".join(
        "%s 档 %d 题" % (k, cnt.get(k, 0)) for k in "ABCD"))
    print("K 值区间： %.2f ~ %.2f" % (min(t["K"] for t in ts), max(t["K"] for t in ts)))
    print("人时与难度 Spearman ρ = %.3f" % rho)
    print("开放题目： %d / 30（默认关闭：%s）" % (
        sum(1 for t in ts if t["open"]),
        "、".join(t["n"] for t in ts if not t["open"]) or "无"))

    if err:
        print("\n自检未通过，共 %d 项：" % len(err))
        for e in err:
            print("  × " + e)
        return 1
    print("\n自检通过：30 题，公式、范围、脚本、演示行号、数据对齐全部正确。")
    return 0


def table():
    ts = decorate()
    print("%-7s %-26s %-4s %-3s %-6s %-5s %-5s %-6s" %
          ("编号", "算法", "层", "★", "S", "K", "档", "人时"))
    print("-" * 72)
    for t in ts:
        print("%-7s %-26s %-4s %-3s %-6.2f %-5.2f %-6s %d" % (
            t["id"], t["n"], t["l"], "★" if t["star"] else "",
            t["S"], t["K"], t["tier"], t["hours"]))
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if arg == "--check":
        sys.exit(check())
    if arg == "--table":
        sys.exit(table())
    if arg == "--emit-md":
        dst = sys.argv[2] if len(sys.argv) > 2 else \
            os.path.join(ROOT, "outputs", "难度系数表.md")
        s = emit_md()
        with io.open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(s)
        print("生成：%s（%d 字符）" % (dst, len(s)))
        sys.exit(0)
    if arg == "--emit-js":
        print(emit_js())
        sys.exit(0)
    print(__doc__)
    sys.exit(2)
