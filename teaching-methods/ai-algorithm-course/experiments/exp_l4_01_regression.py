"""L4-01 线性回归族实验（OLS / Ridge / Lasso / 多项式 / 树模型）

目的：在含非线性与交互项的合成工艺数据上，比较线性模型与非线性模型的表达力与正则化效果。
输入：synth_process(n=1200, seed=42)，特征 [切削速度, 进给量, 切深, 冷却液温度, 主轴振动] -> 表面粗糙度 Ra。
参数：测试集 30%，seed=42；Ridge/Lasso alpha 固定；多项式阶数 2；树 max_depth=5。
"""
import os
import sys

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, data, env_info, mae, r2, rmse  # noqa: E402

SEED = 42


def main(seed=SEED):
    X, y, desc = data.synth_process(seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    exp = Exp(exp_id="L4-01", algo="线性回归 / Ridge / Lasso / 多项式回归",
              layer="L4 监督学习",
              purpose="比较线性、正则化线性与非线性模型在含交互项的工艺数据上的拟合能力",
              dataset="synth_process(n=1200)；%s；测试集 30%%" % desc,
              baselines=["均值预测 DummyRegressor", "决策树 / 随机森林（非线性参照）"],
              metrics=["RMSE", "MAE", "R²", "非零系数个数", "耗时"],
              seed=seed, params={"ridge_alpha": 1.0, "lasso_alpha": 0.01,
                                 "poly_degree": 2, "tree_depth": 5},
              env=env_info())

    models = [
        ("DummyRegressor(均值)", DummyRegressor()),
        ("线性回归 OLS", make_pipeline(StandardScaler(), LinearRegression())),
        ("Ridge(alpha=1)", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("Lasso(alpha=0.01)", make_pipeline(StandardScaler(), Lasso(alpha=0.01))),
        ("多项式回归(2阶)", make_pipeline(StandardScaler(),
                                        PolynomialFeatures(2, include_bias=False),
                                        Ridge(alpha=1.0))),
        ("决策树(max_depth=5)", DecisionTreeRegressor(max_depth=5,
                                                      random_state=seed)),
        ("随机森林(200)", RandomForestRegressor(n_estimators=200,
                                                random_state=seed)),
    ]
    for name, m in models:
        with exp.timer(name):
            m.fit(Xtr, ytr)
            p = m.predict(Xte)
        coef = getattr(m, "coef_", None)
        if coef is None:
            inner = m.steps[-1][1] if hasattr(m, "steps") else m
            coef = getattr(inner, "coef_", None)
        nz = int(np.sum(np.abs(np.ravel(coef)) > 1e-8)) if coef is not None else -1
        exp.add(模型=name, RMSE=rmse(yte, p), MAE=mae(yte, p), R2=r2(yte, p),
                非零系数=nz)

    exp.figure("l4_01_reg", "bar", "回归模型测试集 RMSE 对比",
               [r["模型"] for r in exp.rows[1:]],
               [("RMSE", [r["RMSE"] for r in exp.rows[1:]])],
               ylabel="RMSE（表面粗糙度）")
    lin = [r for r in exp.rows if r["模型"] == "线性回归 OLS"][0]
    rf = [r for r in exp.rows if r["模型"] == "随机森林(200)"][0]
    poly = [r for r in exp.rows if r["模型"] == "多项式回归(2阶)"][0]
    exp.note("线性模型 RMSE %.4f / R² %.4f，随机森林 RMSE %.4f / R² %.4f：数据生成式中含 62·f² 与 f·vib 交互项，"
             "线性假设无法表达，非线性模型显著更优。" % (lin["RMSE"], lin["R2"], rf["RMSE"], rf["R2"]))
    exp.note("显式构造二次特征的多项式回归把 RMSE 降到 %.4f，说明「补特征」是弥补线性模型表达力的低成本手段，"
             "代价是特征维度从 5 膨胀到 20。" % poly["RMSE"])
    exp.note("Lasso 的非零系数个数可用于特征筛选；本例特征间相关性不强，正则化收益主要体现在稳定性而非精度。")
    exp.save()


if __name__ == "__main__":
    main()
