"""一键运行全部实验并生成报告。

用法：
    python run_all.py                 # 运行全部实验
    python run_all.py --only l3       # 只运行文件名含 l3 的实验
    python run_all.py --report-only   # 不重跑，仅根据已有 results/metrics/*.json 生成报告

产物：
    results/metrics/<exp_id>.json     每个实验的数值结果
    results/tables/<exp_id>.md        每个实验的对比表
    results/figures/*.svg             图表（零依赖生成）
    results/REPORT.md                 总报告（含实验清单与主要结论）
    results/实验报告.html             带内嵌图表的完整报告
    results/logs/<script>.log        每个脚本的运行日志
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
LOG = os.path.join(RES, "logs")
os.makedirs(LOG, exist_ok=True)
PY = sys.executable


def run_experiments(only=None):
    files = sorted(glob.glob(os.path.join(ROOT, "exp_*.py")))
    if only:
        files = [f for f in files if only in os.path.basename(f)]
    ok, fail = [], []
    for f in files:
        name = os.path.basename(f)
        t0 = time.perf_counter()
        p = subprocess.run([PY, f], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=ROOT)
        dt = time.perf_counter() - t0
        with open(os.path.join(LOG, name + ".log"), "w", encoding="utf-8") as fh:
            fh.write(p.stdout + ("\n[STDERR]\n" + p.stderr if p.stderr else ""))
        status = "OK  " if p.returncode == 0 else "FAIL"
        print("[%s] %-34s %6.1f s" % (status, name, dt))
        (ok if p.returncode == 0 else fail).append((name, dt))
    print("\n完成：成功 %d，失败 %d，总耗时 %.1f s"
          % (len(ok), len(fail), sum(d for _, d in ok + fail)))
    if fail:
        print("失败清单：" + ", ".join(n for n, _ in fail))
    return ok, fail


def load_metrics():
    out = []
    for fp in sorted(glob.glob(os.path.join(RES, "metrics", "*.json"))):
        with open(fp, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def headline(m):
    """从结果行中挑一句话总结（优先取最后一个数值行的关键信息）。"""
    notes = m.get("notes") or []
    return notes[0] if notes else "（无结论文本）"


def build_report(all_m):
    lines = ["# 《人工智能技术及应用》算法实验报告", ""]
    env = all_m[0]["meta"].get("env", {}) if all_m else {}
    lines += ["运行环境：Python %s / numpy %s / scikit-learn %s（%s）"
              % (env.get("python", "?"), env.get("numpy", "?"),
                 env.get("sklearn", "?"), env.get("platform", "?")), ""]
    lines += ["共 %d 个实验，全部独立可复现：每个脚本固定随机种子、显式声明输入数据与参数，"
              "可直接 `python <脚本>` 单独运行。" % len(all_m), ""]

    lines += ["## 一、实验清单", "",
              "| 编号 | 算法 | 层级 | 实验文件 | 数据集/用例 | 主要结论 |",
              "|---|---|---|---|---|---|"]
    for m in all_m:
        mt = m["meta"]
        lines.append("| %s | %s | %s | `%s` | %s | %s |"
                     % (mt["exp_id"], mt["algo"], mt["layer"],
                        script_of(mt["exp_id"]), short(mt["dataset"]),
                        short(headline(m), 90)))
    lines.append("")

    lines += ["## 二、耗时总览", "", "| 编号 | 算法 | 总耗时(s) | 阶段数 |", "|---|---|---|---|"]
    for m in all_m:
        tot = sum(v for k, v in m["timings"].items() if not k.startswith("重建")
                  and not k.startswith("fit dim"))
        lines.append("| %s | %s | %.2f | %d |"
                     % (m["meta"]["exp_id"], m["meta"]["algo"],
                        sum(m["timings"].values()), len(m["timings"])))
    lines.append("")

    lines += ["## 三、各实验详情", ""]
    for m in all_m:
        mt = m["meta"]
        lines += ["### %s · %s" % (mt["exp_id"], mt["algo"]),
                  "- 实验文件：`%s`" % script_of(mt["exp_id"]),
                  "- 目的：%s" % mt["purpose"],
                  "- 数据集/用例：%s" % mt["dataset"],
                  "- 对比基线：%s" % "；".join(mt["baselines"]),
                  "- 评价指标：%s" % "；".join(mt["metrics"]),
                  "- 随机种子：%s；参数：%s" % (mt["seed"],
                                          json.dumps(mt["params"], ensure_ascii=False)),
                  ""]
        keys = []
        for r in m["rows"]:
            for k in r:
                if k not in keys:
                    keys.append(k)
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---"] * len(keys)) + "|")
        for r in m["rows"]:
            lines.append("| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |")
        lines.append("")
        if m["timings"]:
            lines += ["**耗时**：" + "；".join("%s %.3fs" % (k, v)
                                            for k, v in m["timings"].items()), ""]
        if m["notes"]:
            lines += ["**结果分析**"] + ["- " + n for n in m["notes"]] + [""]
        if m["figures"]:
            lines += ["**图表**：" + ", ".join("`results/%s`" % f for f in m["figures"]), ""]
    lines += ["## 四、难以直接实验的算法与替代方案", "",
              "| 算法 | 原因 | 已采用的替代验证 |", "|---|---|---|",
              "| CNN（端到端训练） | 未安装 PyTorch；显卡损坏，纯 CPU 训练不可复现 | "
              "算子正确性（对照 scipy）+ 固定核卷积特征的分类收益 + 参数量对比（L7-02） |",
              "| LSTM / GRU | 同上，门控单元标准实现依赖 torch | "
              "numpy 从零实现 Elman RNN，与 AR(p)、MLP 窗口基线三方对比（L7-03） |",
              "| Transformer / 大模型 | 需 torch + 大规模语料 + 显存 | "
              "验证自注意力三条数学性质：排列等变性、内容寻址、参数量解耦（L7-04） |",
              "| DQN / PPO | 需 torch 与大量交互步数；gym 未安装 | "
              "自建网格环境 + 表格 Q-Learning，与价值迭代最优解对照（L8-01） |",
              "| GAN / Diffusion | 需 GPU；FID 等指标需预训练网络 | "
              "还原为密度估计与采样：GMM / KDE，用 Wasserstein、MMD、NLL 评价（L8-02） |",
              "| XGBoost | 未安装 | "
              "sklearn HistGradientBoostingClassifier（同为直方图梯度提升）等价替代（L5-02） |",
              "| UMAP | 未安装 | t-SNE + PCA 对照；安装 umap-learn 后脚本自动纳入（L3-04） |", ""]
    return "\n".join(lines)


def script_of(exp_id):
    for f in sorted(glob.glob(os.path.join(ROOT, "exp_*.py"))):
        with open(f, encoding="utf-8") as fh:
            head = fh.read(4000)
        if 'exp_id="%s"' % exp_id in head:
            return os.path.basename(f)
    return "-"


def fmt(v):
    if isinstance(v, float):
        if abs(v) >= 1e5 or (abs(v) < 1e-4 and v != 0):
            return "%.3e" % v
        return "%.4f" % v
    return str(v)


def short(s, n=60):
    s = str(s).replace("\n", " ").replace("|", "/")
    return s if len(s) <= n else s[:n - 1] + "…"


CSS = """
body{background:#f6f7f9;color:#2c2c2a;font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;
     font-size:14px;line-height:1.65;margin:0;padding:0 20px 60px}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:21px;font-weight:500;padding:20px 0 4px}
h2{font-size:17px;font-weight:500;margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid #e3e1da}
h3{font-size:15px;font-weight:500;margin:22px 0 8px;background:#e6f1fb;padding:8px 12px;border-radius:8px}
table{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;overflow:hidden;
      font-size:13px;margin:8px 0 14px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #e3e1da}
th{background:#f1efe8;font-weight:500}
ul{color:#5f5e5a}
code{background:#f1efe8;padding:1px 5px;border-radius:4px;font-size:12px}
.meta{background:#fff;border:1px solid #e3e1da;border-radius:10px;padding:12px 16px;margin:10px 0}
.meta li{margin:3px 0}
figure{margin:10px 0 18px;background:#fff;border:1px solid #e3e1da;border-radius:10px;padding:10px}
.note{background:#faeeda;border-radius:8px;padding:8px 14px;color:#854f0b;font-size:13px;margin:8px 0}
"""


def build_html(all_m):
    p = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>人工智能技术及应用 · 算法实验报告</title>',
         '<style>%s</style></head><body><div class="wrap">' % CSS]
    env = all_m[0]["meta"].get("env", {}) if all_m else {}
    p += ["<h1>《人工智能技术及应用》算法实验报告</h1>",
          '<div class="meta"><ul>',
          "<li>运行环境：Python %s / numpy %s / scikit-learn %s</li>"
          % (env.get("python", "?"), env.get("numpy", "?"), env.get("sklearn", "?")),
          "<li>实验数量：%d；全部独立可复现（固定随机种子、显式声明数据与参数）</li>" % len(all_m),
          "</ul></div>"]
    p += ["<h2>一、实验清单</h2>", "<table><tr><th>编号</th><th>算法</th><th>层级</th>"
          "<th>实验文件</th><th>数据集/用例</th><th>主要结论</th></tr>"]
    for m in all_m:
        mt = m["meta"]
        p.append("<tr><td>%s</td><td>%s</td><td>%s</td><td><code>%s</code></td>"
                 "<td>%s</td><td>%s</td></tr>"
                 % (mt["exp_id"], mt["algo"], mt["layer"],
                    script_of(mt["exp_id"]), short(mt["dataset"], 70),
                    short(headline(m), 110)))
    p.append("</table>")

    p += ["<h2>二、各实验详情</h2>"]
    for m in all_m:
        mt = m["meta"]
        p += ["<h3>%s · %s</h3>" % (mt["exp_id"], mt["algo"]),
              '<div class="meta"><ul>',
              "<li>实验文件：<code>%s</code></li>" % script_of(mt["exp_id"]),
              "<li>目的：%s</li>" % mt["purpose"],
              "<li>数据集/用例：%s</li>" % mt["dataset"],
              "<li>对比基线：%s</li>" % "；".join(mt["baselines"]),
              "<li>评价指标：%s</li>" % "；".join(mt["metrics"]),
              "<li>随机种子：%s；参数：<code>%s</code></li>"
              % (mt["seed"], json.dumps(mt["params"], ensure_ascii=False)),
              "</ul></div>"]
        keys = []
        for r in m["rows"]:
            for k in r:
                if k not in keys:
                    keys.append(k)
        p.append("<table><tr>" + "".join("<th>%s</th>" % k for k in keys) + "</tr>")
        for r in m["rows"]:
            p.append("<tr>" + "".join("<td>%s</td>" % fmt(r.get(k, ""))
                                       for k in keys) + "</tr>")
        p.append("</table>")
        if m["timings"]:
            p.append('<div class="meta">耗时：' +
                     "；".join("%s %.3fs" % (k, v) for k, v in m["timings"].items())
                     + "</div>")
        for n in m["notes"]:
            p.append('<div class="note">%s</div>' % n)
        for f in m["figures"]:
            fp = os.path.join(RES, f)
            if os.path.exists(fp):
                with open(fp, encoding="utf-8") as fh:
                    svg = fh.read()
                svg = svg.replace('width="100%"', 'width="100%" height="auto"', 1)
                p.append("<figure>%s</figure>" % svg)
    p += ["<h2>三、难以直接实验的算法与替代方案</h2>",
          "<table><tr><th>算法</th><th>原因</th><th>已采用的替代验证</th></tr>",
          "<tr><td>CNN（端到端训练）</td><td>未安装 PyTorch；显卡损坏</td>"
          "<td>算子对照 scipy + 固定核卷积特征分类收益 + 参数量对比（L7-02）</td></tr>",
          "<tr><td>LSTM / GRU</td><td>标准实现依赖 torch</td>"
          "<td>numpy 从零实现 Elman RNN，与 AR(p)、MLP 窗口基线对比（L7-03）</td></tr>",
          "<tr><td>Transformer</td><td>需 torch + 语料 + 显存</td>"
          "<td>自注意力三性质：排列等变性 / 内容寻址 / 参数量解耦（L7-04）</td></tr>",
          "<tr><td>DQN / PPO</td><td>需 torch 与大量交互；gym 未安装</td>"
          "<td>自建网格环境 + 表格 Q-Learning，与价值迭代最优解对照（L8-01）</td></tr>",
          "<tr><td>GAN / Diffusion</td><td>需 GPU；FID 需预训练网络</td>"
          "<td>密度估计与采样：GMM / KDE，用 Wasserstein、MMD、NLL 评价（L8-02）</td></tr>",
          "<tr><td>XGBoost</td><td>未安装</td>"
          "<td>HistGradientBoosting 等价替代（L5-02）</td></tr>",
          "<tr><td>UMAP</td><td>未安装</td>"
          "<td>t-SNE + PCA 对照；安装后脚本自动纳入（L3-04）</td></tr>",
          "</table></div></body></html>"]
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只运行文件名包含该关键字的实验")
    ap.add_argument("--report-only", action="store_true", help="只生成报告，不重跑实验")
    a = ap.parse_args()
    if not a.report_only:
        run_experiments(a.only)
    all_m = load_metrics()
    if not all_m:
        print("没有找到实验结果，请先运行实验。")
        return
    md = build_report(all_m)
    with open(os.path.join(RES, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(RES, "实验报告.html"), "w", encoding="utf-8") as f:
        f.write(build_html(all_m))
    print("\n报告已生成：\n  %s\n  %s"
          % (os.path.join(RES, "REPORT.md"),
             os.path.join(RES, "实验报告.html")))


if __name__ == "__main__":
    main()
