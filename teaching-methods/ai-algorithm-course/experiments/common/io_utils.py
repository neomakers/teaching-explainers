"""实验结果记录、计时、表格渲染与零依赖 SVG 图表。

每个实验脚本都通过 Exp 类输出统一结构：
  results/metrics/<exp_id>.json   数值结果（供汇总）
  results/tables/<exp_id>.md      对比表（供报告）
  results/figures/<name>.svg      图表（零依赖，直接嵌入 HTML）
"""
import json
import os
import time
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
for _d in ("metrics", "tables", "figures"):
    os.makedirs(os.path.join(RES, _d), exist_ok=True)


def fmt(v):
    if isinstance(v, float):
        if abs(v) >= 1e5 or (abs(v) < 1e-4 and v != 0):
            return "%.3e" % v
        return "%.4f" % v
    return str(v)


class Exp:
    def __init__(self, exp_id, algo, layer, purpose, dataset, baselines,
                 metrics, seed=42, params=None, env=None):
        self.meta = dict(exp_id=exp_id, algo=algo, layer=layer,
                          purpose=purpose, dataset=dataset,
                          baselines=baselines, metrics=metrics,
                          seed=seed, params=params or {}, env=env or {})
        self.rows, self.timings, self.notes, self.figs = [], {}, [], []

    @contextmanager
    def timer(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + \
                (time.perf_counter() - t0)

    def add(self, **kw):
        self.rows.append(kw)
        return kw

    def note(self, text):
        self.notes.append(text)

    def figure(self, name, kind, title, labels, series, ylabel="", xlabel=""):
        path = os.path.join(RES, "figures", name + ".svg")
        if kind == "bar":
            _bar(path, title, labels, series, ylabel, xlabel)
        else:
            _line(path, title, labels, series, ylabel, xlabel)
        self.figs.append(os.path.relpath(path, RES).replace("\\", "/"))
        return path

    def _tables(self):
        """按行的键集合分组渲染：键结构不同的行自动拆成独立表格。"""
        groups, sigs = [], []
        for r in self.rows:
            sig = tuple(r.keys())
            if sig not in sigs:
                sigs.append(sig)
                groups.append([list(sig), []])
            groups[sigs.index(sig)][1].append(r)
        out = []
        for keys, rows in groups:
            out.append("| " + " | ".join(keys) + " |")
            out.append("|" + "|".join(["---"] * len(keys)) + "|")
            out += ["| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |"
                    for r in rows]
            out.append("")
        return out

    def render_md(self):
        tm = "\n".join("| %s | %.4f s |" % (k, v) for k, v in self.timings.items())
        parts = ["### %s · %s" % (self.meta["exp_id"], self.meta["algo"]),
                 "- 实验目的：%s" % self.meta["purpose"],
                 "- 数据集/用例：%s" % self.meta["dataset"],
                 "- 对比基线：%s" % "；".join(self.meta["baselines"]),
                 "- 评价指标：%s" % "；".join(self.meta["metrics"]),
                 "- 随机种子：%s；参数：%s" % (
                     self.meta["seed"],
                     json.dumps(self.meta["params"], ensure_ascii=False)),
                 ""]
        parts += self._tables()
        if tm:
            parts += ["", "**运行耗时**", "", "| 阶段 | 耗时 |", "|---|---|", tm]
        if self.notes:
            parts += ["", "**结果分析**"] + ["- " + n for n in self.notes]
        if self.figs:
            parts += ["", "**图表**：" + ", ".join("`results/%s`" % f for f in self.figs)]
        return "\n".join(parts)

    def save(self, verbose=True):
        out = dict(meta=self.meta, rows=self.rows, timings=self.timings,
                   notes=self.notes, figures=self.figs)
        with open(os.path.join(RES, "metrics", self.meta["exp_id"] + ".json"),
                  "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        md = self.render_md()
        with open(os.path.join(RES, "tables", self.meta["exp_id"] + ".md"),
                  "w", encoding="utf-8") as f:
            f.write(md)
        if verbose:
            print("\n" + "=" * 78)
            print(md)
        return out


# ---------------- 零依赖 SVG 图表 ----------------
_PAL = ["#185fa5", "#0f6e56", "#ba7517", "#993c1d", "#534ab7", "#5f5e5a"]


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _bar(path, title, labels, series, ylabel, xlabel):
    """series: [(name, [v...]), ...]"""
    n, m = len(labels), max(1, len(series))
    W, H, pad = 720, 340, 56
    iw = W - pad - 24
    vals = [v for _, vs in series for v in vs]
    vmax = max(vals) if vals else 1.0
    vmin = min(0.0, min(vals) if vals else 0.0)
    span = (vmax - vmin) or 1.0
    gw = iw / n
    bw = gw * 0.78 / m
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
           'width="100%%" role="img">' % (W, H)]
    out.append('<title>%s</title>' % _esc(title))
    out.append('<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (W, H))
    out.append('<text x="%d" y="26" font-family="system-ui,sans-serif" '
               'font-size="14" fill="#2c2c2a">%s</text>' % (pad - 40, _esc(title)))
    y0 = H - 60
    for i in range(5):
        yv = vmin + span * i / 4
        y = y0 - (yv - vmin) / span * (y0 - 48)
        out.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="#e3e1da"/>'
                   % (pad - 6, W - 24, y, y))
        out.append('<text x="%d" y="%.1f" font-family="system-ui,sans-serif" '
                   'font-size="11" fill="#888780" text-anchor="end">%s</text>'
                   % (pad - 10, y + 4, ("%.3f" % yv).rstrip("0").rstrip(".")))
    for i, lb in enumerate(labels):
        for j, (nm, vs) in enumerate(series):
            v = vs[i]
            h = max(0.0, (v - vmin) / span * (y0 - 48))
            x = pad + i * gw + j * bw + gw * 0.11
            out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                       'fill="%s"/>' % (x, y0 - h, bw, h, _PAL[j % len(_PAL)]))
        out.append('<text x="%.1f" y="%d" font-family="system-ui,sans-serif" '
                   'font-size="11" fill="#5f5e5a" text-anchor="middle">%s</text>'
                   % (pad + i * gw + gw / 2, y0 + 18, _esc(lb)))
    out.append('<line x1="%d" x2="%d" y1="%d" y2="%d" stroke="#b4b2a9"/>'
               % (pad - 6, W - 24, y0, y0))
    lx = pad - 6
    for j, (nm, _) in enumerate(series):
        out.append('<rect x="%d" y="%d" width="10" height="10" fill="%s"/>'
                   % (lx, H - 26, _PAL[j % len(_PAL)]))
        out.append('<text x="%d" y="%d" font-family="system-ui,sans-serif" '
                   'font-size="11" fill="#5f5e5a">%s</text>'
                   % (lx + 14, H - 18, _esc(nm)))
        lx += 24 + 7 * len(nm)
    if ylabel:
        out.append('<text x="14" y="40" font-family="system-ui,sans-serif" '
                   'font-size="11" fill="#888780">%s</text>' % _esc(ylabel))
    if xlabel:
        out.append('<text x="%d" y="%d" font-family="system-ui,sans-serif" '
                   'font-size="11" fill="#888780" text-anchor="end">%s</text>'
                   % (W - 24, H - 4, _esc(xlabel)))
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def _line(path, title, labels, series, ylabel, xlabel):
    n = len(labels)
    W, H, pad = 720, 340, 56
    iw = W - pad - 24
    vals = [v for _, vs in series for v in vs]
    vmax, vmin = max(vals), min(vals)
    span = (vmax - vmin) or 1.0
    y0, y1 = H - 60, 48
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
           'width="100%%" role="img">' % (W, H)]
    out.append('<title>%s</title>' % _esc(title))
    out.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
    out.append('<text x="14" y="26" font-family="system-ui,sans-serif" '
               'font-size="14" fill="#2c2c2a">%s</text>' % _esc(title))
    for i in range(5):
        yv = vmin + span * i / 4
        y = y0 - (yv - vmin) / span * (y0 - y1)
        out.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="#e3e1da"/>'
                   % (pad - 6, W - 24, y, y))
        out.append('<text x="%d" y="%.1f" font-family="system-ui,sans-serif" '
                   'font-size="11" fill="#888780" text-anchor="end">%s</text>'
                   % (pad - 10, y + 4, ("%.3f" % yv).rstrip("0").rstrip(".")))
    for j, (nm, vs) in enumerate(series):
        pts = " ".join("%.1f,%.1f" % (pad + i * iw / max(1, n - 1),
                                      y0 - (v - vmin) / span * (y0 - y1))
                       for i, v in enumerate(vs))
        out.append('<polyline points="%s" fill="none" stroke="%s" '
                   'stroke-width="2"/>' % (pts, _PAL[j % len(_PAL)]))
        for i, v in enumerate(vs):
            out.append('<circle cx="%.1f" cy="%.1f" r="2.6" fill="%s"/>'
                       % (pad + i * iw / max(1, n - 1),
                          y0 - (v - vmin) / span * (y0 - y1),
                          _PAL[j % len(_PAL)]))
    for i, lb in enumerate(labels):
        if n <= 14 or i % (n // 10 or 1) == 0:
            out.append('<text x="%.1f" y="%d" font-family="system-ui,sans-serif"'
                       ' font-size="10" fill="#5f5e5a" text-anchor="middle">%s</text>'
                       % (pad + i * iw / max(1, n - 1), y0 + 18, _esc(lb)))
    out.append('<line x1="%d" x2="%d" y1="%d" y2="%d" stroke="#b4b2a9"/>'
               % (pad - 6, W - 24, y0, y0))
    lx = pad - 6
    for j, (nm, _) in enumerate(series):
        out.append('<rect x="%d" y="%d" width="10" height="10" fill="%s"/>'
                   % (lx, H - 26, _PAL[j % len(_PAL)]))
        out.append('<text x="%d" y="%d" font-family="system-ui,sans-serif" '
                   'font-size="11" fill="#5f5e5a">%s</text>'
                   % (lx + 14, H - 18, _esc(nm)))
        lx += 24 + 7 * len(nm)
    if xlabel:
        out.append('<text x="%d" y="%d" font-family="system-ui,sans-serif" '
                   'font-size="11" fill="#888780" text-anchor="end">%s</text>'
                   % (W - 24, H - 4, _esc(xlabel)))
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
