"""把实验套件（30 个实验的结果/图表 + 学生指导书）注入交互网页。

用法：
    python tools/build_lab_page.py            # 默认注入 ../../outputs/AI算法大纲-交互版.html

注意：本脚本的「实验档案」区已改为条目视图（与大纲页同风格、行内展开）；
网页当前退回为卡片 + 弹窗版，若重跑本脚本，实验档案会切换为条目版（两者功能等价）。

原理：网页中预留三对锚点注释，脚本把生成内容写入锚点之间，可重复执行（幂等）：
    <!--NAV_LAB:BEGIN--> ... <!--NAV_LAB:END-->      导航按钮
    <!--PANE_LAB:BEGIN--> ... <!--PANE_LAB:END-->    实验中心 / 实验指导书面板
    <!--DATA_LAB:BEGIN--> ... <!--DATA_LAB:END-->    实验数据（JSON，含内嵌 SVG）
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from md2html import convert as md2html, CSS   # noqa: E402

DEFAULT_HTML = os.path.join(
    os.path.dirname(ROOT), "outputs", "AI算法大纲-交互版.html")


def script_of(exp_id):
    """根据 exp_id 反查实验脚本文件名。"""
    for f in sorted(glob.glob(os.path.join(ROOT, "exp_*.py"))):
        with open(f, encoding="utf-8") as fh:
            if 'exp_id="%s"' % exp_id in fh.read(4000):
                return os.path.basename(f)
    return "-"


def load_experiments():
    out = []
    for fp in sorted(glob.glob(os.path.join(RES, "metrics", "*.json"))):
        with open(fp, encoding="utf-8") as fh:
            d = json.load(fh)
        m = d["meta"]
        figs = []
        for rel in d.get("figures", []):
            p = os.path.join(RES, rel)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    svg = fh.read()
                svg = svg.replace('width="100%"', 'width="100%" height="auto"', 1)
                figs.append({"name": os.path.basename(rel), "svg": svg})
        timings = d.get("timings", {})
        out.append({
            "id": m["exp_id"], "algo": m["algo"], "layer": m["layer"],
            "file": script_of(m["exp_id"]), "purpose": m["purpose"],
            "dataset": m["dataset"], "baselines": m["baselines"],
            "metrics": m["metrics"], "seed": m["seed"],
            "params": m["params"], "rows": d.get("rows", []),
            "notes": d.get("notes", []), "figs": figs,
            "total_time": round(sum(timings.values()), 3),
            "timings": {k: round(v, 4) for k, v in timings.items()},
        })
    return out


def nav_html(exps):
    return ('<button data-p="lab">实验中心 (%d)</button>\n'
            '      <button data-p="book">实验指导书</button>' % len(exps))


def pane_html(exps):
    layers = sorted({e["layer"][:2] for e in exps})
    chips = "".join('<button class="chip" data-lf="%s">%s</button>' % (l, l)
                    for l in layers)
    return """
  <section class="pane" id="p-lab">
    <div class="lead">
      <b>实验档案：</b>30 个实验全部独立可复现（固定随机种子 42、显式声明数据集与参数），
      点条目展开即可看到完整方案（数据集 / 基线 / 指标 / 参数）、结果对比表、耗时与图表。
      数值均由 <code>experiments/</code> 下的脚本真实运行产出，可用
      <code>python exp_l3_01_pca.py</code> 单独复现。<br>
      要把实验<b>挂到某个大纲条目</b>上（勾选、改参数、导出计划），请到「大纲 · 关联实验」页操作。
    </div>
    <div class="toolbar">
      <input id="labq" placeholder="搜索算法、数据集或结论关键词，例如：降维、准确率、噪声">
      <button class="chip on" data-lf="all">全部</button>
      %s
      <button class="chip" data-lf="slow">耗时 &gt; 10s</button>
    </div>
    <div id="labList"></div>
  </section>

  <section class="pane" id="p-book">
    <div class="lead">
      <b>学生实验指导书：</b>8 个课内实验单元，每单元含目的、操作步骤、结果记录表、思考题与常见错误；
      附录含 30 个脚本索引、报告模板、评分标准与综合项目选题。页面可直接打印（Ctrl+P）。
    </div>
    <div class="book">%s</div>
  </section>
""" % (chips, book_html())


def book_html():
    """把 outputs/学生实验指导书.md 转成 HTML 片段（去掉外层 html/head/body）。"""
    md_path = os.path.join(os.path.dirname(ROOT), "outputs", "学生实验指导书.md")
    if not os.path.exists(md_path):
        return "<p>（未找到学生实验指导书.md）</p>"
    with open(md_path, encoding="utf-8") as fh:
        md = fh.read()
    full = md2html(md, "学生实验指导书")
    body = full.split('<div class="wrap">', 1)[1]
    body = body.rsplit("<footer>", 1)[0]
    return body


def fmt_js(v):
    if isinstance(v, float):
        if v != v:                       # NaN -> null（JSON 不支持 NaN）
            return "null"
        if abs(v) >= 1e5 or (abs(v) < 1e-4 and v != 0):
            return '"%.3e"' % v
        return '"%.4f"' % v
    return json.dumps(v, ensure_ascii=False)


def clean_nan(obj):
    """NaN 在 JSON 中不合法，统一置空，避免出现裸 NaN 字面量。"""
    if isinstance(obj, float):
        return None if obj != obj else obj
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj


def data_js(exps):
    payload = json.dumps(clean_nan(exps), ensure_ascii=False, allow_nan=False)
    payload = payload.replace("</", "<\\/")     # 防止提前结束 script
    return "const LAB = %s;\n" % payload


LAB_JS = r"""
/* ---------- 实验档案：条目视图（与大纲页同风格，行内展开） ---------- */
function fmtCell(v){
  if(v === null || v === undefined) return '';
  if(typeof v === 'number'){
    if(isNaN(v)) return '';
    if(Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-4 && v !== 0)) return v.toExponential(3);
    return v.toFixed(4);
  }
  return String(v);
}
function labTable(rows){
  if(!rows || !rows.length) return '<div class="mini">（该实验无表格结果）</div>';
  const keys = [];
  rows.forEach(r => Object.keys(r).forEach(k => { if(keys.indexOf(k) < 0) keys.push(k); }));
  let h = '<table><tr>' + keys.map(k => '<th>' + k + '</th>').join('') + '</tr>';
  rows.forEach(r => {
    h += '<tr>' + keys.map(k => '<td>' + fmtCell(r[k]) + '</td>').join('') + '</tr>';
  });
  return h + '</table>';
}
const LABOPEN = (function(){
  try{ return JSON.parse(localStorage.getItem('ai_lab_open')) || {}; }catch(e){ return {}; }
})();
function labSave(){
  try{ localStorage.setItem('ai_lab_open', JSON.stringify(LABOPEN)); }catch(e){}
}
function labClean(s){
  return (typeof cleanText === 'function') ? cleanText(s) : String(s);
}
function labShort(s, n){
  const t = labClean(s);
  return t.length <= n ? t : t.slice(0, n - 1) + '…';
}
function labDetail(e){
  let h = '<div class="detail"><div class="m" style="font-size:12px;color:var(--text2)">'
    + '<b>目的</b>：' + e.purpose + '<br>'
    + '<b>数据集</b>：' + labShort(e.dataset, 120) + '<br>'
    + '<b>基线</b>：' + e.baselines.join('；') + '<br>'
    + '<b>指标</b>：' + e.metrics.join('；') + '<br>'
    + '<b>种子</b>：' + e.seed + '　<b>参数</b>：<code>'
    + JSON.stringify(e.params).replace(/"/g, '') + '</code></div>';
  h += labTable(e.rows);
  const tk = Object.keys(e.timings);
  if(tk.length){
    h += '<div class="mini">分阶段耗时：' + tk.map(k => k + ' ' +
      e.timings[k].toFixed(3) + 's').join('，') + '</div>';
  }
  if(e.notes && e.notes.length){
    h += '<div style="margin-top:8px">' + e.notes.map(n =>
      '<div class="mini" style="color:var(--text2)">· ' + labShort(n, 160) + '</div>').join('')
      + '</div>';
  }
  if(e.figs && e.figs.length){
    h += '<div style="margin-top:8px">' + e.figs.map(f => f.svg).join('') + '</div>';
  }
  return h + '</div>';
}
function renderLab(filter, q){
  const box = document.getElementById('labList');
  if(!box) return;
  box.innerHTML = '';
  const groups = [];
  LAB.forEach(e => {
    const L = String(e.layer).slice(0, 2);
    if(filter !== 'all' && filter !== 'slow' && L !== filter) return;
    if(filter === 'slow' && e.total_time <= 10) return;
    if(q){
      const s = (e.id + e.algo + e.file + e.dataset + e.purpose +
                 e.notes.join('') + e.metrics.join('')).toLowerCase();
      if(s.indexOf(q.toLowerCase()) < 0) return;
    }
    let g = null;
    groups.forEach(x => { if(x.k === L) g = x; });
    if(!g){ g = {k: L, name: e.layer, items: []}; groups.push(g); }
    g.items.push(e);
  });
  groups.forEach(g => {
    const sec = document.createElement('div');
    sec.className = 'layer';
    const hd = document.createElement('div');
    hd.className = 'layer-hd';
    hd.innerHTML = '<span class="tag">' + g.k + '</span><h2>' + g.name +
      '</h2><span class="desc">' + g.items.length + ' 个实验</span>';
    sec.appendChild(hd);
    g.items.forEach(e => {
      const open = !!LABOPEN[e.id];
      const node = document.createElement('div');
      node.className = 'node';
      const note = (e.notes && e.notes.length) ? e.notes[0] : '';
      let h = '<div class="node-hd"><span class="idx">' + e.id + '</span>'
        + '<span class="nm">' + e.algo + '</span>'
        + '<span class="id">' + labShort(note, 90) + '</span>'
        + '<span class="caret">' + (open ? '▾' : '▸') + '</span></div>';
      h += '<div class="expwrap"><div class="expchips">'
        + '<span class="expchip"><code>' + e.file + '</code></span>'
        + '<span class="expchip">总耗时 ' + e.total_time.toFixed(2) + ' s</span>'
        + '</div></div>';
      if(open) h += labDetail(e);
      node.innerHTML = h;
      node.querySelector('.node-hd').onclick = () => {
        LABOPEN[e.id] = !LABOPEN[e.id]; labSave();
        renderLab(labFilter, document.getElementById('labq').value.trim());
      };
      sec.appendChild(node);
    });
    box.appendChild(sec);
  });
  if(!box.children.length){
    box.innerHTML = '<div class="lead">没有匹配的实验，试试「降维」「噪声」「准确率」等关键词。</div>';
  }
}
let labFilter = 'all';
(function initLab(){
  document.querySelectorAll('.chip[data-lf]').forEach(b => {
    b.onclick = () => {
      document.querySelectorAll('.chip[data-lf]').forEach(x => x.classList.remove('on'));
      b.classList.add('on');
      labFilter = b.dataset.lf;
      renderLab(labFilter, document.getElementById('labq').value.trim());
    };
  });
  const qi = document.getElementById('labq');
  if(qi) qi.oninput = ev => renderLab(labFilter, ev.target.value.trim());
  renderLab('all', '');
})();
"""


def inject(html_path, exps):
    with open(html_path, encoding="utf-8") as fh:
        src = fh.read()
    subs = [
        ("NAV_LAB", nav_html(exps)),
        ("PANE_LAB", pane_html(exps)),
        ("DATA_LAB", data_js(exps) + LAB_JS),
    ]
    # 脚本内用 JS 注释作锚点（HTML 注释 <!-- 在 <script> 里会被当单行注释吞掉整行）
    marks = {"NAV_LAB": ("<!--%s:BEGIN-->", "<!--%s:END-->"),
             "PANE_LAB": ("<!--%s:BEGIN-->", "<!--%s:END-->"),
             "DATA_LAB": ("/*%s:BEGIN*/", "/*%s:END*/")}
    for tag, body in subs:
        b, e = marks[tag]
        pat = re.compile(re.escape(b % tag) + r".*?" + re.escape(e % tag), re.S)
        if not pat.search(src):
            raise SystemExit("未找到锚点 %s，请检查网页文件" % tag)
        src = pat.sub(lambda m: (b % tag) + "\n" + body + "\n" + (e % tag),
                      src, count=1)
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(src)
    return len(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=DEFAULT_HTML)
    a = ap.parse_args()
    exps = load_experiments()
    if not exps:
        raise SystemExit("没有实验结果，请先运行 run_all.py")
    n = inject(a.html, exps)
    print("已注入 %d 个实验（%d 张图表）到：%s" %
          (len(exps), sum(len(e["figs"]) for e in exps), a.html))
    print("文件大小：%.1f KB" % (n / 1024))


if __name__ == "__main__":
    main()
