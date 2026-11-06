#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 outputs/ 下的三份独立 HTML 合并成一个自包含单文件「AI课程平台.html」。

=========================== 重跑约定（必读） ===========================
本脚本是**唯一写入口**：三个源文件一个字节都不改，产物是派生快照。
改了下面任一源文件后，先跑对应的注入脚本，再重跑本脚本：

    tools/demos_b*.js         -> python tools/inject_demos.py b2..b6
    tools/difficulty.py       -> python tools/build_pick_page.py
    experiments/results/      -> python experiments/tools/build_lab_page.py
    outputs/项目任务书.md      -> python experiments/tools/md2html.py ... 项目任务书.html
    outputs/难度系数表.md      -> python tools/difficulty.py --emit-md

然后：python tools/build_merged.py
产物 outputs/AI课程平台.html **永不手工编辑**。
========================================================================

用法:
    python tools/build_merged.py            # 构建
    python tools/build_merged.py --check    # 只做冲突自检，不写盘
"""

import io
import os
import re
import sys
import shutil
import time
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "AI课程平台.html")

SRC_ALGO = os.path.join(ROOT, "outputs", "AI算法大纲-交互版.html")
SRC_PICK = os.path.join(ROOT, "outputs", "选题系统.html")
MD_BOOK = os.path.join(ROOT, "outputs", "项目任务书.md")
MD_COEF = os.path.join(ROOT, "outputs", "难度系数表.md")
MD2HTML = os.path.join(ROOT, "experiments", "tools", "md2html.py")

TITLE = "人工智能技术及应用 · 课程综合平台"

# ---------------------------------------------------------------- 读取工具

def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def match_brace(s, i):
    """s[i] == '{'，返回匹配的 '}' 下标。"""
    depth = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("括号不配对")


def split_doc(html):
    """拆出 style 内容、body 内层 markup（已剔除 script）、以及所有内联 script 源码。"""
    m = re.search(r"<style[^>]*>(.*?)</style>", html, re.S)
    css = m.group(1) if m else ""
    mb = re.search(r"<body[^>]*>", html)
    bs = mb.end() if mb else 0
    be = html.rfind("</body>")
    inner = html[bs:be] if be > bs else html[bs:]

    scripts, rest, pos = [], [], 0
    # 注意排除 SKELETON 教学示例里出现的 "<script src>" 文本（无等号也要排除）
    for m in re.finditer(r"<script(?![^>]*\bsrc\b)[^>]*>", inner):
        st = m.end()
        en = inner.find("</script>", st)
        if en < 0:
            break
        scripts.append(inner[st:en])
        rest.append(inner[pos:m.start()])
        pos = en + len("</script>")
    rest.append(inner[pos:])
    return css, "".join(rest), scripts


def split_top(sel):
    """按顶层逗号切分选择器，保护 () [] 与引号内的逗号。"""
    out, buf, depth, q = [], [], 0, None
    for c in sel:
        if q:
            buf.append(c)
            if c == q:
                q = None
            continue
        if c in "\"'":
            q = c
            buf.append(c)
        elif c in "([":
            depth += 1
            buf.append(c)
        elif c in ")]":
            depth -= 1
            buf.append(c)
        elif c == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    out.append("".join(buf))
    return [x.strip() for x in out if x.strip()]


def iter_rules(css):
    """产出 (preamble, body) —— body 是花括号内原文（@media 需递归）。"""
    i, n = 0, len(css)
    while i < n:
        while i < n and css[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        brace = css.find("{", i)
        semi = css.find(";", i)
        if css[i] == "@":
            if brace < 0:
                break
            if semi >= 0 and semi < brace:
                i = semi + 1
                continue
        if brace < 0:
            break
        pre = css[i:brace].strip()
        k = match_brace(css, brace)
        yield pre, css[brace + 1:k]
        i = k + 1


# ---------------------------------------------------------------- CSS 作用域化

def collect_vars(body, bag, origin):
    for decl in body.split(";"):
        if ":" not in decl:
            continue
        k, v = decl.split(":", 1)
        k, v = k.strip(), v.strip()
        if not k.startswith("--"):
            continue
        if k in bag and bag[k][0] != v:
            print("  [WARN] CSS 变量同名异值 %s: %r vs %r (%s)" % (k, bag[k][0], v, origin))
        else:
            bag[k] = (v, origin)


def scope_sel(sel, prefix):
    parts = []
    for p in split_top(sel):
        if p == "*":
            parts.append(prefix + " *")
        elif p in ("body", "html"):
            parts.append(prefix)
        elif p.startswith(":root"):
            parts.append(p)          # 已在上一层剥离
        else:
            parts.append(prefix + " " + p)
    return ", ".join(parts)


def scope_css(css, prefix, bag=None, origin=""):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out = []
    for pre, body in iter_rules(css):
        if pre.startswith("@media") and "print" in pre:
            continue                                  # 打印规则统一由 CHROME 接管
        if pre.startswith("@media") or pre.startswith("@supports"):
            inner = scope_css(body, prefix, bag, origin)
            if inner.strip():
                out.append(pre + "{\n" + inner + "\n}")
            continue
        if pre.replace(" ", "").startswith(":root"):
            if bag is not None:
                collect_vars(body, bag, origin)
            continue
        sel = scope_sel(pre, prefix)
        if sel:
            out.append(sel + "{" + body.strip() + "}")
    return "\n".join(out)


# ---------------------------------------------------------------- 文档页处理

def load_md2html():
    spec = importlib.util.spec_from_file_location("md2html_local", MD2HTML)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SEP_RE = re.compile(r"^\s*[-: |]+\s*$")
# 只需以 | 开头：表格行首尾都带 |，决策树 ASCII 图只开头带 |
TOK_RE = re.compile(r"<p>\|(.*?)</p>")


def _render_pipe_group(grp):
    """连续「裸管道符段落」还原：有分隔行 -> <table>，否则 -> <pre><code>（ASCII 树）。"""
    if len(grp) >= 2 and SEP_RE.match(grp[1]):
        def cells(s):
            cs = [c.strip() for c in s.split("|")]
            if cs and cs[0] == "":
                cs = cs[1:]
            if cs and cs[-1] == "":
                cs = cs[:-1]
            return cs
        t = ["<table><thead><tr>" + "".join("<th>%s</th>" % c for c in cells(grp[0]))
             + "</tr></thead><tbody>"]
        for row in grp[2:]:
            if SEP_RE.match(row):
                continue
            t.append("<tr>" + "".join("<td>%s</td>" % c for c in cells(row)) + "</tr>")
        t.append("</tbody></table>")
        return "\n".join(t)
    return "<pre><code>" + "\n".join("|" + g + "|" for g in grp) + "</code></pre>"


def fix_doc_html(h):
    """修复 md2html 不支持「块引用内表格」导致的裸管道符段落。

    注意：md2html 会把整段块引用压成一行（多个 <p> 相邻），
    因此不能按行匹配，必须按 token 匹配并把相邻 token 归组。
    """
    toks = list(TOK_RE.finditer(h))
    if not toks:
        return h
    groups, cur = [], [toks[0]]
    for t in toks[1:]:
        if h[cur[-1].end():t.start()].strip() == "":
            cur.append(t)
        else:
            groups.append(cur)
            cur = [t]
    groups.append(cur)

    out, pos = [], 0
    for g in groups:
        out.append(h[pos:g[0].start()])
        out.append(_render_pipe_group([m.group(1) for m in g]))
        pos = g[-1].end()
    out.append(h[pos:])
    return "".join(out)


def doc_fragment(md2, mdpath, title):
    doc = md2.convert(read(mdpath), title)
    css, markup, scripts = split_doc(doc)
    markup = re.sub(r"<footer>.*?</footer>", "", markup, flags=re.S)
    markup = fix_doc_html(markup)
    # 文档内指向别的文件的行内代码 -> 站内跳转
    markup = markup.replace(
        "<code>选题系统.html</code>",
        '<a href="#" data-goto="pick" style="color:var(--accent)">「分组选题」模块</a>')
    markup = markup.replace(
        "<code>outputs/AI算法大纲-交互版.html</code>",
        '<a href="#" data-goto="algo" style="color:var(--accent)">「算法大纲」模块</a>')
    markup = re.sub(
        r"<code>AI算法大纲-交互版\.html</code> 第 \d+ 行",
        '「算法大纲」模块该算法的交互演示', markup)
    return css, markup, scripts


# ---------------------------------------------------------------- JS 变换

TAB_DELEGATE_OLD = """(function(){
  const nav = document.querySelector('nav');
  if(!nav) return;
  nav.addEventListener('click', function(ev){
    const b = ev.target.closest ? ev.target.closest('button[data-p]') : null;
    if(!b) return;
    const pane = document.getElementById('p-' + b.getAttribute('data-p'));
    if(!pane) return;
    Array.prototype.forEach.call(nav.querySelectorAll('button[data-p]'),
      function(x){ x.classList.remove('on'); });
    b.classList.add('on');
    Array.prototype.forEach.call(document.querySelectorAll('.pane'),
      function(p){ p.classList.remove('on'); });
    pane.classList.add('on');
    window.scrollTo({top: 0, behavior: 'smooth'});
  });
})();"""

TAB_DELEGATE_NEW = """(function(){
  const root = document.getElementById('pg-algo');
  if(!root) return;
  const nav = root.querySelector('header nav');
  if(!nav) return;
  nav.addEventListener('click', function(ev){
    const b = ev.target.closest ? ev.target.closest('button[data-p]') : null;
    if(!b) return;
    const pane = root.querySelector('#p-' + b.getAttribute('data-p'));
    if(!pane) return;
    Array.prototype.forEach.call(nav.querySelectorAll('button[data-p]'),
      function(x){ x.classList.remove('on'); });
    b.classList.add('on');
    Array.prototype.forEach.call(root.querySelectorAll('.pane'),
      function(p){ p.classList.remove('on'); });
    pane.classList.add('on');
    window.scrollTo({top: 0, behavior: 'smooth'});
  });
})();"""

KEYDOWN_OLD = """document.addEventListener('keydown', e => {
  if(e.key === 'Escape') closeForm();
});"""

KEYDOWN_NEW = """document.addEventListener('keydown', e => {
  const host = document.getElementById('pg-pick');
  if(!host || host.hidden) return;          /* 仅在「分组选题」模块激活时响应 */
  if(e.key === 'Escape') closeForm();
});"""

# 选题系统与主页面硬冲突的 3 个 id —— 只改这三个，其余保持原样
# （全量改名会误伤 SKELETON 教学示例里的 getElementById('app')）
PICK_ID_RULES = [
    (re.compile(r'(?<=\sid=")q(?=")'), "pk-q"),
    (re.compile(r'(?<=\sid=")mask(?=")'), "pk-mask"),
    (re.compile(r'(?<=\sid=")modal(?=")'), "pk-modal"),
    (re.compile(r"getElementById\('q'\)"), "getElementById('pk-q')"),
    (re.compile(r"getElementById\('mask'\)"), "getElementById('pk-mask')"),
    (re.compile(r"getElementById\('modal'\)"), "getElementById('pk-modal')"),
    (re.compile(r"e\.target\.id === 'mask'"), "e.target.id === 'pk-mask'"),
]


def apply_pick_ids(s):
    for rx, rep in PICK_ID_RULES:
        s = rx.sub(rep, s)
    return s


def wrap_iife(src):
    """整体包进 IIFE 做符号隔离。不加 'use strict'（会改变块级函数提升语义）。"""
    return "(function(){\n" + src.rstrip() + "\n})();"


def strip_node_ids(s):
    return re.sub(r'\s*data-page-node-id="[A-Za-z0-9]{22}"', "", s)


# ---------------------------------------------------------------- 骨架

CHROME_CSS = """
html{background:#faf9f5}
body{margin:0;padding:0;font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;
     font-size:14px;line-height:1.65;color:#3c3b37;background:#faf9f5}
:root{--tnav-h:52px}
#topnav{position:sticky;top:0;z-index:60;display:flex;align-items:center;gap:4px;
        height:var(--tnav-h);padding:0 14px;overflow-x:auto;
        background:#ffffff;border-bottom:1px solid #d3d1c7;box-shadow:0 1px 3px rgba(0,0,0,.05)}
#topnav button{flex:0 0 auto;height:32px;padding:0 14px;margin:0;cursor:pointer;
        font:14px "Microsoft YaHei",system-ui;color:#5f5e5a;background:transparent;
        border:1px solid transparent;border-radius:7px;transition:.15s}
#topnav button:hover{background:#f0efe9;color:#2c2b28}
#topnav button.on{background:#185fa5;color:#fff;border-color:#185fa5}
.pg[hidden]{display:none!important}
@media print{
  #topnav{display:none!important}
  .pg[hidden]{display:none!important}
  #pg-algo .pane{display:block!important}
}
"""

# sticky 偏移必须放在各模块 scoped CSS 之后，否则会被原规则覆盖
STICKY_FIX = """
#pg-algo header{top:var(--tnav-h)}
#pg-algo .pickbar{top:calc(var(--tnav-h) + 64px)}
"""

CHROME_JS = """
(function(){
  var MODS = ['algo','pick','book','coef'];
  function show(m, fromHash){
    if(MODS.indexOf(m) < 0) m = 'algo';
    MODS.forEach(function(x){
      var el = document.getElementById('pg-' + x);
      if(el) el.hidden = (x !== m);
    });
    Array.prototype.forEach.call(
      document.querySelectorAll('#topnav button[data-m]'),
      function(b){ b.classList.toggle('on', b.getAttribute('data-m') === m); });
    window.scrollTo(0, 0);
    if(!fromHash){ try{ history.replaceState(null, '', '#' + m); }catch(e){} }
  }
  var nav = document.getElementById('topnav');
  if(nav){
    nav.addEventListener('click', function(ev){
      var b = ev.target.closest ? ev.target.closest('button[data-m]') : null;
      if(b) show(b.getAttribute('data-m'));
    });
  }
  document.addEventListener('click', function(ev){
    var a = ev.target.closest ? ev.target.closest('[data-goto]') : null;
    if(!a) return;
    ev.preventDefault();
    show(a.getAttribute('data-goto'));
  });
  window.addEventListener('hashchange', function(){
    show((location.hash || '').replace('#',''), true);
  });
  show((location.hash || '').replace('#','') || 'algo', true);
})();
"""

NAV_HTML = """<nav id="topnav">
  <button data-m="algo" class="on">算法大纲</button>
  <button data-m="pick">分组选题</button>
  <button data-m="book">项目任务书</button>
  <button data-m="coef">难度系数表</button>
</nav>
"""


# ---------------------------------------------------------------- 主流程

def build(check_only=False):
    print("读取源文件 ...")
    algo_html = read(SRC_ALGO)
    pick_html = read(SRC_PICK)
    md2 = load_md2html()

    algo_css, algo_mk, algo_js = split_doc(algo_html)
    pick_css, pick_mk, pick_js = split_doc(pick_html)
    doc_css, book_mk, _ = doc_fragment(md2, MD_BOOK, "项目任务书")
    _, coef_mk, _ = doc_fragment(md2, MD_COEF, "难度系数表")

    assert len(algo_js) == 1, "主页面应有 1 个 script，实际 %d" % len(algo_js)
    assert len(pick_js) == 1, "选题系统应有 1 个 script，实际 %d" % len(pick_js)

    # ---- 主页面 markup
    algo_mk = strip_node_ids(algo_mk)
    n_btn = algo_mk.replace(
        """<button onclick="location.href='选题系统.html'" title="综合项目分组选题与难度系数登记">分组选题 →</button>""",
        """<button data-goto="pick" title="综合项目分组选题与难度系数登记">分组选题 →</button>""")
    assert n_btn != algo_mk, "主页面「分组选题 →」按钮原文未命中"
    algo_mk = n_btn
    n_a = algo_mk.count('<a href="选题系统.html"') + algo_mk.count('<a href="项目任务书.html"')
    algo_mk = algo_mk.replace('<a href="选题系统.html"', '<a href="#" data-goto="pick"')
    algo_mk = algo_mk.replace('<a href="项目任务书.html"', '<a href="#" data-goto="book"')
    assert n_a == 2, "主页面 #p-plan 内 2 个跨文件链接未命中，实际 %d" % n_a

    # ---- 主页面 script
    ajs = algo_js[0]
    n_tab = ajs.replace(TAB_DELEGATE_OLD, TAB_DELEGATE_NEW)
    assert n_tab != ajs, "主页面二级 tab 委托原文未命中"
    ajs = strip_node_ids(n_tab)

    # ---- 选题系统
    pick_mk = apply_pick_ids(strip_node_ids(pick_mk))
    pjs = apply_pick_ids(strip_node_ids(pick_js[0]))
    n_kd = pjs.replace(KEYDOWN_OLD, KEYDOWN_NEW)
    assert n_kd != pjs, "选题系统 keydown 原文未命中"
    pjs = n_kd

    # ---- 冲突自检（构建期 assert）
    assert ajs.count("document.querySelector('nav')") == 0, "主页面仍有全局 nav 查询"
    assert pjs.count("getElementById('app')") == 1, "SKELETON 教学示例被误改名"
    # 实测值（markup + script）：q=2、mask=6、modal=4
    for tok, cnt in (("pk-q", 2), ("pk-mask", 6), ("pk-modal", 4)):
        assert pick_mk.count(tok) + pjs.count(tok) == cnt, \
            "id 改名计数异常 %s" % tok

    # ---- CSS
    print("合并 CSS ...")
    bag = {}
    out_algo = scope_css(algo_css, "#pg-algo", bag, "主页面")
    out_pick = scope_css(pick_css, "#pg-pick", bag, "选题系统")
    out_doc = scope_css(doc_css, "#pg-book, #pg-coef", bag, "md2html").replace(
        "#pg-book, #pg-coef ", "#pg-book , #pg-coef ")
    # 文档页选择器形如 "#pg-book, #pg-coef h2"，需展开成两条
    out_doc = re.sub(
        r"#pg-book , #pg-coef (.+?)\{",
        lambda m: "#pg-book %s,#pg-coef %s{" % (m.group(1), m.group(1)),
        out_doc)
    root_block = ":root{" + "".join("%s:%s;" % (k, v[0]) for k, v in sorted(bag.items())) + "}"
    print("  :root 变量 %d 个" % len(bag))

    css_all = CHROME_CSS + "\n" + root_block + "\n" + \
        out_algo + "\n" + out_pick + "\n" + out_doc + "\n" + STICKY_FIX

    # ---- 拼装
    print("拼装 ...")
    parts = [
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n",
        '<meta charset="utf-8">\n',
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n',
        "<title>%s</title>\n" % TITLE,
        "<style>\n" + css_all + "\n</style>\n",
        "</head>\n<body>\n",
        NAV_HTML,
        '<section class="pg" id="pg-algo">\n' + algo_mk + "\n</section>\n",
        '<section class="pg" id="pg-pick" hidden>\n' + pick_mk + "\n</section>\n",
        '<section class="pg" id="pg-book" hidden>\n<div class="doc">' + book_mk + "</div>\n</section>\n",
        '<section class="pg" id="pg-coef" hidden>\n<div class="doc">' + coef_mk + "</div>\n</section>\n",
        "<script>" + CHROME_JS + "</script>\n",
        "<script>" + wrap_iife(ajs) + "</script>\n",
        "<script>" + wrap_iife(pjs) + "</script>\n",
        "</body>\n</html>\n",
    ]
    doc = "".join(parts)

    # ---- 结构自检
    assert doc.count("</script>") == 3, "</script> 数量异常: %d" % doc.count("</script>")
    assert doc.count("<style>") == 1 and doc.count("</style>") == 1
    assert "data-page-node-id" not in doc, "噪声属性未剥离干净"
    for frag in (algo_mk, pick_mk, book_mk, coef_mk):
        assert "</script" not in frag, "body 片段内出现 </script"
    assert "<p>|" not in doc, "仍有未还原的裸管道符段落: %d" % doc.count("<p>|")

    print("  产物 %d 字符（%.0f KB）" % (len(doc), len(doc.encode("utf-8")) / 1024.0))

    if check_only:
        print("自检通过（--check，未写盘）")
        return doc

    if os.path.exists(OUT):
        shutil.copy2(OUT, OUT + ".bak")
        print("  旧产物已备份 -> %s.bak" % os.path.basename(OUT))
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(doc)
    print("已写出: %s" % OUT)
    return doc


def backup_src():
    ts = time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(os.environ.get("TEMP", "."), "ai_merge_" + ts, "src")
    os.makedirs(d, exist_ok=True)
    for p in (SRC_ALGO, SRC_PICK, MD_BOOK, MD_COEF):
        shutil.copy2(p, d)
    print("源文件已备份 -> %s" % d)


if __name__ == "__main__":
    if "--check" in sys.argv:
        build(check_only=True)
    else:
        backup_src()
        build()
