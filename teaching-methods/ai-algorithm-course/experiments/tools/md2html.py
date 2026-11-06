"""轻量 Markdown → HTML 转换器（零依赖，仅支持指导书用到的语法）。

支持：标题 #/##/###、无序与有序列表、表格、代码块、引用、分隔线、
      **加粗**、`行内代码`（不支持图片与链接，够用即可）。

用法：
    python md2html.py <input.md> <output.html> [--title 标题]
"""
import argparse
import html
import os
import re

CSS = """
:root{--bg:#f6f7f9;--fg:#2c2c2a;--fg2:#5f5e5a;--line:#e3e1da;--accent:#185fa5;
      --code-bg:#2c2c2a;--code-fg:#e8e8e4}
body{background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,
     "Microsoft YaHei",sans-serif;font-size:15px;line-height:1.75;margin:0;padding:0 20px 80px}
.wrap{max-width:920px;margin:0 auto}
h1{font-size:24px;font-weight:500;padding:28px 0 6px;border-bottom:2px solid var(--accent)}
h2{font-size:19px;font-weight:500;margin:30px 0 12px;padding:8px 12px;background:#e6f1fb;
   border-radius:8px}
h3{font-size:16px;font-weight:500;margin:22px 0 8px;color:#0c447c}
h4{font-size:15px;font-weight:500;margin:16px 0 6px}
p{margin:8px 0}
ul,ol{padding-left:24px;color:var(--fg2)}
li{margin:4px 0}
table{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;overflow:hidden;
      font-size:13.5px;margin:10px 0 18px}
th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line)}
th{background:#f1efe8;font-weight:500}
pre{background:var(--code-bg);color:var(--code-fg);padding:14px 16px;border-radius:10px;
    overflow:auto;font-family:ui-monospace,Consolas,monospace;font-size:13px;line-height:1.6}
code{background:#f1efe8;padding:1px 5px;border-radius:4px;font-size:13px}
pre code{background:none;padding:0;color:inherit}
blockquote{margin:12px 0;padding:10px 16px;background:#faeeda;border-radius:8px;
           color:#854f0b;border-left:4px solid #ba7517}
blockquote p{margin:2px 0}
hr{border:none;border-top:1px solid var(--line);margin:26px 0}
.toc{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 20px}
footer{text-align:center;color:#888780;font-size:12px;margin-top:40px}
@media print{body{background:#fff;font-size:11pt}h2{background:#f1efe8}
  pre{background:#f4f4f2;color:#222}}
"""


def inline(t):
    """行内格式：**粗体** 与 `代码`，先转义再还原。"""
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t


def is_table_sep(line):
    return bool(re.match(r"^\|[\s:|-]+\|$", line.strip()))


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(md_text, title):
    lines = md_text.split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()

        if s.startswith("```"):                       # 代码块
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i], quote=False))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue

        if s in ("---", "***"):
            out.append("<hr>")
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)           # 标题
        if m:
            lv = len(m.group(1))
            out.append("<h%d>%s</h%d>" % (lv, inline(m.group(2)), lv))
            i += 1
            continue

        if s.startswith("|") and i + 1 < n and is_table_sep(lines[i + 1]):
            head = split_row(lines[i])
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            out.append("<table><tr>" +
                       "".join("<th>%s</th>" % inline(c) for c in head) +
                       "</tr>")
            for r in rows:
                cells = "".join("<td>%s</td>" % inline(c) for c in r)
                out.append("<tr>%s</tr>" % cells)
            out.append("</table>")
            continue

        if s.startswith(">"):                          # 引用
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>" +
                       "".join("<p>%s</p>" % inline(b) for b in buf) +
                       "</blockquote>")
            continue

        if re.match(r"^[-*]\s+", s):                   # 无序列表
            buf = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                buf.append(lines[i].strip()[2:])
                i += 1
            out.append("<ul>" +
                       "".join("<li>%s</li>" % inline(b) for b in buf) + "</ul>")
            continue

        if re.match(r"^\d+\.\s+", s):                  # 有序列表
            buf = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                buf.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            out.append("<ol>" +
                       "".join("<li>%s</li>" % inline(b) for b in buf) + "</ol>")
            continue

        if not s:
            i += 1
            continue

        para = [s]
        i += 1
        while i < n:                                     # 连续非空非块首行合并为段落
            nx = lines[i].strip()
            if (not nx or nx.startswith("#") or nx.startswith("|")
                    or nx.startswith(">") or nx.startswith("```")
                    or re.match(r"^[-*]\s+", nx) or re.match(r"^\d+\.\s+", nx)
                    or nx in ("---", "***")):
                break
            para.append(nx)
            i += 1
        out.append("<p>%s</p>" % inline(" ".join(para)))

    return ("<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>%s</title><style>%s</style></head><body><div class=\"wrap\">%s"
            "<footer>由 experiments/tools/md2html.py 自动生成</footer>"
            "</div></body></html>"
            % (html.escape(title), CSS, "\n".join(out)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    with open(a.src, encoding="utf-8") as f:
        md = f.read()
    title = a.title or os.path.splitext(os.path.basename(a.src))[0]
    html_text = convert(md, title)
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    with open(a.dst, "w", encoding="utf-8") as f:
        f.write(html_text)
    print("生成：%s（%d 字符）" % (a.dst, len(html_text)))


if __name__ == "__main__":
    main()
