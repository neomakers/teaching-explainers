# -*- coding: utf-8 -*-
"""把 tools/demos_b*.js 里的交互演示幂等注入主 HTML。

锚点：在 `/* ---------- 主渲染 ---------- */` 之前插入；
每个片段自带 /* DEMOS_xx:BEGIN */ ... /* DEMOS_xx:END */ 标记，重复运行会先剔除旧片段再写入。
"""
import io, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HTML = os.path.join(ROOT, 'outputs', 'AI算法大纲-交互版.html')
ANCHOR = '/* ---------- 主渲染 ---------- */'

def main(part):
    src = os.path.join(ROOT, 'tools', 'demos_%s.js' % part)
    js = io.open(src, encoding='utf-8').read().strip()
    m = re.search(r'/\* (DEMOS_\w+:BEGIN) \*/', js)
    tag = m.group(1).split(':')[0]          # 例如 DEMOS_B2
    html = io.open(HTML, encoding='utf-8').read()

    # 1) 剔除同名旧片段
    pat = re.compile(r'/\* %s:BEGIN \*/.*?/\* %s:END \*/\n?' % (tag, tag), re.S)
    html, k = pat.subn('', html)
    if k:
        print('已剔除旧片段 %s ×%d' % (tag, k))

    # 2) 插入
    if html.count(ANCHOR) != 1:
        print('锚点数量异常：%d' % html.count(ANCHOR))
        return 1
    html = html.replace(ANCHOR, js + '\n\n' + ANCHOR, 1)
    io.open(HTML, 'w', encoding='utf-8', newline='\n').write(html)

    # 3) 统计覆盖
    names = re.findall(r'\{l:"([^"]+)",k:(\d+),n:"([^"]+)"', html)
    demo = set()
    for mm in re.finditer(r"regDemo\(\[([^\]]*)\]", html):
        for a in re.findall(r"'([^']+)'", mm.group(1)):
            demo.add(a)
    miss = [x[2] for x in names if x[2] not in demo]
    print('注入完成：%s（%d 行 JS）' % (src, js.count('\n') + 1))
    print('演示覆盖：%d / %d 个算法' % (len(names) - len(miss), len(names)))
    print('仍缺少演示：%s' % ('、'.join(miss) if miss else '无'))
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'b2'))
