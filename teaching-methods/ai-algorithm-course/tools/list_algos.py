# -*- coding: utf-8 -*-
"""列出大纲里所有算法名，并标注是否已有内嵌交互演示"""
import io, re, os

P = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'AI算法大纲-交互版.html')
s = io.open(os.path.abspath(P), encoding='utf-8').read()

# --- DATA 段（算法条目） ---
i = s.index('const DATA')
j = s.index('\n];', i)
seg = s[i:j]
items = re.findall(r'\{l:"([^"]+)",k:(\d+),n:"([^"]+)"', seg)
layers = [x[0] for x in items]
keys = [x[1] for x in items]
names = [x[2] for x in items]

# --- DEMOS 注册的别名 ---
demo = set()
for m in re.finditer(r"regDemo\(\[([^\]]*)\]", s):
    for a in re.findall(r"'([^']+)'", m.group(1)):
        demo.add(a)

print('算法总数:', len(names))
print('演示别名:', len(demo))
print('-' * 60)
has, miss = [], []
for n, l, k in zip(names, layers, keys):
    (has if n in demo else miss).append((l, n, k))
print('已有演示 %d / %d' % (len(has), len(names)))
for l, n, k in has:
    print('  [OK] %-3s %-22s key=%s' % (l, n, k))
print('-' * 60)
print('缺失演示 %d' % len(miss))
for l, n, k in miss:
    print('  [--] %-3s %-22s key=%s' % (l, n, k))
