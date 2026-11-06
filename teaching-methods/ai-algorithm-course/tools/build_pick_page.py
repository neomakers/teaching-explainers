# -*- coding: utf-8 -*-
"""把 tools/difficulty.py 的选题数据幂等注入 outputs/选题系统.html。

difficulty.py 是唯一事实源，改完难度后跑一次即可同步：
    python tools/build_pick_page.py
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import difficulty  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "outputs", "选题系统.html")
BEGIN = "/*TOPICS:BEGIN*/"
END = "/*TOPICS:END*/"


def main():
    if not os.path.isfile(PAGE):
        print("找不到 %s" % PAGE)
        return 1
    topics = difficulty.decorate()
    seen = set()
    for t in topics:
        t["baseline"] = difficulty._notes_of(t["id"])
        seen.add(t["id"])
    if len(seen) != 30:
        print("题目数应为 30，实际 %d" % len(seen))
        return 1

    # 用紧凑 JSON，避免 HTML 里塞缩进空白导致体积虚高
    import json
    payload = json.dumps(topics, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")  # 防止 </script> 提前闭合

    with io.open(PAGE, encoding="utf-8") as f:
        html = f.read()
    i = html.find(BEGIN)
    j = html.find(END)
    if i < 0 or j < 0:
        print("找不到 TOPICS 锚点，HTML 可能被改坏了")
        return 1
    new = html[:i] + BEGIN + payload + html[j:]
    with io.open(PAGE, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)

    print("已注入 %d 个选题（%d 字符）→ %s" % (len(topics), len(payload), PAGE))
    print("文件大小：%.1f KB" % (os.path.getsize(PAGE) / 1024.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
