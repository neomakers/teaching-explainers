# 《人工智能技术及应用》课程资源包

智能制造 / 机械类本科课程《人工智能技术及应用》的完整配套资产：交互式网页、课程文档、30 个可复现 Python 实验、构建与验收工具。

- **主入口**：[`../ai-algorithm-course.html`](../ai-algorithm-course.html) — 自包含单文件课程平台（4 模块 + 大纲页原有 7 标签），零外部依赖、可离线打开。
- **本包导航**：[`index.html`](index.html) — 在 GitHub Pages 上浏览全部文件与预览图。
- **一次性下载**：[`ai-algorithm-course.zip`](https://neomakers.github.io/teaching-explainers/teaching-methods/ai-algorithm-course/ai-algorithm-course.zip)（ZIP 内不含 ZIP 自身，已解压时无需再下载）。

## 文件对照表

| 原始文件 | 仓库内文件 |
|---|---|
| `outputs/AI课程平台.html` | `teaching-methods/ai-algorithm-course.html` |
| `outputs/AI算法大纲-交互版.html` | `outline-interactive.html` |
| `outputs/选题系统.html` | `topic-picker.html` |
| `outputs/项目任务书.html` / `.md` | `project-brief.html` / `docs/project-brief.md` |
| `outputs/学生实验指导书.html` / `.md` | `lab-guide.html` / `docs/lab-guide.md` |
| `outputs/人工智能技术及应用-算法大纲.md` | `docs/course-outline.md` |
| `outputs/难度系数表.md` | `docs/difficulty-table.md` |
| `experiments/results/实验报告.html` | `experiment-report.html` |
| `experiments/`（不含缓存与日志） | `experiments/` |
| `tools/` | `tools/` |
| `outputs/_shots/` | `previews/` |

## 运行方式

1. 网页：双击任意 `.html`，或用浏览器打开 Pages 链接；不需要服务器、不需要联网。
2. 实验：`pip install numpy scikit-learn matplotlib`，然后 `python experiments/run_all.py`；单个实验可直接 `python experiments/exp_l1_01_svd.py`。全部实验固定 `SEED=42`，结果写入 `experiments/results/`。
3. 重新构建（需在原课程工作区）：`python tools/build_merged.py`，再用 `node tools/check_syntax.js` 与 `tools/verify_merged.js` 验收。构建脚本内保留原始中文路径，本仓库是归档快照。

## 内容与边界

- 算法覆盖 L1 数学基础至 L8 决策与生成选讲，共 8 层 30 个算法，其中 12 个为重点。
- 网页演示为**浏览器内教学用简化模型**，用于说明机理与参数敏感性；数值真值以 `experiments/` 下 seed=42 的 Python 结果为准，不作工程精度依据。
- 评价体系：`得分 = min(100, 原始分 × 难度系数 K)`，K 由数学深度 / 编程复杂度 / 可视化难度 / 概念抽象度 / 资源支撑度五维加权得出，区间 1.05–1.50，详见 `docs/difficulty-table.md`。
- 归档排除：`__pycache__/`、`*.pyc`、`*.log`。

归档日期：2026-11-06。
