# 教学原理讲解页 · Teaching Explainers

面向智能制造、机器人、计算机视觉、具身智能与 AI 教学的**交互式原理讲解页 + 灵感速记合集**。

- **讲解页**：自包含单文件 HTML，零外部依赖、可离线打开、浅色主题、含可操作演示。
- **灵感速记**：轻量 idea 与案例，按同一分类体系打标，收在 `ideas/INDEX.md`。
- 按主题分目录归档，便于长期分类维护；成熟主题才升级为讲解页。

## 内容索引

| 分类 | 类型 | 标题 | 文件 | 要点 | 备注 | 日期 |
|---|---|---|---|---|---|---|
| 计算机视觉 | 讲解页 | ArUco 码基本原理 | [aruco-marker-principles.html](computer-vision/aruco-marker-principles.html) | 结构解剖、汉明纠错字典、8 步检测流水线、单应归一化、PnP 位姿解算、OpenCV 代码与误差清单 | 依据 OpenCV 4.x 官方文档 | 2026-11-06 |

最近速记：**0 条**（见 [ideas/INDEX.md](ideas/INDEX.md)）。

## 分类体系

主分类唯一；交叉主题归入主分类，在索引「备注」列标注交叉标签。

| 目录 | 中文名 | 收纳范围 |
|---|---|---|
| `computer-vision/` | 计算机视觉 | 相机标定、图像特征、视觉标记、位姿估计、机器视觉检测 |
| `robotics/` | 机器人 | 运动学/动力学、坐标变换、机械臂、移动机器人、ROS |
| `additive-manufacturing/` | 增材制造 | 3D 打印工艺、切片、材料、装备、后处理 |
| `ai-deployment/` | AI 部署 | 模型部署、推理加速、端侧 AI、量化、边缘计算 |
| `embodied-ai/` | 具身智能 | VLA、模仿学习、Sim2Real、AI+机器人交叉 |
| `programming-fundamentals/` | 程序设计基础 | C/C++、程序设计基础、数据结构、代码可视化 |
| `control-and-sensing/` | 控制与感知 | 控制理论、传感器、微机原理与接口、PLC、伺服驱动 |
| `teaching-methods/` | 教学法与课程 | 课程设计、实验/项目设计、考核方式、案例化改造、教具 |
| `ideas/` | 灵感速记 | 未成形 idea 与案例，按上述八类分节打标 |

判定原则：**按核心知识点归类，不按来源或载体**。教学组织方式（怎么设计实验、怎么考核）优先归 `teaching-methods/`；无法归类归 `teaching-methods/` 并备注「待细分」；不为单条内容新建主题目录。

## 新增内容约定

1. 文件名一律 ASCII 小写 kebab-case（如 `pid-frequency-intuition.html`），中文标题写在页面与索引里。
2. 讲解页放入对应分类目录 → 在「内容索引」追加一行 → **同步**在 `index.html` 首页补一张卡片。
3. 轻量 idea 在 `ideas/INDEX.md` 对应分节追加一行，并更新本页「最近速记」条数。
4. 提交推送后 Pages 自动更新，通常 1–2 分钟生效。

配套的 WorkBuddy skill `teaching-explainers` 会自动完成分类、生成、索引维护与提交推送。

## 在线浏览

GitHub Pages 首页：<https://neomakers.github.io/teaching-explainers/>

部署方式：Pages 由 `main` 分支根目录直接发布，无需 GitHub Actions；每次 push 到 `main` 自动更新。若 Pages 未启用：仓库 **Settings → Pages → Build and deployment → Source** 选择 **Deploy from a branch**，分支 `main`、目录 `/ (root)`。也可以 clone 后直接双击 HTML 本地打开。

## 说明

- 讲解页中的演示字典与可视化为教学用简化模型，不是 OpenCV 预定义字典；也不模拟镜头畸变、噪声与抗锯齿。演示用于说明机理，不作检测率或测量精度依据。
- 字典规格、参数默认值等数据以 [OpenCV 4.x ArUco 官方文档](https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html) 为准，各页底部附有完整参考列表。
