# 教学原理讲解页 · Teaching Explainers

面向智能制造、机器人、计算机视觉与 AI 教学的**交互式原理讲解页合集**。

每篇是一个自包含单文件 HTML：零外部依赖、可离线打开、浅色主题、含可操作演示。按主题分目录归档，便于长期分类维护。

## 内容索引

| 分类 | 主题 | 文件 | 核心内容 | 日期 |
|---|---|---|---|---|
| 计算机视觉 | ArUco 码基本原理 | [aruco-marker-principles.html](computer-vision/aruco-marker-principles.html) | 结构解剖、汉明纠错字典、8 步检测流水线、单应归一化、PnP 位姿解算、OpenCV 代码与误差清单 | 2026-11-06 |

## 目录约定

| 目录 | 收纳范围 |
|---|---|
| `computer-vision/` | 相机标定、图像特征、视觉标记、位姿估计 |
| `robotics/` | 运动学、坐标变换、传感器、控制基础 |
| `additive-manufacturing/` | 增材制造工艺、切片、材料与装备 |
| `ai-deployment/` | 模型部署、推理加速、端侧 AI |

新增讲解页时：放入对应目录 → 在上表追加一行 → 在 `index.html` 首页补一张卡片。

## 在线浏览

GitHub Pages 首页：<https://neomakers.github.io/teaching-explainers/>

部署方式：Pages 由 `main` 分支根目录直接发布，无需 GitHub Actions；每次 push 到 `main` 自动更新。若 Pages 未启用：仓库 **Settings → Pages → Build and deployment → Source** 选择 **Deploy from a branch**，分支 `main`、目录 `/ (root)`。也可以 clone 后直接双击 HTML 本地打开。

## 说明

- 讲解页中的演示字典与可视化为教学用简化模型，不是 OpenCV 预定义字典；也不模拟镜头畸变、噪声与抗锯齿。演示用于说明机理，不作检测率或测量精度依据。
- 字典规格、参数默认值等数据以 [OpenCV 4.x ArUco 官方文档](https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html) 为准，各页底部附有完整参考列表。
