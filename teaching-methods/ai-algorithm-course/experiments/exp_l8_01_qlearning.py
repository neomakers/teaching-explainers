"""L8-01 强化学习实验（表格 Q-Learning）—— DQN 的替代验证

难以直接实验的原因：DQN 需要 PyTorch + 经验回放 + 目标网络，且训练需要大量交互步数
    （通常 10^5 以上）；gymnasium 也未安装，Atari/连续控制环境不可用。
替代验证方案：自建 5x5 车间 AGV 网格环境（零依赖、完全可复现），训练表格 Q-Learning，
    并与其**精确最优解**（价值迭代，因环境模型已知）对照，从而定量衡量收敛质量。
输入：common.data.GridWorld(size=5, traps={7,11,17}, goal=24)，到达 +10、陷阱 -5、每步 -0.1。
参数：episodes=1500；alpha=0.15；gamma=0.95；eps 0.15→0.02（指数衰减）；seed=42。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Exp, data, env_info                  # noqa: E402
from fromscratch.qlearning import QLearning             # noqa: E402

SEED = 42
EPISODES = 1500


def value_iteration(env, gamma=0.95, theta=1e-8):
    """环境模型已知时的精确最优解（动态规划），作为 Q-Learning 的对照基线。"""
    n, A = env.size ** 2, 4
    V = np.zeros(n)
    env0 = data.GridWorld(size=env.size, slippery=0.0, seed=SEED)
    for _ in range(2000):
        delta = 0.0
        for s in range(n):
            if s == env0.goal or s in env0.traps:
                continue
            best = -np.inf
            for a in range(A):
                env0.state = s
                s2, r, done = env0.step(a)
                best = max(best, r + (0 if done else gamma * V[s2]))
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            break
    return V


def main(seed=SEED):
    exp = Exp(exp_id="L8-01", algo="Q-Learning / DQN（替代验证：表格 Q-Learning）",
              layer="L8 决策与生成",
              purpose="在自建网格环境中训练 Q-Learning，并与价值迭代的最优解对照，验证收敛质量",
              dataset="GridWorld(5x5)：起点 0、终点 24、陷阱 {7,11,17}；到达 +10、陷阱 -5、每步 -0.1",
              baselines=["随机策略", "无学习贪心策略（初始 Q 全 0，退化为任意方向）",
                         "价值迭代（环境模型已知时的最优解）"],
              metrics=["平均回合回报", "最后 100 回合平均回报", "收敛回合数", "耗时"],
              seed=seed, params={"episodes": EPISODES, "alpha": 0.15,
                                "gamma": 0.95, "eps": "0.15→0.02 指数衰减"},
              env=env_info())

    env = data.GridWorld(size=5, slippery=0.0, seed=seed)
    vi = value_iteration(env)
    optimal_return = float(vi[0])
    exp.add(方法="价值迭代（最优基线）", 平均回合回报=optimal_return,
            最后100回合回报=optimal_return, 收敛回合数=0)

    rng = np.random.default_rng(seed)
    rand_returns = []
    for _ in range(300):
        e = data.GridWorld(size=5, slippery=0.0, seed=int(rng.integers(1e6)))
        s, total, done, step = e.reset(), 0.0, False, 0
        while not done and step < 100:
            s, r, done = e.step(int(rng.integers(4)))
            total += (0.95 ** step) * r                # 同口径：折扣回报
            step += 1
        rand_returns.append(total)
    exp.add(方法="随机策略", 平均回合回报=float(np.mean(rand_returns)),
            最后100回合回报=float(np.mean(rand_returns[-100:])), 收敛回合数=0)

    agent = QLearning(n_states=25, n_actions=4, alpha=0.15, gamma=0.95,
                      eps=0.15, seed=seed)
    with exp.timer("Q-Learning 训练 %d 回合" % EPISODES):
        rets = agent.train(env, episodes=EPISODES, max_steps=100)
    mean_r, std_r = agent.evaluate(data.GridWorld(size=5, slippery=0.0,
                                                   seed=seed), episodes=200)
    smooth = np.convolve(rets, np.ones(50) / 50, mode="valid")
    conv = int(np.argmax(smooth > optimal_return - 0.05)) if np.any(
        smooth > optimal_return - 0.05) else -1
    exp.add(方法="Q-Learning(训练)", 平均回合回报=float(rets.mean()),
            最后100回合回报=float(rets[-100:].mean()), 收敛回合数=conv)
    exp.add(方法="Q-Learning(贪心评估)", 平均回合回报=mean_r,
            最后100回合回报=float("nan"), 收敛回合数=0)

    exp.figure("l8_01_rl", "line", "Q-Learning 每回合回报（50 回合滑动平均）",
               [str(i) for i in range(len(smooth))],
               [("滑动平均回报", [float(v) for v in smooth]),
                ("最优回报", [optimal_return] * len(smooth))],
               ylabel="回合回报", xlabel="回合")
    exp.note("回报口径统一为折扣累积（γ=0.95）。价值迭代给出的最优起点价值 %.3f；"
             "Q-Learning 训练后期（最后 100 回合）平均 %.3f，贪心策略评估 %.3f ± %.3f。"
             % (optimal_return, float(rets[-100:].mean()), mean_r, std_r))
    gap = abs(optimal_return - mean_r)
    exp.note("与最优解的差距 %.3f（%.1f%%）：%s"
             % (gap, 100 * gap / abs(optimal_return),
                "已收敛到最优策略。" if gap < 0.05 else
                "仍有差距，原因是 ε-贪心训练中的探索与有限回合数，可通过增加回合数或降低 ε 衰减速度改善。"))
    exp.note("随机策略平均 %.3f（大量回合踩陷阱或超时），与学习后的策略形成鲜明对比，"
             "说明回报的提升来自学习而非环境本身。" % float(np.mean(rand_returns)))
    exp.note("局限与后续：本环境是离散小状态空间，表格 Q 已足够；DQN 的价值在于用网络泛化到"
             "连续/大状态空间（如视觉输入），需 `pip install torch` 且建议 GPU 环境。"
             "教学上可先讲清本实验的 TD 更新与探索-利用，再引入函数近似。")
    exp.save()


if __name__ == "__main__":
    main()
