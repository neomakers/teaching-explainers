"""从零实现：表格 Q-Learning（eps-greedy）。

强化学习实验的可行替代方案：DQN 需要 PyTorch + 大量算力（本机显卡损坏），
但核心思想（时序差分、探索-利用、折扣回报）可在自建网格世界中完整验证。
"""
import numpy as np


class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.15, gamma=0.95,
                 eps=0.15, eps_decay=0.999, eps_min=0.02, seed=42):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.eps_decay, self.eps_min = eps_decay, eps_min
        self.rng = np.random.default_rng(seed)

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.Q.shape[1]))
        q = self.Q[s]
        return int(np.flatnonzero(q == q.max())[0])

    def train(self, env, episodes=1500, max_steps=100):
        returns = []
        for _ in range(episodes):
            s = env.reset()
            total, done, step = 0.0, False, 0
            while not done and step < max_steps:
                a = self.act(s)
                s2, r, done = env.step(a)
                best = self.Q[s2].max()
                self.Q[s, a] += self.alpha * (r + self.gamma * best -
                                              self.Q[s, a])
                total += (self.gamma ** step) * r      # 折扣回报，与价值迭代口径一致
                s, step = s2, step + 1
            returns.append(total)
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return np.array(returns)

    def evaluate(self, env, episodes=200, max_steps=100):
        out = []
        for _ in range(episodes):
            s = env.reset()
            total, done, step = 0.0, False, 0
            while not done and step < max_steps:
                s, r, done = env.step(self.act(s, greedy=True))
                total += (self.gamma ** step) * r      # 折扣回报
                step += 1
            out.append(total)
        return float(np.mean(out)), float(np.std(out))
