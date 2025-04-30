"""
这个文件用来定义单条轨迹类，和buffer类
"""

import numpy as np
import torch
import random
from collections import deque




class Trajectory:
    """
    PPO 使用的轨迹类
    """
    def __init__(self):
        self.states, self.actions = [], []
        self.log_probs, self.rewards, self.values = [], [], []
        self.traj_reward = 0.0

    def add(self, state, action, log_prob, reward, value):
        """
        记录一次交互 (s, a, logπ(a|s), r, V(s))
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)

    def finalize(self, no_cmp_penalty=None):
        """
        结束轨迹时调用：计算累积奖励，可追加未计算惩罚
        """
        self.traj_reward = float(np.sum(self.rewards))
        if no_cmp_penalty is not None:
            self.traj_reward -= float(no_cmp_penalty)

    def length(self):
        return len(self.states)



# ------------------------- Buffer ------------------------- #
class TrajectoryBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def store_a_traj(self, traj: Trajectory):
        self.buffer.append(traj)

    def sample(self, batch_size=20):
        if batch_size >= len(self.buffer):
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)

    def all_traj_rewards(self):
        return [traj.traj_reward for traj in self.buffer]

    # --------------- 修改：避免预热期卡住 ---------------- #
    def ready(self, batch_size=20):
        """
        缓冲区内样本数是否达到一个 batch
        """
        return len(self.buffer) >= batch_size

    def clear(self):
        self.buffer.clear()

    def get_length(self):
        return len(self.buffer)

