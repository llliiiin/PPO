import torch
import numpy as np
import torch.nn.functional as F


class Agent_PPO:
    """
    负责与策略 / 价值网络交互，产生动作与 logπ
    """
    def __init__(self, state_dim, action_dim, actor, critic):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = actor          # π(a|s; θ)
        self.critic = critic        # V(s; ϕ)

    def action(self, obs, epsilon, device):
        """
        根据当前策略产生动作
        返回: action(int), log_prob(float), value(float)

        关键点
        --------
        - 以 ε 概率随机动作，否则按策略采样
        - **无论动作来源，都用当前策略 π(a|s) 计算 log_prob**，这样
          PPO 的 ratio = exp(logπ - logπ_old) 才是正确的。
        """
        state  = torch.tensor(obs, dtype=torch.float32, device=device)

        # 当前策略 π(a|s) 的 logits 与分布
        logits = self.actor(state)
        dist   = torch.distributions.Categorical(logits=logits)

        # ---------- ε-greedy ----------
        if np.random.rand() < epsilon:                     # 随机动作
            action_tensor = torch.tensor(
                np.random.randint(self.action_dim), device=device
            )
        else:                                              # 按策略采样
            action_tensor = dist.sample()

        action   = action_tensor.item()
        log_prob = dist.log_prob(action_tensor)            # 一律用 π 计算
        value    = self.critic(state).squeeze()            # V(s)

        return action, log_prob.item(), value.item()


    def evaluate_actions(self, states, actions, device):
        """
        policy.learn 中使用：批量评估 (s, a)
        """
        states  = torch.tensor(states,  dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long,   device=device)

        logits = self.actor(states)
        dist   = torch.distributions.Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        values    = self.critic(states).squeeze()

        return log_probs, values, entropy
