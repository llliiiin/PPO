import torch
import torch.nn as nn
import numpy as np


class PPO:
    """
    PPO-Clip + GAE
    """
    def __init__(
        self,
        state_shape,
        action_shape,
        actor_optimizer,
        critic_optimizer,
        actor,
        critic,
    ):
        self.state_shape = state_shape
        self.n_actions   = action_shape

        self.actor  = actor
        self.critic = critic
        self.actor_opt  = actor_optimizer
        self.critic_opt = critic_optimizer

        # ---------- 超参数 ---------- #
        self.clip_param    = 0.2
        self.ppo_epochs    = 6
        self.batch_size    = 64
        self.value_coef    = 0.5
        self.entropy_coef  = 0.05
        self.max_grad_norm = 0.5
        self.gae_lambda    = 0.95
        self.gamma         = 0.95
        self.value_clip    = 1e4

        self.value_loss = nn.MSELoss()

    # -------------------------------------------------- #
    def _compute_gae(self, rewards, values, dones, device):
        """
        GAE(λ) —— 对完整轨迹, 最后一条必定 dones[t]=1
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            next_value = values[t + 1] if t + 1 < T else 0.0  # 轨迹终点 bootstrap=0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        adv = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret = torch.tensor(returns, dtype=torch.float32, device=device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # 标准化
        return ret, adv

    # -------------------------------------------------- #
    def learn(self, trajs, buffer, device):
        """
        用采样的若干条 Trajectory 更新一次 Actor / Critic
        """

        # ---------- 把多条轨迹拼成批次 ----------
        states, actions, log_old, rewards, values, dones = [], [], [], [], [], []

        for traj in trajs:
            T = traj.length()
            states.append( np.array(traj.states) )
            actions.append(np.array(traj.actions))
            log_old.append(np.array(traj.log_probs))
            rewards.append(np.array(traj.rewards))
            values.append(np.array(traj.values))

            # dones = [0, 0, ..., 0, 1]  (最后一步为 1)
            dones.append(
                np.concatenate(
                    [np.zeros(T-1, dtype=np.float32),
                     np.ones (1 , dtype=np.float32)]
                )
            )

        states  = np.concatenate(states)
        actions = np.concatenate(actions)
        log_old = np.concatenate(log_old)
        rewards = np.concatenate(rewards)
        values  = np.concatenate(values)
        dones   = np.concatenate(dones)          # ←✔ 正确的 0/1 标记

        # ---------- 计算 GAE & returns ----------
        returns, advantages = self._compute_gae(
            rewards, values, dones, device
        )

        # ---------- 转 Tensor ----------
        states   = torch.tensor(states,  dtype=torch.float32, device=device)
        actions  = torch.tensor(actions, dtype=torch.long,   device=device)
        log_old  = torch.tensor(log_old, dtype=torch.float32, device=device)

        dataset = torch.utils.data.TensorDataset(
            states, actions, log_old, returns, advantages
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        actor_losses, critic_losses = [], []

        # ---------- 多 epoch 迭代 ----------
        for _ in range(self.ppo_epochs):
            for s_b, a_b, log_old_b, ret_b, adv_b in loader:

                if torch.isnan(adv_b).any():
                    print("!! NaN in advantages")
                else:
                    print("adv mean:", adv_b.mean().item(), "std:", adv_b.std().item())

                # π(a|s)、logπ、新 V(s)
                logits = self.actor(s_b)
                if torch.isnan(logits).any():
                    raise RuntimeError("NaN in logits")  # 立刻中断定位
                dist   = torch.distributions.Categorical(logits=logits)
                log_pi = dist.log_prob(a_b)
                entropy= dist.entropy().mean()

                ratio = torch.exp(log_pi - log_old_b)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio,
                                    1 - self.clip_param,
                                    1 + self.clip_param) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean() \
                             - self.entropy_coef * entropy

                # ---- 更新 Actor ----
                self.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                               self.max_grad_norm)
                self.actor_opt.step()

                # ---- 更新 Critic ----
                value_pred_raw = self.critic(s_b).squeeze()
                value_pred = torch.clamp(value_pred_raw, -self.value_clip, self.value_clip)
                # value_pred  = self.critic(s_b).squeeze()
                critic_loss = self.value_loss(value_pred, ret_b) * self.value_coef

                self.critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                               self.max_grad_norm)
                self.critic_opt.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        return float(np.mean(actor_losses) + np.mean(critic_losses))

    # -------------------------------------------------- #
    def save(self, path):
        torch.save(self.actor.state_dict(),  path + "actor.pth")
        torch.save(self.critic.state_dict(), path + "critic.pth")
        print("====== model saved ======")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "actor.pth"))
        self.critic.load_state_dict(torch.load(path + "critic.pth"))
        print("====== model loaded ======")
