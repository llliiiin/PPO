import torch
import os
import numpy as np


class Trainer_PPO:
    def __init__(self, policy, env, buffer, agent, args):
        super().__init__()

        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.agent = agent
        self.batch_size = args.batch_size
        self.num_episode = args.num_episode
        self.sim_timesteps = args.sim_timesteps          # 每次采集样本时模拟的时隙长度
        self.epsilon = args.epsilon                      # 选择动作时epsilon-greedy的参数
        self.epsilon_decay = args.epsilon_decay          # 每轮训练后epsilon衰减率
        self.epsilon_decay_interval = args.epsilon_decay_interval
        self.test_interval = args.test_interval
        self.num_test_episodes = 5                       # 每次测试时运行的轮数
        self.args = args
        self.writer = None                               # For logging.

        self.action_counts = {0: 0, 1: 0}  # 统计本轮采样到的 action 次数
        self.value_buffer = []  # 收集 critic 估值，便于统计均值/方差

    def train(self, check_dir):
        for episode in range(self.num_episode):
            print(f"Episode {episode + 1}/{self.num_episode}")
            self.env.reset()

            # -------- 每次模拟一段时间的道路环境，获得各车辆的轨迹样本，用buffer来存储（运行固定时间长度，从中提取完整样本，样本数量不固定）
            # -------- 多次模拟，直至样本数量足够一个batch --------
            while not self.buffer.ready(batch_size=self.batch_size):
                self.sim_trajs()

            # -------- 从本次采集的样本中取出batch_size个 --------
            trajs = self.buffer.sample(batch_size=self.batch_size)

            # Track average trajectory reward before training
            avg_reward = sum(traj.traj_reward for traj in trajs) / len(trajs)
            print(f"    Average trajectory reward: {avg_reward:.4f}")

            # -------- 更新网络 --------
            loss = self.policy.learn(trajs, self.buffer, self.args.device)
            print(f"    Policy loss: {loss:.4f}")

            if self.writer:
                self.writer.add_scalar('Training/Average_Reward', avg_reward, episode)
                self.writer.add_scalar('Training/Policy_Loss', loss, episode)
                self.writer.add_scalar('Training/Epsilon', self.epsilon, episode)

            # ---------- 新增：每 50 轮输出一次动作/价值统计 ----------
            if (episode + 1) % 50 == 0:
                total_actions = self.action_counts[0] + self.action_counts[1]
                if total_actions > 0:
                    ratio = self.action_counts[1] / total_actions
                    v_mean = np.mean(self.value_buffer) if self.value_buffer else 0.0
                    v_std = np.std(self.value_buffer) if self.value_buffer else 0.0
                    print(f"[Episode {episode + 1}] π(a=1) = {ratio:.3f} | "
                          f"V(s) mean = {v_mean:.2f} ± {v_std:.2f}")
                # 清零统计
                self.action_counts = {0: 0, 1: 0}
                self.value_buffer = []


            # -------- 更新epsilon-greedy参数 --------
            if (episode + 1) % self.epsilon_decay_interval == 0:
                if self.args.epsilon_decay_type == 'multi':
                    self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
                elif self.args.epsilon_decay_type == 'linear':
                    self.epsilon -= (self.args.epsilon - 0.01) / max(1, self.args.num_episode // self.args.epsilon_decay_interval)
                else:
                    raise ValueError("Unknown epsilon decay type")

            # -------- 定期测试模型 --------
            # if (episode + 1) % self.test_interval == 0:
            #     mean_test_reward = self.test(self.num_test_episodes)
            #     if self.writer:
            #         self.writer.add_scalar('Testing/Average_Reward', mean_test_reward, episode)

            # -------- 定期保存模型 --------
            # if (episode + 1) % self.args.save_model_interval == 0:
            #     checkpoints_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), check_dir)
            #     os.makedirs(checkpoints_dir, exist_ok=True)
            #     self.policy.save(os.path.join(checkpoints_dir, f"episode_{episode + 1}_"))

            # -------- 清空buffer，预备下次训练重新采集 --------
            self.buffer.clear()

    def sim_trajs(self):
        '''
        逐时隙运行，每个时隙内，按以下顺序执行：
          1. 每进入一个新的时隙，更新场景中车辆状态（即active_vehicles中的车辆状态，未更新时其中保存的是上一时隙场内车辆）
             更新车辆位置、剩余时间、是否离开、本地持有数据量、传输队列长度（减去上一时隙传输的部分）
             将已超出监测区域的车辆从active_vehicles中移除
             获得已出现车辆中当前时隙仍在场景内的车辆状态，即针对当前时隙的active_vehicles（还没加入当前时隙新到达的）
          2. 添加此时隙新到达的车辆（设置位置、速度），并将id添加到active_vehicles中
             至此，active_vehicles中保存的是当前时隙场景内的所有车辆
          3. 遍历active_vehicles，使每一个非新到达、仍未计算的车辆执行动作决策（新到达的车辆还未接收数据，必定不计算）
             并将其记录进该车的轨迹中
          4. 根据第3步，排除本时隙决定计算的车辆后，获得本时隙参与传输的车辆和车均带宽
             给这些参与传输的车辆更新本时隙的传输速率（速率的计算需要可用带宽）
          5. 基于本时隙及之前的历史车均带宽，给每个新到达车辆选择码率，并设置与码率相关的anytime_alpha系数
          6. 更新所有本时隙参与传输的车辆的传输队列长度（加上此时隙要传的一帧）

        逐步运行完固定时隙长度后，取所有车辆中完整的轨迹样本，放入buffer中
        '''
        for t in range(self.sim_timesteps):
            self.env.current_timeslot = t + 1

            # 1. 更新上一时隙车辆状态
            self.env.update_vehicles()

            # 2. 生成并添加新到达车辆
            vehicles, active_vehicles, new_vehicles = self.env.arrive_new_vehicles()

            # 3. 对每辆“活跃且未计算且非新到达”车辆决策
            for vid in active_vehicles:
                # print('\n【vehicle_id】: ', vid)
                v = vehicles[vid]
                if v.is_computed or vid in new_vehicles:
                    continue

                # ---- 构造 & 归一化状态 ----
                state = self.env.get_vehicle_state(vid)
                # print('====== state ======')
                # print('计算能力： ', state[0])
                # print('剩余时间： ', state[1])
                # print('已接收帧数： ', state[2])
                # print('下一帧的接收比例： ', state[3])
                # print('码率： ', state[4])
                # print('误码率alpha： ', state[5])
                # print('与RSU空间距离： ', state[6])
                # print('其它未计算车辆的数量： ', state[7])
                # print('其它未计算车辆的平均剩余时间： ', state[8])


                state[0] = (state[0] - self.args.f_min) / (self.args.f_max - self.args.f_min)
                state[1] /= 7
                state[2] /= 12
                state[4] = (state[4] - min(self.args.rates_set)) / \
                           (max(self.args.rates_set) - min(self.args.rates_set))
                state[5] = (state[5] - min(self.args.any_alpha_set)) / \
                           (max(self.args.any_alpha_set) - min(self.args.any_alpha_set))
                state[6] = ((state[6] - v.I2V_channel.get_distance(
                    self.args.monitor_range, min(self.args.orth_pos_list))) /
                            (v.I2V_channel.get_distance(0, max(self.args.orth_pos_list)) -
                             v.I2V_channel.get_distance(self.args.monitor_range,
                                                        min(self.args.orth_pos_list))))
                state[7] /= 16
                state[8] /= 7

                # ---- 动作决策 ----
                action, log_prob, value = self.agent.action(state, self.epsilon, self.args.device)
                # print('action: ', action)

                # ---- 记录轨迹 ----
                self.action_counts[action] += 1
                self.value_buffer.append(value)

                if action == 0:
                    reward = 0.0
                    v.compute_step_reward(self.args)
                else:
                    reward = v.compute_step_reward(self.args)
                    v.is_computed = True

                v.traj.add(state, action, log_prob, reward, value)

            # 4. 更新带宽 & 速率
            bw_per_vehicle, to_trans = self.env.update_bandwidth_history()
            for vid in to_trans:
                vehicles[vid].update_trans_rate(self.args.time_slot, bw_per_vehicle)

            # 5. 新到达车辆码率 / α
            for vid in new_vehicles:
                v = vehicles[vid]
                idx = self.env.select_data_rate(vid)
                v.code_rate = self.args.rates_set[idx]
                v.anytime_alpha = self.args.any_alpha_set[idx]

            # 6. 传输队列 +1 帧
            for vid in to_trans:
                vehicles[vid].bits_in_queue += \
                    self.args.frame_size * 1024 * 8 * vehicles[vid].code_rate

            # -------- 模拟结束：整理轨迹 -------- #
        for v in self.env.vehicles.values():
            if v.is_warm_up:
                continue

            penalty = None
            if (not v.is_computed) and v.is_left:
                penalty = self.args.r_S

            v.traj.finalize(no_cmp_penalty=penalty)
            if v.traj.length() > 0:
                self.buffer.store_a_traj(v.traj)

    def test(self, num_test_episodes=5):
        """
        Evaluate the policy without exploration
        Returns average reward across test episodes
        """
        print("Running test evaluation...")
        test_rewards = []
        original_epsilon = self.epsilon  # Save current epsilon
        self.epsilon = 0.0  # Disable exploration during testing

        self.policy.actor.eval()

        for episode in range(num_test_episodes):
            self.env.reset()
            episode_trajs = []

            # Simulate for the same number of timesteps as in training
            self.sim_trajs_test(episode_trajs)

            # Calculate average reward for this test episode
            if episode_trajs:
                avg_reward = sum(traj.traj_reward for traj in episode_trajs) / len(episode_trajs)
                test_rewards.append(avg_reward)

        self.epsilon = original_epsilon  # Restore original epsilon
        self.policy.actor.train()

        if test_rewards:
            mean_test_reward = sum(test_rewards) / len(test_rewards)
            print(f"Test evaluation complete. Mean test reward: {mean_test_reward:.4f}")
            return mean_test_reward
        else:
            print("No complete trajectories collected during testing.")
            return 0.0

    def sim_trajs_test(self, test_trajs):
        """
        与 sim_trajs 类似，但用于测试阶段：
        - 关闭 ε-greedy（self.epsilon 已在 test() 中设为 0）
        - 不向 buffer 存样本，仅收集完整轨迹写入 test_trajs
        """
        for t in range(self.sim_timesteps):
            self.env.current_timeslot = t + 1

            # 1. 更新车辆状态
            self.env.update_vehicles()

            # 2. 新车辆到达
            vehicles, active_vehicles, new_vehicles = self.env.arrive_new_vehicles()

            # 3. 为每辆“活跃且未计算且非新到达”车辆决策
            for vid in active_vehicles:
                v = vehicles[vid]
                if v.is_computed or vid in new_vehicles:
                    continue

                # ---- 构造并归一化状态 ----
                state = self.env.get_vehicle_state(vid)
                state[0] = (state[0] - self.args.f_min) / (self.args.f_max - self.args.f_min)
                state[1] /= 7
                state[2] /= 12
                state[4] = (state[4] - min(self.args.rates_set)) / (
                        max(self.args.rates_set) - min(self.args.rates_set))
                state[5] = (state[5] - min(self.args.any_alpha_set)) / (
                        max(self.args.any_alpha_set) - min(self.args.any_alpha_set))
                state[6] = ((state[6] - v.I2V_channel.get_distance(
                    self.args.monitor_range, min(self.args.orth_pos_list))) /
                            (v.I2V_channel.get_distance(0, max(self.args.orth_pos_list)) -
                             v.I2V_channel.get_distance(self.args.monitor_range,
                                                        min(self.args.orth_pos_list))))
                state[7] /= 16
                state[8] /= 7

                # ---- 动作决策（ε = 0，只用当前策略）----
                action, log_prob, value = self.agent.action(state, self.epsilon, self.args.device)

                # ---- 记录轨迹 ----
                if action == 0:  # 不计算
                    reward = 0.0
                else:  # 执行计算
                    reward = v.compute_step_reward(self.args)
                    v.is_computed = True

                v.traj.add(state, action, log_prob, reward, value)

            # 4. 更新带宽 / 速率 / 码率
            bw_per_vehicle, to_trans = self.env.update_bandwidth_history()
            for vid in to_trans:
                vehicles[vid].update_trans_rate(self.args.time_slot, bw_per_vehicle)

            for vid in new_vehicles:
                v = vehicles[vid]
                idx = self.env.select_data_rate(vid)
                v.code_rate = self.args.rates_set[idx]
                v.anytime_alpha = self.args.any_alpha_set[idx]

            for vid in to_trans:
                vehicles[vid].bits_in_queue += (
                        self.args.frame_size * 1024 * 8 * vehicles[vid].code_rate)

        # -------- 模拟结束：整理轨迹 --------
        for v in self.env.vehicles.values():
            if v.is_warm_up:
                continue

            penalty = None
            if (not v.is_computed) and v.is_left:
                penalty = self.args.r_S

            v.traj.finalize(no_cmp_penalty=penalty)
            if v.traj.length() > 0:
                test_trajs.append(v.traj)
