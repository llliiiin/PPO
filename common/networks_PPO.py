import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """
    策略网络（Actor）- 输出动作概率分布
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 返回logits，未经过softmax


class CriticNetwork(nn.Module):
    """
    价值网络（Critic）- 输出状态价值估计
    """

    def __init__(self, state_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RecurrentActorNetwork(nn.Module):
    """
    带有RNN的策略网络 - 用于处理序列数据
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, rnn_hidden_dim=128):
        super(RecurrentActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, rnn_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'rnn' not in name:
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, states, hidden=None):
        """
        输入:
            states: [batch_size, seq_len, state_dim] 或单个状态 [state_dim]
            hidden: 隐藏状态 (可选)
        """
        # 处理单个状态的情况
        if states.dim() == 1:
            states = states.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        # 处理批次但非序列的情况
        elif states.dim() == 2:
            states = states.unsqueeze(1)  # [batch_size, 1, state_dim]

        batch_size, seq_len, _ = states.size()

        # 转换输入
        x = F.relu(self.fc1(states.view(batch_size * seq_len, -1)))
        x = x.view(batch_size, seq_len, -1)

        # 通过RNN
        self.rnn.flatten_parameters() if hasattr(self.rnn, 'flatten_parameters') else None
        x, hidden = self.rnn(x, hidden)

        # 输出层
        x = F.relu(self.fc2(x.contiguous().view(batch_size * seq_len, -1)))
        logits = self.fc3(x)

        # 返回结果，针对序列的最后一个时间步或者单个状态
        if states.size(1) == 1:  # 单个状态或单步序列
            return logits.squeeze(0)  # [batch_size, action_dim]
        else:  # 返回整个序列的输出
            return logits.view(batch_size, seq_len, -1), hidden


class RecurrentCriticNetwork(nn.Module):
    """
    带有RNN的价值网络 - 用于处理序列数据
    """

    def __init__(self, state_dim, hidden_dim=256, rnn_hidden_dim=128):
        super(RecurrentCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, rnn_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'rnn' not in name:
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, states, hidden=None):
        """
        输入:
            states: [batch_size, seq_len, state_dim] 或单个状态 [state_dim]
            hidden: 隐藏状态 (可选)
        """
        # 处理单个状态的情况
        if states.dim() == 1:
            states = states.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        # 处理批次但非序列的情况
        elif states.dim() == 2:
            states = states.unsqueeze(1)  # [batch_size, 1, state_dim]

        batch_size, seq_len, _ = states.size()

        # 转换输入
        x = F.relu(self.fc1(states.view(batch_size * seq_len, -1)))
        x = x.view(batch_size, seq_len, -1)

        # 通过RNN
        self.rnn.flatten_parameters() if hasattr(self.rnn, 'flatten_parameters') else None
        x, hidden = self.rnn(x, hidden)

        # 输出层
        x = F.relu(self.fc2(x.contiguous().view(batch_size * seq_len, -1)))
        values = self.fc3(x)

        # 返回结果，针对序列的最后一个时间步或者单个状态
        if states.size(1) == 1:  # 单个状态或单步序列
            return values.squeeze()  # [batch_size, 1] -> [batch_size]
        else:  # 返回整个序列的输出
            return values.view(batch_size, seq_len, -1), hidden