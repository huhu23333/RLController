# networks.py - SAC 神经网络结构
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


def init_weights(layer, scale=1.0):
    """正交初始化权重"""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=scale)
        layer.bias.data.zero_()


class ResidualFCBlock(nn.Module):
    def __init__(self, dim, activation = nn.ReLU()):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.fc(x)) + x


class Actor(nn.Module):
    """
    SAC Actor 网络.
    输入: state + future target sequence (sin/cos)
    输出: 力矩序列 (经 tanh 压缩后缩放到 [u_min, u_max])
    """
    def __init__(self, input_dim, action_dim, hidden_dim, u_min=-2.0, u_max=2.0):
        super().__init__()
        self.action_dim = action_dim
        self.action_scale = (u_max - u_min) / 2.0
        self.action_bias = (u_max + u_min) / 2.0

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            ResidualFCBlock(hidden_dim, nn.LeakyReLU()),
            ResidualFCBlock(hidden_dim, nn.LeakyReLU()),
            ResidualFCBlock(hidden_dim, nn.Tanh()),
            ResidualFCBlock(hidden_dim, nn.LeakyReLU()),
            ResidualFCBlock(hidden_dim, nn.Tanh()),
            ResidualFCBlock(hidden_dim, nn.LeakyReLU()),
            ResidualFCBlock(hidden_dim, nn.Tanh()),
            ResidualFCBlock(hidden_dim, nn.LeakyReLU()),
            ResidualFCBlock(hidden_dim, nn.Tanh()),
            ResidualFCBlock(hidden_dim, nn.Tanh()),
            nn.LeakyReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # 小权重初始化，让初始策略接近零力矩
        nn.init.uniform_(self.mean_layer.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.mean_layer.bias, -1e-3, 1e-3)
        nn.init.uniform_(self.log_std_layer.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.log_std_layer.bias, -1e-3, 1e-3)

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state, deterministic=False, with_log_prob=True):
        """
        前向传播.

        Args:
            state: [batch, input_dim] 输入状态 + 目标序列
            deterministic: True 时直接输出均值 (推理模式)
            with_log_prob: True 时返回 log_prob (训练需要)

        Returns:
            action:    [batch, action_dim] 缩放到力矩范围的行动
            log_prob:  [batch, 1] log π(a|s)
            mean:      [batch, action_dim] 均值 (用于调试/记录)
        """
        x = self.net(state)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            # 推理模式: 直接用 tanh(mean)
            raw_action = torch.tanh(mean)
            action = raw_action * self.action_scale + self.action_bias
            log_prob = None
        else:
            # 重参数化采样
            dist = Normal(mean, std)
            z = dist.rsample()                     # z ~ N(mean, std)
            raw_action = torch.tanh(z)
            action = raw_action * self.action_scale + self.action_bias

            if with_log_prob:
                # log π(a|s) = log p(z) - Σ log(1 - tanh²(z_i))
                log_prob_z = dist.log_prob(z)       # [batch, action_dim]
                log_prob_tanh = torch.log(1 - raw_action.pow(2) + 1e-6)
                log_prob = (log_prob_z - log_prob_tanh).sum(dim=-1, keepdim=True)
            else:
                log_prob = None

        return action, log_prob, mean

    def get_action(self, state_np, deterministic=False):
        """
        用于推理的便捷接口，输入 numpy 数组，返回 numpy 数组.

        Args:
            state_np: [input_dim] 或 [batch, input_dim]
            deterministic: 是否确定性

        Returns:
            action_np: [action_dim] 或 [batch, action_dim]
        """
        if state_np.ndim == 1:
            state_np = state_np[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        with torch.no_grad():
            state_t = torch.FloatTensor(state_np)
            action_t, _, _ = self.forward(state_t, deterministic=deterministic, with_log_prob=False)
            action_np = action_t.cpu().numpy()

        return action_np[0] if squeeze else action_np


class Critic(nn.Module):
    """
    SAC Critic (Q-function) 网络.
    输入: state + action
    输出: Q 值 (标量)
    """
    def __init__(self, input_dim, action_dim, hidden_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.LeakyReLU(),
            ResidualFCBlock(hidden_dim, nn.LeakyReLU()),
            ResidualFCBlock(hidden_dim, nn.LeakyReLU()),
            ResidualFCBlock(hidden_dim, nn.LeakyReLU()),
            ResidualFCBlock(hidden_dim, nn.LeakyReLU()),
            ResidualFCBlock(hidden_dim, nn.LeakyReLU()),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(lambda m: init_weights(m, 1.0))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q(x)
