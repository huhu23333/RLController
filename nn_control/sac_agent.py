# sac_agent.py - SAC 智能体，整合 Actor, Critic, 目标网络和训练逻辑
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from networks import Actor, Critic


class SACAgent:
    """
    Soft Actor-Critic 智能体.
    使用双 Q 网络 + 自动熵调节.
    """
    def __init__(self, input_dim, action_dim, hidden_dim=256,
                 lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 gamma=0.99, tau=0.005, target_entropy=None,
                 u_min=-2.0, u_max=2.0, device='cpu'):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # Actor (策略网络)
        self.actor = Actor(input_dim, action_dim, hidden_dim, u_min, u_max).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # 双 Critic (Q 网络)
        self.critic1 = Critic(input_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(input_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # 目标 Critic
        self.critic1_target = Critic(input_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target = Critic(input_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 自动熵调节参数
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.alpha = self.log_alpha.exp().item()

    def select_action(self, state_np, deterministic=False):
        """根据当前策略选择动作 (numpy 接口)"""
        return self.actor.get_action(state_np, deterministic=deterministic)

    def update(self, replay_buffer, batch_size):
        """
        从经验回放缓冲区采样并执行一次 SAC 更新.

        Returns:
            dict: 各项损失和参数值 (用于 TensorBoard 日志)
        """
        if len(replay_buffer) < batch_size:
            return None

        # 采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.FloatTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        # ---- 更新 Critic ----
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor(next_states)
            next_q1 = self.critic1_target(next_states, next_actions)
            next_q2 = self.critic2_target(next_states, next_actions)
            next_q_min = torch.min(next_q1, next_q2)
            target_q = rewards + (1 - dones) * self.gamma * (next_q_min - self.alpha * next_log_prob)

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ---- 更新 Actor ----
        new_actions, log_prob, _ = self.actor(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new_min = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- 更新 Alpha (熵系数) ----
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # ---- 软更新目标网络 ----
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss':   actor_loss.item(),
            'alpha_loss':   alpha_loss.item(),
            'alpha':        self.alpha,
            'q1_mean':      current_q1.mean().item(),
            'q2_mean':      current_q2.mean().item(),
        }

    def _soft_update(self, source, target):
        """软更新: target = tau * source + (1 - tau) * target"""
        with torch.no_grad():
            for param_s, param_t in zip(source.parameters(), target.parameters()):
                param_t.data.copy_(self.tau * param_s.data + (1 - self.tau) * param_t.data)

    def save(self, save_dir, step):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.actor.state_dict(),
                   os.path.join(save_dir, f"actor_{step}.pth"))
        torch.save(self.critic1.state_dict(),
                   os.path.join(save_dir, f"critic1_{step}.pth"))
        torch.save(self.critic2.state_dict(),
                   os.path.join(save_dir, f"critic2_{step}.pth"))
        print(f"模型已保存到 {save_dir} (step={step})")

    def load(self, save_dir, step):
        """加载模型"""
        self.actor.load_state_dict(
            torch.load(os.path.join(save_dir, f"actor_{step}.pth"),
                       map_location=self.device))
        self.critic1.load_state_dict(
            torch.load(os.path.join(save_dir, f"critic1_{step}.pth"),
                       map_location=self.device))
        self.critic2.load_state_dict(
            torch.load(os.path.join(save_dir, f"critic2_{step}.pth"),
                       map_location=self.device))
        print(f"模型已从 {save_dir} 加载 (step={step})")
