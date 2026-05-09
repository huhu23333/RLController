# replay_buffer.py - 经验回放缓冲区
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    固定大小的经验回放缓冲区，用于存储 SAC 训练数据.
    """
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        """存入一条经验"""
        self.buffer.append({
            'state': state.astype(np.float32),
            'action': action.astype(np.float32),
            'reward': np.float32(reward),
            'next_state': next_state.astype(np.float32),
            'done': np.float32(done),
        })

    def sample(self, batch_size):
        """随机采样一个批次"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states      = np.stack([x['state'] for x in batch])
        actions     = np.stack([x['action'] for x in batch])
        rewards     = np.stack([x['reward'] for x in batch])
        next_states = np.stack([x['next_state'] for x in batch])
        dones       = np.stack([x['done'] for x in batch])

        return states, actions, rewards, next_states, dones

    def save(self, filepath):
        """保存缓冲区到文件"""
        data = {
            'states':      np.stack([x['state'] for x in self.buffer]),
            'actions':     np.stack([x['action'] for x in self.buffer]),
            'rewards':     np.array([x['reward'] for x in self.buffer]),
            'next_states': np.stack([x['next_state'] for x in self.buffer]),
            'dones':       np.array([x['done'] for x in self.buffer]),
        }
        np.savez_compressed(filepath, **data)

    def load(self, filepath):
        """从文件加载缓冲区"""
        data = np.load(filepath)
        n = len(data['states'])
        for i in range(n):
            self.buffer.append({
                'state': data['states'][i],
                'action': data['actions'][i],
                'reward': data['rewards'][i],
                'next_state': data['next_states'][i],
                'done': data['dones'][i],
            })
        print(f"已从 {filepath} 加载 {n} 条经验")
