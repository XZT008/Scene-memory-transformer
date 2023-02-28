import numpy as np
import torch
import torch.nn as nn
from config import SMT_Config


class REPLAY_BUFFER(object):
    def __init__(self, config):
        self.max_size = config.max_rb_size
        self.batch_size = config.sample_size
        self.memories = []
        self.actions = []
        self.rewards = []

    def if_full(self):
        return len(self.memories) == len(self.actions) == len(self.rewards) == self.max_size

    def sample_batch(self):
        if not self.if_full():
            print("Replay buffer not full yet!")
        else:
            sample_idx = np.random.choice(self.max_size, self.batch_size, replace=False)
            batch_mem, batch_action, batch_reward = [], [], []
            for idx in sample_idx:
                batch_mem.append(self.memories[idx])
                batch_action.append(self.actions[idx])
                batch_reward.append(self.rewards[idx])
            return batch_mem, batch_action, batch_reward

    def add_traj(self, mems, acts, rewards):
        for mem, act, reward in zip(mems, acts, rewards):
            self.memories.append(mem)
            self.actions.append(act)
            self.rewards.append(reward)

    def update(self):
        self.memories = self.memories[self.max_size/2:]
        self.actions = self.actions[self.max_size / 2:]
        self.rewards = self.rewards[self.max_size / 2:]

