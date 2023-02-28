import os
import numpy
import torch

class SMT_Config:
    def __init__(self):
        self.max_episode_len = 250
        self.actor_num = 10
        self.envs_per_actor = 4
        self.max_rb_size = 200
        self.init_epsilon = 0.1
        self.lam = 5
        self.learning_starts = 1
        self.sample_size = 32
        self.target_update_freq = 10
        self.gamma = 0.99
        self.grad_norm_clipping = 0.1
        self.training_steps = 50


