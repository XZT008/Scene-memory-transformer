# this class is in charge of collecting trajs. Actor runs on CPU

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SMT_Config
import math
from network import FEATURE_EXTRACTOR, DQN_POLICY
from stable_baselines3.common.vec_env import SubprocVecEnv
from replay_buffer import REPLAY_BUFFER

import sys

sys.path.append('./env/')
from box_world import BoxWorld
import numpy as np

ENV_ID = "Simple-Box-v0"


def make_env(env_id, rank):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init():
        env = BoxWorld(isGUI=False)
        env.seed(rank)
        return env

    return _init


class DQN_ACTOR(object):
    def __init__(self, envs, policy_weights, extractor_weights, config):
        self.lam = config.lam
        self.init_epsilon = config.init_epsilon
        self.learning_starts = config.learning_starts
        self.env_num = config.envs_per_actor
        self.envs = envs
        self.max_episode_len = config.max_episode_len
        self.extractor = FEATURE_EXTRACTOR()
        self.policy = DQN_POLICY()
        # self.load_weights(policy_weights, extractor_weights)
        self._reset_storage()

    def load_weights(self, policy_weights, extractor_weights):
        if extractor_weights is not None:
            self.extractor.set_weights(extractor_weights)
        self.policy.set_weights(policy_weights)

    def _reset_storage(self):
        self.time_step = 0
        self.trajectories = {
            'memories': [],
            'rewards': [],
            'actions': []
        }
        self.envs.reset()

    def collect(self, replay_buffer):
        done_masks = np.array([False] * self.env_num)
        done_timestep = [[] for i in range(self.env_num)]
        while self.time_step < self.max_episode_len and np.sum(np.where(done_masks, 1, 0)) != self.env_num:
            perform_random_action = (self.time_step < self.learning_starts) or (np.random.random() < self.init_epsilon)
            if perform_random_action:
                actions = np.random.choice(6, self.env_num, p=[0.25, 0.25, 0.25, 0.25, 0, 0])
            else:
                mem = torch.cat(self.trajectories['memories'], dim=1)
                actions_logit = self.policy(mem, mem[:, -1], None, None)
                actions = torch.argmax(actions_logit, dim=1).numpy()

            obs, rewards, dones, info = self.envs.step(actions)
            done_masks = np.logical_or(done_masks, dones)
            for idx, done in enumerate(dones):
                if done and len(done_timestep[idx]) == 0:
                    done_timestep[idx].append(self.time_step)
            rewards = torch.tensor(rewards).float().unsqueeze(1)
            actions = torch.tensor(actions).float().unsqueeze(1)
            imgs = torch.from_numpy(np.reshape(np.stack(obs[:, 3], axis=0), (-1, 80, 80, 3))).float().permute(0, 3, 1,
                                                                                                              2)
            imgs = imgs * 2 / 255 - 1
            poses = np.stack(obs[:, 1], axis=0)
            normed_poses = [poses[:, 0] / self.lam, poses[:, 1] / self.lam, np.cos(poses[:, 2]), np.sin(poses[:, 2]),
                            np.repeat(np.exp(-self.time_step), self.env_num)]
            normed_poses = torch.tensor(np.stack(normed_poses, axis=1)).float()
            memory = self.extractor(imgs, normed_poses, actions).detach().unsqueeze(1)
            self.trajectories['memories'].append(memory)
            self.trajectories['rewards'].append(rewards)
            self.trajectories['actions'].append(actions)

            self.time_step += 1

        rewards = torch.chunk(torch.cat(self.trajectories['rewards'], dim=1), self.env_num, dim=0)
        actions = torch.chunk(torch.cat(self.trajectories['actions'], dim=1), self.env_num, dim=0)
        memories = torch.chunk(torch.cat(self.trajectories['memories'], dim=1), self.env_num, dim=0)

        reward_lst = []
        action_lst = []
        memory_lst = []
        for idx,(r,a,m,time_step) in enumerate(zip(rewards, actions, memories, done_timestep)):
            if len(time_step) != 0:
                time_step = time_step[0]
                reward_lst.append(r[:,:time_step+1])
                action_lst.append(a[:, :time_step+1])
                memory_lst.append(m[:, :time_step+1])
        replay_buffer.add_traj(memory_lst, action_lst, reward_lst)


if __name__ == '__main__':
    # test env
    """
    env = BoxWorld(isGUI=False)
    obs = env.reset()
    next_obs, reward, dones, info = env.step(np.random.choice(6, 1))
    """

    config = SMT_Config()
    envs = SubprocVecEnv([make_env(ENV_ID, i) for i in range(config.envs_per_actor)], start_method="forkserver")
    actor = DQN_ACTOR(envs, None, None, config)
    rb = REPLAY_BUFFER(config)
    actor.collect(rb)

    pass
