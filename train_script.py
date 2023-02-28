import torch
import torch.nn as nn
from replay_buffer import REPLAY_BUFFER
from config import SMT_Config
from actor import DQN_ACTOR, make_env
from critic import CRITIC
from stable_baselines3.common.vec_env import SubprocVecEnv

config = SMT_Config()
envs = SubprocVecEnv([make_env("Simple-Box-v0", i) for i in range(config.envs_per_actor)], start_method="forkserver")
actor = DQN_ACTOR(envs, None, None, config)
critic = CRITIC(config)
rb = REPLAY_BUFFER(config)

NUM_CYCLE = 100
for i in range(NUM_CYCLE):
    if not rb.if_full():
        for j in range(config.max_rb_size / config.envs_per_actor):
            actor.collect(rb)
    else:
        rb.update()
        for j in range(config.max_rb_size / 2*config.envs_per_actor):
            actor.collect(rb)

    for j in range(config.training_steps):
        batch_mem, batch_action, batch_reward = rb.sample_batch()
        loss = critic.update(batch_mem, batch_action, batch_reward)

    actor.policy.load_state_dict(critic.Q_func.state_dict())
