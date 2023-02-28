# This class is in charge of train policy

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SMT_Config
import math
from network import DQN_POLICY


class CRITIC(nn.Module):
    def __init__(self, config):
        super(CRITIC, self).__init__()
        self.target_update_freq = config.target_update_freq
        self.num_param_updates = 0
        self.gamma = config.gamma
        self.grad_norm_clipping = config.grad_norm_clipping
        self.Q_func = DQN_POLICY()
        self.target_Q_func = DQN_POLICY()
        self.optimizer = torch.optim.Adam(self.Q_func.parameters(), lr=0.0005)

    def get_masks_and_o(self, mem):
        t = mem.size()[1]
        o_curr = mem.squeeze(0)[:-1,:]
        o_next = mem.squeeze(0)[1:,:]

        encoder_masks = torch.ones(t,t,t).cuda()
        decoder_masks = torch.ones(t,1,t).cuda()

        for i in range(t):
            encoder_masks[i,:i,:i] = 0
            decoder_masks[i,:,:i] = 0

        encoder_masks_curr = encoder_masks[:-1]
        encoder_masks_next = encoder_masks[1:]
        decoder_masks_curr = decoder_masks[:-1]
        decoder_masks_next = decoder_masks[1:]

        # convert masks into N*num_heads, num_heads=8
        encoder_masks_curr = encoder_masks_curr.unsqueeze(1).repeat(1,8,1,1).view((t-1)*8, t, t)
        encoder_masks_next = encoder_masks_next.unsqueeze(1).repeat(1,8,1,1).view((t-1)*8, t, t)
        decoder_masks_curr = decoder_masks_curr.unsqueeze(1).repeat(1,8,1,1).view((t-1)*8, 1, t)
        decoder_masks_next = decoder_masks_next.unsqueeze(1).repeat(1,8,1,1).view((t-1)*8, 1, t)
        return (o_curr,encoder_masks_curr, decoder_masks_curr), (o_next, encoder_masks_next, decoder_masks_next)

    # I implemented in Double DQN, where Q_func estimate current_q and target_Q_func estimate best_next_Q
    # and use Q_func to estimate next_ob best action. Update target_Q_func at config.target_update_freq
    def get_loss(self, mem, action, reward):
        mem, action, reward = mem.cuda(), action.cuda(), reward.cuda()
        t = mem.size()[1]
        (o_curr, encoder_masks_curr, decoder_masks_curr), (o_next, encoder_masks_next, decoder_masks_next) = self.get_masks_and_o(mem)
        repeated_mem = mem.repeat(t-1,1,1)
        max_ac = self.Q_func(repeated_mem, o_next, encoder_masks_next, decoder_masks_curr).argmax(-1, True) #t-1,1

        curr_Q = self.Q_func(repeated_mem, o_curr, encoder_masks_curr, decoder_masks_curr)
        curr_Q = curr_Q.gather(-1, action[:,:-1].long().view(-1,1)).squeeze()
        best_next_Q = self.target_Q_func(repeated_mem, o_next, encoder_masks_next, decoder_masks_next)\
            .gather(-1, max_ac).squeeze()
        calc_Q = reward[:,:-1].squeeze()+(self.gamma*best_next_Q)

        return F.smooth_l1_loss(curr_Q, calc_Q)

    def update(self, mems, actions, rewards):
        loss_l = []
        for mem, action, reward in zip(mems, actions, rewards):
            self.optimizer.zero_grad()

            loss = self.get_loss(mem, action, reward)
            loss_l.append(loss.item())
            loss.backward()

            nn.utils.clip_grad_norm(self.Q_func.parameters(), max_norm=self.grad_norm_clipping)
            self.optimizer.step()
            self.num_param_updates += 1

            if self.num_param_updates % self.target_update_freq == 0:
                self.target_Q_func.load_state_dict(self.Q_func.state_dict())
        return loss_l

if __name__ == '__main__':
    config = SMT_Config()
    critic = CRITIC(config)
    critic = critic.cuda()
    memories = [torch.randn(1, 200, 128) for i in range(4)]
    actions = [torch.randint(6, (1, 200)) for i in range(4)]
    rewards = [torch.randn(1, 200) for i in range(4)]
    loss_list = critic.update(memories, actions, rewards)

