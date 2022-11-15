import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import logging
import time
from copy import deepcopy
from model.general import MLP
from model.resnet import ResNet18
from utils.utils import get_time

class Actor(nn.Module):
    def __init__(self, f_dim, a_dim):
        super(Actor, self).__init__()
        self.seq = MLP((f_dim, 128, 128, 128, 128, a_dim), act=nn.ReLU, act_out=nn.Identity)

    def forward(self, feature):
        """
        给出当前状态特征 feature，返回选择的动作 action
        """
        out = self.seq(feature)
        dist = D.Bernoulli(logits=out)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob


class Critic(nn.Module):
    def __init__(self, f_dim, a_dim):
        super(Critic, self).__init__()
        self.seq = MLP((f_dim + a_dim, 128, 128, 128, 128, 1), act=nn.ReLU, act_out=nn.Identity)

    def forward(self, feature, action):
        """
        给出状态特征 feature 和动作 action，返回动作价值 Q
        """
        q = self.seq(torch.cat([feature, action], dim=-1))
        return q


class AC(nn.Module):
    def __init__(self, f_dim, a_dim):
        super(AC, self).__init__()
        self.actor = Actor(f_dim, a_dim)
        self.critic1 = Critic(f_dim, a_dim)
        self.critic2 = Critic(f_dim, a_dim)


class SAC(nn.Module):
    def __init__(self, ARGS):
        super(SAC, self).__init__()
        self.ARGS = ARGS
        self.preprocess_canvas = ResNet18(num_classes=ARGS.F_DIM)
        self.preprocess_vector = MLP((2, 128, 128, ARGS.F_DIM), act=nn.ReLU, act_out=nn.ReLU)
        self.ac = AC(ARGS.F_DIM, ARGS.A_DIM)
        self.ac_ = deepcopy(self.ac)
        for p in self.ac_.parameters():
            p.requires_grad = False
        self.optimizer_0 = torch.optim.Adam([
            {'params': self.preprocess_canvas.parameters()},
            {'params': self.preprocess_vector.parameters()},
            ], lr=ARGS.LR_0, weight_decay=1e-4)
        self.optimizer_a = torch.optim.Adam(self.ac.actor.parameters(), lr=ARGS.LR_A, weight_decay=1e-4)
        self.optimizer_c = torch.optim.Adam([
            {'params': self.ac.critic1.parameters()},
            {'params': self.ac.critic2.parameters()},
        ], lr=ARGS.LR_A, weight_decay=1e-4)
        self.memory = dict(
            state=torch.zeros((ARGS.MEMORY_CAPACITY, 3 * ARGS.FIGSIZE * ARGS.FIGSIZE + 2), device=ARGS.DEVICE),
            action=torch.zeros((ARGS.MEMORY_CAPACITY, ARGS.A_DIM), device=ARGS.DEVICE),
            state_=torch.zeros((ARGS.MEMORY_CAPACITY, 3 * ARGS.FIGSIZE * ARGS.FIGSIZE + 2), device=ARGS.DEVICE),
            reward=torch.zeros((ARGS.MEMORY_CAPACITY, 1), device=ARGS.DEVICE),
            done=torch.zeros((ARGS.MEMORY_CAPACITY, 1), device=ARGS.DEVICE)
        )
        self.pointer = 0

    def forward(self, state):
        """
        根据当前状态产生下一步动作
        canvas 表示游戏界面，大小为 128*128，vector=(player, bomb) 为残机数和炸弹数
        """
        with torch.no_grad():
            feature = self._get_feature(state)
            action, _ = self.ac.actor(feature)
        return action

    def _get_feature(self, state):
        """
        将 canvas 和 vector 转换为 self.ARGS.F_DIM 维的输入
        """
        canvas = state[:, :-2].view(-1, 3, self.ARGS.FIGSIZE, self.ARGS.FIGSIZE)
        vector = state[:, -2:]
        feature = self.preprocess_canvas(canvas) + self.preprocess_vector(vector)
        return feature

    def learn(self):
        if self.pointer < self.ARGS.BATCH_SIZE: 
            return
        indices = np.random.choice(min(self.ARGS.MEMORY_CAPACITY, self.pointer), size=self.ARGS.BATCH_SIZE)
        state = self.memory['state'][indices, :]
        action = self.memory['action'][indices, :]
        state_ = self.memory['state_'][indices, :]
        reward = self.memory['reward'][indices, :]

        feature = self._get_feature(state)
        feature_ = self._get_feature(state_)

        q1 = self.ac.critic1(feature, action)
        q2 = self.ac.critic2(feature, action)
        with torch.no_grad():
            action_, log_prob = self.ac.actor(feature_)
            q1_ = self.ac_.critic1(feature_, action_)
            q2_ = self.ac_.critic2(feature_, action_)
            q_ = torch.min(q1_, q2_)
            q_target = reward + self.ARGS.GAMMA * (q_ - self.ARGS.ALPHA * log_prob)
        td_error1 = (q_target - q1) ** 2
        td_error2 = (q_target - q2) ** 2
        loss_c = torch.mean(td_error1 + td_error2)

        for p in self.ac.critic1.parameters():
            p.requires_grad = False
        for p in self.ac.critic2.parameters():
            p.requires_grad = False

        a, log_prob = self.ac.actor(feature)
        q1 = self.ac.critic1(feature, a)
        q2 = self.ac.critic2(feature, a)
        q = torch.min(q1, q2)
        loss_a = -torch.mean(q - self.ARGS.ALPHA * log_prob)
        
        loss = loss_a + loss_c
        self.optimizer_a.zero_grad()
        self.optimizer_c.zero_grad()
        loss.backward()
        self.optimizer_c.step()
        self.optimizer_a.step()

        for p in self.ac.critic1.parameters():
            p.requires_grad = True
        for p in self.ac.critic2.parameters():
            p.requires_grad = True

    def soft_replace(self):
        for t, e in zip(self.ac_.parameters(), self.ac.parameters()):
            t.data = (1 - self.ARGS.TAU) * t.data + self.ARGS.TAU * e.data

    def store_transition(self, state, action, reward, state_, done):
        for idx in range(state.shape[0]):
            self.memory['state'][self.pointer % self.ARGS.MEMORY_CAPACITY, :] = state[idx, :].detach()
            self.memory['action'][self.pointer % self.ARGS.MEMORY_CAPACITY, :] = action[idx, :].detach()
            self.memory['reward'][self.pointer % self.ARGS.MEMORY_CAPACITY, :] = reward[idx].detach()
            self.memory['state_'][self.pointer % self.ARGS.MEMORY_CAPACITY, :] = state_[idx, :].detach()
            self.memory['done'][self.pointer % self.ARGS.MEMORY_CAPACITY, :] = done[idx, :].detach()
            self.pointer += 1
            if self.pointer == self.ARGS.MEMORY_CAPACITY:
                logging.info('回放池满，开始训练')
            if self.pointer % self.ARGS.MEMORY_CAPACITY == 0:
                with open(f'checkpoint/memory/{get_time()}.bin', 'wb') as f:
                    pickle.dump(self.memory, f)
p