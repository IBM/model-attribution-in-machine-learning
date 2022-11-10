# copied from https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py
# only changes involve keeping track of decoys, adding scanning states, and reduction of action space

from ppo import ActorCritic
from ppo import Memory
import torch
import torch.nn as nn

import numpy as np
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent():
    def __init__(self, input_dims=52, action_space=[i for i in range(150)], lr=0.002, betas=[0.9, 0.990], gamma=0.99, K_epochs=4, eps_clip=0.2, restore=False, ckpt=None, deterministic=False, training=True):

        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip

        self.K_epochs = K_epochs
        self.input_dims = input_dims
        self.restore = restore
        self.ckpt = ckpt
        self.deterministic = deterministic
        self.training = training
        # initialise
        self.set_initial_values(action_space=action_space)




    def get_action(self, observation, action_space=None):
        state = torch.FloatTensor(observation).to(device)
        action = self.old_policy.act(state, self.memory, deterministic=self.deterministic)
        action_ = self.action_space[action]
        return action_

    def store(self, reward, done):
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

    def clear_memory(self):
        self.memory.clear_memory()

    def train(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(self.memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs)).to(device).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = - torch.min(surr1, surr2)

            critic_loss = 0.5 * self.MSE_loss(rewards, state_values) - 0.01 * dist_entropy

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())



    def set_initial_values(self, action_space, observation=None):
        self.memory = Memory()
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        if self.restore:
            pretained_model = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)

        self.old_policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()
