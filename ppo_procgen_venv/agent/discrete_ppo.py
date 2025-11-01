import numpy as np

import torch
import torch.nn as nn
from Cython import returns

from agent.replay_buffer import RolloutBuffer, Sampler
from agent.models import ActorCritic

"""
The code is derived from https://github.com/nikhilbarhate99/PPO-PyTorch
"""

class PPO:
    def __init__(self, state_dim , action_dim, actor_critic_model, lr, gamma, K_epochs, eps_clip, use_gae, gae_lambda, mini_batch_size, use_clipped_value_loss=True, device='cpu'):

        self.device = device

        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.use_clipped_value_loss = use_clipped_value_loss
        self.num_envs = state_dim[0]
        self.mini_batch_size = mini_batch_size

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim[1:], action_dim, base=actor_critic_model).to(
            self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        self.policy_old = ActorCritic(state_dim[1:], action_dim, base=actor_critic_model).to(
            self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()

        self.MseLoss = nn.MSELoss()


    def select_action(self, state):
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state, None, None, deterministic=False)

        return action, action_logprob, state_val

    def calculate_return(self, next_value):
        # Monte Carlo estimate of returns
        batch_size = len(self.buffer.rewards)
        returns = torch.empty((batch_size, self.num_envs), dtype=torch.float32).to(self.device)
        if self.use_gae:
            gae = 0
            for step in reversed(range(batch_size)):
                cur_value = self.buffer.state_values[step]
                delta = self.buffer.rewards[step] + self.gamma * next_value * (1 - self.buffer.is_terminals[step]) - cur_value
                gae = delta + self.gamma * self.gae_lambda * (1 - self.buffer.is_terminals[step]) * gae
                returns[step] = gae + cur_value
                next_value = cur_value
        else:
            for step in reversed(range(batch_size)):
                returns[step] = next_value * self.gamma * (1 - self.buffer.is_terminals[step]) + self.buffer.rewards[step]
                next_value = returns[step]

        return returns.reshape(-1)


    def update(self):
        with torch.no_grad():
            next_value = self.policy_old.get_value(self.buffer.next_states[-1], None, None).detach()

        returns = self.calculate_return(next_value)

        # convert list to tensor
        old_states = torch.concat(self.buffer.states, dim=0).detach().to(self.device)
        old_actions = torch.concat(self.buffer.actions, dim=0).detach().to(self.device)
        old_logprobs = torch.concat(self.buffer.logprobs, dim=0).detach().to(self.device)
        old_state_values = torch.concat(self.buffer.state_values, dim=0).detach().to(self.device)

        # rewards -> (steps)
        # old_states -> (steps, obs_dim)
        # old_state_values -> (steps)

        # calculate advantages
        advantages = returns.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-6)

        # Optimize policy for K epochs
        batch_size = returns.size(0)

        sampler = Sampler(batch_size, self.mini_batch_size, old_states, old_actions, old_logprobs, old_state_values, returns, advantages)
        for _ in range(self.K_epochs):
            for sample in sampler.dataloader():
                batch_old_states, batch_old_actions, batch_old_logprobs, batch_old_state_values, batch_returns, batch_advantages = sample

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate_actions(batch_old_states, None, None, batch_old_actions)

                ratios = torch.exp(logprobs - batch_old_logprobs.detach())
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                if self.use_clipped_value_loss:
                    value_pred_clipped = batch_old_state_values + (state_values - batch_old_state_values).clamp(-self.eps_clip, self.eps_clip)
                    value_losses = (state_values - batch_returns).pow(2)
                    value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
                    value_loss = (0.5 * torch.max(value_losses, value_losses_clipped).mean())
                else:
                    value_loss = (0.5 * (state_values - batch_returns).pow(2).mean())

                loss = -torch.min(surr1, surr2).mean() + 0.5 * value_loss - 0.01 * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()

                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

                self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
