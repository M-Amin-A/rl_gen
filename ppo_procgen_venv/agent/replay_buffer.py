import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.next_states = []
        self.state_values = []
        self.is_terminals = []


    def insert(self, states, actions, logprobs, rewards, next_states, state_values, is_terminals):
        self.states.append(states)
        self.actions.append(actions)
        self.logprobs.append(logprobs)
        self.rewards.append(rewards)
        self.next_states.append(next_states)
        self.state_values.append(state_values)
        self.is_terminals.append(is_terminals)


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.state_values[:]
        del self.is_terminals[:]



class Sampler(object):
    def __init__(self, batch_size, mini_batch_size, *args):
        self.args = args
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

        for arg in args:
            assert isinstance(arg, torch.Tensor)
            assert arg.dim() >= 1
            assert arg.size(0) == batch_size


    def dataloader(self):
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.batch_size)),
            self.mini_batch_size,
            drop_last=True)

        for indices in sampler:
            yield [arg[indices] for arg in self.args]
