import numpy as np
import torch
from torch import nn as nn, optim as optim

import torch.optim
from torch.distributions import Categorical
from torch.utils import data as torch_data
import torch.nn.functional as F

import config


def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data = torch.exp(m.weight) / torch.exp(m.weight).sum(0, keepdim=True)
        assert torch.allclose(m.weight.data.sum(0), torch.tensor(1.))
        # torch.nn.init.orthogonal_(m.weight)


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, h_dim):
        super(ActorCritic, self).__init__()

        self.v = torch.rand(observation_space)
        self.q = torch.rand((observation_space, action_space))
        pi = torch.ones((observation_space, action_space))
        self.pi = pi / pi.sum(1, keepdim=True)

    def policy(self, x):
        x = self.pi[x]
        return x

    def value(self, x):
        v = self.v[x]
        return v

    def q_value(self, x):
        v = self.q[x]
        return v


def get_grad_norm(parameters):
    return torch.norm(torch.cat([p.grad.flatten() for p in parameters]))


class PG:
    def __init__(self, observation_space, action_space, h_dim):
        self._agent = ActorCritic(observation_space, action_space, h_dim)
        self.data = []

    def get_model(self):
        return self._agent

    def policy(self, s):
        logits = self._agent.policy(s)
        pi = F.softmax(logits, dim=-1)
        return pi

    def put_data(self, transition):
        self.data.append(transition)

    def act(self, s):
        s = torch.from_numpy(s).long()
        probs = self.policy(s)
        action = torch.distributions.Categorical(probs=probs).sample().item()
        return action

    def make_batch(self):
        out = list(map(lambda x: torch.tensor(np.stack(x), dtype=torch.float32), list(zip(*self.data))))
        return out

    def train(self):
        # batch x  dim
        s, a, r, s_prime, done_mask = self.make_batch()
        td_stats = self._train_value(s, a, r, s_prime, done_mask)
        pi_stats = self._train_pi(s, a, r, s_prime, done_mask)
        return {**pi_stats, **td_stats}

    def _train_pi(self, s, a, r, s_prime, done):
        s = s.squeeze().long()
        a = a.long().squeeze()
        probs = self.policy(s)
        adv = (probs * self._agent.q_value(s)).sum(dim=1) - self._agent.value(s)
        new_pi = self._agent.pi[s, a] * torch.exp(- config.eta * adv)
        self._agent.pi[s, a] = new_pi
        kl = (probs - self.policy(s)).norm(1)
        return {"adv": adv.mean(), "kl": kl.mean()}

    def _train_value(self, s, a, r, s_prime, done):
        s_prime = s_prime.squeeze().long()
        s = s.squeeze().long()
        pi = self.policy(s_prime)
        a = a.long()
        q_td = r + config.gamma * (pi * self._agent.q_value(s_prime)).sum(dim=-1) - self._agent.q[s, a]
        self._agent.q[s, a] += config.v_lr * (q_td)
        td = r + config.gamma * self._agent.value(s_prime) - self._agent.v[s]
        self._agent.v[s] += config.v_lr * (td)

        return {"q_td": q_td.mean(), "td": td.mean()}


class PPO(PG):

    def policy(self, s):
        probs = self._agent.policy(s)
        pi = Categorical(probs=probs)
        return pi.probs

    def _train_pi(self, s, a, r, s_prime, done):
        s = s.squeeze().long()
        a = a.long().squeeze()
        pi_old = self._agent.policy(s)
        adv = self._agent.q[s, a] - self._agent.value(s)
        self._agent.pi[s, a] = self._agent.pi[s, a] * torch.exp(- config.eta * adv)
        self._agent.pi[s, a] /= self._agent.pi[s].sum(1, keepdim=True)
        kl = (pi_old - self._agent.pi[s]).norm(1)
        return {"adv": adv.mean(), "kl": kl.mean()}


class MirrorDescent(torch.optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            for i, param in enumerate(group['params']):
                w = param * torch.exp(- lr * param.grad)
                w /= w.sum(dim=0, keepdims=True)
                param.data = w
