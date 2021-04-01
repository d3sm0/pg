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
        m.bias.data = torch.ones_like(m.bias.data)
        m.bias.data /= m.bias.data.sum()


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, h_dim):
        super(ActorCritic, self).__init__()

        self.v = torch.rand(observation_space)
        self.q = torch.rand((observation_space, action_space))
        pi = torch.ones((action_space,))
        pi = F.softmax(pi, 0)
        self.pi = nn.Parameter(pi)
        # self.pi = nn.Linear(observation_space, action_space, bias=True)

    def policy(self, x):
        # import torch.nn.functional as F
        # x = F.one_hot(x, self.v.shape[0]).float()
        return self.pi

    def value(self, x):
        v = self.v[x]
        return v


def get_grad_norm(parameters):
    return torch.norm(torch.cat([p.grad.flatten() for p in parameters]))


class PG:
    def __init__(self, observation_space, action_space, h_dim):
        self._agent = ActorCritic(observation_space, action_space, h_dim)
        self.data = []
        self.optim = torch.optim.SGD(params=self._agent.parameters(), lr=config.pi_lr)

    def get_model(self):
        return self._agent

    def policy(self, s):
        logits = self._agent.policy(s)
        pi = F.softmax(logits, dim=-1)
        return pi

    def put_data(self, transition):
        self.data.append(transition)

    def act(self, s):
        with torch.no_grad():
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
        s_prime = s_prime.long().squeeze()
        adv = r + config.gamma * self._agent.value(s_prime) - self._agent.value(s)
        with torch.no_grad():
            probs = self.policy(s)
            pi_old = torch.distributions.Categorical(probs=probs)
        # self._agent.pi = self._agent.pi - self._agent.pi.max(1, keepdims=True)[0]
        for _ in range(config.opt_epochs):
            self.optim.zero_grad()
            pi = torch.distributions.Categorical(probs=self.policy(s))
            loss = - pi.log_prob(a) * adv + config.eta * torch.distributions.kl_divergence(pi_old, pi)
            loss.mean().backward()
            self.optim.step()

        kl = (probs - self.policy(s)).norm(1)
        return {"adv": adv.mean(), "kl": kl.mean()}

    def _train_value(self, s, a, r, s_prime, done):
        s_prime = s_prime.squeeze().long()
        s = s.squeeze().long()
        td = r + config.gamma * self._agent.value(s_prime) - self._agent.v[s]
        self._agent.v[s] += config.v_lr * td

        return {"td": td.mean()}


class PPO(PG):

    def _train_pi(self, s, a, r, s_prime, done):
        s = s.squeeze().long()
        s_prime = s_prime.squeeze().long()
        a = a.long().squeeze()
        with torch.no_grad():
            pi_old = self.policy(s)
            pi_old = torch.distributions.Categorical(probs=pi_old)
        adv = r + config.gamma * self._agent.value(s_prime) - self._agent.value(s)

        #for _ in range(config.opt_epochs):
        #    self.optim.zero_grad()
        #    pi = torch.distributions.Categorical(probs=self.policy(s))
        #    loss = - torch.exp(
        #        pi.log_prob(a) - pi_old.log_prob(a) * adv + config.eta * torch.distributions.kl_divergence(pi_old, pi))
        #    loss.mean().backward()
        #    self.optim.step()
        w = self._agent.pi.data
        w[a] = w[a] + config.eta * adv  ## w[a] * torch.exp(config.eta * adv)
        # w[a] /= w.sum()
        # self._agent.pi.data = w
        kl = (pi_old.probs - torch.softmax(w, 0)).norm(1)
        # kl = (pi_old.probs - pi.probs).norm(1)
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
