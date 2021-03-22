import numpy as np
import torch
from torch import nn as nn, optim as optim
from torch.distributions import Categorical
from torch.utils import data as torch_data

import config


# def debug_grad(m, i, o):
#    print(m)
#    print(i)
#    print(o)
#

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.00)


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, h_dim):
        super(ActorCritic, self).__init__()

        self.v = nn.Sequential(nn.Linear(observation_space, h_dim),
                               nn.ReLU(),
                               nn.Linear(h_dim, h_dim),
                               # nn.Dropout(),
                               nn.ReLU(),
                               nn.Linear(h_dim, 1))
        self.pi = nn.Sequential(nn.Linear(observation_space, action_space))
        # @nn.ELU(),
        # @nn.Linear(h_dim, h_dim),
        # @nn.ELU(),
        #              nn.Linear(h_dim, action_space))
        self.pi.apply(init_weights)
        # self.pi.register_backward_hook(debug_grad)

    def policy(self, x):
        h = self.pi(x)
        return h  # nn.Softmax(-1)(h)

    def value(self, x):
        v = self.v(x)
        return v.squeeze()


def get_grad_norm(parameters):
    grad_norm = 0
    for p in parameters:
        grad_norm += p.grad.norm()
    return grad_norm


class PG:
    def __init__(self, observation_space, action_space, h_dim):
        self._agent = ActorCritic(observation_space, action_space, h_dim)
        self.pi_opt = optim.SGD(self._agent.pi.parameters(), lr=config.learning_rate)
        self.value_opt = optim.SGD(self._agent.v.parameters(), lr=config.learning_rate)
        # self.pi_opt = optim.lr_scheduler.ExponentialLR(self.pi_opt, gamma=config.lr_decay)
        self.data = []

    def get_model(self):
        return self._agent

    def put_data(self, transition):
        self.data.append(transition)

    def act(self, s):
        with torch.no_grad():
            probs = self._agent.policy(torch.from_numpy(s).float())
        action = Categorical(logits=probs).sample().item()
        return action

    def make_batch(self):
        out = list(map(lambda x: torch.tensor(np.stack(x), dtype=torch.float32), list(zip(*self.data))))
        return out

    def train(self):
        # batch x  dim
        s, a, r, s_prime, done_mask = self.make_batch()
        with torch.no_grad():
            probs_old = self._agent.policy(s)
        dataset = torch_data.TensorDataset(*(s, a, r, s_prime, done_mask, probs_old))
        data_loader = torch_data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

        td_stats = self._train_value(data_loader)

        pi_stats = self._train_pi(data_loader)

        return {**td_stats, **pi_stats}

    def _train_pi(self, data_loader):
        for i in range(config.opt_epochs):
            total_loss = 0
            total_kl = 0
            total_entropy = 0
            grad_norm = 0
            for (s, a, r, s_prime, done_mask, probs_old) in data_loader:
                with torch.no_grad():
                    delta = r + config.gamma * self._agent.value(s_prime) * done_mask - self._agent.value(s)
                logits = self._agent.policy(s)
                pi = torch.distributions.Categorical(logits=logits)
                loss = - (pi.log_prob(a) * delta).mean()
                pi_old = torch.distributions.Categorical(logits=probs_old)
                kl = torch.distributions.kl_divergence(pi_old, pi).mean()
                total_loss += loss
                total_kl += kl
                total_entropy += pi.entropy().mean()
                self.pi_opt.zero_grad()
                loss.backward()
                grad_norm = get_grad_norm(self._agent.pi.parameters())
                print(grad_norm)
                self.pi_opt.step()

        return {
            "train/kl": total_kl,
            "train/pi_loss": total_loss,
            "train/entropy": total_entropy,
            "train/grad_norm": grad_norm
        }

    def _train_value(self, data_loader):
        for _ in range(config.opt_epochs):
            total_loss = 0
            for (s, a, r, s_prime, done_mask, _) in data_loader:
                delta = r + config.gamma * self._agent.value(s_prime) * done_mask - self._agent.value(s)
                reg = l2_reg(self._agent.v.named_parameters())
                v_loss = 0.5 * (delta ** 2).mean()  # +  reg
                assert torch.isfinite(v_loss)
                self.value_opt.zero_grad()
                v_loss.backward()
                self.value_opt.step()
                total_loss += v_loss
        return {"train/v_loss": total_loss,
                "train/l2_reg": reg}


def l2_reg(params, reg=1e-3):
    total_norm = 0
    for name, p in params:
        if "bias" not in name:
            total_norm += p.norm()
    return reg * total_norm


class PPO(PG):
    def __init__(self, *args, **kwargs):
        super(PPO, self).__init__(*args, **kwargs)

    def _train_pi(self, data_loader):
        for i in range(config.opt_epochs):
            total_loss = 0
            total_kl = 0
            total_entropy = 0
            grad_norm = 0
            for (s, a, r, s_prime, done_mask, probs_old) in data_loader:
                with torch.no_grad():
                    delta = r + config.gamma * self._agent.value(s_prime) * done_mask - self._agent.value(s)
                pi_old = torch.distributions.Categorical(logits=probs_old)
                probs = self._agent.policy(s)
                pi = torch.distributions.Categorical(logits=probs)
                kl = torch.distributions.kl_divergence(pi_old, pi).mean()
                loss = - torch.exp(pi.log_prob(a) - pi_old.log_prob(a)) * delta  # + config.eta * kl
                total_loss += loss.mean()
                total_kl += kl
                total_entropy += pi.entropy().mean()
                self.pi_opt.zero_grad()
                total_loss.backward()
                grad_norm = get_grad_norm(self._agent.pi.parameters())
                assert torch.isfinite(grad_norm)
                self.pi_opt.step()

        return {
            "train/kl": total_kl,
            "train/pi_loss": total_loss,
            "train/entropy": total_entropy,
            "train/grad_norm": grad_norm,
        }

# ratio = torch.log(probs_old / pi).clamp_min(0.)
# kl = (pi * ratio).sum(dim=-1).mean()
# assert kl >= 0. and kl.isfinite()
# + config.eta * kl
# a = F.one_hot(a.long(), 7)
# p_norm = l2_reg(self._agent.pi.named_parameters(), 1.)
# probs = torch.exp(probs) / torch.exp(probs).sum(dim=1, keepdim=True)
# log_pi = (torch.log(probs) * a).sum(dim=-1)
# self.pi_opt.zero_grad()
