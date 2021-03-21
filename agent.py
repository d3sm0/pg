import numpy as np
import torch
from torch import nn as nn, optim as optim
from torch.distributions import Categorical
from torch.utils import data as torch_data

import config


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, h_dim):
        super(ActorCritic, self).__init__()

        self.v = nn.Sequential(nn.Linear(observation_space, h_dim),
                               nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU(),
                               nn.Linear(h_dim, 1))
        self.pi = nn.Sequential(nn.Linear(observation_space, h_dim), nn.ReLU(),
                                nn.Linear(h_dim, h_dim), nn.ReLU(),
                                nn.Linear(h_dim, action_space))

    def policy(self, x):
        x = self.pi(x)
        return nn.Softmax(-1)(x)

    def value(self, x):
        v = self.v(x)
        return v.squeeze()


def get_grad_norm(parameters):
    return torch.norm(torch.cat([p.grad.flatten() for p in parameters]))


class PG:
    def __init__(self, observation_space, action_space, h_dim):
        self._agent = ActorCritic(observation_space, action_space, h_dim)
        self.pi_opt = optim.SGD(self._agent.pi.parameters(), lr=config.learning_rate)
        self.value_opt = optim.SGD(self._agent.v.parameters(), lr=config.learning_rate)
        self.data = []

    def get_model(self):
        return self._agent

    def put_data(self, transition):
        self.data.append(transition)

    def act(self, s):
        with torch.no_grad():
            probs = self._agent.policy(torch.from_numpy(s).float())
        action = Categorical(probs=probs).sample().item()
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
        data_loader = torch_data.DataLoader(dataset, batch_size=config.batch_size)

        td_stats = self._train_value(data_loader)

        pi_stats = self._train_pi(data_loader)

        return {**pi_stats, **td_stats}

    def _train_pi(self, data_loader):
        for i in range(config.opt_epochs):
            total_loss = 0
            total_kl = 0
            total_entropy = 0
            for (s, a, r, s_prime, done_mask, probs_old) in data_loader:
                with torch.no_grad():
                    delta = r + config.gamma * self._agent.value(s_prime).detach() * done_mask  # - self._agent.value(s)
                pi_old = torch.distributions.Categorical(probs=probs_old)
                pi = torch.distributions.Categorical(probs=self._agent.policy(s))
                kl = torch.distributions.kl_divergence(pi_old, pi).mean()
                assert kl.isfinite().all()
                loss = - (pi.log_prob(a) * delta).mean() + config.eta * kl
                total_loss += loss
                total_kl += kl
                total_entropy += pi.entropy().mean()

                # if config.agent == "ppo":
                #    ratio = torch.exp(pi.log_prob(a) - pi_old.log_prob(a))
                #    surr1 = ratio * delta
                #    surr2 = torch.clamp(ratio, 1 - config.eps_clip, 1 + config.eps_clip) * delta
                #    loss = -torch.min(surr1, surr2).mean()
                # else:
            self.pi_opt.zero_grad()
            total_loss.backward()
            grad_norm = get_grad_norm(self._agent.pi.parameters())
            assert torch.isfinite(grad_norm)
            self.pi_opt.step()

        return {
            "train/kl": total_kl,
            "train/pi_loss": total_loss,
            "train/entropy": total_entropy,
        }

    def _train_value(self, data_loader):
        for _ in range(config.opt_epochs):
            total_loss = 0
            for (s, a, r, s_prime, done_mask, _) in data_loader:
                delta = r + config.gamma * self._agent.value(s_prime).detach() * done_mask - self._agent.value(s)
                v_loss = 0.5 * (delta ** 2).mean()
                assert torch.isfinite(v_loss)
                self.value_opt.zero_grad()
                v_loss.backward()
                self.value_opt.step()
                total_loss += v_loss
        return {"train/v_loss": total_loss}


class PPO(PG):
    def __init__(self, *args, **kwargs):
        super(PPO, self).__init__(*args, **kwargs)

    def _train_pi(self, data_loader):
        for i in range(config.opt_epochs):
            total_loss = 0
            total_kl = 0
            total_entropy = 0
            for (s, a, r, s_prime, done_mask, probs_old) in data_loader:
                with torch.no_grad():
                    delta = r + config.gamma * self._agent.value(s_prime).detach() * done_mask # - self._agent.value(s)
                pi_old = torch.distributions.Categorical(probs=probs_old)
                pi = torch.distributions.Categorical(probs=self._agent.policy(s))
                kl = torch.distributions.kl_divergence(pi_old, pi)
                assert kl.isfinite().all()
                loss = - torch.exp(pi.log_prob(a) - pi_old.log_prob(a)) * delta + config.eta * kl
                # surr1 = ratio * delta
                # surr2 = torch.clamp(ratio, 1 - config.eps_clip, 1 + config.eps_clip) * delta
                # loss = -torch.min(surr1, surr2).mean()
                total_loss += loss.mean()
                total_kl += kl.mean()
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
        }
