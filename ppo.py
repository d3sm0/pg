import copy

import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
import config


def get_grad_norm(parameters):
    return torch.norm(torch.cat([p.grad.flatten() for p in parameters if p.grad is not None]))


def _compute_return(rewards, gamma=0.99):
    value = 0
    for t in range(len(rewards)):
        value += gamma ** t * rewards[t]
    return value


def _make_adv(r, done_mask, gamma=0.99):
    idx, = torch.where(done_mask == 0)
    advs = []
    t_start = 0
    for t_final in idx:
        v = _compute_return(r[t_start:t_final + 1], gamma=gamma)
        for t in range(t_start, t_final + 1):
            reward_slice = r[t + 1:t_final]
            v_next = _compute_return(reward_slice, gamma=gamma)
            adv = r[t] + gamma * v_next - v
            v = v_next
            advs.append(adv)
        t_start = t_final
    advs = torch.stack(advs)
    return advs


class PPO(nn.Module):
    def __init__(self, observation_space, action_space, h_dim):
        super(PPO, self).__init__()
        self.data = []

        self.fc_v = nn.Sequential(nn.Linear(observation_space, h_dim),
                                  nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU(),
                                  nn.Linear(h_dim, 1))
        self.fc_pi = nn.Sequential(nn.Linear(observation_space, h_dim), nn.ReLU(),
                                   nn.Linear(h_dim, h_dim), nn.ReLU(),
                                   nn.Linear(h_dim, action_space))
        self.pi_opt = optim.SGD(self.fc_pi.parameters(), lr=config.learning_rate)
        self.value_opt = optim.SGD(self.fc_v.parameters(), lr=config.learning_rate)

    def pi(self, x):
        x = self.fc_pi(x)
        return nn.Softmax(-1)(x)

    def v(self, x):
        v = self.fc_v(x)
        return v.squeeze()

    def train_net(self, batch):
        # batch x  dim
        # s, a, r, s_prime, done_mask = self.make_batch()
        old_model = copy.deepcopy(self)
        for epoch in range(config.opt_epochs):
            self.pi_opt.zero_grad()
            total_loss = 0
            total_kl = 0
            for trajectory in batch:
                s, a, r, s1, d, adv = trajectory.compute_adv()
                with torch.no_grad():
                    pi_old = torch.distributions.Categorical(probs=old_model.pi(s))
                prob = self.pi(s)
                pi = torch.distributions.Categorical(probs=prob)
                # entropy = pi.entropy().mean()
                # total_entropy += entropy
                kl = torch.distributions.kl_divergence(pi_old, pi).mean()
                total_kl += kl
                assert kl.isfinite().all()
                # total_kl += kl
                # if config.agent == "ppo":
                #    ratio = torch.exp(pi.log_prob(a) - pi_old.log_prob(a))
                #    surr1 = ratio * delta
                #    surr2 = torch.clamp(ratio, 1 - config.eps_clip, 1 + config.eps_clip) * delta
                #    loss = -torch.min(surr1, surr2).mean()
                # else:
                loss = - (pi.log_prob(a) * adv).mean() + config.eta * kl
                total_loss += loss
            (total_loss/len(batch)).backward()
            grad_norm = get_grad_norm(self.parameters())
            self.pi_opt.step()
            # assert torch.isfinite(loss)
            # loss.backward()
            # grad_norm += get_grad_norm(self.parameters())
            # total_loss += loss

        return {
            "train/grad_norm": grad_norm / len(batch),
            "train/kl": total_kl / len(batch),
            # "train/v_loss": total_td / config.opt_epochs,
            "train/pi_loss": total_loss / len(batch)
            # "train/entropy": total_entropy / config.opt_epochs
        }

