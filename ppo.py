import copy
import itertools
from typing import NamedTuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym_minigrid.envs import FourRoomsEnv
from torch.distributions import Categorical

# Hyperparameters
import config
from eval_policy import eval_policy


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

    def put_data(self, transition):
        self.data.append(transition)

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
            total_loss.backward()
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


class MiniGridWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MiniGridWrapper, self).__init__(env)
        # self.action_space = gym.spaces.Discrete(n=3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=19,
            shape=(42,),  # x,y,d, terminal
            dtype=np.float32
        )
        self.t = None
        self.max_steps = 200
        self.ep = 0
        self.reset()

    def step(self, action):
        r = self.reward()
        _, _, d, info = super(MiniGridWrapper, self).step(action)
        s1 = self.observation()
        self.t += 1
        self.returns += r
        info = {"env/reward": r,
                "env/ep": self.ep,
                "env/avg_reward": self.returns / (self.t + 1),
                "env/returns": self.returns,
                "env/steps": self.t}
        if self.t >= self.max_steps:
            self.ep += 1
            d = True
        return s1, r, d, info

    def reward(self):
        x, y = self.env.agent_pos
        goal_x, goal_y = self.env._goal_default_pos
        r = - (abs(x - goal_x) + abs(y - goal_y)) / (goal_x + goal_y)
        return r

    def reset(self, **kwargs):
        s = super(MiniGridWrapper, self).reset()
        self.t = 0
        self.returns = 0
        return self.observation()

    def observation(self):
        x, y = self.env.agent_pos
        z = self.env.agent_dir
        _z = np.zeros(shape=(4,))
        _z[z] = 1
        _x = np.zeros(shape=self.env.grid.width)
        _y = np.zeros(shape=self.env.grid.width)
        _x[x] = 1
        _y[y] = 1
        _obs = np.concatenate([_x, _y, _z])
        return _obs

    def get_state(self):
        return self.observation()


class Trajectory:
    def __init__(self):
        self._data = []

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return f"N:{self.__len__()}"

    def compute_adv(self):
        s, a, r, s1, d = list(map(lambda x: torch.tensor(np.stack(x), dtype=torch.float32), list(zip(*self._data))))
        vs = [torch.tensor(0.)]
        for t in range(len(r)):
            v = _compute_return(r[t:])
            vs.append(v)
        vs = torch.stack(vs)
        adv = r + vs[1:] - vs[:-1]
        return s, a, r, s1, d, adv

    def append(self, transition):
        self._data.append(transition)


class Transition(NamedTuple):
    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    next_state: torch.tensor
    done: torch.tensor


def gather_trajectories(env, model, n_trajectories):
    batch = []
    info = {}
    for ep in range(n_trajectories):
        trajectory, info = _gather_trajectory(env, model)
        # trajectory.compute_adv()
        batch.append(trajectory)
    return batch, info


def _gather_trajectory(env, model):
    s = env.reset()
    trajectory = Trajectory()
    while True:
        logits = model.pi(torch.from_numpy(s).float())
        action = Categorical(probs=logits).sample().item()
        s_prime, r, done, info = env.step(action)
        trajectory.append(Transition(s, action, r, s_prime, 1 - done))
        s = s_prime
        if done:
            break
    return trajectory, info


def main():
    env = FourRoomsEnv(goal_pos=(12, 16))
    torch.manual_seed(config.seed)
    print(config.agent)
    env.seed(config.seed)
    env = MiniGridWrapper(env)
    model = PPO(action_space=env.action_space.n, observation_space=env.observation_space.shape[0], h_dim=config.h_dim)
    # dtm = datetime.now().strftime("%d-%H-%M-%S-%f")
    # writer = tb.SummaryWriter(log_dir=f"logs/{dtm}_as_ppo:{config.as_ppo}")
    for global_step in itertools.count():
        batch, info = gather_trajectories(env, model, config.horizon)
        config.tb.add_scalar("return", info["env/returns"], global_step=global_step)
        losses = model.train_net(batch)
        model.data.clear()
        for k, v in losses.items():
            config.tb.add_scalar(k, v, global_step=global_step)
        if global_step % config.save_interval == 0:
            log_dir = config.tb.add_object('model', model, global_step=global_step)
            # eval_policy(log_dir=log_dir)
        if (global_step * config.horizon) > config.max_steps:
            break

    env.close()


if __name__ == '__main__':
    main()
