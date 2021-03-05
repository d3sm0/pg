import itertools
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
from gym_minigrid.envs import FourRoomsEnv
from torch.distributions import Categorical

# Hyperparameters
import config


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

    def make_batch(self):
        out = list(map(lambda x: torch.tensor(np.stack(x), dtype=torch.float32), list(zip(*self.data))))
        return out

    def train_net(self):
        # batch x  dim
        s, a, r, s_prime, done_mask = self.make_batch()
        with torch.no_grad():
            pi_old = self.pi(s)
        dataset = torch_data.TensorDataset(*(s, a, r, s_prime, done_mask, pi_old))
        data_loader = torch_data.DataLoader(dataset, batch_size=config.batch_size)

        total_loss = 0
        total_kl = 0
        total_td = 0
        total_entropy = 0.
        for i in range(config.opt_epochs):
            for (s, a, r, s_prime, done_mask, _) in data_loader:
                td_target = r + config.gamma * self.v(s_prime).detach() * done_mask
                delta = td_target - self.v(s)
                v_loss = 0.5 * (delta ** 2).mean()
                assert torch.isfinite(v_loss)
                self.value_opt.zero_grad()
                v_loss.backward()
                self.value_opt.step()
                total_td += v_loss

        for i in range(config.opt_epochs):
            for (s, a, r, s_prime, done_mask, pi_old) in data_loader:
                with torch.no_grad():
                    delta = r + config.gamma * self.v(s_prime).detach() * done_mask - self.v(s)
                prob = self.pi(s)
                with torch.no_grad():
                    pi_old = torch.distributions.Categorical(probs=pi_old)
                pi = torch.distributions.Categorical(probs=prob)
                entropy = pi.entropy().mean()
                total_entropy += entropy
                kl = torch.distributions.kl_divergence(pi_old, pi).mean()
                assert kl.isfinite().all()
                total_kl += kl
                if config.as_ppo:
                    ratio = torch.exp(pi.log_prob(a) - pi_old.log_prob(a))
                    surr1 = ratio * delta
                    surr2 = torch.clamp(ratio, 1 - config.eps_clip, 1 + config.eps_clip) * delta
                    loss = -torch.min(surr1, surr2).mean()
                else:
                    loss = - (pi.log_prob(a) * delta).mean() + config.eta * kl

            self.pi_opt.zero_grad()
            assert torch.isfinite(loss)
            loss.backward()
            self.pi_opt.step()
            total_loss += loss

        return {
            "train/kl": total_kl / config.opt_epochs,
            "train/v_loss": total_td / config.opt_epochs,
            "train/pi_loss": total_loss / config.opt_epochs,
            "train/entropy": total_entropy / config.opt_epochs
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
        self.reset()

    def step(self, action):
        r = self.reward()
        s1, _, d, info = super(MiniGridWrapper, self).step(action)
        s1 = self.observation()
        self.t += 1
        self.returns += r
        info = {"env/reward": r,
                "env/avg_reward": self.returns / (self.t + 1),
                "env/returns": self.returns,
                "env/steps": self.t}
        if self.t >= self.max_steps:
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


def gather_trajectory(env, model, horizon):
    s = env.get_state()
    info = {}
    for t in range(horizon):
        logits = model.pi(torch.from_numpy(s).float())
        action = Categorical(probs=logits).sample().item()
        s_prime, r, done, info = env.step(action)
        model.put_data((s, action, r, s_prime, 1 - done))
        s = s_prime
        if done:
            s = env.reset()
    return info


def main():
    env = FourRoomsEnv(goal_pos=(12, 16))
    torch.manual_seed(config.seed)
    env.seed(config.seed)
    env = MiniGridWrapper(env)
    model = PPO(action_space=env.action_space.n, observation_space=env.observation_space.shape[0], h_dim=config.h_dim)
    dtm = datetime.now().strftime("%d-%H-%M-%S-%f")
    # writer = tb.SummaryWriter(log_dir=f"logs/{dtm}_as_ppo:{config.as_ppo}")
    for global_step in itertools.count():
        info = gather_trajectory(env, model, config.horizon)
        config.tb.add_scalar("return", info["env/returns"], global_step=global_step)
        losses = model.train_net()
        model.data.clear()
        for k, v in losses.items():
            config.tb.add_scalar(k, v, global_step=global_step)
        # if global_step % config.save_interval == 0:
        #    torch.save(model, f"{writer.log_dir}/latest.pt")
        if (global_step * config.horizon) == config.max_steps:
            break

    env.close()


if __name__ == '__main__':
    main()
