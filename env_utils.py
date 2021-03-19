from typing import NamedTuple

import gym
import numpy as np
import torch

from ppo import _compute_return


class MiniGridWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MiniGridWrapper, self).__init__(env)
        # self.action_space = gym.spaces.Discrete(n=3)

        shape = 2 * env.unwrapped.width + 4
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1.,
            shape=(shape,),  # x,y,d, terminal
            dtype=np.float32
        )
        self.t = None
        self.max_steps = 200
        self.ep = 0
        self.reset()

    def step(self, action):
        # r = self.reward()
        _, _, d, info = super(MiniGridWrapper, self).step(action)
        r = 0
        s1 = self.observation()
        self.t += 1
        self.returns += r
        info = {"env/reward": r,
                "env/ep": self.ep,
                "env/avg_reward": self.returns / (self.t + 1),
                "env/returns": self.returns,
                "env/steps": self.t}
        if d:
            self.ep += 1
            r = 1.
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

    def compute_adv(self, gamma=0.99):
        s, a, r, s1, d = list(map(lambda x: torch.tensor(np.stack(x), dtype=torch.float32), list(zip(*self._data))))
        vs = []
        for t in range(len(r)):
            v = _compute_return(r[t:])
            vs.append(v)
        #vs.append(torch.tensor(0))
        vs = torch.stack(vs)
        adv = r[:-1] + gamma * vs[1:] - vs[:-1]
        return s, a, r, s1, d, adv

    def append(self, transition):
        self._data.append(transition)


class Transition(NamedTuple):
    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    next_state: torch.tensor
    done: torch.tensor
