import gym
import numpy as np


class MiniGridWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MiniGridWrapper, self).__init__(env)
        #self.action_space = gym.spaces.Discrete(n=10)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(2 * env.unwrapped.width + 4,),
            dtype=np.float32
        )
        self.t = None
        self.max_steps = 200
        self.reset()

    def step(self, action):
        #action = min(action, self.unwrapped.action_space.n - 1)
        s1, _, d, info = super(MiniGridWrapper, self).step(action)
        r = self.reward()
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
        goal_x, goal_y = self.unwrapped.grid.width - 2, self.unwrapped.grid.height - 2
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
