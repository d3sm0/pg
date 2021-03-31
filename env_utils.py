import gym
import numpy as np


class StatisticsWrapper(gym.Wrapper):
    def __init__(self, env):
        super(StatisticsWrapper, self).__init__(env)
        self.t = None
        self.max_steps = 200
        self.reset()

    def step(self, action):
        s, r, d, info = super(StatisticsWrapper, self).step(action)
        self.t += 1
        self.returns += r
        info = {"env/reward": r,
                "env/avg_reward": self.returns / (self.t + 1),
                "env/returns": self.returns,
                "env/steps": self.t}
        if self.t >= self.max_steps:
            d = True
        return s, r, d, info

    def reset(self):
        self.t = 0
        self.returns = 0
        return super().reset()

    def get_state(self):
        return self.unwrapped.get_state()


class MiniGridWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MiniGridWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(n=3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1,),  # * env.unwrapped.width + 4,),
            dtype=np.float32
        )
        self._state_to_idx, self._idx_to_state = enumerate_state_space(env)
        self.n_states = len(self._state_to_idx.keys())
        self.reset()

    def step(self, action):
        s1, _, d, info = super(MiniGridWrapper, self).step(action)
        r = self.reward()
        s1 = self.observation()
        return s1, r, d, info

    def reward(self):
        x, y = self.env.agent_pos
        goal_x, goal_y = self.unwrapped.grid.width - 2, self.unwrapped.grid.height - 2
        r = - (abs(x - goal_x) + abs(y - goal_y)) / (goal_x + goal_y)
        return r

    def reset(self, **kwargs):
        s = super(MiniGridWrapper, self).reset()
        return self.observation()

    def observation(self):
        x, y = self.env.agent_pos
        z = self.env.agent_dir
        obs = self._state_to_idx[(z, x, y)]
        return np.ones((1,)) * obs

    def get_state(self):
        return self.observation()


def enumerate_state_space(env):
    state_dict = {}
    n_states = 0
    n_to_state = {}
    for w in range(env.unwrapped.width):
        for h in range(env.unwrapped.width):
            for d in range(4):
                state_dict[(d, w, h)] = n_states
                n_to_state[n_states] = (d, w, h)
                n_states += 1
    return state_dict, n_to_state
