import os

import torch
from gym.wrappers import monitor
from gym_minigrid.envs import EmptyEnv5x5
from torch.distributions import Categorical


def generate_episode(env, policy):
    s = env.reset()
    d = False
    while not d:
        with torch.no_grad():
            probs = policy.policy(torch.from_numpy(s).float())
        action = Categorical(probs=probs).sample().item()
        s1, r, d, info = env.step(action)
        # env.render()
        s = s1
    env.close()


def eval_policy(log_dir):
    from env_utils import MiniGridWrapper
    # env = FourRoomsEnv(goal_pos=(12, 16))
    env = EmptyEnv5x5()
    env = MiniGridWrapper(env)
    policy = torch.load(log_dir)
    env = monitor.Monitor(env, os.path.dirname(log_dir), force=True)
    generate_episode(env, policy)
    env.close()
    del env
    del policy


if __name__ == '__main__':
    log_dir = "runs/objects/test_Mar19_12-05-01/agent-100.pt"
    eval_policy(log_dir)
