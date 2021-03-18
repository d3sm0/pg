import os

import torch
from gym.wrappers import monitor
from gym_minigrid.envs import FourRoomsEnv
from torch.distributions import Categorical



def generate_episode(env, policy):
    s = env.reset()
    d = False
    while not d:
        logits = policy.pi(torch.from_numpy(s).float())
        action = Categorical(probs=logits).sample().item()
        s1, r, d, info = env.step(action)
        # env.render()
        s = s1
    env.close()


def eval_policy(log_dir):
    from ppo import MiniGridWrapper
    env = FourRoomsEnv(goal_pos=(12, 16))
    env = MiniGridWrapper(env)
    policy = torch.load(log_dir)
    env = monitor.Monitor(env, os.path.dirname(log_dir), force=True)
    generate_episode(env, policy)
    env.close()
    del env
    del policy


if __name__ == '__main__':
    log_dir = "runs/objects/DEBUG_RUN_Mar18_16-00-09/model-0.pt"
    eval_policy(log_dir)
