import collections
import os

import torch
from gym.wrappers import monitor
from gym_minigrid.envs import EmptyEnv
from torch.distributions import Categorical


def generate_episode(env, policy):
    s = env.reset()
    d = False
    info = {}
    while not d:
        with torch.no_grad():
            probs = policy.policy(torch.from_numpy(s).long())
        pi = Categorical(probs=probs)
        action = pi.sample().item()
        s1, r, d, info = env.step(action)
        # env.render()
        s = s1
    env.close()
    return info


def eval_policy(log_dir, eval_runs=1, record_episode=False):
    from env_utils import MiniGridWrapper
    # env = FourRoomsEnv(goal_pos=(12, 16))
    env = EmptyEnv()
    env = MiniGridWrapper(env)
    policy = torch.load(log_dir)
    agg_info = collections.defaultdict(lambda: 0)
    if record_episode:
        env = monitor.Monitor(env, os.path.dirname(log_dir), force=True)
    for _ in range(eval_runs):
        info = generate_episode(env, policy)
        for k in info.keys():
            agg_info[f"eval/{k}"] += info[k] / eval_runs
    env.close()
    del env
    del policy
    return dict(agg_info)


if __name__ == '__main__':
    # log_dir = "runs/objects/no_id_Mar29_23-41-20/agent-0.pt"
    log_dir = "runs/objects/no_id_Mar29_23-46-06/agent-0.pt"
    # log_dir = "runs/objects/no_id_Mar29_23-13-03/agent-3470.pt"
    eval_policy(log_dir, record_episode=True)
