import itertools

import torch
#from simple_rl.tasks import GridWorldMDP
from torch.distributions import Categorical

import config
from env_utils import Trajectory, Transition, MiniGridWrapper
from ppo import PPO


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
    #env = FourRoomsEnv(goal_pos=(12, 16))
    from gym_minigrid.envs import EmptyEnv5x5
    env = EmptyEnv5x5()
    #env = GridWorldMDP()
    torch.manual_seed(config.seed)
    print(config.agent)
    #env.seed(config.seed)
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
