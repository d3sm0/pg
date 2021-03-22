import itertools

import torch
from gym_minigrid.envs import EmptyEnv

import config
from agent import PG, PPO
from env_utils import MiniGridWrapper
from eval_policy import eval_policy


def gather_trajectory(env, model, horizon):
    s = env.get_state()
    info = {}
    for t in range(horizon):
        action = model.act(s)
        s_prime, r, done, info = env.step(action)
        model.put_data((s, action, r, s_prime, 1 - done))
        s = s_prime
        if done:
            s = env.reset()
    return info


def main():
    env = EmptyEnv(size=config.grid_size)  # FourRoomsEnv(goal_pos=(12, 16))
    torch.manual_seed(config.seed)
    print(config.agent)
    env.seed(config.seed)
    env = MiniGridWrapper(env)
    if config.agent == "pg":
        agent = PG(action_space=env.action_space.n, observation_space=env.observation_space.shape[0],
                   h_dim=config.h_dim)
    else:
        agent = PPO(action_space=env.action_space.n, observation_space=env.observation_space.shape[0],
                    h_dim=config.h_dim)

    # writer = tb.SummaryWriter(log_dir=f"logs/{dtm}_as_ppo:{config.as_ppo}")
    for global_step in itertools.count():
        info = gather_trajectory(env, agent, config.horizon)
        config.tb.add_scalar("return", info["env/returns"], global_step=global_step * config.horizon)
        losses = agent.train()
        agent.data.clear()
        for k, v in losses.items():
            config.tb.add_scalar(k, v, global_step=global_step * config.horizon)
        if global_step % config.save_interval == 0:
            path = config.tb.add_object("agent", agent.get_model(), global_step=0)
            eval_info = eval_policy(path, config.eval_runs)
            for k, v in eval_info.items():
                config.tb.add_scalar(k, v, global_step=global_step * config.horizon)
    env.close()


if __name__ == '__main__':
    main()
