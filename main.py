import itertools

import torch
from gym_minigrid.envs import EmptyEnv

import config
from agent import PG, PPO
# from chain_mdp import MDP
from env_utils import MiniGridWrapper, StatisticsWrapper
from eval_policy import eval_policy


def gather_trajectory(env, model, horizon):
    s = env.get_state()
    info = {}
    done = False
    while not done:
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
    env.seed(config.seed)
    # env = MDP()
    env = MiniGridWrapper(env)
    env = StatisticsWrapper(env)
    if config.agent == "pg":
        agent = PG(action_space=env.action_space.n, observation_space=env.env.n_states, h_dim=config.h_dim)
    else:
        agent = PPO(action_space=env.action_space.n, observation_space=env.env.n_states, h_dim=config.h_dim)
    print(config.agent)
    print(config.tb.run.config)
    # plot_value(env, agent, global_step=0)

    # writer = tb.SummaryWriter(log_dir=f"logs/{dtm}_as_ppo:{config.as_ppo}")
    for global_step in itertools.count():
        info = gather_trajectory(env, agent, config.horizon)
        config.tb.add_scalar("return", info["env/returns"], global_step=global_step * config.horizon)
        losses = agent.train()
        config.tb.add_histogram("pi", agent._agent.pi.data, global_step=global_step * config.horizon)
        agent.data.clear()
        for k, v in losses.items():
            config.tb.add_scalar(k, v, global_step=global_step * config.horizon)
        if global_step % config.save_interval == 0:
            path = config.tb.add_object("agent", agent, global_step=0)
            eval_info = eval_policy(path, config.eval_runs, record_episode=False)
            for k, v in eval_info.items():
                config.tb.add_scalar(k, v, global_step=global_step * config.horizon)
            # plot_value(env, agent, global_step * config.horizon)
        if global_step > config.max_steps:
            break
    env.close()
    config.tb.run.finish()


def plot_value(env, agent, global_step):
    value = torch.zeros(4, config.grid_size, config.grid_size)
    q_value = torch.zeros(4, config.grid_size, config.grid_size)
    pi = torch.zeros(4, config.grid_size, config.grid_size)
    for z in range(4):
        for x in range(config.grid_size):
            for y in range(config.grid_size):
                idx = env._state_to_idx[(z, x, y)]
                value[z, x, y] = agent._agent.v[idx]
                q_value[z, x, y] = agent._agent.q[idx].max()
                pi[z, x, y] = agent._agent.pi.weight.data[:, idx].max()

    value = value.max(0)[0]
    q_value = q_value.max(0)[0]
    pi = pi.max(0)[0]
    # q_value = agent._agent.q.reshape((4, config.grid_size, config.grid_size, 3)).max(-1)[0].max(0)[0]

    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 3)
    # fig = plt.figure()
    fig = plt.figure()
    ax = plt.imshow(value)
    config.tb.add_figure("plots/value", fig, global_step)
    fig = plt.figure()
    ax = plt.imshow(q_value)
    config.tb.add_figure("plots/q_value", fig, global_step)
    fig = plt.figure()
    ax = plt.imshow(pi)
    config.tb.add_figure("plots/pi", fig, global_step)
    plt.close()


if __name__ == '__main__':
    main()
