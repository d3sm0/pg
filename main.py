import itertools

import torch
from gym_minigrid.envs import EmptyEnv

import config
from agent import PG
from env_utils import MiniGridWrapper
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
    print(config.agent)
    env.seed(config.seed)
    env = MiniGridWrapper(env)
    agent = PG(action_space=env.action_space.n, observation_space=env.n_states, h_dim=config.h_dim)
    plot_value(agent, global_step=0)

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
            eval_info = eval_policy(path, config.eval_runs, record_episode=False)
            for k, v in eval_info.items():
                config.tb.add_scalar(k, v, global_step=global_step * config.horizon)
            plot_value(agent, global_step)
    env.close()


def plot_value(agent, global_step):
    value = agent._agent.v.reshape((4, config.grid_size, config.grid_size)).max(0)[0]
    q_value = agent._agent.q.reshape((4, config.grid_size, config.grid_size, 3)).max(-1)[0].max(0)[0]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(value)
    ax[1].imshow(q_value)
    config.tb.add_figure("value", fig, global_step)


if __name__ == '__main__':
    main()
