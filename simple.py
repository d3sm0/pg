import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import config
from plot_fn import gridworld_plot_sa
from utils.envs import get_four_rooms, get_cliff
from utils.mdp import get_gridworld
from utils.misc_utils import mdpo, get_star, softmax_ppo, policy_iteration


def main():
    # env = get_shamdp(horizon=config.horizon, c=config.penalty)
    env = get_gridworld(grid_size=config.grid_size, gamma=config.gamma)

    # env = get_cliff(gamma=config.gamma)
    def stop_criterion(t):
        return t >= config.max_steps

    pi_star, _, _, v_star = get_star(env)
    # "pg_clip": softmax_ppo, "ppo": mdpo}  # , "ppo": ppo}  # "ppo": ppo, "pg": pg}
    agent = "pg_clip"; agent_fn = softmax_ppo
    agent = "ppo"; agent_fn = mdpo
    etas = np.linspace(0., 3., 9)
    fig, ax = plt.subplots(3, 3)
    # axs = axs.flatten()
    agent_idx = 0
    # for agent, agent_fn in agents.items():
    np.random.seed(0)
    # ax = axs[agent_idx]
    policy, value, pis, vs = policy_iteration(env, agent_fn, eta, stop_criterion)
    label = f"agent={agent}:v={(env.p0 * value).sum():.2f}"
    gridworld_plot_sa(env, policy, label, ax=ax)

    agent_idx += 1

    plt.savefig(f"plots/gridworld/agent={agent}")
    plt.show()


if __name__ == '__main__':
    main()
