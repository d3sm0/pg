import matplotlib.pyplot as plt
import numpy as np

import config
from plot_fn import chain_plot_vf, gridworld_plot_sa
from utils.envs import get_cliff
from utils.mdp import get_shamdp
from utils.misc_utils import mdpo, get_star, softmax_ppo, policy_iteration
import jax


def main():
    env = get_cliff(gamma=config.gamma)
    eta = 1.

    # env = get_gridworld(grid_size=config.grid_size, gamma=config.gamma)
    # env = get_cliff(gamma=config.gamma)
    def stop_criterion(t):
        return t >= config.max_steps

    pi_star, _, _, v_star = get_star(env)
    agents = {"softmax_ppo": softmax_ppo, "mdpo": mdpo}  # , "ppo": ppo}  # "ppo": ppo, "pg": pg}
    fig, axs = plt.subplots(3, 3)
    axs = axs.flatten()
    # for plot_idx, eta in enumerate(etas[:9]):
    pi = np.ones((env.state_space, env.action_space))
    pi /= pi.sum(1, keepdims=True)
    pi[:, 1] = 4.
    pi = jax.nn.softmax(pi)
    plot_idx = 0
    for agent, agent_fn in agents.items():
        policy, value, pis, vs = policy_iteration(env, agent_fn, eta, stop_criterion, pi)
        label = f"agent={agent}:v={(env.p0 * value).sum():.2f}"
        gridworld_plot_sa(env, pi, f"pi:eta={config.eta:.2f}", log_plot=False, step=config.max_steps + 1, ax=axs)
        #chain_plot_vf(value, ax=axs[plot_idx], label=label)
    axs[plot_idx].legend()
    plt.savefig("plots/shamdp/policy_init")
    plt.show()


if __name__ == '__main__':
    main()
