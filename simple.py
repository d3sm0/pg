import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import config
from plot_fn import gridworld_plot_sa, plot_vf, chain_plot_vf
from utils.envs import get_four_rooms, get_cliff
from utils.mdp import get_gridworld, get_shamdp
from utils.misc_utils import mdpo, get_star, softmax_ppo, policy_iteration


def main():
    #env = get(horizon=config.horizon, c=config.penalty)

#    etas = [0.01, 0.16736842, 0.32473684, 0.48210526, 0.63947368,
#            0.79684211, 0.95421053, 1.11157895, 1.26894737, 1.42631579,
#            1.58368421, 1.74105263, 1.89842105, 2.05578947, 2.21315789,
#            2.37052632, 2.52789474, 2.68526316, 2.84263158, 3.]
#
    etas = [1.]
    env = get_gridworld(grid_size=config.grid_size, gamma=config.gamma)
    # env = get_cliff(gamma=config.gamma)
    pi_star, *_ = get_star(env)
    #plot_vf(env, v, f"vf={config.agent_id}:eta={config.eta:.2f}", log_plot=True, step=config.max_steps + 1)
    gridworld_plot_sa(env, pi_star, title="pi_star", log_plot=False)
    plt.show()
    def stop_criterion(t):
        return t >= config.max_steps

    pi_star, _, _, v_star = get_star(env)
    agents = {"softmax_ppo": softmax_ppo, "mdpo": mdpo}  # , "ppo": ppo}  # "ppo": ppo, "pg": pg}
    fig, axs = plt.subplots(3, 3)
    axs = axs.flatten()
    #for plot_idx, eta in enumerate(etas[:9]):
    for agent, agent_fn in agents.items():
        policy, value, pis, vs = policy_iteration(env, agent_fn, eta, stop_criterion,)
        label = f"agent={agent}:v={(env.p0 * value).sum():.2f}"
        chain_plot_vf(value, ax=axs[plot_idx], label=label)
    axs[plot_idx].legend()
    plt.savefig("plots/shamdp/step_size")
    plt.show()


if __name__ == '__main__':
    main()
