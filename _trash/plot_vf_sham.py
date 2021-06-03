import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import config
from experiments.plot_fn import gridworld_plot_sa, plot_policy_at_state, plot_vf, chain_plot_sa, chain_plot_vf
from utils.mdp import get_gridworld, get_shamdp
from utils.envs import get_four_rooms, get_cliff
from utils.misc_utils import mdpo, get_star, softmax_ppo, policy_iteration, pg


def main():
    etas = np.linspace(0.4, 1., num=2)
    env = get_shamdp(horizon=config.horizon, c=config.penalty)
    # env = get_gridworld(grid_size=config.grid_size)
    # env = get_four_rooms(gamma=config.gamma)
    # env = get_cliff(gamma=config.gamma)
    n_parameters = 1 if config.use_fa else env.state_space
    pi = np.ones(shape=(n_parameters, env.action_space))

    def stop_criterion(t, v, v_old):
        return t >= config.max_steps

    pi /= pi.sum(1, keepdims=True)
    pi = jnp.array(pi)
    # labels = ["left", "right", "up", "down"]
    _, _, _, v_star = get_star(env)
    #agents = {"pg_clip": pg_clip, "ppo"} #, "ppo": ppo}  # "ppo": ppo, "pg": pg}
    agents = {"ppo": mdpo, "pg_clip": softmax_ppo}  # , "ppo": ppo}  # "ppo": ppo, "pg": pg}
    # fig, axs = plt.subplots(3, 1, figsize=(12, 6))
    # axs = axs.flatten()
    plot_idx = 0
    # for eta in etas:
    fig, axs = plt.subplots(2, 1)
    axs = axs.flatten()
    # ax = axs[plot_idx]
    # title = f"{eta:.2f}" # :gap={(env.p0 * (v_star - v_pg)).sum():.3f}:t={t}:eta={eta:.2f}"
    for eta in etas:
        ax = axs[plot_idx]
        for agent, agent_fn in agents.items():
            label = f"agent:{agent}:eta={eta:.2f}"
            e_pg, pg_pi, t, v_pg, pi_s, advs, _, d_s = policy_iteration(agent_fn, env, eta, pi, stop_criterion)
            # gridworld_plot_sa(env, pg_pi, title, ax=ax)
            # plot_policy_at_state(pi, action_label=labels, title=title, ax=ax)
            chain_plot_vf(env, v_pg, label=label, ax=ax)
            # img = plot_vf(env, v_pg, title, ax=ax)

            # chain_plot_sa(env, pi_s[:, :, 0], "", ax=ax, label=label)
        ax.legend(loc='lower right')
        plot_idx += 1
    # plt.show()
    # plt.savefig(f"plots/shamdp/pi")
    plt.show()
    # plt.pause(5)
    # plt.close()


if __name__ == '__main__':
    main()
