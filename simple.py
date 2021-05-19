import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import config
from experiments.plot_fn import gridworld_plot_sa, plot_policy_at_state, plot_vf, chain_plot_sa, chain_plot_vf
from utils.mdp import get_gridworld, get_shamdp
from utils.envs import get_four_rooms, get_cliff
from utils.misc_utils import ppo, get_star, pg_clip, policy_iteration, pg


def main():
    etas = np.linspace(0.1, 1, num=9)
    # env = get_shamdp(horizon=config.horizon, c=config.penalty)
    # env = get_gridworld(grid_size=config.grid_size)
    # env = get_four_rooms(gamma=config.gamma)
    env = get_cliff(gamma=config.gamma)
    n_parameters = 1 if config.use_fa else env.state_space
    pi = np.ones(shape=(n_parameters, env.action_space))

    def stop_criterion(t, v, v_old):
        return jnp.linalg.norm(v - v_old) < config.eps or t >= config.max_steps

    pi /= pi.sum(1, keepdims=True)
    pi = jnp.array(pi)
    labels = ["left", "right", "up", "down"]
    _, _, _, v_star = get_star(env)
    agents = {"pg_clip": pg_clip, "ppo": ppo}  # "ppo": ppo, "pg": pg}
    for agent, agent_fn in agents.items():
        fig, axs = plt.subplots(3, 3, figsize=(12, 6))
        axs = axs.flatten()
        for idx, eta in enumerate(etas):
            ax = axs[idx]
            e_pg, pg_pi, t, v_pg, pi_s, advs, _, d_s = policy_iteration(agent_fn, env, eta, pi, stop_criterion)
            title = f"{agent}:h={e_pg :.3f}:gap={(env.p0 * (v_star - v_pg)).sum():.3f}:t={t}:eta={eta:.2f}"
            # if config.use_fa:
            # else:
            # gridworld_plot_sa(env, pg_pi, title, ax=ax)
            #plot_policy_at_state(pi, action_label=labels, title=title, ax=ax)
            plot_vf(env, v_pg, title, ax=ax)
            plt.tight_layout()
            # chain_plot_sa(env, pi_s[:, :, 0], title, ax=ax)
            # chain_plot_vf(env, v_pg, title, ax=ax)
        # plt.show()
        # plt.legend(["left", "right"])
        plt.savefig(f"plots/cliff/vf:agent={agent}")
        plt.show(block=False)
        plt.pause(5)
        plt.close()


if __name__ == '__main__':
    main()
