import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import config
from misc_utils import ppo, pg, get_star, pg_clip, policy_iteration
from plot_fn import gridworld_plot_sa, plot_policy_at_state
from mdp import get_gridworld  # , get_corridor


def main():
    etas = np.linspace(0.01, 1., num=9)
    # env = get_corridor()
    env = get_gridworld(config.grid_size)
    # env = get_shamdp(100)
    pi = jnp.ones(shape=(1, env.action_space))
    pi /= pi.sum(1, keepdims=True)
    # from main import action_to_text
    labels = ["left", "right", "up", "down"]
    pi_star, _, _, v_star = get_star(env)
    # plt.plot(pi_star)
    gridworld_plot_sa(env, pi_star, f"pi_star")
    plt.savefig("plots/pi_star")
    agents = {"pg_clip": pg_clip, "ppo": ppo, "pg": pg}
    for agent, agent_fn in agents.items():
        fig, axs = plt.subplots(3, 3, figsize=(12, 6), sharex=True)
        axs = axs.flatten()
        for idx, eta in enumerate(etas):
            ax = axs[idx]
            e_pg, pg_pi, t, v_pg, *_ = policy_iteration(agent_fn, env, eta, pi)
            title = f"{agent}:h={e_pg :.3f}:gap={v_star[0] - v_pg[0]:.3f}:t={t}:eta={eta:.2f}"
            plot_policy_at_state(pg_pi, action_label=labels, title=title, ax=ax)
            # ax.plot(v_pg)
            # gridworld_plot_sa(env, pg_pi, title, ax=ax)
            # plot_vf(env, v_pg, f"pg:h={e_pg :.3f}:v={v_pg[0] :.3f}:t={t}:eta={eta:.2f}", ax=ax, frame=(0, 0, 0, 0))
        # plt.legend(["left", "right"])
        plt.savefig(f"plots/large_eta:agent={agent}:size={config.grid_size}")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(5)
        plt.close()


if __name__ == '__main__':
    main()
