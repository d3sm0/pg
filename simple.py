import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import config
from experiments.plot_fn import gridworld_plot_sa, plot_policy_at_state, plot_vf
from mdp import get_gridworld
from utils.envs import get_four_rooms
from utils.misc_utils import ppo, get_star, pg_clip, policy_iteration, pg


def main():
    etas = np.linspace(0.1, 1.0, num=9)
    # etas = [1.]
    env = get_four_rooms(gamma=config.gamma)
    pi = np.ones(shape=(1, env.action_space))
    pi /= pi.sum(1, keepdims=True)
    pi = jnp.array(pi)
    # from main import action_to_text
    labels = ["left", "right", "up", "down"]
    pi_star, _, _, v_star = get_star(env)
    # plt.plot(pi_star)
    gridworld_plot_sa(env, pi_star, f"pi_star")
    plt.savefig("plots/pi_star")
    plt.close()
    agents = {"pg": pg, "pg_clip": pg_clip, "ppo": ppo}
    for agent, agent_fn in agents.items():
        fig, axs = plt.subplots(3, 3, figsize=(12, 6))
        axs = axs.flatten()
        for idx, eta in enumerate(etas):
            ax = axs[idx]
            e_pg, pg_pi, t, v_pg, pi_s, advs, _ = policy_iteration(agent_fn, env, eta, pi)
            title = f"{agent}:h={e_pg :.3f}:gap={jnp.linalg.norm(v_star[10] - v_pg[10]):.3f}:t={t}:eta={eta:.2f}"
            plot_policy_at_state(pg_pi, action_label=labels, title=title, ax=ax)
            # gridworld_plot_sa(env, pg_pi, title, ax=ax)
            # plot_vf(env, v_pg, title, ax=ax, frame=(0, 0, 0, 0))
        # plt.show()
        # plt.legend(["left", "right"])
        plt.tight_layout()
        plt.savefig(f"plots/4rooms_approx/kl_agent={agent}:size={config.grid_size}")
        plt.show(block=False)
        plt.pause(5)
        plt.close()


if __name__ == '__main__':
    main()
