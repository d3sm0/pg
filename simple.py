import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import config
from misc_utils import ppo, pg, get_value, get_pi, get_star, is_prob_mass
from plot_fn import gridworld_plot_sa
from mdp import get_gridworld


def main():
    etas = np.linspace(0.01, 0.3, num=9)
    env = get_gridworld(config.grid_size)
    pi = jnp.ones(shape=(env.state_space, env.action_space))
    pi /= pi.sum(1, keepdims=True)
    # from main import action_to_text
    pi_star, _, _, v_star = get_star(env)
    gridworld_plot_sa(env, pi_star, f"pi_star")
    plt.savefig("plots/pi_star")
    agents = {"pg": pg, "ppo": ppo}
    for agent, agent_fn in agents.items():
        fig, axs = plt.subplots(3, 3, figsize=(12, 6), sharex=True)
        axs = axs.flatten()
        for idx, eta in enumerate(etas):
            ax = axs[idx]
            pg_pi = pi.clone()
            last_v = get_value(env, pi)
            t = 0
            while True:
                pg_pi, _, e_pg, v_pg, _ = get_pi(env, pg_pi, agent_fn, eta=eta)
                if not is_prob_mass(pg_pi):
                    print(f"pi is valid", pi)
                    break
                if np.linalg.norm(last_v[0] - v_pg[0]) < config.eps or t > config.max_steps:
                    break
                t += 1
                last_v = v_pg
            title = f"pg:h={e_pg :.3f}:gap={v_star[0] - v_pg[0]:.3f}:t={t}:eta={eta:.2f}"
            # plot_policy_at_state(pg_pi, action_label=labels, title=title, ax=ax)
            gridworld_plot_sa(env, pg_pi, title, ax=ax)
            # plot_vf(env, v_pg, f"pg:h={e_pg :.3f}:v={v_pg[0] :.3f}:t={t}:eta={eta:.2f}", ax=ax, frame=(0, 0, 0, 0))
        plt.savefig(f"plots/param_small_eta:agent={agent}:size={config.grid_size}")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(5)
        plt.close()


if __name__ == '__main__':
    main()
