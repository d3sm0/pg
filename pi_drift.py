import jax
# trajectory in advantage space
import matplotlib.pyplot as plt
from misc_utils import policy_iteration, pg_clip
# from misc_utils import ppo, pg
from misc_utils import get_pi
import jax.numpy as jnp
import numpy as np
from mdp import get_n_state_chain
from misc_utils import get_value, get_q_value


def ppo(pi, adv):
    return pi * jnp.exp(adv)


def pg(pi, adv):
    return pi + pi * adv




def run():
    env = get_n_state_chain(2)
    env.P[1, 0, 0] = 0
    env.P[1, 0, 1] = 1
    env.P[1, 1, 1] = 1
    pi = jnp.ones((2, 2)) / 2
    agents = {"pg": pg, "ppo": ppo}  # "ppo": ppo, "pg_clip": pg_clip}
    etas = np.linspace(0.01, 1.0, num=9)
    # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    fig, axs = plt.subplots(3, 3, figsize=(12, 6), sharex=True)
    for agent, agent_fn in agents.items():
        axs = axs.flatten()
        for idx, eta in enumerate(etas):
            ax = axs[idx]
            e_pg, pg_pi, t, v_pg, policies, advs, _ = policy_iteration(agent_fn, env, eta, pi)
            ax.plot(advs[:, 0, 0], policies[:, 0, 0], label=f"{agent}:a=0")
            ax.plot(advs[:, 0, 1], policies[:, 0, 1], label=f"{agent}:a=1")

    for idx, ax in enumerate(axs):
        ax.set_xlabel("adv")
        ax.set_ylabel("pi")
        plt.legend()
        ax.grid()
        ax.set_title(f"eta={etas[idx]}")
    plt.tight_layout()
    plt.savefig('plots/pi_vs_adv')
    plt.show()
    print("")


if __name__ == "__main__":
    run()
