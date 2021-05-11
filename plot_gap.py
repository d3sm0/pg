# trajectory in advantage space
import matplotlib.pyplot as plt

from mdp import get_gridworld, get_n_state_chain
from utils.misc_utils import pg_clip, ppo
# from misc_utils import ppo, pg
from utils.misc_utils import get_pi
import jax.numpy as jnp
import numpy as np


def get_init_pi(env, eps):
    pi = np.ones((env.state_space, env.action_space))
    pi[0, 1] = eps
    pi /= pi.sum(1, keepdims=True)
    pi = jnp.array(pi)
    return pi


def run():
    env = get_gridworld(5)
    epsline = np.linspace(0.01, 4., 9)
    fig, ax = plt.subplots(3, 3)
    ax = ax.flatten()
    for idx, eps in enumerate(epsline):
        pi = get_init_pi(env, eps)
        # agents = {"pg": pg, "ppo": ppo}  # "ppo": ppo, "pg_clip": pg_clip}
        etas = np.linspace(0.01, 100., 10)
        as_etas = []
        for eta in etas:
            pg_pi, pi_adv, entropy, new_value, kl = get_pi(env, pi, pg_clip, eta=eta)
            ppo_pi, ppo_adv, entropy, new_value, kl = get_pi(env, pi, ppo, eta=eta)
            as_etas.append((pg_pi[0, 1], ppo_pi[0, 1]))
        ax[idx].plot(etas, as_etas)
        ax[idx].set_title(f"pi'(a)={pi[0,1]:.2f}")
        ax[idx].set_xlabel("eta")
        # ax[idx].set_ylim(0,1)
        ax[idx].set_ylabel("p(a)")
        plt.legend(["pg", "ppo"])
    plt.tight_layout()
    plt.savefig("plots/p_star_chain_gridworld")
    plt.show()
    print("")




if __name__ == "__main__":
    run()
