# TODO plot a 2d transformation of the two densities given the same advatnage funciton and their statistic for every state and action
# what we should observe is what actions are chosen

import numpy as np
import config
from shamdp import get_gridworld
import jax
import jax.numpy as jnp

import haiku as hk
import matplotlib.pyplot as plt

from misc_utils import get_soft_value, entropy_fn, kl_fn, ppo, pg, get_q_value, get_value, get_dpi, get_pi, get_pi_star, \
    get_star

key_gen = hk.PRNGSequence(0)

etas = np.linspace(0.1, 4, num=9)
env = get_gridworld(config.grid_size)
pi = jnp.ones(shape=(env.state_space, env.action_space))
pi /= pi.sum(1, keepdims=True)
# from main import action_to_text
labels = ["left", "right", "up", "down"]
fig, axs = plt.subplots(3, 3, figsize=(12, 6), sharex=True)
axs = axs.flatten()
pi_star, *_ = get_star(env)
n_steps = 4
for idx, eta in enumerate(etas):
    ax = axs[idx]
    pi_ppo = pi.clone()
    for _ in range(n_steps):
        pi_ppo, _, e_ppo, v_ppo = get_pi(env, pi_ppo, ppo, eta=eta)
    ax.hist(np.arange(env.action_space), weights=pi_ppo[0], label=f"agent=ppo:delta:h={e_ppo :.3f}:v={v_ppo[0] :.3f}",
            alpha=0.5)
    pi_pg = pi.clone()
    for _ in range(n_steps):
        pi_pg, _, e_pg, v_pg = get_pi(env, pi_pg, pg, eta=eta)
    ax.hist(np.arange(env.action_space), weights=pi_pg[0], label=f"agent=pg:delta:h={e_pg :.3f}:v={v_pg[0] :.3f}",
            alpha=0.5)
    ax.set_title(f"eta:{eta:.2f}")
    ax.legend()
    plt.xticks(np.arange(env.action_space), labels)
plt.savefig(f"soft_value_{n_steps}")
plt.tight_layout()
plt.show(block=False)
plt.pause(5)
plt.close()
