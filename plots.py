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
    get_star, get_pi_from_log, escort

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
n_steps = 10
for idx, eta in enumerate(etas):
    ax = axs[idx]
    pi_ppo = pi.clone()
    last_v = get_value(env, pi)
    t = 0
    while True:
        pi_ppo, _, e_ppo, v_ppo, _ = get_pi(env, pi_ppo, ppo, eta=eta)
        if np.abs(last_v[0] - v_ppo[0]) < 1e-2:
            break
        t+=1
        last_v = v_ppo
    ax.hist(np.arange(env.action_space), weights=pi_ppo[0], label=f"ppo:h={e_ppo :.3f}:v={v_ppo[0] :.3f}:t:={t}", alpha=0.5)
    pg_pi = pi.clone()
    last_v = get_value(env, pi)
    t = 0
    while True:
        pg_pi, _, e_pg, v_pg, _ = get_pi(env, pg_pi, pg, eta=eta)
        if np.abs(last_v[0] - v_pg[0]) < 1e-2:
            break
        t+=1
        last_v = v_pg
    ax.hist(np.arange(env.action_space), weights=pg_pi[0], label=f"pg:h={e_pg :.3f}:v={v_pg[0] :.3f}", alpha=0.5)
    ax.set_title(f"eta:{eta:.2f}")
    ax.legend()
    plt.xticks(np.arange(env.action_space), labels)
plt.savefig(f"plots/softmax_{n_steps}_{config.grid_size}")
plt.tight_layout()
plt.show(block=False)
plt.pause(5)
plt.close()
