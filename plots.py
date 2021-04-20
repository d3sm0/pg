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
from plot_fn import gridworld_plot_sa, plot_vf

key_gen = hk.PRNGSequence(0)

etas = np.linspace(0.01, 0.3, num=9)
env = get_gridworld(config.grid_size)
pi = jnp.ones(shape=(env.state_space, env.action_space))
pi /= pi.sum(1, keepdims=True)
# from main import action_to_text
labels = ["left", "right", "up", "down"]
pi_star, *_ = get_star(env)
gridworld_plot_sa(env, pi_star, f"pi_star")
plt.savefig("plots/pi_star")
agents = {"ppo": ppo, "pg": pg}
t_max = int(1e3)
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
            assert pg_pi.sum(1).all() and (pg_pi >= 0).all()
            if np.linalg.norm(last_v[0] - v_pg[0]) < 1e-2 or t > t_max:
                break
            t += 1
            last_v = v_pg
        # gridworld_plot_sa(env, pg_pi, f"pg:h={e_pg :.3f}:v={v_pg[0] :.3f}:t={t}:eta={eta:.2f}", ax=ax)
        plot_vf(env, v_pg, f"pg:h={e_pg :.3f}:v={v_pg[0] :.3f}:t={t}:eta={eta:.2f}", ax=ax, frame=(0,0,0,0))
    plt.savefig(f"plots/vf:agent={agent}:size={config.grid_size}")
    plt.tight_layout()
    # plt.show(block=False)
    # plt.pause(5)
    plt.close()
