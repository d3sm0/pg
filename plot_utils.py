import os

import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt

import config


# from main import action_to_text

def action_to_text(a):
    if a == 0:
        return "left"
    if a == 1:
        return "right"
    if a == 2:
        return "up"
    if a == 3:
        return "down"


def plot_pi(pi, title, render=False, savefig=True):
    pi_prob = pi.reshape((config.grid_size, config.grid_size, 4)).max(-1)
    pi_act = pi.reshape((config.grid_size, config.grid_size, 4)).argmax(-1)

    fig, ax = plt.subplots(1, 1)
    out = ax.imshow(pi_prob, interpolation=None, cmap='Blues')
    # ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #  ax.set_title(title)

    n, m = pi_prob.shape
    for x in range(n):
        for y in range(m):
            p_a = pi_prob[x, y]
            a = action_to_text(pi_act[x, y])
            ax.text(x, y, f"p:{p_a:.2f}, A:{a}",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=6,
                    )
    fig.colorbar(out, ax=ax)
    if savefig:
        fig.savefig(os.path.join(config.plot_path, title))
    if render:
        plt.show()
    return fig, ax


def plot_grid(f, title, render=False, savefig=True):
    fig, ax = plt.subplots(1, 1)
    out = ax.imshow(f, interpolation=None, cmap='Blues')
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    n, m = f.shape
    for x in range(n):
        for y in range(m):
            ax.text(x, y, f"{f[x, y]:.2f}")
    fig.colorbar(out, ax=ax)
    if savefig:
        fig.savefig(os.path.join(config.plot_path, title))
    if render:
        plt.show()
    return fig


def eval_policy(env, pi, key_gen):
    s = env.reset()
    done = False
    total_return = 0
    t = 0
    while not done:
        p = jnp.einsum("s, sa->a", s, pi)
        action = jax.random.choice(key=next(key_gen), a=env.action_space, p=p).item()
        s, r, done, info = env.step(action)
        total_return += (env.gamma ** t) * r
        t += 1
        if t > 200:
            break
    return total_return
