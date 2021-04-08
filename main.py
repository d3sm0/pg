# exact update of every policy and parameterfs
from random import random
import os
from emdp.gridworld import GridWorldPlotter

import haiku as hk

from shamdp import get_gridworld
import config
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def get_value(env, pi):
    p_pi = jnp.einsum('xay,xa->xy', env.P, pi)
    r_pi = jnp.einsum('xa,xa->x', env.R, pi)
    v = jnp.linalg.solve((jnp.eye(env.state_space) - env.gamma * p_pi), r_pi)
    return v


def get_q_value(env, pi):
    v = get_value(env, pi)
    v_pi = jnp.einsum('xay,y->xa', env.P, v)
    q = env.R + env.gamma * v_pi
    return q


def kl(p, q):
    kl = (p * jnp.log(p / q)).sum(1).mean()
    return kl


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
        total_return += r
        t += 1
    return {"eval/steps": t, "eval/return": total_return}


def pg(pi, adv, eta):
    pi = pi * jnp.exp(1 + eta * adv)
    pi = jax.nn.softmax(pi)
    return pi


def ppo(pi, adv, eta):
    pi = pi * jnp.exp(eta * adv)
    pi = jax.nn.softmax(pi)
    return pi


def policy_iteration(env, pi_fn, eta, max_iterations=10, key_gen=None):
    pi = jnp.ones((env.state_space, env.action_space))
    pi /= pi.sum(axis=-1, keepdims=True)
    for global_step in range(max_iterations):
        v = get_value(env, pi)
        q = get_q_value(env, pi)
        adv = q - jnp.expand_dims(v, 1)
        pi_old = pi.copy()
        pi = pi_fn(pi, adv, eta)
        eval_stats = {}
        if key_gen is not None:
            eval_stats = eval_policy(env, pi, key_gen)
        _kl = kl(pi, pi_old)
        render(v, q, pi, global_step, {"pi/kl": _kl, **eval_stats})
    return pi


def render(v, q, pi, global_step, stats):
    v = v.reshape((config.grid_size, config.grid_size))
    q = q.reshape((config.grid_size, config.grid_size, 4)).max(-1)
    pi = pi.reshape((config.grid_size, config.grid_size, 4)).max(-1)
    v = plot_grid(v, title="v_star")
    config.tb.add_figure("v_star", v, global_step=global_step)
    q_star = plot_grid(q, title=f"q_star")
    config.tb.add_figure("q_star", q_star, global_step=global_step)
    pi_plot = plot_grid(pi, title=f"pi-star")
    config.tb.add_figure("pi_star", pi_plot, global_step=global_step)
    for k, v in stats.items():
        config.tb.add_scalar(k, v, global_step)


def main():
    key_gen = hk.PRNGSequence(config.seed)
    env = get_gridworld(config.grid_size)
    if config.agent == "pg":
        pi_fn = pg
    else:
        pi_fn = ppo
    pi_star = policy_iteration(env, pi_fn=pi_fn, eta=config.eta, key_gen=key_gen)


if __name__ == '__main__':
    main()
