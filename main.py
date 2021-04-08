# exact update of every policy and parameterfs
import os

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


def get_dpi(env, pi):
    p_pi = jnp.einsum('xay,xa->xy', env.P, pi)
    d_pi = jnp.linalg.inv((jnp.eye(env.state_space) - env.gamma * p_pi)) * (1 - config.gamma)
    d_pi /= d_pi.sum(1, keepdims=True)
    return d_pi


def get_q_value(env, pi):
    v = get_value(env, pi)
    v_pi = jnp.einsum('xay,y->xa', env.P, v)
    q = env.R + env.gamma * v_pi
    return q


def kl(p, q, reduce="mean"):
    kl = (p * jnp.log(p / q)).sum(1)
    if reduce == "mean":
        kl = kl.mean()
    return kl


def plot_pi(pi, title, render=False, savefig=True):
    pi_prob = pi.reshape((config.grid_size, config.grid_size, 4)).max(-1)
    pi_act = pi.reshape((config.grid_size, config.grid_size, 4)).argmax(-1)

    fig, ax = plt.subplots(1, 1)
    out = ax.imshow(pi_prob, interpolation=None, cmap='Blues')
    # ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

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


def pg_loss(pi, pi_old, d_pi, adv):
    _kl = config.eta * kl(pi_old, pi, reduce="")
    pi_grad = (pi_old * jnp.log(pi) * adv).sum(1)
    loss = (d_pi * (pi_grad - _kl)).sum(1)
    return loss


def ppo_loss(pi, pi_old, d_pi, adv):
    _kl = config.eta * kl(pi_old, pi, reduce="")
    pi_grad = (pi / pi_old * adv).sum(1)
    loss = (d_pi * (pi_grad - _kl)).sum(1)
    return loss


def entropy(pi):
    return - (pi * jnp.log(pi)).sum(1).mean()


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
        _entropy = entropy(pi)
        render(v, q, pi, global_step, {"pi/kl": _kl, "pi/entropy": _entropy, **eval_stats})
    return pi


def action_to_text(a):
    if a == 0:
        return "left"
    if a == 1:
        return "right"
    if a == 2:
        return "up"
    if a == 3:
        return "down"


def render(v, q, pi, global_step, stats):
    if not config.REMOTE:
        pi_greedy, _ = plot_pi(q, title=f"pi_greedy:{global_step}")
        config.tb.add_figure("pi_greedy", pi_greedy, global_step=global_step)
        v = v.reshape((config.grid_size, config.grid_size))
        q = q.reshape((config.grid_size, config.grid_size, 4)).max(-1)
        v = plot_grid(v, title=f"v:{global_step}")
        config.tb.add_figure("v", v, global_step=global_step)
        q_star = plot_grid(q, title=f"q:{global_step}")
        config.tb.add_figure("q", q_star, global_step=global_step)
        pi_plot, _ = plot_pi(pi, title=f"pi:{global_step}")
        config.tb.add_figure("pi", pi_plot, global_step=global_step)
    for k, v in stats.items():
        config.tb.add_scalar(k, v, global_step)


def main():
    key_gen = hk.PRNGSequence(config.seed)
    env = get_gridworld(config.grid_size)
    if config.agent == "ppo":
        ppo_star = policy_iteration(env, pi_fn=ppo, eta=config.eta, key_gen=key_gen, max_iterations=config.max_steps)
    else:
        pi_star = policy_iteration(env, pi_fn=pg, eta=config.eta, key_gen=key_gen, max_iterations=config.max_steps)
    # print(ppo_star, pi_star)
    # print(jnp.linalg.norm(ppo_star - pi_star,axis=1, ord=1).sum(0))


def make_gif(prefix="pi"):
    import glob
    from PIL import Image
    fp_in = os.path.join(config.plot_path, f"{prefix}:*.png")
    fp_out = os.path.join(config.plot_path, f"{prefix}.gif")

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)
    # config.tb.add_object(prefix, img, global_step=10)


if __name__ == '__main__':
    main()
