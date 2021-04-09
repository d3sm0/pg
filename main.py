# exact update of every policy and parameterfs
import os

import haiku as hk

from shamdp import get_gridworld
import config
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def v_iteration(env, n_iterations=100, eps=1e-5):
    v = jnp.zeros(env.state_space)
    for _ in range(n_iterations):
        v_new = (env.R + jnp.einsum("xay,y->xa", env.gamma * env.P, v)).max(1)
        if jnp.linalg.norm(v - v_new) < eps:
            break
        v = v_new.clone()
    return v


def q_iteration(env, n_iterations=100, eps=1e-5):
    q = jnp.zeros((env.state_space, env.action_space))
    for _ in range(n_iterations):
        q_star = q.max(1)
        q_new = (env.R + jnp.einsum("xay,y->xa", env.gamma * env.P, q_star))
        if jnp.linalg.norm(q - q_new) < eps:
            break
        q = q_new.clone()
    return q


def get_value(env, pi):
    p_pi = jnp.einsum('xay,xa->xy', env.P, pi)
    r_pi = jnp.einsum('xa,xa->x', env.R, pi)
    v = jnp.linalg.solve((jnp.eye(env.state_space) - env.gamma * p_pi), r_pi)
    return v


def get_dpi(env, pi):
    p_pi = jnp.einsum('xay,xa->xy', env.P, pi)
    rho = np.zeros(env.state_space)
    rho[0] = 1.
    d_pi = jnp.linalg.solve((jnp.eye(env.state_space) - env.gamma * p_pi.T), (1 - env.gamma) * rho)
    d_pi /= d_pi.sum()
    return d_pi


def get_q_value(env, pi):
    v = get_value(env, pi)
    v_pi = jnp.einsum('xay,y->xa', env.P, v)
    q = env.R + env.gamma * v_pi
    return q


def kl_fn(p, q, ds):
    _kl = (p * jnp.log((p + 1e-4) / (q + 1e-4))).sum(1)
    return (ds * _kl).sum(-1)


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
        total_return += (env.gamma ** t) * r
        t += 1
        if t > 200:
            break
    return total_return


def pg(pi, adv, *args):
    pi = pi * jnp.exp(1 + config.eta * adv)
    pi = jax.nn.softmax(pi)
    return pi, {}


def pg_loss(pi, pi_old, d_s, adv):
    kl = 1 / config.eta * kl_fn(pi_old, pi, d_s)
    loss = (pi_old * jnp.log(pi) * adv).sum(1)
    loss = (d_s * loss).sum(0)
    kl = (d_s * kl).sum(0)
    return loss + kl


def ppo(pi, adv, *args):
    pi = pi * jnp.exp(config.eta * adv)
    pi = jax.nn.softmax(pi)
    return pi, {}


def ppo_loss(pi, pi_old, d_s, adv):
    kl = 1 / config.eta * kl_fn(pi_old, pi, d_s)
    loss = (d_s * (pi / pi_old * adv).sum(1)).sum(0)
    kl = (kl * d_s).sum(0)
    return loss + kl


def entropy_fn(pi):
    return - (pi * jnp.log(pi)).sum(1)


def approx_pi(pi_fn):
    d_pi = jax.value_and_grad(pi_fn)

    def _fn(pi, adv, d_s):
        pi_old = pi.copy()
        total_loss = 0
        for step in range(config.opt_epochs):
            loss, grad = d_pi(pi, pi_old, d_s, adv)
            pi = pi + config.pi_lr * grad
            grad_norm = jnp.linalg.norm(grad)
            total_loss += loss
            pi = jax.nn.softmax(pi)
        return pi, {"pi/loss": total_loss / config.opt_epochs, "pi/grad_norm_iter": grad_norm}

    return _fn


def get_pi_star(q):
    pi = np.zeros_like(q)
    idx = q.argmax(1)
    for i in range(pi.shape[0]):
        pi[i, idx[i]] = 1
    return pi


def policy_iteration(env, pi_fn, pi_approx_fn, max_steps=10, key_gen=None):
    pi = jax.random.uniform(key=next(key_gen), shape=(1, env.action_space))
    pi /= pi.sum(axis=-1, keepdims=True)

    v_star = v_iteration(env)
    q_star = q_iteration(env)
    pi_star = get_pi_star(q_star)
    d_star = get_dpi(env, pi_star)
    adv_star = (pi_star * (q_star - jnp.expand_dims(v_star, 1))).sum(1)
    print(eval_policy(env, pi_star, key_gen))

    v = get_value(env, pi_star)
    q = get_q_value(env, pi_star)
    adv = (pi_star * (q - jnp.expand_dims(v, 1))).sum(1)
    assert jnp.allclose(adv, adv_star)

    for global_step in range(max_steps):
        pi_old = pi.clone()
        d_s = get_dpi(env, pi_old)
        pi, adv = get_pi(env, pi_old, pi_fn)
        entropy = (d_s * entropy_fn(pi)).sum(0)
        v = get_value(env, pi)
        kl_star = kl_fn(pi_star, pi, d_star)
        pi_approx, extra = pi_approx_fn(pi_old.clone(), adv, d_s)
        v_approx = get_value(env, pi_approx)
        v_gap_approx = jnp.linalg.norm(v - v_approx)
        v_gap = jnp.linalg.norm(v - v_star)

        stats = {
            "train/v_gap": v_gap,
            "train/v_gap_approx": v_gap_approx,
            "train/entropy": entropy,
            "train/kl_star": kl_star,
                             ** extra}
        save_stats(stats, global_step)
        if key_gen is not None:
            avg_return = 0
            for _ in range(config.eval_episodes):
                total_return = eval_policy(env, pi, key_gen)
                avg_return += total_return
            save_stats({"eval/total_return": avg_return / config.eval_episodes}, global_step)
    return pi


def get_pi(env, pi, pi_fn):
    v = get_value(env, pi)
    q = get_q_value(env, pi)
    adv = q - jnp.expand_dims(v, 1)
    pi, _ = pi_fn(pi, adv)
    return pi, adv


def action_to_text(a):
    if a == 0:
        return "left"
    if a == 1:
        return "right"
    if a == 2:
        return "up"
    if a == 3:
        return "down"


def render(v, q, pi, global_step):
    # if not config.REMOTE:
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


def save_stats(stats, global_step):
    for k, v in stats.items():
        config.tb.add_scalar(k, v, global_step)


def main():
    key_gen = hk.PRNGSequence(config.seed)
    env = get_gridworld(config.grid_size)
    if config.agent == "ppo":
        pi_approx = approx_pi(ppo_loss)
        pi_fn = ppo
    else:
        pi_approx = approx_pi(pg_loss)
        pi_fn = pg

    ppo_star = policy_iteration(env, pi_fn=pi_fn, pi_approx_fn=pi_approx, key_gen=key_gen,
                                max_steps=config.max_steps)


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
