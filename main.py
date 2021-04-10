# exact update of every policy and parameterfs
import functools
import os
import pickle

import tqdm

import haiku as hk
import wandb

from shamdp import get_gridworld
import config
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from plot_utils import plot_grid, plot_pi


def make_gif(prefix="pi"):
    import glob
    from PIL import Image
    fp_in = os.path.join(config.plot_path, f"{prefix}:*.png")
    fp_out = os.path.join(config.plot_path, f"{prefix}.gif")

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)
    # config.tb.add_object(prefix, img, global_step=10)


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


def get_soft_value(env, pi):
    p_pi = jnp.einsum('xay,xa->xy', env.P, pi)
    r_pi = jnp.einsum('xa,xa->x', env.R, pi) - config.eta * jnp.einsum('xa,xa->x', -pi, jnp.log(pi))
    v = jnp.linalg.solve((jnp.eye(env.state_space) - env.gamma * p_pi), r_pi)
    return v


def get_value(env, pi):
    p_pi = jnp.einsum('xay,xa->xy', env.P, pi)
    r_pi = jnp.einsum('xa,xa->x', env.R, pi)
    v = jnp.linalg.solve((jnp.eye(env.state_space) - env.gamma * p_pi), r_pi)
    return v


def get_dpi(env, pi):
    p_pi = jnp.einsum('xay,xa->xy', env.P, pi)
    rho = env.p0
    d_pi = jnp.linalg.solve((jnp.eye(env.state_space) - env.gamma * p_pi.T), (1 - env.gamma) * rho)
    d_pi /= d_pi.sum()
    return d_pi


def get_q_value(env, pi):
    v = get_value(env, pi)
    v_pi = jnp.einsum('xay,y->xa', env.P, v)
    q = env.R + env.gamma * v_pi
    return q


def kl_fn(p, q, ds, reduce=True):
    _kl = (p * jnp.log((p + 1e-4) / (q + 1e-4)))
    _kl = (ds * _kl)
    if reduce:
        _kl = _kl.sum()
    return _kl


def pg(pi, adv, *args):
    pi = pi * jnp.exp(1 + config.eta * adv)
    pi = jax.nn.softmax(pi)
    return pi, {}


def pg_loss(pi, pi_old, d_s, adv, reduce=True):
    loss = (pi_old * jnp.log(pi) * adv)
    loss = (d_s * loss)
    if reduce:
        loss = loss.sum()
    return loss


def ppo(pi, adv, *args):
    pi = pi * jnp.exp(config.eta * adv)
    pi = jax.nn.softmax(pi)
    return pi, {}


def ppo_loss(pi, pi_old, d_s, adv, reduce=True):
    loss = (d_s * (pi * adv))
    if reduce:
        loss = loss.sum()
    return loss


def entropy_fn(pi):
    return - (pi * jnp.log(pi)).sum(1)


def approx_pi(pi_fn, env):
    kl_grad = jax.jacobian(kl_fn, argnums=1)
    d_pi = jax.jacobian(pi_fn)
    eval_pi = functools.partial(get_value, env)

    def pi_landscape(pi, pi_old, d_s, adv):
        loss = pi_fn(pi, pi_old, d_s, adv, reduce=False) - config.eta * kl_fn(pi_old, pi, d_s, reduce=False)
        return loss

    def loss_fn(pi, pi_old, d_s, adv):
        loss = pi_fn(pi, pi_old, d_s, adv)
        kl = config.eta * kl_fn(pi_old, pi, d_s, reduce=True)
        return loss + kl

    d_loss = jax.value_and_grad(loss_fn)

    def _fn(pi, adv, d_s, lr):
        pi_old = pi.copy()
        total_loss = 0
        avg_improve = 0
        step = 0
        while True:
            v_old = eval_pi(pi)
            loss, grad = d_loss(pi, pi_old, d_s, adv)
            log_pi = jnp.log(pi) + lr * grad
            pi = jax.nn.softmax(log_pi)
            grad_norm = jnp.linalg.norm(grad)
            total_loss += loss
            v_half = eval_pi(pi)
            improve = (v_half - v_old)
            avg_improve += (improve).mean()
            _kl_grad = kl_grad(pi_old, pi, d_s)
            pi_grad = d_pi(pi, pi_old, d_s, adv)
            kl_norm = jnp.linalg.norm(_kl_grad)
            pi_norm = jnp.linalg.norm(pi_grad)
            eqilibrium = jnp.linalg.norm(pi_grad - 1 / config.eta * _kl_grad)
            step += 1
            if jnp.linalg.norm(improve) < 1e-5 or step > config.opt_epochs:
                break
        total_loss = total_loss / (step + 1)
        avg_improve = avg_improve / (step + 1)
        landscape = pi_landscape(pi, pi_old, d_s, adv)
        return pi, landscape, {"pi/loss": total_loss,
                               "pi/grad_norm_iter": grad_norm,
                               "pi/equi": eqilibrium,
                               "pi/kl_grad": kl_norm,
                               "pi/pi_grad": pi_norm,
                               "pi/improve": avg_improve
                               }

    return _fn


def get_pi_star(q):
    pi = np.zeros_like(q)
    idx = q.argmax(1)
    for i in range(pi.shape[0]):
        pi[i, idx[i]] = 1
    return pi


def policy_iteration(env, pi_fn, pi_approx_fn, max_steps=10, key_gen=None):
    adv_star, d_star, pi_star, v_star = get_star(env)
    policies = [(pi_star, v_star, adv_star, d_star)]
    lr = config.pi_lr
    eta = config.eta
    # pi = jnp.ones(shape=(env.state_space, env.action_space))
    for global_step in tqdm.trange(max_steps):
        pi = jax.random.uniform(key=next(key_gen), shape=(env.state_space, env.action_space))
        pi /= pi.sum(axis=-1, keepdims=True)
        pi_old = pi.clone()
        d_s = get_dpi(env, pi_old)
        ppo_pi, ppo_adv = get_pi(env, pi_old, ppo, config.eta)
        pg_pi, pg_adv = get_pi(env, pi_old, pg, config.eta)
        if config.agent == "pg":
            pi = pg_pi.clone()
        else:
            pi = ppo_pi.clone()
        # assert np.allclose((pi_old * adv).sum(1), np.zeros(env.state_space), atol=1e-4)
        step_kl = kl_fn(pi_old, pi, d_s, reduce=True)
        policies.append((pg_pi, ppo_pi, pg_adv))

        entropy = (d_s * entropy_fn(pi_old)).sum(0)
        v = get_value(env, pi)
        v_soft = get_soft_value(env, pi)
        kl_star = kl_fn(pi_star, pi, d_star)
        # pi_approx, landscape, extra = pi_approx_fn(pi_old.clone(), adv, d_s, lr)
        # v_approx = get_value(env, pi_approx)
        # v_gap_approx = jnp.linalg.norm(v - v_approx)
        v_gap_star = jnp.linalg.norm(v - v_star, 1)
        v_gap_soft = jnp.linalg.norm(v_soft - v_star, 1)
        v_gap_ = jnp.linalg.norm(v_soft - v, 1)
        if config.render:
            render(v, v_soft, pi, global_step)
        stats = {
            "train/true_return": v[0],
            "train/soft_return": v_soft[0],
            "train/v_gap_star": v_gap_star,
            "train/v_gap_soft": v_gap_soft,
            "train/v_gap": v_gap_,
            "train/entropy": entropy,
            "train/kl_star": kl_star,
            "train/kl": step_kl,
        }
        save_stats(stats, global_step)
    with open(f"policies:{config.agent}.pkl", "wb") as f:
        pickle.dump(policies, f)

    return policies


def get_star(env):
    v_star = v_iteration(env)
    q_star = q_iteration(env)
    pi_star = get_pi_star(q_star)
    d_star = get_dpi(env, pi_star)
    adv_star = ((q_star - jnp.expand_dims(v_star, 1)))
    return adv_star, d_star, pi_star, v_star


def plot_loss_fn_surface(policies, env, title):
    with open(f"policies:{config.agent}.pkl", "rb") as f:
       policies = pickle.load(f)
    n = len(policies)
    ppo_data = []
    pg_data = []
    pi_star, v_star, *_ = policies[0]
    # for a given eta, and a given advantage the next policy in the same policy class (still optimal)
    # but lower entropy
    for idx in range(1, n):
        pg, ppo, adv = policies[idx]
        v_pg = get_soft_value(env, pg)
        v_ppo = get_soft_value(env, ppo)
        ds_pg = get_dpi(env, pg)
        ds_ppo = get_dpi(env, ppo)
        e_pg = (ds_pg * entropy_fn(pg)).sum()
        e_ppo = (ds_ppo * entropy_fn(ppo)).sum()
        ppo_data.append((e_ppo, v_ppo[0], 0))
        pg_data.append((e_pg, v_pg[0], 1))

    fig, ax = plt.subplots(1, 1)
    out = {0: "ppo", 1: "pg"}
    for agent_data in [pg_data, ppo_data]:
        e, v, label = list(zip(*agent_data))
        im = ax.scatter(e, v, label=out[label[0]])
    ax.set_xlabel("entropy")
    ax.set_ylabel("v_0")
    ax.set_title(f"{title}_policy_path")
    plt.legend()
    # wandb.Image(fig)
    config.tb.plot("policy_space", fig, global_step=config.max_steps)
    plt.savefig(f"{title}_policy_path")


#  print(loses)

# s,a
# for every policy
# evaluate loss
# plott loss_value vs policy indx


def get_pi(env, pi, pi_fn, eta):
    # v = get_value(env, pi)
    v = get_soft_value(env, pi)
    q = get_q_value(env, pi)
    adv = q - jnp.expand_dims(v, 1)
    pi, _ = pi_fn(pi, adv, eta)
    pi += 1e-5
    pi /= pi.sum(1, keepdims=True)
    return pi, adv


def render(v, v_soft, pi, global_step):
    # pi_greedy, _ = plot_pi(q, title=f"pi_greedy:{global_step}")
    # config.tb.add_figure("pi_greedy", pi_greedy, global_step=global_step)
    v = v.reshape((config.grid_size, config.grid_size))
    # q = q.reshape((config.grid_size, config.grid_size, 4)).max(-1)
    v = plot_grid(v, title=f"v:{global_step}")
    config.tb.add_figure("v", v, global_step=global_step)

    v_soft = v_soft.reshape((config.grid_size, config.grid_size))
    # q = q.reshape((config.grid_size, config.grid_size, 4)).max(-1)
    v_soft = plot_grid(v_soft, title=f"v_soft:{global_step}")
    config.tb.add_figure("v", v_soft, global_step=global_step)

    # q_star = plot_grid(q, title=f"q:{global_step}")
    # config.tb.add_figure("q", q_star, global_step=global_step)

    pi_plot, _ = plot_pi(pi, title=f"pi:{global_step}")
    config.tb.add_figure("pi", pi_plot, global_step=global_step)

    # pi_plot, _ = plot_pi(pi_approx, title=f"pi:{global_step}")
    # config.tb.add_figure("pi_approx", pi_plot, global_step=global_step)

    # landscape = landscape.reshape((config.grid_size, config.grid_size, 4)).max(1)
    # loss_lanscape = plot_grid(landscape, title=f"loss_landscape:{global_step}")
    # config.tb.add_figure("max_loss", loss_lanscape, global_step=global_step)


def save_stats(stats, global_step):
    for k, v in stats.items():
        config.tb.add_scalar(k, v, global_step)


def main():
    key_gen = hk.PRNGSequence(config.seed)
    env = get_gridworld(config.grid_size)
    if config.agent == "ppo":
        pi_approx = approx_pi(ppo_loss, env)
        pi_fn = ppo
    else:
        pi_approx = approx_pi(pg_loss, env)
        pi_fn = pg

    policies = policy_iteration(env, pi_fn=pi_fn, pi_approx_fn=pi_approx, key_gen=key_gen,
                                max_steps=config.max_steps)
    plot_loss_fn_surface(policies, env, title=f"surface_plot_0_{str(config.eta).replace('.', '_')}")
    print("done")
    config.tb.run.finish()


if __name__ == '__main__':
    main()
