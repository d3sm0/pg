# exact update of every policy and parameterfs
import functools

import jax
import jax.numpy as jnp

import config
# from misc_utils import get_value, kl_fn, get_dpi, entropy_fn, pg_loss, ppo_loss, ppo, pg, get_pi
from env_utils import eval_policy
from experiments.plot_fn import gridworld_plot_sa, plot_vf
from utils.envs import get_four_rooms
from utils.misc_utils import get_star, entropy_fn, get_dpi, get_pi, kl_fn, get_value, ppo_loss, ppo, pg_loss, \
    is_prob_mass, pg_clip
from mdp import get_gridworld


def approx_pi(pi_fn, env):
    kl_grad = jax.jacobian(kl_fn, argnums=1)
    d_pi = jax.jacobian(pi_fn)
    eval_pi = functools.partial(get_value, env)

    def loss_fn(pi, pi_old, d_s, adv):
        loss = pi_fn(pi, pi_old, d_s, adv)
        kl = 1 / config.eta * kl_fn(pi_old, pi, d_s)
        return loss + kl

    d_loss = jax.value_and_grad(loss_fn)

    def _fn(pi, adv, d_s):
        pi_old = pi.copy()
        total_loss = 0
        avg_improve = 0
        step = 0
        while True:
            v_old = eval_pi(pi)
            loss, grad = d_loss(pi, pi_old, d_s, adv)
            log_pi = jnp.log(pi) + config.pi_lr * grad
            pi = jax.nn.softmax(log_pi)
            grad_norm = jnp.linalg.norm(grad)
            total_loss += loss
            v_half = eval_pi(pi)
            improve = (v_half - v_old)
            avg_improve += improve.mean()
            _kl_grad = kl_grad(pi_old, pi, d_s)
            pi_grad = d_pi(pi, pi_old, d_s, adv)
            kl_norm = jnp.linalg.norm(_kl_grad)
            pi_norm = jnp.linalg.norm(pi_grad)
            eqilibrium = jnp.linalg.norm(pi_grad + config.eta * _kl_grad)
            step += 1
            if jnp.linalg.norm(improve) < 1e-5 or step > config.opt_epochs:
                break
        total_loss = total_loss / (step + 1)
        avg_improve = avg_improve / (step + 1)

        return pi, v_half, {"pi/loss": total_loss,
                            "pi/grad_norm_iter": grad_norm,
                            "pi/equi": eqilibrium,
                            "pi/kl_grad": kl_norm,
                            "pi/pi_grad": pi_norm,
                            "pi/improve": avg_improve,
                            }

    return _fn


def policy_iteration(env, pi_fn, pi_approx_fn, max_steps=10):
    pi_star, adv_star, d_star, v_star = get_star(env)
    pi = jnp.ones(shape=(1, env.action_space))
    pi /= pi.sum(axis=-1, keepdims=True)
    global_step = 0
    last_v = get_value(env, pi)
    while True:
        pi_old = pi.clone()
        d_s = get_dpi(env, pi_old)
        entropy = (d_s * entropy_fn(pi_old)).sum(0)
        pi, adv, entropy_new, v, kl, _ = get_pi(env, pi_old, pi_fn, config.eta)
        # pi_approx, v_approx, pi_stats = pi_approx_fn(pi, adv, d_s)
        kl_star = kl_fn(pi_star, pi, d_star)
        v_gap_star = jnp.linalg.norm(v_star - last_v)
        sampled_return = eval_policy(env, pi)
        stats = {
            "train/v_star": v_star[10],
            "train/return": v[10],
            "train/sampled_return": sampled_return,
            "train/v_gap_star": v_gap_star,
            "train/entropy": entropy,
            "train/kl_star": kl_star,
            "train/kl": kl,
            "train/t": global_step,
            # **pi_stats,
        }
        save_stats(stats, global_step)
        condition = kl
        # condition = jnp.linalg.norm(v - last_v)
        eps = config.eps * (1 - config.gamma) / 2 * config.gamma
        if condition < eps or global_step > max_steps:
            # print(sampled_return - v[10])
            # gridworld_plot_sa(env, pi, f"pi:eta={config.eta:.2f}", log_plot=True, step=global_step)
            # plot_vf(env, v, f"vf:eta={config.eta:.2f}")
            break
        if not is_prob_mass(pi):
            break
        global_step += 1
        last_v = v


def save_stats(stats, global_step):
    for k, v in stats.items():
        config.tb.add_scalar(k, v, global_step)


def main():
    # env = get_gridworld(config.grid_size)
    # env = get_f(config.grid_size)
    env = get_four_rooms(config.gamma)
    if config.agent == "ppo":
        pi_approx = approx_pi(ppo_loss, env)
        pi_fn = ppo
    elif config.agent == "pg_clip":
        pi_approx = approx_pi(pg_loss, env)
        pi_fn = pg_clip
    else:
        pi_approx = approx_pi(pg_loss, env)
        pi_fn = pg_clip
        # pi_fn = pg

    policy_iteration(env, pi_fn=pi_fn, pi_approx_fn=pi_approx, max_steps=config.max_steps)
    print("done")
    config.tb.run.finish()


if __name__ == '__main__':
    main()
