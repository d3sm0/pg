import jax.nn
from jax import numpy as jnp

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


def get_soft_value(env, pi, eta):
    p_pi = jnp.einsum('xay,xa->xy', env.P, pi)
    # for large eta -> pi become deterministic  entropy goes to 0
    # for small eta pi becomes more randomized and i don't move
    r_pi = jnp.einsum('xa,xa->x', env.R, pi) - 1 / eta * jnp.einsum('xa,xa->x', pi, jnp.log(pi + 1e-6))
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


def kl_fn(p, q, ds):
    _kl = (p * jnp.log((p + 1e-4) / (q + 1e-4))).sum(1)
    _kl = (ds * _kl).sum()
    return _kl


def pg(pi, adv, eta):
    pi = pi * (1 + eta * adv)
    pi = escort(pi)
    return pi


def ppo(pi, adv, eta):
    pi = pi.clone()
    pi = pi * jnp.exp(eta * adv)
    pi = pi / pi.sum(1, keepdims=True)
    return pi


def pg_loss(pi, pi_old, d_s, adv):
    loss = (pi_old * jnp.log(pi) * adv).sum(1)
    loss = (d_s * loss).sum()
    return loss


def ppo_loss(pi, pi_old, d_s, adv):
    loss = (d_s * (pi * adv).sum(1)).sum()
    return loss


def entropy_fn(pi):
    return - (pi * jnp.log(pi + 1e-6)).sum(1)


def get_pi(env, pi_old, pi_fn, eta):
    pi = pi_old.clone()
    v = get_value(env, pi)
    q = get_q_value(env, pi)
    d_s = get_dpi(env, pi)
    adv = q - jnp.expand_dims(v, 1)
    pi = pi_fn(pi, adv, eta)
    new_value = get_value(env, pi)
    entropy = (d_s * entropy_fn(pi)).sum()
    d_s = get_dpi(env, pi)
    kl = kl_fn(pi_old, pi, d_s)
    return pi, adv, entropy, new_value, kl


def escort(log_pi):
    pi = jnp.abs(log_pi) / jnp.abs(log_pi).sum(1, keepdims=True)
    return pi


def get_pi_from_log(env, log_pi, pi_fn, eta):
    # pi = jax.nn.softmax(log_pi)
    pi = escort(log_pi)
    v = get_value(env, pi)
    q = get_q_value(env, pi)
    d_s = get_dpi(env, pi)
    adv = q - jnp.expand_dims(v, 1)
    log_pi = pg(log_pi, adv, eta)
    new_value = get_value(env, pi)
    entropy = (d_s * entropy_fn(pi)).sum()
    d_s = get_dpi(env, pi)
    kl = kl_fn(pi, jax.nn.softmax(log_pi), d_s)
    return log_pi, adv, entropy, new_value, kl


def get_star(env):
    v_star = v_iteration(env)
    q_star = q_iteration(env)
    pi_star = get_pi_star(q_star)
    d_star = get_dpi(env, pi_star)
    adv_star = ((q_star - jnp.expand_dims(v_star, 1)))
    return pi_star, adv_star, d_star, v_star


def get_pi_star(q):
    pi = np.zeros_like(q)
    idx = q.argmax(1)
    for i in range(pi.shape[0]):
        pi[i, idx[i]] = 1
    return pi
