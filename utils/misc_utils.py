import itertools

from jax import numpy as jnp

import numpy as np

import config
import jax

from experiments.plot_fn import gridworld_plot_sa


def v_iteration(env, n_iterations=1000, eps=1e-6):
    v = jnp.zeros(env.state_space)
    for _ in range(n_iterations):
        v_new = (env.R + jnp.einsum("xay,y->xa", env.gamma * env.P, v)).max(1)
        # if jnp.linalg.norm(v - v_new) < eps:
        #    break
        v = v_new.clone()
    return v


def q_iteration(env, n_iterations=1000, eps=1e-6):
    q = jnp.zeros((env.state_space, env.action_space))
    for _ in range(n_iterations):
        q_star = q.max(1)
        q_new = (env.R + jnp.einsum("xay,y->xa", env.gamma * env.P, q_star))
        # if jnp.linalg.norm(q - q_new) < eps:
        #    break
        q = q_new.clone()
    return q


@jax.partial(jax.jit, static_argnums=(0,))
def get_value(env, pi):
    p_pi = jnp.einsum('xay,xa->xy', env.P, pi)
    r_pi = jnp.einsum('xa,xa->x', env.R, pi)
    v = jnp.linalg.solve((jnp.eye(env.state_space) - env.gamma * p_pi), r_pi)
    return v


@jax.partial(jax.jit, static_argnums=(0,))
def get_dpi(env, pi):
    p_pi = jnp.einsum('xay,xa->xy', env.P, pi)
    rho = env.p0
    d_pi = jnp.linalg.solve((jnp.eye(env.state_space) - env.gamma * p_pi.T), (1 - env.gamma) * rho)
    d_pi /= d_pi.sum()
    return d_pi


@jax.partial(jax.jit, static_argnums=(0,))
def get_q_value(env, pi):
    v = get_value(env, pi)
    v_pi = jnp.einsum('xay,y->xa', env.P, v)
    q = env.R + env.gamma * v_pi
    return q


def kl_fn(p, q, ds):
    _kl = ((p + 1e-6) * jnp.log((p + 1e-6) / (q + 1e-6))).sum(1)
    _kl = (ds * _kl).sum()
    # assert _kl >=0
    return jnp.clip(_kl, 0.)


def pg(pi, adv, eta):
    # This number might be negative
    pi = pi * (1 + eta * adv)
    pi = pi / pi.sum(1, keepdims=True)
    return pi


def pg_clip(pi, adv, eta):
    # This number might be negative
    pi = pi * (1 + eta * adv)
    if not is_prob_mass(pi):
        pi = jnp.clip(pi, a_min=1e-3)
        pi = pi / pi.sum(1, keepdims=True)
    return pi


def ppo(pi, adv, eta):
    pi = pi.clone()
    pi = pi * (jnp.exp(eta * adv))
    denom = pi.sum(1, keepdims=True)
    pi = pi / denom
    assert (denom >= 1 - 1e-4).all()
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


import jax


def get_pi(env, pi_old, pi_fn, eta):
    pi = pi_old.clone()
    v = get_value(env, pi)
    q = get_q_value(env, pi)
    d_s = get_dpi(env, pi)
    adv = q - jnp.expand_dims(v, 1)

    if config.use_fa:
        # d_s = jax.nn.softmax(jax.nn.softmax(adv.max(axis=1))*d_s)
        _adv = jnp.expand_dims(jnp.einsum('i,ij->j', d_s, adv), 0)
    else:
        _adv = adv
    pi = pi_fn(pi, _adv, eta)
    new_value = get_value(env, pi)
    entropy = (d_s * entropy_fn(pi)).sum()
    d_s = get_dpi(env, pi)
    kl = kl_fn(pi, pi_old, d_s)
    return pi, adv, entropy, new_value, kl, d_s


def get_star(env):
    v_star = v_iteration(env)
    q_star = q_iteration(env)
    pi_star = get_pi_star(q_star)
    d_star = get_dpi(env, pi_star)
    adv_star = q_star - jnp.expand_dims(v_star, 1)
    return pi_star, adv_star, d_star, v_star


def get_pi_star(q):
    pi = np.zeros_like(q)
    idx = q.argmax(1)
    for i in range(pi.shape[0]):
        pi[i, idx[i]] = 1
    return pi


def is_prob_mass(pg_pi):
    return jnp.allclose(pg_pi.sum(1), 1) and (pg_pi.min() >= 0).all()


import matplotlib.pyplot as plt


def policy_iteration(agent_fn, env, eta, pi, stop_criterion):
    pi_old = pi.clone()
    data = []
    v_last = jnp.zeros(shape=env.state_space)
    for step in itertools.count():
        pi, adv, e_pg, v, kl, d_s = get_pi(env, pi_old, agent_fn, eta=eta)
        print(pi)
        data.append((pi, adv, v, d_s))
        if stop_criterion(step, v, v_last) or not is_prob_mass(pi):
            break
        step += 1
        pi_old = pi
        v_last = v.copy()
    pis, advs, vs, d_s = list(map(lambda x: jnp.stack(x, 0), list(zip(*data))))
    return e_pg, pi, step, v, pis, advs, vs, d_s
