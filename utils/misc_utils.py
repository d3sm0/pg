import functools
import itertools

import jax
import numpy as np
from jax import numpy as jnp

import config

import haiku as hk


def v_iteration(env, n_iterations=1000):
    v = jnp.zeros(env.state_space)
    for _ in range(n_iterations):
        v_new = (env.R + jnp.einsum("xay,y->xa", env.gamma * env.P, v)).max(1)
        v = v_new.clone()
    return v


def q_iteration(env, n_iterations=1000, eps=1e-6):
    q = jnp.zeros((env.state_space, env.action_space))
    for _ in range(n_iterations):
        q_star = q.max(1)
        q_new = (env.R + jnp.einsum("xay,y->xa", env.gamma * env.P, q_star))
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
def get_q_value(env, pi, v):
    v = get_value(env, pi)
    q_pi = jnp.einsum('xay,y->xa', env.P, v)
    q = env.R + env.gamma * q_pi
    return q


def kl_fn(weight, p, q):
    _kl = ((p + 1e-6) * jnp.log((p + 1e-6) / (q + 1e-6))).sum(1)
    _kl = (weight * _kl).sum()
    return jnp.clip(_kl, 0.)


def line_search(pi, adv, eta_0, step_size):
    eta = eta_0
    while True:
        pi_new = softmax_ppo(pi, adv, eta)
        if not is_prob_mass(pi_new):
            return softmax_ppo(pi, adv, eta-step_size)
        eta = eta + step_size


@jax.jit
def softmax_ppo(pi_old, adv, eta, clip=0.):
    pi = pi_old * (1 + eta * adv)
    pi = jnp.clip(pi, a_min=clip)
    pi = pi / pi.sum(1, keepdims=True)
    return pi


# TODO implement safe exp
@jax.jit
def mdpo(pi, adv, eta):
    pi = pi * (jnp.exp(eta * adv))
    denom = pi.sum(1, keepdims=True)
    pi = pi / denom
    # assert (denom >= 1 - 1e-4).all()
    return pi


def pg_loss(pi, pi_old, d_s, adv):
    loss = (pi_old * jnp.log(pi) * adv).sum(1)
    loss = (d_s * loss).sum()
    return loss


def ppo_loss(pi, pi_old, d_s, adv):
    loss = jnp.einsum('s, sa->a', d_s, pi * adv).sum()
    return loss


@jax.partial(jax.jit, static_argnums=(0,))
def entropy_fn(env, pi):
    d_s = get_dpi(env, pi)
    entropy = - (pi * jnp.log(pi + 1e-6)).sum(1)
    out = jnp.einsum('s,sa', d_s, entropy)
    return out


key_gen = hk.PRNGSequence(0)


def sample_value(env, pi, n_samples):
    vs = []
    for _ in range(n_samples):
        v = _sample_value(env, pi)
        vs.append(v)
    return jnp.stack(vs, 0).mean(0)


def _sample_value(env, pi):
    s = env.reset()
    state_idx = s.argmax()
    total_return = 0
    for t in itertools.count():
        p = jnp.einsum("s, sa->a", s, pi)
        action = jax.random.choice(key=next(key_gen), a=env.action_space, p=p).item()
        s, r, done, info = env.step(action)
        total_return += (env.gamma ** t) * r
        if done or t > 1e2:
            break
    v_hat = np.zeros(env.state_space)
    v_hat[state_idx] = total_return
    return jnp.array(v_hat) / (1 - env.gamma)


# change shape of advantage funciton
# TODO refactor this taking pi and jit everything
def improve_pi(env, pi_old, pi_fn, eta):
    pi = pi_old
    _adv, v_k = get_adv(env, pi_old)
    #if config.agent_id is "softmax_ppo":
    #    pi = line_search(pi.clone(), _adv, eta_0=eta, step_size=0.01)
    #else:
    pi = pi_fn(pi, _adv, eta)
    v_kp1 = get_value(env, pi)
    kl = kl_fn(get_dpi(env, pi), pi, pi_old)

    stats = {
        "pi/delta_v": (env.p0 @ (v_kp1 - v_k)),
        "pi/return": (env.p0 @ v_kp1),
        "pi/kl": kl
    }
    return pi, v_kp1, stats


def get_adv(env, pi):
    v_k = get_value(env, pi)
    q = get_q_value(env, pi, v_k)
    d_s = get_dpi(env, pi)
    adv = q - jnp.expand_dims(v_k, 1)
    if config.use_fa:
        _adv = jnp.expand_dims(jnp.einsum('s,sa->a', d_s, adv), 0)
    else:
        _adv = adv
    return _adv, v_k


def get_star(env):
    v_star = v_iteration(env)
    q_star = q_iteration(env)
    pi_star = get_pi_star(q_star)
    d_star = get_dpi(env, pi_star)
    adv_star = q_star - jnp.expand_dims(v_star, 1)
    return pi_star, v_star


def get_pi_star(q):
    pi = np.zeros_like(q)
    idx = q.argmax(1)
    for i in range(pi.shape[0]):
        pi[i, idx[i]] = 1
    return pi


def is_prob_mass(pg_pi):
    return jnp.allclose(pg_pi.sum(1), 1) and (pg_pi.min() >= 0).all()


def save_stats(stats, global_step):
    for k, v in stats.items():
        config.tb.add_scalar(k, v, global_step)


def init_pi(env):
    if config.use_fa:
        state_params = 1
    else:
        state_params = env.state_space
    pi = jnp.ones(shape=(state_params, env.action_space))
    pi /= pi.sum(axis=-1, keepdims=True)
    return pi


def policy_iteration(env, pi_opt, eta, stop_criterion, policy=None):
    if policy is None:
        policy = init_pi(env)
    data = []
    value = get_value(env, policy)
    stats = {"pi/return": env.p0 @ value}
    prob_right = policy[0, 0]
    stats["pi/prob_right"] = prob_right
    save_stats(stats, global_step=0)
    for step in itertools.count(1):
        policy, value, stats = improve_pi(env, policy, pi_opt, eta=eta)
        data.append((policy, value))
        prob_right = policy[0, 0]
        stats["pi/prob_right"] = prob_right
        save_stats(stats, global_step=step)
        if stop_criterion(step) or not is_prob_mass(policy):
            break
    pis, vs = list(map(lambda x: jnp.stack(x, 0), list(zip(*data))))
    return policy, value, pis, vs


def approx_policy_iteration(env, pi_fn, stop_criterion, write_stats=False):
    policy = init_pi(env)
    data = []
    value = jnp.zeros(shape=env.state_space)
    for step in itertools.count():
        adv, _ = get_adv(env, policy)
        d_s = get_dpi(env, policy)
        policy, value, stats = pi_fn(policy, adv, d_s)
        data.append((policy, value))
        if write_stats:
            save_stats(stats, global_step=step)
        if stop_criterion(step) or not is_prob_mass(policy):
            break
    pis, vs = list(map(lambda x: jnp.stack(x, 0), list(zip(*data))))
    return policy, value, pis, vs


def approx_improve_pi(env, pi_fn, iterations=10):
    eval_pi = functools.partial(get_value, env)

    def loss_fn(pi, pi_old, d_s, adv):
        loss = pi_fn(pi, pi_old, d_s, adv)
        kl = config.eta * kl_fn(d_s, pi_old, pi)
        return loss - kl

    d_loss = jax.value_and_grad(loss_fn)

    def _fn(pi, adv, d_s):
        pi_old = pi.copy()
        v_old = eval_pi(pi)
        for t in range(iterations):
            loss, grad = d_loss(pi, pi_old, d_s, adv)
            pi = jnp.log(pi) + config.lr * grad
            pi = jax.nn.softmax(pi)
        v = eval_pi(pi)
        d_s = get_dpi(env, pi)
        kl = kl_fn(d_s, pi, pi_old)
        stats = {
            "pi/return": (env.p0 @ v),
            "pi/improve": (env.p0 @ (v - v_old)),
            "pi/kl_fwd": kl
        }
        return pi, v, stats

    return _fn
