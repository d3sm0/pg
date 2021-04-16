import haiku as hk
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import config
# from main import approx_pi
from misc_utils import ppo, pg, get_pi, get_star, pg_loss, get_dpi, get_adv, ppo_loss, entropy_fn, get_value, kl_fn
from shamdp import get_gridworld

import jax

key_gen = hk.PRNGSequence(0)

etas = np.linspace(0.1, 4, num=9)
env = get_gridworld(config.grid_size)
pi = np.ones(shape=(env.state_space, env.action_space))
# pi = jax.random.uniform(next(key_gen), shape=(env.state_space, env.action_space))
#pi[0, 1] = 0.2
#pi[0, 3] = 0.2
pi /= pi.sum(1, keepdims=True)
# pi /= pi.sum(1, keepdims=True)
pi = jnp.array(pi)
# from main import action_to_text
labels = ["left", "right", "up", "down"]
fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
axs = axs.flatten()

ppo_grad = jax.jacobian(ppo_loss)
pg_grad = jax.jacobian(pg_loss)
kl_grad = jax.jacobian(kl_fn, argnums=1)


def reverse_kl(p, q, d_s):
    return - kl_fn(p, q, d_s)


mirror_kl = jax.jacobian(reverse_kl, argnums=0)

# pi_star, *_ = get_star(env)
MAX_T = int(1e4)
eta = 1.0


# adv =adv +  jax.random.uniform(key=next(key_gen), shape=(env.state_space, env.action_space))
def approx_pi(pi, d_pi, reg):
    pi_old = pi.clone()
    adv, d_s = get_adv(env, pi)
    adv = np.zeros_like(adv)
    adv[0,1] = -1
    d_s = jnp.ones_like(d_s)
    last_v = get_value(env, pi)
    t = 0
    grads = []
    while True:
        grad_log_pi = jax.grad(lambda pi: jnp.log(pi))(pi[0, 1])
        grad_pi = jax.grad(lambda pi: pi)(pi[0, 1])
        assert jnp.allclose(grad_pi, pi[0, 1] * grad_log_pi)
        diff = (pi_old[0, 1] * grad_log_pi, grad_pi)
        grad_pi = d_pi(pi, pi_old, d_s, adv)
        grads.append(diff)
        log_pi = pi + grad_pi #- eta * reg(pi_old, pi, d_s))
        pi = jax.nn.softmax(log_pi, -1)
        # pi = jnp.abs(log_pi)/jnp.abs(log_pi).sum(1, keepdims=True)
        v = get_value(env, pi)
        if jnp.linalg.norm(v - last_v) < 1e-3 or t > MAX_T:
            break
        last_v = v
        t += 1

    return pi, v, t, grads


# ppo_grads = jnp.stack(grads_ppo)
# pg_grads = jnp.stack(grads_pi)
#
# ppo_grads = ppo_grads.reshape((-1, 4, 4, 4))[:, 0, 0]
# pg_grads = pg_grads.reshape((-1, 4, 4, 4))[:, 0, 0]
#
# axs[0].plot(ppo_grads[:, 1], label="ppo_right")
## axs[0].plot(pg_grads[:, 1], label="pg_right")
# axs[0].legend()
# axs[1].plot(ppo_grads[:, 3], label="ppo_down")
## axs[1].plot(pg_grads[:, 3], label="pg_down")
# axs[1].legend()
#
# axs[2].plot(ppo_grads[:, 0], label="ppo_left")
## axs[2].plot(pg_grads[:, 0], label="pg_left")
# axs[3].legend()
# axs[3].plot(ppo_grads[:, 3], label="ppo_up")
## axs[3].plot(pg_grads[:, 3], label="pg_up")
# plt.legend()
# plt.xlabel("iterations")
# plt.xlabel("grad_pi")
# plt.savefig("best_action")

pg_pi, v_pg, t_pg, grads_pi = approx_pi(pi, pg_grad, kl_grad)

log_pi, grad_pi = list(zip(*grads_pi))
plt.plot(log_pi, label="grad_log_pi")
plt.plot(grad_pi, label="grad_pi")
plt.legend()
plt.show()

ppo_pi, v_ppo, t, grads_ppo = approx_pi(pi, ppo_grad, kl_grad)
axs[0].hist(np.arange(env.action_space), weights=ppo_pi[0], label=f"agent=ppo:v={v_ppo[0] :.3f}:t={t}", alpha=0.5)
axs[1].hist(np.arange(env.action_space), weights=pg_pi[0], label=f"agent=pg:v={v_pg[0] :.3f}:t={t_pg}", alpha=0.5)
mirror_ppo_pi, mirror_v_ppo, mirror_t_ppo, _ = approx_pi(pi, ppo_grad, mirror_kl)
axs[2].hist(np.arange(env.action_space), weights=mirror_ppo_pi[0],
            label=f"agent=ppo_mirror:v={mirror_v_ppo[0] :.3f}:t_ppo={mirror_t_ppo}", alpha=0.5)
mirror_pg_pi, mirror_v_pg, mirror_t_pg, _ = approx_pi(pi, pg_grad, mirror_kl)
axs[3].hist(np.arange(env.action_space), weights=mirror_pg_pi[0],
            label=f"agent=pg_mirror:v={mirror_v_pg[0] :.3f}:t_pg={mirror_t_pg}", alpha=0.5)

axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()

plt.xticks(np.arange(env.action_space), labels)
plt.savefig(f"uniform_eta=1_random_adv=1")
plt.tight_layout()
plt.show(block=False)
plt.pause(5)
plt.close()

# for idx, eta in enumerate(etas):
# ax = axs[idx]
# yet here is a tiny difference in the gradient
# print(_pg_grad - _ppo_grad)

# e_ppo = (d_s * entropy_fn(ppo_pi)).sum()
# pg_pi = pi.clone()
# for _ in range(n_steps):
#    adv, d_s = get_adv(env, pg_pi)
#    pg_pi, v_pg, _ = pg_approx(pi, adv, d_s)
# _, d_s = get_adv(env, pg_pi)
# e_pg = (d_s * entropy_fn(pg_pi)).sum()
# ax.hist(np.arange(env.action_space), weights=pg_pi[0], label=f"agent=pg:delta:h={e_pg :.3f}:v={v_pg[0] :.3f}",
#        alpha=0.5)
# ax.set_title(f"eta:{eta:.2f}")
