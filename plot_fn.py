import glob
import pickle

import jax.nn
import matplotlib.pyplot as plt
import jax.numpy as jnp
from matplotlib import pyplot as plt

from misc_utils import get_value, get_q_value
from shamdp import get_gridworld


def ppo(pi, adv, eta):
    pi = pi.clone()
    pi = pi * jnp.exp(adv * eta)
    out = pi / pi.sum(1)
    print(out)
    return out


def pg(pi, adv, eta):
    pi = pi.clone()
    pi = pi * jnp.exp(1 + adv * eta)
    out = pi / pi.sum(1)
    print(out)
    return out

def aggregate_polices():
    path = "data/*.pkl"
    polices = {}
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.xlabel("entropy")
    plt.ylabel("v_0")
    plt.xlim(0, 1.5)
    ax[0].grid(True)
    ax[1].grid(True)
    for p in glob.glob(path):
        with open(p, "rb") as f:
            pi, meta = pickle.load(f)
            idx = 1
            if "pg" in meta["agent"]:
                idx = 0
            c = meta["eta"]
            _, _, im = plot_loss_fn_surface(pi, c, meta["agent"], fig, ax[idx])

    fig.colorbar(im, ax=ax.ravel())
    fig.subplots_adjust(right=0.8)
    ax[0].set_title("pg")
    ax[1].set_title("ppo")
    plt.savefig("plot_policies")
    plt.show()


def plot_loss_fn_surface(policies, c, label, fig=None, ax=None):
    _, entropy, v_soft = list(zip(*policies))
    im = ax.scatter(entropy, v_soft, c=(c,) * len(entropy), alpha=0.5)
    return fig, ax, im

env = get_gridworld(2)
pi_old = jnp.ones(shape=(env.state_space, env.action_space))
pi_old /= pi_old.sum(1, keepdims=True)
# from main import action_to_text
eta = 4
pi = pi_old.clone()
v = get_value(env, pi)
q = get_q_value(env, pi)
adv = q - jnp.expand_dims(v, 1)
ppo_next = ppo(pi, adv, eta)
pg_next = pg(pi, adv, eta)
print(ppo_next, pg_next)

