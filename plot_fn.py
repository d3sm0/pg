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


import random

import jax.numpy as np
import math
import matplotlib.pyplot as plt
import time
# np.set_printoptions(precision=4, suppress=True, threshold=200)
from emdp import actions

import config

task0 = None


def plot_step_4states(env, pis, rew, hierarchical_policy, vf, ir, task_idx):
    global task0
    pis, rew, hierarchical_policy, vf = pis.primal, rew.primal, hierarchical_policy.primal, vf.primal
    if task_idx != len(pis) - 1:
        task0 = pis, rew, hierarchical_policy, vf
        return

    # pis0, r0, mu0, vf0 = task0
    # assert np.allclose(pis, task0[0])
    if len(pis) > 1:
        pis, (r0, r1), (mu0, mu1), (vf0, vf1) = task0[0], (task0[1], rew), (task0[2], hierarchical_policy), (
        task0[3], vf)
    else:
        r0, mu0, vf0 = rew, hierarchical_policy, vf

    r, c = 3, 4

    fig = plt.figure(figsize=[4.5, 4.5], dpi=200.)
    idx = 0

    idx += 1
    ax = plt.subplot(r, c, idx)
    gridworld_plot_sa(env, ir[0], f"ir0", (fig, ax))

    idx += 1
    ax = plt.subplot(r, c, idx)
    gridworld_plot_sa(env, pis[0], f"pi0", (fig, ax))

    idx += 1
    plot_vf(env, r0.T[0], idx, r, c, 0)

    idx += 1
    if rew.shape[1] > 1:
        plot_vf(env, r1.T[0], idx, r, c, 1)

    if pis.shape[0] == 2:
        idx += 1
        ax = plt.subplot(r, c, idx)
        gridworld_plot_sa(env, ir[1], f"ir1", (fig, ax))

        idx += 1
        ax = plt.subplot(r, c, idx)
        gridworld_plot_sa(env, pis[1], f"pi1", (fig, ax))

        idx += 1
        plot_vf(env, r0.T[1], idx, r, c, "r0")

        idx += 1
        plot_vf(env, r1.T[1], idx, r, c, "r1")
    else:
        idx += 4

    idx += 1
    ax = plt.subplot(r, c, idx)
    gridworld_plot_sa(env, mu0, f"mu0", (fig, ax))

    idx += 1
    plot_vf(env, vf0, idx, r, c, "mu0_vf")

    if rew.shape[1] > 1:
        idx += 1
        ax = plt.subplot(r, c, idx)
        gridworld_plot_sa(env, mu1, f"mu1", (fig, ax))

        idx += 1
        plot_vf(env, vf1, idx, r, c, "mu1_vf")

    else:
        idx += 2

    fig.subplots_adjust(hspace=0.9)
    plt.tight_layout(pad=0.)
    plt.show()


def plot_step(env, pis, rews, mus, vfs, irs, single_plot=True, step=None):
    if hasattr(pis, "primal"):
        pis, rews, mus, vfs = pis.primal, rews.primal, mus.primal, vfs.primal

    num_options = pis.shape[0]
    num_tasks = vfs.shape[0]
    # pis0, r0, mu0, vf0 = task0
    # assert np.allclose(pis, task0[0])
    r, c = num_options + 1, num_tasks + 2
    if num_tasks * 2 > c:
        r += math.ceil((num_tasks * 2) / c) - 1
    if env.size == 3:
        frame = (1, 0, 0, 0)
    elif env.size == 9:
        frame = (1, 1, 1, 1)
    else:
        frame = (0, 0, 0, 0)

    if single_plot:
        fig = plt.figure(dpi=400., figsize=[c, r])  # ,
    idx = 0

    for opt_idx, (ir, pi, r_o) in enumerate(zip(irs, pis, rews.T)):
        idx += 1
        ax = plt.subplot(r, c, idx) if single_plot else None
        gridworld_plot_sa(env, ir, f"intrinsic_reward{opt_idx}", ax, frame=frame, step=step)

        idx += 1
        ax = plt.subplot(r, c, idx) if single_plot else None
        gridworld_plot_sa(env, pi, f"pi{opt_idx}", ax, frame=frame, step=step)

        for tsk_idx, r_tsk in enumerate(r_o.T):
            idx += 1
            ax = plt.subplot(r, c, idx) if single_plot else None
            plot_vf(env, r_tsk, f"option{opt_idx}/task{tsk_idx}", frame, ax, step=step)

    for tsk_idx, (mu, vf) in enumerate(zip(mus, vfs)):
        idx += 1
        ax = plt.subplot(r, c, idx) if single_plot else None
        gridworld_plot_sa(env, mu, f"mu{tsk_idx}", ax, step=step)

        idx += 1
        ax = plt.subplot(r, c, idx) if single_plot else None
        plot_vf(env, vf, f"mu{tsk_idx}", frame, ax, step)

    if single_plot:
        fig.subplots_adjust(hspace=0.9)
        plt.tight_layout(pad=0.)
        plt.savefig(f"/tmp/{time.time()}.jpg")
        plt.show()


def plot_omd_step(env, mus, vfs, P_theta, P, step=None):
    raise NotImplementedError("Plot P_pi, P is too hard to see")
    if hasattr(mus, "primal"):
        mus, vfs, P = mus.primal, vfs.primal, P.primal

    num_tasks = vfs.shape[0]

    if env.size == 3:
        frame = (1, 0, 0, 0)
    elif env.size == 9:
        frame = (1, 1, 1, 1)
    else:
        frame = (0, 0, 0, 0)

    idx = 0

    P_ = np.einsum("ast,ast->ast", P, P_theta).max(-1).transpose()
    gridworld_plot_sa(env, P_, f"P", None, frame=frame, step=step)

    for tsk_idx, (mu, vf) in enumerate(zip(mus, vfs)):
        idx += 1
        gridworld_plot_sa(env, mu, f"mu{tsk_idx}", None, step=step)

        idx += 1
        plot_vf(env, vf, f"mu{tsk_idx}", frame, None, step)


def plot_vf(env, vf, title, frame, ax, step):
    log_plot = False
    if ax is None:
        ax = plt.gca()
        log_plot = True

    x0, x1, y0, y1 = frame
    vf = vf.reshape(env.size, env.size)
    num_cols, num_rows = vf.shape

    tag = f"plots/vf/{title}"
    title = f"{title}_{vf.max():.5f}_{step}"

    ax.set_title(title, fontdict={'fontsize': 8, 'fontweight': 'medium'})

    ax.set_xlim((x0 - 0.5, num_cols - x1 - 0.5))
    ax.set_ylim((y0 - 0.5, num_rows - y1 - 0.5)[::-1])

    ax.set_xticks(np.arange(x0, num_cols - x1, 1))
    ax.xaxis.set_tick_params(labelsize=5)
    ax.set_yticks(np.arange(y0, num_rows - y1, 1))
    ax.yaxis.set_tick_params(labelsize=5)

    ax.imshow(vf, origin='lower')

    ax.set_aspect(1)

    if log_plot:
        config.tb.plot(tag, plt, step)
        plt.clf()


def plot_vf_policy(env, vf, policy, title):
    if vf.ndim == 1:
        vf = vf.reshape(*vf.shape, -1)
        policy = policy.reshape(-1, *policy.shape)
    fig = plt.figure()
    # fig.tight_layout()
    plots = len(policy)
    for idx, (r, pi) in enumerate(zip(vf.T, policy)):
        ax = plt.subplot(plots, 2, (idx * 2) + 1)
        gridworld_plot_sa(env, pi, f"{title}_pi_{idx}", (fig, ax))

        ax = plt.subplot(plots, 2, (idx * 2) + 2)
        vf_i = r.primal if hasattr(r, "primal") else r
        ax.imshow(vf_i.reshape(env.size, env.size), origin='upper')
        ax.title.set_text(f"{title}_vf_{idx}_{vf_i.max():.5f}")
    fig.subplots_adjust(hspace=0.3)
    plt.show()
    # plt.savefig(f"/tmp/option_vf_{title}")


def plot_rw_and_grad_rw(env, rw, grad, rw1, title):
    fig = plt.figure()
    fig.suptitle(title)
    fig.tight_layout()
    plots = len(rw)
    num_cols = 3
    for idx, (r0, g, r1) in enumerate(zip(rw, grad, rw1)):
        ax = plt.subplot(plots, num_cols, (idx * num_cols) + 1)
        gridworld_plot_sa(env, r0, f"rw0_{idx}", (fig, ax))
        ax = plt.subplot(plots, num_cols, (idx * num_cols) + 2)
        gridworld_plot_sa(env, g, f"grad_{idx}", (fig, ax))
        ax = plt.subplot(plots, num_cols, (idx * num_cols) + 3)
        gridworld_plot_sa(env, r1, f"rw1_{idx}", (fig, ax))
    fig.subplots_adjust(hspace=0.3)
    plt.show()
    # plt.savefig(f"/tmp/param_{title}")


def gridworld_plot_sa(env, data, title, ax=None, scale_data=False, frame=(0, 0, 0, 0), step=None):
    """
    This is going to generate a quiver plot to visualize the policy graphically.
    It is useful to see all the probabilities assigned to the four possible actions
    in each state
    """
    log_plot = False
    if ax is None:
        ax = plt.gca()
        # log_plot = False
        # assert step is not None

    if scale_data:
        scale = np.abs(data).max()
        data = data / (scale * 1.1)

    num_cols = env.ncol if hasattr(env, "ncol") else env.size
    num_rows = env.ncol if hasattr(env, "nrow") else env.size

    num_obs, num_actions = data.shape

    direction = [
        np.array((-1, 0)),  # left
        np.array((1, 0)),  # right
        np.array((0, 1)),  # up
        np.array((0, -1)),  # down
    ]

    if hasattr(data, "primal"):
        data = data.primal

    x, y = np.meshgrid(np.arange(env.size), np.arange(env.size))
    x, y = x.flatten(), y.flatten()

    for base, a in zip(direction, range(num_actions)):
        quivers = np.einsum("d,m->md", base, data[:, a])

        pos = data[:, a] > 0
        ax.quiver(x[pos], y[pos], *quivers[pos].T, units='xy', scale=2.0, color='g')

        pos = data[:, a] < 0
        ax.quiver(x[pos], y[pos], *-quivers[pos].T, units='xy', scale=2.0, color='r')

    x0, x1, y0, y1 = frame
    # set axis limits / ticks / etc... so we have a nice grid overlay
    ax.set_xlim((x0 - 0.5, num_cols - x1 - 0.5))
    ax.set_ylim((y0 - 0.5, num_rows - y1 - 0.5)[::-1])

    # ax.set_xticks(xs)
    # ax.set_yticks(ys)

    # major ticks

    ax.set_xticks(np.arange(x0, num_cols - x1, 1))
    ax.xaxis.set_tick_params(labelsize=5)
    ax.set_yticks(np.arange(y0, num_rows - y1, 1))
    ax.yaxis.set_tick_params(labelsize=5)

    # minor ticks
    ax.set_xticks(np.arange(*ax.get_xlim(), 1), minor=True)
    ax.set_yticks(np.arange(*ax.get_ylim()[::-1], 1), minor=True)

    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    ax.set_aspect(1)

    tag = f"plots/{title}"
    #if hasattr(scale, "primal"):
    #    title += f"_{float(scale.primal):.4f}"
    ##else:
    #    title += f"_{scale:.4f}"
    ax.set_title(title, fontdict={'fontsize': 8, 'fontweight': 'medium'})

    if log_plot:
        config.tb.plot(tag, plt, step)
        plt.clf()
