import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import config
from experiments.plot_fn import gridworld_plot_sa, plot_policy_at_state, plot_vf, chain_plot_sa, chain_plot_vf
from utils.mdp import get_gridworld, get_shamdp
from utils.envs import get_four_rooms
from utils.misc_utils import ppo, get_star, pg_clip, policy_iteration, pg


def plot_ds(d_s, ax=None, label=None):
    n_states = d_s.shape[0]
    if ax is None:
        ax = plt.gca()
    # ax.hist(np.arange(n_states), weights=d_s, label=label)
    ax.imshow(d_s)
    ax.set_xlabel("state_idx")
    ax.set_ylabel("d_pi(s)")


def plot_adv(adv, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.imshow(adv.T)
    # labels = ["right", "left_0", "left_1", "left_2"]
    # ax.set_ylabel()
    # ax.set_xlim(labels)
    ax.set_ylabel("actions")
    ax.set_xlabel("states")


def plot_params(params, ax=None):
    if ax is None:
        ax = plt.gca()
    _, n_actions = params.shape
    ax.scatter(np.arange(n_actions), params)
    ax.set_ylabel("avg_grad(action)")
    ax.set_xlabel("action")


def plot_batch(batch, plot_fn, grid_size=(4, 4), path=""):
    n, *_ = batch.shape
    fig, axs = plt.subplots(*grid_size, figsize=(12, 6))
    axs = axs.flatten()
    batch = batch[:grid_size[0] ** 2]
    for idx, element in enumerate(batch):
        plot_fn(element, ax=axs[idx])
    plt.tight_layout()
    # plt.savefig(path)
    # plt.close()


def main():
    # etas = np.linspace(-7., -2., num=9)
    # etas = np.linspace(0.1, 2., num=9)
    # env = get_four_rooms(gamma=config.gamma)
    # horizon 2, 0.9 (4, 0.9), (6, 0.2)
    env = get_shamdp(horizon=4, c=0.1)
    n_parameters = 1 if config.use_fa else env.state_space
    pi = np.ones(shape=(n_parameters, env.action_space))

    def stop_criterion(t, v, v_old):
        return (jnp.linalg.norm(v - v_old) < config.eps and t >= 5) or t >= config.max_steps

    pi_star, adv_star, d_star, v_star = get_star(env)
    pi /= pi.sum(1, keepdims=True)
    pi = jnp.array(pi)
    agents = {"pg_clip": pg_clip}
    # agents = {"pg_clip": pg_clip, "ppo": ppo}
    eta = 0.01
    for agent, agent_fn in agents.items():
        # ax = axs[idx]
        e_pg, pg_pi, t, v_pg, pi_s, advs, _, d_s = policy_iteration(agent_fn, env, eta, pi, stop_criterion)
        plot_batch(pi_s, plot_fn=plot_ds)
        print(pg_pi)
        # plot_batch(d_s, plot_fn=plot_ds, path=f"plots/shamdp/state_distribution_prob_{43}")
        # plot_batch(advs, plot_fn=plot_adv, path=f"plots/shamdp/adv_{43}")
        # plot_batch(avg_adv, plot_fn=plot_params) #, path=f"plots/shamdp/avg_adv{44}")
        plt.show()

        # for idx, eta in enumerate(etas):
        # title = f"{agent}:h={e_pg :.3f}:gap={(env.p0 * (v_star - v_pg)).sum():.3f}:t={t}:eta={eta:.2f}"

        # if config.use_fa:
        #    plot_policy_at_state(pg_pi, action_label=labels, title=title, ax=ax)
        # else:
        #    gridworld_plot_sa(env, pi, title, ax=ax)
        # plot_vf(env, v_pg, title, ax=ax)
        # chain_plot_sa(env, pi_s[:, 0, 0], title, ax=ax)
        # chain_plot_vf(env, v_pg, title, ax=ax)
        # plt.show()
        # plt.legend(["left", "right"])
        # plt.savefig(f"plots/shamdp/prob_right_agent={agent}:size={config.grid_size}")
        # plt.show(block=False)
        # plt.pause(5)
        # plt.close()


if __name__ == '__main__':
    main()
