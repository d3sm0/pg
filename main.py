import jax.numpy as jnp

import config
from plot_fn import plot_vf, gridworld_plot_sa
from utils.envs import get_cliff, get_four_rooms, get_frozen_lake
import utils.misc_utils
from utils.mdp import get_gridworld


def train(env):
    def cb(step):
        return step > config.max_steps

    pi_fn = utils.misc_utils.softmax_ppo
    pi, v, *_ = utils.misc_utils.policy_iteration(env, pi_opt=pi_fn, stop_criterion=cb, eta=config.eta)
    # plot_policy_at_state(pi, action_label=labels, title=title)
    plot_vf(env, v, f"vf:eta={config.eta:.2f}", log_plot=True, step=config.max_steps)
    gridworld_plot_sa(env, pi, f"pi:eta={config.eta:.2f}", log_plot=True, step=config.max_steps)


def train_approx(env):
    def cb(step):
        return step > config.max_steps

    # pi_star, adv_star, d_star, v_star = utils.misc_utils.get_star(env)
    pi_fn = utils.misc_utils.approx_improve_pi(env, utils.misc_utils.ppo_loss, iterations=config.opt_epochs)
    pi, v, *_ = utils.misc_utils.approx_policy_iteration(env, stop_criterion=cb, pi_fn=pi_fn)
    plot_vf(env, v, f"vf:eta={config.eta:.2f}", log_plot=True, step=config.max_steps + 1)
    gridworld_plot_sa(env, pi, f"pi:eta={config.eta:.2f}", log_plot=True, step=config.max_steps + 1)


def main():
    # env = get_gridworld(config.grid_size)
    # env = get_cliff(config.gamma)
    # env = get_shamdp(config.horizon, c=config.penalty)
    env = get_four_rooms(config.gamma)
    # env = get_frozen_lake(config.gamma)
    train_approx(env)
    # train(env)

    config.tb.run.finish()


if __name__ == '__main__':
    main()
