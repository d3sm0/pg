import jax.numpy as jnp

import config
from plot_fn import plot_vf, gridworld_plot_sa
from utils.envs import get_cliff, get_four_rooms
import utils.misc_utils


def train(env, pi_fn):
    def cb(step):
        return step > config.max_steps

    pi, v, *_ = utils.misc_utils.policy_iteration(env, pi_opt=pi_fn, stop_criterion=cb, eta=config.eta)
    # plot_policy_at_state(pi, action_label=labels, title=title)
    plot_vf(env, v, f"vf:eta={config.eta:.2f}", log_plot=True, step=config.max_steps)
    gridworld_plot_sa(env, pi, f"pi:eta={config.eta:.2f}", log_plot=True, step=config.max_steps)


def train_approx(env):
    def cb(step):
        return step > config.max_steps

    pi_fn = utils.misc_utils.approx_improve_pi(env, utils.misc_utils.ppo_loss, iterations=config.opt_epochs)
    pi, v, *_ = utils.misc_utils.approx_policy_iteration(env, stop_criterion=cb, pi_fn=pi_fn)
    plot_vf(env, v, f"vf:eta={config.eta:.2f}", log_plot=True, step=config.max_steps)
    gridworld_plot_sa(env, pi, f"pi:eta={config.eta:.2f}", log_plot=True, step=config.max_steps)


def main():
    # env = get_gridworld(config.grid_size)
    # env = get_cliff(config.gamma)
    # env = get_shamdp(config.horizon, c=config.penalty)
    env = get_four_rooms(config.gamma)
    # 4if config.agent == "mdpo":
    # 4    pi_fn = utils.misc_utils.mdpo
    # 4elif config.agent == "softmax_ppo":
    # pi_fn = utils.misc_utils.softmax_ppo
    train_approx(env)

    config.tb.run.finish()


if __name__ == '__main__':
    main()
