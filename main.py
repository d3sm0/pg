import jax.nn
import jax.numpy as jnp
import numpy as np

import config
import utils.misc_utils
from plot_fn import plot_vf, gridworld_plot_sa
from utils.envs import get_cliff, get_four_rooms
from utils.mdp import get_shamdp


def train(env, pi_fn):
    def cb(step):
        return step > config.max_steps

    # left right up down
    pi = np.ones(shape=(env.state_space, env.action_space))
    pi_star, v_star = utils.misc_utils.get_star(env)
    master_key = jax.random.PRNGKey(config.seed)
    # a_offset, b_offset = jax.random.uniform(jax.random.split(master_key)[1], shape=(2,), minval=5., maxval=10.)

    pi /= pi.sum(1, keepdims=True)
    policy = jnp.array(pi)
    gridworld_plot_sa(env, pi, f"pi={config.agent_id}:eta={config.eta:.2f}", log_plot=False, step=config.max_steps + 1)
    pi, v, *_ = utils.misc_utils.policy_iteration(env, pi_opt=pi_fn, stop_criterion=cb, eta=config.eta, policy=policy)
    plot_vf(env, v, f"vf={config.agent_id}:eta={config.eta:.2f}", log_plot=True, step=config.max_steps + 1)
    gridworld_plot_sa(env, pi, f"pi={config.agent_id}:eta={config.eta:.2f}", log_plot=True, step=config.max_steps + 1)


# def train_approx(env):
#    def cb(step):
#        return step > config.max_steps
#
#    pi_fn = utils.misc_utils.approx_improve_pi(env, utils.misc_utils.ppo_loss, iterations=config.opt_epochs)
#    pi, v, *_ = utils.misc_utils.approx_policy_iteration(env, stop_criterion=cb, pi_fn=pi_fn)
#    plot_vf(env, v, f"vf:eta={config.eta:.2f}", log_plot=True, step=config.max_steps)
#    gridworld_plot_sa(env, pi, f"pi:eta={config.eta:.2f}", log_plot=True, step=config.max_steps)

# TODO
def main():
    if config.env_id == "cliff":
        env = get_cliff(config.gamma)
    elif config.env_id == "sham":
        env = get_shamdp(config.horizon, c=config.penalty)
    elif config.env_id == "frozen":
        env = get_four_rooms(config.gamma)
    else:
        raise NotImplementedError()
    # else:
    #    env = get_gridworld(config.grid_size, gamma=config.gamma)

    if config.agent_id == "mdpo":
        pi_fn = utils.misc_utils.mdpo
    elif config.agent_id == "softmax_ppo":
        pi_fn = utils.misc_utils.softmax_ppo
    else:
        raise NotImplementedError()
    train(env, pi_fn)
    config.tb.run.finish()


if __name__ == '__main__':
    main()
