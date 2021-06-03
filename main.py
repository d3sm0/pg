import jax.nn
import jax.numpy as jnp
# TODO experiment with cliff + linesearch
# TODO experiment with different initialization
# TODO experiment with chain + different eta
import config
from plot_fn import plot_vf, gridworld_plot_sa, plot_policy_at_state, chain_plot_vf
from utils.envs import get_cliff, get_four_rooms
import utils.misc_utils
from utils.mdp import get_gridworld, get_shamdp
import matplotlib.pyplot as plt
import numpy as np


def train(env, pi_fn):
    def cb(step):
        return step > config.max_steps

    # left right up down
    pi = np.ones(shape=(env.state_space, env.action_space))
    pi_star, v_star = utils.misc_utils.get_star(env)
    print(env.p0 @ v_star, pi_star[0, 0])
    pi[:, 1] += config.shift * 10
    pi[:, 2] += config.shift * 10
    pi /= pi.sum(1, keepdims=True)
    policy = jnp.array(pi)
    print(pi[env.p0.argmax()])
    pi, v, *_ = utils.misc_utils.policy_iteration(env, pi_opt=pi_fn, stop_criterion=cb, eta=config.eta, policy=policy)
    #chain_plot_vf(v, title=f"vf={config.agent_id}:{env.p0 @ v}", log_plot=True, step=config.max_steps + 1)


    #plot_policy_at_state(pi, action_label=labels, title=title)
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
