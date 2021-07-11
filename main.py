import jax.numpy as jnp
import numpy as np

import config
import src.algorithms
from src import get_env, get_agent
from src.utils.plot_utils import gridworld_plot_sa, plot_vf


def get_initial_policy(env):
    pi = np.ones(shape=(env.state_space, env.action_space))
    pi /= pi.sum(1, keepdims=True)
    policy = jnp.array(pi)
    return policy


def stop_criterion(step):
    return step > config.max_steps


def train(env, pi_fn):
    policy = get_initial_policy(env)
    pi, v, *_ = src.algorithms.policy_iteration(env, pi_opt=pi_fn, stop_criterion=stop_criterion, eta=config.eta,
                                                policy=policy)


def main():
    env = get_env(config.env_id, **config.env_kwargs)
    policy_update_fn = get_agent(config.agent_id, config.approximate_pi)
    train(env, policy_update_fn)
    config.tb.run.finish()


if __name__ == '__main__':
    main()
