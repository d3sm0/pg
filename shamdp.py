from emdp.chainworld import build_chain_MDP
import numpy as np
import itertools
from emdp import actions
from gym.spaces import Discrete


def process_action(state, action):
    if state.argmax() == 0:
        return actions.RIGHT
    if action == 0:
        action = actions.RIGHT
    else:
        action = actions.LEFT
    return action


def get_policy_class(n_states, n_actions, slack=0.1):
    directions = []
    for a in range(n_actions):
        direction = np.zeros(n_actions, dtype=np.float32)
        direction[a] = 1
        # direction[a] = 1 - slack
        # direction[direction == 0] = slack / len(direction[direction == 0])
        directions.append(direction)
    policy = np.array(list(itertools.product(directions, repeat=n_states)))
    return policy


def get_shamdp(horizon=2):
    slack = 0.
    state_distribution = np.zeros(shape=(horizon + 2,))
    state_distribution[0] = 1
    mdp = build_chain_MDP(n_states=horizon + 2, p_success=1.0, reward_spec=[(horizon + 1, actions.RIGHT, 1), (horizon+1, actions.LEFT, 1)],
                          starting_distribution=state_distribution,
                          terminal_states=[horizon + 1], gamma=horizon / (horizon + 1))
    mdp.action_space = 4
    mdp.observation_space = 0
    mdp.reward_range = 0
    mdp.metadata = 0
    return mdp
