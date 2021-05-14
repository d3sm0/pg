import emdp
from emdp.chainworld import build_chain_MDP
import emdp.gridworld as gw
from emdp.gridworld.builder_tools import TransitionMatrixBuilder, create_reward_matrix
from emdp.gridworld import GridWorldPlotter, GridWorldMDP

import numpy as np
import itertools
from emdp import actions
from emdp.gridworld.helper_utilities import flatten_state
from emdp.gridworld.txt_utilities import get_char_matrix

from utils.envs import ascii_to_walls


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


def get_n_state_chain(n_states=2):
    state_distribution = np.zeros(shape=(n_states,))
    state_distribution[0] = 1
    reward_spec = np.zeros(shape=(n_states, 2))
    reward_spec[-1, 1] = 1.
    reward_spec[-1, 0] = 1.
    mdp = build_chain_MDP(n_states=n_states, p_success=1.0,
                          reward_spec=reward_spec,
                          starting_distribution=state_distribution,
                          terminal_states=[n_states], gamma=0.9)
    mdp.P[1, 0, 0] = 0
    mdp.P[1, 0, 1] = 1
    mdp.P[1, 1, 1] = 1
    return mdp


def get_shamdp(horizon=2):
    state_distribution = np.zeros(shape=(horizon + 2,))
    state_distribution[0] = 1
    reward_spec = np.zeros(shape=(horizon + 2, 2))
    reward_spec[:, 1] = -1
    reward_spec[-1, 1] = 20
    mdp = build_chain_MDP(n_states=horizon + 2, p_success=1.0,
                          reward_spec=reward_spec,
                          starting_distribution=state_distribution,
                          terminal_states=[horizon + 1], gamma=horizon / (horizon + 1))
    mdp.action_space = 2
    mdp.observation_space = 0
    mdp.reward_range = 0
    mdp.metadata = 0
    return mdp


def grid_to_idx(idx, grid_size):
    x = idx % grid_size
    y = idx // grid_size
    assert x < grid_size and y < grid_size
    return x, y


def idx_to_grid(x, y, grid_size):
    idx = x + grid_size * y
    return idx


def get_corridor():
    grid_size = 5
    gamma = 0.9
    terminal_states = [(grid_size - 1, grid_size - 1)]
    from emdp.gridworld.builder_tools import TransitionMatrixBuilder
    builder = TransitionMatrixBuilder(grid_size=5, has_terminal_state=True)
    builder.add_grid(terminal_states, p_success=1)
    builder.add_wall_at((4, 2))
    builder.add_wall_at((3, 2))
    builder.add_wall_at((2, 2))
    builder.add_wall_at((1, 2))

    # P = gw.build_simple_grid(size=grid_size, p_success=1., terminal_states=[(grid_size - 1, grid_size - 1)])
    n_states, n_actions = builder.P.shape[:2]
    p0 = np.zeros(n_states)
    p0[0] = 1.
    R = np.zeros((n_states, n_actions))
    idx = idx_to_grid(grid_size - 1, grid_size - 1, grid_size)
    assert idx < R.shape[0]
    R[idx, :] = 1
    mdp = gw.GridWorldMDP(builder.P, R, gamma, p0, terminal_states, grid_size)
    return mdp


def get_gridworld(grid_size):
    """
    first set the probability of all actions from state 1 to zero
    now set the probability of going from 1 to 21 with prob 1 for all actions
    first set the probability of all actions from state 3 to zero
    now set the probability of going from 3 to 13 with prob 1 for all actions
    """

    P = gw.build_simple_grid(size=grid_size, p_success=1., terminal_states=[(grid_size - 1, grid_size - 1)])
    n_states, n_actions = P.shape[:2]
    R = np.zeros((n_states, n_actions))
    idx = idx_to_grid(grid_size - 1, grid_size - 1, grid_size)
    assert idx < R.shape[0]
    R[idx, :] = 1

    p0 = np.zeros(n_states)
    p0[0] = 1.
    gamma = 0.9
    terminal_states = [(grid_size - 1, grid_size - 1)]
    mdp = gw.GridWorldMDP(P, R, gamma, p0, terminal_states, grid_size)
    return mdp


def build_chain_mdp(H, n_actions, end_reward, right_penalty=0.):
    n_states = H + 2
    P = np.zeros((n_states, n_actions, n_states))  # (s, a, s')
    r = np.zeros((n_states, n_actions))

    # taking action 0 at absorbing state gives you reward 1
    r[-1, 0] = 1.

    # optional penalty of going right at every state but the last
    r[:-1, 0] = right_penalty

    # populate the transition matrix
    # forward actions
    for s in range(n_states - 1):
        P[s, 0, s + 1] = 1.
    P[n_states - 1, :, n_states - 1] = 1.  # irrespective of the action, you end up in the last state forever
    # backward actions
    for s in range(1, n_states - 1):
        P[s, 1:, s - 1] = 1.
    P[0, 1:, 0] = 1.
    return P, r


def shamdp():
    H = 20
    gamma = H / (H + 1)
    n_states = H + 2
    n_actions = 4

    # original MDP
    P, r = build_chain_mdp(H, n_actions, 1, 0)

    # modified MDP
    # going right except in the absorbing state incurs a penalty
    penalty = -gamma ** (H // 1.6)
    _, rhat = build_chain_mdp(H, n_actions, 1, penalty)  # same transition dynamics as original MDP

    # starting state distribution
    rho = np.zeros(n_states)
    rho[0] = 1.

    # optimal policy of this MDP
    pistar = np.zeros((n_states, n_actions))
    pistar[:, 0] = 1.

    # uniform starting state distribution
    rho = np.ones(n_states) / n_states

    #@jax.jit
    #def policy_objective(P, r, gamma, pi, rho):
    #    v_pi = get_V(P, r, gamma, pi)
    #    return jnp.sum(v_pi * rho)

    ## verify that the return for going right is still positive
    #print(f"return of go-right policy/optimal is {policy_objective(P, rhat, gamma, pistar, rho)}")


def _test(env):
    s = env.reset()
    done = False
    while not done:
        action = 0
        s, r, done, _ = env.step(action)
        print(s, r, done)


if __name__ == '__main__':
    env = get_gridworld(grid_size=2)
    _test(env)
