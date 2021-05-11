import random

import emdp.gridworld.builder_tools
from emdp import actions, MDP
from emdp.gridworld import GridWorldMDP
from emdp.gridworld.builder_tools import create_reward_matrix
from emdp.gridworld.helper_utilities import flatten_state
from emdp.gridworld.txt_utilities import get_char_matrix  # , ascii_to_walls
from jax import numpy as jnp
import numpy as np


def two_states_gw():
    builder = emdp.gridworld.builder_tools.TransitionMatrixBuilder(grid_size=3, has_terminal_state=False)
    builder.add_grid([], p_success=1)
    builder.add_wall_at((0, 0))
    builder.add_wall_at((0, 1))
    builder.add_wall_at((0, 2))
    builder.add_wall_at((1, 0))
    builder.add_wall_at((1, 2))
    # builder.add_wall_at((2, 0))
    # builder.add_wall_at((2, 2))

    reward_spec = {(2, 1): +1}
    R = create_reward_matrix(builder.P.shape[0], builder.grid_size, reward_spec, action_space=builder.P.shape[1])

    # target_state = pos_to_state(builder, (2, 2))
    p0 = flatten_state((1, 1), builder.grid_size, R.shape[0])
    # p0[[2, 4, 5, 8]] = .25
    gw = GridWorldMDP(builder.P, R, 0.9, p0, terminal_states=(), size=builder.grid_size)
    return gw


def four_states_gw(goal):
    builder = emdp.gridworld.builder_tools.TransitionMatrixBuilder(grid_size=3, has_terminal_state=False)
    builder.add_grid([], p_success=1)
    builder.add_wall_at((0, 0))
    builder.add_wall_at((0, 1))
    builder.add_wall_at((0, 2))
    builder.add_wall_at((1, 0))
    builder.add_wall_at((1, 2))

    reward_spec = {goal: +1}
    R = create_reward_matrix(builder.P.shape[0], builder.grid_size, reward_spec, action_space=builder.P.shape[1])

    # target_state = pos_to_state(builder, (2, 2))
    p0 = flatten_state((1, 1), builder.grid_size, R.shape[0])
    # p0[[2, 4, 5, 8]] = .25
    gw = GridWorldMDP(builder.P, R, 0.9, p0, terminal_states=(), size=builder.grid_size)
    return gw


def pos_to_state(builder, pos):
    target_state = flatten_state(pos, builder.grid_size, builder.state_space)
    target_state = target_state.argmax()
    return target_state


def get_two_states():
    env = two_states_gw()
    return env


def get_four_states():
    env = four_states_gw((2, 0))
    return env


def get_four_rooms(gamma):
    env = four_rooms_gw(gamma)
    return env


def ascii_to_walls(ascii_room):
    walls = []
    empty = []
    for row_idx, r in enumerate(ascii_room):
        for column_idx, c in enumerate(r):
            if c == "#":
                walls.append((row_idx, column_idx))
            else:
                empty.append((row_idx, column_idx))
    return walls, empty


def four_rooms_gw(gamma):
    ascii_room = """
    #########
    #   #   #
    #       #
    #   #   #
    ## ### ##
    #   #   #
    #       #
    #   #   #
    #########"""[1:].split('\n')
    ascii_room = [row.strip() for row in ascii_room]
    # a = list(ascii_room[goal[0]])
    # a[goal[1]] = "g"
    # ascii_room[goal[0]] = "".join(a)

    char_matrix = get_char_matrix(ascii_room)

    grid_size = len(char_matrix[0])
    reward_spec = {(grid_size - 2, grid_size - 2): +1}
    builder = emdp.gridworld.builder_tools.TransitionMatrixBuilder(grid_size=grid_size, has_terminal_state=False)

    walls, empty_ = ascii_to_walls(char_matrix)  # hacks
    empty = []
    for e in empty_:
        (e,), = flatten_state(e, grid_size, grid_size * grid_size).nonzero()
        empty.append(e)
    builder.add_grid(p_success=1, terminal_states=[(grid_size - 2, grid_size - 2)])
    for (r, c) in walls:
        builder.add_wall_at((r, c))

    R = create_reward_matrix(builder.P.shape[0], builder.grid_size, reward_spec, action_space=builder.P.shape[1])
    p0 = np.zeros(R.shape[0])
    #p0[10] = 1.
    p0[empty] = 1 / len(empty)
    gw = GridWorldMDP(builder.P, R, gamma, p0, terminal_states=[(grid_size - 2, grid_size - 2)], size=builder.grid_size)
    return gw
