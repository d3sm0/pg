import jax.numpy as jnp

from mdp import get_n_state_chain
from utils.misc_utils import mdpo, pg, policy_iteration

env = get_n_state_chain(2)
env.P[1, 0, 0] = 0
env.P[1, 0, 1] = 1
env.P[1, 1, 1] = 1
pi = jnp.ones((2, 2)) / 2
eta = 1
e_pg, pg_pi, t, v_pg, ppo_pi, _, _ = policy_iteration(mdpo, env, eta, pi)
e_pg, pg_pi, t, v_pg, pg_pi, _, _ = policy_iteration(pg, env, eta, pi)

print(ppo_pi)

ppo_pi[:, 0, 0] < pg_pi[:, 0, 0]