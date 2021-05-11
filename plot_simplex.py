import numpy as np

import matplotlib.pyplot as plt

# env = get_gridworld(5)
from utils.envs import get_four_rooms
import jax.numpy as jnp

from utils.misc_utils import policy_iteration, ppo, pg, pg_clip, get_value, get_star, get_dpi
from experiments.plot_fn import gridworld_plot_sa, plot_policy_at_state

env = get_four_rooms()
pi = np.ones(shape=(1, env.action_space))
pi /= pi.sum(1, keepdims=True)
pi = jnp.array(pi)
#pi = [0.25 , 0.25, 0.25, 0.25]
#pi = [0.1, 0.4, 0.1, 0.4]
get_dpi(env,pi)
pi = jnp.array(np.array(pi).reshape((1, 4)))
eta = 0.01
pi_star, *_, v_star = get_star(env)
v_0 = get_value(env, pi)
e_pg, pg_pi, t, v_pg, pi_s, advs, _ = policy_iteration(pg_clip, env, eta, pi)
action_symbols = ['↑', '↓', '←', '→']
plot_policy_at_state(pg_pi, title="lovely", action_label=action_symbols)
# gridworld_plot_sa(env, pg_pi, "lovely")
plt.show()
print("")
