import itertools
# TODO plot movement on the simplex

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# ax = plt.axes(projection='3d')
fig = plt.figure(figsize=(12, 4))
ax = plt.gca()
adv = jnp.load('adv.npy')
s, a = adv.shape
ax.imshow(adv.T, alpha=0.8)
for r in range(s):
    for c in range(a):
        ax.text(r, c, f"{adv[r, c]:.2f}", color='w', va='center', ha='center')
ax.text(20, 0, f"{adv[20, 0]:.2f}", color='r', va='center', ha='center')

ax.set_xticks(np.arange(s))
ax.set_yticks(np.arange(a))
ax.set_xlabel('state_idx')
ax.set_ylabel('action_idx')
ax.set_title('Advantage')
# ax.set_yticklabels(['R', 'L' 'L', 'L'])
fig.tight_layout()
plt.savefig("plots/shamdp/adv")
# plt.show()
