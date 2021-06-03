import sys
import os
import experiment_buddy

use_fa = False
horizon = 6 if use_fa else 20
penalty = 0.15 if use_fa else 1.4  # 1.6
eps = 1e-4
gamma = 0.99
eta =3.6
grid_size = 5
opt_epochs = 10
agent_id = "mdpo"
env_id = "sham"
save_interval = 10
seed = 984
mask_prob = 1.
lr = 1.
shift = 0.95

REMOTE = 1
RUN_SWEEP = REMOTE
NUM_PROCS = 5
sweep_yaml = "experiments/sweep_seeds.yaml" if RUN_SWEEP else False
HOST = "mila" if REMOTE else ""  # in host
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
max_steps = int(1e3)

render = not DEBUG
experiment_buddy.register(locals())
tb = experiment_buddy.deploy(host=HOST, sweep_yaml=sweep_yaml, proc_num=NUM_PROCS,
                             wandb_kwargs=dict(mode="online"))  # "disabled" if DEBUG else "online"))

plot_path = os.path.join(tb.objects_path, "plots")
os.makedirs(plot_path, exist_ok=True)
