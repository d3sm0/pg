import sys
import os
import experiment_buddy

use_fa = False
horizon = 6 if use_fa else 20
penalty = 0.15 if use_fa else 1.4  # 1.6
eps = 1e-4
gamma = 0.9
eta = 0.1
grid_size = 10
opt_epochs = 20
agent = "pg_clip"
save_interval = 10
max_steps = int(1e3)
seed = 984
lr = 1.

REMOTE = 1
RUN_SWEEP = REMOTE
NUM_PROCS = 5
sweep_yaml = "experiments/sweep.yaml" if RUN_SWEEP else False
HOST = "mila" if REMOTE else ""  # in host
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

render = not DEBUG
experiment_buddy.register(locals())
tb = experiment_buddy.deploy(host=HOST, sweep_yaml=sweep_yaml, proc_num=NUM_PROCS,
                             wandb_kwargs=dict(mode="disabled" if DEBUG else "online"))

plot_path = os.path.join(tb.objects_path, "plots")
os.makedirs(plot_path, exist_ok=True)
