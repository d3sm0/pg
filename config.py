import sys
import os
import experiment_buddy

pi_lr = 0.01
gamma = 0.9
eps_clip = 0.1
opt_epochs = 10
horizon = 19  # if DEBUG else 200
eta = 1.0
grid_size = 5
agent = "pg"
save_interval = 10
max_steps = int(1e2)
seed = 984
eval_episodes = 10
data = "data"
use_fa = True
use_kl = False
eps = 1e-6 if use_kl else 1e-4

REMOTE = 1
RUN_SWEEP = REMOTE
NUM_PROCS = 5
sweep_yaml = "sweep_seeds.yaml" if RUN_SWEEP else False
HOST = "mila" if REMOTE else ""  # in host
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

render = not DEBUG
experiment_buddy.register(locals())
tb = experiment_buddy.deploy(host=HOST, sweep_yaml=sweep_yaml, proc_num=NUM_PROCS,
                             wandb_kwargs=dict(mode="disabled" if DEBUG else "online"))

os.makedirs(data, exist_ok=True)
plot_path = os.path.join(tb.objects_path, "plots")
os.makedirs(plot_path, exist_ok=True)
