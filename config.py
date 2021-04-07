import sys
import wandb
import functools

wandb.init = functools.partial(wandb.init, mode="disabled")

import experiment_buddy

eval_runs = 1
pi_lr = 1e-1
v_lr = 1e-5
gamma = 0.99
eps_clip = 0.1
opt_epochs = 20
horizon = 200  # if DEBUG else 200
mdp_horizon = 8
batch_size = 32
eta = 0.001
grid_size = 8
agent = "pg"
save_interval = 10
max_steps = int(1e3)
seed = 984
h_dim = 32
# wandb_mode = "online" if DEBUG else "offline"

use_cuda = False

experiment_buddy.register(locals())

################################################################
# Derivative parameters
################################################################
# esh = """
# #SBATCH --mem=24GB
# """

REMOTE = 1
RUN_SWEEP = 1
NUM_PROCS = 5
sweep_yaml = "sweep_params.yaml" if RUN_SWEEP else False
HOST = "mila" if REMOTE else ""  # in host
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

tb = experiment_buddy.deploy(host=HOST, sweep_yaml=sweep_yaml, proc_num=NUM_PROCS)
