import getpass
import sys

import experiment_buddy
import torch

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
REMOTE = 1
RUN_SWEEP = REMOTE
NUM_PROCS = 5 if not DEBUG else 1
HOST = "mila" if REMOTE else ""  # in host

sweep_yaml = "sweep_seeds.yaml" if RUN_SWEEP else False

user = getpass.getuser()
learning_rate = 1e-3
# lr_decay = 0.99
gamma = 0.99
eps_clip = 0.1
opt_epochs = 10
horizon = 200 if DEBUG else int(1e4)
batch_size = 32
eta = 0.
grid_size = 8
agent = "ppo"
save_interval = 5
max_steps = int(50)
seed = 984
h_dim = 32
eval_runs = 10
# wandb_mode = "online" if DEBUG else "offline"

use_cuda = False

experiment_buddy.register(locals())
device = torch.device("cuda" if use_cuda else "cpu")

################################################################
# Derivative parameters
################################################################
# esh = """
# #SBATCH --mem=24GB
# """
tb = experiment_buddy.deploy(host=HOST, sweep_yaml=sweep_yaml, proc_num=NUM_PROCS)
