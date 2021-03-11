import sys

import mila_tools
import torch

RUN_SWEEP = 1
REMOTE = 1
NUM_PROCS = 1

sweep_yaml = "sweep_seeds.yaml" if RUN_SWEEP else False
HOST = "mila" if REMOTE else ""  # in host
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

learning_rate = 2e-3
gamma = 0.99
eps_clip = 0.1
opt_epochs = 5
horizon = 64
batch_size = 32
eta = 1.
agent = "pg"
save_interval = 200
max_steps = int(5e6)
seed = 0
h_dim = 32
wandb_mode = "offline"

use_cuda = False

mila_tools.register(locals())
device = torch.device("cuda" if use_cuda else "cpu")

################################################################
# Derivative parameters
################################################################
# esh = """
# #SBATCH --mem=24GB
# """
esh = ""
tb = mila_tools.deploy(host=HOST, sweep_yaml=sweep_yaml, extra_slurm_headers=esh, proc_num=NUM_PROCS)
