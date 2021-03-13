import sys

import mila_tools
import torch

RUN_SWEEP = 1
REMOTE = 1
NUM_PROCS = 20

sweep_yaml = "sweep_seeds.yaml" if RUN_SWEEP else False
HOST = "mila" if REMOTE else ""  # in host
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

learning_rate = 2e-3
gamma = 0.99
eps_clip = 0.1
opt_epochs = 5
horizon = 2048
batch_size = 32
eta = 1.
agent = "pg"
save_interval = 200
max_steps = int(1e7)
seed = 984
h_dim = 32
#wandb_mode = "online" if DEBUG else "offline"

use_cuda = False

mila_tools.register(locals())
device = torch.device("cuda" if use_cuda else "cpu")

################################################################
# Derivative parameters
################################################################
# esh = """
# #SBATCH --mem=24GB
# """
esh = """
#SBATCH --job-name=spython
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --time=2-00:00
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=long
#SBATCH --get-user-env=L
"""
tb = mila_tools.deploy(host=HOST, sweep_yaml=sweep_yaml, extra_slurm_headers=esh, proc_num=NUM_PROCS)
