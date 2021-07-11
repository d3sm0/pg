import sys
import os
import experiment_buddy

agent_id = "mdpo"
env_id = "four_rooms"

# env params
gamma = 0.90
env_kwargs = {
    "gamma": gamma
}
eta = 0.1

# approx pi
approximate_pi = False
opt_epochs = 10
lr = 1.

# training params
save_interval = 10
seed = 984
max_steps = int(1e2)

experiment_buddy.register(locals())

# Make sure you have wandb.ai to use this
REMOTE = 0
RUN_SWEEP = REMOTE
NUM_PROCS = 5
sweep_yaml = "experiments/sweep_seeds.yaml" if RUN_SWEEP else False
HOST = ""  # "mila" if REMOTE else ""
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
tb = experiment_buddy.deploy(host=HOST, sweep_yaml=sweep_yaml, proc_num=NUM_PROCS, wandb_kwargs=dict(mode="disabled"))

plot_path = os.path.join(tb.objects_path, "plots")
os.makedirs(plot_path, exist_ok=True)
