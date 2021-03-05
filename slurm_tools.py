import ast
import time

# import flax.core
import tensorboardX as tb
import wandb

# from _config import config
# from flax.core import FrozenDict
import config


# TODO ad safety csv here
# TODO push video policy

class SummaryWriter:
    def __init__(self, project, experiment_id, log_dir):
        self.log_dir = log_dir
        self.writer = tb.SummaryWriter(logdir=log_dir)
        self.run = self._connect(project, experiment_id, config.wandb_mode)
        self._metrics = {}

    @staticmethod
    def _connect(project, name, mode, max_trials=10):
        trial = 0
        while trial < max_trials:
            trial += 1
            try:
                run = wandb.init(project=project, name=name, mode=mode, entity="ihvg")
                print("Connected to wandb")
                return run
            except wandb.Error as e:
                print(e)
                time.sleep(10)
        else:
            run = wandb.init(project=project, name=name, mode="disabled")
            print("Failed to conncet wandb, running in disabled mode")
        return run

    def add_scalar(self, k, v, global_step):
        self._metrics[k] = float(v)
        self.writer.add_scalar(k, v, global_step)

    def add_figure(self, k, v, global_step):
        self.run.log({k: v, "global_step": global_step})

    #def save_checkpoint(self, target, step=None, prefix='checkpoint_'):
    #    return save_checkpoint(self.log_dir, target, step, prefix)

    def add_config(self, config):
        print("updating config")
        self.run.config.update(config)

    def flush(self, global_step):
        self._metrics["global_step"] = global_step
        self.run.log(self._metrics)
        self._metrics = {}

    def add_scalars(self, stats, step):
        for k, v in stats.items():
            self.add_scalar(k, v, step)


#def restore_checkpoint(ckpt_dir, target, step=None, prefix='checkpoint_'):
#    out = flax.training.checkpoints.restore_checkpoint(ckpt_dir, None, step, prefix)
#    return FrozenDict(out)
#
#
#def save_checkpoint(ckpt_dir, target, step=None, prefix='checkpoint_', parallel=True):
#    out = flax.training.checkpoints.save_checkpoint(ckpt_dir, target, step, prefix, parallel)
#    return out


def parse_args(argv):
    # following k,v convetion
    _args = {}
    for arg in argv:
        k, v = arg[2:].split("=")
        try:
            v = ast.literal_eval(v)
        except Exception as e:
            print(e, v)
        _args[k] = v
    return _args

# TODO add local writer
# TODO check code experiment


# def git_sync(experiment_id, git_repo):
#    active_branch = git_repo.active_branch.name
#    os.system(f"git add -u .")
#    os.system(f"git commit -m '{experiment_id}'")
#    experiment_branch = f"_snapshot_{active_branch}"
#    for b in git_repo.branches:
#        if b.name == experiment_branch:
#            os.system(f"git checkout {experiment_branch}")
#            os.system(f"git merge {active_branch}")
#            break
#    else:
#        os.system(f"git checkout -b {experiment_branch}")
#    os.system(f"git push --set-upstream origin {experiment_branch}")  # send to online repo
#    git_hash = git_repo.commit().hexsha
#    os.system(f"git checkout {active_branch}")  # return on active branch
#    return git_hash
