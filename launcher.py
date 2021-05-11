import multiprocessing as mp
import numpy as np

import os

etas = np.linspace(0.01, 1., 9)
agents = ["pg", "ppo", "pg_clip"]
import itertools

os.environ["BUDDY_IS_DEPLOYED"] = "1"


def run(arg):
    eta, agent = arg
    return os.system(f"python main.py --eta={eta} --agent={agent}")


out = itertools.product(etas, agents)


def main():
    with mp.Pool(4) as p:
        p.map(run, out)


if __name__ == '__main__':
    main()
