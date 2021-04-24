import multiprocessing as mp

import os

etas = [0.03, 0.06, 0.12, 0.24, 0.36, 0.48, 0.55, 0.60]
agents = ["pg", "ppo"]
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
