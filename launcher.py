import multiprocessing as mp

etas = [0.01, 0.05, 0.1, 1.0, 1.5, 2.]

import os

os.environ["BUDDY_IS_DEPLOYED"] = "1"


def run(eta):
    return os.system(f"python main.py --eta={eta}")


def main():
    with mp.Pool(4) as p:
        p.map(run, etas)


if __name__ == '__main__':
    main()
