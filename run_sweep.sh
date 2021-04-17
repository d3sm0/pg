#/usr/bin/env bash

export BUDDY_IS_DEPLOYED=1

ETAS [ 0.24, 0.36, 0.48, 0.55, 0.60, 0.64,0.75, 0.96 ]
for eta in etas
  do
  python main.py --eta=eta

  done