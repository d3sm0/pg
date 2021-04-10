#/usr/bin/env bash

export BUDDY_IS_DEPLOYED=1

for eta in etas
  do
  python main.py --eta=eta

  done