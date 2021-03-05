#!/usr/bin/env bash
echo "args revived" $1 $2 $3

ssh cluster "bash -l run.sh $1 $2 $3"
