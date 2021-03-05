#!/usr/bin/env bash

echo "Ready to start $1 with branch $2. Experiment id $3"

PROJECT_DIR=$1
rm -rf jobs/*
cd $PROJECT_DIR

echo $PWD

sbatch run_job.sh $1 $2 $3
