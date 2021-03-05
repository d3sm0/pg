#!/usr/bin/env bash
#SBATCH --job-name=experiment
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --constraint="intel"
#SBATCH --workdir=/homedtic/stotaro/ihvg
#SBATCH --output=/homedtic/stotaro/jobs/%N.%J.ihvg.out # STDOUT
#SBATCH --error=/homedtic/stotaro/jobs/%N.%J.ihvg.err # STDOUT

set -x
PROJECT_NAME=$1
module load numpy
echo $PWD
git status
source /homedtic/stotaro/ihvg_env/bin/activate
#python -O gpomdp.py

wandb sweep sweep_envs.yaml -e $1 -p $PROJECT_NAME --name $2 &>sweep_id.log
SWEEP_ID=$(grep -o 'ID: .*' sweep_id.log | cut -f2 --d ' ')
SWEEP_ADDR=ihvg/$PROJECT_NAME/$SWEEP_ID
echo $SWEEP_ADDR
NUM_PROC=25
for i in $(seq 0 $NUM_PROC)
  do
    echo "starting machine $i for $SWEEP_ADDR"
    sbatch distributed_sweep.sh $SWEEP_ADDR
    sleep 5
  done