#!/usr/bin/env bash

#SBATCH --job-name=sweep
#SBATCH --cpus-per-task=4
#SBATCH --priority="TOP"
#SBATCH --mem=8GB
#SBATCH --constraint="intel"
#SBATCH --workdir=/homedtic/stotaro/ihvg
#SBATCH --output=/homedtic/stotaro/jobs/%N.%J.ihvg.out # STDOUT
#SBATCH --error=/homedtic/stotaro/jobs/%N.%J.ihvg.err # STDOUT

set -x
module load numpy
source /homedtic/stotaro/ihvg_env/bin/activate

wandb agent $1
