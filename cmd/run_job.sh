#!/usr/bin/env bash

#SBATCH --job-name=pull
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --constraint="intel"
#SBATCH --workdir=/homedtic/stotaro/ihvg
#SBATCH --output=/homedtic/stotaro/jobs/%N.%J.ihvg.out # STDOUT
#SBATCH --error=/homedtic/stotaro/jobs/%N.%J.ihvg.err # STDOUT

set -x
PROJECT_NAME=$1
echo "Ready to start $1 at branch $2"
git checkout '_snapshot_'$2
git pull origin '_snapshot_'$2
echo "Pulled. Load modules"

echo $PWD
sbatch run_experiment.sh $1 $3


