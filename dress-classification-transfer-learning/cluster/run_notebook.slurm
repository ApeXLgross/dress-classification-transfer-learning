#!/bin/bash
#SBATCH --job-name=dress-classification
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: sbatch $0 notebooks/<notebook>.ipynb"
  exit 2
fi

NOTEBOOK="$1"
OUTPUT="${NOTEBOOK%.ipynb}_executed.ipynb"

# Adapt these lines to the cluster's module and environment setup.
module load python
source .venv/bin/activate

jupyter nbconvert \
  --to notebook \
  --execute "$NOTEBOOK" \
  --output "$OUTPUT" \
  --ExecutePreprocessor.timeout=-1
