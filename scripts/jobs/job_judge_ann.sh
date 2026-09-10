#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512gb
#SBATCH --time=01:00:00
#SBATCH --account=
#SBATCH --partition=
#SBATCH --output=out_slurm/job-%j.out
#SBATCH --gres=gpu:4

export BASE_DIR=$(python -c "from config.paths import BASE_DIR; print(BASE_DIR)")
model=$1
echo $model
module load Python/3.10.4
source $BASE_DIR/.venv/bin/activate
cd $BASE_DIR
exam=None
model=None
python -m scripts.run_scripts.run_answers_llm_judge $exam $model
