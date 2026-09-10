#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512gb
#SBATCH --time=03:00:00
#SBATCH --account=
#SBATCH --partition=
#SBATCH --output=out_slurm/judge-%j.out
#SBATCH --gres=gpu:4
export BASE_DIR=$(python -c "from config.paths import BASE_DIR; print(BASE_DIR)")
model=$1
echo $model
module load Python/3.10.4
source $BASE_DIR/.venv/bin/activate
cd $BASE_DIR
for exam in r_lek r_ldek r_pes_latest r_diagnostics r_pharmacy r_lek_en r_ldek_en;do
    python -m scripts.run_scripts.run_answers_llm_judge $exam $model
    done
