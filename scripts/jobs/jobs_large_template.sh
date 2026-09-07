#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512GB
#SBATCH --time=03:00:00
#SBATCH --account=
#SBATCH --partition=
#SBATCH --output=out_slurm/job-%j.out
#SBATCH --gres=gpu:4
export BASE_DIR=$(python -c "from config.paths import BASE_DIR; print(BASE_DIR)")
model=$1
module load Python/3.10.4
module load CUDA/12.8.0
source $BASE_DIR/.venv/bin/activate
echo $HF_HOME
hf auth login
cd $BASE_DIR

for exam in r_lek_pandas r_ldek_pandas r_pes_latest_pandas r_diagnostics_pandas r_pharmacy_pandas
# for exam in r_lek r_ldek
do
	python -m scripts.run_scripts.evaluation_all_splits "$exam" "$model"
done

for exam in r_lek_en_pandas r_ldek_en_pandas
do
	python -m scripts.run_scripts.evaluation_all_eng "$exam" "$model"
done
