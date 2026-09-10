export BASE_DIR=$(python -c "from config.paths import BASE_DIR; print(BASE_DIR)")
model=$1
cd $BASE_DIR
exam=None
model=None
python -m scripts.run_scripts.run_answers_llm_judge $exam $model
