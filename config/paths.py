from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_CREATOR = BASE_DIR / "data_creator"
SCRIPTS_PATH = BASE_DIR / "scripts" 
OUTPUT_PATH = BASE_DIR / "outputs"
ANSWERS_PATH = OUTPUT_PATH / "answers"
MODEL_OUTPUT = OUTPUT_PATH / "model_outputs"
FIG_DIR = BASE_DIR / "figures"
HF_REPO = "NASK-PIB/Reassess-Polish-Medical-Exams"