curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate

uv pip install openai
uv pip install dotenv
uv pip install nltk
uv pip install pandas
uv pip install matplotlib seaborn
uv pip install huggingface_hub
uv pip install datasets
uv pip install vllm==0.19.0
uv pip install -U transformers
uv pip list