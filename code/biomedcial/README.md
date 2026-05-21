# AIDE

This directory contains the minimal code needed to reproduce the dataset-discovery pipeline described in the paper. It keeps the core logic (CLI entry point plus `src/AIDE` package) while omitting exploratory notebooks, figures, large outputs, and private scripts.

## What’s Included
- `main.py`: CLI entry point that wires configuration to the agent.
- `src/AIDE`: core implementation (keyword generation, HF search, analysis, advanced processing, utilities).
- `pyproject.toml` / `uv.lock`: dependency metadata and lockfile for deterministic installs.

## Prerequisites
- Python 3.13 (matches the requirement in `pyproject.toml`).
- [`uv`](https://docs.astral.sh/uv/) for environment management.
- Access to the OpenAI API (or compatible endpoint) and optional Hugging Face Hub token.

## Environment Setup
Unzip the provided archive and run the following commands from the extracted `release` directory.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv if needed
uv sync --frozen                  # create .venv 
source .venv/bin/activate
```

Set the required environment variables before running:
- `OPENAI_API_KEY`: key for the chosen OpenAI-compatible endpoint, i.e. 'sk-xxxx'
- `OPENAI_API_BASE` (optional): override the default OpenAI API base URL if using OpenAI-compatible gateways. e.g., https://openrouter.ai/api/v1
- `HF_TOKEN` (optional but recommended): increases Hugging Face Hub rate limits when searching datasets.

## Running the Pipeline
`main.py` exposes the dataclass defined in `AIDE.configs.InstAgentConfig` as CLI arguments via `jsonargparse`. Refer to `src/AIDE/configs.py` for every knob and its default.

Before running, you can define a few shell variables to keep the long command readable. Adjust the defaults as needed:

```bash
export SEARCH_PROMPT="Identify high-quality biomedical QA datasets for instruction tuning"
export HF_KEYWORDS_MODEL="openai/gpt-4o-2024-11-20"  # LLM used to propose HF search keywords (i.e., stage 1)
export BA_MODEL="openai/gpt-4o-2024-11-20"           # LLM used for the basic analysis stage (i.e., stage 2)
export ADVANCED_PROCESS_MODEL="openai/gpt-4o-2024-11-20"  # LLM used during advanced dataset inspection (i.e., stage 3)
export CACHE_DIR="./cache"                 # Location for cached downloads and API results
export OUTPUT_DIR="./outputs"             # Final reports + cached prompt-specific artifacts
export MAX_RESULTS_PER_TERM=1             # HF dataset hits to keep per generated keyword
export MAX_DATASETS=1                     # Global cap on datasets processed in one run
```

**Run the command**
```bash
uv run python main.py \
  "$SEARCH_PROMPT" \
  --use_hf_keywords_generation true \
  --hf_keywords_model "$HF_KEYWORDS_MODEL" \
  --ba_model "$BA_MODEL" \
  --advanced_process_type inst \
  --advanced_process_model "$ADVANCED_PROCESS_MODEL" \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --max_results_per_term "$MAX_RESULTS_PER_TERM" \
  --max_datasets "$MAX_DATASETS"
```

Key behaviors:
- `cache_dir` now defaults to `./cache`, ensuring all HF downloads stay local to the release directory.
- Each run materializes `outputs/<prompt-hash>/user_prompt.txt`, cached intermediate JSON files, and generated advanced-processing artifacts (see structure below).
- Re-running the same prompt reuses cache files (`keywords.json`, `datasets.json`, `basic_analysis.json`) to avoid repeated API calls. Delete the corresponding `outputs/<hash>/cache` folder to force a refresh.

## Tips & Troubleshooting
- To inspect available CLI flags, run `uv run python main.py --help`.
