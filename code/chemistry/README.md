# Project Setup

This repository is managed with [`uv`](https://docs.astral.sh/uv/), a fast Python package and environment manager.

## Environment Setup

1. Install `uv`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Create the virtual environment and install dependencies:
   ```bash
   uv sync --frozen
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Managing Dependencies

To add a new Python dependency:
```bash
uv pip install <package-name>
```
The command updates both the lockfile and the virtual environment.

## Instruction-Tuning Data Collection

1. Update `src/instagent/configs.py` with your credentials:
   - `base_url: str = "YOUR_OPENAI_API_BASE_URL"`
   - `api_key: str = "YOUR_OPENAI_API_KEY"`
2. Run the data collection pipeline:
   ```bash
   bash run.sh
   ```
3. Collected samples are stored in `dataflow/gpu_pipelines/cache/dataflow_cache_step_step2.jsonl`.