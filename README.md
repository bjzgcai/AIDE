# AIDE

Agentic Intelligent Data Engine for Scientific Large Language Models.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Package Manager](https://img.shields.io/badge/package%20manager-uv-6f42c1)

<p align="center">
  <img src="assets/aide-banner.png" alt="AIDE data-curation workflow banner" width="100%">
</p>

AIDE curates scientific datasets for large language model development. Given a
natural-language data requirement, it searches for candidate Hugging Face
datasets, selects useful datasets with an LLM quality gate, and organizes the
selected data into instruction-response examples or retrieval corpora.

## Install

```bash
git clone <repository-url>
cd <repository-name>
uv sync
```

Python 3.11 or newer is required.

## Offline Check

Run the local demo before using external services:

```bash
uv run aide-demo --output-dir outputs/demo
uv run python -m unittest discover -s tests
```

The demo writes a small synthetic curation run under `outputs/demo/` and does
not require API keys or network access.

## Configure Models

AIDE uses an OpenAI-compatible chat endpoint.

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
LLM_API_KEY=your-openai-compatible-api-key
LLM_API_BASE=https://api.openai.com/v1
LLM_EXTRA_BODY_JSON=
HF_TOKEN=
```

`HF_TOKEN` is optional for public Hugging Face datasets, but can help with rate
limits.

## Run AIDE

RAG corpus curation:

```bash
uv run aide-curate --config configs/smoke_biomedical_rag.yaml
```

Instruction-data curation:

```bash
uv run aide-curate --config configs/smoke_chemistry_instruction.yaml
```

The smoke configs keep dataset search and processing small. Use
`configs/biomedical_rag.yaml` or `configs/chemistry_instruction.yaml` as
starting points for larger runs.

## Outputs

Runs write artifacts under `outputs/`, including selected datasets, generated
processors, organized data, logs, and optional stage metrics. Generated files are
kept so that each curation decision can be inspected.

Instruction runs produce per-task `train.json` files. To export instruction and
output JSONL pairs:

```bash
uv run aide-dedup \
  --input-dir outputs/<run-dir> \
  --train-output outputs/<run-dir>/instruction_train.jsonl \
  --val-output outputs/<run-dir>/instruction_val.jsonl \
  --summary-path outputs/<run-dir>/dedup_summary.txt \
  --seed 0 \
  --force
```

## Configuration

Common fields:

| Field | Purpose |
| --- | --- |
| `prompt` | Natural-language data requirement. |
| `organization_format` | `instruction` or `rag`. |
| `selection_model` | Model used for dataset selection. |
| `organization_model` | Model used for data organization. |
| `max_results_per_term`, `max_datasets` | Bounds for dataset search. |
| `instruction_max_items_per_dataset` | Item cap for instruction runs. |
| `rag_max_items_per_dataset` | Item cap for RAG runs. |

See [docs/configuration.md](docs/configuration.md) for the full configuration
reference and [docs/end_to_end.md](docs/end_to_end.md) for live smoke tests.

## Development

```bash
uv sync --extra dev
uv run ruff format --check .
uv run ruff check src tests
uv run pytest -q
```

## Citation

If you use AIDE in research, please cite the accompanying paper and this
software release. A machine-readable citation file is available at
[CITATION.cff](CITATION.cff).

```bibtex
@software{aide_curator_2026,
  title = {Agentic Intelligent Data Engine for Scientific Large Language Models},
  author = {Xie, Shufang and Liu, Zequn and Deng, Pan and Luo, Renqian and Xia, Yingce and Qin, Tao and Yan, Rui},
  year = {2026},
  version = {0.1.0},
  license = {MIT}
}
```
