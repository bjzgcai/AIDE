# Live Smoke Tests

This guide runs AIDE with real model calls and public Hugging Face datasets while
keeping the job small.

## 1. Prepare

```bash
uv sync
cp .env.example .env
```

Set `LLM_API_KEY` and, if needed, `LLM_API_BASE` in `.env`. `HF_TOKEN` is
optional for public datasets.

## 2. Offline Baseline

```bash
uv run aide-demo --output-dir outputs/demo
uv run python -m unittest discover -s tests
```

Do this first. If the offline path fails, fix the local environment before
spending model or Hugging Face calls.

## 3. RAG Smoke Test

```bash
uv run aide-curate --config configs/smoke_biomedical_rag.yaml
```

Expected run shape:

```text
outputs/smoke_biomedical_rag/
  cache/
  log.txt
  stage_metrics.json
  *_rag.jsonl
  *_process_data.py
```

Pass criteria:

- `log.txt` reports at least one selected dataset.
- A `*_rag.jsonl` file exists and contains JSON objects with `id` and
  `contents`.
- The generated `*_process_data.py` file is present for inspection.

## 4. Instruction Smoke Test

```bash
uv run aide-curate --config configs/smoke_chemistry_instruction.yaml
```

Expected run shape:

```text
outputs/smoke_chemistry_instruction/
  cache/
  log.txt
  stage_metrics.json
  <dataset>/<task>/raw/extract_code.py
  <dataset>/<task>/train.json
```

Pass criteria:

- `log.txt` reports at least one organized dataset.
- At least one task directory contains `raw/extract_code.py`.
- At least one `train.json` contains tab-separated instruction-response
  strings.

Export SFT-style JSONL pairs:

```bash
uv run aide-dedup \
  --input-dir outputs/smoke_chemistry_instruction \
  --train-output outputs/smoke_chemistry_instruction/instruction_train.jsonl \
  --val-output outputs/smoke_chemistry_instruction/instruction_val.jsonl \
  --summary-path outputs/smoke_chemistry_instruction/dedup_summary.txt \
  --seed 0 \
  --force
```

## 5. Cleanup

```bash
rm -rf outputs cache
```

Keep `.env` local. Do not paste API keys, private endpoints, or proprietary
dataset contents into issues or pull requests.
